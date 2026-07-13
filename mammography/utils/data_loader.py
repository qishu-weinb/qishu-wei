"""
CBIS-DDSM 乳腺钼靶数据集加载器

支持:
  - DICOM 文件读取与窗宽窗位转换
  - CSV 标注文件解析
  - 病灶ROI提取 (基于标注的bbox/crop路径)
  - 训练/验证集划分

CBIS-DDSM 数据结构:
  mammography/
  ├── patients/
  │   ├── Mass-Training_P_00001_LEFT_CC/
  │   │   ├── 1-1.dcm          # full mammogram
  │   │   ├── 1-1_crop_00.png  # cropped ROI (已提供)
  │   │   └── ...
  │   └── ...
  ├── calc_case_description_train_set.csv
  ├── calc_case_description_test_set.csv
  ├── mass_case_description_train_set.csv
  └── mass_case_description_test_set.csv
"""

import csv
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pydicom
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import (
    CLASS_NAMES,
    CSV_FILES,
    DATA_ROOT,
    LABEL_MAP,
    NORM_MEAN,
    NORM_STD,
    NUM_CLASSES,
    PATIENTS_DIR,
    RANDOM_SEED,
    TRAIN_SPLIT_RATIO,
    WINDOW_LEVEL,
    WINDOW_WIDTH,
)


def read_dicom(path: str) -> np.ndarray:
    """
    读取DICOM文件并转为8-bit RGB图像

    处理流程:
      1. 读取DICOM → 提取像素数组 (12/14-bit)
      2. 应用窗宽窗位 → 映射到0-255
      3. 灰度→RGB三通道复制

    参数:
        path: DICOM文件路径

    返回:
        image: RGB图像 (H, W, 3), uint8
    """
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)

    # 窗宽窗位映射
    ww = getattr(ds, "WindowWidth", WINDOW_WIDTH)
    wl = getattr(ds, "WindowCenter", WINDOW_LEVEL)
    if isinstance(ww, (list, pydicom.multival.MultiValue)):
        ww = float(ww[0])
    if isinstance(wl, (list, pydicom.multival.MultiValue)):
        wl = float(wl[0])

    low = wl - ww / 2
    high = wl + ww / 2
    img = np.clip(img, low, high)
    img = ((img - low) / (high - low) * 255).astype(np.uint8)

    # 灰度→RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img


def read_image(path: str) -> np.ndarray:
    """
    统一读取图像文件 (支持 dcm / png / jpg)

    参数:
        path: 图像文件路径

    返回:
        image: RGB图像 (H, W, 3), uint8
    """
    ext = Path(path).suffix.lower()
    if ext == ".dcm":
        return read_dicom(path)
    else:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"无法读取图像: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class CBISDDSMDataset(Dataset):
    """
    CBIS-DDSM 数据集 (PyTorch Dataset)

    从CSV标注文件加载数据，支持:
      - 全乳图像 (full mammogram) 用于分割训练
      - 裁剪ROI图像 (cropped image) 用于分类训练
      - 肿块(mass)和钙化(calcification)两种病灶类型
    """

    def __init__(
        self,
        data_root: str = None,
        csv_names: Optional[list] = None,
        transform=None,
        mode: str = "classification",
        split: str = "train",
        train_ratio: float = TRAIN_SPLIT_RATIO,
        seed: int = RANDOM_SEED,
    ):
        """
        参数:
            data_root: 数据根目录
            csv_names: 要加载的CSV文件列表
            transform: 数据增强/预处理
            mode: "classification" (ROI分类) 或 "segmentation" (全图分割)
            split: "train" / "val" / "all"
            train_ratio: 训练集比例
            seed: 随机种子
        """
        self.data_root = Path(data_root) if data_root else DATA_ROOT
        self.transform = transform
        self.mode = mode
        self.split = split

        if csv_names is None:
            csv_names = [
                CSV_FILES["mass_train"],
                CSV_FILES["mass_test"],
                CSV_FILES["calc_train"],
                CSV_FILES["calc_test"],
            ]

        # 解析CSV → 样本列表
        self.samples = []
        for csv_name in csv_names:
            csv_path = self.data_root / csv_name
            if csv_path.exists():
                self._parse_csv(csv_path)

        # 训练/验证集划分
        random.seed(seed)
        random.shuffle(self.samples)
        n_train = int(len(self.samples) * train_ratio)

        if split == "train":
            self.samples = self.samples[:n_train]
        elif split == "val":
            self.samples = self.samples[n_train:]
        # "all" → 全部

    def _parse_csv(self, csv_path: Path):
        """解析单个CSV标注文件"""
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pathology = row.get("pathology", "").strip()
                if pathology not in LABEL_MAP:
                    continue

                # 获取裁剪图像路径
                cropped_path = row.get("cropped image file path", "").strip().strip('"')
                # 获取全乳图像路径
                full_path = row.get("image file path", "").strip().strip('"')

                label = LABEL_MAP[pathology]

                # 处理乳腺摄影特殊标签 (左右+视角)
                laterality = row.get("left or right breast", "").strip()
                view = row.get("image view", "").strip()

                sample = {
                    "cropped_path": cropped_path,
                    "full_path": full_path,
                    "label": label,
                    "pathology": pathology,
                    "laterality": laterality,
                    "view": view,
                    "abnormality_type": row.get("abnormality type", "").strip(),
                    "patient_id": row.get("patient_id", "").strip(),
                }
                self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        if self.mode == "classification":
            # 加载裁剪的ROI图像
            image_path = self.data_root / sample["cropped_path"]
            if not image_path.exists():
                image_path = self._find_image(sample["cropped_path"])
        else:
            # 加载全乳图像
            image_path = self.data_root / sample["full_path"]
            if not image_path.exists():
                image_path = self._find_image(sample["full_path"])

        if image_path is None or not image_path.exists():
            # 返回blank占位
            image = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            image = read_image(str(image_path))

        label = sample["label"]

        # 应用transform
        if self.transform:
            image = self.transform(image)

        if isinstance(image, np.ndarray):
            label_tensor = torch.tensor(label, dtype=torch.long)
        else:
            label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            "image": image,
            "label": label_tensor,
            "sample": sample,
        }

    def _find_image(self, rel_path: str) -> Optional[Path]:
        """在patients目录中递归查找图像文件"""
        filename = Path(rel_path).name
        folder_name = Path(rel_path).parts[0] if "/" in rel_path else ""

        # 先在patients目录中找对应文件夹
        if folder_name:
            patient_dir = PATIENTS_DIR / folder_name
            if patient_dir.exists():
                for ext in [".png", ".jpg", ".jpeg", ".dcm"]:
                    candidate = patient_dir / f"{Path(filename).stem}{ext}"
                    if candidate.exists():
                        return candidate
                # 递归搜索
                for f in patient_dir.rglob(f"*{Path(filename).stem}*"):
                    return f

        # 全局搜索（慢，仅备用）
        for f in PATIENTS_DIR.rglob(f"*{Path(filename).stem}*"):
            return f

        return None

    def get_class_distribution(self) -> dict:
        """获取各类别样本数"""
        dist = defaultdict(int)
        for s in self.samples:
            dist[s["label"]] += 1
        return dict(dist)


def create_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
    mode: str = "classification",
    transform_train=None,
    transform_val=None,
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证DataLoader

    参数:
        batch_size: 批次大小
        num_workers: 数据加载线程
        mode: "classification" 或 "segmentation"
        transform_train: 训练集数据增强
        transform_val: 验证集预处理

    返回:
        train_loader, val_loader
    """
    train_dataset = CBISDDSMDataset(
        transform=transform_train,
        mode=mode,
        split="train",
    )
    val_dataset = CBISDDSMDataset(
        transform=transform_val,
        mode=mode,
        split="val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    print(f"类别分布: {train_dataset.get_class_distribution()}")

    return train_loader, val_loader
