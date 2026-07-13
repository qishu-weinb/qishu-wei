"""
CBIS-DDSM 乳腺钼靶数据集加载器

适配 Kaggle 版本 (awsaf49/cbis-ddsm-breast-cancer-image-dataset):
  - CSV 标注: csv/*.csv
  - JPEG 图像: jpeg/<SOPInstanceUID>/<N-XXX>.jpg

映射关系:
  - 训练CSV的 cropped_image_file_path 最后一段UID = SOPInstanceUID
  - SOPInstanceUID → jpeg/<UID>/ 目录
  - 目录内 1-XXX.jpg = 全图/ROI mask, 2-XXX.jpg = 裁剪ROI
  - dicom_info.csv 提供 PatientID ↔ SeriesDescription ↔ image_path 的完整映射
"""

import csv
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import (
    CLASS_NAMES,
    CSV_FILES,
    DATA_ROOT,
    LABEL_MAP,
    NUM_CLASSES,
    RANDOM_SEED,
    TRAIN_SPLIT_RATIO,
)


# ====================================================================
# 路径解析
# ====================================================================

def _extract_sop_uid(csv_path: str) -> Optional[str]:
    """从 CSV 中的 DICOM 路径提取 SOP Instance UID (最后一个UID段)"""
    for part in reversed(csv_path.replace("\\", "/").split("/")):
        if part.startswith("1.3.6.1.4.1.9590.100.1.2."):
            return part
    return None


class CBISDDSMPathResolver:
    """CBIS-DDSM 路径解析器

    将训练CSV中的DICOM路径映射到实际磁盘上的JPEG文件。
    """

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self.csv_dir = self.data_root / "csv"
        self.jpeg_dir = self.data_root / "jpeg"

        # 构建 SOPInstanceUID → JPEG目录 的缓存
        self._uid_to_dir: dict = {}
        self._build_uid_cache()

    def _build_uid_cache(self):
        """扫描 jpeg/ 目录，建立 UID → 目录路径 的快速查找"""
        if not self.jpeg_dir.is_dir():
            return
        for d in self.jpeg_dir.iterdir():
            if d.is_dir() and d.name.startswith("1.3.6.1.4.1.9590.100.1.2."):
                self._uid_to_dir[d.name] = d

    def resolve_cropped(self, csv_cropped_path: str) -> Optional[Path]:
        """解析裁剪ROI图像的磁盘路径

        Args:
            csv_cropped_path: 训练CSV中的 cropped image file path

        Returns:
            JPEG文件路径，或 None
        """
        uid = _extract_sop_uid(csv_cropped_path)
        if uid is None:
            return None

        jpeg_dir = self._uid_to_dir.get(uid)
        if jpeg_dir is None:
            return None

        # 裁剪ROI通常是 2-XXX.jpg
        for f in sorted(jpeg_dir.glob("2-*.jpg")):
            return f
        # 回退到任意jpg
        for f in sorted(jpeg_dir.glob("*.jpg")):
            return f
        return None

    def resolve_full(self, csv_full_path: str) -> Optional[Path]:
        """解析全乳图像的磁盘路径"""
        uid = _extract_sop_uid(csv_full_path)
        if uid is None:
            return None

        jpeg_dir = self._uid_to_dir.get(uid)
        if jpeg_dir is None:
            return None

        # 全图通常是 1-XXX.jpg
        for f in sorted(jpeg_dir.glob("1-*.jpg")):
            return f
        for f in sorted(jpeg_dir.glob("*.jpg")):
            return f
        return None


# ====================================================================
# 图像读取
# ====================================================================

def read_dicom(path: str) -> np.ndarray:
    """读取DICOM文件并转为8-bit RGB (兼容旧接口)"""
    import pydicom
    from ..config import WINDOW_LEVEL, WINDOW_WIDTH

    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)

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

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def read_image(path: str) -> np.ndarray:
    """统一读取图像 (支持 dcm/png/jpg)"""
    ext = Path(path).suffix.lower()
    if ext == ".dcm":
        return read_dicom(path)
    else:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"无法读取图像: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ====================================================================
# 主数据集类
# ====================================================================

class CBISDDSMDataset(Dataset):
    """CBIS-DDSM PyTorch Dataset

    从CSV标注文件加载数据，支持:
      - classification: 裁剪ROI → BCHI-CovNet 分类训练
      - segmentation: 全乳图像 → YOLO 分割训练
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
        self.data_root = Path(data_root) if data_root else DATA_ROOT
        self.transform = transform
        self.mode = mode
        self.split = split
        self.resolver = CBISDDSMPathResolver(self.data_root)

        # CSF 文件位于 csv/ 子目录
        csv_dir = self.data_root / "csv"

        if csv_names is None:
            csv_names = [
                csv_dir / CSV_FILES["mass_train"],
                csv_dir / CSV_FILES["mass_test"],
                csv_dir / CSV_FILES["calc_train"],
                csv_dir / CSV_FILES["calc_test"],
            ]

        # 解析CSV → 样本列表
        self.samples = []
        for csv_path in csv_names:
            if csv_path.exists():
                self._parse_csv(csv_path)
            else:
                # 兼容旧路径 (CSV直接在data_root下)
                alt = self.data_root / csv_path.name
                if alt.exists():
                    self._parse_csv(alt)

        # 训练/验证划分
        if len(self.samples) > 0:
            random.seed(seed)
            random.shuffle(self.samples)
            n_train = int(len(self.samples) * train_ratio)

            if split == "train":
                self.samples = self.samples[:n_train]
            elif split == "val":
                self.samples = self.samples[n_train:]

    def _parse_csv(self, csv_path: Path):
        """解析单个CSV标注文件"""
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pathology = (row.get("pathology") or "").strip()
                if pathology not in LABEL_MAP:
                    continue

                cropped_path = (row.get("cropped image file path") or "").strip().strip('"')
                full_path = (row.get("image file path") or "").strip().strip('"')
                label = LABEL_MAP[pathology]

                sample = {
                    "cropped_path": cropped_path,
                    "full_path": full_path,
                    "label": label,
                    "pathology": pathology,
                    "laterality": (row.get("left or right breast") or "").strip(),
                    "view": (row.get("image view") or "").strip(),
                    "abnormality_type": (row.get("abnormality type") or "").strip(),
                    "patient_id": (row.get("patient_id") or "").strip(),
                }
                self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = None

        if self.mode == "classification":
            jpeg_path = self.resolver.resolve_cropped(sample["cropped_path"])
        else:
            jpeg_path = self.resolver.resolve_full(sample["full_path"])

        if jpeg_path and jpeg_path.exists():
            image = read_image(str(jpeg_path))

        if image is None:
            image = np.zeros((256, 256, 3), dtype=np.uint8)

        label = sample["label"]

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "sample": sample,
        }

    def get_class_distribution(self) -> dict:
        dist = defaultdict(int)
        for s in self.samples:
            dist[CLASS_NAMES[s["label"]]] += 1
        return dict(dist)


# ====================================================================
# DataLoader 工厂
# ====================================================================

def create_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
    mode: str = "classification",
    transform_train=None,
    transform_val=None,
) -> Tuple[DataLoader, DataLoader]:
    """创建训练和验证 DataLoader"""
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
    if len(train_dataset) > 0:
        print(f"类别分布: {train_dataset.get_class_distribution()}")

    return train_loader, val_loader
