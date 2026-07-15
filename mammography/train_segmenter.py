"""
YOLOv8-seg 乳腺钼靶病灶分割器训练脚本

训练流程:
  1. 从CBIS-DDSM的CSV标注构建YOLO格式数据集 (BBox + Mask)
  2. 使用预训练yolov8n-seg.pt作为起点微调
  3. 保存最佳模型到 checkpoints/

数据集格式转换:
  CBIS-DDSM CSV 标注 → YOLO格式 (类别ID x_center y_center w h + mask points)

用法:
  python -m mammography.train_segmenter --epochs 50 --batch 8
"""

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from .config import (
    CSV_FILES,
    DATA_ROOT,
    DEVICE,
    MODEL_DIR,
    PATIENTS_DIR,
    RANDOM_SEED,
    SEG_MODEL_PATH,
    YOLO_INPUT_SIZE,
)
from .utils.dicom_utils import find_dicom_files, get_dicom_pixel_array
from .utils.preprocessing import apply_windowing


def prepare_yolo_dataset(
    output_dir: str,
    train_ratio: float = 0.8,
    img_size: int = 1024,
):
    """
    将CBIS-DDSM数据转换为YOLO分割格式

    YOLO格式要求:
      dataset/
      ├── images/
      │   ├── train/
      │   └── val/
      ├── labels/
      │   ├── train/
      │   │   └── image_001.txt  (class_id x1 y1 x2 y2 ... [归一化坐标])
      │   └── val/
      └── dataset.yaml

    参数:
        output_dir: YOLO格式数据集输出目录
        train_ratio: 训练/验证划分比例
        img_size: 图像保存尺寸
    """
    import random
    random.seed(RANDOM_SEED)

    out = Path(output_dir)
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    # 收集所有标注
    all_samples = []

    for csv_key, csv_name in CSV_FILES.items():
        csv_path = DATA_ROOT / csv_name
        if not csv_path.exists():
            print(f"跳过不存在的CSV: {csv_path}")
            continue

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pathology = row.get("pathology", "").strip()
                if pathology not in ["MALIGNANT", "BENIGN", "BENIGN_WITHOUT_CALLBACK"]:
                    continue

                full_path = row.get("image file path", "").strip().strip('"')
                cropped_path = row.get("cropped image file path", "").strip().strip('"')

                # 类别: mass=0, calcification=1
                abnormality = row.get("abnormality type", "").strip().lower()
                class_id = 0 if "mass" in abnormality else 1

                # 获取bbox (从cropped image的路径推断)
                if cropped_path:
                    # 尝试获取原图中的坐标
                    # CBIS-DDSM的cropped image是预裁剪的，需要反推坐标
                    cropped_full_path = DATA_ROOT / cropped_path
                    if cropped_full_path.exists():
                        all_samples.append({
                            "full_path": full_path,
                            "cropped_path": cropped_path,
                            "class_id": class_id,
                            "pathology": pathology,
                        })

    print(f"总样本数: {len(all_samples)}")

    # 打乱并划分
    random.shuffle(all_samples)
    n_train = int(len(all_samples) * train_ratio)

    splits = {"train": all_samples[:n_train], "val": all_samples[n_train:]}

    for split_name, samples in splits.items():
        for idx, sample in enumerate(samples):
            # 尝试加载全乳图像
            full_img_path = DATA_ROOT / sample["full_path"]
            crop_img_path = DATA_ROOT / sample["cropped_path"]

            if not full_img_path.exists() and not crop_img_path.exists():
                continue

            # 加载图像
            if full_img_path.exists():
                try:
                    if full_img_path.suffix.lower() == ".dcm":
                        pixels = get_dicom_pixel_array(str(full_img_path))
                        image = apply_windowing(pixels)
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    else:
                        image = cv2.imread(str(full_img_path))
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                except Exception:
                    continue
            else:
                try:
                    image = cv2.imread(str(crop_img_path))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                except Exception:
                    continue

            if image is None:
                continue

            h, w = image.shape[:2]

            # 保存图像
            img_name = f"{split_name}_{idx:06d}"
            img_save_path = out / f"images/{split_name}/{img_name}.jpg"
            cv2.imwrite(str(img_save_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            # 生成YOLO标注 (简化: 使用全图作为大致bbox)
            # TODO: 从CSV获取精确的bbox坐标或使用ROI mask
            label_path = out / f"labels/{split_name}/{img_name}.txt"
            with open(label_path, "w") as lf:
                # 暂时使用全图作为label
                # 格式: class_id cx cy w h
                lf.write(f"{sample['class_id']} 0.5 0.5 0.8 0.8\n")

        print(f"{split_name}: {len(samples)} → 已处理")

    # 创建 dataset.yaml
    yaml_content = f"""
path: {out.absolute()}
train: images/train
val: images/val

names:
  0: mass
  1: calcification
"""
    yaml_path = out / "dataset.yaml"
    yaml_path.write_text(yaml_content.strip(), encoding="utf-8")
    print(f"数据集配置已保存: {yaml_path}")

    return str(yaml_path)


def train_segmenter(
    data_yaml: Optional[str] = None,
    epochs: int = 50,
    batch_size: int = 8,
    img_size: int = YOLO_INPUT_SIZE,
    resume: bool = False,
):
    """
    训练YOLOv8-seg病灶分割模型

    参数:
        data_yaml: YOLO格式数据集配置文件路径
        epochs: 训练轮数
        batch_size: 批次大小
        img_size: 输入图像尺寸
        resume: 是否从checkpoint恢复
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 准备数据集
    if data_yaml is None:
        dataset_dir = DATA_ROOT / "yolo_dataset"
        data_yaml = prepare_yolo_dataset(str(dataset_dir), img_size=img_size)
        if not Path(data_yaml).exists():
            raise RuntimeError("数据集准备失败")

    # 加载预训练模型
    model = YOLO("yolov8n-seg.pt")

    # 训练
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=DEVICE,
        workers=4,
        patience=10,
        save=True,
        save_period=5,
        project=str(MODEL_DIR),
        name="mammo_seg",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=1e-3,
        lrf=1e-4,
        cos_lr=True,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.2,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
    )

    # 导出最佳模型
    best_path = MODEL_DIR / "mammo_seg" / "weights" / "best.pt"
    if best_path.exists():
        shutil.copy(best_path, SEG_MODEL_PATH)
        print(f"最佳模型已保存到: {SEG_MODEL_PATH}")

    return results


def main():
    parser = argparse.ArgumentParser(description="训练YOLOv8-seg乳腺钼靶病灶分割器")
    parser.add_argument("--data", type=str, default=None, help="YOLO格式数据集yaml路径")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch", type=int, default=8, help="批次大小")
    parser.add_argument("--img-size", type=int, default=YOLO_INPUT_SIZE, help="输入尺寸")
    parser.add_argument("--prepare-only", action="store_true", help="仅准备数据集，不训练")
    parser.add_argument("--resume", action="store_true", help="从checkpoint恢复训练")

    args = parser.parse_args()

    if args.prepare_only:
        prepare_yolo_dataset(str(DATA_ROOT / "yolo_dataset"))
    else:
        train_segmenter(
            data_yaml=args.data,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.img_size,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
