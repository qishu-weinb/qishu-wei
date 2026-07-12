"""
多模态乳腺图像分类训练脚本

支持三种模态：
1. Ultrasound (超声) - PNG图像，benign/malignant文件夹分类
2. Mammography (钼靶) - DICOM图像，CSV标签关联
3. MRI - DICOM图像，ACRIN 6667表格标签关联，T2+DCE配对

使用CNN分类器，支持训练/验证分割、数据增强、模型保存。
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from results.multimodal.common.datasets import (
    ImageFolderClassification,
    MammographyDataset,
    MRIPairedDataset,
)
from results.multimodal.common.models import CNNClassifier


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "multimodal")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "multimodal")
IMAGE_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-4
VAL_RATIO = 0.2


def train_modality(name, dataset, num_classes):
    """训练单个模态的模型。"""
    if len(dataset) == 0:
        print(f"\n[{name}] 跳过: 无样本")
        return None

    print(f"\n{'='*60}")
    print(f"[{name}] 开始训练 - 样本数: {len(dataset)}, 类别数: {num_classes}")
    print(f"{'='*60}")

    # 分割训练集和验证集
    n_val = max(1, int(len(dataset) * VAL_RATIO))
    n_train = len(dataset) - n_val

    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"  训练集: {len(train_subset)}, 验证集: {len(val_subset)}")

    # 统计标签分布
    train_labels = []
    for i in train_indices:
        _, label = dataset[i]
        train_labels.append(label.item())
    val_labels = []
    for i in val_indices:
        _, label = dataset[i]
        val_labels.append(label.item())

    print(f"  训练集标签分布: {dict(zip(*np.unique(train_labels, return_counts=True)))}")
    print(f"  验证集标签分布: {dict(zip(*np.unique(val_labels, return_counts=True)))}")

    # 初始化模型
    model = CNNClassifier(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(OUTPUT_DIR, f"{name}_cnn.pth")

    for epoch in range(EPOCHS):
        # 训练
        model.train()
        train_loss = 0.0
        train_preds = []
        train_gts = []

        for batch in tqdm(train_loader, desc=f"[{name}] Epoch {epoch+1}/{EPOCHS}", leave=False):
            images, labels = batch

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_gts.extend(labels.cpu().numpy())

        train_loss /= len(train_subset)
        train_acc = accuracy_score(train_gts, train_preds)

        # 验证
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_gts = []

        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_gts.extend(labels.cpu().numpy())

        val_loss /= len(val_subset)
        val_acc = accuracy_score(val_gts, val_preds)
        val_f1 = f1_score(val_gts, val_preds, average="weighted", zero_division=0)
        val_precision = precision_score(val_gts, val_preds, average="weighted", zero_division=0)
        val_recall = recall_score(val_gts, val_preds, average="weighted", zero_division=0)

        print(f"[{name}] Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"  -> 保存最佳模型到 {model_path}")

    print(f"\n[{name}] 训练完成! 最佳验证准确率: {best_val_acc:.4f}")
    return {"best_val_acc": best_val_acc, "model_path": model_path}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}

    # 1. 超声
    print("\n" + "="*60)
    print("开始训练超声(Ultrasound)模型")
    print("="*60)

    us_dir = os.path.join(DATA_ROOT, "ultrasound")
    if os.path.isdir(us_dir):
        us_dataset = ImageFolderClassification(us_dir, image_size=IMAGE_SIZE)
        results["ultrasound"] = train_modality("ultrasound", us_dataset, num_classes=2)
    else:
        print(f"[ultrasound] 目录不存在: {us_dir}")

    # 2. 钼靶
    print("\n" + "="*60)
    print("开始训练钼靶(Mammography)模型")
    print("="*60)

    mg_dir = os.path.join(DATA_ROOT, "mammography")
    if os.path.isdir(mg_dir):
        mg_dataset = MammographyDataset(mg_dir, image_size=IMAGE_SIZE)
        results["mammography"] = train_modality("mammography", mg_dataset, num_classes=2)
    else:
        print(f"[mammography] 目录不存在: {mg_dir}")

    # 3. MRI
    print("\n" + "="*60)
    print("开始训练MRI模型")
    print("="*60)

    mri_dir = os.path.join(DATA_ROOT, "mri")
    acrin_dir = os.path.join(mri_dir, "ACRIN 6667 Contralateral Breast MRI Clinical Data Anonymized")
    if os.path.isdir(mri_dir):
        mri_dataset = MRIPairedDataset(mri_dir, acrin_dir, image_size=IMAGE_SIZE)
        results["mri"] = train_modality("mri", mri_dataset, num_classes=2)
    else:
        print(f"[mri] 目录不存在: {mri_dir}")

    # 汇总
    print("\n" + "="*60)
    print("训练汇总")
    print("="*60)
    for name, result in results.items():
        if result:
            print(f"  {name}: 最佳验证准确率={result['best_val_acc']:.4f}, 模型={result['model_path']}")
        else:
            print(f"  {name}: 训练失败或无数据")


if __name__ == "__main__":
    main()
