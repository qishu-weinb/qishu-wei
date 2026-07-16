"""
BCHI-CovNet 分类器训练脚本

训练两阶段流水线的第二阶段分类器:
  - 使用CBIS-DDSM裁剪ROI图像训练
  - 支持多种数据增强策略
  - 类别不平衡处理 (加权损失 + 过采样)
  - 四分类: normal / benign / inSitu / invasive

用法:
  python -m mammography.train_classifier --epochs 60 --batch 32
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .config import (
    BCHI_BATCH_SIZE,
    BCHI_DROPOUT,
    BCHI_EPOCHS,
    BCHI_LABEL_SMOOTHING,
    BCHI_LEARNING_RATE,
    BCHI_WEIGHT_DECAY,
    CLASS_NAMES,
    CLS_MODEL_PATH,
    DEVICE,
    MODEL_DIR,
    NORM_MEAN,
    NORM_STD,
    NUM_CLASSES,
    RANDOM_SEED,
)
from .models.bchi_covnet import BCHICovNet
from .utils.data_loader import CBISDDSMDataset, create_dataloaders

# 设置随机种子
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def get_transforms(train: bool = True) -> T.Compose:
    """
    获取数据增强/预处理管道

    训练集: 强数据增强 (模拟临床图像变化)
    验证集: 仅标准化

    关键设计: 不对ROI做resize，保留原始分辨率纹理。
    """
    if train:
        return T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.15, contrast=0.15),
            T.ToTensor(),
            T.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ])
    else:
        return T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
            T.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ])


def compute_class_weights(dataset: CBISDDSMDataset) -> torch.Tensor:
    """
    计算类别权重 (处理类别不平衡)

    参数:
        dataset: 训练数据集

    返回:
        weights: 各类别权重张量 [num_classes]
    """
    dist = dataset.get_class_distribution()
    total = sum(dist.values())

    weights = []
    for c in range(NUM_CLASSES):
        count = dist.get(c, 0)
        if count > 0:
            weights.append(total / (NUM_CLASSES * count))
        else:
            weights.append(1.0)

    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


def train_epoch(
    model: BCHICovNet,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
) -> dict:
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()

        # 梯度裁剪 (防止梯度爆炸)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": acc,
        "f1": f1,
    }


@torch.no_grad()
def validate_epoch(
    model: BCHICovNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> dict:
    """验证一个epoch"""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    all_probs = []

    for batch in tqdm(loader, desc="Validation", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    # AUC (one-vs-rest)
    try:
        all_probs_arr = np.array(all_probs)
        all_labels_arr = np.array(all_labels)
        auc = roc_auc_score(
            np.eye(NUM_CLASSES)[all_labels_arr],
            all_probs_arr,
            multi_class="ovr",
            average="weighted",
        )
    except Exception:
        auc = 0.0

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc,
        "confusion_matrix": cm.tolist(),
        "predictions": all_preds,
        "labels": all_labels,
    }


def train_classifier(
    epochs: int = BCHI_EPOCHS,
    batch_size: int = BCHI_BATCH_SIZE,
    lr: float = BCHI_LEARNING_RATE,
    weight_decay: float = BCHI_WEIGHT_DECAY,
    patience: int = 15,
    resume_from: Optional[str] = None,
    use_weighted_sampler: bool = True,
):
    """
    训练BCHI-CovNet分类器

    参数:
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        weight_decay: 权重衰减
        patience: 早停耐心值
        resume_from: 从checkpoint恢复
        use_weighted_sampler: 是否使用加权采样
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 数据加载
    transform_train = get_transforms(train=True)
    transform_val = get_transforms(train=False)

    train_dataset = CBISDDSMDataset(
        transform=transform_train,
        mode="classification",
        split="train",
    )
    val_dataset = CBISDDSMDataset(
        transform=transform_val,
        mode="classification",
        split="val",
    )

    print(f"训练集: {len(train_dataset)} | 验证集: {len(val_dataset)}")
    print(f"训练集分布: {train_dataset.get_class_distribution()}")

    # 加权采样器
    sampler = None
    if use_weighted_sampler:
        class_counts = train_dataset.get_class_distribution()
        total = sum(class_counts.values())
        sample_weights = []
        for s in train_dataset.samples:
            count = class_counts.get(s["label"], 1)
            sample_weights.append(total / (NUM_CLASSES * count))
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        print("使用加权采样器处理类别不平衡")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    ) if sampler else DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 模型
    model = BCHICovNet(
        in_channels=3,
        num_classes=NUM_CLASSES,
        dropout=BCHI_DROPOUT,
        use_attention=True,
    )

    if resume_from and Path(resume_from).exists():
        state = torch.load(resume_from, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state, strict=False)
        print(f"从checkpoint恢复: {resume_from}")

    model = model.to(DEVICE)

    # 损失函数 (类别加权 + 标签平滑)
    class_weights = compute_class_weights(train_dataset)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=BCHI_LABEL_SMOOTHING,
    )

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # 学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        verbose=True,
    )

    # 训练
    best_val_f1 = 0.0
    best_epoch = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    patience_counter = 0

    print(f"\n开始训练 BCHI-CovNet | 设备: {DEVICE}")
    print(f"参数: epochs={epochs}, batch={batch_size}, lr={lr}")
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_metrics = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_metrics = validate_epoch(model, val_loader, criterion, DEVICE)

        scheduler.step(val_metrics["f1"])

        elapsed = time.time() - epoch_start

        # 记录历史
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"T Loss: {train_metrics['loss']:.4f} | T Acc: {train_metrics['accuracy']:.4f} | "
            f"V Loss: {val_metrics['loss']:.4f} | V Acc: {val_metrics['accuracy']:.4f} | "
            f"V F1: {val_metrics['f1']:.4f} | V AUC: {val_metrics['auc']:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # 保存最佳模型
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            patience_counter = 0

            torch.save(model.state_dict(), CLS_MODEL_PATH)
            print(f"  ✓ 保存最佳模型 (F1={best_val_f1:.4f}) -> {CLS_MODEL_PATH}")

            # 保存最佳epoch的详细指标
            metrics_path = MODEL_DIR / "best_metrics.json"
            metrics_path.write_text(
                json.dumps({
                    "epoch": epoch,
                    "val_accuracy": val_metrics["accuracy"],
                    "val_f1": val_metrics["f1"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "val_auc": val_metrics["auc"],
                    "confusion_matrix": val_metrics["confusion_matrix"],
                    "class_names": CLASS_NAMES,
                }, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n早停触发! 最佳F1={best_val_f1:.4f} at epoch {best_epoch}")
                break

    print(f"\n训练完成! 最佳F1={best_val_f1:.4f} at epoch {best_epoch}")

    # 保存训练历史
    history_path = MODEL_DIR / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    return model, history


def evaluate_model(
    model_path: Optional[str] = None,
):
    """
    在测试集上评估模型

    参数:
        model_path: 模型权重路径
    """
    if model_path is None:
        model_path = str(CLS_MODEL_PATH)

    # 加载模型
    model = BCHICovNet(in_channels=3, num_classes=NUM_CLASSES)
    state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=False)
    model = model.to(DEVICE)
    model.eval()

    # 测试数据
    transform_val = get_transforms(train=False)
    test_dataset = CBISDDSMDataset(
        transform=transform_val,
        mode="classification",
        split="all",
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    metrics = validate_epoch(model, test_loader, criterion, DEVICE)

    print("\n" + "=" * 60)
    print("  测试集评估结果")
    print("=" * 60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"AUC:       {metrics['auc']:.4f}")
    print(f"\n混淆矩阵:")
    cm = np.array(metrics["confusion_matrix"])
    header = "          " + " ".join(f"{n:>8}" for n in CLASS_NAMES)
    print(header)
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:>8}  " + " ".join(f"{cm[i][j]:>8}" for j in range(NUM_CLASSES)))

    print(f"\n分类报告:")
    print(classification_report(
        metrics["labels"],
        metrics["predictions"],
        target_names=CLASS_NAMES,
        zero_division=0,
    ))

    return metrics


def main():
    parser = argparse.ArgumentParser(description="训练BCHI-CovNet乳腺钼靶ROI分类器")
    parser.add_argument("--epochs", type=int, default=BCHI_EPOCHS)
    parser.add_argument("--batch", type=int, default=BCHI_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=BCHI_LEARNING_RATE)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的checkpoint路径")
    parser.add_argument("--evaluate", action="store_true", help="仅评估模型")
    parser.add_argument("--model-path", type=str, default=None, help="评估用的模型路径")

    args = parser.parse_args()

    if args.evaluate:
        evaluate_model(args.model_path)
    else:
        train_classifier(
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            patience=args.patience,
            resume_from=args.resume,
        )


if __name__ == "__main__":
    main()
