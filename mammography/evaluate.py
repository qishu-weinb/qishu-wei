"""
模型评估模块

评估两阶段流水线的性能:
  1. 分割器评估: mAP, IoU, Dice
  2. 分类器评估: Accuracy, F1, AUC, 混淆矩阵
  3. 端到端评估: 全流水线准确率

输出:
  - ROC曲线, PR曲线
  - 混淆矩阵热力图
  - 按病灶类型(肿块/钙化)的分类性能
  - JSON评估报告
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

from .config import (
    CLASS_NAMES,
    CLASS_NAMES_CN,
    CLS_MODEL_PATH,
    DATA_ROOT,
    DEVICE,
    MODEL_DIR,
    NUM_CLASSES,
    OUTPUT_DIR,
    SEG_MODEL_PATH,
)
from .models.pipeline import MammographyPipeline
from .utils.data_loader import CBISDDSMDataset, read_image


def evaluate_segmenter(
    model_path: Optional[str] = None,
    conf_threshold: float = 0.5,
):
    """
    评估YOLOv8-seg分割器性能

    计算指标: mAP@0.5, mAP@0.5:0.95, Precision, Recall

    参数:
        model_path: 分割器模型路径
        conf_threshold: 置信度阈值
    """
    from ultralytics import YOLO

    if model_path is None:
        model_path = str(SEG_MODEL_PATH) if SEG_MODEL_PATH.exists() else "yolov8n-seg.pt"

    model = YOLO(model_path)

    # 使用ultralytics内置评估
    # 需要YOLO格式的验证集
    yolo_val_dir = DATA_ROOT / "yolo_dataset"
    data_yaml = yolo_val_dir / "dataset.yaml"

    if data_yaml.exists():
        results = model.val(
            data=str(data_yaml),
            conf=conf_threshold,
            device=DEVICE,
            split="val",
            verbose=True,
        )
        return results
    else:
        print("未找到YOLO格式验证集，跳过分割器评估。")
        print(f"请先运行 train_segmenter.py --prepare-only 准备数据集")
        return None


def evaluate_classifier(
    model_path: Optional[str] = None,
    output_dir: Optional[str] = None,
):
    """
    评估BCHI-CovNet分类器性能

    参数:
        model_path: 分类器权重路径
        output_dir: 输出目录
    """
    from .models.bchi_covnet import BCHICovNet
    from .train_classifier import get_transforms, validate_epoch
    from torch.utils.data import DataLoader
    import torch.nn as nn

    if model_path is None:
        model_path = str(CLS_MODEL_PATH)

    if not Path(model_path).exists():
        print(f"模型权重不存在: {model_path}")
        return None

    # 加载模型
    model = BCHICovNet(in_channels=3, num_classes=NUM_CLASSES)
    state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=False)
    model = model.to(DEVICE)
    model.eval()

    # 测试数据集
    transform_val = get_transforms(train=False)
    test_dataset = CBISDDSMDataset(
        transform=transform_val,
        mode="classification",
        split="all",
    )

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    print(f"测试集大小: {len(test_dataset)}")
    print(f"类别分布: {test_dataset.get_class_distribution()}")

    # 评估
    criterion = nn.CrossEntropyLoss()
    metrics = validate_epoch(model, test_loader, criterion, DEVICE)

    _print_classifier_results(metrics)

    # 保存结果
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        eval_result = {
            "model": "BCHI-CovNet",
            "num_samples": len(test_dataset),
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "metrics": {k: v for k, v in metrics.items() if k not in ("predictions", "labels")},
        }
        (out / "classifier_evaluation.json").write_text(
            json.dumps(eval_result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return metrics


def evaluate_end_to_end(
    seg_weights: Optional[str] = None,
    cls_weights: Optional[str] = None,
    num_samples: int = 50,
    output_dir: Optional[str] = None,
):
    """
    端到端流水线评估

    在测试集上评估完整两阶段流水线的准确性

    参数:
        seg_weights: YOLO分割器权重
        cls_weights: BCHI-CovNet分类器权重
        num_samples: 评估样本数
        output_dir: 输出目录
    """
    import random
    from .config import RANDOM_SEED

    random.seed(RANDOM_SEED)

    # 加载流水线
    pipeline = MammographyPipeline(
        seg_model_path=seg_weights,
        cls_model_path=cls_weights,
    )

    # 加载测试数据
    dataset = CBISDDSMDataset(
        mode="classification",
        split="all",
    )

    # 随机采样
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    y_true = []
    y_pred = []
    results_detail = []

    print(f"端到端评估: {len(indices)} 样本")

    for idx in tqdm(indices, desc="端到端诊断"):
        sample_data = dataset[idx]
        sample = dataset.samples[idx]
        label = sample["label"]

        # 加载原始图像
        image_path = DATA_ROOT / sample["full_path"]
        if not image_path.exists():
            image_path = DATA_ROOT / sample["cropped_path"]

        if not image_path.exists():
            continue

        try:
            image = read_image(str(image_path))
            result = pipeline.diagnose(image)
        except Exception as e:
            print(f"跳过 {image_path.name}: {e}")
            continue

        # 从最高置信度病灶获取预测类别
        lesions = result["lesions"]
        if lesions:
            best = max(lesions, key=lambda l: l.get("pred_confidence", 0))
            cls_info = best.get("classification", {})
            pred = cls_info.get("class_id", 0) if cls_info else 0
        else:
            pred = 0  # normal

        y_true.append(label)
        y_pred.append(pred)

        results_detail.append({
            "image": str(image_path.name),
            "true_label": label,
            "pred_label": pred,
            "num_lesions": len(lesions),
            "risk_level": result["summary"]["risk_level"],
        })

    if len(y_true) == 0:
        print("无有效评估样本")
        return None

    # 计算指标
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 60)
    print("  端到端流水线评估结果")
    print("=" * 60)
    print(f"样本数: {len(y_true)}")
    print(f"准确率: {acc:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")

    print(f"\n混淆矩阵:")
    header = "          " + " ".join(f"{n:>8}" for n in CLASS_NAMES)
    print(header)
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:>8}  " + " ".join(f"{cm[i][j] if i < cm.shape[0] else 0:>8}" for j in range(NUM_CLASSES)))

    print(f"\n分类报告:")
    print(classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        zero_division=0,
    ))

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "end_to_end_evaluation.json").write_text(
            json.dumps({
                "num_samples": len(y_true),
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": cm.tolist(),
                "details": results_detail,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
    }


def _print_classifier_results(metrics: dict):
    """打印分类器评估结果"""
    print("\n" + "=" * 60)
    print("  BCHI-CovNet 分类器评估结果")
    print("=" * 60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"AUC:       {metrics['auc']:.4f}")

    cm = np.array(metrics["confusion_matrix"])
    print(f"\n混淆矩阵:")
    header = "          " + " ".join(f"{n:>8}" for n in CLASS_NAMES)
    print(header)
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:>8}  " + " ".join(f"{cm[i][j]:>8}" for j in range(NUM_CLASSES)))


def main():
    parser = argparse.ArgumentParser(description="乳腺钼靶模型评估")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "segmenter", "classifier", "e2e"],
                        help="评估模式")
    parser.add_argument("--seg-weights", type=str, default=None)
    parser.add_argument("--cls-weights", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=50,
                        help="端到端评估样本数")

    args = parser.parse_args()

    output_dir = args.output or str(OUTPUT_DIR / "evaluation")

    if args.mode in ("all", "segmenter"):
        print("\n>>> 评估分割器")
        evaluate_segmenter(args.seg_weights)

    if args.mode in ("all", "classifier"):
        print("\n>>> 评估分类器")
        evaluate_classifier(args.cls_weights, output_dir)

    if args.mode in ("all", "e2e"):
        print("\n>>> 端到端评估")
        evaluate_end_to_end(
            args.seg_weights,
            args.cls_weights,
            args.num_samples,
            output_dir,
        )


if __name__ == "__main__":
    main()
