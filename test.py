"""
模型测试脚本

功能说明：
- 评估LightweightUNet模型在训练集、验证集和测试集上的性能
- 计算并输出多种评估指标：准确率、精确率、召回率、F1分数
- 生成混淆矩阵图和类别分布图
- 支持真实数据集和模拟数据集

测试流程：
1. 加载训练/验证/测试数据集
2. 加载训练好的模型权重
3. 在三个数据集上评估性能
4. 输出评估报告和可视化结果
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免显示图像
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 设置字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import seaborn as sns

from models.lightweight_unet import LightweightUNet
from datasets.breast_cancer_dataset import BreastCancerDataset
from config import TRAIN_DIR, VAL_DIR, TEST_DIR, UNET_MODEL_PATH, UNET_INPUT_SIZE, BATCH_SIZE, DEVICE, NUM_CLASSES, CLASSES


def evaluate_model(model, dataloader, device):
    """
    评估模型性能的函数
    
    参数：
        model: 待评估的模型
        dataloader: 数据加载器
        device: 运行设备
    
    返回：
        results: 评估结果字典，包含损失、准确率、精确率、召回率、F1等
    """
    # 设置模型为评估模式
    model.eval()
    all_preds = []      # 存储所有预测结果
    all_labels = []     # 存储所有真实标签
    all_probs = []      # 存储所有预测概率
    total_loss = 0.0    # 总损失
    criterion = nn.CrossEntropyLoss()  # 损失函数

    # 关闭梯度计算（评估阶段不需要）
    with torch.no_grad():
        for images, labels in dataloader:
            # 将数据移动到指定设备
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            cls_out = model(images)
            # 计算损失
            loss = criterion(cls_out, labels)

            # 累积损失
            total_loss += loss.item() * images.size(0)

            # 获取概率和预测类别
            probs = torch.softmax(cls_out, dim=1)
            preds = torch.argmax(probs, dim=1)

            # 收集预测结果
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 计算平均损失
    total_loss = total_loss / len(dataloader.dataset)

    # 计算评估指标
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # 计算混淆矩阵和分类报告
    cm = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, target_names=CLASSES, zero_division=0)

    return {
        'loss': total_loss,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'classification_report': class_report,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


def plot_confusion_matrix(cm, save_path=None):
    """
    绘制混淆矩阵图
    
    参数：
        cm: 混淆矩阵
        save_path: 保存路径（可选）
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_class_distribution(labels, predictions, save_path=None):
    """
    绘制类别分布图
    
    参数：
        labels: 真实标签
        predictions: 预测标签
        save_path: 保存路径（可选）
    """
    plt.figure(figsize=(12, 5))

    # 真实类别分布
    plt.subplot(1, 2, 1)
    unique, counts = np.unique(labels, return_counts=True)
    plt.bar(['Benign', 'Malignant'], counts, color=['blue', 'red'])
    plt.title('True Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')

    # 预测类别分布
    plt.subplot(1, 2, 2)
    unique, counts = np.unique(predictions, return_counts=True)
    plt.bar(['Benign', 'Malignant'], counts, color=['blue', 'red'])
    plt.title('Predicted Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """
    主函数：执行完整的模型评估流程
    """
    # 数据预处理管道（与训练时保持一致，但不使用随机增广）
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((UNET_INPUT_SIZE, UNET_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 初始化数据集变量
    train_dataset = None
    val_dataset = None
    test_dataset = None

    # 加载训练数据集
    if os.path.exists(TRAIN_DIR):
        train_dataset = BreastCancerDataset(TRAIN_DIR, transform=transform)
        print(f"训练数据集大小: {len(train_dataset)}")
    else:
        print(f"警告: 训练目录 {TRAIN_DIR} 不存在")

    # 加载验证数据集
    if os.path.exists(VAL_DIR):
        val_dataset = BreastCancerDataset(VAL_DIR, transform=transform)
        print(f"验证数据集大小: {len(val_dataset)}")
    else:
        print(f"警告: 验证目录 {VAL_DIR} 不存在")

    # 加载测试数据集
    if os.path.exists(TEST_DIR):
        test_dataset = BreastCancerDataset(TEST_DIR, transform=transform)
        print(f"测试数据集大小: {len(test_dataset)}")
    else:
        print(f"警告: 测试目录 {TEST_DIR} 不存在")

    # 如果所有数据集都不存在，使用模拟数据
    if train_dataset is None and val_dataset is None and test_dataset is None:
        print("使用模拟数据进行测试...")
        train_dataset = create_dummy_dataset(100, transform)
        val_dataset = create_dummy_dataset(30, transform)
        test_dataset = create_dummy_dataset(50, transform)
        print(f"模拟训练集: {len(train_dataset)}, 模拟验证集: {len(val_dataset)}, 模拟测试集: {len(test_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 初始化模型
    model = LightweightUNet(n_channels=3, n_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    # 加载预训练模型权重
    if os.path.exists(UNET_MODEL_PATH):
        print(f"加载预训练模型: {UNET_MODEL_PATH}")
        try:
            model.load_state_dict(torch.load(UNET_MODEL_PATH, map_location=DEVICE, weights_only=True))
            print("模型加载成功!")
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("使用随机初始化模型进行测试")
    else:
        print(f"警告: 模型文件 {UNET_MODEL_PATH} 不存在，使用随机初始化模型")

    # 输出评估报告标题
    print("\n" + "="*60)
    print("模型评估报告")
    print("="*60)

    # 评估训练集
    print("\n--- 训练集评估 ---")
    train_results = evaluate_model(model, train_loader, DEVICE)
    print(f"Loss: {train_results['loss']:.4f}")
    print(f"准确率: {train_results['accuracy']:.4f}")
    print(f"精确率: {train_results['precision']:.4f}")
    print(f"召回率: {train_results['recall']:.4f}")
    print(f"F1分数: {train_results['f1']:.4f}")
    print("\n分类报告:")
    print(train_results['classification_report'])

    # 评估验证集
    print("\n--- 验证集评估 ---")
    val_results = evaluate_model(model, val_loader, DEVICE)
    print(f"Loss: {val_results['loss']:.4f}")
    print(f"准确率: {val_results['accuracy']:.4f}")
    print(f"精确率: {val_results['precision']:.4f}")
    print(f"召回率: {val_results['recall']:.4f}")
    print(f"F1分数: {val_results['f1']:.4f}")
    print("\n分类报告:")
    print(val_results['classification_report'])

    # 评估测试集
    print("\n--- 测试集评估 ---")
    test_results = evaluate_model(model, test_loader, DEVICE)
    print(f"Loss: {test_results['loss']:.4f}")
    print(f"准确率: {test_results['accuracy']:.4f}")
    print(f"精确率: {test_results['precision']:.4f}")
    print(f"召回率: {test_results['recall']:.4f}")
    print(f"F1分数: {test_results['f1']:.4f}")
    print("\n分类报告:")
    print(test_results['classification_report'])

    # 创建结果目录
    os.makedirs('results', exist_ok=True)

    # 生成混淆矩阵图
    print("\n--- 生成混淆矩阵图 ---")
    plot_confusion_matrix(test_results['confusion_matrix'], save_path='results/confusion_matrix.png')
    print("混淆矩阵已保存到 results/confusion_matrix.png")

    # 生成类别分布图
    print("\n--- 生成类别分布图 ---")
    plot_class_distribution(test_results['labels'], test_results['predictions'], save_path='results/class_distribution.png')
    print("类别分布图已保存到 results/class_distribution.png")

    # 输出完成信息
    print("\n" + "="*60)
    print("评估完成!")
    print("="*60)


def create_dummy_dataset(size, transform):
    """
    创建模拟数据集（仅用于演示）
    
    参数：
        size: 数据集大小
        transform: 数据预处理管道
    
    返回：
        DummyDataset: 模拟数据集实例
    """
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size, transform):
            self.size = size
            self.transform = transform

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            # 生成随机像素图像
            image = np.random.randint(0, 255, (UNET_INPUT_SIZE, UNET_INPUT_SIZE, 3), dtype=np.uint8)
            # 交替分配标签
            label = idx % NUM_CLASSES

            if self.transform:
                image = self.transform(image)

            return image, label

    return DummyDataset(size, transform)


# 命令行入口
if __name__ == '__main__':
    main()