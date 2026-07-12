"""
LightweightUNet训练脚本

功能说明：
- 训练LightweightUNet模型对乳腺癌病灶进行分类
- 支持真实数据集和模拟数据集（当真实数据不存在时自动使用）
- 使用交叉熵损失函数和Adam优化器
- 训练过程中保存验证准确率最高的模型

训练流程：
1. 加载训练和验证数据集
2. 初始化LightweightUNet模型
3. 设置损失函数和优化器
4. 循环训练：
   - 前向传播计算损失
   - 反向传播更新权重
   - 在验证集上评估性能
5. 保存最佳模型权重
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import numpy as np

from models.lightweight_unet import LightweightUNet
from datasets.breast_cancer_dataset import BreastCancerDataset
from config import TRAIN_DIR, VAL_DIR, MODEL_DIR, UNET_MODEL_PATH, UNET_INPUT_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, DEVICE, NUM_CLASSES


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """
    训练模型的核心函数
    
    参数：
        model: 待训练的模型
        train_loader: 训练集数据加载器
        val_loader: 验证集数据加载器
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练轮数
        device: 运行设备（CPU或GPU）
    
    返回：
        model: 训练完成的模型
    """
    # 记录最佳验证准确率（用于保存最佳模型）
    best_val_acc = 0.0

    # 遍历每个epoch
    for epoch in range(num_epochs):
        # 设置模型为训练模式（开启dropout等训练特有的操作）
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        # 遍历训练集的每个batch
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # 将数据移动到指定设备
            images = images.to(device)
            labels = labels.to(device)

            # 清空梯度（防止梯度累积）
            optimizer.zero_grad()

            # 前向传播：获取模型输出
            cls_out = model(images)
            # 计算损失
            loss = criterion(cls_out, labels)

            # 反向传播：计算梯度
            loss.backward()
            # 更新权重
            optimizer.step()

            # 累积训练损失和预测结果
            train_loss += loss.item() * images.size(0)
            # 获取预测类别（概率最大的类别）
            preds = torch.argmax(cls_out, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())

        # 计算训练集的平均损失和指标
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='weighted')

        # 设置模型为评估模式（关闭dropout等）
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        # 在验证集上评估（关闭梯度计算，节省内存）
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                # 前向传播
                cls_out = model(images)
                # 计算损失
                loss = criterion(cls_out, labels)

                # 累积验证损失和预测结果
                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(cls_out, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())

        # 计算验证集的平均损失和指标
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds, average='weighted', zero_division=0)
        val_recall = recall_score(val_labels, val_preds, average='weighted', zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')

        # 打印训练结果
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f} | Val F1: {val_f1:.4f}")

        # 如果当前验证准确率是最佳的，保存模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 确保模型目录存在
            os.makedirs(MODEL_DIR, exist_ok=True)
            # 保存模型权重
            torch.save(model.state_dict(), UNET_MODEL_PATH)
            print(f"保存最佳模型到 {UNET_MODEL_PATH}")

    # 训练完成，打印最佳验证准确率
    print(f"\n训练完成! 最佳验证准确率: {best_val_acc:.4f}")
    return model


def main():
    """
    主函数：准备数据、初始化模型、开始训练
    """
    # 训练集数据增强：随机翻转、旋转、颜色抖动等
    # 这些操作可以增加数据多样性，防止过拟合
    transform = transforms.Compose([
        transforms.ToPILImage(),                          # numpy数组转PIL图像
        transforms.Resize((UNET_INPUT_SIZE, UNET_INPUT_SIZE)),  # 调整为256x256
        transforms.RandomHorizontalFlip(),                # 随机水平翻转
        transforms.RandomVerticalFlip(),                  # 随机垂直翻转
        transforms.RandomRotation(15),                    # 随机旋转（-15到15度）
        transforms.ColorJitter(                          # 随机颜色抖动
            brightness=0.2,   # 亮度变化范围
            contrast=0.2,     # 对比度变化范围
            saturation=0.2,   # 饱和度变化范围
            hue=0.1           # 色相变化范围
        ),
        transforms.ToTensor(),                            # PIL转张量（0-1范围）
        transforms.Normalize(                             # 标准化（使用ImageNet均值和标准差）
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 验证集数据预处理：不使用随机增广，只进行resize和标准化
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((UNET_INPUT_SIZE, UNET_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 加载训练数据集
    if os.path.exists(TRAIN_DIR):
        # 如果训练目录存在，加载真实数据
        train_dataset = BreastCancerDataset(TRAIN_DIR, transform=transform)
        print(f"训练数据集大小: {len(train_dataset)}")
    else:
        # 如果训练目录不存在，使用模拟数据（仅用于演示）
        print(f"警告: 训练目录 {TRAIN_DIR} 不存在，使用模拟数据进行演示")
        train_dataset = create_dummy_dataset(100, transform)

    # 加载验证数据集
    if os.path.exists(VAL_DIR):
        # 如果验证目录存在，加载真实数据
        val_dataset = BreastCancerDataset(VAL_DIR, transform=val_transform)
        print(f"验证数据集大小: {len(val_dataset)}")
    else:
        # 如果验证目录不存在，使用模拟数据
        print(f"警告: 验证目录 {VAL_DIR} 不存在，使用模拟数据进行演示")
        val_dataset = create_dummy_dataset(20, val_transform)

    # 创建数据加载器（负责批量读取和打乱数据）
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 初始化LightweightUNet模型
    model = LightweightUNet(n_channels=3, n_classes=NUM_CLASSES)
    # 将模型移动到指定设备
    model = model.to(DEVICE)

    # 设置损失函数：交叉熵损失（适用于多分类问题）
    criterion = nn.CrossEntropyLoss()
    # 设置优化器：Adam（自适应学习率优化器）
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 开始训练
    print(f"开始训练, 设备: {DEVICE}")
    train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS, DEVICE)


def create_dummy_dataset(size, transform):
    """
    创建模拟数据集（仅用于演示，非真实医学图像）
    
    参数：
        size: 数据集大小
        transform: 数据预处理管道
    
    返回：
        DummyDataset: 模拟数据集实例
    """
    from datasets.breast_cancer_dataset import BreastCancerDataset

    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size, transform):
            self.size = size
            self.transform = transform

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            # 生成随机像素图像（256x256x3）
            image = np.random.randint(0, 255, (UNET_INPUT_SIZE, UNET_INPUT_SIZE, 3), dtype=np.uint8)
            # 交替分配标签：0=良性，1=恶性
            label = idx % NUM_CLASSES

            # 应用数据预处理
            if self.transform:
                image = self.transform(image)

            return image, label

    return DummyDataset(size, transform)


# 命令行入口
if __name__ == '__main__':
    main()