"""
LightweightUNet - 轻量级UNet分类模型

模型架构说明：
- 采用UNet的编码器结构（下采样路径），不包含解码器部分
- 通过5层下采样提取图像特征，通道数从32递增到512
- 最终使用全局平均池化(GAP)将特征图压缩为向量
- 通过全连接层输出分类结果（良性/恶性）

设计意图：
- 只保留编码器部分，减少参数量和计算量，适合轻量化部署
- 使用UNet的编码器能有效提取医学图像中的病灶特征
- 全局平均池化比全连接层更高效，且不易过拟合
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    双重卷积模块：由两个3x3卷积层组成
    
    每个卷积层后接BatchNorm和ReLU激活函数
    这种结构能提取更丰富的特征，同时保持特征图尺寸不变
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            # 第一个卷积层：in_channels -> out_channels
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),  # 批量归一化：加速训练，防止过拟合
            nn.ReLU(inplace=True),          # ReLU激活函数：引入非线性
            # 第二个卷积层：out_channels -> out_channels
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """前向传播：输入特征图经过双重卷积"""
        return self.conv(x)


class Down(nn.Module):
    """
    下采样模块：MaxPooling + DoubleConv
    
    通过2x2最大池化将特征图尺寸缩小一半（宽高各减半）
    通道数翻倍，保持特征总量不变
    """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),              # 2x2最大池化：下采样，尺寸减半
            DoubleConv(in_channels, out_channels)  # 双重卷积：通道数翻倍
        )

    def forward(self, x):
        """前向传播：先下采样，再卷积提取特征"""
        return self.maxpool_conv(x)


class LightweightUNet(nn.Module):
    """
    LightweightUNet主模型类
    
    网络结构（编码器部分）：
    Input (256x256x3) 
        -> inc (DoubleConv 3->32)     -> x1 (256x256x32)
        -> down1 (MaxPool + Conv 32->64)  -> x2 (128x128x64)
        -> down2 (MaxPool + Conv 64->128) -> x3 (64x64x128)
        -> down3 (MaxPool + Conv 128->256) -> x4 (32x32x256)
        -> down4 (MaxPool + Conv 256->512) -> x5 (16x16x512)
        -> GlobalAvgPool (16x16x512 -> 1x1x512)
        -> Flatten (512)
        -> FC (512 -> 2) -> Output (2类分类结果)
    """
    def __init__(self, n_channels=3, n_classes=2):
        """
        初始化模型
        
        参数：
            n_channels: 输入图像通道数，RGB图像为3
            n_classes: 输出类别数，良性/恶性为2
        """
        super(LightweightUNet, self).__init__()
        self.n_channels = n_channels  # 输入通道数
        self.n_classes = n_classes    # 输出类别数

        # 编码器层：从低分辨率到高分辨率特征提取
        self.inc = DoubleConv(n_channels, 32)   # 初始卷积：3通道 -> 32通道
        self.down1 = Down(32, 64)               # 下采样1：32通道 -> 64通道
        self.down2 = Down(64, 128)              # 下采样2：64通道 -> 128通道
        self.down3 = Down(128, 256)             # 下采样3：128通道 -> 256通道
        self.down4 = Down(256, 512)             # 下采样4：256通道 -> 512通道

        # 分类头：将特征图转换为分类结果
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
        self.fc = nn.Linear(512, n_classes)                   # 全连接层：512 -> 2

    def forward(self, x):
        """
        前向传播：输入图像 -> 特征提取 -> 分类结果
        
        参数：
            x: 输入图像张量，形状 [batch_size, 3, 256, 256]
        
        返回：
            cls_out: 分类输出，形状 [batch_size, 2]，每个元素代表对应类别的logit值
        """
        # 编码器前向传播，提取不同层级的特征
        x1 = self.inc(x)    # 初始特征：256x256x32
        x2 = self.down1(x1) # 第一层下采样：128x128x64
        x3 = self.down2(x2) # 第二层下采样：64x64x128
        x4 = self.down3(x3) # 第三层下采样：32x32x256
        x5 = self.down4(x4) # 第四层下采样：16x16x512（最深层特征）

        # 分类头：将最深层特征转换为分类结果
        feat = self.global_avg_pool(x5)  # 全局平均池化：16x16x512 -> 1x1x512
        feat = torch.flatten(feat, 1)    # 展平：512维向量
        cls_out = self.fc(feat)          # 全连接层：512 -> 2

        return cls_out

    def predict(self, x):
        """
        预测方法：返回类别索引和概率（用于推理阶段）
        
        参数：
            x: 输入图像张量，形状 [batch_size, 3, 256, 256]
        
        返回：
            cls_pred: 预测类别索引，形状 [batch_size]
            probabilities: 各类别概率，形状 [batch_size, 2]
        """
        self.eval()  # 设置为评估模式（关闭dropout等训练特有的操作）
        with torch.no_grad():  # 关闭梯度计算，节省内存和计算时间
            cls_out = self.forward(x)
            cls_pred = torch.argmax(cls_out, dim=1)  # 获取概率最大的类别索引
            probabilities = torch.softmax(cls_out, dim=1)  # 将logit转换为概率
        return cls_pred, probabilities