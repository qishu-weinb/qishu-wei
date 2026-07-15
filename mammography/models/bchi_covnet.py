"""
BCHI-CovNet: Breast Cancer Histopathology Imaging Convolutional Network

专为乳腺钼靶ROI纹理分类设计的轻量级CNN。

架构特点:
  - 支持可变尺寸输入 (通过自适应全局平均池化)
  - SE (Squeeze-and-Excitation) 通道注意力机制
  - 4层卷积块逐步提取纹理特征
  - 接收原始分辨率ROI，保留完整纹理信息
  - 4分类输出: normal / benign / inSitu / invasive

设计理念:
  传统方案通常将ROI缩放到固定尺寸(如256×256)，这会丢失关键的纹理细节。
  BCHI-CovNet通过自适应池化接受任意尺寸的ROI，保留原始分辨率下的
  微钙化形态、肿块边缘特征等关键诊断信息。
"""

import torch
import torch.nn as nn
from typing import Optional


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation 通道注意力模块

    通过学习每个通道的重要性权重，增强有用特征、抑制无用特征。
    对乳腺钼靶中的微钙化和细小纹理特别有效。
    """

    def __init__(self, channels: int, reduction: int = 16):
        """
        参数:
            channels: 输入通道数
            reduction: 压缩比，控制全连接层的瓶颈维度
        """
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y


class ConvBlock(nn.Module):
    """
    基础卷积块: Conv3x3 → BN → ReLU → Conv3x3 → BN → ReLU

    双卷积结构提取更丰富的局部纹理特征。
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class BCHICovNet(nn.Module):
    """
    BCHI-CovNet 主模型

    网络结构:
      Input (H×W×3, 任意尺寸)
        → ConvBlock (3→32)          → H×W×32
        → MaxPool + ConvBlock (32→64)   → H/2×W/2×64
        → MaxPool + ConvBlock (64→128)  → H/4×W/4×128
        → MaxPool + ConvBlock (128→256) → H/8×W/8×256
        → SEBlock (256, reduction=16)
        → ConvBlock (256→512, stride=2) → H/16×W/16×512
        → SEBlock (512, reduction=16)
        → AdaptiveAvgPool2d(1)          → 1×1×512
        → Flatten + Dropout
        → FC (512→256) + ReLU + Dropout
        → FC (256→4)                    → 分类输出
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        base_channels: int = 32,
        num_blocks: int = 4,
        dropout: float = 0.5,
        use_attention: bool = True,
        reduction: int = 16,
    ):
        """
        参数:
            in_channels: 输入通道数 (灰度图=1, RGB=3)
            num_classes: 分类类别数 (4: normal/benign/inSitu/invasive)
            base_channels: 第一层通道数，后续每层翻倍
            num_blocks: 卷积块数量
            dropout: Dropout比例
            use_attention: 是否使用SE注意力
            reduction: SE注意力压缩比
        """
        super().__init__()
        self.use_attention = use_attention

        # 编码器卷积块
        self.stem = ConvBlock(in_channels, base_channels)

        self.blocks = nn.ModuleList()
        ch = base_channels
        for i in range(num_blocks):
            out_ch = ch * 2
            self.blocks.append(
                nn.Sequential(
                    nn.MaxPool2d(2),
                    ConvBlock(ch, out_ch),
                )
            )
            ch = out_ch

        # 额外一个下采样块 + SE
        self.deep_conv = ConvBlock(ch, ch * 2, stride=2)
        ch = ch * 2

        # 注意力模块
        if use_attention:
            self.se1 = SEBlock(ch // 2, reduction)
            self.se2 = SEBlock(ch, reduction)

        # 分类头
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(ch, ch // 2)
        self.fc2 = nn.Linear(ch // 2, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Kaiming 初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x: 输入图像张量 [B, C, H, W], 支持任意 H×W

        返回:
            logits: 分类 logit [B, num_classes]
        """
        x = self.stem(x)  # [B, 32, H, W]

        for i, block in enumerate(self.blocks):
            x = block(x)  # 每次 MaxPool 尺寸减半
            if self.use_attention and i == len(self.blocks) - 2:
                x = self.se1(x)  # 在倒数第二层后加 SE

        if self.use_attention:
            x = self.se2(x)

        x = self.deep_conv(x)

        # 全局池化 → 分类
        x = self.global_pool(x)  # [B, C, 1, 1]
        x = torch.flatten(x, 1)  # [B, C]
        x = self.dropout(x)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        预测方法: 返回类别和概率

        参数:
            x: 输入图像张量 [B, C, H, W]

        返回:
            preds: 预测类别索引 [B]
            probs: 各类别概率 [B, num_classes]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs

    def get_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        提取特征向量 (用于特征可视化或检索)

        参数:
            x: 输入图像张量 [B, C, H, W]

        返回:
            features: 512维特征向量 [B, 512]
            logits: 分类logit [B, num_classes]
        """
        self.eval()
        with torch.no_grad():
            # 共享编码器部分
            feat = self.stem(x)
            for i, block in enumerate(self.blocks):
                feat = block(feat)
                if self.use_attention and i == len(self.blocks) - 2:
                    feat = self.se1(feat)
            if self.use_attention:
                feat = self.se2(feat)
            feat = self.deep_conv(feat)
            feat = self.global_pool(feat)
            features = torch.flatten(feat, 1)
            logits = self.fc2(nn.functional.relu(self.fc1(features)))
        return features, logits


def create_bchi_covnet(
    num_classes: int = 4,
    pretrained_path: Optional[str] = None,
    device: str = "cpu",
) -> BCHICovNet:
    """
    工厂函数: 创建 BCHI-CovNet 实例并可选加载预训练权重

    参数:
        num_classes: 分类数
        pretrained_path: 预训练权重路径
        device: 运行设备

    返回:
        model: BCHICovNet 实例
    """
    model = BCHICovNet(
        in_channels=3,
        num_classes=num_classes,
        base_channels=32,
        num_blocks=4,
        dropout=0.5,
        use_attention=True,
    )

    if pretrained_path is not None:
        state = torch.load(pretrained_path, map_location=device, weights_only=True)
        # 支持直接加载完整模型或仅权重
        if isinstance(state, dict) and "fc2.weight" in state:
            model.load_state_dict(state, strict=False)
        else:
            model.load_state_dict(state, strict=False)
        print(f"已加载预训练权重: {pretrained_path}")

    model = model.to(device)
    return model
