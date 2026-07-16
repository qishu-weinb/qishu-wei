from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpFuse(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class MultiTaskEfficientNet(nn.Module):
    """EfficientNet encoder with a three-class head and lesion-mask decoder."""

    def __init__(self, pretrained: bool = True, num_classes: int = 3) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        self.features = backbone.features
        self.classifier = nn.Sequential(
            nn.Dropout(0.25), nn.Linear(1280, num_classes)
        )
        self.decode32 = ConvBlock(1280, 256)
        self.up16 = UpFuse(256, 80, 128)
        self.up8 = UpFuse(128, 40, 96)
        self.up4 = UpFuse(96, 24, 64)
        self.up2 = UpFuse(64, 32, 32)
        self.mask_head = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skips = {}
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in (0, 2, 3, 4):
                skips[index] = x
        deep = x
        logits = self.classifier(nn.functional.adaptive_avg_pool2d(deep, 1).flatten(1))
        y = self.decode32(deep)
        y = self.up16(y, skips[4])
        y = self.up8(y, skips[3])
        y = self.up4(y, skips[2])
        y = self.up2(y, skips[0])
        y = nn.functional.interpolate(y, size=(x.shape[-2] * 32, x.shape[-1] * 32), mode="bilinear", align_corners=False)
        mask_logits = self.mask_head(y)
        return logits, mask_logits
