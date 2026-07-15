"""
模型定义 - ResNet50 + DenseNet121 双模型集成
"""

import torch
import torch.nn as nn
from torchvision import models


class EnsembleModel(nn.Module):
    """
    ResNet50 + DenseNet121 双模型集成
    硬投票规则：两者都预测为1（恶性）才判定为恶性
    """

    def __init__(self):
        super().__init__()

        # ResNet50
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)

        # DenseNet121
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, 2)

    def forward(self, x):
        resnet_out = self.resnet(x)
        densenet_out = self.densenet(x)
        return resnet_out, densenet_out


def hard_vote_predict(resnet_out, densenet_out):
    """硬投票集成预测"""
    r_pred = torch.argmax(resnet_out, dim=1)
    d_pred = torch.argmax(densenet_out, dim=1)
    return (r_pred + d_pred) // 2


def load_model(weight_path, device):
    """加载训练好的模型"""
    model = EnsembleModel().to(device)
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model