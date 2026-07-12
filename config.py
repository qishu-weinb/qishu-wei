"""
乳腺癌病灶检测模型 - 配置文件

此文件定义了整个项目的路径、超参数和全局常量。
所有脚本都从这里读取配置，确保一致性。
"""

import os
import torch

# ==================== 基础路径配置 ====================
# 项目根目录：当前文件所在的文件夹
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据目录：存放训练/验证/测试数据集
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')      # 训练集目录
VAL_DIR = os.path.join(DATA_DIR, 'val')          # 验证集目录
TEST_DIR = os.path.join(DATA_DIR, 'test')        # 测试集目录

# 模型目录：存放训练好的模型权重文件
MODEL_DIR = os.path.join(BASE_DIR, 'models')
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, 'yolov8n-seg.pt')      # YOLOv8n分割模型路径
UNET_MODEL_PATH = os.path.join(MODEL_DIR, 'lightweight_unet.pth') # LightweightUNet分类模型路径

# 结果目录：存放测试结果、混淆矩阵、类别分布图等
RESULT_DIR = os.path.join(BASE_DIR, 'results')

# ==================== 数据集配置 ====================
# 分类类别：良性(benign) 和 恶性(malignant) - 适用于超声图像
CLASSES = ['benign', 'malignant']
NUM_CLASSES = len(CLASSES)  # 类别数量：2

# 病理图像分类类别：正常乳腺组织、良性、原位癌、浸润性癌
# 目录结构: normal/, benign/, malignant/inSitu/, malignant/invasive/
PATHOLOGY_CLASSES = ['normal', 'benign', 'malignant/inSitu', 'malignant/invasive']
PATHOLOGY_NUM_CLASSES = len(PATHOLOGY_CLASSES)  # 类别数量：4

# ==================== 模型输入尺寸配置 ====================
YOLO_INPUT_SIZE = 640       # YOLOv8n的输入图像尺寸
UNET_INPUT_SIZE = 256       # LightweightUNet的输入图像尺寸

# ==================== 训练超参数配置 ====================
BATCH_SIZE = 16             # 批次大小：每批处理16张图像
EPOCHS = 50                 # 训练轮数：训练50个epoch
LEARNING_RATE = 1e-4        # 学习率：0.0001

# ==================== 设备配置 ====================
# 自动检测是否有可用的GPU，如果有则使用CUDA，否则使用CPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'