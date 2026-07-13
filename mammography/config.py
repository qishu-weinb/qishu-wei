"""
乳腺钼靶诊断模块 - 配置文件

定义数据路径、模型超参数、类别映射等全局常量。
支持数据集: CBIS-DDSM, VinDr-Mammo
"""

import os
import torch
from pathlib import Path

# ==================== 基础路径 ====================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

# 数据目录
DATA_ROOT = PROJECT_DIR / "data" / "mammography"
PATIENTS_DIR = DATA_ROOT / "patients"

# VinDr-Mammo 数据路径 (下载后自动填充)
VINDR_ROOT = DATA_ROOT / "vindr_mammo"
VINDR_IMAGES_DIR = VINDR_ROOT / "images"
VINDR_FINDING_CSV = VINDR_ROOT / "finding_annotations.csv"
VINDR_BREAST_CSV = VINDR_ROOT / "breast-level_annotations.csv"

# 预处理后的训练数据
YOLO_DATASET_DIR = DATA_ROOT / "yolo_dataset"
CLS_DATASET_DIR = DATA_ROOT / "classification_dataset"

# CBIS-DDSM CSV 标注文件
CSV_FILES = {
    "calc_train": "calc_case_description_train_set.csv",
    "calc_test": "calc_case_description_test_set.csv",
    "mass_train": "mass_case_description_train_set.csv",
    "mass_test": "mass_case_description_test_set.csv",
}

# 模型保存目录
MODEL_DIR = BASE_DIR / "checkpoints"
SEG_MODEL_PATH = MODEL_DIR / "yolov8_mammo_seg.pt"
CLS_MODEL_PATH = MODEL_DIR / "bchi_covnet_best.pth"

# 输出目录
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 数据集配置 ====================

# 病理标签映射 (CBIS-DDSM)
# BENIGN / BENIGN_WITHOUT_CALLBACK → 0 (benign)
# MALIGNANT → 分为 inSitu(2) 和 invasive(3)
LABEL_MAP = {
    "BENIGN": 1,
    "BENIGN_WITHOUT_CALLBACK": 1,
    "MALIGNANT": 3,  # 默认归为invasive，后续可通过病理子类型细分
}

# 四分类 (BCHI-CovNet 输出)
CLASS_NAMES = ["normal", "benign", "inSitu", "invasive"]
CLASS_NAMES_CN = ["正常组织", "良性病变", "原位癌", "浸润性癌"]
NUM_CLASSES = len(CLASS_NAMES)

# 病灶类型
LESION_TYPES = ["mass", "calcification", "asymmetry", "architectural_distortion"]
LESION_TYPES_CN = ["肿块", "钙化", "不对称", "结构扭曲"]

# ==================== YOLOv8-seg 分割器配置 ====================
YOLO_MODEL_NAME = "yolov8n-seg.pt"           # 预训练模型 (自动下载)
YOLO_INPUT_SIZE = 1024                         # 乳腺钼靶使用较大输入尺寸
YOLO_CONF_THRESHOLD = 0.25                     # 检测置信度阈值 (较低以捕捉小病灶)
YOLO_IOU_THRESHOLD = 0.45                      # NMS IoU 阈值
YOLO_PADDING = 20                              # ROI 扩展像素 (保留病灶周围组织)

# ==================== BCHI-CovNet 分类器配置 ====================
# 模型架构
BCHI_BASE_CHANNELS = 32                        # 第一层卷积通道数
BCHI_NUM_BLOCKS = 4                            # 卷积块数量
BCHI_DROPOUT = 0.5                             # Dropout 比例
BCHI_USE_ATTENTION = True                      # 是否使用 SE 通道注意力
BCHI_REDUCTION_RATIO = 16                      # SE 注意力压缩比

# 训练
BCHI_BATCH_SIZE = 32
BCHI_EPOCHS = 60
BCHI_LEARNING_RATE = 1e-4
BCHI_WEIGHT_DECAY = 1e-4
BCHI_LABEL_SMOOTHING = 0.1                     # 标签平滑

# ==================== 训练通用配置 ====================
TRAIN_SPLIT_RATIO = 0.8                        # 训练/验证分割比例
RANDOM_SEED = 42
NUM_WORKERS = 4

# ==================== 图像预处理配置 ====================
# 乳腺钼靶窗宽窗位 (12-bit DICOM)
WINDOW_WIDTH = 4096
WINDOW_LEVEL = 2048

# 归一化参数 (ImageNet 统计)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# ==================== 设备配置 ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== 推理配置 ====================
# 最小ROI尺寸过滤 (过小的区域可能是噪声)
MIN_ROI_AREA = 64 * 64

# 高置信度阈值
HIGH_CONFIDENCE_THRESHOLD = 0.85
