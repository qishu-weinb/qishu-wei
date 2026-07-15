"""
乳腺钼靶图像预处理工具

包含:
  - 窗宽窗位调整 (多种方案: DICOM默认、乳腺专用、多窗宽)
  - 胸肌去除与背景抑制
  - 图像增强 (CLAHE直方图均衡化)
  - 伪彩色映射
"""

import cv2
import numpy as np
from typing import Optional, Tuple


def apply_windowing(
    image: np.ndarray,
    window_width: float = 4096,
    window_level: float = 2048,
) -> np.ndarray:
    """
    窗宽窗位映射 (12/14-bit → 8-bit)

    参数:
        image: float32 原始DICOM像素值
        window_width: 窗宽
        window_level: 窗位 (窗中心)

    返回:
        image_8bit: uint8 0-255
    """
    low = window_level - window_width / 2
    high = window_level + window_width / 2
    image = np.clip(image, low, high)
    image = ((image - low) / (high - low) * 255).astype(np.uint8)
    return image


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    CLAHE 自适应直方图均衡化

    增强局部对比度，对乳腺致密组织和微钙化的显示特别有效。

    参数:
        image: 灰度图 (H, W), uint8
        clip_limit: 对比度限制
        tile_size: 网格大小

    返回:
        enhanced: 增强后的图像
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(gray)

    if len(image.shape) == 3:
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    return enhanced


def remove_pectoral_muscle(
    image: np.ndarray,
    threshold: int = 30,
) -> np.ndarray:
    """
    去除胸肌区域 (MLO位摄片)

    使用阈值分割 + 形态学操作去除胸肌高密度区域。

    参数:
        image: 灰度图 (H, W), uint8
        threshold: 胸肌分割阈值

    返回:
        masked: 去掉胸肌后的图像
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # OTSU阈值
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # 形态学闭运算
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 找到最大连通区域 (胸肌)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest], -1, 255, -1)

        # 反掩码 (保留非胸肌区域)
        result = cv2.bitwise_and(gray, gray, mask=cv2.bitwise_not(mask))
    else:
        result = gray

    if len(image.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

    return result


def suppress_background(
    image: np.ndarray,
    threshold: int = 10,
    margin: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    背景抑制：检测并裁剪掉纯黑背景区域

    乳腺钼靶图像通常有大量黑色背景，
    裁剪后可以提高检测效率。

    参数:
        image: 灰度或RGB图像
        threshold: 背景像素阈值
        margin: 裁剪边距

    返回:
        cropped: 裁剪后的图像
        roi_bbox: 裁剪区域 (用于映射回原图) [x1, y1, x2, y2]
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # 二值化
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # 找非背景区域
    coords = cv2.findNonZero(binary)
    if coords is None:
        return image, [0, 0, image.shape[1], image.shape[0]]

    x, y, w, h = cv2.boundingRect(coords)

    # 添加margin
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(image.shape[1], x + w + margin)
    y2 = min(image.shape[0], y + h + margin)

    cropped = image[y1:y2, x1:x2]

    return cropped, [x1, y1, x2, y2]


def preprocess_mammogram(
    image: np.ndarray,
    apply_clahe_flag: bool = True,
    remove_background: bool = True,
    to_rgb: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    完整的乳腺钼靶预处理流水线

    参数:
        image: 输入图像 (可以是12-bit float或8-bit)
        apply_clahe_flag: 是否应用CLAHE
        remove_background: 是否去除背景
        to_rgb: 是否转为RGB三通道

    返回:
        processed: 处理后的图像
        metadata: 预处理元数据 (裁剪坐标等)
    """
    metadata = {"original_shape": image.shape}

    # 如果是float (DICOM原始数据)，先窗宽窗位
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = apply_windowing(image)

    # CLAHE增强
    if apply_clahe_flag:
        image = apply_clahe(image)

    # RGB三通道
    if to_rgb and len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # 背景去除
    if remove_background:
        image, bbox = suppress_background(image)
        metadata["roi_bbox"] = bbox

    metadata["processed_shape"] = image.shape

    return image, metadata


def multi_window_enhancement(
    image_float: np.ndarray,
) -> np.ndarray:
    """
    多窗宽融合增强

    使用多个窗宽窗位组合增强不同密度组织:
      - 软组织窗: 显示肿块
      - 钙化窗: 显示微钙化
      - 宽窗: 显示整体结构

    参数:
        image_float: 原始float32 DICOM像素

    返回:
        fused: 3通道增强图像 (RGB)
    """
    windows = [
        (4096, 2048),   # 软组织窗
        (2048, 1024),   # 钙化细节窗
        (6144, 3072),   # 宽窗
    ]

    channels = []
    for ww, wl in windows:
        ch = apply_windowing(image_float.copy(), ww, wl)
        channels.append(ch)

    return cv2.merge(channels)
