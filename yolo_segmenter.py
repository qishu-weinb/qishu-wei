"""
YOLOSegmenter - YOLOv8n分割器

功能说明：
- 使用YOLOv8n-seg模型对乳腺图像进行病灶检测和分割
- 从原始图像中提取病灶区域（ROI），用于后续分类
- 支持置信度阈值、IoU阈值和padding参数调节

工作流程：
1. 加载YOLOv8n-seg预训练模型
2. 对输入图像进行推理，获取检测框和分割mask
3. 根据mask提取病灶区域，应用padding扩展边界
4. 返回裁剪后的病灶图像和相关信息
"""

import cv2
import numpy as np
from ultralytics import YOLO
from config import YOLO_MODEL_PATH, YOLO_INPUT_SIZE, DEVICE


class YOLOSegmenter:
    """
    YOLOv8n分割器类
    
    负责：
    - 加载YOLO模型
    - 检测图像中的病灶区域
    - 提取并裁剪病灶ROI
    """
    
    def __init__(self, model_path=None):
        """
        初始化分割器
        
        参数：
            model_path: YOLO模型权重文件路径，默认为config中配置的路径
                       如果为None，会自动下载预训练的yolov8n-seg.pt
        """
        if model_path is None:
            model_path = YOLO_MODEL_PATH
        # 加载YOLOv8n分割模型
        self.model = YOLO(model_path)
        # 设置运行设备（CPU或GPU）
        self.device = DEVICE

    def detect_and_segment(self, image, conf_threshold=0.5, iou_threshold=0.45):
        """
        对图像进行病灶检测和分割
        
        参数：
            image: 输入图像，numpy数组，RGB格式
            conf_threshold: 置信度阈值，低于此值的检测结果会被过滤，默认0.5
            iou_threshold: IoU阈值，用于NMS（非极大值抑制），默认0.45
        
        返回：
            results: YOLO推理结果对象，包含检测框(boxes)、分割mask(masks)等信息
        """
        # 运行YOLO推理
        results = self.model(
            image, 
            imgsz=YOLO_INPUT_SIZE,  # 输入图像尺寸
            conf=conf_threshold,     # 置信度阈值
            iou=iou_threshold,       # IoU阈值
            device=self.device       # 运行设备
        )
        # 返回第一张图像的检测结果
        return results[0]

    def extract_lesion_regions(self, image, results, padding=10):
        """
        从检测结果中提取病灶区域
        
        参数：
            image: 原始输入图像，numpy数组，RGB格式
            results: YOLO推理结果对象
            padding: 病灶区域周围的扩展像素数，默认10像素
        
        返回：
            lesions: 病灶区域列表，每个元素包含：
                - image: 裁剪后的病灶图像（已应用mask）
                - mask: 病灶mask
                - bbox: 检测框坐标 [x1, y1, x2, y2]
                - confidence: YOLO检测置信度
        """
        lesions = []
        # 获取检测结果中的mask和box信息
        masks = results.masks
        boxes = results.boxes

        # 如果没有检测到任何病灶，返回空列表
        if masks is None or len(masks) == 0:
            return lesions

        # 获取原始图像尺寸
        original_shape = image.shape[:2]  # (height, width)

        # 遍历每个检测到的病灶
        for i in range(len(masks)):
            # 获取第i个mask并转换为numpy数组（0/1）
            mask = masks.data[i].cpu().numpy().astype(np.uint8)
            # 将mask调整为原始图像尺寸（YOLO内部会resize）
            mask = cv2.resize(mask, (original_shape[1], original_shape[0]))

            # 获取第i个检测框坐标（左上角x1,y1，右下角x2,y2）
            box = boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box

            # 应用padding扩展边界（确保不超出图像范围）
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(original_shape[1], x2 + padding)
            y2 = min(original_shape[0], y2 + padding)

            # 裁剪病灶区域
            lesion_region = image[y1:y2, x1:x2]
            # 裁剪对应的mask
            lesion_mask = mask[y1:y2, x1:x2]

            # 使用mask提取病灶（背景置为黑色）
            masked_lesion = cv2.bitwise_and(lesion_region, lesion_region, mask=lesion_mask)

            # 将病灶信息添加到列表
            lesions.append({
                'image': masked_lesion,     # 裁剪后的病灶图像
                'mask': lesion_mask,        # 病灶mask
                'bbox': [x1, y1, x2, y2],   # 检测框坐标
                'confidence': boxes.conf[i].cpu().numpy()  # YOLO检测置信度
            })

        return lesions

    def process_image(self, image_path, conf_threshold=0.5, iou_threshold=0.45, padding=10):
        """
        完整处理流程：读取图像 -> 检测分割 -> 提取病灶
        
        参数：
            image_path: 图像文件路径
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
            padding: 扩展像素数
        
        返回：
            image: 原始图像（RGB格式）
            lesions: 病灶区域列表
            results: YOLO推理结果
        """
        # 使用OpenCV读取图像（默认BGR格式）
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 转换为RGB格式（YOLO期望RGB输入）
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 执行检测和分割
        results = self.detect_and_segment(image, conf_threshold, iou_threshold)
        # 提取病灶区域
        lesions = self.extract_lesion_regions(image, results, padding)

        return image, lesions, results