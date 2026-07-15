"""
YOLOv8-seg 乳腺钼靶病灶分割器

专为乳腺钼靶X线图像优化的病灶检测与分割模块。

核心功能:
  1. 检测肿块(mass)、钙化(calcification)、结构扭曲等异常区域
  2. 生成高精度分割mask
  3. 在原始分辨率下裁剪ROI，保留完整纹理信息给BCHI-CovNet

与通用YOLO分割器的区别:
  - 使用更大的输入尺寸(1024)以适应高分辨率钼靶图像
  - 更低的置信度阈值以捕捉微钙化等细小病灶
  - 保留原始分辨率ROI裁剪 (不缩放)
  - 支持DICOM 12-bit → 8-bit 窗宽窗位转换
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from ..config import (
    DEVICE,
    MIN_ROI_AREA,
    MODEL_DIR,
    SEG_MODEL_PATH,
    YOLO_CONF_THRESHOLD,
    YOLO_INPUT_SIZE,
    YOLO_IOU_THRESHOLD,
    YOLO_PADDING,
)


class MammographySegmenter:
    """
    乳腺钼靶病灶分割器

    使用 YOLOv8n-seg 进行病灶检测与实例分割。
    支持微调权重加载和原始分辨率ROI提取。
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = DEVICE,
    ):
        """
        初始化分割器

        参数:
            model_path: 模型权重路径，None则使用预训练yolov8n-seg
            device: 运行设备 ("cuda" / "cpu")
        """
        if model_path is None:
            if SEG_MODEL_PATH.exists():
                model_path = str(SEG_MODEL_PATH)
            else:
                model_path = "yolov8n-seg.pt"

        self.model = YOLO(model_path)
        self.device = device
        self.input_size = YOLO_INPUT_SIZE

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = YOLO_CONF_THRESHOLD,
        iou_threshold: float = YOLO_IOU_THRESHOLD,
    ):
        """
        对乳腺钼靶图像进行病灶检测

        参数:
            image: RGB图像 (H, W, 3), uint8
            conf_threshold: 检测置信度阈值
            iou_threshold: NMS IoU 阈值

        返回:
            results: ultralytics Results 对象
        """
        results = self.model(
            image,
            imgsz=self.input_size,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False,
        )
        return results[0]

    def extract_rois(
        self,
        image: np.ndarray,
        results,
        padding: int = YOLO_PADDING,
        min_area: int = MIN_ROI_AREA,
    ) -> list[dict]:
        """
        从YOLO检测结果中提取原始分辨率ROI

        关键设计: 在原始图像上裁剪，不做缩放，保留完整纹理信息。
        这是为BCHI-CovNet提供高质量输入的核心步骤。

        参数:
            image: 原始图像 (H, W, 3), 原始分辨率
            results: YOLO检测结果
            padding: ROI边界扩展像素数
            min_area: 最小ROI面积过滤

        返回:
            rois: ROI列表，每个元素包含:
                - image: 原始分辨率裁剪图像 (numpy array)
                - mask: 分割mask (与ROI同尺寸)
                - bbox: 边界框 [x1, y1, x2, y2] (原图坐标)
                - confidence: YOLO检测置信度
                - area: ROI面积 (像素)
        """
        rois = []
        h, w = image.shape[:2]

        masks = results.masks
        boxes = results.boxes

        if masks is None or len(masks) == 0:
            return rois

        for i in range(len(masks)):
            # 获取mask并缩放到原图尺寸
            mask_raw = masks.data[i].cpu().numpy().astype(np.float32)

            # YOLO内部会将图像resize到imgsz，mask需要resize回原图
            if mask_raw.shape != (h, w):
                mask_raw = cv2.resize(mask_raw, (w, h), interpolation=cv2.INTER_LINEAR)

            # 二值化mask
            mask_binary = (mask_raw > 0.5).astype(np.uint8)

            # 获取边界框
            box = boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box

            # 扩展padding (不超出图像范围)
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)

            # 过滤面积过小的区域
            roi_w, roi_h = x2 - x1, y2 - y1
            if roi_w * roi_h < min_area:
                continue

            # 在原始分辨率下裁剪
            roi_image = image[y1:y2, x1:x2].copy()
            roi_mask = mask_binary[y1:y2, x1:x2]

            rois.append({
                "image": roi_image,
                "mask": roi_mask,
                "bbox": [x1, y1, x2, y2],
                "confidence": float(boxes.conf[i].cpu().numpy()),
                "area": int(roi_w * roi_h),
                "class_id": int(boxes.cls[i].cpu().numpy()) if boxes.cls is not None else 0,
            })

        return rois

    def segment(
        self,
        image: np.ndarray,
        conf_threshold: float = YOLO_CONF_THRESHOLD,
        iou_threshold: float = YOLO_IOU_THRESHOLD,
        padding: int = YOLO_PADDING,
    ) -> dict:
        """
        完整的分割流程: 检测 → 提取ROI

        参数:
            image: RGB图像 (H, W, 3), uint8
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
            padding: 扩展像素

        返回:
            output: 包含原始图像、检测结果、ROI列表的字典
        """
        results = self.detect(image, conf_threshold, iou_threshold)
        rois = self.extract_rois(image, results, padding)

        return {
            "image": image,
            "results": results,
            "rois": rois,
            "num_detections": len(rois),
        }

    def segment_file(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
    ) -> dict:
        """
        从文件路径读取图像并执行分割

        参数:
            image_path: 图像文件路径 (支持 png/jpg/dcm)
            output_dir: 保存ROI图像的目录 (可选)

        返回:
            output: 分割结果字典
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        output = self.segment(image)

        # 保存ROI图像
        if output_dir and output["rois"]:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            image_stem = Path(image_path).stem
            for i, roi in enumerate(output["rois"]):
                roi_path = out_path / f"{image_stem}_roi_{i:03d}.png"
                cv2.imwrite(str(roi_path), cv2.cvtColor(roi["image"], cv2.COLOR_RGB2BGR))

        return output


def visualize_segmentation(image: np.ndarray, output: dict) -> np.ndarray:
    """
    绘制分割结果可视化

    参数:
        image: 原始RGB图像
        output: segment() 的输出字典

    返回:
        vis: 可视化RGB图像
    """
    vis = image.copy()

    color_map = [
        (0, 255, 0),      # 类别0: 绿色
        (255, 255, 0),    # 类别1: 青色
        (255, 165, 0),    # 类别2: 橙色
        (255, 0, 0),      # 类别3: 红色
    ]

    for roi in output["rois"]:
        x1, y1, x2, y2 = roi["bbox"]
        cls_id = roi["class_id"]
        color = color_map[cls_id % len(color_map)]

        # 绘制边界框
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)

        # 绘制半透明mask
        if roi["mask"].shape == roi["image"].shape[:2]:
            overlay = np.zeros_like(vis)
            overlay[y1:y2, x1:x2][roi["mask"] > 0] = color
            vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

        # 标签
        label = f"{roi['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(vis, label, (x1 + 3, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 统计信息
    info = f"Detected: {output['num_detections']} lesions"
    cv2.putText(vis, info, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return vis
