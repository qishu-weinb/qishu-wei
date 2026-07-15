"""
两阶段乳腺钼靶诊断流水线

Stage 1 - YOLOv8-seg 病灶分割:
  对全乳钼靶X线图像进行病灶检测，生成高精度分割mask，
  在原始分辨率下裁剪ROI区域（不缩放）。

Stage 2 - BCHI-CovNet 纹理分类:
  将每个原始分辨率ROI送入BCHI-CovNet进行四分类：
  normal / benign / inSitu / invasive

输出:
  - 每个病灶的位置、分割mask、分类结果和置信度
  - 整体风险评估
  - 可视化结果
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as T

from .bchi_covnet import BCHICovNet
from .yolo_segmenter import MammographySegmenter, visualize_segmentation
from ..config import (
    CLASS_NAMES,
    CLASS_NAMES_CN,
    CLS_MODEL_PATH,
    DEVICE,
    HIGH_CONFIDENCE_THRESHOLD,
    NORM_MEAN,
    NORM_STD,
    NUM_CLASSES,
    YOLO_CONF_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    YOLO_PADDING,
)


class MammographyPipeline:
    """
    乳腺钼靶两阶段诊断流水线

    使用方式:
        pipeline = MammographyPipeline()
        results = pipeline.diagnose("path/to/mammogram.png")
        print(results["report"])
    """

    def __init__(
        self,
        seg_model_path: Optional[str] = None,
        cls_model_path: Optional[str] = None,
        device: str = DEVICE,
    ):
        """
        初始化流水线

        参数:
            seg_model_path: YOLOv8-seg 权重路径
            cls_model_path: BCHI-CovNet 权重路径
            device: 运行设备
        """
        self.device = device

        # Stage 1: 分割器
        self.segmenter = MammographySegmenter(
            model_path=seg_model_path,
            device=device,
        )

        # Stage 2: 分类器
        if cls_model_path is None and CLS_MODEL_PATH.exists():
            cls_model_path = str(CLS_MODEL_PATH)

        self.classifier = BCHICovNet(
            in_channels=3,
            num_classes=NUM_CLASSES,
        )
        if cls_model_path:
            state = torch.load(cls_model_path, map_location=device, weights_only=True)
            self.classifier.load_state_dict(state, strict=False)
            print(f"已加载分类器权重: {cls_model_path}")
        else:
            print("警告: 未加载分类器权重，将使用随机初始化的BCHI-CovNet")

        self.classifier = self.classifier.to(device)
        self.classifier.eval()

        # 分类图像预处理 (ImageNet标准化)
        self.cls_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ])

    def _preprocess_roi(self, roi: np.ndarray) -> torch.Tensor:
        """
        预处理ROI图像送入BCHI-CovNet

        ROI保持原始分辨率，不做resize。
        转换为Tensor并标准化后直接送入网络（网络内部有自适应池化）。

        参数:
            roi: RGB图像 (H, W, 3), uint8, 原始分辨率

        返回:
            tensor: [1, 3, H, W] 处理后的张量
        """
        # 确保是RGB三通道
        if len(roi.shape) == 2:
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
        elif roi.shape[2] == 1:
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)

        # 转为Tensor并标准化
        tensor = self.cls_transform(roi).unsqueeze(0).to(self.device)
        return tensor

    def _classify_rois(self, rois: list[dict]) -> list[dict]:
        """
        对ROI列表进行BCHI-CovNet分类 (支持批量)

        参数:
            rois: segmenter.extract_rois() 的输出列表

        返回:
            rois: 添加了分类结果的ROI列表
        """
        if not rois:
            return rois

        # 逐ROI分类 (因为尺寸不一致无法批量，保持原始分辨率)
        for roi in rois:
            roi_img = roi["image"]
            if roi_img is None or roi_img.size == 0:
                roi["classification"] = None
                roi["pred_class"] = "normal"
                roi["pred_confidence"] = 0.0
                continue

            tensor = self._preprocess_roi(roi_img)

            with torch.no_grad():
                logits = self.classifier(tensor)
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                conf = probs[0, pred].item()

            probs_dict = {
                CLASS_NAMES[i]: round(probs[0, i].item(), 4)
                for i in range(NUM_CLASSES)
            }

            roi["classification"] = {
                "class_id": pred,
                "class_name": CLASS_NAMES[pred],
                "class_name_cn": CLASS_NAMES_CN[pred],
                "confidence": conf,
                "probabilities": probs_dict,
            }
            roi["pred_class"] = CLASS_NAMES[pred]
            roi["pred_confidence"] = conf

        return rois

    def _assess_risk(self, rois: list[dict]) -> dict:
        """
        基于所有病灶分类结果进行整体风险评估

        风险评估规则:
          - 如果检测到invasive → 高风险
          - 如果检测到inSitu → 中高风险
          - 如果仅有benign → 低风险
          - 如果无病灶 → 正常

        参数:
            rois: 已分类的ROI列表

        返回:
            assessment: 风险评估字典
        """
        if not rois:
            return {
                "status": "normal",
                "risk_level": "low",
                "risk_description": "未检测到明显病灶，乳腺组织影像正常。",
                "recommendation": "建议定期随访，保持常规筛查。",
                "overall_confidence": 1.0,
            }

        class_ids = [r["classification"]["class_id"] for r in rois if r.get("classification")]
        confidences = [r["classification"]["confidence"] for r in rois if r.get("classification")]

        if not class_ids:
            return {
                "status": "uncertain",
                "risk_level": "medium",
                "risk_description": "检测到异常区域但分类不明确。",
                "recommendation": "建议进一步影像学检查或临床评估。",
                "overall_confidence": 0.0,
            }

        # 最高风险类别决定整体风险
        max_class = max(class_ids)
        avg_conf = np.mean(confidences) if confidences else 0.0

        if max_class == 3:  # invasive
            risk = {
                "status": "abnormal",
                "risk_level": "high",
                "risk_description": "检测到疑似浸润性癌病灶，需立即临床关注。",
                "recommendation": "强烈建议立即进行活检和病理确认，尽快安排临床干预。",
            }
        elif max_class == 2:  # inSitu
            risk = {
                "status": "abnormal",
                "risk_level": "high",
                "risk_description": "检测到疑似原位癌病灶。",
                "recommendation": "建议进行活检确认，原位癌早期干预预后良好。",
            }
        elif max_class == 1:  # benign
            risk = {
                "status": "abnormal",
                "risk_level": "medium",
                "risk_description": "检测到良性病变特征。",
                "recommendation": "建议定期随访观察，短期内复查确认稳定性。",
            }
        else:
            risk = {
                "status": "uncertain",
                "risk_level": "low",
                "risk_description": "检测到异常区域，倾向正常组织。",
                "recommendation": "建议结合临床症状综合判断。",
            }

        risk["overall_confidence"] = round(float(avg_conf), 4)
        return risk

    def diagnose(
        self,
        image: np.ndarray,
        conf_threshold: float = YOLO_CONF_THRESHOLD,
        iou_threshold: float = YOLO_IOU_THRESHOLD,
        padding: int = YOLO_PADDING,
    ) -> dict:
        """
        执行完整的两阶段诊断

        参数:
            image: RGB乳腺钼靶图像 (H, W, 3), uint8
            conf_threshold: YOLO置信度阈值
            iou_threshold: YOLO IoU阈值
            padding: ROI扩展像素

        返回:
            result: 完整诊断结果字典
        """
        start_time = time.time()

        # Stage 1: 病灶分割
        seg_output = self.segmenter.segment(image, conf_threshold, iou_threshold, padding)

        # Stage 2: ROI分类
        rois = self._classify_rois(seg_output["rois"])

        # 风险评估
        risk = self._assess_risk(rois)

        elapsed = time.time() - start_time

        return {
            "timestamp": datetime.now().isoformat(),
            "image_size": image.shape[:2],
            "processing_time_seconds": round(elapsed, 2),
            "stage1_segmentation": {
                "num_detections": seg_output["num_detections"],
                "conf_threshold": conf_threshold,
            },
            "stage2_classification": {
                "model": "BCHI-CovNet",
                "num_classified": len([r for r in rois if r.get("classification")]),
            },
            "lesions": rois,
            "lesion_count": len(rois),
            "summary": risk,
        }

    def diagnose_file(
        self,
        image_path: str,
        save_visualization: bool = True,
        output_dir: Optional[str] = None,
    ) -> dict:
        """
        从文件路径读取图像并执行诊断

        参数:
            image_path: 图像文件路径
            save_visualization: 是否保存可视化结果
            output_dir: 输出目录

        返回:
            result: 完整诊断结果
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = self.diagnose(image)

        if save_visualization:
            if output_dir is None:
                output_dir = str(Path(image_path).parent / "diagnosis_output")
            self.save_results(result, output_dir, Path(image_path).stem)

        return result

    def visualize(self, image: np.ndarray, result: dict) -> np.ndarray:
        """
        生成完整的可视化诊断图

        包含: 边界框、分类标签、置信度、风险评估

        参数:
            image: 原始RGB图像
            result: diagnose() 的输出

        返回:
            vis: 可视化RGB图像
        """
        vis = image.copy()
        h, w = image.shape[:2]

        # 颜色映射
        color_map = {
            "normal": (0, 255, 0),
            "benign": (0, 255, 255),
            "inSitu": (255, 165, 0),
            "invasive": (255, 0, 0),
        }

        # 绘制每个病灶
        for i, lesion in enumerate(result["lesions"]):
            x1, y1, x2, y2 = lesion["bbox"]
            cls_info = lesion.get("classification", {})
            class_name = cls_info.get("class_name", "unknown") if cls_info else "unknown"
            conf = lesion.get("pred_confidence", 0.0)
            color = color_map.get(class_name, (128, 128, 128))

            # 边界框
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)

            # 半透明mask叠加
            if lesion["mask"].shape[:2] == lesion["image"].shape[:2]:
                overlay = vis.copy()
                overlay[y1:y2, x1:x2][lesion["mask"] > 0] = color
                vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)

            # 标签 (类别 + 置信度)
            cn_name = cls_info.get("class_name_cn", class_name) if cls_info else class_name
            label = f"#{i+1} {cn_name} {conf:.1%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), baseline = cv2.getTextSize(label, font, 0.55, 2)
            cv2.rectangle(vis, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
            cv2.putText(vis, label, (x1 + 4, y1 - 4), font, 0.55, (255, 255, 255), 2)

        # 顶部信息栏
        summary = result["summary"]
        risk_color = {
            "high": (0, 0, 255),
            "medium": (0, 165, 255),
            "low": (0, 255, 0),
        }.get(summary["risk_level"], (255, 255, 255))

        header_h = 100
        cv2.rectangle(vis, (0, 0), (w, header_h), (0, 0, 0), -1)

        lines = [
            f"Risk: {summary['risk_level'].upper()}  |  Lesions: {result['lesion_count']}",
            f"Status: {summary['status']}  |  Confidence: {summary['overall_confidence']:.1%}",
            f"Time: {result['processing_time_seconds']:.1f}s",
        ]
        for i, line in enumerate(lines):
            cv2.putText(vis, line, (15, 25 + i * 25), font, 0.6, (255, 255, 255), 2)

        return vis

    def save_results(
        self,
        result: dict,
        output_dir: str,
        prefix: str = "diagnosis",
    ):
        """
        保存诊断结果到文件

        输出:
          - {prefix}_visualization.png: 可视化图像
          - {prefix}_report.txt: 文本诊断报告
          - {prefix}_rois/: ROI裁剪图像

        参数:
            result: diagnose() 的输出
            output_dir: 输出目录
            prefix: 文件名前缀
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 可视化
        if result.get("image") is not None:
            vis = self.visualize(result["image"], result)
            vis_path = out / f"{prefix}_visualization.png"
            cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            print(f"可视化已保存: {vis_path}")

        # ROI裁剪
        if result["lesions"]:
            roi_dir = out / f"{prefix}_rois"
            roi_dir.mkdir(exist_ok=True)
            for i, lesion in enumerate(result["lesions"]):
                cls_name = lesion.get("pred_class", "unknown")
                roi_path = roi_dir / f"roi_{i:03d}_{cls_name}.png"
                cv2.imwrite(str(roi_path), cv2.cvtColor(lesion["image"], cv2.COLOR_RGB2BGR))
            print(f"ROI已保存: {roi_dir} ({len(result['lesions'])} 个)")

        # 文本报告
        report = self._generate_report(result)
        report_path = out / f"{prefix}_report.txt"
        report_path.write_text(report, encoding="utf-8")
        print(f"报告已保存: {report_path}")

    def _generate_report(self, result: dict) -> str:
        """生成文本诊断报告"""
        lines = []
        lines.append("=" * 60)
        lines.append("  乳腺钼靶X线摄影 AI 辅助诊断报告")
        lines.append("=" * 60)
        lines.append(f"检查时间: {result['timestamp']}")
        lines.append(f"图像尺寸: {result['image_size']}")
        lines.append(f"处理耗时: {result['processing_time_seconds']:.2f} 秒")
        lines.append("")

        summary = result["summary"]
        lines.append("【整体评估】")
        lines.append(f"  状态: {summary['status']}")
        lines.append(f"  风险等级: {summary['risk_level'].upper()}")
        lines.append(f"  综合置信度: {summary['overall_confidence']:.2%}")
        lines.append("")
        lines.append(f"  评估说明: {summary['risk_description']}")
        lines.append(f"  临床建议: {summary['recommendation']}")
        lines.append("")

        if result["lesions"]:
            lines.append(f"【病灶详情】共检测到 {result['lesion_count']} 个病灶区域")
            lines.append("")
            for i, lesion in enumerate(result["lesions"]):
                cls_info = lesion.get("classification", {})
                lines.append(f"  病灶 #{i+1}:")
                x1, y1, x2, y2 = lesion["bbox"]
                lines.append(f"    位置: [{x1}, {y1}, {x2}, {y2}]")
                lines.append(f"    YOLO检测置信度: {lesion['confidence']:.2%}")
                lines.append(f"    病灶面积: {lesion['area']} px²")

                if cls_info:
                    lines.append(f"    分类: {cls_info['class_name_cn']} ({cls_info['class_name']})")
                    lines.append(f"    分类置信度: {cls_info['confidence']:.2%}")
                    lines.append(f"    概率分布:")
                    for cn, en in zip(["正常", "良性", "原位癌", "浸润性癌"], result["lesions"][0].get("classification", {}).get("probabilities", {}).keys() if i == 0 else []):
                        pass
                    for cls_name, prob in cls_info.get("probabilities", {}).items():
                        cn_map = dict(zip(["normal", "benign", "inSitu", "invasive"], ["正常", "良性", "原位癌", "浸润性癌"]))
                        lines.append(f"      {cn_map.get(cls_name, cls_name)}: {prob:.2%}")
                lines.append("")
        else:
            lines.append("【未检测到明确病灶】")
            lines.append("  乳腺组织影像无明显异常征象。")
            lines.append("")

        lines.append("=" * 60)
        lines.append("⚠ 免责声明：本报告由AI自动生成，仅供临床参考，")
        lines.append("  不能替代专业放射科医生的诊断意见。")
        lines.append("=" * 60)

        return "\n".join(lines)


def create_pipeline(
    seg_weights: Optional[str] = None,
    cls_weights: Optional[str] = None,
) -> MammographyPipeline:
    """
    工厂函数: 创建诊断流水线

    参数:
        seg_weights: YOLO分割器权重路径
        cls_weights: BCHI-CovNet分类器权重路径

    返回:
        pipeline: 配置好的 MammographyPipeline
    """
    return MammographyPipeline(
        seg_model_path=seg_weights,
        cls_model_path=cls_weights,
    )
