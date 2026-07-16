"""
VinDr-Mammo 数据集适配器
=======================

解析 PhysioNet 发布的 VinDr-Mammo 数据集，转换为本项目的两阶段流水线格式：
  Stage 1 (YOLOv8-seg): 将 bbox 标注转为 YOLO 分割格式
  Stage 2 (BCHI-CovNet): 从原始分辨率图像中裁剪 ROI 用于分类

数据集结构:
  vindr_mammo/
  ├── images/                    # DICOM (.dcm) 文件
  ├── breast-level_annotations.csv
  ├── finding_annotations.csv    # 病灶级标注 (bbox, BI-RADS)
  └── metadata.csv

参考文档:
  - PhysioNet: https://physionet.org/content/vindr-mammo/1.0.0/
  - 论文: Nguyen et al., "VinDr-Mammo: A large-scale benchmark dataset
    for computer-aided diagnosis in full-field digital mammography"
"""

import csv
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pydicom

logger = logging.getLogger(__name__)

# VinDr-Mammo 病灶类型 -> 本项目类别映射
FINDING_CATEGORIES = {
    "Mass": "mass",
    "Suspicious Calcification": "calcification",
    "Focal Asymmetry": "asymmetry",
    "Architectural Distortion": "distortion",
    "Asymmetry": "asymmetry",
    "Skin Thickening": "other",
    "Skin Retraction": "other",
    "Nipple Retraction": "other",
    "Lymph Node": "other",
}

# BI-RADS 分类 (0-5) -> 风险等级
BIRADS_TO_CLASS = {
    0: 0,   # Incomplete -> normal (保守)
    1: 1,   # Negative -> benign
    2: 1,   # Benign -> benign
    3: 1,   # Probably benign -> benign
    4: 2,   # Suspicious -> in situ
    5: 3,   # Highly suggestive -> invasive
}


class VinDrDataset:
    """VinDr-Mammo 数据集的解析和转换工具"""

    def __init__(self, root_dir: Union[str, Path]):
        """
        Args:
            root_dir: VinDr-Mammo 数据集根目录
                      应包含 images/ 和 annotation CSV 文件
        """
        self.root = Path(root_dir)
        self.images_dir = self.root / "images"
        self._breast_df = None
        self._finding_df = None
        self._metadata = None

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------

    def load_annotations(self):
        """加载所有 CSV 标注文件"""
        import pandas as pd

        # 病灶级标注
        finding_csv = self.root / "finding_annotations.csv"
        if finding_csv.exists():
            self._finding_df = pd.read_csv(finding_csv, dtype={
                "study_id": str, "series_id": str, "image_id": str,
            })
            logger.info("已加载病灶标注: %d 条记录", len(self._finding_df))
        else:
            logger.warning("未找到 finding_annotations.csv，路径: %s", finding_csv)
            self._finding_df = None

        # 乳房级标注 (BI-RADS 密度)
        breast_csv = self.root / "breast-level_annotations.csv"
        if breast_csv.exists():
            self._breast_df = pd.read_csv(breast_csv, dtype={
                "study_id": str, "series_id": str, "image_id": str,
            })
            logger.info("已加载乳房标注: %d 条记录", len(self._breast_df))
        else:
            logger.warning("未找到 breast-level_annotations.csv")
            self._breast_df = None

    # ------------------------------------------------------------------
    # DICOM 工具
    # ------------------------------------------------------------------

    def find_dicom(self, study_id: str, image_id: str) -> Optional[Path]:
        """根据 study_id 和 image_id 定位 DICOM 文件

        VinDr-Mammo 中 DICOM 文件名包含 image_id(后缀不含 .dcm)。
        搜索 images/<study_id>/ 目录下匹配的文件。
        """
        study_dir = self.images_dir / study_id
        if not study_dir.is_dir():
            return None

        for f in study_dir.iterdir():
            if f.suffix.lower() == ".dcm" and image_id in f.stem:
                return f
        return None

    def read_dicom_info(self, dcm_path: Path) -> dict:
        """读取 DICOM 关键元数据"""
        ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
        return {
            "path": str(dcm_path),
            "rows": getattr(ds, "Rows", None),
            "cols": getattr(ds, "Columns", None),
            "pixel_spacing": getattr(ds, "ImagerPixelSpacing", None),
            "laterality": getattr(ds, "ImageLaterality", ""),
            "view_position": getattr(ds, "ViewPosition", ""),
        }

    # ------------------------------------------------------------------
    # YOLO 分割格式转换
    # ------------------------------------------------------------------

    def export_yolo_dataset(
        self,
        output_dir: Union[str, Path],
        image_subdir: str = "images",
        label_subdir: str = "labels",
        min_size: int = 10,
    ) -> Tuple[int, int]:
        """将 VinDr-Mammo bbox 标注导出为 YOLO 分割格式

        每张 DICOM 生成一张对应的 YOLO 标注 .txt 文件。
        格式: class_id x_center y_center width height (均归一化到 [0,1])

        Args:
            output_dir: 输出目录 (会创建 images/ 和 labels/ 子目录)
            image_subdir: 图像子目录名
            label_subdir: 标注子目录名
            min_size: 最小 bbox 尺寸 (像素)，滤除过小目标

        Returns:
            (exported_images, exported_bboxes) 导出统计
        """
        output_dir = Path(output_dir)
        (output_dir / image_subdir).mkdir(parents=True, exist_ok=True)
        (output_dir / label_subdir).mkdir(parents=True, exist_ok=True)

        if self._finding_df is None:
            self.load_annotations()
        if self._finding_df is None:
            logger.error("无可用的病灶标注")
            return 0, 0

        class_map = self._build_class_map()
        exported_images = 0
        exported_bboxes = 0
        skipped_no_dcm = 0
        skipped_too_small = 0

        # 按 (study_id, image_id) 分组
        for (study_id, image_id), group in self._finding_df.groupby(
            ["study_id", "image_id"]
        ):
            dcm_path = self.find_dicom(str(study_id), str(image_id))
            if dcm_path is None:
                skipped_no_dcm += 1
                continue

            # 获取图像尺寸
            try:
                ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
                rows = ds.Rows
                cols = ds.Columns
            except Exception:
                skipped_no_dcm += 1
                continue

            # 复制/链接 DICOM 文件（转 PNG 最佳在预处理阶段统一做）
            dst_img = output_dir / image_subdir / f"{study_id}_{image_id}.dcm"
            if not dst_img.exists():
                shutil.copy2(dcm_path, dst_img)

            # 写 YOLO 标注
            label_lines = []
            for _, row in group.iterrows():
                xmin = row.get("xmin")
                ymin = row.get("ymin")
                xmax = row.get("xmax")
                ymax = row.get("ymax")

                if any(v is None or pd.isna(v) for v in [xmin, ymin, xmax, ymax]):
                    continue

                xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
                w, h = xmax - xmin, ymax - ymin

                if w < min_size or h < min_size:
                    skipped_too_small += 1
                    continue

                finding_cat = row.get("finding_categories", "other")
                class_id = class_map.get(str(finding_cat).strip(), 0)

                x_center = (xmin + w / 2) / cols
                y_center = (ymin + h / 2) / rows
                width_norm = w / cols
                height_norm = h / rows

                label_lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} "
                    f"{width_norm:.6f} {height_norm:.6f}"
                )

            if label_lines:
                label_path = (
                    output_dir / label_subdir / f"{study_id}_{image_id}.txt"
                )
                label_path.write_text("\n".join(label_lines))
                exported_images += 1
                exported_bboxes += len(label_lines)

        logger.info(
            "YOLO 导出完成: %d 张图, %d 个 bbox, "
            "跳过: %d(无DCM) %d(过小)",
            exported_images, exported_bboxes, skipped_no_dcm, skipped_too_small,
        )
        return exported_images, exported_bboxes

    # ------------------------------------------------------------------
    # BCHI-CovNet 分类数据集生成
    # ------------------------------------------------------------------

    def export_classification_dataset(
        self,
        output_dir: Union[str, Path],
        image_size: Optional[int] = None,
        margin: float = 0.1,
    ) -> int:
        """从 VinDr-Mammo 生成 BCHI-CovNet 分类数据集

        使用 bbox 从原始 DICOM 中裁剪 ROI（不缩放），保存为 PNG。
        每张图可能生成多个 ROI。

        Args:
            output_dir: 输出目录 (按类别分子目录: benign/inSitu/invasive)
            image_size: 如果指定，等比例缩放 ROI 的最长边到此尺寸 (None = 不缩放)
            margin: bbox 扩展比例 (0.1 = 扩展10%)

        Returns:
            生成的 ROI 总数
        """
        output_dir = Path(output_dir)
        for c in ["benign", "inSitu", "invasive", "normal"]:
            (output_dir / c).mkdir(parents=True, exist_ok=True)

        if self._finding_df is None:
            self.load_annotations()
        if self._finding_df is None:
            logger.error("无可用标注")
            return 0

        import cv2
        from PIL import Image

        total_rois = 0

        for _, row in self._finding_df.iterrows():
            study_id = str(row["study_id"])
            image_id = str(row["image_id"])

            dcm_path = self.find_dicom(study_id, image_id)
            if dcm_path is None:
                continue

            try:
                ds = pydicom.dcmread(str(dcm_path))
                img = ds.pixel_array.astype(np.float32)
            except Exception as e:
                logger.warning("读取 DICOM 失败 %s: %s", dcm_path, e)
                continue

            xmin = row.get("xmin")
            ymin = row.get("ymin")
            xmax = row.get("xmax")
            ymax = row.get("ymax")
            if any(v is None or pd.isna(v) for v in [xmin, ymin, xmax, ymax]):
                continue

            xmin, ymin, xmax, ymax = (
                int(float(xmin)), int(float(ymin)),
                int(float(xmax)), int(float(ymax)),
            )

            # 扩展 margin
            h_box, w_box = ymax - ymin, xmax - xmin
            dh, dw = int(h_box * margin), int(w_box * margin)
            y1 = max(0, ymin - dh)
            y2 = min(img.shape[0], ymax + dh)
            x1 = max(0, xmin - dw)
            x2 = min(img.shape[1], xmax + dw)

            roi = img[y1:y2, x1:x2]

            # 归一化到 [0, 255]
            roi = self._normalize_roi(roi)

            # 可选缩放
            if image_size is not None and roi.size > 0:
                h, w = roi.shape[:2]
                scale = image_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

            # 类别映射
            birads = row.get("breast_birads", 0)
            try:
                birads = int(float(birads))
            except (ValueError, TypeError):
                birads = 0
            cls_id = BIRADS_TO_CLASS.get(birads, 1)
            cls_names = {0: "normal", 1: "benign", 2: "inSitu", 3: "invasive"}
            cls_name = cls_names[cls_id]

            # 保存
            roi_name = (
                f"{study_id}_{image_id}_"
                f"x{xmin}y{ymin}w{w_box}h{h_box}_b{birads}.png"
            )
            roi_path = str(output_dir / cls_name / roi_name)
            Image.fromarray(roi.astype(np.uint8)).save(roi_path)

            total_rois += 1

        logger.info("分类数据导出完成: %d 个 ROI", total_rois)
        return total_rois

    # ------------------------------------------------------------------
    # 统计与分析
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """返回数据集统计摘要"""
        if self._finding_df is None:
            self.load_annotations()

        stats = {"total_findings": 0, "categories": {}, "birads_dist": {}}

        if self._finding_df is not None:
            stats["total_findings"] = len(self._finding_df)
            stats["total_images"] = self._finding_df["image_id"].nunique()
            stats["total_studies"] = self._finding_df["study_id"].nunique()

            # 病灶类型分布
            if "finding_categories" in self._finding_df.columns:
                cat_counts = self._finding_df["finding_categories"].value_counts()
                stats["categories"] = cat_counts.to_dict()

            # BI-RADS 分布
            if "breast_birads" in self._finding_df.columns:
                birads_counts = self._finding_df["breast_birads"].value_counts()
                stats["birads_dist"] = birads_counts.to_dict()

        return stats

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _build_class_map() -> Dict[str, int]:
        """构建病灶类型 -> YOLO class_id 的映射"""
        unique_cats = set(FINDING_CATEGORIES.values())
        return {cat: i for i, cat in enumerate(sorted(unique_cats))}

    @staticmethod
    def _normalize_roi(roi: np.ndarray) -> np.ndarray:
        """将 ROI 像素值归一化到 [0, 255]"""
        roi = roi.astype(np.float32)
        p_low, p_high = np.percentile(roi, [1, 99])
        if p_high > p_low:
            roi = np.clip((roi - p_low) / (p_high - p_low) * 255, 0, 255)
        else:
            roi = np.clip(roi, 0, 255)
        return roi


# ====================================================================
# 便捷函数 - 供外部调用
# ====================================================================

def prepare_vindr_for_training(
    vindr_root: str,
    output_root: str,
    yolo_subdir: str = "yolo",
    cls_subdir: str = "classification",
) -> dict:
    """一站式准备 VinDr-Mammo 训练数据

    生成 YOLO 分割数据集 + BCHI-CovNet 分类数据集

    Args:
        vindr_root: VinDr-Mammo 数据根目录
        output_root: 输出根目录
        yolo_subdir: YOLO 数据集子目录名
        cls_subdir: 分类数据集子目录名

    Returns:
        统计字典
    """
    output_root = Path(output_root)
    dataset = VinDrDataset(vindr_root)

    # 打印摘要
    summary = dataset.summary()
    logger.info("VinDr-Mammo 数据摘要: %s", json.dumps(summary, indent=2, ensure_ascii=False))

    # 导出 YOLO 格式
    yolo_dir = output_root / yolo_subdir
    n_yolo_imgs, n_yolo_bboxes = dataset.export_yolo_dataset(yolo_dir)

    # 导出分类 ROI
    cls_dir = output_root / cls_subdir
    n_cls_rois = dataset.export_classification_dataset(cls_dir)

    return {
        "yolo_images": n_yolo_imgs,
        "yolo_bboxes": n_yolo_bboxes,
        "classification_rois": n_cls_rois,
        "summary": summary,
    }
