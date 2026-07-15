"""
DICOM 工具函数

乳腺钼靶DICOM文件特有处理:
  - 读取患者信息和采集参数
  - 提取乳腺摄影特有标签 (Laterality, ViewPosition, etc.)
  - 批量DICOM→PNG转换
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pydicom

# 乳腺钼靶相关DICOM标签
MAMMOGRAPHY_TAGS = {
    "PatientID": (0x0010, 0x0020),
    "PatientName": (0x0010, 0x0010),
    "PatientAge": (0x0010, 0x1010),
    "PatientSex": (0x0010, 0x0040),
    "StudyDate": (0x0008, 0x0020),
    "StudyDescription": (0x0008, 0x1030),
    "Modality": (0x0008, 0x0060),
    "Laterality": (0x0020, 0x0062),
    "ViewPosition": (0x0018, 0x5101),
    "ImageLaterality": (0x0020, 0x0062),
    "WindowWidth": (0x0028, 0x1051),
    "WindowCenter": (0x0028, 0x1050),
    "PixelSpacing": (0x0028, 0x0030),
    "BitsStored": (0x0028, 0x0101),
    "Rows": (0x0028, 0x0010),
    "Columns": (0x0028, 0x0011),
    "Manufacturer": (0x0008, 0x0070),
    "BodyPartThickness": (0x0018, 0x11A0),
    "CompressionForce": (0x0018, 0x11A2),
    "KVP": (0x0018, 0x0060),
    "Exposure": (0x0018, 0x1152),
}


def read_dicom_metadata(dcm_path: str) -> dict:
    """
    读取DICOM文件元数据

    参数:
        dcm_path: DICOM文件路径

    返回:
        metadata: 包含患者信息、采集参数等
    """
    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)

    metadata = {}
    for name, tag in MAMMOGRAPHY_TAGS.items():
        if tag in ds:
            value = ds[tag].value
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            elif isinstance(value, pydicom.valuerep.PersonName):
                value = str(value)
            metadata[name] = value
        else:
            metadata[name] = None

    return metadata


def get_dicom_pixel_array(dcm_path: str) -> np.ndarray:
    """
    读取DICOM像素数组（不做窗宽窗位映射）

    参数:
        dcm_path: DICOM文件路径

    返回:
        pixels: float32像素数组
    """
    ds = pydicom.dcmread(dcm_path)
    return ds.pixel_array.astype(np.float32)


def find_dicom_files(
    directory: str,
    pattern: str = "*.dcm",
    recursive: bool = True,
) -> List[Path]:
    """
    查找目录下的所有DICOM文件

    参数:
        directory: 根目录
        pattern: 文件名匹配模式
        recursive: 是否递归搜索

    返回:
        paths: DICOM文件路径列表
    """
    dir_path = Path(directory)
    if recursive:
        return sorted(dir_path.rglob(pattern))
    else:
        return sorted(dir_path.glob(pattern))


def batch_dicom_to_png(
    input_dir: str,
    output_dir: str,
    window_width: float = 4096,
    window_level: float = 2048,
):
    """
    批量DICOM → PNG转换

    参数:
        input_dir: 输入DICOM目录
        output_dir: 输出PNG目录
        window_width: 窗宽
        window_level: 窗位
    """
    import cv2
    from .preprocessing import apply_windowing

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    dcm_files = find_dicom_files(input_dir)
    print(f"找到 {len(dcm_files)} 个DICOM文件")

    for dcm_path in dcm_files:
        pixels = get_dicom_pixel_array(str(dcm_path))
        img_8bit = apply_windowing(pixels, window_width, window_level)

        rel_path = dcm_path.relative_to(input_dir)
        png_path = out_path / rel_path.with_suffix(".png")
        png_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(png_path), img_8bit)

    print(f"转换完成，保存至: {output_dir}")
