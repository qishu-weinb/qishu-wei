"""
端到端乳腺钼靶诊断推理脚本

完整的临床级诊断流水线:
  Stage 1: YOLOv8-seg 病灶分割
  Stage 2: BCHI-CovNet ROI分类

支持:
  - 单张图像诊断
  - 批量诊断
  - DICOM直接输入
  - 结果可视化 + 报告生成

用法:
  python -m mammography.inference --image path/to/mammogram.dcm
  python -m mammography.inference --dir path/to/images/ --output results/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import (
    CLS_MODEL_PATH,
    DEVICE,
    OUTPUT_DIR,
    SEG_MODEL_PATH,
)
from .models.pipeline import MammographyPipeline
from .utils.dicom_utils import get_dicom_pixel_array
from .utils.preprocessing import apply_windowing, preprocess_mammogram


def diagnose_single(
    pipeline: MammographyPipeline,
    image_path: str,
    output_dir: Optional[str] = None,
    save_json: bool = True,
    show: bool = False,
) -> dict:
    """
    对单张图像执行诊断

    参数:
        pipeline: 诊断流水线
        image_path: 图像路径 (支持 dcm / png / jpg)
        output_dir: 输出目录
        save_json: 是否保存JSON结果
        show: 是否显示可视化结果

    返回:
        result: 诊断结果字典
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"图像不存在: {image_path}")

    print(f"正在诊断: {image_path.name}")

    # 读取图像
    ext = image_path.suffix.lower()
    if ext == ".dcm":
        pixels = get_dicom_pixel_array(str(image_path))
        image = apply_windowing(pixels)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 预处理 (可选CLAHE增强)
    image, preprocess_meta = preprocess_mammogram(
        image,
        apply_clahe_flag=True,
        remove_background=False,  # 保留完整图像坐标
    )

    # 执行诊断
    result = pipeline.diagnose(image)

    # 输出结果摘要
    summary = result["summary"]
    print(f"  检测到病灶: {result['lesion_count']} 个")
    print(f"  风险等级: {summary['risk_level'].upper()}")
    print(f"  综合置信度: {summary['overall_confidence']:.2%}")
    print(f"  处理耗时: {result['processing_time_seconds']:.1f}s")

    for i, lesion in enumerate(result["lesions"]):
        cls_info = lesion.get("classification", {})
        if cls_info:
            print(f"  病灶 #{i+1}: {cls_info['class_name_cn']} ({cls_info['confidence']:.2%})")
        else:
            print(f"  病灶 #{i+1}: 未分类")

    # 保存结果
    if output_dir:
        pipeline.save_results(result, output_dir, image_path.stem)

    if save_json:
        json_path = Path(output_dir or OUTPUT_DIR) / f"{image_path.stem}_result.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        # 序列化时排除numpy数组
        serializable = _make_serializable(result)
        json_path.write_text(
            json.dumps(serializable, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(f"JSON结果已保存: {json_path}")

    if show:
        vis = pipeline.visualize(image, result)
        cv2.imshow("Diagnosis Result", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result


def diagnose_batch(
    pipeline: MammographyPipeline,
    input_dir: str,
    output_dir: Optional[str] = None,
    pattern: str = "*",
) -> list[dict]:
    """
    批量诊断目录下的所有图像

    参数:
        pipeline: 诊断流水线
        input_dir: 输入图像目录
        output_dir: 输出目录
        pattern: 文件名匹配模式

    返回:
        results: 所有诊断结果列表
    """
    input_path = Path(input_dir)
    exts = {".dcm", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    image_files = sorted([
        f for f in input_path.rglob(pattern)
        if f.suffix.lower() in exts
    ])

    print(f"找到 {len(image_files)} 张图像")
    print("-" * 50)

    all_results = []
    summary_stats = {
        "total": len(image_files),
        "high_risk": 0,
        "medium_risk": 0,
        "low_risk": 0,
        "total_lesions": 0,
    }

    for i, img_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] {img_path.name}")

        try:
            result = diagnose_single(
                pipeline,
                str(img_path),
                output_dir=output_dir,
                save_json=False,
            )
            all_results.append({
                "image": str(img_path.name),
                "result": result,
            })

            # 统计数据
            risk = result["summary"]["risk_level"]
            summary_stats[f"{risk}_risk"] += 1
            summary_stats["total_lesions"] += result["lesion_count"]

        except Exception as e:
            print(f"  ❌ 诊断失败: {e}")
            all_results.append({
                "image": str(img_path.name),
                "error": str(e),
            })

    # 保存批量统计
    print("\n" + "=" * 50)
    print("批量诊断统计")
    print("=" * 50)
    print(f"总图像数: {summary_stats['total']}")
    print(f"高风险: {summary_stats['high_risk']}")
    print(f"中风险: {summary_stats['medium_risk']}")
    print(f"低风险: {summary_stats['low_risk']}")
    print(f"总病灶数: {summary_stats['total_lesions']}")

    if output_dir:
        stats_path = Path(output_dir) / "batch_summary.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(
            json.dumps(summary_stats, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"统计数据已保存: {stats_path}")

    return all_results


def _make_serializable(result: dict) -> dict:
    """将结果中的numpy数组转换为可序列化的格式"""
    out = {}
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            out[key] = f"<numpy array shape={value.shape} dtype={value.dtype}>"
        elif isinstance(value, list):
            out[key] = [
                _make_serializable(item) if isinstance(item, dict) else
                (f"<numpy array shape={item.shape}>" if isinstance(item, np.ndarray) else item)
                for item in value
            ]
        elif isinstance(value, dict):
            out[key] = _make_serializable(value)
        else:
            out[key] = value
    return out


def main():
    parser = argparse.ArgumentParser(
        description="乳腺钼靶AI诊断 - 端到端推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m mammography.inference --image mammogram.dcm
  python -m mammography.inference --dir ./images/ --output ./results/
  python -m mammography.inference --image test.png --show
        """,
    )
    parser.add_argument("--image", type=str, help="单张图像路径")
    parser.add_argument("--dir", type=str, help="批量诊断的图像目录")
    parser.add_argument("--output", type=str, default=None, help="输出目录")
    parser.add_argument("--seg-weights", type=str, default=None, help="YOLO分割器权重路径")
    parser.add_argument("--cls-weights", type=str, default=None, help="BCHI-CovNet分类器权重路径")
    parser.add_argument("--show", action="store_true", help="显示可视化结果")
    parser.add_argument("--no-preprocess", action="store_true", help="跳过图像预处理")

    args = parser.parse_args()

    if not args.image and not args.dir:
        parser.print_help()
        sys.exit(1)

    # 创建流水线
    print("初始化诊断流水线...")
    pipeline = MammographyPipeline(
        seg_model_path=args.seg_weights or (str(SEG_MODEL_PATH) if SEG_MODEL_PATH.exists() else None),
        cls_model_path=args.cls_weights or (str(CLS_MODEL_PATH) if CLS_MODEL_PATH.exists() else None),
    )
    print("流水线就绪!\n")

    # 单张诊断
    if args.image:
        diagnose_single(
            pipeline,
            args.image,
            output_dir=args.output or str(OUTPUT_DIR),
            show=args.show,
        )

    # 批量诊断
    if args.dir:
        diagnose_batch(
            pipeline,
            args.dir,
            output_dir=args.output or str(OUTPUT_DIR / "batch"),
        )


if __name__ == "__main__":
    main()
