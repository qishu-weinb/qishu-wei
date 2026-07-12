import argparse
import os

from results.multimodal.common.wsi_utils import process_wsi_folder


def main():
    parser = argparse.ArgumentParser(description="WSI预处理脚本 - 将全切片病理图像裁剪为tiles")
    parser.add_argument("--input-dir", required=True, help="输入WSI文件夹路径，包含normal/benign/malignant_in_situ/malignant_invasive子目录")
    parser.add_argument("--output-dir", required=True, help="输出tiles文件夹路径")
    parser.add_argument("--tile-size", type=int, default=256, help="tile尺寸（默认256x256）")
    parser.add_argument("--stride", type=int, default=256, help="步长（默认等于tile尺寸，无重叠）")
    parser.add_argument("--level", type=int, default=0, help="WSI层级（0为最高分辨率）")
    parser.add_argument("--magnification", type=float, default=None, help="目标放大倍数（如20x），设置后会自动选择合适的层级")
    parser.add_argument("--min-tissue-ratio", type=float, default=0.1, help="最小组织比例阈值，低于此值的tile会被过滤（默认0.1）")
    
    args = parser.parse_args()
    
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"tile尺寸: {args.tile_size}")
    print(f"步长: {args.stride}")
    print(f"层级: {args.level}")
    print(f"目标放大倍数: {args.magnification}x" if args.magnification else "使用指定层级")
    print(f"最小组织比例: {args.min_tissue_ratio}")
    print()
    
    process_wsi_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        tile_size=args.tile_size,
        stride=args.stride,
        level=args.level,
        min_tissue_ratio=args.min_tissue_ratio,
        magnification=args.magnification
    )
    
    print("\nWSI预处理完成!")


if __name__ == "__main__":
    main()
