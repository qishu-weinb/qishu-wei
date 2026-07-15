"""
推理主程序 - 支持单张和批量图像预测
"""

import argparse
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

from config import WEIGHT_PATH, IMG_SIZE, NORM_MEAN, NORM_STD, DEVICE, CLASS_LABELS
from model import load_model, hard_vote_predict


def get_transform():
    """获取图像预处理流程"""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])


def predict_single(model, image_path, device):
    """单张图像预测"""
    transform = get_transform()
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        resnet_out, densenet_out = model(input_tensor)
        pred = hard_vote_predict(resnet_out, densenet_out)

        r_pred = torch.argmax(resnet_out, dim=1)
        d_pred = torch.argmax(densenet_out, dim=1)
        confidence = 1.0 if r_pred.item() == d_pred.item() else 0.5

    return pred.item(), confidence


def predict_batch(model, image_paths, device):
    """批量图像预测"""
    transform = get_transform()
    batch_tensors = []
    valid_paths = []

    for path in image_paths:
        try:
            image = Image.open(path).convert('RGB')
            batch_tensors.append(transform(image))
            valid_paths.append(path)
        except Exception as e:
            print(f"警告: 无法加载 {path.name}: {e}")

    if not batch_tensors:
        return [], [], []

    batch = torch.stack(batch_tensors).to(device)

    with torch.no_grad():
        resnet_out, densenet_out = model(batch)
        preds = hard_vote_predict(resnet_out, densenet_out)

        r_preds = torch.argmax(resnet_out, dim=1)
        d_preds = torch.argmax(densenet_out, dim=1)
        confidences = [1.0 if r.item() == d.item() else 0.5
                       for r, d in zip(r_preds, d_preds)]

    return preds.cpu().numpy(), confidences, valid_paths


def get_image_paths(directory):
    """获取目录下所有图像文件"""
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")

    extensions = ['.png', '.jpg', '.jpeg']
    paths = []
    for ext in extensions:
        paths.extend(dir_path.glob(f'*{ext}'))
        paths.extend(dir_path.glob(f'*{ext.upper()}'))
    return sorted(set(paths))


def main():
    parser = argparse.ArgumentParser(description='病理图像智能分类系统')

    parser.add_argument('--image', type=str, help='单张图像路径')
    parser.add_argument('--dir', type=str, help='图像目录路径（批量推理）')
    parser.add_argument('--weight', type=str, default=str(WEIGHT_PATH), help='权重文件路径')
    parser.add_argument('--confidence', action='store_true', help='显示置信度')

    args = parser.parse_args()

    if not args.image and not args.dir:
        print("错误: 请指定 --image 或 --dir 参数")
        parser.print_help()
        return

    device = torch.device(DEVICE)
    print(f"使用设备: {device}")

    weight_path = Path(args.weight)
    if not weight_path.exists():
        print(f"错误: 权重文件不存在: {weight_path}")
        return

    print(f"加载模型: {weight_path}")
    model = load_model(weight_path, device)
    print("模型加载成功！")

    # ---------- 单张推理 ----------
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"错误: 图像不存在: {image_path}")
            return

        print(f"\n{'='*50}")
        print(f"单张推理: {image_path.name}")
        print(f"{'='*50}")

        pred, conf = predict_single(model, image_path, device)
        label = CLASS_LABELS.get(pred, '未知')

        print(f"预测类别: {pred} ({label})")
        if args.confidence:
            print(f"置信度: {conf:.0%}")

    # ---------- 批量推理 ----------
    elif args.dir:
        image_paths = get_image_paths(args.dir)

        if not image_paths:
            print(f"警告: 在 {args.dir} 中未找到任何图像文件")
            return

        print(f"\n{'='*50}")
        print(f"批量推理: {args.dir}")
        print(f"发现 {len(image_paths)} 张图像")
        print(f"{'='*50}\n")

        preds, confidences, valid_paths = predict_batch(model, image_paths, device)

        stats = {0: 0, 1: 0}
        print(f"{'序号':<6} {'文件名':<40} {'预测结果':<12} {'置信度':<10}")
        print("-" * 70)

        for i, (path, pred, conf) in enumerate(zip(valid_paths, preds, confidences), 1):
            label = CLASS_LABELS.get(int(pred), '未知')
            stats[int(pred)] += 1
            conf_str = f"{conf:.0%}" if args.confidence else "-"
            print(f"{i:<6} {path.name:<40} {label:<12} {conf_str:<10}")

        print("-" * 70)
        total = sum(stats.values())
        print(f"\n统计汇总:")
        print(f"  良性 (0): {stats[0]} 张 ({stats[0]/total*100:.1f}%)")
        print(f"  恶性 (1): {stats[1]} 张 ({stats[1]/total*100:.1f}%)")
        print(f"  总计: {total} 张")

    print("\n推理完成！")


if __name__ == '__main__':
    main()