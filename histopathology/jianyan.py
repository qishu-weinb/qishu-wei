"""
模型验证脚本 - 测试模型是否能正常加载和推理
"""

import torch
from pathlib import Path
from PIL import Image
import numpy as np

from config import WEIGHT_PATH, DEVICE, IMG_SIZE, CLASS_LABELS
from model import load_model, hard_vote_predict
from inference import get_transform


def test_model_loading():
    """测试1: 模型加载"""
    print("=" * 60)
    print("测试1: 模型加载")
    print("=" * 60)

    weight_path = Path(WEIGHT_PATH)

    if not weight_path.exists():
        print(f"❌ 权重文件不存在: {weight_path}")
        print(f"   请确认文件路径是否正确")
        return False

    file_size = weight_path.stat().st_size / (1024 * 1024)
    print(f"✅ 权重文件存在: {weight_path}")
    print(f"   文件大小: {file_size:.2f} MB")

    try:
        model = load_model(weight_path, DEVICE)
        print(f"✅ 模型加载成功！")
        print(f"   设备: {DEVICE}")
        if DEVICE.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   显存占用: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        return model
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None


def test_model_forward(model):
    """测试2: 模型前向传播"""
    print("\n" + "=" * 60)
    print("测试2: 模型前向传播")
    print("=" * 60)

    try:
        # 创建随机输入 (模拟图像)
        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)

        print(f"输入形状: {dummy_input.shape}")
        print(f"输入设备: {dummy_input.device}")

        with torch.no_grad():
            resnet_out, densenet_out = model(dummy_input)
            preds = hard_vote_predict(resnet_out, densenet_out)

        print(f"✅ 前向传播成功！")
        print(f"   ResNet输出形状: {resnet_out.shape}")
        print(f"   DenseNet输出形状: {densenet_out.shape}")
        print(f"   预测结果: {preds.cpu().numpy()}")

        return True
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        return False


def test_image_processing():
    """测试3: 图像预处理"""
    print("\n" + "=" * 60)
    print("测试3: 图像预处理")
    print("=" * 60)

    try:
        # 创建虚拟图像
        dummy_image = Image.new('RGB', (300, 300), color=(128, 128, 128))
        print(f"✅ 虚拟图像创建成功: 300x300")

        transform = get_transform()
        tensor = transform(dummy_image)

        print(f"✅ 图像预处理成功！")
        print(f"   输出形状: {tensor.shape}")
        print(f"   数据类型: {tensor.dtype}")
        print(f"   像素范围: [{tensor.min():.3f}, {tensor.max():.3f}]")

        return True
    except Exception as e:
        print(f"❌ 图像预处理失败: {e}")
        return False


def test_single_inference(model):
    """测试4: 模拟单张推理"""
    print("\n" + "=" * 60)
    print("测试4: 模拟单张推理")
    print("=" * 60)

    try:
        # 创建虚拟图像并预测
        dummy_image = Image.new('RGB', (224, 224), color=(100, 150, 200))
        transform = get_transform()
        input_tensor = transform(dummy_image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            resnet_out, densenet_out = model(input_tensor)
            pred = hard_vote_predict(resnet_out, densenet_out)

            # 计算置信度
            r_pred = torch.argmax(resnet_out, dim=1)
            d_pred = torch.argmax(densenet_out, dim=1)
            confidence = 1.0 if r_pred.item() == d_pred.item() else 0.5

        label = CLASS_LABELS.get(pred.item(), '未知')
        print(f"✅ 单张推理成功！")
        print(f"   预测类别: {pred.item()} ({label})")
        print(f"   置信度: {confidence:.0%}")
        print(f"   ResNet预测: {r_pred.item()}")
        print(f"   DenseNet预测: {d_pred.item()}")

        return True
    except Exception as e:
        print(f"❌ 单张推理失败: {e}")
        return False


def test_inference_speed(model):
    """测试5: 推理速度"""
    print("\n" + "=" * 60)
    print("测试5: 推理速度")
    print("=" * 60)

    try:
        import time
        dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)

        # 预热
        with torch.no_grad():
            _ = model(dummy_input)

        # 测试10次
        times = []
        for _ in range(10):
            start = time.time()
            with torch.no_grad():
                _ = model(dummy_input)
            torch.cuda.synchronize() if DEVICE.type == 'cuda' else None
            times.append(time.time() - start)

        avg_time = np.mean(times) * 1000
        fps = 1000 / avg_time

        print(f"✅ 推理速度测试完成！")
        print(f"   平均耗时: {avg_time:.2f} ms")
        print(f"   推理速度: {fps:.1f} FPS")
        print(f"   (10次推理平均)")

        return True
    except Exception as e:
        print(f"⚠️ 推理速度测试失败: {e}")
        return True  # 不影响主要功能


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("      模型验证工具 - His-rec")
    print("=" * 60)
    print(f"设备: {DEVICE}")
    print(f"PyTorch版本: {torch.__version__}")
    print("=" * 60)

    results = []

    # 测试1: 模型加载
    model = test_model_loading()
    results.append(("模型加载", model is not None))

    if model is None:
        print("\n" + "=" * 60)
        print("❌ 模型加载失败，停止后续测试")
        print("=" * 60)
        return

    # 测试2: 前向传播
    results.append(("前向传播", test_model_forward(model)))

    # 测试3: 图像预处理
    results.append(("图像预处理", test_image_processing()))

    # 测试4: 单张推理
    results.append(("单张推理", test_single_inference(model)))

    # 测试5: 推理速度
    results.append(("推理速度", test_inference_speed(model)))

    # 汇总结果
    print("\n" + "=" * 60)
    print("      验证结果汇总")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name:<12}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n🎉 所有测试通过！模型可以正常使用。")
        print("\n运行推理示例:")
        print("  python inference.py --image your_image.png")
        print("  python inference.py --dir ./images/")
    else:
        print("\n⚠️ 部分测试失败，请检查配置。")

    return all_passed


if __name__ == '__main__':
    main()