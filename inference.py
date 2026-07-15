"""
BreastCancerDetector - 乳腺癌病灶检测推理管道

功能说明：
- 端到端的乳腺癌病灶检测系统
- 第一阶段：使用YOLOv8n-seg检测并分割病灶区域
- 第二阶段：使用LightweightUNet对病灶区域进行良性/恶性分类
- 支持结果可视化和保存

完整推理流程：
1. 加载YOLOv8n-seg和LightweightUNet模型
2. 读取输入图像
3. YOLO检测病灶，提取ROI
4. 对每个病灶ROI进行UNet分类
5. 返回检测结果和可视化图像
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from models.yolo_segmenter import YOLOSegmenter
from models.lightweight_unet import LightweightUNet
from config import UNET_MODEL_PATH, UNET_INPUT_SIZE, DEVICE, CLASSES, NUM_CLASSES


class BreastCancerDetector:
    """
    乳腺癌病灶检测器类
    
    整合YOLOv8n分割和LightweightUNet分类的端到端系统
    """
    
    def __init__(self, yolo_model_path=None, unet_model_path=None):
        """
        初始化检测器
        
        参数：
            yolo_model_path: YOLO模型权重路径，默认使用预训练模型
            unet_model_path: UNet模型权重路径，默认使用训练好的模型
        """
        # 初始化YOLO分割器
        self.yolo_segmenter = YOLOSegmenter(yolo_model_path)

        # 初始化LightweightUNet分类模型
        self.unet = LightweightUNet(n_channels=3, n_classes=NUM_CLASSES)
        # 将模型移动到指定设备（CPU或GPU）
        self.unet = self.unet.to(DEVICE)

        # 如果提供了UNet模型权重路径，加载权重
        if unet_model_path is not None:
            try:
                self.unet.load_state_dict(
                    torch.load(unet_model_path, map_location=DEVICE, weights_only=True)
                )
            except Exception as e:
                print(f"加载模型权重失败: {e}")

        # 定义图像预处理管道（与训练时保持一致）
        self.transform = transforms.Compose([
            transforms.ToPILImage(),                          # numpy数组转PIL图像
            transforms.Resize((UNET_INPUT_SIZE, UNET_INPUT_SIZE)),  # 调整为256x256
            transforms.ToTensor(),                            # PIL转张量（0-1范围）
            transforms.Normalize(                             # 标准化（使用ImageNet均值和标准差）
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # 将UNet设置为评估模式
        self.unet.eval()

    def classify_lesion(self, lesion_image):
        """
        对单个病灶区域进行分类
        
        参数：
            lesion_image: 病灶图像，numpy数组，RGB格式
        
        返回：
            pred_class: 预测类别索引（0=良性，1=恶性），失败返回None
            confidence: 预测置信度（0-1），失败返回0.0
        """
        # 检查图像是否有效
        if lesion_image is None or lesion_image.size == 0:
            return None, 0.0

        # 图像预处理：转换为张量并标准化
        input_tensor = self.transform(lesion_image).unsqueeze(0).to(DEVICE)

        # 推理阶段：关闭梯度计算
        with torch.no_grad():
            # 前向传播获取分类结果
            cls_out = self.unet(input_tensor)
            # 将logit转换为概率
            probabilities = torch.softmax(cls_out, dim=1)
            # 获取概率最大的类别索引
            pred_class = torch.argmax(probabilities, dim=1).item()
            # 获取最大概率值
            confidence = probabilities[0, pred_class].item()

        return pred_class, confidence

    def detect(self, image_path, conf_threshold=0.5, padding=10):
        """
        完整检测流程：检测病灶并分类
        
        参数：
            image_path: 输入图像路径
            conf_threshold: YOLO置信度阈值
            padding: 病灶区域扩展像素数
        
        返回：
            original_image: 原始图像（RGB格式）
            detections: 检测结果列表，每个元素包含：
                - bbox: 检测框坐标
                - class: 分类类别（'benign'或'malignant'）
                - confidence: 分类置信度
                - yolo_confidence: YOLO检测置信度
                - mask: 病灶mask
                - image: 病灶图像
            results: YOLO推理结果
        """
        # 使用YOLO处理图像，获取原始图像、病灶列表和YOLO结果
        original_image, lesions, results = self.yolo_segmenter.process_image(
            image_path, 
            conf_threshold=conf_threshold, 
            padding=padding
        )

        # 存储检测结果
        detections = []
        # 遍历每个检测到的病灶
        for lesion in lesions:
            # 使用UNet对病灶进行分类
            pred_class, confidence = self.classify_lesion(lesion['image'])

            # 如果分类成功
            if pred_class is not None:
                detections.append({
                    'bbox': lesion['bbox'],           # 检测框坐标
                    'class': CLASSES[pred_class],      # 分类类别名称
                    'confidence': confidence,          # 分类置信度
                    'yolo_confidence': lesion['confidence'],  # YOLO检测置信度
                    'mask': lesion['mask'],            # 病灶mask
                    'image': lesion['image']           # 病灶图像
                })

        return original_image, detections, results

    def visualize_results(self, image, detections):
        """
        可视化检测结果：在图像上绘制检测框和标签
        
        参数：
            image: 原始图像（RGB格式）
            detections: 检测结果列表
        
        返回：
            vis_image: 可视化后的图像（RGB格式）
        """
        # 复制图像以避免修改原始图像
        vis_image = image.copy()

        # 遍历每个检测结果
        for det in detections:
            # 获取检测框坐标
            x1, y1, x2, y2 = det['bbox']
            # 创建标签文本（类别 + 置信度）
            label = f"{det['class']}: {det['confidence']:.2f}"
            # 设置颜色：良性为绿色，恶性为红色
            color = (0, 255, 0) if det['class'] == 'benign' else (255, 0, 0)

            # 绘制检测框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

            # 计算标签文本尺寸
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            # 绘制标签背景框
            cv2.rectangle(vis_image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            # 绘制标签文本
            cv2.putText(vis_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return vis_image


def main(image_path, output_path=None):
    """
    主函数：运行检测并输出结果
    
    参数：
        image_path: 输入图像路径
        output_path: 输出图像路径（可选）
    
    返回：
        detections: 检测结果列表
    """
    # 创建检测器实例
    detector = BreastCancerDetector()
    # 执行检测
    image, detections, results = detector.detect(image_path)

    # 打印检测结果
    print(f"检测到 {len(detections)} 个病灶区域")
    for i, det in enumerate(detections):
        print(f"病灶 {i+1}: 类别={det['class']}, 置信度={det['confidence']:.4f}, "
              f"YOLO置信度={det['yolo_confidence']:.4f}, BBOX={det['bbox']}")

    # 如果检测到病灶，进行可视化
    if detections:
        vis_image = detector.visualize_results(image, detections)
        # 转换为BGR格式（OpenCV保存需要BGR）
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)

        # 如果指定了输出路径，保存图像
        if output_path:
            cv2.imwrite(output_path, vis_image)
            print(f"结果已保存到: {output_path}")
        else:
            # 否则显示图像
            cv2.imshow('Breast Cancer Detection', vis_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return detections


# 命令行入口
if __name__ == '__main__':
    import argparse

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='乳腺癌病灶检测')
    parser.add_argument('--image', required=True, help='输入图像路径')
    parser.add_argument('--output', help='输出图像路径')
    parser.add_argument('--conf', type=float, default=0.5, help='YOLO置信度阈值')

    # 解析参数
    args = parser.parse_args()
    # 运行主函数
    main(args.image, args.output)