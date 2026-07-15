"""
结果可视化模块

支持：
- 病灶检测结果可视化
- 诊断报告生成
- 热力图绘制
"""

import cv2
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt


def visualize_diagnosis(image: np.ndarray, results: Dict) -> np.ndarray:
    """
    可视化诊断结果
    
    参数：
        image: 原始图像 (RGB)
        results: 诊断结果字典
    
    返回：
        vis_image: 可视化图像 (RGB)
    """
    vis_image = image.copy()
    h, w = image.shape[:2]
    
    if not results['lesions']:
        return vis_image
    
    # 颜色映射
    color_map = {
        'normal': (0, 255, 0),      # 绿色
        'benign': (0, 255, 255),    # 黄色
        'in_situ': (255, 165, 0),   # 橙色
        'invasive': (255, 0, 0),    # 红色
        'other': (128, 0, 128)      # 紫色
    }
    
    for lesion in results['lesions']:
        # 获取信息
        bbox = lesion['location']['bbox']
        x1, y1, x2, y2 = bbox
        class_name = lesion['classification']['class_name']
        confidence = lesion['classification']['confidence']
        
        # 获取颜色
        color = color_map.get(class_name, (128, 128, 128))
        
        # 绘制边界框
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 3)
        
        # 绘制标签
        label = f"{class_name}: {confidence:.2%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        # 标签背景
        cv2.rectangle(
            vis_image,
            (x1, y1 - text_h - 10),
            (x1 + text_w + 10, y1),
            color,
            -1
        )
        
        # 标签文字
        cv2.putText(
            vis_image,
            label,
            (x1 + 5, y1 - 5),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )
    
    # 添加整体诊断信息
    summary_text = f"Risk: {results['summary']['risk_level']} | Lesions: {results['lesion_count']}"
    cv2.putText(
        vis_image,
        summary_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )
    
    return vis_image


def create_diagnosis_report(results: Dict) -> str:
    """
    生成文本诊断报告
    
    参数：
        results: 诊断结果字典
    
    返回：
        report: 诊断报告文本
    """
    report = []
    report.append("=" * 60)
    report.append("乳腺 X 线摄影 AI 诊断报告")
    report.append("=" * 60)
    report.append(f"时间: {results['timestamp']}")
    report.append(f"图像大小: {results['image_size']}")
    report.append("")
    
    # 整体评估
    report.append("【整体评估】")
    summary = results['summary']
    report.append(f"状态: {summary['status']}")
    report.append(f"风险等级: {summary['risk_level'].upper()}")
    report.append(f"平均置信度: {summary['overall_confidence']:.2%}")
    report.append("")
    
    # 病灶详情
    if results['lesions']:
        report.append(f"【检测到 {results['lesion_count']} 个病灶】")
        report.append("")
        
        for lesion in results['lesions']:
            report.append(f"病灶 {lesion['id']+1}:")
            report.append(f"  位置: [{lesion['location']['x1']}, {lesion['location']['y1']}, "
                         f"{lesion['location']['x2']}, {lesion['location']['y2']}]")
            
            # 分割信息
            seg = lesion['segmentation']
            report.append(f"  分割置信度: {seg['confidence']:.2%}")
            report.append(f"  病灶面积: {seg['area_pixels']} px²")
            report.append(f"  Mask 覆盖: {seg['mask_coverage_ratio']:.2%}")
            
            # 分类信息
            clf = lesion['classification']
            report.append(f"  分类: {clf['class_name']} ({clf['description']})")
            report.append(f"  置信度: {clf['confidence']:.2%}")
            report.append(f"  概率分布:")
            for class_name, prob in clf['probabilities'].items():
                report.append(f"    - {class_name}: {prob:.2%}")
            
            report.append(f"  建议: {lesion['recommendation']}")
            report.append("")
    else:
        report.append("【未检测到病灶】")
        report.append("乳腺组织正常，建议定期随访。")
        report.append("")
    
    # 临床建议
    report.append("【临床建议】")
    report.append(summary['recommendation'])
    report.append("")
    
    report.append("=" * 60)
    report.append("免责声明：本报告仅供参考，不能替代专业医学诊断。")
    report.append("=" * 60)
    
    return "\n".join(report)


def plot_training_history(history_dict: Dict, save_path: str = None):
    """
    绘制训练历史曲线
    
    参数：
        history_dict: 训练历史字典
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    axes[0].plot(history_dict['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history_dict['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 精度曲线
    axes[1].plot(history_dict['train_acc'], label='Train Acc', marker='o')
    axes[1].plot(history_dict['val_acc'], label='Val Acc', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"训练曲线已保存到 {save_path}")
    else:
        plt.show()


if __name__ == '__main__':
    # 测试
    sample_results = {
        'timestamp': '2024-01-01T12:00:00',
        'image_size': (1024, 1024),
        'lesion_count': 2,
        'summary': {
            'status': 'abnormal',
            'risk_level': 'high',
            'overall_confidence': 0.92,
            'recommendation': '建议立即活检或手术'
        },
        'lesions': [
            {
                'id': 0,
                'location': {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200},
                'segmentation': {'confidence': 0.85, 'area_pixels': 5000, 'mask_coverage_ratio': 0.75},
                'classification': {
                    'class_name': 'invasive',
                    'confidence': 0.95,
                    'description': '浸润性癌'
                },
                'recommendation': '建议立即活检或手术'
            }
        ]
    }
    
    report = create_diagnosis_report(sample_results)
    print(report)
