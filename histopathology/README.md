\# His-rec - 病理图像智能分类系统



基于 \*\*ResNet50 + DenseNet121\*\* 双模型集成投票的病理组织图像二分类系统，用于辅助判断组织样本为\*\*良性\*\*或\*\*恶性\*\*。



\---



\## 📊 模型性能



| 指标 | 数值 |

|------|------|

| 测试集准确率 | \*\*91.35%\*\* |

| 训练集规模 | 222,020 张 |

| 测试集规模 | 55,504 张 |

| 模型架构 | ResNet50 + DenseNet121 硬投票集成 |



\---



\## 🚀 快速开始



环境要求

\- Python 3.8+

\- CUDA 11.0+ (可选，GPU加速)



单张图像推理

python inference.py --image /path/to/your/image.png

输出示例：

使用设备: cuda

图像: /path/to/image.png

预测结果: 恶性 (类别: 1)



批量推理

python inference.py --dir /path/to/image/folder/

输出示例：

使用设备: cuda

image\_001.png: 良性

image\_002.png: 恶性

image\_003.png: 良性



指定权重文件

python inference.py --image test.png --weight weights/ensemble\_best\_acc0.9135.pth



数据集结构如下：

dataset/

├── patient\_001/

│   ├── 0/          # 良性样本

│   │   ├── img1.png

│   │   └── img2.png

│   └── 1/          # 恶性样本

│       ├── img3.png

│       └── img4.png

├── patient\_002/

│   ├── 0/

│   └── 1/

└── ...



推理参数说明

参数	说明	默认值

\--image	单张图像路径	无

\--dir	图像目录路径（批量推理）	无

\--weight	模型权重文件路径	weights/ensemble\_best\_acc0.9135.pth

注意：--image 和 --dir 必须指定其中之一。



模型原理

本系统采用硬投票集成策略：

ResNet50 和 DenseNet121 分别对输入图像进行预测

每个模型输出二分类结果（0 或 1）

最终结果 = (resnet\_pred + densenet\_pred) // 2

即两个模型都预测为恶性（1）时，才判定为恶性，有效降低假阳性率。



项目结构

His-rec/

├── README.md                    # 项目说明

├── requirements.txt             # Python依赖

├── inference.py                 # 推理主程序

├── model.py                     # 模型定义

├── config.py                    # 配置文件

├── weights/

│   └── ensemble\_best\_acc0.9135.pth  # 预训练权重

└── dataset/                     # 数据集（用户自备）

&#x20;   └── ...



注意事项

图像格式：支持 .png、.jpg、.jpeg 格式

图像尺寸：输入图像会自动缩放到 224×224

归一化：使用 ImageNet 均值和标准差进行归一化

GPU支持：自动检测 CUDA，若无 GPU 则使用 CPU

