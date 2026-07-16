"""
乳腺钼靶X线摄影 AI 诊断模块

两阶段流水线：
  第一阶段 - YOLOv8-seg: 高精度病灶检测与分割
  第二阶段 - BCHI-CovNet: 原始分辨率ROI纹理分类

支持 CBIS-DDSM 数据集格式。
"""
