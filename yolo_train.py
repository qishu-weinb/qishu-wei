import os
from ultralytics import YOLO
from config import DATA_DIR, MODEL_DIR


def create_data_yaml():
    yaml_content = f"""
path: {DATA_DIR}/yolo_dataset
train: images/train
val: images/val
test: images/test

names:
  0: lesion
"""
    yaml_path = os.path.join(DATA_DIR, 'yolo_dataset', 'data.yaml')
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)

    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    return yaml_path


def train_yolo(data_yaml_path, epochs=100, batch_size=16, imgsz=640):
    model = YOLO('yolov8n-seg.pt')

    os.makedirs(MODEL_DIR, exist_ok=True)

    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device='cuda' if __import__('torch').cuda.is_available() else 'cpu',
        save=True,
        save_period=10,
        plots=True,
        augment=True
    )

    best_model_path = os.path.join(MODEL_DIR, 'yolov8n-seg-best.pt')
    os.rename(results.save_dir + '/weights/best.pt', best_model_path)

    print(f"YOLOv8n训练完成! 最佳模型已保存到 {best_model_path}")

    return best_model_path


def main():
    print("创建YOLO数据集配置文件...")
    data_yaml_path = create_data_yaml()
    print(f"配置文件已创建: {data_yaml_path}")

    print("开始训练YOLOv8n-seg...")
    train_yolo(data_yaml_path)


if __name__ == '__main__':
    main()