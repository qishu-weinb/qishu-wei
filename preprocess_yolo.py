import os
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from config import DATA_DIR, CLASSES, YOLO_INPUT_SIZE, DEVICE


def extract_lesions_from_dataset(input_dir, output_dir, yolo_model_path=None, conf_threshold=0.5, padding=10):
    if yolo_model_path is None:
        yolo_model_path = 'yolov8n-seg.pt'

    model = YOLO(yolo_model_path)
    model = model.to(DEVICE)

    for cls_name in CLASSES:
        cls_input_dir = os.path.join(input_dir, cls_name)
        cls_output_dir = os.path.join(output_dir, cls_name)

        if not os.path.exists(cls_input_dir):
            continue

        os.makedirs(cls_output_dir, exist_ok=True)

        image_files = [f for f in os.listdir(cls_input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]

        for img_name in tqdm(image_files, desc=f"处理 {cls_name}"):
            img_path = os.path.join(cls_input_dir, img_name)
            image = cv2.imread(img_path)

            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = model(image, imgsz=YOLO_INPUT_SIZE, conf=conf_threshold, device=DEVICE)
            masks = results[0].masks
            boxes = results[0].boxes

            if masks is None or len(masks) == 0:
                continue

            original_shape = image.shape[:2]

            for i in range(len(masks)):
                mask = masks.data[i].cpu().numpy().astype(np.uint8)
                mask = cv2.resize(mask, (original_shape[1], original_shape[0]))

                box = boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = box

                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(original_shape[1], x2 + padding)
                y2 = min(original_shape[0], y2 + padding)

                lesion_region = image[y1:y2, x1:x2]
                lesion_mask = mask[y1:y2, x1:x2]

                masked_lesion = cv2.bitwise_and(lesion_region, lesion_region, mask=lesion_mask)

                base_name = os.path.splitext(img_name)[0]
                output_name = f"{base_name}_lesion_{i}.png"
                output_path = os.path.join(cls_output_dir, output_name)

                masked_lesion_bgr = cv2.cvtColor(masked_lesion, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, masked_lesion_bgr)

    print(f"预处理完成! 病灶图像已保存到 {output_dir}")


def main():
    input_train_dir = os.path.join(DATA_DIR, 'train_raw')
    input_val_dir = os.path.join(DATA_DIR, 'val_raw')
    output_train_dir = TRAIN_DIR
    output_val_dir = VAL_DIR

    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(output_val_dir, exist_ok=True)

    if os.path.exists(input_train_dir):
        print("处理训练数据集...")
        extract_lesions_from_dataset(input_train_dir, output_train_dir)

    if os.path.exists(input_val_dir):
        print("处理验证数据集...")
        extract_lesions_from_dataset(input_val_dir, output_val_dir)


if __name__ == '__main__':
    from config import TRAIN_DIR, VAL_DIR
    main()