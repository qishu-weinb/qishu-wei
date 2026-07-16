import os

data_root = r'd:\软件下载\TRAE\rxaa\data\multimodal\mammography'

image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.dcm']

extension_counts = {}
total_images = 0

for root, dirs, files in os.walk(data_root):
    for filename in files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in image_extensions:
            extension_counts[ext] = extension_counts.get(ext, 0) + 1
            total_images += 1

print(f'乳腺钼靶数据目录: {data_root}')
print(f'总图像文件数: {total_images}')
print('按扩展名分布:')
for ext, count in sorted(extension_counts.items(), key=lambda x: -x[1]):
    print(f'  {ext}: {count}')

print('\n=== 检查benign和malignant目录 ===')
benign_dir = os.path.join(data_root, 'benign')
malignant_dir = os.path.join(data_root, 'malignant')

for dir_name, dir_path in [('benign', benign_dir), ('malignant', malignant_dir)]:
    if os.path.isdir(dir_path):
        image_count = 0
        for f in os.listdir(dir_path):
            ext = os.path.splitext(f)[1].lower()
            if ext in image_extensions:
                image_count += 1
        print(f'{dir_name}: {image_count} 张图像')
    else:
        print(f'{dir_name}: 目录不存在')
