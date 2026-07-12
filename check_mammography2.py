import os
import csv

data_root = r'd:\软件下载\TRAE\rxaa\data\multimodal\mammography'
patients_dir = os.path.join(data_root, 'patients')

csv_files = [
    'calc_case_description_train_set.csv',
    'calc_case_description_test_set.csv',
    'mass_case_description_train_set.csv',
    'mass_case_description_test_set.csv',
]

csv_folder_names = set()
actual_folders = {}

for csv_name in csv_files:
    csv_path = os.path.join(data_root, csv_name)
    if not os.path.exists(csv_path):
        continue
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pathology = row.get('pathology', '').strip()
            if pathology not in ['MALIGNANT', 'BENIGN', 'BENIGN_WITHOUT_CALLBACK']:
                continue
            cropped_path = row.get('cropped image file path', '').strip().strip('"')
            if not cropped_path:
                continue
            folder_name = cropped_path.split('/')[0]
            csv_folder_names.add(folder_name)

for folder in os.listdir(patients_dir):
    folder_path = os.path.join(patients_dir, folder)
    if os.path.isdir(folder_path):
        actual_folders[folder] = folder_path

print('=== 检查文件夹命名差异 ===')
print(f'CSV文件夹数: {len(csv_folder_names)}')
print(f'实际文件夹数: {len(actual_folders)}')

matching = 0
csv_only = 0
actual_only = 0

for csv_folder in csv_folder_names:
    if csv_folder in actual_folders:
        matching += 1
    else:
        csv_only += 1
        
for actual_folder in actual_folders:
    if actual_folder not in csv_folder_names:
        actual_only += 1

print(f'完全匹配: {matching}')
print(f'CSV独有: {csv_only}')
print(f'实际独有: {actual_only}')

print('\n=== 检查实际独有的文件夹是否可以通过去掉数字后缀匹配 ===')
matched_by_trimming = 0
for actual_folder in actual_folders:
    if actual_folder not in csv_folder_names:
        parts = actual_folder.split('_')
        if len(parts) >= 5 and parts[-1].isdigit():
            trimmed = '_'.join(parts[:-1])
            if trimmed in csv_folder_names:
                matched_by_trimming += 1
                if matched_by_trimming <= 5:
                    print(f'  {actual_folder} -> {trimmed} (匹配)')

print(f'通过去掉数字后缀匹配: {matched_by_trimming}')

print('\n=== 检查CSV独有的文件夹是否可以通过添加数字后缀匹配 ===')
matched_by_adding = 0
for csv_folder in csv_folder_names:
    if csv_folder not in actual_folders:
        parts = csv_folder.split('_')
        if len(parts) >= 5 and parts[-1].isdigit():
            trimmed = '_'.join(parts[:-1])
            if trimmed in actual_folders:
                matched_by_adding += 1
                if matched_by_adding <= 5:
                    print(f'  {csv_folder} -> {trimmed} (匹配)')

print(f'通过添加数字后缀匹配: {matched_by_adding}')
