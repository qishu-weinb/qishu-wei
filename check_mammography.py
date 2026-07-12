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

patient_labels = {}
all_folder_names = set()

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
            all_folder_names.add(folder_name)

print(f'CSV中文件夹名称总数: {len(all_folder_names)}')

actual_folders = set()
for folder in os.listdir(patients_dir):
    if os.path.isdir(os.path.join(patients_dir, folder)):
        actual_folders.add(folder)

print(f'实际文件夹名称总数: {len(actual_folders)}')
common = all_folder_names.intersection(actual_folders)
print(f'匹配的文件夹数: {len(common)}')

print('\nCSV中独有的文件夹（实际不存在）:')
for f in sorted(all_folder_names - actual_folders)[:10]:
    print(f'  {f}')

print('\n实际独有的文件夹（CSV中没有）:')
for f in sorted(actual_folders - all_folder_names)[:10]:
    print(f'  {f}')
