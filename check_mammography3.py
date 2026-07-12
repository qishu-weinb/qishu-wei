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

patient_id_labels = {}

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
            patient_id = row.get('patient_id', '').strip()
            if not patient_id:
                continue
            label = 1 if pathology == 'MALIGNANT' else 0
            patient_id_labels[patient_id] = label

print(f'CSV中不同patient_id数量: {len(patient_id_labels)}')

folder_patient_ids = set()
for folder in os.listdir(patients_dir):
    folder_path = os.path.join(patients_dir, folder)
    if os.path.isdir(folder_path):
        for pid in patient_id_labels:
            if pid in folder:
                folder_patient_ids.add(pid)
                break

print(f'实际文件夹中匹配到的patient_id数量: {len(folder_patient_ids)}')

print('\nCSV中patient_id示例:')
for i, pid in enumerate(list(patient_id_labels.keys())[:10]):
    print(f'  {pid}: {patient_id_labels[pid]}')

print('\n实际文件夹名称示例:')
for i, folder in enumerate(sorted(os.listdir(patients_dir))[:10]):
    folder_path = os.path.join(patients_dir, folder)
    if os.path.isdir(folder_path):
        print(f'  {folder}')

print('\n=== 检查哪些patient_id无法匹配 ===')
unmatched_pids = []
for pid in patient_id_labels:
    found = False
    for folder in os.listdir(patients_dir):
        folder_path = os.path.join(patients_dir, folder)
        if os.path.isdir(folder_path) and pid in folder:
            found = True
            break
    if not found:
        unmatched_pids.append(pid)

print(f'无法匹配的patient_id数量: {len(unmatched_pids)}')
print(f'前20个无法匹配的patient_id:')
for pid in unmatched_pids[:20]:
    print(f'  {pid}')
