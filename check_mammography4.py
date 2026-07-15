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

print(f'CSV中patient_id数量: {len(patient_id_labels)}')

all_dcm_files = []
for root, dirs, files in os.walk(patients_dir):
    for filename in files:
        if filename.lower().endswith('.dcm'):
            all_dcm_files.append(os.path.relpath(os.path.join(root, filename), patients_dir))

print(f'总DICOM文件数: {len(all_dcm_files)}')

matched_count = 0
unmatched_files = []

for dcm_rel_path in all_dcm_files:
    folder_parts = dcm_rel_path.split(os.sep)
    matched_patient_id = None
    for pid in patient_id_labels:
        if any(pid in part for part in folder_parts):
            matched_patient_id = pid
            break
    if matched_patient_id:
        matched_count += 1
    else:
        unmatched_files.append(dcm_rel_path)

print(f'匹配到的DICOM文件数: {matched_count}')
print(f'未匹配的DICOM文件数: {len(unmatched_files)}')

print('\n未匹配的DICOM文件路径示例:')
for f in unmatched_files[:10]:
    print(f'  {f}')

print('\npatient_id示例:')
for i, pid in enumerate(list(patient_id_labels.keys())[:10]):
    print(f'  {pid}')
