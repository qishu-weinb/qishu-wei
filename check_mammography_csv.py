import os
import csv

data_root = r'd:\软件下载\TRAE\rxaa\data\multimodal\mammography'

csv_files = [
    'calc_case_description_train_set.csv',
    'calc_case_description_test_set.csv',
    'mass_case_description_train_set.csv',
    'mass_case_description_test_set.csv',
]

total_rows = 0
label_counts = {'MALIGNANT': 0, 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0}
patient_ids = set()

for csv_name in csv_files:
    csv_path = os.path.join(data_root, csv_name)
    if not os.path.exists(csv_path):
        print(f'{csv_name}: 不存在')
        continue
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        print(f'{csv_name}: {len(rows)} 行')
        total_rows += len(rows)
        
        for row in rows:
            pathology = row.get('pathology', '').strip()
            if pathology in label_counts:
                label_counts[pathology] += 1
            pid = row.get('patient_id', '').strip()
            if pid:
                patient_ids.add(pid)

print(f'\n总CSV行数: {total_rows}')
print(f'不同patient_id数: {len(patient_ids)}')
print(f'标签分布:')
for label, count in label_counts.items():
    print(f'  {label}: {count}')
