import csv
from collections import defaultdict

csv_path = r'd:\软件下载\TRAE\rxaa\data\multimodal\mri\labels.csv'

patient_series = defaultdict(list)

with open(csv_path, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        patient_id = row['PatientID'].strip()
        series_desc = row['SeriesDescription'].strip()
        patient_series[patient_id].append(series_desc)

print("所有患者及其序列描述:\n")
for patient_id in sorted(patient_series.keys()):
    series = patient_series[patient_id]
    print(f"患者 {patient_id}:")
    for s in series:
        is_t2 = 'T2' in s or 'T2TSE' in s
        is_dce = 'dyn' in s.lower() or 'DCE' in s or 'enhanced' in s.lower() or 'DYN' in s
        tag = ""
        if is_t2:
            tag = "[T2]"
        elif is_dce:
            tag = "[DCE]"
        else:
            tag = "[未识别]"
        print(f"  {tag} {s}")
    print()