import os

data_root = r'd:\软件下载\TRAE\rxaa\data'

dicom_counts = {}

for root, dirs, files in os.walk(data_root):
    dcm_count = sum(1 for f in files if f.lower().endswith('.dcm'))
    if dcm_count > 0:
        rel_path = os.path.relpath(root, data_root)
        dicom_counts[rel_path] = dcm_count

print('DICOM文件分布:')
for rel_path, count in sorted(dicom_counts.items(), key=lambda x: -x[1])[:20]:
    print(f'  {rel_path}: {count}')
