import os

data_root = r'd:\软件下载\TRAE\rxaa\data\multimodal\mammography'

all_dcm_paths = []
for root, dirs, files in os.walk(data_root):
    for filename in files:
        if filename.lower().endswith('.dcm'):
            all_dcm_paths.append(os.path.relpath(os.path.join(root, filename), data_root))

print(f'总DICOM文件数: {len(all_dcm_paths)}')

folder_counts = {}
for rel_path in all_dcm_paths:
    parts = rel_path.split(os.sep)
    if parts:
        first_folder = parts[0]
        folder_counts[first_folder] = folder_counts.get(first_folder, 0) + 1

print('\n按一级目录分布:')
for folder, count in sorted(folder_counts.items(), key=lambda x: -x[1]):
    print(f'  {folder}: {count}')

print('\n前20个DICOM文件路径:')
for path in all_dcm_paths[:20]:
    print(f'  {path}')
