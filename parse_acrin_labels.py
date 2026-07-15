import os
import csv
import sys
from pathlib import Path
from collections import defaultdict


class ACRINLabelParser:
    def __init__(self, acrin_dir, mri_patients_dir):
        self.acrin_dir = Path(acrin_dir)
        self.mri_patients_dir = Path(mri_patients_dir)
        self.patient_labels = {}

    def parse_m3_table(self):
        m3_path = self.acrin_dir / "6667_M3 reviewed.csv"
        if not m3_path.exists():
            print(f"[错误] M3表格不存在: {m3_path}")
            return

        print(f"[信息] 正在读取M3表格: {m3_path}")
        with open(m3_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                cn = int(row['cn'])
                
                m3e68 = row.get('m3e68', '').strip()
                if m3e68:
                    m3e68 = int(m3e68)
                
                m3e69 = row.get('m3e69', '').strip()
                if m3e69:
                    m3e69 = int(m3e69)
                
                m3e70 = row.get('m3e70', '').strip()
                if m3e70:
                    m3e70 = int(m3e70)

                if cn not in self.patient_labels:
                    self.patient_labels[cn] = {
                        'm3e68': [],
                        'm3e69': [],
                        'm3e70': [],
                        'pae10': None,
                        'final_label': None,
                        'label_source': None,
                    }
                
                if m3e68 or m3e68 == 0:
                    self.patient_labels[cn]['m3e68'].append(m3e68)
                if m3e69 and m3e69 != 0:
                    self.patient_labels[cn]['m3e69'].append(m3e69)
                if m3e70 and m3e70 != 0:
                    self.patient_labels[cn]['m3e70'].append(m3e70)

        print(f"[信息] M3表格读取完成，共 {len(self.patient_labels)} 个患者")

    def parse_pa_table(self):
        pa_path = self.acrin_dir / "6667_PA reviewed.csv"
        if not pa_path.exists():
            print(f"[错误] PA表格不存在: {pa_path}")
            return

        print(f"[信息] 正在读取PA表格: {pa_path}")
        with open(pa_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                cn = int(row['cn'])
                pae10 = row.get('pae10', '').strip()
                
                if pae10:
                    pae10 = int(pae10)
                    
                    if cn not in self.patient_labels:
                        self.patient_labels[cn] = {
                            'm3e68': [],
                            'm3e69': [],
                            'm3e70': [],
                            'pae10': None,
                            'final_label': None,
                            'label_source': None,
                        }
                    
                    if self.patient_labels[cn]['pae10'] is None:
                        self.patient_labels[cn]['pae10'] = pae10

        print(f"[信息] PA表格读取完成")

    def _determine_label_from_pae10(self, pae10):
        if pae10 == 1:
            return 0, 'benign'
        elif pae10 == 3:
            return 1, 'malignant_in_situ'
        elif pae10 == 4:
            return 1, 'malignant_invasive'
        return None, None

    def _determine_label_from_m3e68(self, m3e68_values):
        if not m3e68_values:
            return None, None
        
        max_val = max(int(v) for v in m3e68_values)
        
        if max_val == 0:
            return 0, 'incomplete'
        elif max_val <= 2:
            return 0, 'benign'
        elif max_val == 3:
            return 0, 'probably_benign'
        elif max_val == 4:
            return 1, 'suspicious_malignant'
        elif max_val == 5:
            return 1, 'highly_suggestive_malignant'
        return None, None

    def assign_final_labels(self):
        print(f"\n[信息] 开始分配最终标签...")
        
        benign_count = 0
        malignant_count = 0
        unknown_count = 0
        pa_source_count = 0
        m3_source_count = 0

        for cn, labels in self.patient_labels.items():
            final_label = None
            label_source = None

            if labels['pae10'] is not None:
                label, label_type = self._determine_label_from_pae10(labels['pae10'])
                if label is not None:
                    final_label = label
                    label_source = 'PA'
                    pa_source_count += 1
            
            if final_label is None and labels['m3e68']:
                label, label_type = self._determine_label_from_m3e68(labels['m3e68'])
                if label is not None:
                    final_label = label
                    label_source = 'M3'
                    m3_source_count += 1

            labels['final_label'] = final_label
            labels['label_source'] = label_source

            if final_label == 0:
                benign_count += 1
            elif final_label == 1:
                malignant_count += 1
            else:
                unknown_count += 1

        print(f"[信息] 标签分配完成:")
        print(f"  良性: {benign_count}")
        print(f"  恶性: {malignant_count}")
        print(f"  未知: {unknown_count}")
        print(f"  来自PA表格: {pa_source_count}")
        print(f"  来自M3表格: {m3_source_count}")

    def get_mri_patients(self):
        mri_patients = {}
        
        if not self.mri_patients_dir.exists():
            print(f"[错误] MRI患者目录不存在: {self.mri_patients_dir}")
            return mri_patients

        for patient_dir in self.mri_patients_dir.iterdir():
            if patient_dir.is_dir():
                patient_name = patient_dir.name
                if patient_name.startswith('ACRIN-Contralateral-Breast-MR-'):
                    patient_id = int(patient_name.replace('ACRIN-Contralateral-Breast-MR-', ''))
                    mri_patients[patient_id] = patient_dir

        print(f"\n[信息] 发现 {len(mri_patients)} 个MRI患者")
        return mri_patients

    def correlate_mri_with_labels(self):
        mri_patients = self.get_mri_patients()
        
        correlated = {}
        missing_labels = []
        
        for patient_id, patient_dir in mri_patients.items():
            if patient_id in self.patient_labels:
                labels = self.patient_labels[patient_id]
                if labels['final_label'] is not None:
                    correlated[patient_id] = {
                        'dir': patient_dir,
                        'label': labels['final_label'],
                        'source': labels['label_source'],
                        'pae10': labels['pae10'],
                        'm3e68': labels['m3e68'],
                    }
                else:
                    missing_labels.append(patient_id)
            else:
                missing_labels.append(patient_id)

        benign_mri = [p for p, v in correlated.items() if v['label'] == 0]
        malignant_mri = [p for p, v in correlated.items() if v['label'] == 1]

        print(f"\n[信息] MRI数据与ACRIN标签关联结果:")
        print(f"  有标签的MRI患者: {len(correlated)}")
        print(f"    - 良性: {len(benign_mri)}")
        print(f"    - 恶性: {len(malignant_mri)}")
        print(f"  无标签的MRI患者: {len(missing_labels)}")
        
        if missing_labels:
            print(f"    无标签患者ID: {sorted(missing_labels)}")

        return correlated

    def print_summary(self):
        print("\n" + "="*70)
        print("ACRIN 6667 标签解析报告")
        print("="*70)

        total_patients = len(self.patient_labels)
        labeled_patients = sum(1 for v in self.patient_labels.values() if v['final_label'] is not None)
        benign = sum(1 for v in self.patient_labels.values() if v['final_label'] == 0)
        malignant = sum(1 for v in self.patient_labels.values() if v['final_label'] == 1)

        print(f"\n[ACRIN表格统计]")
        print(f"  总患者数: {total_patients}")
        print(f"  有标签的患者: {labeled_patients}")
        print(f"  良性: {benign}")
        print(f"  恶性: {malignant}")

        pa_count = sum(1 for v in self.patient_labels.values() if v['pae10'] is not None)
        m3_count = sum(1 for v in self.patient_labels.values() if v['m3e68'])
        print(f"\n[标签来源统计]")
        print(f"  PA表格(病理): {pa_count}")
        print(f"  M3表格(MRI诊断): {m3_count}")

        print("\n[标签定义]")
        print("  PA表格 pae10:")
        print("    1 = 良性")
        print("    3 = 原位癌(恶性)")
        print("    4 = 浸润性癌(恶性)")
        print("  M3表格 M3e68:")
        print("    1 = 阴性")
        print("    2 = 良性")
        print("    3 = 可能良性")
        print("    4 = 可疑恶性")
        print("    5 = 高度提示恶性")

        print("\n" + "="*70)


def main():
    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    acrin_dir = base_dir / "data" / "multimodal" / "mri" / "ACRIN 6667 Contralateral Breast MRI Clinical Data Anonymized"
    mri_patients_dir = base_dir / "data" / "multimodal" / "mri" / "patients"

    print(f"[信息] ACRIN目录: {acrin_dir}")
    print(f"[信息] MRI患者目录: {mri_patients_dir}")

    parser = ACRINLabelParser(acrin_dir, mri_patients_dir)
    
    parser.parse_m3_table()
    parser.parse_pa_table()
    parser.assign_final_labels()
    parser.correlate_mri_with_labels()
    parser.print_summary()


if __name__ == '__main__':
    main()