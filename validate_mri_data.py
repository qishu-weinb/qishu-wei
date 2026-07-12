import os
import sys
import csv
from pathlib import Path
from collections import defaultdict

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    print("[警告] pydicom未安装，将跳过DICOM文件内容验证")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class MRIPairedDatasetValidator:
    def __init__(self, data_root, csv_path):
        self.data_root = Path(data_root)
        self.csv_path = Path(csv_path)
        self.patient_series = defaultdict(lambda: {'t2': [], 'dce': [], 't1_pre': [], 't1_post': [], 'other': []})
        self.stats = {
            'total_patients': 0,
            'patients_with_t2_and_dce': 0,
            'patients_with_t1_and_dce': 0,
            'patients_only_dce': 0,
            'patients_only_t2': 0,
            'patients_missing_both': 0,
            't2_slice_counts': [],
            'dce_slice_counts': [],
            't1_pre_slice_counts': [],
            't1_post_slice_counts': [],
            'valid_patients': [],
            'invalid_patients': [],
        }

    def _classify_series(self, series_desc):
        desc = series_desc.lower()
        upper_desc = series_desc.upper()
        
        if 'T2' in upper_desc or 'T2TSE' in upper_desc:
            return 't2'
        elif 'dyn' in desc or 'DCE' in upper_desc or 'enhanced' in desc:
            return 'dce'
        elif 'precon' in desc or 'pregad' in desc or 'pre gad' in desc:
            return 't1_pre'
        elif 'post gad' in desc or 'post_gad' in desc or 'spgr post' in desc or 'rodeo post' in desc:
            return 't1_post'
        elif '3d grass' in desc or 'spgr' in desc:
            if 'pre' in desc:
                return 't1_pre'
            elif 'post' in desc:
                return 't1_post'
            else:
                return 't1_post'
        else:
            return 'other'

    def load_idc_csv(self):
        if not self.csv_path.exists():
            print(f"[错误] CSV文件不存在: {self.csv_path}")
            return False

        print(f"[信息] 正在读取IDC CSV文件: {self.csv_path}")
        with open(self.csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                patient_id = row['PatientID'].strip()
                series_desc = row['SeriesDescription'].strip()
                local_path = row['S5cmdManifestPath'].strip()

                series_type = self._classify_series(series_desc)
                self.patient_series[patient_id][series_type].append({
                    'desc': series_desc,
                    'path': local_path
                })

        total_series = sum(sum(len(v) for v in types.values()) for types in self.patient_series.values())
        print(f"[信息] 共读取 {len(self.patient_series)} 个患者, {total_series} 个序列")
        return True

    def validate_samples(self):
        print("\n[信息] 开始验证样本...")
        
        for patient_id, series in self.patient_series.items():
            t2_list = series['t2']
            dce_list = series['dce']
            t1_pre_list = series['t1_pre']
            t1_post_list = series['t1_post']

            print(f"\n--- 患者 {patient_id} ---")
            print(f"T2序列: {len(t2_list)}, DCE序列: {len(dce_list)}, T1预增强: {len(t1_pre_list)}, T1后增强: {len(t1_post_list)}")

            has_t2 = False
            has_dce = False
            has_t1 = False
            used_t2_path = None
            used_dce_path = None

            if t2_list:
                t2_dir = Path(t2_list[0]['path'])
                print(f"T2路径: {t2_dir}")
                if t2_dir.exists():
                    has_t2 = True
                    used_t2_path = t2_dir
                    t2_slices = sorted(list(t2_dir.glob('*.dcm')))
                    self.stats['t2_slice_counts'].append(len(t2_slices))
                    print(f"T2切片数量: {len(t2_slices)}")
                    
                    if PYDICOM_AVAILABLE and t2_slices:
                        self._validate_dicom(t2_slices[0], "T2")
                else:
                    print(f"[错误] T2目录不存在")

            if dce_list:
                dce_dir = Path(dce_list[0]['path'])
                print(f"DCE路径: {dce_dir}")
                if dce_dir.exists():
                    has_dce = True
                    used_dce_path = dce_dir
                    dce_slices = sorted(list(dce_dir.glob('*.dcm')))
                    self.stats['dce_slice_counts'].append(len(dce_slices))
                    print(f"DCE切片数量: {len(dce_slices)}")
                    
                    if PYDICOM_AVAILABLE and dce_slices:
                        self._validate_dicom(dce_slices[0], "DCE")
                else:
                    print(f"[错误] DCE目录不存在")
            elif t1_post_list:
                t1_post_dir = Path(t1_post_list[0]['path'])
                print(f"T1后增强路径: {t1_post_dir}")
                if t1_post_dir.exists():
                    has_dce = True
                    used_dce_path = t1_post_dir
                    dce_slices = sorted(list(t1_post_dir.glob('*.dcm')))
                    self.stats['dce_slice_counts'].append(len(dce_slices))
                    print(f"T1后增强切片数量: {len(dce_slices)}")
                    
                    if PYDICOM_AVAILABLE and dce_slices:
                        self._validate_dicom(dce_slices[0], "T1_POST")
                else:
                    print(f"[错误] T1后增强目录不存在")

            if not has_t2 and t1_pre_list:
                t1_pre_dir = Path(t1_pre_list[0]['path'])
                print(f"T1预增强路径: {t1_pre_dir}")
                if t1_pre_dir.exists():
                    has_t1 = True
                    used_t2_path = t1_pre_dir
                    t1_slices = sorted(list(t1_pre_dir.glob('*.dcm')))
                    self.stats['t1_pre_slice_counts'].append(len(t1_slices))
                    print(f"T1预增强切片数量: {len(t1_slices)}")
                    
                    if PYDICOM_AVAILABLE and t1_slices:
                        self._validate_dicom(t1_slices[0], "T1_PRE")

            if has_t2 and has_dce:
                self.stats['patients_with_t2_and_dce'] += 1
                self.stats['valid_patients'].append(patient_id)
            elif has_t1 and has_dce:
                self.stats['patients_with_t1_and_dce'] += 1
                self.stats['valid_patients'].append(patient_id)
            elif has_dce and not has_t2 and not has_t1:
                self.stats['patients_only_dce'] += 1
                self.stats['valid_patients'].append(patient_id)
            elif has_t2 and not has_dce:
                self.stats['patients_only_t2'] += 1
                self.stats['invalid_patients'].append(patient_id)
            else:
                self.stats['patients_missing_both'] += 1
                self.stats['invalid_patients'].append(patient_id)

            self.stats['total_patients'] += 1

    def _validate_dicom(self, dicom_path, modality):
        try:
            ds = pydicom.dcmread(str(dicom_path))
            print(f"{modality} DICOM信息:")
            print(f"  - 患者ID: {getattr(ds, 'PatientID', 'N/A')}")
            print(f"  - 序列描述: {getattr(ds, 'SeriesDescription', 'N/A')}")
            print(f"  - 图像尺寸: {ds.Rows} x {ds.Columns}")
            print(f"  - 位深度: {ds.BitsAllocated} bits")
            print(f"  - 切片位置: {getattr(ds, 'SliceLocation', 'N/A')}")
            print(f"  - 实例编号: {getattr(ds, 'InstanceNumber', 'N/A')}")
        except Exception as e:
            print(f"[警告] 读取{modality} DICOM失败: {e}")

    def print_summary(self):
        print("\n" + "="*60)
        print("IDC MRI数据集验证报告")
        print("="*60)

        print(f"\n[患者统计]")
        print(f"  总患者数: {self.stats['total_patients']}")
        print(f"  T2+DCE: {self.stats['patients_with_t2_and_dce']}")
        print(f"  T1+DCE: {self.stats['patients_with_t1_and_dce']}")
        print(f"  仅DCE: {self.stats['patients_only_dce']}")
        print(f"  仅T2: {self.stats['patients_only_t2']}")
        print(f"  两者都缺失: {self.stats['patients_missing_both']}")
        print(f"  有效患者: {len(self.stats['valid_patients'])}")

        if self.stats['t2_slice_counts']:
            print(f"\n[T2切片统计]")
            print(f"  最小切片数: {min(self.stats['t2_slice_counts'])}")
            print(f"  最大切片数: {max(self.stats['t2_slice_counts'])}")
            print(f"  平均切片数: {sum(self.stats['t2_slice_counts']) / len(self.stats['t2_slice_counts']):.2f}")

        if self.stats['dce_slice_counts']:
            print(f"\n[DCE/T1后增强切片统计]")
            print(f"  最小切片数: {min(self.stats['dce_slice_counts'])}")
            print(f"  最大切片数: {max(self.stats['dce_slice_counts'])}")
            print(f"  平均切片数: {sum(self.stats['dce_slice_counts']) / len(self.stats['dce_slice_counts']):.2f}")

        if self.stats['t1_pre_slice_counts']:
            print(f"\n[T1预增强切片统计]")
            print(f"  最小切片数: {min(self.stats['t1_pre_slice_counts'])}")
            print(f"  最大切片数: {max(self.stats['t1_pre_slice_counts'])}")
            print(f"  平均切片数: {sum(self.stats['t1_pre_slice_counts']) / len(self.stats['t1_pre_slice_counts']):.2f}")

        if self.stats['invalid_patients']:
            print(f"\n[无效患者列表]")
            for pid in self.stats['invalid_patients']:
                print(f"  - {pid}")

        print("\n" + "="*60)

        if len(self.stats['valid_patients']) == self.stats['total_patients']:
            print("[成功] 所有患者数据验证通过！")
            return True
        else:
            print("[警告] 部分患者数据存在问题，请检查上述无效患者")
            return False


def main():
    mri_data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'multimodal', 'mri')
    csv_path = os.path.join(mri_data_root, 'labels.csv')
    
    print(f"[信息] MRI数据根目录: {mri_data_root}")
    print(f"[信息] CSV文件路径: {csv_path}")
    
    if not os.path.exists(mri_data_root):
        print(f"[错误] 数据目录不存在: {mri_data_root}")
        return
    
    if not os.path.exists(csv_path):
        print(f"[错误] CSV文件不存在: {csv_path}")
        return

    validator = MRIPairedDatasetValidator(mri_data_root, csv_path)
    
    if not validator.load_idc_csv():
        print("[错误] CSV加载失败")
        return

    validator.validate_samples()
    validator.print_summary()


if __name__ == '__main__':
    main()