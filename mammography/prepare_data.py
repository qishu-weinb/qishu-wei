"""
数据准备脚本 - 乳腺钼靶数据集下载与预处理
========================================

支持两种数据集:
  1. CBIS-DDSM (Kaggle) - 自动下载
  2. VinDr-Mammo (PhysioNet) - 需手动下载后运行预处理

用法:
  # 下载并准备 CBIS-DDSM
  python mammography/prepare_data.py --dataset cbis

  # 准备 VinDr-Mammo (需先手动下载到 data/mammography/vindr_mammo/)
  python mammography/prepare_data.py --dataset vindr

  # 一站式准备所有数据
  python mammography/prepare_data.py --dataset all
"""

import argparse
import logging
import os
import subprocess
import sys
import zipfile
from pathlib import Path

# 将项目根目录加入 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("prepare_data")


# ====================================================================
# CBIS-DDSM
# ====================================================================

def download_cbis_ddsm(data_dir: Path) -> bool:
    """从 Kaggle 下载 CBIS-DDSM 数据集

    Args:
        data_dir: 下载目标目录

    Returns:
        是否成功
    """
    kaggle_dataset = "awsaf49/cbis-ddsm-breast-cancer-image-dataset"

    # 检查是否已存在
    if (data_dir / "manifest-1667316001.8478514-OAiURc").exists() or \
       (data_dir / "cbis-ddsm-breast-cancer-image-dataset").exists():
        logger.info("CBIS-DDSM 数据已存在，跳过下载")
        return True

    logger.info("正在从 Kaggle 下载 CBIS-DDSM (约 5.6GB)...")
    data_dir.mkdir(parents=True, exist_ok=True)

    # 使用无代理环境
    env = os.environ.copy()
    env["HTTP_PROXY"] = ""
    env["HTTPS_PROXY"] = ""
    env["http_proxy"] = ""
    env["https_proxy"] = ""

    kaggle_bin = os.path.join(os.path.dirname(sys.executable), "Scripts", "kaggle.exe")
    if not os.path.exists(kaggle_bin):
        # 尝试查找 kaggle
        kaggle_bin = shutil.which("kaggle") or kaggle_bin

    try:
        result = subprocess.run(
            [kaggle_bin, "datasets", "download", kaggle_dataset,
             "-p", str(data_dir), "--unzip"],
            env=env, capture_output=True, text=True, timeout=3600,
        )
        if result.returncode == 0:
            logger.info("CBIS-DDSM 下载完成!")
            return True
        else:
            logger.error("下载失败: %s", result.stderr)
            return False
    except subprocess.TimeoutExpired:
        logger.error("下载超时 (1小时)")
        return False
    except FileNotFoundError:
        logger.error(
            "找不到 kaggle CLI。请先安装: "
            "pip install kaggle (或使用 prepare_data.py 同级目录下的环境)"
        )
        return False


def verify_cbis_ddsm(data_dir: Path) -> dict:
    """验证 CBIS-DDSM 数据完整性"""
    import pandas as pd

    from mammography.config import CSV_FILES

    result = {"csv_files": {}, "patient_dirs": 0, "status": "unknown"}

    # 检查 CSV
    for key, fname in CSV_FILES.items():
        csv_path = data_dir / fname
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                result["csv_files"][key] = len(df)
            except Exception:
                result["csv_files"][key] = 0
        else:
            result["csv_files"][key] = 0

    # 检查患者 DICOM 目录
    patients_dir = data_dir / "patients"
    if patients_dir.is_dir():
        result["patient_dirs"] = len(list(patients_dir.iterdir()))

    total_csv = sum(result["csv_files"].values())
    if total_csv > 0 and result["patient_dirs"] > 0:
        result["status"] = "ok"
    elif total_csv > 0:
        result["status"] = "csv_only"
    else:
        result["status"] = "missing"

    return result


# ====================================================================
# VinDr-Mammo
# ====================================================================

def prepare_vindr(data_dir: Path) -> bool:
    """准备 VinDr-Mammo 数据集 (需手动下载)

    数据集需从 PhysioNet 下载并解压到此目录:
    https://physionet.org/content/vindr-mammo/1.0.0/

    Expected structure:
      vindr_mammo/
      ├── images/
      │   ├── <study_id>/
      │   │   └── *.dcm
      ├── finding_annotations.csv
      ├── breast-level_annotations.csv
    """
    vindr_root = data_dir / "vindr_mammo"

    if not vindr_root.exists():
        logger.error("VinDr-Mammo 目录不存在: %s", vindr_root)
        logger.info(
            "请从 https://physionet.org/content/vindr-mammo/ 手动下载，"
            "解压到: %s", vindr_root
        )
        return False

    # 检查关键文件
    required = [
        vindr_root / "finding_annotations.csv",
        vindr_root / "breast-level_annotations.csv",
    ]
    missing = [f for f in required if not f.exists()]
    if missing:
        logger.error("缺少 CSV 标注文件: %s", missing)
        return False

    images_dir = vindr_root / "images"
    if not images_dir.is_dir() or not any(images_dir.iterdir()):
        logger.error("images 目录为空: %s", images_dir)
        return False

    logger.info("VinDr-Mammo 数据验证通过!")

    # 转换为训练格式
    from mammography.utils.vindr_loader import prepare_vindr_for_training

    stats = prepare_vindr_for_training(
        vindr_root=str(vindr_root),
        output_root=str(data_dir),
    )
    logger.info("VinDr-Mammo 预处理完成: %s", stats)
    return True


# ====================================================================
# 主入口
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="乳腺钼靶数据集准备工具"
    )
    parser.add_argument(
        "--dataset", type=str, default="cbis",
        choices=["cbis", "vindr", "all"],
        help="要准备的数据集 (default: cbis)",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="数据目录 (default: <project>/data/mammography)",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="仅验证数据完整性，不下载",
    )
    args = parser.parse_args()

    # 确定数据目录
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        project_dir = Path(__file__).resolve().parent.parent
        data_dir = project_dir / "data" / "mammography"

    data_dir.mkdir(parents=True, exist_ok=True)

    # ---- CBIS-DDSM ----
    if args.dataset in ("cbis", "all"):
        logger.info("=" * 50)
        logger.info("CBIS-DDSM")
        logger.info("=" * 50)

        if args.verify_only:
            stats = verify_cbis_ddsm(data_dir)
            logger.info("验证结果: %s", stats)
        else:
            success = download_cbis_ddsm(data_dir)
            if success:
                stats = verify_cbis_ddsm(data_dir)
                logger.info("数据验证: %s", stats)

    # ---- VinDr-Mammo ----
    if args.dataset in ("vindr", "all"):
        logger.info("=" * 50)
        logger.info("VinDr-Mammo")
        logger.info("=" * 50)

        if not args.verify_only:
            prepare_vindr(data_dir)

    logger.info("数据准备完成!")


if __name__ == "__main__":
    main()
