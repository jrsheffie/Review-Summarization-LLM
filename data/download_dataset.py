"""
download_dataset.py

Downloads the Amazon Product Reviews dataset from Kaggle.

Prerequisites (one-time setup):
    1. Create a free Kaggle account.
    2. Go to Account -> Create New API Token. This downloads kaggle.json.
    3. Place it at ~/.kaggle/kaggle.json (Colab: see notebooks/00_colab_setup.ipynb
       for how to upload it there instead).
    4. chmod 600 ~/.kaggle/kaggle.json   (permissions requirement on Linux/Colab)

Usage:
    pip install kaggle
    python data/download_dataset.py
"""

import subprocess
import sys
from pathlib import Path

DATASET_SLUG = "arhamrumi/amazon-product-reviews"
RAW_DIR = Path(__file__).resolve().parent / "raw"


def download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import kaggle  # noqa: F401
    except (ImportError, OSError) as e:
        sys.exit(
            "Kaggle package not found or kaggle.json not configured.\n"
            "Run: pip install kaggle, then place kaggle.json in ~/.kaggle/\n"
            f"Original error: {e}"
        )

    subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", DATASET_SLUG,
            "-p", str(RAW_DIR),
            "--unzip",
        ],
        check=True,
    )
    print(f"Dataset downloaded to {RAW_DIR}")
    print("Next: check the downloaded CSV's column names against COLUMN_MAP in data/preprocess.py")


if __name__ == "__main__":
    download()
