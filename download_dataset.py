"""
download_dataset.py  —  Robust Kaggle dataset downloader with progress reporting
"""
import os
import sys
import subprocess
import zipfile
import shutil

DATASET  = "wasifmahmood01/custom-waste-classification-dataset"
DEST_ZIP = "dataset_raw"
FINAL    = "dataset"

# Kaggle expects credentials at ~/.kaggle/kaggle.json
kaggle_src = os.path.join(os.path.expanduser("~"), "Downloads", "kaggle.json")
kaggle_dst = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
os.makedirs(os.path.dirname(kaggle_dst), exist_ok=True)
if os.path.exists(kaggle_src) and not os.path.exists(kaggle_dst):
    shutil.copy(kaggle_src, kaggle_dst)
    os.chmod(kaggle_dst, 0o600)
    print(f"[OK] kaggle.json copied to {kaggle_dst}")

if not os.path.exists(kaggle_dst):
    print("[ERROR] kaggle.json not found. Place it at ~/Downloads/kaggle.json")
    sys.exit(1)

print(f"[START] Downloading: {DATASET}")
print("       This may take 5–15 minutes depending on internet speed...\n")

result = subprocess.run(
    [sys.executable, "-m", "kaggle", "datasets", "download",
     "-d", DATASET, "--unzip", "-p", FINAL],
    capture_output=False
)

if result.returncode == 0:
    # Count images per class
    total = 0
    for root, dirs, files in os.walk(FINAL):
        imgs = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if imgs:
            cls = os.path.basename(root)
            print(f"  [{cls}]  {len(imgs)} images")
            total += len(imgs)
    print(f"\n[DONE] {total} images ready in '{FINAL}/' folder.")
else:
    print(f"\n[FAIL] Download error. Return code: {result.returncode}")
    print("Tip: Try running:  .venv\\Scripts\\python.exe -m kaggle datasets download -d wasifmahmood/custom-waste-classification-dataset --unzip -p dataset")
