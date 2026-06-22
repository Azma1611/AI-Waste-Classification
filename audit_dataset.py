import os
import hashlib
import numpy as np
from PIL import Image

def get_image_hash(img_path):
    """Generate MD5 hash of resized, grayscale image to find duplicates."""
    try:
        with Image.open(img_path) as img:
            img_gray = img.convert("L").resize((32, 32), Image.Resampling.BILINEAR)
            data = list(img_gray.getdata())
            return hashlib.md5(bytes(data)).hexdigest()
    except Exception:
        return None

def audit_classes(class_list=["Plastic", "Glass"]):
    hashes = {}
    duplicates = []
    corrupted = []
    
    for cls in class_list:
        folder = os.path.join("data", cls)
        if not os.path.exists(folder):
            continue
        print(f"Auditing folder: {folder}...")
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            
            # Verify load integrity
            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception:
                corrupted.append(path)
                continue
                
            h = get_image_hash(path)
            if h is not None:
                if h in hashes:
                    duplicates.append((path, hashes[h]))
                else:
                    hashes[h] = path
                    
    print("\n" + "="*50)
    print(" DATASET AUDIT SUMMARY")
    print("="*50)
    print(f"Corrupted Images Found: {len(corrupted)}")
    for c in corrupted:
        print(f"  [CORRUPTED] {c}")
        
    print(f"\nDuplicate / Identical Images Found: {len(duplicates)}")
    for dup, orig in duplicates[:10]:
        print(f"  Duplicate: {os.path.basename(dup)} (Matches: {os.path.basename(orig)})")
    if len(duplicates) > 10:
        print(f"  ... and {len(duplicates) - 10} more.")
    print("="*50)

if __name__ == "__main__":
    audit_classes()
