"""
data_pipeline.py
================
AI-Powered Smart Waste Classification System
Data Pipeline & Processing Engine

Responsibilities:
  1. Directory structure creation for 6 waste classes.
  2. MD5-based strict duplicate image elimination.
  3. In-memory image preprocessing (resize + normalize).
  4. Sequential data augmentation (rotation, flip, zoom, brightness).
  5. EDA reporting: bar chart, pie chart, RGB histogram.

Author  : AI & Data Science Engineering Team
Project : Smart Waste Classification & Recycling Recommendation System
"""

import os
import hashlib
import logging
import random
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                 # Non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageEnhance

# ─── Logging Configuration ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
WASTE_CLASSES   = ["Plastic", "Paper", "Glass", "Metal", "Organic Waste", "E-Waste"]
DATA_ROOT       = "data"
EDA_OUTPUT_DIR  = os.path.join(DATA_ROOT, "eda_reports")
VALID_EXTS      = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TARGET_SIZE     = (224, 224)          # MobileNetV2-compatible input size
AUGMENT_PER_IMG = 3                   # Augmented copies to generate per source image


# ─── 1. Directory Structure Creation ─────────────────────────────────────────
def create_directory_structure(root: str, classes: list[str]) -> dict[str, str]:
    """
    Create the root data directory and one sub-folder per waste class.

    Returns
    -------
    class_dirs : dict mapping class name → absolute path
    """
    class_dirs: dict[str, str] = {}
    os.makedirs(root, exist_ok=True)
    os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)

    for cls in classes:
        path = os.path.join(root, cls)
        os.makedirs(path, exist_ok=True)
        class_dirs[cls] = path
        logger.info("Directory ensured: %s", path)

    logger.info("Directory structure creation complete (%d classes).", len(classes))
    return class_dirs


# ─── 2. Duplicate Elimination via MD5 Hashing ────────────────────────────────
def _md5_of_file(filepath: str) -> str:
    """Return the hex MD5 digest of a file's raw bytes (streaming, memory-safe)."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def remove_duplicates(class_dirs: dict[str, str]) -> int:
    """
    Scan every class directory for exact duplicate image files using MD5 hashing.
    The first occurrence of each hash is kept; subsequent duplicates are deleted.

    Returns
    -------
    total_removed : int — cumulative count of deleted files across all classes
    """
    total_removed = 0

    for cls, directory in class_dirs.items():
        seen_hashes: dict[str, str] = {}   # hash → first file path
        removed_in_class = 0

        for filename in sorted(os.listdir(directory)):
            if os.path.splitext(filename)[1].lower() not in VALID_EXTS:
                continue

            filepath = os.path.join(directory, filename)
            digest   = _md5_of_file(filepath)

            if digest in seen_hashes:
                logger.warning(
                    "Duplicate found in [%s]: '%s' matches '%s'. Removing.",
                    cls, filename, os.path.basename(seen_hashes[digest])
                )
                os.remove(filepath)
                removed_in_class += 1
            else:
                seen_hashes[digest] = filepath

        logger.info(
            "[%s] Duplicate scan complete. Removed %d duplicate(s).",
            cls, removed_in_class
        )
        total_removed += removed_in_class

    logger.info("Total duplicates removed across all classes: %d", total_removed)
    return total_removed


# ─── 3. In-Memory Image Preprocessing ────────────────────────────────────────
def preprocess_image(filepath: str) -> np.ndarray:
    """
    Load one image file, resize it to TARGET_SIZE, convert to RGB, and
    normalize pixel values to the range [0.0, 1.0].

    Returns
    -------
    img_array : float32 ndarray of shape (224, 224, 3)
    """
    img       = Image.open(filepath).convert("RGB")
    img       = img.resize(TARGET_SIZE, Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0   # strict [0, 1] normalization
    assert img_array.min() >= 0.0 and img_array.max() <= 1.0, \
        f"Normalization out of range for {filepath}"
    return img_array


# ─── 4. Sequential Data Augmentation ─────────────────────────────────────────
def augment_image(img: Image.Image) -> Image.Image:
    """
    Apply a sequential chain of randomised augmentations to a PIL Image:
      - Random rotation  : ±20°
      - Random horizontal flip
      - Random zoom crop : 85–100% of the image area
      - Random brightness: 0.75× – 1.25× of original

    The output is always resized back to TARGET_SIZE so shape is consistent.
    """
    # 1. Random rotation — up to 20° in either direction
    angle = random.uniform(-20.0, 20.0)
    img   = img.rotate(angle, resample=Image.BICUBIC, expand=False)

    # 2. Random horizontal flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # 3. Random zoom (crop then resize back to original dimensions)
    zoom_factor = random.uniform(0.85, 1.0)     # 1.0 = no zoom; <1.0 = zoom in
    w, h        = img.size
    crop_w      = int(w * zoom_factor)
    crop_h      = int(h * zoom_factor)
    left        = random.randint(0, w - crop_w)
    top         = random.randint(0, h - crop_h)
    img         = img.crop((left, top, left + crop_w, top + crop_h))
    img         = img.resize(TARGET_SIZE, Image.LANCZOS)

    # 4. Random brightness variation (0.75 = darker, 1.25 = brighter)
    brightness_factor = random.uniform(0.75, 1.25)
    enhancer          = ImageEnhance.Brightness(img)
    img               = enhancer.enhance(brightness_factor)

    return img


def generate_augmented_images(class_dirs: dict[str, str], copies_per_image: int = AUGMENT_PER_IMG) -> int:
    """
    For every valid source image in each class directory, produce `copies_per_image`
    augmented variants and save them with an `_aug<n>` suffix.

    Returns
    -------
    total_generated : int — total augmented images written to disk
    """
    total_generated = 0

    for cls, directory in class_dirs.items():
        source_files = [
            f for f in os.listdir(directory)
            if os.path.splitext(f)[1].lower() in VALID_EXTS
            and "_aug" not in f                # skip already-augmented files
        ]

        for filename in source_files:
            filepath = os.path.join(directory, filename)
            stem, ext = os.path.splitext(filename)

            try:
                source_img = Image.open(filepath).convert("RGB").resize(TARGET_SIZE, Image.LANCZOS)
            except Exception as exc:
                logger.error("Cannot open image '%s': %s", filepath, exc)
                continue

            for n in range(1, copies_per_image + 1):
                augmented   = augment_image(source_img)
                out_name    = f"{stem}_aug{n}{ext}"
                out_path    = os.path.join(directory, out_name)
                augmented.save(out_path)
                total_generated += 1

        logger.info("[%s] Augmentation complete — %d new images generated.", cls, len(source_files) * copies_per_image)

    logger.info("Total augmented images saved: %d", total_generated)
    return total_generated


# ─── 5. EDA Reporting & Visualisations ────────────────────────────────────────
def _collect_class_counts(class_dirs: dict[str, str]) -> pd.DataFrame:
    """
    Count valid image files in each class directory and return a tidy DataFrame.

    Returns
    -------
    df : pd.DataFrame with columns ['Class', 'Count']
    """
    records = []
    for cls, directory in class_dirs.items():
        count = sum(
            1 for f in os.listdir(directory)
            if os.path.splitext(f)[1].lower() in VALID_EXTS
        )
        records.append({"Class": cls, "Count": count})
    return pd.DataFrame(records)


def _sample_pixel_arrays(class_dirs: dict[str, str], max_samples: int = 300) -> np.ndarray:
    """
    Randomly sample up to `max_samples` images across all class directories,
    preprocess each one, and stack them into a single float32 array.

    Returns
    -------
    pixel_stack : ndarray of shape (N, 224, 224, 3)
    """
    all_paths: list[str] = []
    for directory in class_dirs.values():
        all_paths.extend(
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.splitext(f)[1].lower() in VALID_EXTS
        )

    sampled  = random.sample(all_paths, min(max_samples, len(all_paths)))
    arrays   = []
    for path in sampled:
        try:
            arrays.append(preprocess_image(path))
        except Exception as exc:
            logger.warning("Skipping image during pixel sampling: %s — %s", path, exc)

    if not arrays:
        logger.warning("No valid images found for pixel sampling.")
        return np.empty((0, *TARGET_SIZE, 3), dtype=np.float32)

    return np.stack(arrays, axis=0)


def generate_eda_reports(class_dirs: dict[str, str]) -> None:
    """
    Produce and save exactly three EDA figures to EDA_OUTPUT_DIR:
      Figure 1 — Bar chart   : image count per class
      Figure 2 — Pie chart   : percentage distribution across classes
      Figure 3 — Histogram   : per-channel (R, G, B) pixel intensity distribution
    """
    df = _collect_class_counts(class_dirs)
    logger.info("Class distribution summary:\n%s", df.to_string(index=False))

    palette = sns.color_palette("viridis", len(WASTE_CLASSES))

    # ── Figure 1: Bar Chart ──────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    # hue=Class + legend=False avoids the seaborn v0.14 palette deprecation warning
    sns.barplot(data=df, x="Class", y="Count", hue="Class", legend=False,
                palette=palette, ax=ax1, edgecolor="black", linewidth=0.6)
    ax1.set_title("Image Count Distribution Across Waste Classes", fontsize=14, fontweight="bold", pad=12)
    ax1.set_xlabel("Waste Category", fontsize=11)
    ax1.set_ylabel("Number of Images", fontsize=11)
    ax1.tick_params(axis="x", rotation=20)
    for bar in ax1.patches:
        height = bar.get_height()
        if not math.isnan(height):
            ax1.annotate(
                f"{int(height)}",
                (bar.get_x() + bar.get_width() / 2.0, height),
                ha="center", va="bottom", fontsize=9, fontweight="bold"
            )
    fig1.tight_layout()
    bar_path = os.path.join(EDA_OUTPUT_DIR, "fig1_class_distribution_bar.png")
    fig1.savefig(bar_path, dpi=150)
    plt.close(fig1)
    logger.info("Saved Figure 1 (bar chart) → %s", bar_path)

    # ── Figure 2: Pie Chart ──────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    total_images = df["Count"].sum()
    if total_images > 0:
        wedge_props = {"linewidth": 0.8, "edgecolor": "white"}
        ax2.pie(
            df["Count"],
            labels=df["Class"],
            autopct="%1.1f%%",
            startangle=140,
            colors=palette,
            wedgeprops=wedge_props,
            pctdistance=0.82,
        )
    else:
        # Render a placeholder equal-slice pie when no data exists yet
        ax2.pie(
            [1] * len(WASTE_CLASSES),
            labels=WASTE_CLASSES,
            colors=palette,
            wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
            pctdistance=0.82,
        )
        ax2.text(0, 0, "No images\nyet", ha="center", va="center",
                 fontsize=13, fontweight="bold", color="grey")
        logger.warning("Pie chart rendered as placeholder — dataset folders are empty.")
    ax2.set_title("Waste Class Percentage Distribution", fontsize=14, fontweight="bold", pad=16)
    fig2.tight_layout()
    pie_path = os.path.join(EDA_OUTPUT_DIR, "fig2_class_distribution_pie.png")
    fig2.savefig(pie_path, dpi=150)
    plt.close(fig2)
    logger.info("Saved Figure 2 (pie chart) → %s", pie_path)

    # ── Figure 3: RGB Pixel Intensity Histogram ──────────────────────────────
    pixel_stack = _sample_pixel_arrays(class_dirs)

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    if pixel_stack.shape[0] > 0:
        channel_configs = [
            ("R", pixel_stack[:, :, :, 0], "#e63946"),
            ("G", pixel_stack[:, :, :, 1], "#2a9d8f"),
            ("B", pixel_stack[:, :, :, 2], "#457b9d"),
        ]
        bins = np.linspace(0.0, 1.0, 64)
        for label, channel_data, color in channel_configs:
            ax3.hist(
                channel_data.ravel(),
                bins=bins,
                alpha=0.55,
                color=color,
                label=f"{label} channel",
                density=True,
            )
        ax3.set_title("RGB Pixel Intensity Distribution (Normalised [0, 1])", fontsize=14, fontweight="bold", pad=12)
        ax3.set_xlabel("Pixel Intensity", fontsize=11)
        ax3.set_ylabel("Density", fontsize=11)
        ax3.legend(fontsize=10)
    else:
        ax3.text(0.5, 0.5, "No images available for histogram.",
                 ha="center", va="center", fontsize=12, transform=ax3.transAxes)
        ax3.set_title("RGB Pixel Intensity Distribution", fontsize=14, fontweight="bold")

    fig3.tight_layout()
    hist_path = os.path.join(EDA_OUTPUT_DIR, "fig3_rgb_pixel_histogram.png")
    fig3.savefig(hist_path, dpi=150)
    plt.close(fig3)
    logger.info("Saved Figure 3 (RGB histogram) → %s", hist_path)

    logger.info("EDA reporting complete. All 3 figures saved to '%s'.", EDA_OUTPUT_DIR)


# ─── Pipeline Orchestrator ────────────────────────────────────────────────────
def run_pipeline() -> None:
    """
    Orchestrate the full data pipeline in a strict, sequential order:
      Step 1 → Create directory structure
      Step 2 → Remove duplicate images
      Step 3 → Generate augmented images
      Step 4 → Run EDA and save visualisations
    """
    logger.info("=" * 60)
    logger.info("SMART WASTE CLASSIFICATION — DATA PIPELINE STARTED")
    logger.info("=" * 60)

    # Step 1: Directory structure
    logger.info("STEP 1: Creating directory structure...")
    class_dirs = create_directory_structure(DATA_ROOT, WASTE_CLASSES)

    # Step 2: Duplicate elimination
    logger.info("STEP 2: Scanning for and removing duplicate images...")
    duplicates_removed = remove_duplicates(class_dirs)
    logger.info("Duplicates eliminated: %d", duplicates_removed)

    # Step 3: Data augmentation
    logger.info("STEP 3: Generating augmented images (%d copies per source)...", AUGMENT_PER_IMG)
    augmented_count = generate_augmented_images(class_dirs)
    logger.info("Augmentation complete. Total new images: %d", augmented_count)

    # Step 4: EDA
    logger.info("STEP 4: Generating EDA reports and visualisations...")
    generate_eda_reports(class_dirs)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE. All outputs saved under '%s/'.", DATA_ROOT)
    logger.info("=" * 60)


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline()
