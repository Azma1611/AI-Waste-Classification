"""
evaluate.py
===========
MODULE 7: Detailed Model Evaluation & Validation Metrics

Loads the final selected waste model (model/waste_model.keras), runs inference
on the held-out test split, generates classification report, confusion matrix
heatmap, and outputs a per-class metrics table.

Author  : AI & Data Science Engineering Team
Project : AI-Powered Smart Waste Classification & Recycling System
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support
)

# Suppress TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Import datasets config from train.py to ensure identical splits
try:
    from train import create_datasets, CLASSES, DATA_DIR, MODEL_DIR, RESULTS_DIR
except ImportError:
    print("[ERROR] Could not import from train.py. Ensure train.py is in the same directory.")
    sys.exit(1)

MODEL_PATH = os.path.join(MODEL_DIR, "waste_model.keras")


def run_evaluation():
    print("\n" + "=" * 70)
    print("  STANDALONE EVALUATION MODULE — BEST SAVED MODEL")
    print("=" * 70)

    # 1. Load saved model
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Saved model not found at '{MODEL_PATH}'.")
        print("Please train a model first by running: python train.py")
        sys.exit(1)

    print(f"[INFO] Loading final model from '{MODEL_PATH}'...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    # 2. Get dataset splits
    print("[INFO] Loading dataset test split...")
    ds, _, test_data = create_datasets(DATA_DIR)
    
    test_paths, test_labels = test_data
    total_test_samples = len(test_paths)
    print(f"Test split contains {total_test_samples} images.")

    # 3. Predict on test split
    print("\nRunning inference on test dataset (batch processing)...")
    y_true, y_pred, y_prob = [], [], []

    for idx, (images, labels) in enumerate(ds["test"]):
        preds = model.predict(images, verbose=0)
        y_prob.extend(preds)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # 4. Compute Global Metrics
    acc = accuracy_score(y_true, y_pred)
    prec_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_w = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\n" + "-" * 45)
    print("  GLOBAL PERFORMANCE SUMMARY")
    print("-" * 45)
    print(f"  Test Accuracy : {acc * 100:.2f}%")
    print(f"  Precision     : {prec_w:.4f} (weighted)")
    print(f"  Recall        : {rec_w:.4f} (weighted)")
    print(f"  F1-Score      : {f1_w:.4f} (weighted)")
    print("-" * 45)

    # Save summary metrics to text file
    summary_path = os.path.join(RESULTS_DIR, "global_metrics_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("AI-Powered Smart Waste Classifier - Global Metrics\n")
        f.write("==================================================\n")
        f.write(f"Test Accuracy : {acc * 100:.2f}%\n")
        f.write(f"Precision     : {prec_w:.4f}\n")
        f.write(f"Recall        : {rec_w:.4f}\n")
        f.write(f"F1-Score      : {f1_w:.4f}\n")
    print(f"Global metrics saved to '{summary_path}'")

    # 5. Compute Per-Class Performance Table
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(CLASSES)), zero_division=0
    )

    per_class_records = []
    for idx, name in enumerate(CLASSES):
        per_class_records.append({
            "Class Name": name,
            "Precision": round(precision[idx], 4),
            "Recall": round(recall[idx], 4),
            "F1-Score": round(f1[idx], 4),
            "Support (Images)": int(support[idx]),
        })

    df_per_class = pd.DataFrame(per_class_records)
    print("\nPER-CLASS PERFORMANCE METRICS:")
    print(df_per_class.to_string(index=False))

    per_class_csv_path = os.path.join(RESULTS_DIR, "per_class_metrics.csv")
    df_per_class.to_csv(per_class_csv_path, index=False)
    print(f"\n[SAVED] Per-class metrics table CSV -> {per_class_csv_path}")

    # 6. Classification Report
    report = classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0)
    print("\nCLASSIFICATION REPORT:")
    print(report)
    
    report_path = os.path.join(RESULTS_DIR, "classification_report_best.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[SAVED] Text classification report -> {report_path}")

    # 7. Confusion Matrix Heatmap
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages for confusion matrix
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_percent = np.nan_to_num(cm_percent)

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Annotate with count and percentage
    annotations = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f"{cm[i, j]}\n({cm_percent[i, j]*100:.1f}%)"

    sns.heatmap(
        cm,
        annot=annotations,
        fmt="",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        cbar=True,
        ax=ax,
        edgecolor="grey",
        linewidths=0.5
    )
    
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title(f"Confusion Matrix (Best Model)\nAccuracy: {acc*100:.2f}%", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix_best_model.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"[SAVED] Confusion matrix image -> {cm_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_evaluation()
