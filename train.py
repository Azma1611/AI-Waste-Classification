"""
train.py
========
MODULES 4, 5, 6, 7: Model Training, Transfer Learning, and Comparisons

Builds, trains, and compares three deep learning architectures for waste classification:
  1. Module 4: Custom CNN
  2. Module 5: MobileNetV2 Transfer Learning & Fine-Tuning
  3. Module 6: ResNet50 Transfer Learning & Fine-Tuning
  4. Module 7: Evaluation, Comparison, and Best Model Selection

Author  : AI & Data Science Engineering Team
Project : AI-Powered Smart Waste Classification & Recycling System
"""

import os
import sys
import time
import json
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
)
from sklearn.utils.class_weight import compute_class_weight

# Suppress noisy TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = "data"
MODEL_DIR = "model"
RESULTS_DIR = os.path.join(MODEL_DIR, "results")
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

# 6 canonical classes
CLASSES = ["Plastic", "Paper", "Glass", "Metal", "Organic Waste", "E-Waste"]

# Create directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATASET LOADING & SPLITTING
# ══════════════════════════════════════════════════════════════════════════════
def create_datasets(data_dir: str, test_mode: bool = False):
    """
    Scans the data directory, maps images, checks integrity,
    and returns train/val/test tf.data.Dataset splits (80/10/10).
    """
    if not os.path.isdir(data_dir):
        print(f"[ERROR] Data directory not found: {data_dir}. Run data_pipeline.py first!")
        sys.exit(1)

    all_paths, all_labels = [], []
    class_counts = {cls: 0 for cls in CLASSES}

    # Gather image file paths
    for idx, cls in enumerate(CLASSES):
        cls_folder = os.path.join(data_dir, cls)
        if not os.path.exists(cls_folder):
            continue
        for filename in os.listdir(cls_folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                all_paths.append(os.path.join(cls_folder, filename))
                all_labels.append(idx)
                class_counts[cls] += 1

    total = len(all_paths)
    if total == 0:
        print("[ERROR] Zero images found. Check data folder contents.")
        sys.exit(1)

    print(f"\n[INFO] Found {total} preprocessed images in total.")
    print("Class breakdown:")
    for cls, count in class_counts.items():
        print(f"  {cls:15s}  {count:5d} images")

    # Shuffle and split 80/10/10
    idx_shuffled = np.random.RandomState(SEED).permutation(total)
    paths_shuffled = np.array(all_paths)[idx_shuffled]
    labels_shuffled = np.array(all_labels)[idx_shuffled]

    if test_mode:
        print("[DEBUG] Slicing dataset to a tiny subset (128 images) for fast debug verification...")
        paths_shuffled = paths_shuffled[:128]
        labels_shuffled = labels_shuffled[:128]
        total = len(paths_shuffled)

    n_train = int(total * 0.80)
    n_val = int(total * 0.10)

    splits = {
        "train": (paths_shuffled[:n_train], labels_shuffled[:n_train]),
        "val": (paths_shuffled[n_train:n_train + n_val], labels_shuffled[n_train:n_train + n_val]),
        "test": (paths_shuffled[n_train + n_val:], labels_shuffled[n_train + n_val:]),
    }

    # tf.data.Dataset parsing functions
    def load_image(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0  # normalize [0, 1]
        label = tf.one_hot(label, len(CLASSES))
        return img, label

    def augment(img, label):
        # Apply moderate on-the-fly augmentation to training dataset only
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.15)
        img = tf.image.random_contrast(img, 0.85, 1.15)
        return img, label

    AUTOTUNE = tf.data.AUTOTUNE
    ds = {}

    for split, (paths, labels) in splits.items():
        d = tf.data.Dataset.from_tensor_slices((paths, labels))
        d = d.map(load_image, num_parallel_calls=AUTOTUNE)
        
        # Suppress image decoding errors gracefully
        try:
            d = d.ignore_errors(log_warning=False)
        except Exception:
            d = d.apply(tf.data.experimental.ignore_errors())

        if split == "train":
            d = d.map(augment, num_parallel_calls=AUTOTUNE)
            try:
                d = d.ignore_errors(log_warning=False)
            except Exception:
                d = d.apply(tf.data.experimental.ignore_errors())
            d = d.shuffle(1024, seed=SEED)

        d = d.batch(BATCH_SIZE).prefetch(AUTOTUNE)
        ds[split] = d

    return ds, splits["train"][1], splits["test"]


# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL ARCHITECTURES
# ══════════════════════════════════════════════════════════════════════════════

def build_custom_cnn(input_shape, num_classes) -> tf.keras.Model:
    """Module 4: Build Custom CNN Architecture."""
    inputs = tf.keras.Input(shape=input_shape)
    
    # Layer 1
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Layer 2
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Layer 3
    x = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Fully Connected
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    
    return tf.keras.Model(inputs, outputs, name="Custom_CNN")


def build_mobilenetv2_transfer(input_shape, num_classes):
    """Module 5: Transfer Learning with MobileNetV2 base."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    # Preprocess scaling to [-1, 1] as expected by MobileNetV2
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    
    model = tf.keras.Model(inputs, outputs, name="MobileNetV2_Transfer")
    return model, base_model


def build_resnet50_transfer(input_shape, num_classes):
    """Module 6: Transfer Learning with ResNet50 base."""
    base_model = tf.keras.applications.ResNet50(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    # Preprocess scaling as expected by ResNet50
    x = tf.keras.applications.resnet50.preprocess_input(inputs * 255.0)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    
    model = tf.keras.Model(inputs, outputs, name="ResNet50_Transfer")
    return model, base_model


# ══════════════════════════════════════════════════════════════════════════════
# 3. TRAINING & EVALUATION HELPER
# ══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(
    model_func,
    model_name: str,
    ds,
    train_labels,
    test_data,
    epochs_p1: int,
    epochs_p2: int = 0,
    base_model=None,
    fine_tune_unfreeze_count: int = 0,
):
    """
    Builds, trains, and evaluates a single model.
    Includes Phase 1 (classification head) and optional Phase 2 (fine-tuning).
    """
    print("\n" + "=" * 70)
    print(f"  TRAINING ARCHITECTURE: {model_name}")
    print("=" * 70)
    
    # 1. Compute class weights to balance dataset
    unique_labels = np.unique(train_labels)
    cw = compute_class_weight("balanced", classes=unique_labels, y=train_labels)
    class_weight = {int(k): float(v) for k, v in zip(unique_labels, cw)}

    # 2. Build model
    if base_model is None:
        # Custom CNN
        model = model_func((*IMG_SIZE, 3), len(CLASSES))
    else:
        # Transfer learning
        model, backbone = model_func((*IMG_SIZE, 3), len(CLASSES))

    # Compile for Phase 1
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        ),
    ]

    print(f"[Phase 1] Training classification head for up to {epochs_p1} epochs...")
    t0 = time.time()
    h1 = model.fit(
        ds["train"],
        epochs=epochs_p1,
        validation_data=ds["val"],
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )
    history_acc = h1.history["accuracy"]
    history_val_acc = h1.history["val_accuracy"]
    history_loss = h1.history["loss"]
    history_val_loss = h1.history["val_loss"]

    # Optional Phase 2: Fine-Tuning
    if epochs_p2 > 0 and base_model is not None and fine_tune_unfreeze_count > 0:
        print(f"\n[Phase 2] Fine-tuning. Unfreezing top {fine_tune_unfreeze_count} layers of base backbone...")
        backbone.trainable = True
        # Freeze all layers except the last N
        for layer in backbone.layers[:-fine_tune_unfreeze_count]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # very small LR for FT
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        h2 = model.fit(
            ds["train"],
            epochs=epochs_p2,
            validation_data=ds["val"],
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1,
        )
        history_acc += h2.history["accuracy"]
        history_val_acc += h2.history["val_accuracy"]
        history_loss += h2.history["loss"]
        history_val_loss += h2.history["val_loss"]

    elapsed_mins = (time.time() - t0) / 60.0
    print(f"[INFO] Completed training {model_name} in {elapsed_mins:.1f} minutes.")

    # Save individual training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history_acc, label="Train Acc")
    plt.plot(history_val_acc, label="Val Acc")
    if epochs_p2 > 0:
        plt.axvline(epochs_p1 - 1, color="red", linestyle="--", label="Fine-tune Start")
    plt.title(f"{model_name} Accuracy")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history_loss, label="Train Loss")
    plt.plot(history_val_loss, label="Val Loss")
    if epochs_p2 > 0:
        plt.axvline(epochs_p1 - 1, color="red", linestyle="--", label="Fine-tune Start")
    plt.title(f"{model_name} Loss")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{model_name.replace(' ', '_')}_curves.png"), dpi=150)
    plt.close()

    # 3. Evaluate on held-out test split
    print(f"Evaluating {model_name} on held-out test set...")
    test_paths, test_labels = test_data
    
    y_true, y_pred = [], []
    for images, labels in ds["test"]:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASSES, yticklabels=CLASSES, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix: {model_name}\nAccuracy: {acc*100:.1f}%")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{model_name.replace(' ', '_')}_cm.png"), dpi=150)
    plt.close()

    print(f"[{model_name}] Test Accuracy: {acc*100:.1f}%, F1-Score: {f1:.4f}")
    
    metrics = {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "Training Time (Min)": round(elapsed_mins, 2),
    }

    # Save model weights to temp file to compare sizes
    temp_model_path = os.path.join(MODEL_DIR, f"temp_{model_name.replace(' ', '_')}.keras")
    model.save(temp_model_path)
    
    return metrics, temp_model_path, acc


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Parse test mode command line argument ─────────────────────────────────
    # If "test" is passed as an argument, run a very quick 1-epoch demo
    test_mode = len(sys.argv) > 1 and sys.argv[1].lower() == "test"
    
    epochs_cnn = 1 if test_mode else 15
    epochs_tl_p1 = 1 if test_mode else 12
    epochs_tl_p2 = 1 if test_mode else 10

    if test_mode:
        print("\n>>> RUNNING IN TEST/DEBUG MODE (1 epoch per phase) <<<")

    # Step 1: Create dataset pipelines
    print("\n[INFO] Loading datasets...")
    ds, train_labels, test_data = create_datasets(DATA_DIR, test_mode=test_mode)

    # Store results for final comparison
    comparison_records = []
    saved_models = {}

    # Run Model 1: Custom CNN
    cnn_metrics, cnn_path, cnn_acc = train_and_evaluate(
        model_func=build_custom_cnn,
        model_name="Custom CNN",
        ds=ds,
        train_labels=train_labels,
        test_data=test_data,
        epochs_p1=epochs_cnn,
        epochs_p2=0,
        base_model=None,
    )
    comparison_records.append(cnn_metrics)
    saved_models["Custom CNN"] = (cnn_path, cnn_acc)

    # Run Model 2: MobileNetV2
    mobilenet_metrics, mobilenet_path, mobilenet_acc = train_and_evaluate(
        model_func=build_mobilenetv2_transfer,
        model_name="MobileNetV2 Transfer",
        ds=ds,
        train_labels=train_labels,
        test_data=test_data,
        epochs_p1=epochs_tl_p1,
        epochs_p2=epochs_tl_p2,
        base_model=True,
        fine_tune_unfreeze_count=50,
    )
    comparison_records.append(mobilenet_metrics)
    saved_models["MobileNetV2 Transfer"] = (mobilenet_path, mobilenet_acc)

    # Run Model 3: ResNet50
    resnet_metrics, resnet_path, resnet_acc = train_and_evaluate(
        model_func=build_resnet50_transfer,
        model_name="ResNet50 Transfer",
        ds=ds,
        train_labels=train_labels,
        test_data=test_data,
        epochs_p1=epochs_tl_p1,
        epochs_p2=epochs_tl_p2,
        base_model=True,
        fine_tune_unfreeze_count=30,
    )
    comparison_records.append(resnet_metrics)
    saved_models["ResNet50 Transfer"] = (resnet_path, resnet_acc)

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 7: MODEL COMPARISONS & SELECTION
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  MODULE 7: DEEP LEARNING MODEL COMPARISON")
    print("=" * 70)
    
    df_compare = pd.DataFrame(comparison_records)
    print(df_compare.to_string(index=False))
    
    # Save comparison report as CSV
    comparison_csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    df_compare.to_csv(comparison_csv_path, index=False)
    print(f"\n[SAVED] Comparison report CSV -> {comparison_csv_path}")

    # Generate visual model comparison chart
    plt.figure(figsize=(10, 5))
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score"]
    df_melted = df_compare.melt(id_vars="Model", value_vars=metrics_to_plot, var_name="Metric", value_name="Value")
    
    sns.barplot(data=df_melted, x="Metric", y="Value", hue="Model", palette="muted", edgecolor="black", linewidth=0.5)
    plt.ylim(0, 1.1)
    plt.title("Performance Comparison Across All Architectures", fontsize=14, fontweight="bold", pad=12)
    plt.ylabel("Score")
    plt.xlabel("Evaluation Metric")
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    comparison_png_path = os.path.join(RESULTS_DIR, "model_comparison_bar.png")
    plt.savefig(comparison_png_path, dpi=150)
    plt.close()
    print(f"[SAVED] Comparison chart PNG -> {comparison_png_path}")

    # Identify best model
    best_model_name = df_compare.loc[df_compare["Accuracy"].idxmax()]["Model"]
    best_model_acc = df_compare.loc[df_compare["Accuracy"].idxmax()]["Accuracy"]
    best_model_path, _ = saved_models[best_model_name]
    
    final_best_path = os.path.join(MODEL_DIR, "waste_model.keras")
    
    # Overwrite best model path
    if os.path.exists(final_best_path):
        try:
            os.remove(final_best_path)
        except Exception:
            pass
            
    import shutil
    shutil.copy2(best_model_path, final_best_path)
    print(f"\nWINNING ARCHITECTURE: {best_model_name} ({best_model_acc*100:.1f}% Accuracy)")
    print(f"   Saved as: {final_best_path}\n")

    # Clean up temp keras files
    for name, (path, _) in saved_models.items():
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass


if __name__ == "__main__":
    main()
