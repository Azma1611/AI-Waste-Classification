"""
train_engine.py  —  High-Accuracy Waste Classification Training
================================================================
Architecture : MobileNetV2 (Transfer Learning + Fine-Tuning)
Target       : ≥90% validation accuracy
Strategy     : Phase 1 → train head only | Phase 2 → fine-tune top 50 layers
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score,
                             recall_score, f1_score)
from sklearn.utils.class_weight import compute_class_weight

# ── Suppress noisy TF logs ────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR  = "dataset"          # root folder of downloaded images
MODEL_DIR    = "model"
MODEL_PATH   = os.path.join(MODEL_DIR, "waste_model.keras")
RESULTS_DIR  = os.path.join(MODEL_DIR, "results")
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 16
PHASE1_EPOCHS = 20     # head-only training
PHASE2_EPOCHS = 15     # fine-tuning
SEED         = 42

# The 6 official assignment categories + their dataset folder names
# (adjust right-hand values if your dataset uses different folder names)
CLASS_MAP = {
    "Plastic"      : ["Plastic",       "plastic", "plastic waste"],
    "Paper"        : ["Paper",         "paper", "paper waste"],
    "Glass"        : ["Glass",         "glass", "glass waste"],
    "Metal"        : ["Metal",         "metal", "metal waste"],
    "Organic Waste": ["Organic Waste", "organic", "Organic", "organic waste"],
    "E-Waste"      : ["E-Waste",       "ewaste",  "e-waste", "E_waste", "e-waste"],
}
CLASSES = list(CLASS_MAP.keys())   # canonical order

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATASET DISCOVERY & PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def create_datasets(dataset_root: str):
    """Build train/val/test tf.data.Dataset objects from the raw folder."""
    if not os.path.isdir(dataset_root):
        print(f"[ERROR] Dataset directory not found: {dataset_root}")
        sys.exit(1)

    all_paths, all_labels = [], []
    
    # Precompute lowercase aliases for fast matching
    alias_to_class_idx = {}
    for idx, (canon, aliases) in enumerate(CLASS_MAP.items()):
        for a in aliases:
            alias_to_class_idx[a.lower()] = idx

    # Recursively walk the dataset
    class_counts = {cls: 0 for cls in CLASSES}
    
    for dirpath, _, files in os.walk(dataset_root):
        folder_name = os.path.basename(dirpath).lower()
        
        # Check if this folder matches any of our known class aliases
        if folder_name in alias_to_class_idx:
            class_idx = alias_to_class_idx[folder_name]
            canon_name = CLASSES[class_idx]
            
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    all_paths.append(os.path.join(dirpath, f))
                    all_labels.append(class_idx)
                    class_counts[canon_name] += 1

    total = len(all_paths)
    if total == 0:
        print("[ERROR] Zero images found. Check dataset path and folder names.")
        sys.exit(1)

    print(f"\n[INFO] Found {total} total images. Running integrity check...")
    from PIL import Image
    from concurrent.futures import ThreadPoolExecutor

    valid_paths, valid_labels = [], []
    corrupted = 0

    def check_img(idx):
        path = all_paths[idx]
        try:
            with Image.open(path) as img:
                img.load()
                img.convert("RGB")
            return idx, True
        except Exception:
            return idx, False

    with ThreadPoolExecutor(max_workers=32) as executor:
        for idx, is_valid in executor.map(check_img, range(total)):
            if is_valid:
                valid_paths.append(all_paths[idx])
                valid_labels.append(all_labels[idx])
            else:
                corrupted += 1
                
    if corrupted > 0:
        print(f"[WARN] Dropped {corrupted} corrupted or unreadable images.")
        
    all_paths = valid_paths
    all_labels = valid_labels
    total = len(all_paths)

    print("\n[INFO] Images found per category (valid only):")
    for cls, count in class_counts.items():
        print(f"  {cls:15s}  {count:5d} images")
    print(f"\n  Total images found: {total}\n")

    # Shuffle then split  80 / 10 / 10
    idx   = np.random.RandomState(SEED).permutation(total)
    paths = np.array(all_paths)[idx]
    labels = np.array(all_labels)[idx]

    n_train = int(total * 0.80)
    n_val   = int(total * 0.10)

    splits = {
        "train": (paths[:n_train],          labels[:n_train]),
        "val":   (paths[n_train:n_train+n_val], labels[n_train:n_train+n_val]),
        "test":  (paths[n_train+n_val:],    labels[n_train+n_val:]),
    }

    def load_image(path, label):
        raw   = tf.io.read_file(path)
        img   = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img   = tf.image.resize(img, IMG_SIZE)
        img   = tf.cast(img, tf.float32) / 255.0
        label = tf.one_hot(label, len(CLASSES))
        return img, label

    def augment(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_saturation(img, 0.8, 1.2)
        img = tf.image.random_hue(img, 0.05)
        # Random zoom via crop-and-resize
        boxes   = tf.random.uniform([1, 4], minval=[0,0,0,0],
                                    maxval=[0.2, 0.2, 1.0, 1.0])
        boxes   = tf.clip_by_value(boxes, 0.0, 1.0)
        img     = tf.image.crop_and_resize(
            tf.expand_dims(img, 0),
            boxes, [0], IMG_SIZE)[0]
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img, label

    AUTOTUNE = tf.data.AUTOTUNE
    
    ds = {}
    for split, (p, l) in splits.items():
        d = tf.data.Dataset.from_tensor_slices((p, l))
        d = d.map(load_image, num_parallel_calls=AUTOTUNE)
        try:
            d = d.ignore_errors(log_warning=True)
        except Exception:
            d = d.apply(tf.data.experimental.ignore_errors())
            
        if split == "train":
            d = d.map(augment, num_parallel_calls=AUTOTUNE)
            try:
                d = d.ignore_errors(log_warning=True)
            except Exception:
                d = d.apply(tf.data.experimental.ignore_errors())
            d = d.shuffle(2048, seed=SEED)
            
        d = d.batch(BATCH_SIZE).prefetch(AUTOTUNE)
        ds[split] = d

    return ds, splits["train"][1]   # also return raw train labels for class weights


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL  (MobileNetV2 Transfer Learning)
# ══════════════════════════════════════════════════════════════════════════════
def build_model(num_classes: int) -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False   # Phase 1: frozen

    inputs  = tf.keras.Input(shape=(*IMG_SIZE, 3))
    # MobileNetV2 preprocess_input scales to [-1, +1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation="relu",
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation="relu",
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="WasteNet_MobileNetV2")
    return model, base


# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def train(model, base, ds, train_labels):
    # ── Class weights (handles imbalanced datasets) ───────────────────────────
    unique_labels = np.unique(train_labels)
    cw = compute_class_weight("balanced", classes=unique_labels, y=train_labels)
    class_weight = {int(k): float(v) for k, v in zip(unique_labels, cw)}

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    # ── PHASE 1: Train classification head (frozen backbone) ─────────────────
    print("\n" + "="*60)
    print("  PHASE 1 — Training classification head (backbone frozen)")
    print("="*60)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    h1 = model.fit(
        ds["train"], epochs=PHASE1_EPOCHS,
        validation_data=ds["val"],
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # ── PHASE 2: Fine-tune top 50 layers of backbone ─────────────────────────
    print("\n" + "="*60)
    print("  PHASE 2 — Fine-tuning top 50 layers of MobileNetV2")
    print("="*60)
    base.trainable = True
    for layer in base.layers[:-50]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # very low LR
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    h2 = model.fit(
        ds["train"], epochs=PHASE2_EPOCHS,
        validation_data=ds["val"],
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    return h1, h2


# ══════════════════════════════════════════════════════════════════════════════
# 5. EVALUATION & REPORTS
# ══════════════════════════════════════════════════════════════════════════════
def evaluate(model, ds_test):
    print("\n" + "="*60)
    print("  EVALUATION on held-out test set")
    print("="*60)

    y_true, y_pred = [], []
    for images, labels in ds_test:
        preds  = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    results = {
        "Model": "MobileNetV2 (Fine-tuned)",
        "Accuracy":  f"{acc*100:.2f}%",
        "Precision": f"{prec:.4f}",
        "Recall":    f"{rec:.4f}",
        "F1-Score":  f"{f1:.4f}",
    }
    df = pd.DataFrame([results])
    print("\n", df.to_string(index=False))
    df.to_csv(os.path.join(RESULTS_DIR, "metrics.csv"), index=False)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=CLASSES, yticklabels=CLASSES,
                cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix  |  Accuracy: {acc*100:.2f}%")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
    print(f"\n[SAVED] Confusion matrix → {RESULTS_DIR}/confusion_matrix.png")

    # Classification report
    report = classification_report(y_true, y_pred, target_names=CLASSES)
    print("\nClassification Report:\n", report)
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    return acc


# ══════════════════════════════════════════════════════════════════════════════
# 6. TRAINING CURVES
# ══════════════════════════════════════════════════════════════════════════════
def plot_history(h1, h2):
    acc  = h1.history["accuracy"]  + h2.history["accuracy"]
    val  = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss = h1.history["loss"] + h2.history["loss"]
    vloss= h1.history["val_loss"] + h2.history["val_loss"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(acc,  label="Train Acc")
    ax1.plot(val,  label="Val Acc")
    ax1.axvline(len(h1.history["accuracy"]), color="gray", linestyle="--",
                label="Fine-tune start")
    ax1.set_title("Accuracy")
    ax1.legend()

    ax2.plot(loss,  label="Train Loss")
    ax2.plot(vloss, label="Val Loss")
    ax2.axvline(len(h1.history["loss"]), color="gray", linestyle="--",
                label="Fine-tune start")
    ax2.set_title("Loss")
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "training_curves.png")
    plt.savefig(path, dpi=150)
    print(f"[SAVED] Training curves → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    print("\n" + "="*60)
    print("  AI Waste Classification — High-Accuracy Training Engine")
    print("="*60)
    print(f"  TensorFlow : {tf.__version__}")
    print(f"  GPU        : {[g.name for g in tf.config.list_physical_devices('GPU')] or 'CPU only'}")

    # Step 1 & 2 — Discover and Build data pipelines
    print("\n[INFO] Building data pipeline...")
    ds, train_labels = create_datasets(DATASET_DIR)

    # Step 3 — Build model
    print("\n[INFO] Building MobileNetV2 model...")
    model, base = build_model(len(CLASSES))
    model.summary(line_length=80)

    # Step 4 — Train (Phase 1 + Phase 2)
    t0 = time.time()
    h1, h2 = train(model, base, ds, train_labels)
    elapsed = time.time() - t0
    print(f"\n[INFO] Total training time: {elapsed/60:.1f} minutes")

    # Step 5 — Evaluate
    acc = evaluate(model, ds["test"])

    # Step 6 — Save training curves
    plot_history(h1, h2)

    # Step 7 — Save the final model explicitly
    model.save(MODEL_PATH)
    print(f"\n[SAVED] Model → {MODEL_PATH}")

    # Summary
    print("\n" + "="*60)
    if acc >= 0.90:
        print(f"  ✅  TARGET ACHIEVED: {acc*100:.2f}% ≥ 90% accuracy!")
    else:
        print(f"  ⚠️   Accuracy: {acc*100:.2f}%  (target: ≥90%)")
        print("       Try: more epochs, larger dataset, or stronger augmentation.")
    print("="*60 + "\n")
