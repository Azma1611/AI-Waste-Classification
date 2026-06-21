"""
train_engine.py
===============
AI-Powered Smart Waste Classification System
Deep Learning Training, Evaluation, and Export Engine

Workflows (executed sequentially, zero code duplication):
  1. Data generator construction — built once, shared across all models.
  2. Model instantiation — Custom CNN, MobileNetV2, ResNet50.
  3. Centralised training loop with EarlyStopping & ModelCheckpoint callbacks.
  4. Centralised evaluation — Accuracy, Precision, Recall, F1-Score.
  5. Comparison DataFrame output.
  6. Confusion matrix heatmap for the best model.
  7. Best model exported as `model/waste_model.keras`.

Author  : AI & Data Science Engineering Team
Project : Smart Waste Classification & Recycling Recommendation System
"""

import os
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Global Constants ─────────────────────────────────────────────────────────
WASTE_CLASSES   = ["Plastic", "Paper", "Glass", "Metal", "Organic Waste", "E-Waste"]
NUM_CLASSES     = len(WASTE_CLASSES)
IMG_SIZE        = (224, 224)
INPUT_SHAPE     = (224, 224, 3)
BATCH_SIZE      = 32
EPOCHS          = 30          # EarlyStopping will halt training earlier if needed
DATA_ROOT       = "data"
MODEL_DIR       = "model"
RESULTS_DIR     = os.path.join(MODEL_DIR, "results")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "waste_model.keras")

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA GENERATORS (built once, reused across all model pipelines)
# ════════════════════════════════════════════════════════════════════════════════

def build_data_generators(data_root: str):
    """
    Construct three ImageDataGenerator flows (train / validation / test) from
    the directory structure created by data_pipeline.py.

    The generators are created once and passed to every model's training loop —
    no data split is ever rebuilt for a second model.

    Returns
    -------
    train_gen, val_gen, test_gen : DirectoryIterator objects
    class_indices                : dict mapping class-name → integer label
    """
    # Training augmentation (augmentation only on training split)
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.10,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.80, 1.20],
        validation_split=0.20,        # reserve 20 % of data for validation
    )

    # Test/validation: only rescale — no stochastic augmentation
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        data_root,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=WASTE_CLASSES,
        subset="training",
        shuffle=True,
        seed=42,
    )

    val_gen = train_datagen.flow_from_directory(
        data_root,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=WASTE_CLASSES,
        subset="validation",
        shuffle=False,
        seed=42,
    )

    # Test generator (shuffle=False → ordered predictions for confusion matrix)
    test_gen = eval_datagen.flow_from_directory(
        data_root,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=WASTE_CLASSES,
        shuffle=False,
    )

    logger.info(
        "Generators ready — train: %d samples | val: %d samples | test: %d samples",
        train_gen.samples, val_gen.samples, test_gen.samples,
    )
    return train_gen, val_gen, test_gen, train_gen.class_indices


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MODEL ARCHITECTURES
# ════════════════════════════════════════════════════════════════════════════════

def build_custom_cnn(num_classes: int, input_shape: tuple) -> tf.keras.Model:
    """
    Custom CNN built from scratch:
      Conv2D (32) → MaxPool → Conv2D (64) → MaxPool →
      Conv2D (128) → MaxPool → Conv2D (256) → MaxPool →
      Flatten → Dense (256) → Dropout → Dense (num_classes, softmax)
    """
    model = models.Sequential(name="Custom_CNN")

    # Block 1
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 2
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 3
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 4
    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Classifier head
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.50))
    model.add(layers.Dense(num_classes, activation="softmax"))

    logger.info("Custom CNN architecture built — %d parameters.", model.count_params())
    return model


def build_mobilenetv2(num_classes: int, input_shape: tuple) -> tf.keras.Model:
    """
    MobileNetV2 transfer-learning model:
      - Base: pre-trained ImageNet weights, top=False.
      - Base layers frozen initially; top 30 layers unfrozen for fine-tuning.
      - Custom head: GlobalAveragePooling → Dense(128) → Dropout → Softmax.
    """
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    # Freeze entire base first, then unfreeze top 30 layers for fine-tuning
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.40)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="MobileNetV2_FineTuned")
    logger.info("MobileNetV2 architecture built — %d parameters.", model.count_params())
    return model


def build_resnet50(num_classes: int, input_shape: tuple) -> tf.keras.Model:
    """
    ResNet50 transfer-learning model:
      - Base: pre-trained ImageNet weights, top=False.
      - All base layers frozen; only the classification head is trained.
      - Custom head: GlobalAveragePooling → Dense(256) → Dropout → Softmax.
    """
    base = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    base.trainable = False   # Freeze entire base (fine-tune head only)

    inputs = tf.keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.50)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="ResNet50_FineTuned")
    logger.info("ResNet50 architecture built — %d parameters.", model.count_params())
    return model


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CENTRALISED TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════════════

def get_callbacks(model_name: str) -> list:
    """
    Return a standard callback suite for every model:
      - EarlyStopping : halts training when val_loss stops improving (patience=5).
      - ModelCheckpoint: saves the epoch-best weights to disk.
      - ReduceLROnPlateau: halves LR when val_loss plateaus (patience=3).
    """
    ckpt_path = os.path.join(RESULTS_DIR, f"{model_name}_best_weights.keras")
    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def compile_and_train(
    model: tf.keras.Model,
    train_gen,
    val_gen,
    learning_rate: float = 1e-3,
) -> tf.keras.callbacks.History:
    """
    Compile a model with Adam + categorical cross-entropy, then fit it on the
    pre-built generators.  The same function is called for every architecture —
    no compilation or training logic is duplicated.

    Returns
    -------
    history : Keras History object
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    logger.info("Training model: %s", model.name)
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=get_callbacks(model.name),
        verbose=1,
    )
    logger.info("Training complete: %s", model.name)
    return history


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CENTRALISED EVALUATION (Accuracy, Precision, Recall, F1)
# ════════════════════════════════════════════════════════════════════════════════

def evaluate_model(model: tf.keras.Model, test_gen) -> dict:
    """
    Run inference on the full test generator and compute four metrics.
    All metric calculations are performed in this single function —
    no metric logic is repeated elsewhere.

    Returns
    -------
    metrics : dict with keys: Model, Accuracy, Precision, Recall, F1-Score
    """
    test_gen.reset()
    y_pred_probs = model.predict(test_gen, verbose=0)
    y_pred       = np.argmax(y_pred_probs, axis=1)
    y_true       = test_gen.classes

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    logger.info(
        "[%s] Acc=%.4f | Prec=%.4f | Rec=%.4f | F1=%.4f",
        model.name, acc, prec, rec, f1,
    )
    return {
        "Model":     model.name,
        "Accuracy":  round(acc,  4),
        "Precision": round(prec, 4),
        "Recall":    round(rec,  4),
        "F1-Score":  round(f1,   4),
    }


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 5 — EVALUATION OUTPUT: COMPARISON DATAFRAME
# ════════════════════════════════════════════════════════════════════════════════

def build_comparison_dataframe(metrics_list: list[dict]) -> pd.DataFrame:
    """
    Convert the list of per-model metric dicts into a formatted DataFrame,
    sort by F1-Score descending, and save it as a CSV.

    Returns
    -------
    df : pd.DataFrame
    """
    df = pd.DataFrame(metrics_list).sort_values("F1-Score", ascending=False).reset_index(drop=True)
    csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    df.to_csv(csv_path, index=False)
    logger.info("Model comparison table:\n%s", df.to_string(index=False))
    logger.info("Comparison saved → %s", csv_path)
    return df


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 6 — CONFUSION MATRIX HEATMAP (best model only)
# ════════════════════════════════════════════════════════════════════════════════

def save_confusion_matrix(model: tf.keras.Model, test_gen, class_names: list) -> None:
    """
    Predict on the test generator, compute the NxN confusion matrix, and save
    a seaborn heatmap annotated with counts to RESULTS_DIR.
    """
    test_gen.reset()
    y_pred = np.argmax(model.predict(test_gen, verbose=0), axis=1)
    y_true = test_gen.classes

    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model.name}", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()

    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix_best_model.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved → %s", cm_path)

    # Also log the per-class classification report
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    logger.info("Classification report (best model):\n%s", report)


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 7 — MODEL EXPORT
# ════════════════════════════════════════════════════════════════════════════════

def export_best_model(model: tf.keras.Model, export_path: str) -> None:
    """
    Save the best-performing model in the native Keras v3 format (.keras).
    This format produces a single, compressed file with a typical footprint
    of 8–14 MB for MobileNetV2-scale networks — well under the 15 MB target.
    """
    model.save(export_path)
    size_mb = os.path.getsize(export_path) / (1024 ** 2)
    logger.info(
        "Best model exported → %s  (%.2f MB)",
        export_path, size_mb,
    )
    if size_mb > 15.0:
        logger.warning(
            "Model file exceeds the 15 MB target (%.2f MB). "
            "Consider post-training quantization to reduce size.", size_mb,
        )


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 8 — TRAINING HISTORY PLOT UTILITY
# ════════════════════════════════════════════════════════════════════════════════

def save_training_curves(history: tf.keras.callbacks.History, model_name: str) -> None:
    """
    Plot and save accuracy and loss training curves for a single model.
    Called once per trained model — no curve-plotting logic is duplicated.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history["accuracy"],     label="Train Accuracy", color="#2a9d8f")
    ax1.plot(history.history["val_accuracy"], label="Val Accuracy",   color="#e9c46a")
    ax1.set_title(f"{model_name} — Accuracy", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(history.history["loss"],     label="Train Loss", color="#e63946")
    ax2.plot(history.history["val_loss"], label="Val Loss",   color="#457b9d")
    ax2.set_title(f"{model_name} — Loss", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    curve_path = os.path.join(RESULTS_DIR, f"{model_name}_training_curves.png")
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    logger.info("Training curves saved → %s", curve_path)


# ════════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR — run_training_engine
# ════════════════════════════════════════════════════════════════════════════════

def run_training_engine() -> None:
    """
    Top-level orchestrator. Executes all seven sections in strict sequence:
      1. Build shared data generators (once).
      2. Instantiate all three model architectures.
      3. Train each model using the centralised compile_and_train function.
      4. Evaluate each model using the centralised evaluate_model function.
      5. Build and save the comparison DataFrame.
      6. Save the confusion matrix for the best model.
      7. Export the best model as waste_model.keras.
    """
    logger.info("=" * 65)
    logger.info("SMART WASTE CLASSIFICATION — TRAINING ENGINE STARTED")
    logger.info("=" * 65)

    # ── Step 1: Data Generators (built exactly once) ─────────────────────────
    logger.info("STEP 1: Building shared data generators from '%s'...", DATA_ROOT)
    train_gen, val_gen, test_gen, class_indices = build_data_generators(DATA_ROOT)

    if train_gen.samples == 0:
        logger.error(
            "No training images found in '%s'. "
            "Run data_pipeline.py and populate the class directories first.",
            DATA_ROOT,
        )
        return

    # ── Step 2: Instantiate all three architectures ───────────────────────────
    logger.info("STEP 2: Instantiating model architectures...")
    model_specs = [
        # (model_object, learning_rate)
        (build_custom_cnn(NUM_CLASSES, INPUT_SHAPE),    1e-3),
        (build_mobilenetv2(NUM_CLASSES, INPUT_SHAPE),   1e-4),   # lower LR for fine-tuning
        (build_resnet50(NUM_CLASSES, INPUT_SHAPE),      1e-4),
    ]

    # ── Steps 3 & 4: Train then immediately evaluate each model ──────────────
    logger.info("STEP 3+4: Training and evaluating each model sequentially...")
    metrics_list: list[dict] = []
    trained_models: list[tuple] = []    # stores (model, metrics_dict)

    for model, lr in model_specs:
        # Train
        history = compile_and_train(model, train_gen, val_gen, learning_rate=lr)
        save_training_curves(history, model.name)

        # Evaluate (centralised — called identically for every model)
        metrics = evaluate_model(model, test_gen)
        metrics_list.append(metrics)
        trained_models.append((model, metrics))

    # ── Step 5: Comparison DataFrame ─────────────────────────────────────────
    logger.info("STEP 5: Building model comparison DataFrame...")
    comparison_df = build_comparison_dataframe(metrics_list)
    print("\n" + "=" * 55)
    print("       MODEL PERFORMANCE COMPARISON")
    print("=" * 55)
    print(comparison_df.to_string(index=False))
    print("=" * 55 + "\n")

    # ── Step 6 & 7: Best model → confusion matrix + export ───────────────────
    best_model_name = comparison_df.iloc[0]["Model"]
    best_model      = next(m for m, _ in trained_models if m.name == best_model_name)
    best_f1         = comparison_df.iloc[0]["F1-Score"]

    logger.info(
        "STEP 6: Generating confusion matrix for best model: %s (F1=%.4f)",
        best_model_name, best_f1,
    )
    save_confusion_matrix(best_model, test_gen, WASTE_CLASSES)

    logger.info("STEP 7: Exporting best model → %s", BEST_MODEL_PATH)
    export_best_model(best_model, BEST_MODEL_PATH)

    logger.info("=" * 65)
    logger.info("TRAINING ENGINE COMPLETE.")
    logger.info("Best model : %s | F1 = %.4f", best_model_name, best_f1)
    logger.info("Artifacts  : %s/", RESULTS_DIR)
    logger.info("Model file : %s", BEST_MODEL_PATH)
    logger.info("=" * 65)


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Suppress verbose TensorFlow INFO/WARNING logs in the console
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    run_training_engine()
