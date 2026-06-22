import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = "data"
MODEL_PATH = "model/waste_model.keras"
CLASSES = ["Plastic", "Paper", "Glass", "Metal", "Organic Waste", "E-Waste"]
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

def load_test_dataset():
    all_paths, all_labels = [], []
    for idx, cls in enumerate(CLASSES):
        cls_folder = os.path.join(DATA_DIR, cls)
        if not os.path.exists(cls_folder):
            continue
        for f in os.listdir(cls_folder):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                all_paths.append(os.path.join(cls_folder, f))
                all_labels.append(idx)
    
    # Deterministic Split matching train.py seed
    total = len(all_paths)
    idx_shuffled = np.random.RandomState(42).permutation(total)
    paths_shuffled = np.array(all_paths)[idx_shuffled]
    labels_shuffled = np.array(all_labels)[idx_shuffled]
    
    # Extract 10% test split
    n_train = int(total * 0.80)
    n_val = int(total * 0.10)
    test_paths = paths_shuffled[n_train + n_val:]
    test_labels = labels_shuffled[n_train + n_val:]
    
    return test_paths, test_labels

def parse_image(path, label):
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def run_evaluation():
    os.makedirs("scratch", exist_ok=True)
    test_paths, test_labels = load_test_dataset()
    ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
    ds = ds.map(parse_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    model = tf.keras.models.load_model(MODEL_PATH)
    
    y_true, y_pred, y_probs = [], [], []
    for images, labels in ds:
        preds = model.predict(images, verbose=0)
        y_probs.extend(preds)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())
        
    y_true, y_pred, y_probs = np.array(y_true), np.array(y_pred), np.array(y_probs)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASSES, yticklabels=CLASSES, cmap="Blues")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.title("Evaluation Confusion Matrix")
    plt.savefig("scratch/confusion_matrix.png", dpi=150)
    plt.close()
    
    # 2. Classification Report
    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv("scratch/per_class_report.csv")
    
    # 3. Misclassification Detail Analysis
    misclass_records = []
    for i in range(len(test_paths)):
        if y_true[i] != y_pred[i]:
            misclass_records.append({
                "File": test_paths[i],
                "Actual": CLASSES[y_true[i]],
                "Predicted": CLASSES[y_pred[i]],
                "Prob_Actual": y_probs[i][y_true[i]],
                "Prob_Predicted": y_probs[i][y_pred[i]]
            })
            
    df_misclass = pd.DataFrame(misclass_records)
    df_misclass.to_csv("scratch/misclassifications.csv", index=False)
    
    print("\n--- Evaluation Finished ---")
    print(f"Per-Class metrics saved to: scratch/per_class_report.csv")
    print(f"Detailed misclassifications saved to: scratch/misclassifications.csv")
    print(f"Visual Confusion Matrix plot saved to: scratch/confusion_matrix.png")

if __name__ == "__main__":
    run_evaluation()
