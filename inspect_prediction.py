import os
import numpy as np
import tensorflow as tf
from PIL import Image

# Config
MODEL_PATH = "model/waste_model.keras"
CLASSES = ["Plastic", "Paper", "Glass", "Metal", "Organic Waste", "E-Waste"]

def inspect_image_prediction(image_path):
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    # 1. Load Model
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # 2. Load and Preprocess Image
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # 3. Model Inference
    preds = model.predict(img_batch, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    
    # 4. Extract Top-3 Predictions
    top_3_indices = np.argsort(preds)[::-1][:3]
    
    print("\n" + "="*50)
    print(f" DIAGNOSTICS FOR: {os.path.basename(image_path)}")
    print("="*50)
    print(f"Raw Prediction Vector:\n{preds}\n")
    print(f"Predicted Class Index: {pred_idx} -> {CLASSES[pred_idx]}")
    print(f"Confidence:            {preds[pred_idx] * 100:.4f}%\n")
    
    print("Top-3 Probabilities:")
    for i, idx in enumerate(top_3_indices):
        print(f"  {i+1}. {CLASSES[idx]:<15s} : {preds[idx] * 100:.4f}%")
    print("="*50)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python inspect_prediction.py <path_to_image>")
    else:
        inspect_image_prediction(sys.argv[1])
