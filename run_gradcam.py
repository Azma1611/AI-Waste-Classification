import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import gradcam

MODEL_PATH = "model/waste_model.keras"
CLASSES = ["Plastic", "Paper", "Glass", "Metal", "Organic Waste", "E-Waste"]

def get_gradcam_overlay(image_path, target_class_name="Glass"):
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return
        
    model = tf.keras.models.load_model(MODEL_PATH)
    
    if target_class_name not in CLASSES:
        print(f"Error: target_class_name '{target_class_name}' must be one of {CLASSES}")
        return
    target_idx = CLASSES.index(target_class_name)

    # 1. Load Image
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    orig_img = Image.open(image_path).convert("RGB")
    
    # 2. Get predictions and generate heatmap using our robust pipeline
    img_resized = orig_img.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Find layer and generate Grad-CAM++ heatmap
    conv_layer_name = gradcam.find_target_explain_layer(model)
    print(f"Targeting layer for explanation: {conv_layer_name}")
    
    heatmap = gradcam.generate_gradcam_heatmap(
        model,
        img_batch,
        target_class_idx=target_idx,
        conv_layer_name=conv_layer_name,
        use_gradcam_plusplus=True
    )
    
    # 3. Generate visual overlay with Guided Image Filtering (Research-Quality)
    superimposed = gradcam.overlay_heatmap_on_image(img_resized, heatmap, alpha=0.5)

    # 4. Save ONLY the final single overlay image directly (no subplots)
    os.makedirs("scratch", exist_ok=True)
    out_path = "scratch/gradcam_waste_diagnostic.png"
    
    # Convert RGB to BGR for cv2 writing
    cv2.imwrite(out_path, cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))
    print(f"Saved research-quality diagnostic overlay directly to: {out_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python run_gradcam.py <path_to_image> [target_class_name]")
    else:
        target_cls = sys.argv[2] if len(sys.argv) > 2 else "Glass"
        get_gradcam_overlay(sys.argv[1], target_cls)
