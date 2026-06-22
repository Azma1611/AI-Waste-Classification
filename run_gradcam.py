import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

MODEL_PATH = "model/waste_model.keras"
CLASSES = ["Plastic", "Paper", "Glass", "Metal", "Organic Waste", "E-Waste"]

def get_gradcam_overlay(image_path, target_class_name="Glass"):
    model = tf.keras.models.load_model(MODEL_PATH)
    target_idx = CLASSES.index(target_class_name)

    # 1. Preprocess input image
    orig_img = Image.open(image_path).convert("RGB")
    img_resized = orig_img.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # 2. Automatically discover last Conv layer
    last_conv_layer_name = None
    for layer in model.layers:
        if hasattr(layer, "layers"): # nested model check
            for sub in layer.layers:
                if isinstance(sub, tf.keras.layers.Conv2D):
                    last_conv_layer_name = sub.name
                if sub.name in ("out_relu", "conv5_block3_3_relu"):
                    last_conv_layer_name = sub.name
        elif isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name

    print(f"Discovered last Conv layer: {last_conv_layer_name}")

    # Retrieve layers to build gradient model
    # Handle nested architecture lookup
    try:
        backbone = model.get_layer("mobilenetv2_1.00_224") # default layer name
        target_layer = backbone.get_layer(last_conv_layer_name)
        grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[target_layer.output, model.output]
        )
    except Exception:
        try:
            target_layer = model.get_layer(last_conv_layer_name)
            grad_model = tf.keras.Model(inputs=model.input, outputs=[target_layer.output, model.output])
        except Exception as e:
            # Direct backup
            print(f"Error accessing layer: {e}. Attempting direct submodel extraction...")
            # Extract nested layers manually
            sub_model = [l for l in model.layers if hasattr(l, "layers")][0]
            target_layer = sub_model.get_layer(last_conv_layer_name)
            grad_model = tf.keras.Model(inputs=model.input, outputs=[target_layer.output, model.output])

    # 3. Compute Gradients
    img_tensor = tf.cast(img_batch, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, target_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() != 0:
        heatmap = heatmap / heatmap.max()

    # 4. Generate visual overlay
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    orig_np = np.array(img_resized)
    superimposed = cv2.addWeighted(orig_np, 0.6, heatmap_colored, 0.4, 0)

    # Save output
    os.makedirs("scratch", exist_ok=True)
    out_path = "scratch/gradcam_waste_diagnostic.png"
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_resized)
    plt.title("Original Input")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(superimposed)
    plt.title(f"Grad-CAM overlay for '{target_class_name}'")
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved diagnostic overlay: {out_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python run_gradcam.py <path_to_image>")
    else:
        get_gradcam_overlay(sys.argv[1])
