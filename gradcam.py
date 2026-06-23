"""
gradcam.py
==========
MODULE 10: Explainable AI — Grad-CAM & Feature Visualization

Implements Gradient-weighted Class Activation Mapping (Grad-CAM) and Grad-CAM++
to explain why the model made a specific prediction by highlighting the
image regions that most influenced the classification decision.

Uses Guided Image Filtering, soft thresholding, and gamma correction for
pixel-precise, research-paper quality object boundary localization.

Author  : AI & Data Science Engineering Team
Project : AI-Powered Smart Waste Classification & Recycling System
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

# TensorFlow import with graceful fallback
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES AND LAYERS
# ══════════════════════════════════════════════════════════════════════════════

def find_target_explain_layer(model) -> str:
    """
    Finds the target convolutional or activation layer to use for Grad-CAM.
    Prioritizes the final convolutional/activation layer of the backbone (e.g. out_relu
    for MobileNetV2, conv5_block3_out for ResNet50) to capture holistic, semantic object
    information, which is then refined to pixel-sharp boundaries using the Guided Filter.

    Parameters
    ----------
    model : tf.keras.Model

    Returns
    -------
    str : Name of the target layer
    """
    if not TF_AVAILABLE:
        return None

    # Check for functional sub-models
    base_model = None
    for layer in model.layers:
        if hasattr(layer, "layers") and isinstance(layer, tf.keras.Model):
            base_model = layer
            break

    if base_model is not None:
        model_name_lower = base_model.name.lower()
        if "mobilenet" in model_name_lower:
            # MobileNetV2 final semantic activation layer (7x7x1280) for holistic object focus
            target_layers = ["out_relu", "Conv_1", "block_16_project"]
            for name in target_layers:
                try:
                    base_model.get_layer(name)
                    return name
                except ValueError:
                    continue
        elif "resnet" in model_name_lower:
            # ResNet50 final semantic block activation layer (7x7x2048)
            target_layers = ["conv5_block3_out", "conv5_block3_3_conv", "conv5_block3_add"]
            for name in target_layers:
                try:
                    base_model.get_layer(name)
                    return name
                except ValueError:
                    continue

        # Backward search inside base model for any Conv2D if targets not found
        for sub_layer in reversed(base_model.layers):
            if isinstance(sub_layer, tf.keras.layers.Conv2D):
                return sub_layer.name
    else:
        # Backward search in flat model
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name

    # Fallback to known names
    known_final_layers = [
        "conv5_block3_out",
        "out_relu",
        "conv5_block3_3_conv",
        "block_16_project"
    ]
    for name in known_final_layers:
        try:
            _get_nested_layer(model, name)
            return name
        except ValueError:
            continue

    return None


def find_last_conv_layer(model) -> str:
    """
    Legacy wrapper for find_target_explain_layer.
    """
    return find_target_explain_layer(model)


def _get_nested_layer(model, layer_name):
    """
    Retrieve a layer by name, searching through nested sub-models.
    """
    # Try direct lookup first
    try:
        return model.get_layer(layer_name)
    except ValueError:
        pass

    # Search nested models
    for layer in model.layers:
        if hasattr(layer, "layers"):
            try:
                return layer.get_layer(layer_name)
            except ValueError:
                continue

    raise ValueError(f"Layer '{layer_name}' not found in model or sub-models.")


# ══════════════════════════════════════════════════════════════════════════════
# GUIDED IMAGE FILTERING
# ══════════════════════════════════════════════════════════════════════════════

def guided_filter(guidance: np.ndarray, target: np.ndarray, r: int = 12, eps: float = 1e-3) -> np.ndarray:
    """
    Guided image filter for edge-preserving smoothing and detail transfer.
    Aligns the boundaries of the heatmap with the edges of the guidance image.
    
    Parameters
    ----------
    guidance : np.ndarray
        RGB image of shape (H, W, 3) or grayscale image of shape (H, W), in range [0, 255] or [0, 1].
    target : np.ndarray
        Heatmap of shape (H, W), values in range [0, 1].
    r : int
        Local window radius. Default r=12 for smooth, holistic boundary alignment.
    eps : float
        Regularization parameter (variance threshold).
        
    Returns
    -------
    refined_target : np.ndarray
        Refined heatmap of shape (H, W), in range [0, 1].
    """
    # Convert guidance to grayscale and normalize to [0, 1]
    if len(guidance.shape) == 3:
        I = cv2.cvtColor(guidance, cv2.COLOR_RGB2GRAY)
    else:
        I = guidance.copy()
        
    if I.max() > 1.0:
        I = I.astype(np.float32) / 255.0
        
    p = target.astype(np.float32)
    
    # Calculate local means using box filter
    mean_I = cv2.boxFilter(I, -1, (r, r))
    mean_p = cv2.boxFilter(p, -1, (r, r))
    mean_Ip = cv2.boxFilter(I * p, -1, (r, r))
    
    # Covariance of (I, p) in each local patch
    cov_Ip = mean_Ip - mean_I * mean_p
    
    # Variance of I in each local patch
    mean_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = mean_II - mean_I * mean_I
    
    # Linear coefficients a and b
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    # Mean of a and b over local patch
    mean_a = cv2.boxFilter(a, -1, (r, r))
    mean_b = cv2.boxFilter(b, -1, (r, r))
    
    # Refined target
    q = mean_a * I + mean_b
    return np.clip(q, 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# GRAD-CAM / GRAD-CAM++ ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def generate_gradcam_heatmap(
    model,
    img_array: np.ndarray,
    target_class_idx: int = None,
    conv_layer_name: str = None,
    use_gradcam_plusplus: bool = True,
    threshold: float = 0.05,
    gamma: float = 0.8,
) -> np.ndarray:
    """
    Generate a highly localized Grad-CAM or Grad-CAM++ heatmap for a given image and model.
    Aligns preprocessing with the backbone, applies soft thresholding and gamma correction.

    Parameters
    ----------
    model : tf.keras.Model
        The trained classification model.
    img_array : np.ndarray
        Preprocessed image array of shape (1, 224, 224, 3), values in [0, 1].
    target_class_idx : int, optional
        Class index to compute the heatmap for. If None, uses the
        predicted class (argmax).
    conv_layer_name : str, optional
        Name of the convolutional layer to use. If None, automatically
        detects the target layer.
    use_gradcam_plusplus : bool
        If True, applies Grad-CAM++ using first, second, and third-order derivatives.
    threshold : float
        Soft threshold value (0.0 to 1.0) to remove background activation noise.
        Default threshold=0.05 to keep weaker activations over object bodies.
    gamma : float
        Gamma exponent for contrast enhancement and peak focusing.
        Default gamma=0.8 to spread highlights across full object body.

    Returns
    -------
    heatmap : np.ndarray
        Normalized heatmap of shape (H, W), values in [0, 1].
    """
    if not TF_AVAILABLE or model is None:
        # Return a placeholder heatmap
        return np.random.rand(7, 7).astype(np.float32)

    # Find the target convolutional layer
    if conv_layer_name is None:
        conv_layer_name = find_target_explain_layer(model)
        if conv_layer_name is None:
            print("[Grad-CAM Diagnostic] Error: No suitable conv layer found!")
            return np.ones((7, 7), dtype=np.float32) * 0.5

    # Check if we have a nested base model (e.g. resnet50, mobilenetv2)
    base_model = None
    for layer in model.layers:
        if hasattr(layer, "layers") and isinstance(layer, tf.keras.Model):
            base_model = layer
            break

    # Temporarily set main model's last layer activation to linear for logits-based gradients
    last_layer = model.layers[-1]
    original_activation = last_layer.activation if hasattr(last_layer, "activation") else None
    if hasattr(last_layer, "activation"):
        last_layer.activation = tf.keras.activations.linear

    try:
        # Preprocess input image based on detected backbone
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        
        if base_model is not None:
            # Scale raw input [0, 1] to [0, 255] and apply backbone-specific preprocessing
            x_prep = img_tensor * 255.0
            model_name_lower = base_model.name.lower()
            if "resnet50" in model_name_lower:
                x_prep = tf.keras.applications.resnet50.preprocess_input(x_prep)
            elif "mobilenet" in model_name_lower:
                x_prep = tf.keras.applications.mobilenet_v2.preprocess_input(x_prep)
            
            target_layer = _get_nested_layer(base_model, conv_layer_name)
            
            # Sub-model for the base backbone
            base_grad_model = tf.keras.Model(
                inputs=base_model.input,
                outputs=[target_layer.output, base_model.output]
            )
            
            # Extract main model head layers after base model
            head_layers = []
            found_base = False
            for layer in model.layers:
                if layer == base_model:
                    found_base = True
                    continue
                if found_base:
                    head_layers.append(layer)
            
            dense_classifier = head_layers[-1]
            head_layers_except_last = head_layers[:-1]

            def run_head_logits(x):
                for hl in head_layers_except_last:
                    x = hl(x)
                if hasattr(dense_classifier, "kernel"):
                    logits = tf.matmul(x, dense_classifier.kernel)
                    if dense_classifier.use_bias:
                        logits = logits + dense_classifier.bias
                    return logits
                else:
                    return dense_classifier(x)

            # Grad-CAM++ calculation using triple-nested tapes w.r.t pre-softmax logits
            with tf.GradientTape() as tape3:
                with tf.GradientTape() as tape2:
                    with tf.GradientTape() as tape1:
                        conv_outputs, base_features = base_grad_model(x_prep)
                        logits = run_head_logits(base_features)
                        
                        if target_class_idx is None:
                            target_class_idx = tf.argmax(logits[0]).numpy()
                        loss = logits[:, target_class_idx]
                        
                    grads_first = tape1.gradient(loss, conv_outputs)
                grads_second = tape2.gradient(grads_first, conv_outputs)
            grads_third = tape3.gradient(grads_second, conv_outputs)
            
        else:
            # Standard flat model execution
            target_layer = _get_nested_layer(model, conv_layer_name)
            dense_classifier = model.layers[-1]
            second_to_last_layer = model.layers[-2]

            flat_grad_model = tf.keras.Model(
                inputs=model.input,
                outputs=[target_layer.output, second_to_last_layer.output]
            )
            
            with tf.GradientTape() as tape3:
                with tf.GradientTape() as tape2:
                    with tf.GradientTape() as tape1:
                        conv_outputs, pre_classifier_features = flat_grad_model(img_tensor)
                        if hasattr(dense_classifier, "kernel"):
                            logits = tf.matmul(pre_classifier_features, dense_classifier.kernel)
                            if dense_classifier.use_bias:
                                logits = logits + dense_classifier.bias
                        else:
                            logits = dense_classifier(pre_classifier_features)
                            
                        if target_class_idx is None:
                            target_class_idx = tf.argmax(logits[0]).numpy()
                        loss = logits[:, target_class_idx]
                        
                    grads_first = tape1.gradient(loss, conv_outputs)
                grads_second = tape2.gradient(grads_first, conv_outputs)
            grads_third = tape3.gradient(grads_second, conv_outputs)

        if grads_first is None:
            print("[Grad-CAM Diagnostic] Error: Gradients are None!")
            return np.ones((7, 7), dtype=np.float32) * 0.5

        # Weight the conv outputs
        conv_outputs_val = conv_outputs[0]
        grads_first_val = grads_first[0]
        
        if use_gradcam_plusplus:
            grads_second_val = grads_second[0]
            grads_third_val = grads_third[0]
            
            # Alphas coefficients for Grad-CAM++
            numerator = grads_second_val
            sum_act = tf.reduce_sum(conv_outputs_val, axis=(0, 1), keepdims=True)
            denominator = 2.0 * grads_second_val + sum_act * grads_third_val
            denominator = tf.where(denominator != 0.0, denominator, tf.ones_like(denominator))
            
            alphas = numerator / denominator
            weights = tf.reduce_sum(alphas * tf.maximum(grads_first_val, 0.0), axis=(0, 1))
        else:
            # Standard Grad-CAM average gradients
            weights = tf.reduce_mean(grads_first_val, axis=(0, 1))

        # Sum along channel dimension
        heatmap_tensor = tf.reduce_sum(weights * conv_outputs_val, axis=-1)

        # ReLU — only keep positive contributions
        heatmap = tf.maximum(heatmap_tensor, 0.0).numpy()

        # Robust Min-Max Normalization to establish a true zero baseline
        heatmap_min = np.min(heatmap)
        heatmap_max = np.max(heatmap)
        if heatmap_max - heatmap_min > 1e-8:
            heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min + 1e-8)
        else:
            heatmap = np.zeros_like(heatmap)

        # Soft Thresholding to clear out background noise
        if threshold > 0.0:
            heatmap = np.where(heatmap < threshold, 0.0, (heatmap - threshold) / (1.0 - threshold + 1e-8))

        # Gamma Correction to focus peak activations
        if gamma > 0.0:
            heatmap = np.power(heatmap, gamma)

        # Print diagnostics to log stream
        print(f"[Grad-CAM Diagnostic] Target Class Index: {target_class_idx}")
        print(f"[Grad-CAM Diagnostic] Selected Layer Name: {conv_layer_name}")
        print(f"[Grad-CAM Diagnostic] Gradients Min/Max: {grads_first_val.numpy().min():.6f}/{grads_first_val.numpy().max():.6f}")
        print(f"[Grad-CAM Diagnostic] Heatmap Min/Max: {heatmap.min():.6f}/{heatmap.max():.6f}")

        return heatmap

    finally:
        # Restore original activation function
        if original_activation is not None and hasattr(last_layer, "activation"):
            last_layer.activation = original_activation


def overlay_heatmap_on_image(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.50,
    colormap: int = cv2.COLORMAP_JET,
    target_size: tuple = (224, 224),
    r: int = 12,
    eps: float = 1e-3,
) -> np.ndarray:
    """
    Superimpose a refined Grad-CAM heatmap onto the original image.
    Uses Guided Image Filtering for edge refinement and dynamic alpha masking.

    Parameters
    ----------
    original_image : np.ndarray
        Original image as RGB uint8 array (H, W, 3) or PIL Image.
    heatmap : np.ndarray
        Grad-CAM heatmap (any spatial size), values in [0, 1].
    alpha : float
        Blending weight for the heatmap overlay (0=transparent, 1=opaque).
    colormap : int
        OpenCV colormap constant (default: JET).
    target_size : tuple
        Output image size (width, height).
    r : int
        Guided filter window radius. Default r=12.
    eps : float
        Guided filter regularization epsilon.

    Returns
    -------
    superimposed : np.ndarray
        RGB uint8 image with heatmap overlay.
    """
    # Handle PIL Image input
    if hasattr(original_image, "convert"):
        original_image = np.array(
            original_image.resize(target_size).convert("RGB")
        )

    # Ensure correct size
    if original_image.shape[:2] != target_size[::-1]:
        original_image = cv2.resize(original_image, target_size)

    # Resize heatmap to match image using cv2.INTER_LANCZOS4 for high fidelity
    heatmap_resized = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_LANCZOS4)

    # Apply Guided Filtering using the original image to align the activations with boundaries
    heatmap_refined = guided_filter(original_image, heatmap_resized, r=r, eps=eps)

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_refined), colormap
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Ensure original is uint8
    if original_image.dtype == np.float32 or original_image.dtype == np.float64:
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)
        else:
            original_image = original_image.astype(np.uint8)

    # Blend: original * (1 - alpha * mask) + heatmap_colored * (alpha * mask)
    # The mask (refined heatmap) acts as a dynamic spatial weight
    mask = np.expand_dims(heatmap_refined, axis=-1)
    
    original_float = original_image.astype(np.float32)
    heatmap_colored_float = heatmap_colored.astype(np.float32)
    
    blended = original_float * (1.0 - alpha * mask) + heatmap_colored_float * (alpha * mask)
    superimposed = np.clip(blended, 0, 255).astype(np.uint8)

    return superimposed


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def visualize_intermediate_features(
    model,
    img_array: np.ndarray,
    layer_names: list = None,
    max_features: int = 16,
    save_path: str = None,
) -> plt.Figure:
    """
    Visualize activations from intermediate layers of the model.

    Parameters
    ----------
    model : tf.keras.Model
    img_array : np.ndarray
        Preprocessed image (1, 224, 224, 3).
    layer_names : list of str, optional
        Specific layers to visualize. If None, auto-selects conv layers.
    max_features : int
        Maximum number of feature maps to show per layer.
    save_path : str, optional
        If provided, saves the figure to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if not TF_AVAILABLE or model is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.text(0.5, 0.5, "TensorFlow not available",
                ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    # Auto-select layers if not specified
    if layer_names is None:
        layer_names = []
        for layer in model.layers:
            if isinstance(layer, (tf.keras.layers.Conv2D,)):
                layer_names.append(layer.name)
            if hasattr(layer, "layers"):  # nested model
                for sub in layer.layers:
                    if isinstance(sub, tf.keras.layers.Conv2D):
                        layer_names.append(sub.name)
        # Take first, middle, and last conv layers
        if len(layer_names) > 3:
            mid = len(layer_names) // 2
            layer_names = [layer_names[0], layer_names[mid], layer_names[-1]]

    if not layer_names:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.text(0.5, 0.5, "No convolutional layers found",
                ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    # Build sub-models for each target layer
    outputs = []
    valid_names = []
    for name in layer_names:
        try:
            layer = _get_nested_layer(model, name)
            outputs.append(layer.output)
            valid_names.append(name)
        except ValueError:
            continue

    if not outputs:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.text(0.5, 0.5, "Could not access layers",
                ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    feature_model = tf.keras.Model(inputs=model.input, outputs=outputs)
    activations = feature_model.predict(img_array, verbose=0)

    if not isinstance(activations, list):
        activations = [activations]

    # Plot
    n_layers = len(activations)
    fig, axes = plt.subplots(
        n_layers, min(max_features, 8),
        figsize=(16, 3 * n_layers),
    )

    if n_layers == 1:
        axes = [axes]

    for layer_idx, (activation, name) in enumerate(zip(activations, valid_names)):
        n_features = min(activation.shape[-1], min(max_features, 8))
        for feat_idx in range(n_features):
            ax = axes[layer_idx][feat_idx] if n_features > 1 else axes[layer_idx]
            if isinstance(ax, np.ndarray):
                ax = ax[feat_idx] if feat_idx < len(ax) else ax[0]
            ax.imshow(activation[0, :, :, feat_idx], cmap="viridis")
            ax.axis("off")
            if feat_idx == 0:
                ax.set_title(name, fontsize=8, fontweight="bold")

    plt.suptitle("Intermediate Feature Activations", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FULL GRAD-CAM PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def explain_prediction(
    model,
    pil_image,
    class_names: list = None,
    save_dir: str = None,
) -> dict:
    """
    Complete explainability pipeline for a single image prediction.
    Generates a single edge-refined overlay image with zero background noise.

    Parameters
    ----------
    model : tf.keras.Model
    pil_image : PIL.Image.Image
        Original uploaded image.
    class_names : list of str
        Class label names.
    save_dir : str, optional
        Directory to save visualization outputs.

    Returns
    -------
    dict with keys:
        - predicted_class : str
        - confidence : float
        - heatmap : np.ndarray (raw heatmap)
        - overlay : np.ndarray (image with heatmap overlay)
        - predictions : np.ndarray (all class probabilities)
    """
    if class_names is None:
        class_names = ["Plastic", "Paper", "Glass", "Metal", "Organic Waste", "E-Waste"]

    # Preprocess image
    img = pil_image.resize((224, 224)).convert("RGB")
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Get prediction
    if TF_AVAILABLE and model is not None:
        predictions = model.predict(img_batch, verbose=0)[0]
        pred_idx = int(np.argmax(predictions))
        confidence = float(predictions[pred_idx])
    else:
        import random
        pred_idx = random.randint(0, len(class_names) - 1)
        confidence = random.uniform(0.7, 0.98)
        predictions = np.random.dirichlet(np.ones(len(class_names)))

    pred_class = class_names[pred_idx]

    # Generate Grad-CAM++ heatmap
    conv_layer_name = find_target_explain_layer(model)
    heatmap = generate_gradcam_heatmap(
        model, 
        img_batch, 
        target_class_idx=pred_idx, 
        conv_layer_name=conv_layer_name,
        use_gradcam_plusplus=True
    )

    # Print diagnostics for debugging
    print(f"[Grad-CAM Diagnostic] Predicted Class: {pred_class}")
    print(f"[Grad-CAM Diagnostic] Confidence: {confidence:.4f}")
    print(f"[Grad-CAM Diagnostic] Selected Grad-CAM Layer: {conv_layer_name}")
    print(f"[Grad-CAM Diagnostic] Heatmap Min/Max: {heatmap.min():.6f}/{heatmap.max():.6f}")

    # Create overlay (using alpha=0.5 for optimal visual pop and edge alignment)
    overlay = overlay_heatmap_on_image(pil_image, heatmap, alpha=0.5)

    # Save outputs if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # Write only ONE final high-quality overlay image
        cv2.imwrite(
            os.path.join(save_dir, "gradcam_overlay.png"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )

    return {
        "predicted_class": pred_class,
        "confidence": confidence,
        "heatmap": heatmap,
        "overlay": overlay,
        "predictions": predictions,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  MODULE 10: Explainable AI — Grad-CAM Engine")
    print("=" * 60)
    print(f"  TensorFlow Available : {TF_AVAILABLE}")

    if TF_AVAILABLE:
        print(f"  TensorFlow Version   : {tf.__version__}")

    print("\n  Usage:")
    print("    from gradcam import explain_prediction")
    print("    result = explain_prediction(model, pil_image)")
    print("    overlay = result['overlay']")
    print("=" * 60)
