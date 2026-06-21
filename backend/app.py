"""
AI-Powered Smart Waste Classification & Recycling Recommendation System
========================================================================
Flask backend that loads a trained MobileNetV2 model and classifies
uploaded waste images into four categories: Hazardous, Recyclable,
Non-Recyclable, and Organic — with recycling recommendations.

Author  : EcoScan AI Team
Version : 1.0.0
"""

import os
import io
import logging
from datetime import datetime

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from flask import Flask, request, jsonify, send_from_directory
# pyrefly: ignore [missing-import]
from flask_cors import CORS
# pyrefly: ignore [missing-import]
from PIL import Image

# ── TensorFlow import with GPU memory growth ─────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF info logs

import tensorflow as tf
from tensorflow.keras.models import load_model
# pyrefly: ignore [missing-import]
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "waste_model.h5")
FRONTEND_DIR = os.path.join(PROJECT_DIR, "frontend")

# Image settings
IMG_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "gif"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

# Waste classification classes (must match model training order)
WASTE_CLASSES = ["Hazardous", "Non-Recyclable", "Organic", "Recyclable"]

# Recycling recommendations for each waste class
RECYCLING_RECOMMENDATIONS = {
    "Hazardous": {
        "recommendation": "Dispose at hazardous waste collection center.",
        "icon": "☣️",
        "color": "#ef233c",
        "bin": "Hazardous Waste",
        "details": [
            "Store in original containers with labels intact.",
            "Keep away from children, pets, and heat sources.",
            "Locate your nearest hazardous waste collection facility.",
            "Transport carefully in sealed containers in a ventilated vehicle.",
            "NEVER place hazardous waste in regular trash or recycling.",
        ],
        "donts": [
            "Don't pour chemicals, paint, or oil down drains.",
            "Don't mix different hazardous materials together.",
            "Don't burn hazardous waste — releases toxic fumes.",
        ],
        "impact": {
            "decomposition": "Varies (toxic indefinitely)",
            "fact": "Proper hazardous waste disposal prevents soil and groundwater contamination.",
        },
    },
    "Recyclable": {
        "recommendation": "Place in recycling bin.",
        "icon": "♻️",
        "color": "#00b4d8",
        "bin": "Recycling Bin",
        "details": [
            "Rinse out any food residue with water.",
            "Check for recycling symbols and resin codes.",
            "Remove caps and lids if required by your facility.",
            "Flatten bottles and containers to save space.",
            "Place in your curbside recycling bin.",
        ],
        "donts": [
            "Don't bag recyclables in plastic bags.",
            "Don't include items smaller than a credit card.",
            "Don't 'wish-cycle' — putting non-recyclables contaminates the stream.",
        ],
        "impact": {
            "decomposition": "200 – 1,000,000 years (if not recycled)",
            "fact": "Recycling one aluminum can saves enough energy to run a TV for 3 hours.",
        },
    },
    "Non-Recyclable": {
        "recommendation": "Dispose in general waste bin.",
        "icon": "🗑️",
        "color": "#6c757d",
        "bin": "General Waste Bin",
        "details": [
            "Check if the item can be donated or repurposed first.",
            "Look for specialized recycling programs (textiles, etc.).",
            "Break down large items for easier disposal.",
            "Place in your regular waste bin if no recycling option exists.",
            "Contact your local waste authority for guidance on unusual items.",
        ],
        "donts": [
            "Don't dump large items illegally — use scheduled bulk pickup.",
            "Don't mix waste types — contamination reduces recycling efficiency.",
            "Don't include hazardous materials in general waste.",
        ],
        "impact": {
            "decomposition": "Varies by material",
            "fact": "Reducing waste at the source is the single most effective environmental action.",
        },
    },
    "Organic": {
        "recommendation": "Compost or biodegradable waste bin.",
        "icon": "🌿",
        "color": "#00d4aa",
        "bin": "Compost / Green Bin",
        "details": [
            "Collect food scraps in a kitchen compost bin or bag.",
            "Include fruit/vegetable peels, coffee grounds, and eggshells.",
            "Add yard waste like leaves, grass clippings, and small branches.",
            "Turn or aerate your compost regularly if home composting.",
            "Use municipal green bin if available for curbside composting.",
        ],
        "donts": [
            "Don't compost meat, dairy, or oily foods in home compost.",
            "Don't add diseased plants or treated wood.",
            "Don't include pet waste in food compost.",
        ],
        "impact": {
            "decomposition": "2 weeks – 6 months",
            "fact": "Composting reduces methane by keeping organics out of landfills.",
        },
    },
}

# ═══════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# FLASK APP INITIALIZATION
# ═══════════════════════════════════════════════════════════════════

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app)

# ═══════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════

model = None


def load_waste_model():
    """Load the trained MobileNetV2 model from disk."""
    global model
    if model is not None:
        return model

    if not os.path.exists(MODEL_PATH):
        logger.warning(
            f"Model file not found at {MODEL_PATH}. "
            "The /predict endpoint will use simulated predictions. "
            "Place your trained waste_model.h5 in the model/ directory."
        )
        return None

    try:
        logger.info(f"Loading model from {MODEL_PATH}...")
        model = load_model(MODEL_PATH)
        logger.info("✅ Model loaded successfully!")
        logger.info(f"   Input shape : {model.input_shape}")
        logger.info(f"   Output shape: {model.output_shape}")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════


def allowed_file(filename):
    """Check if the uploaded file has a valid image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_bytes):
    """
    Preprocess an image for MobileNetV2 prediction.
    - Open with PIL
    - Convert to RGB
    - Resize to 224×224
    - Convert to numpy array
    - Apply MobileNetV2 preprocessing (scale to [-1, 1])
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise ValueError("Cannot open image. File may be corrupted or not a valid image.")

    # Convert to RGB (handles RGBA, grayscale, palette, etc.)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize to 224×224 with high-quality resampling
    image = image.resize(IMG_SIZE, Image.LANCZOS)

    # Convert to numpy array and preprocess
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # MobileNetV2 preprocessing

    return img_array


def simulate_prediction():
    """
    Generate a simulated prediction when the model is not available.
    Used for demo/testing purposes so the frontend still works.
    """
    import random

    idx = random.randint(0, len(WASTE_CLASSES) - 1)
    confidence = round(random.uniform(0.70, 0.98), 4)

    # Generate realistic confidence scores for all classes
    remaining = 1.0 - confidence
    all_confidences = []
    for i in range(len(WASTE_CLASSES)):
        if i == idx:
            all_confidences.append(confidence)
        else:
            share = remaining / (len(WASTE_CLASSES) - 1)
            jitter = random.uniform(-share * 0.5, share * 0.5)
            all_confidences.append(max(0, share + jitter))

    # Normalize
    total = sum(all_confidences)
    all_confidences = [c / total for c in all_confidences]
    all_confidences[idx] = max(all_confidences)

    return idx, all_confidences


# ═══════════════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════════════


@app.route("/")
def serve_frontend():
    """Serve the frontend index.html."""
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:path>")
def serve_static(path):
    """Serve static frontend files (CSS, JS, images)."""
    return send_from_directory(FRONTEND_DIR, path)


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    ─────────────
    Accept an image upload, preprocess it, run prediction through
    the MobileNetV2 model, and return the predicted waste class,
    confidence score, and recycling recommendation.

    Request:
        - Content-Type: multipart/form-data
        - Field "image": image file (JPG, PNG, WebP, BMP, GIF)

    Response (JSON):
        {
            "success": true,
            "prediction": {
                "class": "Recyclable",
                "confidence": 0.9432,
                "confidence_percent": "94.32%",
                "all_predictions": { ... }
            },
            "recommendation": { ... },
            "timestamp": "2024-01-15T10:30:00"
        }
    """
    # ── Validate request ──────────────────────────────────────────
    if "image" not in request.files:
        return (
            jsonify(
                {
                    "success": False,
                    "error": "No image file provided. Please upload an image using the 'image' field.",
                }
            ),
            400,
        )

    file = request.files["image"]

    if file.filename == "":
        return (
            jsonify({"success": False, "error": "No file selected. Please choose an image to upload."}),
            400,
        )

    if not allowed_file(file.filename):
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}",
                }
            ),
            400,
        )

    # ── Read and preprocess image ─────────────────────────────────
    try:
        image_bytes = file.read()
        if len(image_bytes) == 0:
            return jsonify({"success": False, "error": "Uploaded file is empty."}), 400

        img_array = preprocess_image(image_bytes)
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        return (
            jsonify({"success": False, "error": "Failed to process image. Please try a different image."}),
            400,
        )

    # ── Run prediction ────────────────────────────────────────────
    try:
        if model is not None:
            # Real model prediction
            predictions = model.predict(img_array, verbose=0)
            predicted_idx = int(np.argmax(predictions[0]))
            all_confidences = predictions[0].tolist()
        else:
            # Simulated prediction (model not available)
            logger.info("Using simulated prediction (model not loaded)")
            predicted_idx, all_confidences = simulate_prediction()

        predicted_class = WASTE_CLASSES[predicted_idx]
        confidence = float(all_confidences[predicted_idx])

        # Build all predictions dict
        all_predictions = {}
        for i, cls in enumerate(WASTE_CLASSES):
            all_predictions[cls] = {
                "confidence": round(float(all_confidences[i]), 4),
                "confidence_percent": f"{float(all_confidences[i]) * 100:.2f}%",
            }

        # Get recycling recommendation
        recommendation = RECYCLING_RECOMMENDATIONS.get(
            predicted_class, RECYCLING_RECOMMENDATIONS["Non-Recyclable"]
        )

        response = {
            "success": True,
            "prediction": {
                "class": predicted_class,
                "confidence": round(confidence, 4),
                "confidence_percent": f"{confidence * 100:.2f}%",
                "all_predictions": all_predictions,
            },
            "recommendation": recommendation,
            "timestamp": datetime.now().isoformat(),
            "model_loaded": model is not None,
        }

        logger.info(
            f"🔍 Prediction: {predicted_class} ({confidence * 100:.1f}%) "
            f"{'[MODEL]' if model is not None else '[SIMULATED]'}"
        )

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return (
            jsonify({"success": False, "error": "Prediction failed. Please try again."}),
            500,
        )


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": model is not None,
            "model_path": MODEL_PATH,
            "classes": WASTE_CLASSES,
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/classes", methods=["GET"])
def get_classes():
    """Return the list of waste classes and their recommendations."""
    return jsonify(
        {
            "classes": WASTE_CLASSES,
            "recommendations": RECYCLING_RECOMMENDATIONS,
        }
    )


# ═══════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ═══════════════════════════════════════════════════════════════════


@app.errorhandler(413)
def too_large(e):
    return (
        jsonify(
            {
                "success": False,
                "error": f"File too large. Maximum size is {MAX_CONTENT_LENGTH // (1024 * 1024)} MB.",
            }
        ),
        413,
    )


@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "Resource not found."}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"success": False, "error": "Internal server error. Please try again."}), 500


# ═══════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  EcoScan AI — Waste Classification Server")
    print("=" * 60)
    print(f"  Model path : {MODEL_PATH}")
    print(f"  Frontend   : {FRONTEND_DIR}")
    print(f"  Classes    : {', '.join(WASTE_CLASSES)}")
    print("=" * 60 + "\n")

    # Load the model at startup
    load_waste_model()

    # Start Flask development server
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=True,
    )
