# 🌿 EcoScan AI — Smart Waste Classification & Recycling Recommendation System

An **AI-powered waste classification system** that uses a **MobileNetV2** deep learning model to classify uploaded waste images into four categories and provide recycling recommendations.

Built with **Python, TensorFlow/Keras, Flask, HTML, CSS, and JavaScript**.

---

## 📋 Features

- 🤖 **MobileNetV2 Deep Learning** — Trained model classifies waste images with high accuracy
- 🏷️ **4 Waste Categories** — Hazardous, Recyclable, Non-Recyclable, Organic
- ♻️ **Recycling Recommendations** — Detailed disposal instructions for each category
- 📊 **Confidence Scores** — Probability breakdown across all classes
- 🖼️ **Drag & Drop Upload** — Modern image upload with preview
- 🎨 **Premium Dark UI** — Glassmorphism design with micro-animations
- 📱 **Fully Responsive** — Works on desktop, tablet, and mobile
- 🔒 **Error Handling** — Validates file types, sizes, and image integrity
- 📜 **Classification History** — Tracks past predictions (localStorage)
- 🌍 **Eco-Tips Ticker** — Rotating environmental awareness facts

---

## 🗂️ Project Structure

```
project/
├── model/
│   └── waste_model.h5          # Trained MobileNetV2 model
├── backend/
│   └── app.py                  # Flask server + prediction API
├── frontend/
│   ├── index.html              # Main web page
│   ├── styles.css              # Premium dark-mode styling
│   └── app.js                  # Frontend logic & API client
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🏷️ Waste Classes & Recommendations

| Class | Icon | Recommendation |
|-------|------|----------------|
| **Hazardous** | ☣️ | Dispose at hazardous waste collection center |
| **Recyclable** | ♻️ | Place in recycling bin |
| **Non-Recyclable** | 🗑️ | Dispose in general waste bin |
| **Organic** | 🌿 | Compost or biodegradable waste bin |

---

## 🚀 Installation & Setup

### Prerequisites

- **Python 3.9+** installed
- **pip** package manager
- (Optional) A trained `waste_model.h5` file

### Step 1: Clone or Download

```bash
cd project
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** TensorFlow installation may take a few minutes. On machines without GPU, TensorFlow will use CPU automatically.

### Step 3: Add Your Model (Optional)

Place your trained MobileNetV2 model in the `model/` directory:

```
model/waste_model.h5
```

> If no model is found, the server runs in **demo mode** with simulated predictions — perfect for testing the UI and API.

### Step 4: Run the Server

```bash
python backend/app.py
```

Or from the backend directory:

```bash
cd backend
python app.py
```

### Step 5: Open in Browser

Navigate to: **http://localhost:5000**

---

## 🔗 API Endpoints

### `POST /predict`

Classify an uploaded waste image.

**Request:**
```
Content-Type: multipart/form-data
Field: "image" (file)
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "class": "Recyclable",
    "confidence": 0.9432,
    "confidence_percent": "94.32%",
    "all_predictions": {
      "Hazardous": { "confidence": 0.0123, "confidence_percent": "1.23%" },
      "Non-Recyclable": { "confidence": 0.0234, "confidence_percent": "2.34%" },
      "Organic": { "confidence": 0.0211, "confidence_percent": "2.11%" },
      "Recyclable": { "confidence": 0.9432, "confidence_percent": "94.32%" }
    }
  },
  "recommendation": {
    "recommendation": "Place in recycling bin.",
    "icon": "♻️",
    "color": "#00b4d8",
    "bin": "Recycling Bin",
    "details": ["..."],
    "donts": ["..."],
    "impact": { "decomposition": "...", "fact": "..." }
  },
  "timestamp": "2024-01-15T10:30:00",
  "model_loaded": true
}
```

### `GET /health`

Health check endpoint.

```json
{
  "status": "healthy",
  "model_loaded": true,
  "classes": ["Hazardous", "Non-Recyclable", "Organic", "Recyclable"],
  "timestamp": "2024-01-15T10:30:00"
}
```

### `GET /classes`

List all waste classes and recommendations.

---

## 🧠 Training Your Own Model

To train a custom MobileNetV2 model for waste classification:

```python
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load pre-trained MobileNetV2 (without top classification layer)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base layers

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dense(4, activation='softmax')(x)  # 4 classes

model = Model(inputs=base_model.input, outputs=x)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train on your dataset
# model.fit(train_data, epochs=10, validation_data=val_data)

# Save the model
model.save('model/waste_model.h5')
```

> **Important:** Ensure your training class order matches: `["Hazardous", "Non-Recyclable", "Organic", "Recyclable"]`

---

## 🛡️ Error Handling

The system handles these error cases:
- ❌ No image file uploaded
- ❌ Invalid file type (non-image)
- ❌ File too large (>16 MB)
- ❌ Corrupted or unreadable images
- ❌ Model prediction failures
- ❌ Server connectivity issues

---

## 🖥️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | HTML5, CSS3, JavaScript (ES6+) |
| **Backend** | Python, Flask, Flask-CORS |
| **AI/ML** | TensorFlow 2.x, Keras, MobileNetV2 |
| **Image Processing** | Pillow (PIL) |
| **Design** | Custom CSS, Glassmorphism, Google Fonts (Inter, Outfit) |

---

## 📝 Notes for Demonstration

1. **Without a model file:** The system automatically runs in demo mode with simulated predictions. This is ideal for demonstrating the frontend, API design, and overall architecture.

2. **With a model file:** Place your trained `waste_model.h5` in the `model/` directory. The server loads it on startup and uses it for real predictions.

3. **API Testing:** You can test the `/predict` endpoint with tools like Postman or cURL:
   ```bash
   curl -X POST -F "image=@test_image.jpg" http://localhost:5000/predict
   ```

---

## 📄 License

This project is created for educational purposes. Feel free to use, modify, and distribute.

---

**Built with ❤️ and AI for a greener planet** 🌍
