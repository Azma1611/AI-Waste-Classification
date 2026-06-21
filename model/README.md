# Model Directory

Place your trained MobileNetV2 model here as `waste_model.h5`.

## Expected Model Specification

- **Architecture:** MobileNetV2 (transfer learning)
- **Input Shape:** (224, 224, 3)
- **Output:** 4 classes with softmax activation
- **Class Order:** ["Hazardous", "Non-Recyclable", "Organic", "Recyclable"]
- **Format:** Keras HDF5 (.h5)

## Without a Model

If no `waste_model.h5` file is present, the backend runs in **demo mode**
with simulated predictions — the frontend and API still work perfectly
for demonstration purposes.
