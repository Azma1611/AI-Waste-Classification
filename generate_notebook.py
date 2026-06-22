import nbformat as nbf

nb = nbf.v4.new_notebook()

# Markdown and code cells for the notebook
cells = [
    nbf.v4.new_markdown_cell("# Assignment 3: AI-Powered Smart Waste Classification\n\n## Phase 1: Data Preprocessing, EDA, and Model Training\n\nThis notebook covers Data Collection, Exploratory Data Analysis (EDA), Image Preprocessing, MobileNetV2 Training, Evaluation, and Explainable AI (Grad-CAM)."),
    
    nbf.v4.new_code_cell("""import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import warnings
warnings.filterwarnings('ignore')"""),
    
    nbf.v4.new_markdown_cell("### 1. Data Collection & Preprocessing\n\nConfigure the paths to the dataset and apply data augmentation (rotation, flip, zoom, brightness)."),
    
    nbf.v4.new_code_cell("""DATASET_DIR_TRAIN = 'dataset/Training Set'
DATASET_DIR_TEST = 'dataset/Test Set'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data Augmentation & Normalization
datagen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

datagen_test = ImageDataGenerator(rescale=1./255)

if os.path.exists(DATASET_DIR_TRAIN) and os.path.exists(DATASET_DIR_TEST):
    train_generator = datagen_train.flow_from_directory(
        DATASET_DIR_TRAIN,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    val_generator = datagen_test.flow_from_directory(
        DATASET_DIR_TEST,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    CLASSES = list(train_generator.class_indices.keys())
    print("Classes:", CLASSES)
else:
    print(f"Dataset directories not found. Please run the download script and ensure 'Training Set' and 'Test Set' exist inside 'dataset/'.")"""),
    
    nbf.v4.new_markdown_cell("### 2. Exploratory Data Analysis (EDA)\n\nAnalyze class distribution and visualize sample images."),
    
    nbf.v4.new_code_cell("""if os.path.exists(DATASET_DIR_TRAIN):
    # Class Distribution Bar Chart
    class_counts = {c: len(os.listdir(os.path.join(DATASET_DIR_TRAIN, c))) for c in CLASSES}
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
    plt.title('Waste Category Distribution')
    plt.ylabel('Number of Images')
    plt.xlabel('Category')
    plt.show()
    
    # Visualize Sample Images
    imgs, labels = next(train_generator)
    plt.figure(figsize=(12, 12))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(imgs[i])
        plt.title(CLASSES[np.argmax(labels[i])])
        plt.axis('off')
    plt.show()"""),
    
    nbf.v4.new_markdown_cell("### 3. Model Training (MobileNetV2)\n\nWe build a transfer learning model using MobileNetV2 to ensure fast and accurate classification on edge devices or web servers."),
    
    nbf.v4.new_code_cell("""# Build Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
base_model.trainable = False  # Freeze base layers initially

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(len(CLASSES) if os.path.exists(DATASET_DIR_TRAIN) else 9, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()"""),
    
    nbf.v4.new_code_cell("""# Train Model (Uncomment to run training)
# EPOCHS = 10
# if os.path.exists(DATASET_DIR_TRAIN):
#     history = model.fit(
#         train_generator,
#         validation_data=val_generator,
#         epochs=EPOCHS
#     )
#     
#     # Save Model
#     os.makedirs('model', exist_ok=True)
#     model.save('model/waste_model.keras')
#     print("Model saved to model/waste_model.keras")"""),

    nbf.v4.new_markdown_cell("### 4. Evaluation\n\nGenerate evaluation metrics (Accuracy, Precision, Recall, F1) and Confusion Matrix."),
    
    nbf.v4.new_code_cell("""# if os.path.exists(DATASET_DIR_TRAIN):
#     Y_pred = model.predict(val_generator)
#     y_pred = np.argmax(Y_pred, axis=1)
#     
#     print('Confusion Matrix')
#     cm = confusion_matrix(val_generator.classes, y_pred)
#     sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES)
#     plt.show()
#     
#     print('Classification Report')
#     print(classification_report(val_generator.classes, y_pred, target_names=CLASSES))"""),

    nbf.v4.new_markdown_cell("### 5. Explainable AI (Grad-CAM)\n\nVisualizing what the model focuses on when making a prediction."),
    
    nbf.v4.new_code_cell("""def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Example usage (Uncomment when you have a trained model and test image)
# img_path = 'test_image.jpg' # Replace with actual image path
# img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
# img_array = tf.keras.preprocessing.image.img_to_array(img)
# img_array = np.expand_dims(img_array, axis=0)
# heatmap = make_gradcam_heatmap(img_array, model, 'out_relu')
# plt.matshow(heatmap)
# plt.show()""")
]

nb['cells'] = cells
with open('c:\\Main\\new one\\train_model.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook 'train_model.ipynb' generated successfully.")
