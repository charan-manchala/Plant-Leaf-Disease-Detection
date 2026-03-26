import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf

BASE_DIR = r"C:\Users\chara\OneDrive\Desktop\plant_disease_project"
MODEL_PATH = os.path.join(BASE_DIR, "models", "disease_model.h5")
CLASS_PATH = os.path.join(BASE_DIR, "models", "class_indices.json")
CURE_PATH = os.path.join(BASE_DIR, "data", "cure.json")
PLANT_TYPE_PATH = os.path.join(BASE_DIR, "data", "plant_type.json")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r") as f:
    class_indices = json.load(f)

with open(CURE_PATH, "r") as f:
    cure_data = json.load(f)

with open(PLANT_TYPE_PATH, "r") as f:
    plant_type_data = json.load(f)

index_to_class = {v: k for k, v in class_indices.items()}

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def extract_plant_name(class_name):
    return class_name.split("_")[0]

def extract_status(class_name):
    return "Healthy" if "healthy" in class_name.lower() else "Diseased"

def format_disease_name(class_name):
    if "healthy" in class_name.lower():
        return "No Disease"
    parts = class_name.split("_")
    return " ".join(parts[1:])

def predict_leaf(image):
    processed = preprocess_image(image)
    prediction = model.predict(processed, verbose=0)
    predicted_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction) * 100)

    predicted_class = index_to_class[predicted_index]
    plant_name = extract_plant_name(predicted_class)
    plant_type = plant_type_data.get(plant_name, "Unknown")
    leaf_status = extract_status(predicted_class)
    disease_name = format_disease_name(predicted_class)
    cure = cure_data.get(predicted_class, "No cure information available.")

    if confidence >= 75:
        is_supported = True
        warning_level = "none"
    elif confidence >= 60:
        is_supported = True
        warning_level = "medium"
    else:
        is_supported = False
        warning_level = "high"

    return {
        "is_supported": is_supported,
        "warning_level": warning_level,
        "plant_name": plant_name if is_supported else "Unsupported",
        "plant_type": plant_type if is_supported else "Unknown",
        "leaf_status": leaf_status if is_supported else "Unknown",
        "disease_name": disease_name if is_supported else "Unknown",
        "confidence": round(confidence, 2),
        "cure": cure if is_supported else "Please upload a clearer image from a supported plant."
    }