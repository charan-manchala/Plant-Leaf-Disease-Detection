import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = r"C:\Users\chara\OneDrive\Desktop\plant_disease_project"
MODEL_PATH = os.path.join(BASE_DIR, "models", "disease_model.h5")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

model = tf.keras.models.load_model(MODEL_PATH)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_labels))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))