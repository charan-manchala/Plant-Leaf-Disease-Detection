import os
import shutil
import random

random.seed(42)

# CHANGE THIS PATH to your extracted PlantVillage folder
SOURCE_DATASET = r"C:\Users\chara\OneDrive\Desktop\archive\plantvillage dataset\color"

# Do not change this unless your project folder is elsewhere
TARGET_DATASET = r"C:\Users\chara\OneDrive\Desktop\plant_disease_project\dataset"

selected_classes = {
    "Tomato___Early_blight": "Tomato_Early_blight",
    "Tomato___Late_blight": "Tomato_Late_blight",
    "Tomato___healthy": "Tomato_healthy",
    "Potato___Early_blight": "Potato_Early_blight",
    "Potato___Late_blight": "Potato_Late_blight",
    "Potato___healthy": "Potato_healthy",
    "Pepper_bell___Bacterial_spot": "Pepper_Bacterial_spot",
    "Pepper,_bell___healthy": "Pepper_healthy",
}

for split in ["train", "val", "test"]:
    for class_name in selected_classes.values():
        os.makedirs(os.path.join(TARGET_DATASET, split, class_name), exist_ok=True)

for old_name, new_name in selected_classes.items():
    source_folder = os.path.join(SOURCE_DATASET, old_name)

    if not os.path.exists(source_folder):
        print(f"Missing folder: {source_folder}")
        continue

    images = [
        img for img in os.listdir(source_folder)
        if img.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    total = len(images)
    train_end = int(total * 0.70)
    val_end = int(total * 0.85)

    train_files = images[:train_end]
    val_files = images[train_end:val_end]
    test_files = images[val_end:]

    for file in train_files:
        shutil.copy2(
            os.path.join(source_folder, file),
            os.path.join(TARGET_DATASET, "train", new_name, file)
        )

    for file in val_files:
        shutil.copy2(
            os.path.join(source_folder, file),
            os.path.join(TARGET_DATASET, "val", new_name, file)
        )

    for file in test_files:
        shutil.copy2(
            os.path.join(source_folder, file),
            os.path.join(TARGET_DATASET, "test", new_name, file)
        )

    print(f"Done: {new_name}")

print("Dataset split completed successfully.")