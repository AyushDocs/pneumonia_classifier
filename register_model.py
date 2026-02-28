import os
import sys

import bentoml
import joblib
import torch

# Define constants (matching training pipeline)
MODEL_PATH = r"j:\Users\ayush\Desktop\code\pneumonia_classifier\artifacts\02_12_2025_08_52_04\model_training\model.pt"
TRANSFORMS_PATH = r"j:\Users\ayush\Desktop\code\pneumonia_classifier\artifacts\02_12_2025_08_52_04\data_transformation\train_transforms.pkl"
MODEL_NAME = "pneumonia_classifier_model"
TRANSFORMS_KEY = "pneumonia_classifier_train_transforms"

def register():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return
    if not os.path.exists(TRANSFORMS_PATH):
        print(f"Error: Transforms file not found at {TRANSFORMS_PATH}")
        return

    print("Loading model...")
    # Load model (it was saved as a whole object)
    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

    print("Loading transforms...")
    transforms_obj = joblib.load(TRANSFORMS_PATH)

    print(f"Registering model '{MODEL_NAME}' in BentoML...")
    bentoml.pytorch.save_model(
        name=MODEL_NAME,
        model=model,
        custom_objects={
            TRANSFORMS_KEY: transforms_obj
        }
    )
    print("Registration successful!")

if __name__ == "__main__":
    register()
