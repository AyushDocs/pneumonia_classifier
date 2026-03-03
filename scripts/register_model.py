import os

import bentoml
import torch

from pneumonia_classifier.config import TRANSFORM, config

# Define constants
MODEL_NAME = "pneumonia_classifier_model"
TRANSFORMS_KEY = "pneumonia_classifier_train_transforms"

def register():
    model_path = config.PT_MODEL_PATH

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    try:
        # The synced model is a state_dict or whole object.
        # For BentoML native persistence, we prefer the whole object if available
        # or we instantiate and load state_dict.
        from pneumonia_classifier.ml.model.arch import Net
        model = Net()
        loaded_data = torch.load(model_path, map_location='cpu', weights_only=False)

        if isinstance(loaded_data, dict):
            model.load_state_dict(loaded_data, strict=False)
        else:
            model = loaded_data

        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print(f"Registering model '{MODEL_NAME}' in BentoML...")
    try:
        bentoml.pytorch.save_model(
            name=MODEL_NAME,
            model=model,
            custom_objects={
                TRANSFORMS_KEY: TRANSFORM
            }
        )
        print("Registration successful!")
    except Exception as e:
        print(f"Failed to register model: {e}")

if __name__ == "__main__":
    register()
if __name__ == "__main__":
    register()
