import os

import torch
from pydantic_settings import BaseSettings
from torchvision import transforms


class Settings(BaseSettings):
    # Model Configuration
    PT_MODEL_PATH: str = os.path.join(os.getcwd(), "models", "pneumonia_classifier_cnn_uza7heywpgthvahb.pt")
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # DB Configuration
    DB_PATH: str = os.path.join(os.getcwd(), "data", "patient_history.db")
    # Redis Configuration
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    REPORT_TEMP_DIR: str = os.path.join(os.getcwd(), "data", "tmp")

    class Config:
        env_file = ".env"

# Instantiate global config
config = Settings()

# Fixed Transforms (Not strictly config, but fits here for inference pipeline)
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
