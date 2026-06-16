import os

import torch
from huggingface_hub import hf_hub_download
from pydantic_settings import BaseSettings
from torchvision import transforms

HF_REPO = "24f2004275/pneumonia_classifier"
DEFAULT_MODEL = "pneumonia_classifier_cnn_uza7heywpgthvahb.pt"


def _get_model_path(model_filename: str = DEFAULT_MODEL) -> str:
    """Download model from HF Hub (cached after first download)."""
    return hf_hub_download(repo_id=HF_REPO, filename=model_filename)


class Settings(BaseSettings):
    # Model Configuration — resolved at runtime via HF Hub
    PT_MODEL_PATH: str = ""
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

# Lazy model path — downloaded on first access
_model_path = None


def get_model_path() -> str:
    global _model_path
    if _model_path is None:
        _model_path = _get_model_path()
    return _model_path


# Fixed Transforms (Not strictly config, but fits here for inference pipeline)
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
