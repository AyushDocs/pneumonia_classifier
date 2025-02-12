import os
from dataclasses import dataclass

from torch import device

from pneumonia_classifier.constant.training_pipeline import TIMESTAMP,BUCKET_NAME,S3_DATA_FOLDER,ARTIFACT_DIR,TRAIN_TRANSFORMS_FILE,TEST_TRANSFORMS_FILE,TRAINED_MODEL_NAME,BRIGHTNESS,CONTRAST,SATURATION,HUE,RESIZE,CENTERCROP,RANDOMROTATION,NORMALIZE_LIST_1,NORMALIZE_LIST_2,BATCH_SIZE,SHUFFLE,PIN_MEMORY,EPOCH,STEP_SIZE,GAMMA,DEVICE,BENTOML_MODEL_NAME,BENTOML_SERVICE_NAME,TRAIN_TRANSFORMS_KEY,BENTOML_ECR_URI


@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.s3_data_folder: str = S3_DATA_FOLDER

        self.bucket_name: str = BUCKET_NAME

        self.artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)

        self.data_path: str = os.path.join(
            self.artifact_dir, "data_ingestion", self.s3_data_folder
        )

        self.train_data_path: str = os.path.join(self.data_path, "train")
        os.makedirs(self.train_data_path, exist_ok=True)

        self.test_data_path: str = os.path.join(self.data_path, "test")
        os.makedirs(self.test_data_path, exist_ok=True)

        os.makedirs(os.path.join(self.train_data_path,'NORMAL'),exist_ok=True)
        os.makedirs(os.path.join(self.train_data_path,'PNEUMONIA'),exist_ok=True)
        os.makedirs(os.path.join(self.test_data_path,'NORMAL'),exist_ok=True)
        os.makedirs(os.path.join(self.test_data_path,'PNEUMONIA'),exist_ok=True)



@dataclass
class DataTransformationConfig:
    def __init__(self):
        self.color_jitter_transforms: dict = {
            "brightness": BRIGHTNESS,
            "contrast": CONTRAST,
            "saturation": SATURATION,
            "hue": HUE,
        }

        self.RESIZE: int = RESIZE

        self.CENTERCROP: int = CENTERCROP

        self.RANDOMROTATION: int = RANDOMROTATION

        self.normalize_transforms: dict = {
            "mean": NORMALIZE_LIST_1,
            "std": NORMALIZE_LIST_2,
        }

        self.data_loader_params: dict = {
            "batch_size": BATCH_SIZE,
            "shuffle": SHUFFLE,
            "pin_memory": PIN_MEMORY,
        }

        self.artifact_dir: str = os.path.join(
            ARTIFACT_DIR, TIMESTAMP, "data_transformation"
        )

        self.train_transforms_file: str = os.path.join(
            self.artifact_dir, TRAIN_TRANSFORMS_FILE
        )

        self.test_transforms_file: str = os.path.join(
            self.artifact_dir, TEST_TRANSFORMS_FILE
        )




@dataclass
class ModelTrainerConfig:
    def __init__(self):
        self.artifact_dir: int = os.path.join(ARTIFACT_DIR, TIMESTAMP, "model_training")

        self.trained_bentoml_model_name: str = "xray_model"

        self.trained_model_path: int = os.path.join(
            self.artifact_dir, TRAINED_MODEL_NAME
        )

        self.train_transforms_key: str = TRAIN_TRANSFORMS_KEY

        self.epochs: int = EPOCH

        self.optimizer_params: dict = {"lr": 0.01, "momentum": 0.8}

        self.scheduler_params: dict = {"step_size": STEP_SIZE, "gamma": GAMMA}

        self.device: device = DEVICE

@dataclass
class ModelEvaluationConfig:
    def __init__(self):
        self.device: device = DEVICE

        self.test_loss: int = 0

        self.test_accuracy: int = 0

        self.total: int = 0

        self.total_batch: int = 0

        self.optimizer_params: dict = {"lr": 0.01, "momentum": 0.8}

# Model Pusher Configurations
@dataclass
class ModelPusherConfig:
    def __init__(self):
        self.bentoml_model_name: str = BENTOML_MODEL_NAME

        self.bentoml_service_name: str = BENTOML_SERVICE_NAME

        self.train_transforms_key: str = TRAIN_TRANSFORMS_KEY

        self.bentoml_ecr_image: str = BENTOML_ECR_URI
