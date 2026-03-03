import base64
import io
import json
import logging
import os
import time

import cv2
import numpy as np
import redis
import torch
from celery import Celery
from PIL import Image

from pneumonia_classifier.config import TRANSFORM, config
from pneumonia_classifier.ml.explainability import get_medical_heatmap
from pneumonia_classifier.ml.model.arch import Net
from pneumonia_classifier.utils.database import save_drift_log, save_prediction

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis & Celery Setup
REDIS_URL = config.REDIS_URL
celery_app = Celery("inference_tasks", broker=REDIS_URL, backend=REDIS_URL)

# Global model placeholders
pt_model = None

@celery_app.task(name="process_inference")
def process_inference(job_id: str, b64_image: str, patient_id: str, requester_id: str, requester_ip: str):
    """Background worker that runs the PyTorch model and updates the Redis result store."""
    try:
        global pt_model
        if pt_model is None:
            pt_model = Net()
            loaded_data = torch.load(config.PT_MODEL_PATH, map_location='cpu', weights_only=False)

            # Ensure we get the state_dict
            if not isinstance(loaded_data, dict) and hasattr(loaded_data, "state_dict"):
                state_dict = loaded_data.state_dict()
            else:
                state_dict = loaded_data

            pt_model.load_state_dict(state_dict, strict=False)
            pt_model.eval()
            logger.info(f"Worker loaded PyTorch model from {config.PT_MODEL_PATH}")

        image_bytes = base64.b64decode(b64_image)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = TRANSFORM(image).unsqueeze(0)

        # Analytics: Input Statistics
        try:
            input_data = input_tensor.numpy()
            mean_val = float(np.mean(input_data))
            std_val = float(np.std(input_data))
            save_drift_log(job_id, mean_val, std_val)
        except Exception as drift_e:
            logger.warning(f"Failed to log drift metrics: {drift_e}")

        # Core Inference (PyTorch)
        with torch.no_grad():
            outputs_log_softmax = pt_model(input_tensor)
            probabilities = torch.exp(outputs_log_softmax)

        # Platt Scaling Calibration (Simulated)
        pred_idx = int(probabilities.argmax().item())
        prediction = "Pneumonia" if pred_idx == 1 else "Normal"
        winning_raw_conf = probabilities[0][pred_idx].item()

        # Use raw probability as confidence percent
        calibrated_conf = winning_raw_conf * 100

        logger.info(f"[Job {job_id}] Prediction: {prediction} ({calibrated_conf:.1f}%)")

        # Grad-CAM Explainability (Shadow Model Strategy)
        heatmap_base64 = ""
        heatmap_path = ""
        try:
            # Generate heatmap using standardized utility
            heatmap_img_bgr = get_medical_heatmap(pt_model, input_tensor, image)

            if heatmap_img_bgr is not None:
                # 1. Persistence for offline retrieval
                heatmap_filename = f"data/heatmaps/{patient_id}_{int(time.time())}.png"
                os.makedirs("data/heatmaps", exist_ok=True)
                cv2.imwrite(heatmap_filename, heatmap_img_bgr)
                heatmap_path = heatmap_filename

                # 2. Base64 for instant delivery to Streamlit
                h_img_rgb = cv2.cvtColor(heatmap_img_bgr, cv2.COLOR_BGR2RGB)
                buffered = io.BytesIO()
                Image.fromarray(h_img_rgb).save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                heatmap_base64 = f"data:image/png;base64,{img_str}"

        except Exception as cam_e:
            logger.error(f"Grad-CAM Failure: {cam_e}")

        # Save to SQLite
        save_prediction(patient_id, prediction, f"{calibrated_conf:.1f}%", heatmap_path, requester_id, requester_ip)

        # Final Result Update in Redis
        redis_conn = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        redis_conn.set(job_id, json.dumps({
            "status": "completed",
            "prediction": prediction,
            "confidence": f"{calibrated_conf:.1f}%",
            "heatmap": heatmap_base64,
            "original_image": f"data:image/png;base64,{b64_image}",
            "patient_id": patient_id,
            "message": f"Analysis complete for Patient {patient_id}."
        }), ex=86400)

        return f"Analysis Complete: {prediction} ({calibrated_conf:.1f}%) | Patient: {patient_id}"

    except Exception as e:
        logger.error(f"Worker Error: {e}")
        redis_conn = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        redis_conn.set(job_id, json.dumps({
            "status": "failed",
            "message": str(e)
        }), ex=3600)
