import base64
from io import BytesIO

import bentoml
import cv2
import torch
from PIL import Image as PILImage
from torchvision import transforms

from pneumonia_classifier.ml.explainability import get_medical_heatmap
from pneumonia_classifier.ml.model.arch import Net

# Allowlist Net for PyTorch 2.6+ unpickling
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([Net])

@bentoml.service(
    name="pneumonia_classifier_service",
    traffic={"timeout": 60},
)
class svc:
    # Use the recommended BentoModel reference for class attributes
    model_ref = bentoml.models.BentoModel("pneumonia_classifier_model:latest")

    def __init__(self):
        # Load the model directly into the service instance
        self.model = bentoml.pytorch.load_model(self.model_ref,weights_only=False)
        self.model.eval()

        # Define the inference transform
        self.inference_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _pil_to_base64(self, img: PILImage.Image) -> str:
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @bentoml.api
    async def predict(self, input_img: PILImage.Image) -> dict:
        # Preprocess
        img_tensor = self.inference_transform(input_img).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            output = self.model(img_tensor)

        # Postprocess
        probabilities = torch.nn.functional.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()

        label = "PNEUMONIA" if prediction == 1 else "NORMAL"

        return {
            "prediction": label,
            "class_index": prediction,
            "confidence": float(confidence)
        }

    @bentoml.api
    async def explain(self, input_img: PILImage.Image) -> dict:
        # 1. Prediction Path
        img_tensor = self.inference_transform(input_img).unsqueeze(0)

        with torch.no_grad():
            output = self.model(img_tensor)

        probabilities = torch.nn.functional.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
        label = "PNEUMONIA" if prediction == 1 else "NORMAL"

        # 2. XAI Path (Grad-CAM)
        heatmap_img_np = get_medical_heatmap(self.model, img_tensor, input_img)

        if heatmap_img_np is not None:
            # Convert BGR (OpenCV) to RGB for PIL
            heatmap_img_rgb = PILImage.fromarray(cv2.cvtColor(heatmap_img_np, cv2.COLOR_BGR2RGB))
            heatmap_base64 = self._pil_to_base64(heatmap_img_rgb)
        else:
            heatmap_base64 = None

        return {
            "prediction": label,
            "class_index": prediction,
            "confidence": float(confidence),
            "heatmap": heatmap_base64
        }
