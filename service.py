import bentoml
import numpy as np
import torch
from bentoml.io import JSON, Image
from PIL import Image as PILImage
from torchvision import transforms

# Load the runner
pneumonia_runner = bentoml.pytorch.get("pneumonia_classifier_model:latest").to_runner()

# Create the service
svc = bentoml.Service("pneumonia_classifier_service", runners=[pneumonia_runner])

# Define the inference transform (standard for your model)
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@svc.api(input=Image(), output=JSON())
async def predict(input_img: PILImage.Image) -> dict:
    # Preprocess
    img_tensor = inference_transform(input_img).unsqueeze(0)
    
    # Run inference
    output = await pneumonia_runner.run_async(img_tensor)
    
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
