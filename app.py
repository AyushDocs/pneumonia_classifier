import os
import random

import uvicorn
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Pneumonia Classifier")

# Setup templates and static files
templates = Jinja2Templates(directory="template")

# Create static directory if not exists for sample images
if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Dummy sample images for now
    samples = [
        {"id": "sample1", "url": "/static/sample1.jpg", "label": "Normal"},
        {"id": "sample2", "url": "/static/sample2.jpg", "label": "Pneumonia"}
    ]
    return templates.TemplateResponse("index.html", {"request": request, "samples": samples})

@app.post("/predict")
async def predict(file: UploadFile = File(None), sample_id: str = None):
    # Mock prediction logic: Random selection
    prediction = random.choice(["Normal", "Pneumonia"])
    confidence = random.uniform(0.85, 0.99)

    return {
        "status": "success",
        "prediction": prediction,
        "confidence": f"{confidence * 100:.2f}%",
        "message": f"Our model identifies this image as {prediction} with {confidence * 100:.1f}% confidence."
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
