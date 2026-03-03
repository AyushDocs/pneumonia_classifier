# PnuemoCheck AI: Pneumonia Classifier
[![Live Demo](https://img.shields.io/badge/Demo-Live_Now-brightgreen?style=for-the-badge&logo=github)](https://ayushdocs.github.io/pneumonia_classifier/)

Pneumonia classification using deep learning.
An end-to-end medical imaging project using Deep Learning to detect pneumonia from Chest X-rays, featuring explainability and production-ready deployment.

## 🌟 Key Features

- **Explainable AI (XAI)**: Integrated Grad-CAM to visualize model focus areas.
- **Optimization**: Hyperparameter tuning via Optuna (Achieved **1.0 F1-Score**).
- **Experiment Tracking**: Full lifecycle tracking with MLflow (parameters, metrics, and model artifacts).
- **Research Foundation**: Comprehensive notebooks covering Augmentation, Quantization, and XAI.

## 🛠️ Tech Stack

- **Core**: Python 3.10, PyTorch, FastAPI
- **XAI**: OpenCV, Grad-CAM (Standardized in `docs/XAI_WORKFLOW.md`)
- **Ops**: BentoML (Model Store & Serving), MLflow (Experiment Tracking & XAI Explanations)
- **UI**: Vanilla HTML/CSS/JS with a premium, responsive glassmorphism design.

## 🚀 Quick Start

1. **Environment Setup**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run Web UI**:

   ```bash
   python app.py
   ```

   Visit `http://localhost:8000` to upload an X-ray and see the diagnosis + heatmap.

3. **View Experiments**:

   ```bash
   mlflow server --backend-store-uri file:///$(pwd)/notebooks/mlruns --port 5001
   ```

- [Detailed Research Findings](docs/RESEARCH_FINDINGS.md)

## 📂 Project Structure

- `notebooks/`: Research pipeline (01-10)
- `pneumonia_classifier/`: Core ML package and architecture
- `static/`: Frontend assets (styles, interactions, samples)
- `app.py`: FastAPI backend with integrated XAI

---
*Disclaimer: This is for educational/DA purposes and not for actual clinical diagnosis.*
