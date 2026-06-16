# PnuemoCheck AI: Pneumonia Classifier
[![Live Demo](https://img.shields.io/badge/Demo-Live_Now-brightgreen?style=for-the-badge&logo=github)](https://ayushdocs.github.io/pneumonia_classifier/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit)](https://pneumonia-classifie.streamlit.app/)

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

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 98% |
| **Pneumonia Precision** | 97% |
| **Pneumonia Recall** | 100% |
| **Normal Precision** | 100% |
| **Normal Recall** | 96% |
| **Macro F1-Score** | 0.98 |
| **ROC AUC** | 1.00 |

> Evaluated on 57 test samples with optimized threshold (0.1) for high sensitivity.

### Cross-Validation (5-Fold Stratified)

| Fold | Accuracy | F1-Score |
|------|----------|----------|
| 1 | 92.86% | 0.927 |
| 2 | 97.62% | 0.976 |
| 3 | 92.86% | 0.933 |
| 4 | 95.24% | 0.950 |
| 5 | 97.62% | 0.977 |
| **Mean** | **95.24%** | **0.953** |

### Quantization (INT8)

| | FP32 | INT8 | Improvement |
|--|------|------|-------------|
| Size | 243 KB | 52 KB | **4.64x** smaller |
| Latency | 108 ms | 27 ms | **3.99x** faster |

## 📂 Project Structure

- `notebooks/`: Research pipeline (01-10)
- `pneumonia_classifier/`: Core ML package and architecture
- `static/`: Frontend assets (styles, interactions, samples)
- `app.py`: FastAPI backend with integrated XAI

---
*Disclaimer: This is for educational/DA purposes and not for actual clinical diagnosis.*
