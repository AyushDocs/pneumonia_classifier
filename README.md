# Pneumonia Classifier

A modular deep learning project to classify pneumonia from chest X-ray images using PyTorch and BentoML.

## 🚀 Overview

This project implements a complete machine learning pipeline for medical image classification. It is designed with a modular architecture to handle data ingestion, transformation, model training, evaluation, and deployment.

## 📁 Project Structure

```text
├── pneumonia_classifier/
│   ├── components/       # Modular pipeline components
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluvation.py
│   │   └── model_pusher.py
│   ├── pipeline/         # Training & prediction pipelines
│   ├── ml/               # Model definitions & service logic
│   └── entity/           # Configuration & artifact entities
├── notebooks/            # Experimental development notebooks
├── main.py               # Entry point for training
├── app.py                # Service entry point
└── bentofile.yaml        # BentoML deployment configuration
```

## 🛠️ Tech Stack

- **Deep Learning:** PyTorch, Torchvision
- **Deployment:** BentoML
- **Data Handling:** NumPy, Pandas, Scikit-learn
- **Visualization:** Matplotlib, Seaborn

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AyushDocs/pneumonia_classifier.git
   cd pneumonia_classifier
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Training the Model
To start the end-to-end training pipeline:
```bash
python main.py
```

### Serving the Model
To serve the model using BentoML:
```bash
bentoml serve .
```

## 📝 Components

- **Data Ingestion:** Downloads and prepares clinical X-ray datasets.
- **Data Transformation:** Handles image preprocessing and augmentation.
- **Model Trainer:** Executes deep learning training using PyTorch.
- **Model Evaluation:** Validates model performance against test metrics.
- **Model Pusher:** Deploys the validated model for production use.
