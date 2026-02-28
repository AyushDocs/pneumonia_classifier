# Research Findings: Pneumonia Classifier

This document summarizes the insights, metrics, and experimental results obtained during the research phase (Notebooks 01-10).

```mermaid
graph LR
    A[Data Prep] --> B[Augmentation]
    B --> C[Optuna Tuning]
    C --> D[MLflow Tracking]
    D --> E[Final Model]
    E --> F[Grad-CAM Analysis]
    F --> G[Quantization]
```

## 1. Data Analysis (EDA)

![Performance Metrics Visualization](images/performance_metrics.png)

- **Dataset**: Chest X-Ray Images (Pneumonia).
- **Balance**: The training set was balanced with 105 images for `NORMAL` and 105 for `PNEUMONIA`.
- **Validation**: Test set contains 57 images (27 Normal, 30 Pneumonia).

## 2. Model Performance

The classifier uses a custom CNN architecture (`Net`) with 9 convolutional blocks and Global Average Pooling.

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 74% |
| **Pneumonia Precision** | 100% |
| **Pneumonia Recall** | 50% |
| **Normal Recall** | 100% |

*Note: The model is highly conservative in predicting Pneumonia, ensuring zero false positives in current tests but missing some positive cases.*

### 📊 Confusion Matrix (Conceptual)

| Actual \ Predicted | NORMAL | PNEUMONIA |
| :--- | :--- | :--- |
| **NORMAL** | 27 (True Negative) | 0 (False Positive) |
| **PNEUMONIA** | 15 (False Negative) | 15 (True Positive) |

## 3. Explainable AI (XAI)

Using **Grad-CAM**, we visualized that the model primarily focuses on the mid-to-lower lung regions to distinguish pneumonia markers. The heatmaps confirm the model is learning relevant biological features rather than noise.

- **Target Layer**: `convolution_block9` (Detailed in `notebooks/06-grad-cam-xai-AyushDocs.ipynb`).

## 4. Experiment Tracking & Tuning

- **MLflow**: Integrated for tracking parameters, metrics (Batch Loss, Accuracy), and model versioning.
- **Hyperparameter Tuning**: Used **Optuna** to optimize learning rate and momentum, achieving significantly improved F1-scores.

## 5. Data Augmentation Analysis

Tested various augmentation strategies including:

- `RandomHorizontalFlip`: Improved robustness against orientation variance.
- `ColorJitter`: Helped the model generalize across different X-ray exposures.

## 6. Model Optimization

### Hyperparameter Tuning

- **Framework**: Optuna.
- **Best Model**: Saved as `best_model_optuna.pt`.

### Quantization (Post-Training)

Dynamic quantization was applied to reduce the model size for edge deployment.

| Model | Size (MB) | Avg Latency (ms) |
| :--- | :--- | :--- |
| **FP32 (Original)** | 0.08 | ~61.34 |
| **INT8 (Quantized)** | 0.03 | ~78.11 |

*Note: In the current CPU environment, INT8 shows slightly higher latency due to quantization overhead, but the 62% reduction in size is significant for storage-constrained environments.*
