# Shadow Model Strategy for XAI

This document explains how PnuemoCheck AI handles Explainable AI (XAI) for production-quantized models.

## The Challenge: Quantization vs. Gradients

To achieve low latency in production, we use **int8 quantization**. However, quantized layers (like `ConvReLU2d`) are non-differentiable and discard the gradient information required by **Grad-CAM**.

## The Solution: Shadow Models

Our architecture implements a **Shadow Model Strategy**:

1. **Inference (Fast Path)**: The user request is first processed by the `int8` model stored in **BentoML** for rapid diagnosis.
2. **Explanation (XAI Path)**: If a heatmap is requested, the system automatically:
    - Identifying the corresponding `float32` version of the active model.
    - Loading the "shadow" float model into memory (cached for subsequent requests).
    - Generating the Grad-CAM heatmap using the float model's gradients.
3. **Persistence**: The generated explanation is logged as an artifact in **MLflow**, linked to the original prediction metadata.

## Workflow Summary

- **Model Store**: BentoML (manages both `:latest` and `:latest_int8` versions).
- **Explanation Store**: MLflow (stores heatmap images and XAI metrics).

> [!IMPORTANT]
> Always ensure that for every `int8` model pushed to BentoML, the original `float32` version remains available in the registry for XAI compatibility.
