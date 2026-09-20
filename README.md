# Early Skin Cancer Risk Prediction in Bovines

## Overview

Early Skin Cancer Risk Prediction in Bovines is an AI-based image screening system designed to provide an initial assessment of skin-related risk in bovines using images.

The system uses a Convolutional Neural Network (CNN) based on the MobileNetV2 architecture to classify uploaded bovine skin images into three risk categories:

- Low Risk
- Medium Risk
- High Risk

The application provides a risk level, confidence score, possible skin condition, and a recommendation based on the predicted risk level.

> **Disclaimer:** This system is intended for early screening and awareness purposes only. It does not provide a veterinary diagnosis and should not replace examination by a qualified veterinarian.

---

## Problem Statement

Skin-related conditions in bovines may not always be noticed at an early stage, particularly when veterinary facilities and regular veterinary examinations are limited.

An image-based screening system can help provide an initial indication of potential skin-related risk and encourage timely veterinary attention when necessary.

---

## Objectives

- Develop an AI-based image screening system for bovine skin conditions.
- Classify images into Low, Medium, and High risk categories.
- Use MobileNetV2 for image classification.
- Provide a simple and farmer-friendly web interface.
- Display the uploaded image along with the prediction.
- Provide confidence information and recommendations.
- Encourage veterinary consultation for potentially serious cases.

---

## Key Features

### 1. Image Upload

Users can upload an image of bovine skin through the web interface.

### 2. Image Preprocessing

The uploaded image is:

- Converted to RGB format
- Resized to 96 × 96 pixels
- Converted into a PyTorch tensor
- Normalized using ImageNet mean and standard deviation

### 3. AI-Based Risk Prediction

The MobileNetV2 model classifies the image into:

| Risk Level | Description |
|---|---|
| Low | Skin appears normal |
| Medium | Possible minor skin-related issue |
| High | Potentially serious skin-related condition |

### 4. Confidence Score

The system calculates the prediction probability using Softmax and displays the resulting confidence score.

### 5. Recommendation

A recommendation is generated according to the predicted risk level.

For example, high-risk predictions recommend consultation with a veterinarian.

### 6. Web Interface

The application is implemented using Flask and provides an interface for uploading images and viewing prediction results.

---

## Machine Learning Model

The project uses **MobileNetV2**, a lightweight convolutional neural network architecture suitable for image classification.

### Model Configuration

- Architecture: MobileNetV2
- Framework: PyTorch
- Input image size: 96 × 96 pixels
- Number of classes: 3
- Optimizer: Adam
- Learning rate: 0.0005
- Loss function: Cross Entropy Loss
- Batch size: 16
- Training epochs: 10
- Train/Test split: 85% / 15%

The final trained model is saved as:

```text
m.pth