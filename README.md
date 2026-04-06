# FakeFrame - Explainable Deepfake Image Detector

## What is FakeFrame?
FakeFrame is a deepfake image detection system that classifies images as REAL or FAKE and explains WHY using Grad-CAM heatmaps — showing exactly which regions the CNN focused on when making its decision.
Most deepfake detectors are black boxes. FakeFrame makes the decision explainable and transparent.

User uploads an image -> EfficientNet-B0 (CNN) analyzes it -> REAL or FAKE + confidence score -> Grad-CAM heatmap shows WHERE the CNN found evidence

## How FakeFrame differs from existing work:
Most existing deepfake detection projects, achieved 87.38% accuracy using a custom CNN built with TensorFlow and Keras, focus solely on binary classification — outputting a simple real or fake label with no further explanation. FakeFrame takes this a step further by integrating Grad-CAM explainability directly into the detection pipeline, revealing exactly which regions of an image the CNN identified as suspicious. Rather than building a CNN from scratch, FakeFrame uses EfficientNet-B0 with transfer learning — a more modern and efficient approach that achieves 89% accuracy while requiring only 2,562 trainable parameters instead of millions. Additionally, FakeFrame provides a live interactive Streamlit dashboard where anyone can upload any image and receive a real-time verdict, confidence score, and visual heatmap — making it a complete end-to-end explainable AI system rather than just a model.

## Results
- **Accuracy:** 89.0%  
- **ROC-AUC:** 0.957  
- **F1 Score:** 0.89  
- **Training Time:** ~20 mins on T4 GPU  
- **Test Images:** 20,000  

## Dataset - CIFAKE
**Download:** [CIFAKE Dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)

- **Total Images:** 120,000  
- **Classes:**  
  - REAL: 60,000 (CIFAR-10 images)  
  - FAKE: 60,000 (AI-generated using Stable Diffusion)

**CIFAR-10 Categories:**  
`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`

**Folder Structure:**
data/
  train/
    REAL/    ← 50,000 images
    FAKE/    ← 50,000 images
  test/
    REAL/    ← 10,000 images
    FAKE/    ← 10,000 images


## Model Architecture
- **Base:** EfficientNet-B0 pretrained on ImageNet  
- **Approach:** Transfer learning — freeze all layers, replace final layer  
- **Trainable Params:** 2,562 out of 4,010,110 total

**Architecture Flow:**
Input (224×224) -> EfficientNet-B0 - FROZEN -> Linear(1280 → 2) - TRAINABLE -> FAKE or REAL -> Grad-CAM heatmap


## Tech Stack
- PyTorch + torchvision — Training and inference  
- EfficientNet-B0 — Pretrained CNN backbone  
- pytorch-grad-cam — Heatmap generation  
- scikit-learn — Evaluation metrics  
- Streamlit — Dashboard  
- Google Colab T4 — Free GPU  
- pyngrok — Public dashboard URL  

## How To Run
1. Install dependencies:
```bash
pip install torch torchvision grad-cam opencv-python streamlit pyngrok pillow plotly scikit-learn matplotlib
```
2. Then run every cell and attach your images to know whether they are real or fake.
   
## Why This Matters
Deepfakes can spread misinformation and enable identity fraud.  
Explainable detection helps forensic analysts **understand the evidence**, making AI decisions **transparent and trustworthy**.

## Limitations
- Trained on CIFAKE (32×32 upscaled) — performance may differ on high-resolution, real-world deepfakes  
- Best performance on CIFAR-10 categories (animals, vehicles)  
- Out-of-distribution images may produce less reliable results
