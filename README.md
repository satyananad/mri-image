# Brain Tumor Detection using Hybrid CNN–GNN–SVM Architecture

## Overview

This project implements a **hybrid deep learning framework for brain tumor detection from MRI images**.
The system combines **Convolutional Neural Networks (CNN), Graph Neural Networks (GNN), and Support Vector Machine (SVM)** to improve feature representation and classification performance.

The model extracts spatial features using **DenseNet121**, converts them into **graph structures**, and processes them using **Graph Attention Networks (GAT)** for relational learning.
Finally, classification is performed using an **RBF-SVM with probability calibration**.

The pipeline also integrates **Explainable AI (XAI)** techniques such as **Grad-CAM and SHAP** to interpret model predictions.

---

## Key Features

* Hybrid **CNN + GNN + SVM architecture**
* Advanced **MRI preprocessing and edge-aware feature extraction**
* **Hybrid graph construction** combining spatial and feature similarity edges
* **Multi-task hybrid loss function**

  * Focal Loss
  * Triplet Loss
  * Gradient Penalty
* **Explainable AI**

  * Grad-CAM visualization
  * SHAP feature importance
* **Data leakage prevention**

  * SHA-1 hashing
  * Perceptual hashing
* **Stratified K-Fold Cross Validation**
* **Ensemble learning across folds**

---

## Model Architecture

MRI Image
↓
Image Preprocessing (CLAHE + Bilateral Filter + Edge Detection)
↓
DenseNet121 Feature Extraction
↓
7×7 Feature Map Graph Construction
↓
Graph Attention Network (GAT)
↓
Graph Embedding (128D)
↓
RBF-SVM Classification
↓
Tumor / Healthy Prediction

---

## Preprocessing Pipeline

The preprocessing stage includes:

* Global intensity normalization
* Bilateral filtering for noise reduction
* CLAHE contrast enhancement
* Hybrid edge detection

  * Sobel (60%)
  * Canny (40%)

The final image representation contains:

* **3 RGB channels**
* **1 edge map channel**

Total input channels = **4**

---

## Explainable AI

Two explainability techniques are used:

### Grad-CAM

Highlights important image regions responsible for predictions.

### SHAP

Explains feature importance for the final SVM classifier.

These techniques improve **model interpretability for medical applications**.

---

## Performance Metrics

Model efficiency results:

* **Model Size:** ~31 MB
* **Trainable Parameters:** ~8.15M
* **Computational Complexity:** ~2.96 GFLOPs
* **Inference Speed:** ~83 samples/sec
* **Prediction Time:** ~12 ms per sample

---

## Evaluation Metrics

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Brier Score
* Confusion Matrix
* Reliability Diagram

Cross-validation is performed using **5-fold stratified splitting**.

---

## Dataset Structure

Dataset should be organized as:

dataset/
├── YES/
│   ├── tumor1.jpg
│   ├── tumor2.jpg
│
└── NO/
├── healthy1.jpg
├── healthy2.jpg

YES → Tumor images
NO → Healthy images

---

## Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git

Navigate to the project folder:

cd YOUR_REPOSITORY_NAME

Install dependencies:

pip install -r requirements.txt

---

## Running the Project

Run the training pipeline:

python train_pipeline.py

If using a notebook:

Open the notebook and run all cells sequentially.

---

## Visualization Outputs

The project generates multiple visualizations:

* Confusion Matrix
* ROC Curve
* Reliability Diagram
* Grad-CAM heatmaps
* t-SNE embedding visualization

---

## Results

Example outputs include:

* Ensemble test predictions
* Model calibration curves
* Embedding visualizations
* Explainability heatmaps

---

## Research Applications

This framework can be extended to:

* Multi-class brain tumor classification
* Medical image segmentation
* Other medical imaging tasks

---

## Author

Satyanand Shukla
M.Tech Bioinformatics
Machine Learning and Medical Imaging Research

---

## License

MIT License
