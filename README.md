Here is the complete, self-contained **`README.md`** file ready to copy and paste directly into your GitHub repository:

```markdown
# Brain Tumor Detection using Hybrid CNN–GNN–SVM Architecture

An end-to-end, medical-grade deep learning pipeline for binary **Brain Tumor Classification (Tumor vs. Healthy)** from T1/T2-weighted Brain MRI scans.

This framework integrates a modified **4-channel DenseNet-121** convolutional backbone, an **Edge-Aware Graph Attention Network (GAT)** operating over hybrid spatial-topological graphs, an **Isotonically Calibrated RBF-SVM** decision head, and multi-modal **Explainable AI (Grad-CAM & SHAP)**. Strict multi-tier data deduplication (SHA-1, perceptual hashing, and Faiss cosine similarity) is enforced to eliminate data leakage.

---

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Key Innovations & Features](#key-innovations--features)
3. [Technology Stack](#technology-stack)
4. [Data Curation & Leakage Prevention Protocol](#data-curation--leakage-prevention-protocol)
5. [Preprocessing & Feature Engineering](#preprocessing--feature-engineering)
6. [Model Architecture in Detail](#model-architecture-in-detail)
7. [Multi-Task Hybrid Loss Formulation](#multi-task-hybrid-loss-formulation)
8. [Second-Stage Calibrated SVM Classifier](#second-stage-calibrated-svm-classifier)
9. [Explainable AI (XAI) Framework](#explainable-ai-xai-framework)
10. [Experimental Setup & Hyperparameters](#experimental-setup--hyperparameters)
11. [Empirical Benchmark & Evaluation Results](#empirical-benchmark--evaluation-results)
12. [Repository Structure](#repository-structure)
13. [Installation & Quick Start](#installation--quick-start)
14. [Author & Citation](#author--citation)
15. [License](#license)

---

## System Architecture

```text
               Raw T1/T2-Weighted Brain MRI Scan
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │       Multi-Tier Image Preprocessing & Edge Fusion          │
 │  • Global Percentile Intensity Normalization (2% - 98%)     │
 │  • Bilateral Filtering (d=9, sigmaColor=75, sigmaSpace=75)  │
 │  • Contrast Limited Adaptive Histogram Equalization (CLAHE) │
 │  • Hybrid Edge Map: 0.6 * Sobel (k=5) + 0.4 * Canny + Otsu  │
 └──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼  (Input: 224 × 224 × 4 Tensor)
 ┌─────────────────────────────────────────────────────────────┐
 │         Modified DenseNet-121 Feature Extractor             │
 │  • 4-Channel conv0 (Channel 4 initialized via RGB mean)     │
 │  • Multi-layer Dense Blocks + Transition Layers             │
 │  • Dropout (p=0.6) + Adaptive Average Pooling to 7 × 7      │
 └──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼  (C=1024, H=7, W=7 -> N=49 Nodes)
 ┌─────────────────────────────────────────────────────────────┐
 │               Hybrid Graph Construction                     │
 │  • Static Edges: 8-Connected Spatial Grid Topology          │
 │  • Dynamic Edges: Cosine Feature Similarity (top-k=3, ≥0.6) │
 │  • Batched PyG Graph Indexing across GPU Batches            │
 └──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │             Simplified Edge-Aware GAT (PyG)                 │
 │  • Layer 1: GATConv (1024 -> 128, heads=4), DropEdge (0.3)  │
 │             Residual Connection + LayerNorm + ReLU          │
 │  • Layer 2: GATConv (512 -> 128, heads=1), Residual + LN    │
 │  • Readout: Global Mean Pooling -> 128-D Graph Embedding    │
 └──────────────────────────────┬──────────────────────────────┘
                                │
                 ┌──────────────┴──────────────┐
                 ▼                             ▼
   ┌───────────────────────────┐ ┌───────────────────────────┐
   │ End-to-End Auxiliary Head │ │ Stage 2: Calibrated SVM   │
   │ • Multi-Task Hybrid Loss: │ │ • PCA Reduction (d=64)    │
   │   - Class-Weighted Focal  │ │ • StandardScaler          │
   │   - Hard Triplet Mining   │ │ • RBF-Kernel Support      │
   │   - Gradient Penalty (GP) │ │   Vector Machine (SVC)    │
   └───────────────────────────┘ │ • Isotonic Calibration    │
                                 │ • Optimal Macro-F1 Cutoff │
                                 └─────────────┬─────────────┘
                                               │
                                               ▼
                                  Final Diagnostic Output:
                                     Tumor vs. Healthy
                               (Accuracy: 99.80%, AUC: 0.9997)

```

---

## Key Innovations & Features

* **4-Channel Fused Input:** Supplements 3-channel RGB MRI with an anatomically responsive hybrid edge channel ($0.6 \times \text{Sobel} + 0.4 \times \text{Canny}$ thresholded via Otsu's algorithm) to explicitly preserve subtle intracranial boundary definitions.
* **Hybrid Static-Dynamic Graph Modeling:** Transposes deep feature maps ($7 \times 7$) into graph nodes, bridging immediate 8-neighbor spatial proximity with long-range semantic interactions using dynamic cosine thresholding.
* **Multi-Task Hybrid Loss with Gradient Penalty:** Simultaneously optimizes classification confidence via class-weighted focal loss, tightens intra-class clusters via hard negative triplet mining, and penalizes sharp embedding gradient norms to smooth decision manifolds.
* **Two-Stage Calibrated SVM Head:** Replaces standard linear heads with an RBF-SVM trained on dimensionality-reduced (PCA $n=64$) GNN embeddings, calibrated via out-of-fold isotonic regression to output clinically dependable probabilities (Brier score $0.0077$).
* **Data Leakage Immunity:** Enforces strict SHA-1 and perceptual hashing to purge duplicated records across splits, followed by Faiss-accelerated cosine similarity deduplication on the training set.

---

## Technology Stack

| Category | Component / Library | Version / Details | Purpose in Pipeline |
| --- | --- | --- | --- |
| **Deep Learning** | **PyTorch** | `2.6.0+cu124` | Core automatic differentiation, tensor execution, and model compilation. |
|  | **Torchvision** | `0.21.0` | Pretrained DenseNet-121 and ResNet-18 vision backbones. |
|  | **PyTorch Geometric (PyG)** | `2.7.0` | Sparse graph data handling, `GATConv`, and `global_mean_pool`. |
|  | **PyG C++ Extensions** | `pt26cu124` | `torch-scatter`, `torch-sparse`, `torch-cluster`, `torch-spline-conv`. |
| **CUDA Acceleration** | **NVIDIA CUDA Toolkit** | `12.4.127` | GPU-accelerated tensor arithmetic. |
|  | **NVIDIA cuDNN & cuBLAS** | `9.1.0` / `12.4.5` | Deep neural network primitives and matrix multiplication kernels. |
| **Image Processing** | **OpenCV (`cv2`)** | `4.10+` | Bilateral filtering, CLAHE, Sobel gradients, and Canny edge analysis. |
|  | **Pillow (PIL)** | `11.3.0` | Image loading, tensor transformations, and color-space casting. |
|  | **ImageHash** | `4.3.1` | Perceptual image hashing (`phash`) for structural duplicate tracking. |
| **Vector Search** | **Meta Faiss (`faiss-cpu`)** | `1.13.0` | High-throughput inner-product embedding search for deduplication. |
| **Classical ML & Stats** | **Scikit-Learn** | `1.6+` | RBF-SVM, PCA, StandardScaler, CalibratedClassifierCV, StratifiedKFold. |
|  | **NumPy / SciPy** | `1.26.4` / `1.15.3` | Matrix manipulation, distance computations, and percentile statistics. |
| **Explainable AI (XAI)** | **Grad-CAM (Custom)** | Native PyTorch | Feature map & gradient backpropagation hooks on `features.norm5`. |
|  | **SHAP** | `0.46+` | KernelExplainer for feature importance attribution on the SVM. |
| **Benchmarking** | **THOP** | `0.1.1` | FLOPs and parameter profiling (`profile`, `clever_format`). |
|  | **AMP (`GradScaler`)** | Native PyTorch | FP16/FP32 mixed-precision training. |

---

## Data Curation & Leakage Prevention Protocol

Medical imaging benchmarks often suffer from inflated performance metrics due to dataset contamination (identical or transformed scans appearing across train and test sets). This pipeline enforces a strict 4-stage filtering procedure:

```text
    Raw Dataset (12,524 images: 6,340 Tumor, 6,184 Healthy)
                              │
                              ▼
    Stage 1: Exact & Perceptual Hash Deduplication
             - SHA-1 content hashing
             - Perceptual image hashing (pHash)
             - 4,476 duplicate scans purged
                              │
                              ▼
    Deduplicated Cohort (8,048 images: 5,420 Tumor, 2,628 Healthy)
                              │
                              ▼
    Stage 2: Stratified Split (75% Train/Val : 25% Test)
             - Train/Val Raw Pool: 6,036 scans
             - Test Raw Pool:      2,012 scans
                              │
                              ▼
    Stage 3: Cross-Split Contamination Verification
             - Hash intersection between Train and Test = 0 overlaps
             - Hash check within Test split = 0 intra-test duplicates
                              │
                              ▼
    Stage 4: Feature-Level Deduplication (Applied ONLY to Training Set)
             - Deep embeddings extracted using Pretrained ResNet-18
             - Indexed via Faiss IndexFlatIP (cosine similarity threshold = 0.98)
             - 585 redundant training scans removed (Test set left untouched)
                              │
                              ▼
    Final Working Dataset:
    • Train/Val Set: 5,451 scans (3,804 Tumor, 1,647 Healthy)
    • Unseen Test Set: 2,012 scans (1,355 Tumor, 657 Healthy)

```

---

## Preprocessing & Feature Engineering

Each scan undergoes deterministic preprocessing with fold-isolated normalization parameters:

1. **Fold-Level Normalization:** Global intensity percentiles ($p_{low} = 2\%, p_{high} = 98\%$) are calculated solely on the training fold to prevent distribution leakage. The test set is normalized using the average parameters calculated across all cross-validation folds ($\text{Min}=0.00, \text{Max}=211.80$).
2. **Noise Reduction:** A bilateral filter ($d=9, \sigma_{\text{color}}=75, \sigma_{\text{space}}=75$) preserves critical tissue margins while smoothing acquisition artifacts.
3. **Contrast Enhancement:** Contrast Limited Adaptive Histogram Equalization (CLAHE) is applied with a clip limit of $2.0$ over an $8 \times 8$ grid.
4. **Hybrid Edge Synthesis:**

$$\mathbf{E}_{\text{hybrid}} = \text{Otsu}\Big(0.6 \cdot \text{Sobel}(\mathbf{I}_{\text{gray}}, k=5) + 0.4 \cdot \text{Canny}(\mathbf{I}_{\text{gray}}, 50, 150)\Big)$$


5. **Tensor Assembly:** The enhanced 3-channel RGB image is stacked with the hybrid edge map to produce a normalized $4 \times 224 \times 224$ input tensor.
6. **Class-Conditional Augmentation:** To counter class imbalance, healthy brain scans receive stronger conditional augmentations (elastic transformations $\alpha=80, \sigma=6$, affine shifts, random rotations, Gaussian blurring, and intensity variations).

---

## Model Architecture in Detail

### 1. Convolutional Backbone (Modified DenseNet-121)

* **First Convolution Adaptation:** The standard ImageNet `conv0` kernel ($7 \times 7, \text{stride}=2, \text{padding}=3$) is modified from 3 to 4 input channels.
* **Weight Initialization:** Weights for channels 0–2 are copied from pretrained weights; channel 3 (edge channel) is initialized using the channel-wise arithmetic mean of the RGB weights:

$$\mathbf{W}_{\text{conv0}}[:, 3, :, :] = \frac{1}{3}\sum_{c=0}^{2} \mathbf{W}_{\text{conv0}}[:, c, :, :]$$


* **Feature Projection:** Feature maps are processed through 4 dense connection blocks, followed by ReLU, adaptive average pooling to $7 \times 7$, and feature dropout ($p=0.6$).

### 2. Hybrid Graph Construction

* **Node Generation:** The $1024 \times 7 \times 7$ output tensor yields $N = 49$ spatial nodes per image, each represented by a $C=1024$ dimensional feature vector, normalized via LayerNorm.
* **Spatial Edges (Static):** Nodes are linked based on an 8-connected grid neighborhood graph $\mathcal{E}_{\text{spatial}}$, cached in GPU memory.
* **Semantic Edges (Dynamic):** A dynamic affinity graph $\mathcal{E}_{\text{feature}}$ is constructed by calculating pairwise cosine similarities:

$$\mathcal{S}_{ij} = \frac{\mathbf{f}_i \cdot \mathbf{f}_j}{\Vert{}\mathbf{f}_i\Vert{}_2 \Vert{}\mathbf{f}_j\Vert{}_2}, \quad \text{for } i \neq j$$



Top-$k$ ($k=3$) connections meeting the threshold $\mathcal{S}_{ij} \ge 0.6$ are retained.
* **Unified Graph:** Static and dynamic edge indices are merged: $\mathcal{E} = \text{Unique}(\mathcal{E}_{\text{spatial}} \cup \mathcal{E}_{\text{feature}})$. Graphs are batched using index offsets across mini-batches for parallel GPU execution.

### 3. Edge-Aware Graph Attention Network (GAT)

* **Layer 1:** Multi-head `GATConv` ($1024 \to 128$, $\text{heads}=4$, $\text{concat}=\text{True}$, output dimension $512$). Includes linear residual projection, LayerNorm, ReLU, and DropEdge regularization (rate $0.3$).
* **Layer 2:** Single-head `GATConv` ($512 \to 128$, $\text{heads}=1$, $\text{concat}=\text{False}$, output dimension $128$). Includes residual projection and LayerNorm.
* **Graph Pooling:** Node representations are pooled using global mean readout:

$$\mathbf{h}_{\mathcal{G}} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{h}_i^{(2)} \in \mathbb{R}^{128}$$



---

## Multi-Task Hybrid Loss Formulation

The end-to-end network is optimized using a composite objective function:

$$\mathcal{L}_{\text{total}} = 0.8 \cdot \mathcal{L}_{\text{Focal}} + 0.2 \cdot \mathcal{L}_{\text{Triplet}} + 0.1 \cdot \mathcal{L}_{\text{GP}}$$

### 1. Class-Weighted Focal Loss with Label Smoothing

To prevent majority-class gradient saturation, asymmetric label smoothing ($\epsilon_0 = 0.1, \epsilon_1 = 0.25$) and class weights ($\alpha_0 = 3.0, \alpha_1 = 1.5$) are applied:


$$\tilde{y} = y(1 - \epsilon) + (1 - y)\epsilon$$

$$\mathcal{L}_{\text{Focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t), \quad \gamma = 2.0$$

### 2. Hard-Mining Triplet Loss

Embeddings are $L_2$-normalized. Triplet pairs are mined dynamically by pairing each anchor with the most distant positive and nearest negative sample:


$$\mathcal{L}_{\text{Triplet}} = \max\Big(0, \Vert{}\mathbf{z}_a - \mathbf{z}_p\Vert{}_2^2 - \Vert{}\mathbf{z}_a - \mathbf{z}_n\Vert{}_2^2 + m_t\Big)$$


where class-dependent margins are set to $m_0 = 0.8$ and $m_1 = 1.2$.

### 3. Gradient Penalty ($\mathcal{L}_{\text{GP}}$)

Enforces a 1-Lipschitz continuity constraint over the latent manifold to prevent sharp decision boundaries:


$$\mathcal{L}_{\text{GP}} = \mathbb{E}_{\hat{\mathbf{z}}}\left[ \left( \Vert{}\nabla_{\hat{\mathbf{z}}} D(\hat{\mathbf{z}})\Vert{}_2 - 1 \right)^2 \right]$$

---

## Second-Stage Calibrated SVM Classifier

To ensure optimal decision boundaries and reliable probability outputs:

1. **Feature Extraction:** 128-D graph embeddings are extracted from the trained GNN.
2. **Dimension Reduction:** Principal Component Analysis (PCA) reduces the latent dimension to $n=64$ components.
3. **Hyperparameter Tuning:** A 3-fold cross-validated grid search explores:
* Regularization parameter: $C \in \{0.1, 1.0, 10.0, 100.0\}$
* Kernel coefficient: $\gamma \in \{\text{'scale'}, \text{'auto'}, 0.001, 0.01, 0.1\}$


4. **Isotonic Calibration:** An independent 30% calibration split fits an isotonic regression model (`CalibratedClassifierCV`) on top of the pre-fit SVM to map decision distances into reliable posterior probabilities.
5. **Macro-F1 Threshold Optimization:** The decision threshold is selected by optimizing the macro-averaged F1-score across precision-recall sweeps on validation folds.

---

## Explainable AI (XAI) Framework

Medical verification requires transparent model reasoning. The pipeline implements two complementary attribution frameworks:

```text
               ┌────────────────────────────────────────────────┐
               │              Explainable AI (XAI)              │
               └───────┬────────────────────────────────┬───────┘
                       │                                │
                       ▼                                ▼
         ┌───────────────────────────┐    ┌───────────────────────────┐
         │     Spatial Grad-CAM      │    │     Latent Feature SHAP   │
         │  (Backpropagation from    │    │  (KernelExplainer on PCA- │
         │   GNN to features.norm5)  │    │   Transformed Embeddings) │
         └─────────────┬─────────────┘    └─────────────┬─────────────┘
                       │                                │
                       ▼                                ▼
            Anatomical Heatmaps &             Global Feature Impact &
         Lesion Localization Overlays        Biomarker Directional Plots

```

1. **Spatial Localization (Grad-CAM):** Backward gradients flow from the target prediction score back to the final convolutional normalization layer (`cnn.backbone.features.norm5`). Global average-pooled gradients weigh the activation maps, followed by ReLU thresholding and bilinear upsampling to $224 \times 224$.
2. **Feature Attribution (SHAP):** A `KernelExplainer` models prediction probability shifts over the PCA-transformed embedding dimensions to determine which latent components drive tumor predictions.

---

## Experimental Setup & Hyperparameters

```yaml
Hardware Environment:
  Device: NVIDIA GPU (CUDA 12.4 enabled)
  cuBLAS Workspace Config: ":4096:8"
  Deterministic Algorithms: Enabled (warn_only=True)

Data Pipeline:
  Input Resolution: 224 x 224 pixels
  Input Channels: 4 (RGB + Hybrid Edge Map)
  Stratified K-Fold: 5 Folds
  Batch Size: 32
  Oversampling Ratio: 2.5 (WeightedRandomSampler)

Optimization & Training:
  Optimizer: AdamW
  Initial Learning Rate: 5e-5
  Weight Decay: 5e-5
  Warmup Epochs: 5 (Linear Warmup)
  LR Decay Schedule: Cosine Annealing
  Maximum Epochs: 40
  Early Stopping: Patience = 3 (Monitored on Validation Loss)
  Mixed Precision: PyTorch AMP (GradScaler)
  Gradient Clipping: Max Norm = 1.0

Architecture Configurations:
  DenseNet Feature Dim: 1024
  Grid Spatial Size: 7 x 7 (49 nodes)
  GAT Hidden Dimension: 128
  GAT Attention Heads: 4
  Embedding Dimension: 128
  Dropout Rates: CNN Backbone = 0.6, GNN Layers = 0.8
  DropEdge Regularization: 0.3

```

---

## Empirical Benchmark & Evaluation Results

### 1. Computational Efficiency Profile

Measured on the complete 4-channel input architecture using `thop`:

| Metric | Measured Value |
| --- | --- |
| **Model Parameter Count** | **8.15 Million** |
| **Memory Footprint (Weights + Buffers)** | **31.41 MB** |
| **Computational Complexity** | **2.96 GFLOPs** |
| **Inference Throughput** | **82.97 samples / second** |
| **Latency per Scan** | **12.05 milliseconds** |

---

### 2. 5-Fold Stratified Cross-Validation Results

The network was evaluated across 5 stratified folds on the training split ($N = 5,451$):

| Fold Index | Epochs | Train Acc @ Best AUC | Val Accuracy | Val Precision | Val Recall | Val F1-Score | Val ROC-AUC | Val Brier Score | Optimal Threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Fold 1** | 30 | 98.78% | 99.45% | 0.9987 | 0.9934 | 0.9960 | 0.9971 | 0.0062 | 0.500 |
| **Fold 2** | 19 | 97.29% | 97.89% | 0.9817 | 0.9882 | 0.9849 | 0.9969 | 0.0209 | 0.269 |
| **Fold 3** | 24 | 98.32% | 98.53% | 0.9934 | 0.9855 | 0.9894 | 0.9896 | 0.0139 | 0.750 |
| **Fold 4** | 26 | 98.76% | 98.81% | 1.0000 | 0.9829 | 0.9914 | 0.9960 | 0.0123 | 0.239 |
| **Fold 5** | 14 | 96.51% | 97.52% | 0.9986 | 0.9658 | 0.9819 | 0.9971 | 0.0260 | 0.207 |
| **Mean ± Std** | — | **97.93% ± 0.89%** | **98.44% ± 0.68%** | **0.9945 ± 0.0068** | **0.9832 ± 0.0094** | **0.9888 ± 0.0049** | **0.9953 ± 0.0029** | **0.0159 ± 0.0069** | **0.393 ± 0.231** |

---

### 3. Held-Out Unseen Test Set Performance ($N = 2,012$)

Predictions from all 5 fold models were ensembled via soft probability voting. The final classification cutoff was set to $\tau = 0.255$ based on optimal validation macro-F1 tuning:

| Metric | Ensemble Test Performance |
| --- | --- |
| **Test Accuracy** | **99.80%** (2,008 / 2,012 scans correct) |
| **Precision** | **99.78%** |
| **Recall (Sensitivity)** | **99.93%** (1,354 / 1,355 tumors detected) |
| **Specificity** | **99.54%** (654 / 657 healthy scans correct) |
| **F1-Score** | **99.85%** |
| **ROC-AUC** | **0.9997** |
| **Brier Score** | **0.0077** |

#### Detailed Test Classification Report

```text
              precision    recall  f1-score   support

     Healthy       1.00      1.00      1.00       657
       Tumor       1.00      1.00      1.00      1355

    accuracy                           1.00      2012
   macro avg       1.00      1.00      1.00      2012
weighted avg       1.00      1.00      1.00      2012

```

#### Test Set Confusion Matrix Breakdown

* **True Positives (Tumor correctly identified):** 1,354
* **True Negatives (Healthy correctly identified):** 654
* **False Positives (Healthy misclassified as Tumor):** 3
* **False Negatives (Tumor missed):** 1

---

### 2. Dependency Installation

Install dependencies configured for CUDA 12.4:

```bash
# 1. Install pinned NumPy
pip install numpy==1.26.4

# 2. Install Faiss CPU without dependency overrides
pip install faiss-cpu --no-deps

# 3. Install PyTorch Geometric and CUDA 12.4 wheels
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f [https://data.pyg.org/whl/torch-2.6.0+cu124.html](https://data.pyg.org/whl/torch-2.6.0+cu124.html)

# 4. Install remaining vision and diagnostic libraries
pip install opencv-python pillow==11.3.0 imagehash thop scikit-learn shap matplotlib seaborn tqdm

```

### 3. Executing the Pipeline

Organize your dataset under the `dataset/` directory (`YES/` and `NO/` folders), then trigger the complete pipeline:

```bash
python train_pipeline.py

```

---

**Satyanand Shukla**

*M.Tech Bioinformatics*

Specialization: Machine Learning, Graph Neural Networks, and Medical Image Analytics

```

```
