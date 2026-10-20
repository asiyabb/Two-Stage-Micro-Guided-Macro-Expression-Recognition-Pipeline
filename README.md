# Two-Stage Micro-Guided Macro-Expression Recognition Pipeline

This repository implements a novel two-stage micro-guided macro-expression recognition framework. The core intuition is to leverage fine-grained, short-term temporal motion dynamics from subtle micro-expressions (captured via a specialized micro-clue branch) to guide and enhance the recognition of broader, more visible macro-expressions[cite: 20, 21].

---

## 🏗️ Architectural Overview

The proposed pipeline processes video sequences through two complementary branches before fusing them via multi-head cross-attention and adaptive sigmoid gating:

1. **ViT Feature Extraction (`vit_extractor.py`)**: 
   - Takes video sequences of shape $[B, T, 3, H, W]$ ($T=32$ frames, $224 \times 224$ resolution)[cite: 16, 17, 21].
   - Extracts frame-level spatial features using a Vision Transformer (`vit_b_16`, 768-dim), yielding $F_{\mathrm{ViT}} \in \mathbb{R}^{B \times T \times 768}$[cite: 21].
2. **Macro Branch (`macro_guided_model.py`)**: 
   - Processes the full sequence of ViT features through a Bidirectional GRU (BiGRU) to summarize global temporal context[cite: 12, 21].
   - Outputs a global macro representation vector of shape $[B, 512]$[cite: 21].
3. **Micro-Clue Branch (`micro_encoder.py`)**: 
   - Applies short overlapping sliding windows (window size $8$, stride $4$) over the feature sequence[cite: 16, 21].
   - Computes consecutive temporal differences within windows to isolate motion dynamics rather than static appearance[cite: 21].
   - Processes windows through a shared BiGRU to produce local latent micro-clues $[B, W, 256]$ ($W=7$ windows)[cite: 13, 21].
4. **Cross-Attention & Adaptive Gating (`macro_guided_model.py`)**: 
   - Uses the global macro vector as a query to attend over the micro-clue sequence via multi-head cross-attention[cite: 12, 21].
   - Employs an adaptive sigmoid gating mechanism to control the contribution of the retrieved micro clues before fusing them with the macro representation[cite: 21].
5. **Classifier**: 
   - Passes the fused latent representation $[B, 512]$ through a multi-layer perceptron to predict the final macro-expression emotion class (anger, disgust, fear, happiness, sadness, surprise)[cite: 12, 16, 21].

---

## 🧪 Training Strategies

The model supports two distinct training strategies alongside self-supervised pretraining:
* **Stage 1 — Micro-Encoder Pretraining (`pretrain_micro.py`)**: Initializes the micro-encoder using contrastive window loss (InfoNCE) and temporal consistency loss to cluster similar motion dynamics and smooth temporal fluctuations[cite: 18, 21].
* **Strategy 1 (Two-Stage)**: Pretrains the micro-encoder, freezes its weights, and trains the full macro-guided classification model[cite: 21].
* **Strategy 2 (Joint End-to-End)**: Jointly trains the micro-encoder and the macro-guided network end-to-end using cross-entropy loss[cite: 21].

---

## 📊 Baseline Experiment

To rigorously validate the proposed architecture, performance is compared against a separate **Macro-Only Baseline** located in `MMEW_Macro_Pipeline(Baseline)/`:
* **Architecture**: Processes 1050-dimensional combined feature vectors (1024-dim ShuffleNetV2 facial appearance features + 26-dim MediaPipe geometric facial landmark angles/distances) through a 2-layer Bidirectional LSTM.
* **Evaluation**: Evaluated on the exact same subject-independent splits to ensure an unbiased comparison[cite: 5, 17].

---

## 📁 Directory Structures

### 1. Proposed Model Directory (`mmew_guided_recognition/`)
```text
mmew_guided_recognition/
│
├── config/
│   └── config.py                   # Global hyperparameters, paths, and training settings
├── data/
│   └── dataset.py                  # Sequence loading, augmentations, and GroupShuffleSplit
├── losses/
│   └── micro_losses.py             # ContrastiveWindowLoss (InfoNCE) and TemporalConsistencyLoss
├── models/
│   ├── vit_extractor.py            # ViT-B/16 frame-level spatial feature extractor
│   ├── micro_encoder.py            # Short-window temporal difference BiGRU micro-encoder
│   └── macro_guided_model.py       # Macro branch, cross-attention, adaptive gate, and classifier
│
├── evaluate_all_strategies.py      # Unified script running Strategy 1, Strategy 2, and comparative metrics
├── evaluate_baseline.py            # Isolated evaluation script for the ShuffleNet + LSTM baseline
├── pretrain_micro.py               # Stage 1 micro-encoder pretraining script
├── strategy_curves_comparison.png  # Generated training/validation loss and accuracy curves
└── confusion_matrix_guided.png     # Generated normalized confusion matrix visualization


### 2. Baseline Pipeline Directory (`MMEW_Macro_Pipeline(Baseline)/`)

```text
MMEW_Macro_Pipeline(Baseline)/

├── Combined_MMEW_1050d.json        # Horizontal combination of ShuffleNet and Geometric features
├── Shufle_MMEW_1024d.json          # 1024-dim ShuffleNetV2 facial features
├── Geometric_MMEW_26d.json         # 26-dim MediaPipe geometric features
├── best_model_mmew.pth             # Pretrained baseline LSTM weights
├── shufle_mmew.py                  # ShuffleNet feature extraction script
├── geometric_features_mmew.py      # MediaPipe geometric feature extractor
├── cocatenacion_mmew.py            # Feature concatenation script
└── Training_LSTM_MMEW.py           # Baseline LSTM training script

```

---

## 📈 Experimental Results

Subject-independent validation splits demonstrate the effectiveness of incorporating motion-guided micro-clues over traditional macro-only pipelines:

| Model / Training Strategy | Accuracy | Macro F1-Score |
| --- | --- | --- |
| **Macro-Only Baseline (ShuffleNet + LSTM 1050d)**<br> | **38.89%**<br> | **0.3418**<br> |
| **Strategy 1 (Frozen Micro Branch)**<br> | **40.00%** | **0.3634** |
| **Strategy 2 (Joint End-to-End)**<br> | **60.00%** | **0.5888** |

---

## 🚀 Getting Started

### Prerequisites

Install dependencies inside your virtual environment (`gpuenv`):

```bash
pip install torch torchvision opencv-python mediapipe scikit-learn matplotlib seaborn pandas tqdm

```

### Running the Pipeline

1. **Evaluate the Baseline**:
```bash
python evaluate_baseline.py

```


2. **Run Strategies & Generate Comparative Metrics and Plots**:
```bash
python evaluate_all_strategies.py


