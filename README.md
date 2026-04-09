<div align="center">

# 🧠 SegSAM3: Parameter-Efficient Medical 3D Segmentation

**An Adapter-Based Fine-Tuning Framework for Segment Anything Model 3 on Brain Tumor Data**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](#dependencies)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](#dependencies)
[![MONAI](https://img.shields.io/badge/MONAI-1.3+-00A4DC)](#dependencies)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[English](#abstract) · [简体中文](./README_zh.md) · [Architecture Diagrams](./画图.md)

</div>

---

## 📖 Abstract

Vision Foundation Models (VFMs) like the 600M+ parameter SAM 3 exhibit remarkable zero-shot capabilities in natural images but face severe challenges in 3D medical image segmentation. These challenges stem from massive domain gaps, catastrophic forgetting of visual priors during full fine-tuning, and insurmountable GPU memory bottlenecks when processing 3D volumes. 

**SegSAM3** introduces a mathematically rigorous, structurally focused **Parameter-Efficient Fine-Tuning (PEFT)** paradigm that resolves these bottlenecks. By freezing 99.5% of the SAM 3 backbone and injecting lightweight bottleneck adapters, SegSAM3 successfully maps the foundational visual intelligence of SAM 3 into the specialized domain of brain tumor segmentation (BraTS dataset). 

> **Research Focus:** This repository serves as the official implementation for my graduation thesis. It is designed to demonstrate high-quality academic software engineering, reproducible baseline evaluations, and a rigorous theoretical approach to medical image analysis.

---

## ✨ Key Research Contributions

1. **Parameter-Efficient Domain Adaptation (PEFT)**  
   Designed a zero-initialized, residual adapter module injected alongside every SAM 3 ViT block. This mechanism strictly prevents catastrophic forgetting by starting off entirely transparent to the foundation model, subsequently learning disease-specific anomalies while updating only **~0.5%** of the overall parameter footprint.

2. **Robust Multi-Objective Loss Formulation**  
   Medical segmentation inherently suffers from extreme spatial voxel imbalance (tumor pixels often represent `< 1%` of the volume). SegSAM3 intercepts this via a rigorous dual-path gradient formulation:
   - **Tversky Loss:** Configured with stringent parameters ($\alpha=0.3$, $\beta=0.7$) to aggressively penalize false negatives, ensuring high recall on critical anomalies.
   - **Log-Dice Loss:** Amplifies early-stage gradients exponentially (`−log(Dice + ε)`) to escape local minima.
   - **Combined Target:** Computes $L = 0.5 \times L_{Tversky} + 0.5 \times L_{LogDice}$, balancing sensitivity and convergence stability.

3. **RoPE Frequency Translation**  
   SAM 3's native Rotary Position Embeddings (RoPE) are hard-bound to 1024×1024 resolutions. SegSAM3 executes dynamic frequency interpolation (`freqs_cis`) to natively down-translate the foundation coordinate space to standard medical GPU bounds (256×256) without structurally warping the transformer attention maps.

4. **Probabilistic Foreground-Aware 2.5D Slicing**  
   NIfTI medical volumes contain massive voids of empty background spatial data. Instead of blindly passing 3D tensors, we deploy a customized, statistically-weighted PyTorch DataLoader that parses voxel density on the fly. It probabilistically bypasses blank slices, effectively maximizing dense feature extraction logic per GPU cycle.

---

## 🏗 System Architecture

SegSAM3 adopts a highly decentralized, object-oriented design optimized for extensibility and strictly decoupled logic components.

The inference pipeline operates as follows: Raw 3D NIfTI medical volumes are first transformed into **2.5D image batches** via a **foreground-aware slicing strategy** that probabilistically bypasses empty background slices. These batches are then fed into the **frozen SAM 3 ViT backbone** for feature extraction. Within each ViT encoder block, features are routed through a bypass branch into **zero-initialized adapters (Zero-Init Adapters)**, whose outputs are added back to the backbone via residual connections — enabling domain adaptation without disrupting the pretrained representations. The resulting feature maps are passed to a **lightweight mask decoder** to produce the final segmentation predictions. During training, gradients are computed through a **compound loss function (0.5 × Tversky + 0.5 × Log-Dice)**, and the AdamW optimizer **updates only the adapter parameters**, achieving efficient fine-tuning with less than 0.5% trainable parameters.

---

## 🔬 Evaluation & Metrics Standards

Academic integrity relies on rigorous quantitative assessment. SegSAM3 computes volume-level metrics exclusively, avoiding the bias of slice-by-slice averaging.

- **Dice Coefficient (DSC):** Measures generic volumetric overlap $\frac{2|A \cap B|}{|A| + |B|}$.
- **Intersection over Union (IoU):** Validates strict boundary overlapping $\frac{|A \cap B|}{|A \cup B|}$.
- **95th Percentile Hausdorff Distance (HD95):** Captures extreme boundary divergence and topological consistency. Only calculated on valid foreground pairs.

---

## 🚀 Reproducibility & Quick Start

A cornerstone of this repository is complete, painless reproducibility.

### 1. Environment Setup
```bash
# Strongly recommended to use a virtual environment
pip install -r requirements.txt
```

### 2. Sanity Validation (Dry Run)
Verify the computational graph, data loaders, and mathematical stability without requiring the 600M+ parameter checkpoint.
```bash
python main.py --config configs/exp_group_a.yaml --epochs 1 --use_mock
```

### 3. Full Experimental Execution
Trigger the centralized pipeline. The model tracks hyper-variables dynamically via TensorBoard.
```bash
python main.py \
  --config configs/exp_group_a.yaml \
  --data_root data/brats_volumes \
  --epochs 60 \
  --batch_size 1 \
  --lr 5e-5 \
  --device cuda
```
*(Requires the SAM 3 ViT baseline weights located at `data/checkpoints/sam3.pt`)*

---

## 📂 Engineering & Code Organization

The codebase strictly adheres to the Single Responsibility Principle (SRP) to eliminate coupled spaghetti code, ensuring logical purity and straightforward ablation studies.

```text
SegSAM3/
├── main.py                              # Centralized trigger interface
├── configs/
│   └── exp_group_a.yaml                 # Deterministic YAML variables
├── src/
│   ├── integrated_model.py              # PEFT topological assembly
│   ├── improved_losses.py               # Robust mathematical loss calculus
│   ├── metrics.py                       # Pure 3D academic formulations
│   └── models/
│       ├── adapter.py                   # The mathematical PEFT bottleneck
│       └── freeze_utils.py              # Gradient requirement manipulation
├── data/
│   └── dataset_loader.py                # Statistical & probabilistic ingestion
└── scripts/                             # Automation tooling
```

---
