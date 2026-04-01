
# SegSAM3

[Back to Home](./README.md) | [简体中文](./README_zh.md)

> A research-oriented framework for **heterogeneous multimodal federated medical image segmentation** based on **SAM3 + Adapter + federated aggregation**.

---

## Introduction

SegSAM3 is a research-oriented codebase for **federated medical image segmentation**, designed with the following goals:

- Support **heterogeneous federated clients**
- Use **SAM3** as the visual backbone
- Reduce communication and optimization cost through **Adapters / lightweight fine-tuning**
- Provide a unified training framework for:
  - `text_only`
  - `image_only`
  - `multimodal`
- Offer a full experimentation pipeline:
  - federated training
  - validation
  - checkpoint recovery
  - plotting
  - mask export

The default config `configs/exp_group_a.yaml` corresponds to a controlled **GroupA image-only** experiment, while the underlying code already supports heterogeneous multimodal federated learning extensions.

---

## Key Features

- **SAM3-based medical segmentation**
  - real SAM3 checkpoint loading
  - mock mode for pipeline smoke tests
  - RoPE reset and input-size adaptation

- **Lightweight parameter tuning**
  - Adapter injection into Transformer blocks
  - frozen backbone with trainable lightweight modules
  - parameter-group-specific learning rates

- **Heterogeneous client support**
  - `text_only`
  - `image_only`
  - `multimodal`

- **Complete federated training loop**
  - serial client-side training
  - server-side aggregation
  - global image/text representation maintenance
  - early stopping, checkpointing, logging, and visualization

- **Research-friendly diagnostics**
  - Dice / IoU / HD95 evaluation
  - JSON training history export
  - automatic plotting
  - predicted and ground-truth mask export

---

## Repository Structure

```text
SegSAM3/
├─ main.py
├─ configs/
│  └─ exp_group_a.yaml
├─ src/
│  ├─ config_manager.py
│  ├─ federated_trainer.py
│  ├─ integrated_model.py
│  ├─ client.py
│  ├─ server.py
│  └─ ...
├─ data/
│  ├─ heterogeneous_dataset_loader.py
│  └─ ...
├─ scripts/
├─ tests/
├─ requirements.txt
└─ requirements-lock.txt
````

---

## Core Modules

### `main.py`

The main training entrypoint. It is responsible for:

* parsing CLI arguments
* loading YAML configs
* applying CLI overrides
* creating `FederatedTrainer`
* launching federated training

---

### `src/config_manager.py`

Centralized configuration manager that flattens nested YAML fields into `FederatedConfig`, covering training, validation, logging, checkpointing, and device settings.

---

### `src/federated_trainer.py`

The main federated training controller, responsible for:

* initializing the global model
* initializing the server aggregator
* preparing clients and state caches
* building validation loaders
* running each federated round
* validation, early stopping, checkpointing, and plotting

---

### `src/integrated_model.py`

Core model class `SAM3MedicalIntegrated`, which integrates:

* SAM3 backbone loading
* Adapter injection
* frozen backbone strategy
* image/text projection heads
* text prompt injection
* output channel adaptation
* medical segmentation head initialization
* contrastive-learning-related utilities

---

### `src/client.py`

Client-side trainer implementation with one abstract base class and three subclasses:

* `TextOnlyTrainer`
* `ImageOnlyTrainer`
* `MultimodalTrainer`

---

### `src/server.py`

Server-side aggregator `CreamAggregator`, responsible for:

* client weight aggregation
* modality-decoupled routing
* global image/text representation maintenance
* gradient conflict probing
* heterogeneous client tuple aggregation

---

### `data/heterogeneous_dataset_loader.py`

Dataset loader for heterogeneous clients, supporting:

* `text_only`
* `image_only`
* `multimodal`

It implements:

* BraTS-style case loading
* 2D slice construction
* `.nii / .nii.gz` image and mask loading
* `.npy` text feature loading
* train / validation / public-data loading

---

## Supported Client Types

### `text_only`

Uses only text features and does not perform segmentation.

* Private format: `(text_feature,)`
* Public format: `(text_feature,)`

---

### `image_only`

Uses only images and segmentation masks. This is the default experiment path.

* Private format: `(image, mask)`
* Public format: `(image,)`

---

### `multimodal`

Uses images, masks, and text features jointly.

* Private format: `(image, mask, text_feature)`
* Public format: `(image, text_feature)`

---

## Data Layout

The current implementation is designed for a BraTS-style directory structure, such as:

```text
data_root/
  train/
    client_x/
      private/
        BraTSxxxx/
          *_flair.nii.gz / *_t1.nii.gz
          *_seg.nii.gz
          *_text.npy
      public/
        ...
  val/
    client_x/
      private/
        ...
```

### Notes

* `private/`: private client data for training/validation
* `public/`: public data for representation extraction or contrastive learning
* `*_text.npy`: text feature file
* `*_seg.nii.gz`: segmentation mask

---

## Mask Encoding

In the current implementation, BraTS masks are converted into three channels:

* `channel0 = background`
* `channel1 = WT`
* `channel2 = ET`

The validation pipeline currently evaluates **WT** by default for Dice / IoU / HD95.

---

## Default Experiment Config (GroupA)

The default config file is:

```text
configs/exp_group_a.yaml
```

### Key settings

* active client: `client_2`
* modality: `image_only`
* batch_size = 1
* accumulation_steps = 4
* effective_batch_size = 4
* learning_rate = 5e-5
* rounds = 60
* local_epochs = 1
* aggregation_method = `fedavg`
* lambda_cream = 0.0
* num_classes = 3
* device = cuda
* validation every round
* mask saving enabled

This is a controlled **image-only federated segmentation experiment** by default.

---

## Installation

A dedicated virtual environment is recommended.

### Install dependencies

```bash
pip install -r requirements.txt
```

### For strict reproducibility

```bash
pip install -r requirements-lock.txt
```

---

## Main Dependencies

* torch
* torchvision
* numpy
* timm
* tensorboard
* monai
* nibabel
* matplotlib
* scikit-image
* scikit-learn
* pandas
* pyyaml
* fvcore
* fairscale
* hydra-core

---

## Running the Project

### 1. Quick smoke test

```bash
python main.py --config configs/exp_group_a.yaml --rounds 1 --use_mock
```

---

### 2. Standard run

```bash
python main.py --config configs/exp_group_a.yaml
```

---

### 3. Common CLI overrides

```bash
python main.py \
  --config configs/exp_group_a.yaml \
  --data_root data/federated_split \
  --rounds 10 \
  --batch_size 1 \
  --lr 5e-5 \
  --device cuda
```

### Main CLI options

* `--config`
* `--data_root`
* `--rounds`
* `--batch_size`
* `--lr`
* `--lambda_cream`
* `--use_mock`
* `--device`

---

## Outputs

### 1. Checkpoints

Located at:

```text
<log_dir>/checkpoints/
```

Includes:

* `checkpoint_round_*.pth`
* `latest_checkpoint.pth`
* `best_model.pth`
* `final_model.pth`

---

### 2. Training history

Exported as:

```text
training_history.json
```

Records:

* per-round training losses
* segmentation / cream losses
* validation metrics
* learning-rate history
* GPU peak memory
* per-round time cost
* gradient conflict angle

---

### 3. Plots

Automatically generated:

* `loss_curve.png`
* `metrics_dice_iou.png`
* `metrics_hd95.png`

---

### 4. Segmentation results

If `save_masks` is enabled:

* predicted masks
* ground-truth masks

---

## Validation Metrics

Supported metrics:

* Dice
* IoU
* HD95

### Characteristics

* global pixel accumulation over the whole validation set
* Dice / IoU computed once after full validation
* HD95 computed only when GT is non-empty
* WT channel used by default

---

## Resume Training

To resume training, set:

```yaml
checkpoint:
  resume_from_checkpoint: path/to/checkpoint.pth
```

The trainer restores:

* server state
* training history
* client states

---

## Important Notes Before Use

### 1. Real training requires a SAM3 checkpoint

Example:

```text
data/checkpoints/sam3.pt
```

If unavailable, use `--use_mock`.

---

### 2. External dependencies

The project dynamically appends:

```text
core_projects/sam3-main
core_projects/SAM-Adapter-PyTorch-main
core_projects/CreamFL-main/src
core_projects/FedFMS-main
```

This repository is **not fully self-contained**.

---

### 3. Data is not included

Not provided:

* raw medical images
* text features
* logs
* trained checkpoints

---

## Recommended Usage Order

### Step 1: Verify pipeline

```bash
python main.py --config configs/exp_group_a.yaml --rounds 1 --use_mock
```

### Step 2: Run with real data

```bash
python main.py --config configs/exp_group_a.yaml
```

### Step 3: Extend to heterogeneous experiments

Modify:

* `config.clients`
* data splits
* `lambda_cream`
* `aggregation_method`

---

## Summary

SegSAM3 provides a strong experimental backbone for research:

* SAM3-based medical segmentation
* lightweight Adapter tuning
* heterogeneous client training
* modality-decoupled aggregation
* global representation maintenance
* medical segmentation validation and diagnostics

---

## Acknowledgements

This repository is inspired by:

* SAM-style segmentation models
* Adapter / PEFT methods
* CreamFL-style aggregation
* heterogeneous federated learning for medical imaging

---

## Citation

If you use this repository in research or coursework, please cite the relevant foundational works and acknowledge this implementation.

```
```
