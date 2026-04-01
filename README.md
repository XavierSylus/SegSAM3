# SegSAM3

[简体中文](./README_zh.md) | [English](./README_en.md)

> 基于 **SAM3 + Adapter + Federated Aggregation** 的异构多模态医学图像分割实验框架  
> A research-oriented framework for **heterogeneous multimodal federated medical image segmentation** based on **SAM3 + Adapter + federated aggregation**.

---

## Overview

SegSAM3 is a research-oriented codebase for federated medical image segmentation with support for:

- `text_only` clients
- `image_only` clients
- `multimodal` clients
- SAM3-based segmentation backbone
- adapter-based lightweight tuning
- federated training, validation, checkpointing, plotting, and mask export

当前默认配置 `configs/exp_group_a.yaml` 对应一个 **GroupA image-only** 受控实验；底层代码已支持扩展到异构多模态联邦学习。

The default config `configs/exp_group_a.yaml` corresponds to a controlled **GroupA image-only** experiment, while the underlying code already supports extensions to heterogeneous multimodal federated learning.

---

## Quick Start

### Smoke test

```bash id="m8n4pm"
python main.py --config configs/exp_group_a.yaml --rounds 1 --use_mock
```

### Standard run

```bash id="m8n4pm"
python main.py --config configs/exp_group_a.yaml
```
