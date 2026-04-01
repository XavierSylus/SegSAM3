# SegSAM3 GroupA (Thesis Release)

This repository is a clean release for undergraduate thesis submission and demo.
It only contains code needed for **GroupA** training, validation, and reproduction.

## Included

- Entry point: `main.py`
- GroupA config: `configs/exp_group_a.yaml`
- Core training implementation: `src/`
- Data-processing source code (code only):
  - `data/*.py`
  - `data/data_allocation.json`
- GroupA utility scripts:
  - `scripts/setup_serial_clients.py`
  - `scripts/serial_training_utils.py`
  - `scripts/generate_groupa_color_figures.py`
- GroupA validation test:
  - `tests/test_validation_wt_channel.py`
- Environment files:
  - `requirements.txt`
  - `requirements-lock.txt`

## Excluded

- Raw datasets and patient data
- Checkpoints, logs, and experiment artifacts
- Non-GroupA project assets

## Environment Setup

```bash
pip install -r requirements.txt
```

For strict reproducibility:

```bash
pip install -r requirements-lock.txt
```

## Data Layout

Data is not bundled in this repository.
Expected structure:

```text
data/
  federated_split/
    train/
    val/
    client2_image_only/
      dataset.json
```

## Run GroupA

Quick smoke run:

```bash
python main.py --config configs/exp_group_a.yaml --rounds 1 --use_mock
```

Standard run:

```bash
python main.py --config configs/exp_group_a.yaml
```

## Reproducibility Checklist

Before final thesis submission, archive these items:

- Random seed settings
- Runtime environment (Python/CUDA/package versions)
- Git commit hash used in experiments

Commands:

```bash
python -V
pip freeze > requirements-lock.txt
git rev-parse HEAD
```
