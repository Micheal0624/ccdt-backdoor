# CCDT Backdoor Attack — Reproduction Code (Anonymous Release)

> **Anonymous release.** This repository contains no author identities, affiliations, or
> private data. Absolute machine paths, cloud-environment paths, and training logs have been
> removed. It is intended for anonymous sharing alongside the paper submission.

## What is CCDT

CCDT is a backdoor attack against image classifiers designed to remain effective even when
standard backdoor **defenses** are applied. This repository provides the code, configurations,
run commands, evaluation/metric code, and processed results needed to reproduce the paper's
experiments. Experiments cover **CIFAR-10, CIFAR-100 and GTSRB** with **ResNet-18** and
**VGG-11**, and evaluate the attack against four defenses: **STRIP**, **Spectral Signatures**,
**Fine-Pruning**, and **Neural Cleanse**.

## Repository layout

```
ccdt_backdoor/
├── README.md              # this file
├── requirements.txt       # Python dependencies
├── src/                   # 1) core method implementation
├── configs/               # 2) experiment configurations (dataset × architecture)
├── scripts/               # 3) run commands (shell scripts)
│   ├── *.sh               #    main CCDT experiment grids
│   ├── adaptive/          #    adaptive-attack variant runners
│   └── baselines/         #    baseline-port & clean-reference runners
├── tools/                 # 4) evaluation / aggregation / figure code
│   └── figures/           #    figure-generation scripts (final versions)
└── results/               # 5) processed results (no raw per-run dumps)
    ├── tables/            #    aggregated metric tables (CSV / TXT)
    ├── figures/           #    paper figures (PNG / PDF)
    └── paper_results/     #    curated per-experiment summaries
```

### Where the key pieces live

| Paper artifact | Location |
|----------------|----------|
| **Core attack implementation** | `src/` — `models.py`, `triggers.py`, `poison.py`, `datasets.py`, `train.py`, `train_adaptive.py`, `adaptive_loss.py`, `config_search.py` |
| **Metric code** (ASR / clean-acc / CSG) | `src/evaluate.py` |
| **Defense implementations + evaluation** | `tools/run_defense_eval.py` (STRIP, Neural Cleanse, Spectral Signatures, Fine-Pruning) |
| **Result aggregation** | `tools/aggregate_*.py`, `tools/reevaluate_*.py`, `tools/evaluate_*_out_of_library_invalid.py` |
| **Figure generation** | `tools/figures/*.py` |
| **Configurations & run commands** | `configs/*.yaml`, `scripts/**/*.sh` |
| **Processed results** | `results/tables/`, `results/figures/`, `results/paper_results/` |

## Installation

```bash
pip install -r requirements.txt
```

The code is built on **PyTorch**. Install a CUDA-compatible `torch` / `torchvision` build that
matches your environment (see https://pytorch.org). Remaining dependencies: numpy, pandas,
matplotlib, scikit-learn, tqdm, pyyaml, opencv-python, pillow, tensorboard.

## Datasets & weights (NOT included)

Per the data-sharing policy, **datasets and trained weights are not included**:

- **Datasets** (CIFAR-10 / CIFAR-100 / GTSRB): download them and set the `data_root` field in
  `configs/*.yaml` to your local copy.
- **Weights**: all reported numbers are produced by running the code below. `output_root` /
  `checkpoint_root` in the configs default to `./results` and `./checkpoints` under the repo.

Every reported metric is already available in `results/tables/` and `results/paper_results/`.

## Usage

All scripts resolve the repository root relative to their own location, so they run from any
machine or path.

```bash
# Train a poisoned model (CIFAR-10, ResNet-18, full method, poison rate 0.05, seed 0, kc=2)
python -m src.train --config configs/cifar10_resnet18.yaml \
    --method full --poison-rate 0.05 --seed 0 --kc 2

# Evaluate a defense on a trained model
python tools/run_defense_eval.py --dataset cifar10 --model resnet18 --method full \
    --poison-rate 0.05 --seed 0 --kc 2 --ckpt-name last \
    --defense spectral_signatures --ss-samples 5000 --ss-remove-fraction 0.1
```

Convenience grid scripts under `scripts/` reproduce the experiment grids
(e.g. `scripts/run_main_cifar10_dynamic_grid.sh`, `scripts/run_defense_manifest_seed0.sh`).
`scripts/baselines/` covers clean-reference and external-baseline comparisons; the
external-baseline scripts assume the referenced external repositories are available locally.

### Key arguments

| Argument | Meaning |
|----------|---------|
| `--method` | `single` / `naive_dual` / `wo_invalid` / `full` |
| `--poison-rate` | fraction of training data poisoned |
| `--kc` | number of (target) classes involved |
| `--seed` | random seed |
| `--defense` | `strip` / `spectral_signatures` / `fine_pruning` / `neural_cleanse` |

## Results

- **Aggregated tables**: `results/tables/` — e.g. `scalability_runs.csv`,
  `static_dynamic_v2_runs.csv`, `out_of_library_invalid_full_last_runs.csv`, `anomaly_runs.csv`.
- **Figures**: `results/figures/` (main + appendix).
- **Per-experiment summaries**: `results/paper_results/` (clean-reference, adaptive, baseline
  comparison, hard-ablation, etc.).

Raw per-run output directories and training logs are intentionally omitted to keep the release
lightweight; the aggregated CSV/JSON artifacts above contain every number reported in the paper.

## Code availability

The code is available at: <https://anonymous.4open.science/r/ccdt-backdoor-68A6/>
