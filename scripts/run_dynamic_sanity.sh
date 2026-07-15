#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):$PYTHONPATH"

python -m src.train \
  --config configs/cifar10_resnet18.yaml \
  --method full \
  --poison-rate 0.05 \
  --seed 0 \
  --kc 2 \
  --position-mode dynamic \
  --search-grid 4 \
  --search-subset 2048 \
  --epochs 40 \
  2>&1 | tee logs/dynamic_sanity_cifar10_resnet18_full_pr005_seed0.log
