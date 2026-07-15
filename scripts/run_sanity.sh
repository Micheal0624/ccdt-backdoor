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
  --epochs 40
