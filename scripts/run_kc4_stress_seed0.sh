#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):$PYTHONPATH"

CONFIG=configs/cifar10_resnet18.yaml
POISON=0.05
SEED=0
KC=4
EPOCHS=40

for METHOD in single naive_dual wo_invalid full
do
  echo "============================================================"
  echo "Running Kc=${KC}, seed=${SEED}, method=${METHOD}"
  echo "============================================================"

  python -m src.train \
    --config ${CONFIG} \
    --method ${METHOD} \
    --poison-rate ${POISON} \
    --seed ${SEED} \
    --kc ${KC} \
    --epochs ${EPOCHS} \
    2>&1 | tee logs/kc4_cifar10_resnet18_${METHOD}_pr005_seed0.log
done
