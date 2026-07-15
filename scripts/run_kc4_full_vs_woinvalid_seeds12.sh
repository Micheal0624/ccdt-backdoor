#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):$PYTHONPATH"

CONFIG=configs/cifar10_resnet18.yaml
POISON=0.05
KC=4
EPOCHS=40

for SEED in 1 2
do
  for METHOD in wo_invalid full
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
      2>&1 | tee logs/kc4_cifar10_resnet18_${METHOD}_pr005_seed${SEED}.log
  done
done
