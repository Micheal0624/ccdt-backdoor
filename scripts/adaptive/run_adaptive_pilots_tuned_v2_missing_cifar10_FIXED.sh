#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(cd "$(dirname "$0")" && pwd)"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

STAMP=$(date +%Y%m%d_%H%M%S)

COMMON_ARGS="
  --kc 8
  --poison-rate 0.01
  --seed 0
  --position-mode dynamic
  --adaptive-mode feedback
  --epochs 40
  --search-grid 6
  --search-subset 2048
  --feedback-size 2048
  --feedback-ema 0.9
  --feedback-temperature 1.0
  --min-weight 0.25
  --max-weight 2.0
  --warmup-epochs 3
  --target-valid-asr 0.985
  --target-single-leak 0.02
  --target-invalid-leak 0.02
  --target-wrong-asr 0.01
  --clean-tolerance 0.005
  --asr-tolerance 0.01
  --leak-tolerance 0.02
  --wrong-tolerance 0.01
"

echo "========== Running CIFAR10 / VGG11 tuned v2 =========="
python -m src.train_adaptive \
  --config configs/cifar10_vgg11.yaml \
  ${COMMON_ARGS} \
  --run-tag "pilot_tuned_FIXED_cifar10_vgg11_kc8_pr001_${STAMP}" \
  2>&1 | tee "logs_adaptive_pilot_tuned/pilot_tuned_FIXED_cifar10_vgg11_kc8_pr001_${STAMP}.out"

echo "========== Running CIFAR10 / ResNet18 tuned v2 =========="
python -m src.train_adaptive \
  --config configs/cifar10_resnet18.yaml \
  ${COMMON_ARGS} \
  --run-tag "pilot_tuned_FIXED_cifar10_resnet18_kc8_pr001_${STAMP}" \
  2>&1 | tee "logs_adaptive_pilot_tuned/pilot_tuned_FIXED_cifar10_resnet18_kc8_pr001_${STAMP}.out"

echo "========== Correct CIFAR10 tuned v2 pilots finished =========="
