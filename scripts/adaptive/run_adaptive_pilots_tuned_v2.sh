#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(cd "$(dirname "$0")" && pwd)"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

STAMP=$(date +%Y%m%d_%H%M%S)

pick_config () {
  DATASET=$1
  MODEL=$2

  CFG=$(find configs -type f \( -name "*.yaml" -o -name "*.yml" -o -name "*.json" \) 2>/dev/null \
    | grep -i "${DATASET}" \
    | grep -i "${MODEL}" \
    | head -1 || true)

  if [ -z "${CFG}" ]; then
    echo "[ERROR] 找不到 config: dataset=${DATASET}, model=${MODEL}" >&2
    echo "当前 configs 文件如下：" >&2
    find configs -type f 2>/dev/null | sort >&2 || true
    exit 2
  fi

  echo "${CFG}"
}

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

run_one () {
  DATASET=$1
  MODEL=$2
  TAG=$3
  CFG=$(pick_config "${DATASET}" "${MODEL}")
  LOG="logs_adaptive_pilot_tuned/${TAG}_${STAMP}.out"

  echo "========== Running ${DATASET} ${MODEL} ${TAG} =========="
  echo "Config: ${CFG}"
  echo "Log: ${LOG}"

  python -m src.train_adaptive \
    --config "${CFG}" \
    ${COMMON_ARGS} \
    --run-tag "${TAG}_${STAMP}" \
    2>&1 | tee "${LOG}"
}

run_one cifar10  vgg11    pilot_tuned_c10_vgg11_kc8_pr001
run_one gtsrb    vgg11    pilot_tuned_gtsrb_vgg11_kc8_pr001
run_one cifar100 vgg11    pilot_tuned_c100_vgg11_kc8_pr001
run_one cifar10  resnet18 pilot_tuned_c10_resnet18_kc8_pr001

echo "========== Tuned pilots finished =========="
