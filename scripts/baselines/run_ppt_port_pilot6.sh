#!/usr/bin/env bash
set -u
set -o pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SCRIPT=${ROOT}/external_ports/ppt_port/train_ppt_port.py
OUT=${ROOT}/paper_results/external_baselines/ppt_port_pilot6
LOG=${ROOT}/logs_external_baselines/ppt_port_pilot6
STATUS=${OUT}/ppt_port_pilot6_status.csv

mkdir -p "$OUT" "$LOG"

echo "dataset,model,poison_rate,seed,status,log_path,summary_path,start_time,end_time" > "$STATUS"

run_one () {
  DATASET="$1"
  MODEL="$2"
  PR="$3"
  SEED="$4"

  TAG="ppt_port_${DATASET}_${MODEL}_pr005_seed${SEED}"
  RUN_OUT="${OUT}/${TAG}"
  LOG_PATH="${LOG}/${TAG}.out"
  SUMMARY_PATH="${RUN_OUT}/summary.json"
  START=$(date "+%Y-%m-%d %H:%M:%S")

  mkdir -p "$RUN_OUT"

  echo
  echo "===================================================================================================="
  echo "START $TAG"
  echo "===================================================================================================="

  set +e
  python "$SCRIPT" \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --poison-rate "$PR" \
    --seed "$SEED" \
    --clean-epochs 1 \
    --poison-epochs 1 \
    --batch-size 128 \
    --num-workers 2 \
    --clean-lr 0.01 \
    --poison-lr 0.01 \
    --pgd-steps 3 \
    --out-dir "$RUN_OUT" \
    2>&1 | tee "$LOG_PATH"
  CODE=${PIPESTATUS[0]}
  set -e

  if [ "$CODE" -eq 0 ] && [ -f "$SUMMARY_PATH" ]; then
    STATUS_STR="DONE"
  else
    STATUS_STR="FAILED"
  fi

  END=$(date "+%Y-%m-%d %H:%M:%S")
  echo "${DATASET},${MODEL},${PR},${SEED},${STATUS_STR},${LOG_PATH},${SUMMARY_PATH},${START},${END}" >> "$STATUS"
  echo "END $TAG status=$STATUS_STR"
}

run_one cifar10  resnet18 0.05 0
run_one cifar10  vgg11    0.05 0
run_one cifar100 resnet18 0.05 0
run_one cifar100 vgg11    0.05 0
run_one gtsrb    resnet18 0.05 0
run_one gtsrb    vgg11    0.05 0

echo
echo "========== STATUS =========="
cat "$STATUS"
