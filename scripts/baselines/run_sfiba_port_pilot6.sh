#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

OUT_ROOT=paper_results/external_baselines/sfiba_port_pilot6
LOG_ROOT=logs_external_baselines/sfiba_port_pilot6
STATUS=$OUT_ROOT/sfiba_port_pilot6_status.csv

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

if [ ! -f "$STATUS" ]; then
  echo "dataset,model,poison_rate,seed,epochs,alpha,eps,patch_size,status,log_path,summary_path,start_time,end_time" > "$STATUS"
fi

DATASETS=("cifar10" "cifar100" "gtsrb")
MODELS=("resnet18" "vgg11")

PR="0.05"
SEED="0"
EPOCHS="1"
ALPHA="0.15"
EPS="0.03137254901960784"
PATCH_SIZE="8"

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do

    RUN_NAME="sfiba_port_${dataset}_${model}_pr005_seed0_pilot"
    RUN_DIR="$OUT_ROOT/$RUN_NAME"
    LOG_PATH="$LOG_ROOT/${RUN_NAME}.out"
    SUMMARY_PATH="$RUN_DIR/summary.json"

    if grep -q "^${dataset},${model},${PR},${SEED},${EPOCHS},${ALPHA},${EPS},${PATCH_SIZE},DONE," "$STATUS"; then
      echo "[SKIP DONE] $RUN_NAME"
      continue
    fi

    if [ -e "$RUN_DIR" ] && [ ! -f "$SUMMARY_PATH" ]; then
      echo "[FATAL] Existing incomplete run dir, refusing to overwrite: $RUN_DIR"
      exit 1
    fi

    if [ -f "$SUMMARY_PATH" ]; then
      echo "[FATAL] Existing summary without DONE status, refusing to reuse: $SUMMARY_PATH"
      exit 1
    fi

    mkdir -p "$RUN_DIR"

    START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[START] $RUN_NAME at $START_TIME"

    python external_ports/sfiba_port/train_sfiba_port.py \
      --dataset "$dataset" \
      --model "$model" \
      --poison-rate "$PR" \
      --seed "$SEED" \
      --epochs "$EPOCHS" \
      --batch-size 128 \
      --num-workers 4 \
      --lr 0.01 \
      --alpha "$ALPHA" \
      --eps "$EPS" \
      --patch-size "$PATCH_SIZE" \
      --out-dir "$RUN_DIR" \
      > "$LOG_PATH" 2>&1

    EXIT_CODE=$?
    END_TIME=$(date "+%Y-%m-%d %H:%M:%S")

    if [ "$EXIT_CODE" -eq 0 ] && [ -f "$SUMMARY_PATH" ] && grep -q "DONE" "$LOG_PATH"; then
      STATUS_STR="DONE"
    else
      STATUS_STR="FAILED"
    fi

    echo "${dataset},${model},${PR},${SEED},${EPOCHS},${ALPHA},${EPS},${PATCH_SIZE},${STATUS_STR},${ROOT}/${LOG_PATH},${ROOT}/${SUMMARY_PATH},${START_TIME},${END_TIME}" >> "$STATUS"
    echo "[${STATUS_STR}] $RUN_NAME exit_code=$EXIT_CODE at $END_TIME"

    if [ "$STATUS_STR" != "DONE" ]; then
      echo "[ERROR] $RUN_NAME failed. See log:"
      echo "$LOG_PATH"
      exit 1
    fi

  done
done

echo "SFIBA_PORT_PILOT6_DONE"
