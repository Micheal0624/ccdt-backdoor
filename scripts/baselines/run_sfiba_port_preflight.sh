#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

OUT_ROOT=paper_results/external_baselines/sfiba_port_preflight
LOG_ROOT=logs_external_baselines/sfiba_port_preflight
STATUS=$OUT_ROOT/sfiba_port_preflight_status.csv

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

echo "dataset,model,status,log_path,summary_path,start_time,end_time" > "$STATUS"

DATASETS=("cifar10" "cifar100" "gtsrb")
MODELS=("resnet18" "vgg11")

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    RUN_NAME="sfiba_preflight_${dataset}_${model}"
    RUN_DIR="$OUT_ROOT/$RUN_NAME"
    LOG_PATH="$LOG_ROOT/${RUN_NAME}.out"
    SUMMARY_PATH="$RUN_DIR/summary.json"

    mkdir -p "$RUN_DIR"

    START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[START] $RUN_NAME"

    python external_ports/sfiba_port/train_sfiba_port.py \
      --dataset "$dataset" \
      --model "$model" \
      --poison-rate 0.05 \
      --seed 0 \
      --epochs 1 \
      --batch-size 16 \
      --num-workers 2 \
      --alpha 0.15 \
      --eps 0.03137254901960784 \
      --patch-size 8 \
      --out-dir "$RUN_DIR" \
      --dry-run \
      > "$LOG_PATH" 2>&1

    EXIT_CODE=$?
    END_TIME=$(date "+%Y-%m-%d %H:%M:%S")

    if [ "$EXIT_CODE" -eq 0 ] && grep -q "SFIBA_PORT_PREFLIGHT_OK" "$LOG_PATH"; then
      STATUS_STR="DONE"
    else
      STATUS_STR="FAILED"
    fi

    echo "${dataset},${model},${STATUS_STR},${ROOT}/${LOG_PATH},${ROOT}/${SUMMARY_PATH},${START_TIME},${END_TIME}" >> "$STATUS"
    echo "[${STATUS_STR}] $RUN_NAME exit_code=$EXIT_CODE"

    if [ "$STATUS_STR" != "DONE" ]; then
      echo "[ERROR] See $LOG_PATH"
      exit 1
    fi
  done
done

echo "SFIBA_PORT_PREFLIGHT_DONE"
