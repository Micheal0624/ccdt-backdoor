#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

OUT_ROOT=paper_results/clean_reference/clean_ref_preflight
LOG_ROOT=logs_clean_reference/clean_ref_preflight
STATUS=$OUT_ROOT/clean_ref_preflight_status.csv

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

echo "dataset,model,status,log_path,start_time,end_time" > "$STATUS"

DATASETS=("cifar10" "cifar100" "gtsrb")
MODELS=("resnet18" "vgg11")

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    RUN_NAME="clean_ref_preflight_${dataset}_${model}"
    RUN_DIR="$OUT_ROOT/$RUN_NAME"
    LOG_PATH="$LOG_ROOT/${RUN_NAME}.out"

    mkdir -p "$RUN_DIR"

    START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[START] $RUN_NAME"

    python external_ports/clean_ref/train_clean_ref.py \
      --dataset "$dataset" \
      --model "$model" \
      --seed 0 \
      --epochs 1 \
      --batch-size 16 \
      --num-workers 2 \
      --out-dir "$RUN_DIR" \
      --dry-run \
      > "$LOG_PATH" 2>&1

    EXIT_CODE=$?
    END_TIME=$(date "+%Y-%m-%d %H:%M:%S")

    if [ "$EXIT_CODE" -eq 0 ] && grep -q "CLEAN_REF_PREFLIGHT_OK" "$LOG_PATH"; then
      STATUS_STR="DONE"
    else
      STATUS_STR="FAILED"
    fi

    echo "${dataset},${model},${STATUS_STR},${ROOT}/${LOG_PATH},${START_TIME},${END_TIME}" >> "$STATUS"
    echo "[${STATUS_STR}] $RUN_NAME exit_code=$EXIT_CODE"

    if [ "$STATUS_STR" != "DONE" ]; then
      echo "[ERROR] See $LOG_PATH"
      exit 1
    fi
  done
done

echo "CLEAN_REF_PREFLIGHT_DONE"
