#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

OUT_ROOT=paper_results/clean_reference/clean_ref_main18_e40
LOG_ROOT=logs_clean_reference/clean_ref_main18_e40
STATUS=$OUT_ROOT/clean_ref_main18_e40_status.csv

EPOCHS="40"
BATCH_SIZE="128"
NUM_WORKERS="4"
LR="0.01"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

if [ ! -f "$STATUS" ]; then
  echo "dataset,model,seed,epochs,batch_size,num_workers,lr,status,log_path,summary_path,start_time,end_time" > "$STATUS"
fi

DATASETS=("cifar10" "cifar100" "gtsrb")
MODELS=("resnet18" "vgg11")
SEEDS=("0" "1" "2")

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do

      RUN_NAME="clean_ref_${dataset}_${model}_seed${seed}_e40"
      RUN_DIR="$OUT_ROOT/$RUN_NAME"
      LOG_PATH="$LOG_ROOT/${RUN_NAME}.out"
      SUMMARY_PATH="$RUN_DIR/summary.json"

      if grep -q "^${dataset},${model},${seed},${EPOCHS},${BATCH_SIZE},${NUM_WORKERS},${LR},DONE," "$STATUS"; then
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

      python external_ports/clean_ref/train_clean_ref.py \
        --dataset "$dataset" \
        --model "$model" \
        --seed "$seed" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --lr "$LR" \
        --out-dir "$RUN_DIR" \
        > "$LOG_PATH" 2>&1

      EXIT_CODE=$?
      END_TIME=$(date "+%Y-%m-%d %H:%M:%S")

      if [ "$EXIT_CODE" -eq 0 ] && [ -f "$SUMMARY_PATH" ] && grep -q "DONE" "$LOG_PATH"; then
        STATUS_STR="DONE"
      else
        STATUS_STR="FAILED"
      fi

      echo "${dataset},${model},${seed},${EPOCHS},${BATCH_SIZE},${NUM_WORKERS},${LR},${STATUS_STR},${ROOT}/${LOG_PATH},${ROOT}/${SUMMARY_PATH},${START_TIME},${END_TIME}" >> "$STATUS"
      echo "[${STATUS_STR}] $RUN_NAME exit_code=$EXIT_CODE at $END_TIME"

      if [ "$STATUS_STR" != "DONE" ]; then
        echo "[ERROR] $RUN_NAME failed. See log:"
        echo "$LOG_PATH"
        exit 1
      fi

    done
  done
done

echo "CLEAN_REF_MAIN18_E40_DONE"
