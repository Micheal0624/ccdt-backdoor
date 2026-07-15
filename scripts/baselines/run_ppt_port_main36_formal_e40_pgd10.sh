#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

TRAIN=external_ports/ppt_port/train_ppt_port.py

OUT_ROOT=paper_results/external_baselines/ppt_port_main36_formal_e40_pgd10
LOG_ROOT=logs_external_baselines/ppt_port_main36_formal_e40_pgd10
STATUS=$OUT_ROOT/ppt_port_main36_formal_e40_pgd10_status.csv

CLEAN_EPOCHS=40
POISON_EPOCHS=40
PGD_STEPS=10
BATCH_SIZE=128
NUM_WORKERS=4

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

if [ ! -f "$TRAIN" ]; then
  echo "[FATAL] Missing train script: $TRAIN"
  exit 1
fi

if [ ! -f "$STATUS" ]; then
  echo "dataset,model,poison_rate,seed,clean_epochs,poison_epochs,pgd_steps,batch_size,num_workers,status,log_path,summary_path,start_time,end_time" > "$STATUS"
fi

DATASETS=("cifar10" "cifar100" "gtsrb")
MODELS=("resnet18" "vgg11")
SEEDS=("0" "1" "2")
PRS=("0.05" "0.01")

for pr in "${PRS[@]}"; do
  if [ "$pr" = "0.05" ]; then
    PR_TAG="005"
  elif [ "$pr" = "0.01" ]; then
    PR_TAG="001"
  else
    PR_TAG=$(echo "$pr" | sed 's/\.//g')
  fi

  for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
      for seed in "${SEEDS[@]}"; do

        RUN_NAME="ppt_port_${dataset}_${model}_pr${PR_TAG}_seed${seed}_e40_pgd10"
        RUN_DIR="$OUT_ROOT/$RUN_NAME"
        LOG_PATH="$LOG_ROOT/${RUN_NAME}.out"
        SUMMARY_PATH="$RUN_DIR/summary.json"

        if grep -q "^${dataset},${model},${pr},${seed},${CLEAN_EPOCHS},${POISON_EPOCHS},${PGD_STEPS},${BATCH_SIZE},${NUM_WORKERS},DONE," "$STATUS"; then
          echo "[SKIP DONE] $RUN_NAME"
          continue
        fi

        if [ -e "$RUN_DIR" ]; then
          echo "[FATAL] Existing run dir, refusing to overwrite: $RUN_DIR"
          exit 1
        fi

        mkdir -p "$RUN_DIR"

        START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
        echo "[START] $RUN_NAME at $START_TIME"

        python "$TRAIN" \
          --dataset "$dataset" \
          --model "$model" \
          --poison-rate "$pr" \
          --seed "$seed" \
          --clean-epochs "$CLEAN_EPOCHS" \
          --poison-epochs "$POISON_EPOCHS" \
          --pgd-steps "$PGD_STEPS" \
          --batch-size "$BATCH_SIZE" \
          --num-workers "$NUM_WORKERS" \
          --out-dir "$RUN_DIR" \
          > "$LOG_PATH" 2>&1

        EXIT_CODE=$?
        END_TIME=$(date "+%Y-%m-%d %H:%M:%S")

        if [ "$EXIT_CODE" -eq 0 ] && [ -f "$SUMMARY_PATH" ]; then
          STATUS_STR="DONE"
        else
          STATUS_STR="FAILED"
        fi

        echo "${dataset},${model},${pr},${seed},${CLEAN_EPOCHS},${POISON_EPOCHS},${PGD_STEPS},${BATCH_SIZE},${NUM_WORKERS},${STATUS_STR},${ROOT}/${LOG_PATH},${ROOT}/${SUMMARY_PATH},${START_TIME},${END_TIME}" >> "$STATUS"
        echo "[${STATUS_STR}] $RUN_NAME exit_code=$EXIT_CODE at $END_TIME"

        if [ "$STATUS_STR" != "DONE" ]; then
          echo "[ERROR] $RUN_NAME failed. See log:"
          echo "$LOG_PATH"
          exit 1
        fi

      done
    done
  done
done

echo "PPT_PORT_MAIN36_FORMAL_E40_PGD10_DONE"
