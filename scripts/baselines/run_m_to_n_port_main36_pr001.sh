#!/usr/bin/env bash
set -u
set -o pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SCRIPT=${ROOT}/external_ports/m_to_n_port/train_m_to_n_port.py
OUT=${ROOT}/paper_results/external_baselines/m_to_n_port_main36_pr001
LOG=${ROOT}/logs_external_baselines/m_to_n_port_main36_pr001
STATUS=${OUT}/m_to_n_port_main36_pr001_status.csv

mkdir -p "$OUT" "$LOG"

echo "dataset,model,poison_rate,n_targets,m_triggers,seed,status,log_path,summary_path,start_time,end_time" > "$STATUS"

run_one () {
  DATASET="$1"
  MODEL="$2"
  PR="$3"
  N_TARGETS="$4"
  M_TRIGGERS="$5"
  SEED="$6"

  TAG="m_to_n_port_${DATASET}_${MODEL}_pr001_N${N_TARGETS}_M${M_TRIGGERS}_seed${SEED}"
  RUN_OUT="${OUT}/${TAG}"
  LOG_PATH="${LOG}/${TAG}.out"
  SUMMARY_PATH="${RUN_OUT}/summary.json"
  START=$(date "+%Y-%m-%d %H:%M:%S")

  mkdir -p "$RUN_OUT"

  if [ -f "$SUMMARY_PATH" ]; then
    echo "SKIP existing $TAG"
    END=$(date "+%Y-%m-%d %H:%M:%S")
    echo "${DATASET},${MODEL},${PR},${N_TARGETS},${M_TRIGGERS},${SEED},DONE_EXISTING,${LOG_PATH},${SUMMARY_PATH},${START},${END}" >> "$STATUS"
    return 0
  fi

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
    --epochs 40 \
    --batch-size 128 \
    --num-workers 2 \
    --lr 0.01 \
    --alpha 0.2 \
    --m-triggers "$M_TRIGGERS" \
    --n-targets "$N_TARGETS" \
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
  echo "${DATASET},${MODEL},${PR},${N_TARGETS},${M_TRIGGERS},${SEED},${STATUS_STR},${LOG_PATH},${SUMMARY_PATH},${START},${END}" >> "$STATUS"
  echo "END $TAG status=$STATUS_STR"
}

for SEED in 0 1 2; do
  for N in 4 8; do
    run_one cifar10  resnet18 0.01 "$N" 2 "$SEED"
    run_one cifar10  vgg11    0.01 "$N" 2 "$SEED"
    run_one cifar100 resnet18 0.01 "$N" 2 "$SEED"
    run_one cifar100 vgg11    0.01 "$N" 2 "$SEED"
    run_one gtsrb    resnet18 0.01 "$N" 2 "$SEED"
    run_one gtsrb    vgg11    0.01 "$N" 2 "$SEED"
  done
done

echo
echo "========== STATUS =========="
cat "$STATUS"
