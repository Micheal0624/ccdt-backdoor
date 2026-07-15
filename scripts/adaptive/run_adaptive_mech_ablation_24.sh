#!/usr/bin/env bash
set -u
set -o pipefail

export PYTHONPATH="$(cd "$(dirname "$0")" && pwd)"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${ROOT}/logs_adaptive_mech_ablation"
OUT_DIR="${ROOT}/paper_results/adaptive_mech_ablation"
STATUS_CSV="${OUT_DIR}/adaptive_mech_ablation_24_status.csv"
PLAN_CSV="${OUT_DIR}/adaptive_mech_ablation_24_plan.csv"

mkdir -p "${LOG_DIR}" "${OUT_DIR}"

echo "variant,dataset,model,config,kc,poison_rate,seed,run_tag" > "${PLAN_CSV}"
echo "variant,dataset,model,kc,poison_rate,seed,run_tag,status,exit_code,summary_path,log_path,start_time,end_time" > "${STATUS_CSV}"

COMMON_BASE=(
  --kc 8
  --poison-rate 0.01
  --position-mode dynamic
  --epochs 40
  --search-grid 6
  --search-subset 2048
  --feedback-size 2048
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
)

run_one () {
  VARIANT="$1"
  DATASET="$2"
  MODEL="$3"
  CONFIG="$4"
  SEED="$5"

  KC=8
  PR=0.01
  RUN_TAG="mech24_${VARIANT}_${DATASET}_${MODEL}_kc8_pr001_seed${SEED}"
  LOG_PATH="${LOG_DIR}/${RUN_TAG}.out"

  echo "${VARIANT},${DATASET},${MODEL},${CONFIG},${KC},${PR},${SEED},${RUN_TAG}" >> "${PLAN_CSV}"

  EXISTING_SUMMARY=$(find "${ROOT}/results" -maxdepth 2 -path "*${RUN_TAG}*/summary.json" -print -quit 2>/dev/null || true)
  if [ -n "${EXISTING_SUMMARY}" ]; then
    NOW=$(date "+%Y-%m-%d %H:%M:%S")
    echo "========== SKIP existing ${RUN_TAG} =========="
    echo "${VARIANT},${DATASET},${MODEL},${KC},${PR},${SEED},${RUN_TAG},SKIP_EXISTING,0,${EXISTING_SUMMARY},${LOG_PATH},${NOW},${NOW}" >> "${STATUS_CSV}"
    return 0
  fi

  START_TIME=$(date "+%Y-%m-%d %H:%M:%S")

  echo
  echo "===================================================================================================="
  echo "START ${RUN_TAG}"
  echo "Config: ${CONFIG}"
  echo "Start: ${START_TIME}"
  echo "Log: ${LOG_PATH}"
  echo "===================================================================================================="

  if [ "${VARIANT}" = "equal" ]; then
    MODE_ARGS=(--adaptive-mode equal --feedback-ema 0.9)
  elif [ "${VARIANT}" = "no_ema" ]; then
    MODE_ARGS=(--adaptive-mode feedback --feedback-ema 0.0)
  else
    echo "[ERROR] Unknown variant: ${VARIANT}"
    return 2
  fi

  set +e
  python -m src.train_adaptive \
    --config "${CONFIG}" \
    --seed "${SEED}" \
    "${COMMON_BASE[@]}" \
    "${MODE_ARGS[@]}" \
    --run-tag "${RUN_TAG}" \
    2>&1 | tee "${LOG_PATH}"
  EXIT_CODE=${PIPESTATUS[0]}
  set -e

  END_TIME=$(date "+%Y-%m-%d %H:%M:%S")
  SUMMARY_PATH=$(find "${ROOT}/results" -maxdepth 2 -path "*${RUN_TAG}*/summary.json" -print -quit 2>/dev/null || true)

  if [ "${EXIT_CODE}" -eq 0 ] && [ -n "${SUMMARY_PATH}" ]; then
    STATUS="DONE"
  elif [ "${EXIT_CODE}" -eq 0 ] && [ -z "${SUMMARY_PATH}" ]; then
    STATUS="NO_SUMMARY"
  else
    STATUS="FAILED"
  fi

  echo "===================================================================================================="
  echo "END ${RUN_TAG}"
  echo "Status: ${STATUS}"
  echo "Exit code: ${EXIT_CODE}"
  echo "Summary: ${SUMMARY_PATH}"
  echo "End: ${END_TIME}"
  echo "===================================================================================================="

  echo "${VARIANT},${DATASET},${MODEL},${KC},${PR},${SEED},${RUN_TAG},${STATUS},${EXIT_CODE},${SUMMARY_PATH},${LOG_PATH},${START_TIME},${END_TIME}" >> "${STATUS_CSV}"
}

echo "===================================================================================================="
echo "Adaptive mechanism ablation 24 started"
echo "===================================================================================================="

for VARIANT in equal no_ema; do
  for SEED in 0 1 2; do
    run_one "${VARIANT}" cifar10  vgg11    configs/cifar10_vgg11.yaml     "${SEED}"
    run_one "${VARIANT}" gtsrb    vgg11    configs/gtsrb_vgg11.yaml       "${SEED}"
    run_one "${VARIANT}" cifar100 vgg11    configs/cifar100_vgg11.yaml    "${SEED}"
    run_one "${VARIANT}" cifar10  resnet18 configs/cifar10_resnet18.yaml  "${SEED}"
  done
done

echo
echo "===================================================================================================="
echo "Adaptive mechanism ablation 24 finished"
echo "Status CSV: ${STATUS_CSV}"
echo "Plan CSV: ${PLAN_CSV}"
echo "===================================================================================================="

python - <<'PY'
import csv
from collections import Counter
p = "paper_results/adaptive_mech_ablation/adaptive_mech_ablation_24_status.csv"
with open(p, newline="") as f:
    rows = list(csv.DictReader(f))
print("total rows:", len(rows))
print(Counter(r["status"] for r in rows))
bad = [r for r in rows if r["status"] not in ("DONE", "SKIP_EXISTING")]
if bad:
    print("bad rows:")
    for r in bad:
        print(r["variant"], r["dataset"], r["model"], "seed", r["seed"], r["status"], r["log_path"])
PY
