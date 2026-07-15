#!/usr/bin/env bash
set -u
set -o pipefail

export PYTHONPATH="$(cd "$(dirname "$0")" && pwd)"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${ROOT}/logs_adaptive_main72"
OUT_DIR="${ROOT}/paper_results/adaptive_main72_tuned_v2"
STATUS_CSV="${OUT_DIR}/adaptive_main72_tuned_v2_status.csv"
PLAN_CSV="${OUT_DIR}/adaptive_main72_tuned_v2_plan.csv"

mkdir -p "${LOG_DIR}" "${OUT_DIR}"

echo "dataset,model,config,kc,poison_rate,seed,run_tag" > "${PLAN_CSV}"
echo "dataset,model,kc,poison_rate,seed,run_tag,status,exit_code,summary_path,log_path,start_time,end_time" > "${STATUS_CSV}"

COMMON_ARGS=(
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
)

run_one () {
  DATASET="$1"
  MODEL="$2"
  CONFIG="$3"
  KC="$4"
  PR="$5"
  SEED="$6"

  if [ "${PR}" = "0.01" ]; then
    PRTAG="pr001"
  elif [ "${PR}" = "0.05" ]; then
    PRTAG="pr005"
  else
    PRTAG="pr${PR}"
  fi

  RUN_TAG="adaptive72_tuned_v2_${DATASET}_${MODEL}_kc${KC}_${PRTAG}_seed${SEED}"
  LOG_PATH="${LOG_DIR}/${RUN_TAG}.out"

  echo "${DATASET},${MODEL},${CONFIG},${KC},${PR},${SEED},${RUN_TAG}" >> "${PLAN_CSV}"

  EXISTING_SUMMARY=$(find "${ROOT}/results" -maxdepth 2 -path "*${RUN_TAG}*/summary.json" -print -quit 2>/dev/null || true)
  if [ -n "${EXISTING_SUMMARY}" ]; then
    NOW=$(date "+%Y-%m-%d %H:%M:%S")
    echo "========== SKIP existing ${RUN_TAG} =========="
    echo "summary: ${EXISTING_SUMMARY}"
    echo "${DATASET},${MODEL},${KC},${PR},${SEED},${RUN_TAG},SKIP_EXISTING,0,${EXISTING_SUMMARY},${LOG_PATH},${NOW},${NOW}" >> "${STATUS_CSV}"
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

  set +e
  python -m src.train_adaptive \
    --config "${CONFIG}" \
    --kc "${KC}" \
    --poison-rate "${PR}" \
    --seed "${SEED}" \
    "${COMMON_ARGS[@]}" \
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

  echo "${DATASET},${MODEL},${KC},${PR},${SEED},${RUN_TAG},${STATUS},${EXIT_CODE},${SUMMARY_PATH},${LOG_PATH},${START_TIME},${END_TIME}" >> "${STATUS_CSV}"
}

echo "===================================================================================================="
echo "Adaptive main72 tuned v2 started"
echo "Status CSV: ${STATUS_CSV}"
echo "Plan CSV: ${PLAN_CSV}"
echo "===================================================================================================="

for KC in 4 8; do
  for PR in 0.01 0.05; do
    for SEED in 0 1 2; do

      run_one cifar10  resnet18 configs/cifar10_resnet18.yaml  "${KC}" "${PR}" "${SEED}"
      run_one cifar10  vgg11    configs/cifar10_vgg11.yaml     "${KC}" "${PR}" "${SEED}"

      run_one cifar100 resnet18 configs/cifar100_resnet18.yaml "${KC}" "${PR}" "${SEED}"
      run_one cifar100 vgg11    configs/cifar100_vgg11.yaml    "${KC}" "${PR}" "${SEED}"

      run_one gtsrb    resnet18 configs/gtsrb_resnet18.yaml    "${KC}" "${PR}" "${SEED}"
      run_one gtsrb    vgg11    configs/gtsrb_vgg11.yaml       "${KC}" "${PR}" "${SEED}"

    done
  done
done

echo
echo "===================================================================================================="
echo "Adaptive main72 tuned v2 finished"
echo "Status CSV: ${STATUS_CSV}"
echo "Plan CSV: ${PLAN_CSV}"
echo "===================================================================================================="

echo
echo "========== Final status count =========="
python - <<'PY'
import csv
from collections import Counter
p = "paper_results/adaptive_main72_tuned_v2/adaptive_main72_tuned_v2_status.csv"
with open(p, newline="") as f:
    rows = list(csv.DictReader(f))
print("total rows:", len(rows))
print(Counter(r["status"] for r in rows))
bad = [r for r in rows if r["status"] not in ("DONE", "SKIP_EXISTING")]
if bad:
    print("bad rows:")
    for r in bad:
        print(r["dataset"], r["model"], "kc", r["kc"], "pr", r["poison_rate"], "seed", r["seed"], r["status"], r["log_path"])
PY
