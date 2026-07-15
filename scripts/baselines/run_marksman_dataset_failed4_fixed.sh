#!/usr/bin/env bash
set -u
set -o pipefail

export PYTHONPATH=${EXTERNAL_REPOS_ROOT:-./external_backdoor_repos}/backdoor_attacks/python
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

REPO=${EXTERNAL_REPOS_ROOT:-./external_backdoor_repos}/backdoor_attacks
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT=${ROOT}/paper_results/external_baselines/marksman_dataset_pilot_fixed
LOG=${ROOT}/logs_external_baselines/marksman_dataset_pilot_fixed
STATUS=${OUT}/marksman_dataset_failed4_fixed_status.csv

mkdir -p "$OUT" "$LOG"
cd "$REPO"

echo "dataset,model,poison_rate,seed,target_label,clsmodel,stage1_status,stage2_status,stage1_log,stage2_log,start_time,end_time" > "$STATUS"

run_one () {
  DATASET="$1"
  MODEL="$2"
  PR="$3"
  SEED="$4"
  TARGET="$5"

  if [ "$MODEL" = "resnet18" ]; then
    CLSMODEL="PreActResNet18"
  elif [ "$MODEL" = "vgg11" ]; then
    CLSMODEL="vgg11"
  else
    echo "bad model $MODEL"
    exit 2
  fi

  TAG="marksman_${DATASET}_${MODEL}_pr005_seed${SEED}_target${TARGET}_fixed"
  RUN_OUT="${OUT}/${TAG}"
  STAGE1_LOG="${LOG}/${TAG}_stage1.out"
  STAGE2_LOG="${LOG}/${TAG}_stage2.out"
  START=$(date "+%Y-%m-%d %H:%M:%S")

  mkdir -p "$RUN_OUT"

  echo
  echo "===================================================================================================="
  echo "START $TAG"
  echo "===================================================================================================="

  set +e
  python python/marksman_conditional_trigger_generation.py \
    --dataset "$DATASET" \
    --data_root "${ROOT}/data" \
    --clsmodel "$CLSMODEL" \
    --path "$RUN_OUT" \
    --epochs 1 \
    --train-epoch 1 \
    --mode all2one \
    --target_label "$TARGET" \
    --epochs_per_external_eval 1 \
    --cls_test_epochs 1 \
    --verbose 1 \
    --batch-size 128 \
    --alpha 0.5 \
    --eps 0.1 \
    --attack_portion "$PR" \
    --avoid_cls_reinit \
    --num-workers 2 \
    --seed "$SEED" \
    2>&1 | tee "$STAGE1_LOG"
  S1=${PIPESTATUS[0]}
  set -e

  if [ "$S1" -eq 0 ]; then
    S1_STATUS="DONE"
  else
    S1_STATUS="FAILED"
  fi

  set +e
  python python/marksman_conditional_backdoor_injection.py \
    --dataset "$DATASET" \
    --data_root "${ROOT}/data" \
    --clsmodel "$CLSMODEL" \
    --path "$RUN_OUT" \
    --epochs 1 \
    --train-epoch 1 \
    --mode all2one \
    --target_label "$TARGET" \
    --epochs_per_external_eval 1 \
    --cls_test_epochs 1 \
    --verbose 1 \
    --batch-size 128 \
    --alpha 0.5 \
    --eps 0.1 \
    --attack_portion "$PR" \
    --test_attack_portion "$PR" \
    --test_epochs 1 \
    --test_lr 0.01 \
    --schedulerC_lambda 0.1 \
    --schedulerC_milestones 1 \
    --test_optimizer sgd \
    --avoid_cls_reinit \
    --num-workers 2 \
    --seed "$SEED" \
    2>&1 | tee "$STAGE2_LOG"
  S2=${PIPESTATUS[0]}
  set -e

  if [ "$S2" -eq 0 ]; then
    S2_STATUS="DONE"
  else
    S2_STATUS="FAILED"
  fi

  END=$(date "+%Y-%m-%d %H:%M:%S")
  echo "${DATASET},${MODEL},${PR},${SEED},${TARGET},${CLSMODEL},${S1_STATUS},${S2_STATUS},${STAGE1_LOG},${STAGE2_LOG},${START},${END}" >> "$STATUS"

  echo "END $TAG : stage1=$S1_STATUS stage2=$S2_STATUS"
}

run_one gtsrb    resnet18 0.05 0 0
run_one gtsrb    vgg11    0.05 0 0
run_one cifar100 resnet18 0.05 0 0
run_one cifar100 vgg11    0.05 0 0

echo
echo "========== STATUS =========="
cat "$STATUS"
