#!/usr/bin/env bash
set -u
set -o pipefail

export PYTHONPATH=${EXTERNAL_REPOS_ROOT:-./external_backdoor_repos}/backdoor_attacks/python
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

REPO=${EXTERNAL_REPOS_ROOT:-./external_backdoor_repos}/backdoor_attacks
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT=${ROOT}/paper_results/external_baselines/marksman_pilot4
LOG=${ROOT}/logs_external_baselines/marksman_pilot4
STATUS=${OUT}/marksman_pilot4_status.csv

mkdir -p "$OUT" "$LOG"

cd "$REPO"

if [ ! -f python/mt_conditional_trigger_generation.py ]; then
  cp python/marksman_conditional_trigger_generation.py python/mt_conditional_trigger_generation.py
fi

echo "dataset,model,kc,poison_rate,seed,target_label,clsmodel,stage1_status,stage2_status,stage1_log,stage2_log,start_time,end_time" > "$STATUS"

run_one () {
  DATASET="$1"
  MODEL="$2"
  KC="$3"
  PR="$4"
  SEED="$5"
  TARGET="$6"

  if [ "$MODEL" = "resnet18" ]; then
    CLSMODEL="PreActResNet18"
  elif [ "$MODEL" = "vgg11" ]; then
    CLSMODEL="vgg11"
  else
    echo "bad model $MODEL"
    exit 2
  fi

  TAG="marksman_${DATASET}_${MODEL}_kc${KC}_pr005_seed${SEED}_target${TARGET}"
  STAGE1_LOG="${LOG}/${TAG}_stage1.out"
  STAGE2_LOG="${LOG}/${TAG}_stage2.out"
  START=$(date "+%Y-%m-%d %H:%M:%S")

  echo
  echo "===================================================================================================="
  echo "START $TAG"
  echo "===================================================================================================="

  python python/marksman_conditional_trigger_generation.py \
    --dataset "$DATASET" \
    --data_root "${ROOT}/data" \
    --clsmodel "$CLSMODEL" \
    --path "$OUT" \
    --epochs 5 \
    --train-epoch 1 \
    --mode all2one \
    --target_label "$TARGET" \
    --epochs_per_external_eval 5 \
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

  if [ "$S1" -eq 0 ]; then
    S1_STATUS="DONE"
  else
    S1_STATUS="FAILED"
  fi

  python python/marksman_conditional_backdoor_injection.py \
    --dataset "$DATASET" \
    --data_root "${ROOT}/data" \
    --clsmodel "$CLSMODEL" \
    --path "$OUT" \
    --epochs 5 \
    --train-epoch 1 \
    --mode all2one \
    --target_label "$TARGET" \
    --epochs_per_external_eval 5 \
    --cls_test_epochs 1 \
    --verbose 1 \
    --batch-size 128 \
    --alpha 0.5 \
    --eps 0.1 \
    --attack_portion "$PR" \
    --test_attack_portion "$PR" \
    --test_epochs 5 \
    --test_lr 0.01 \
    --schedulerC_lambda 0.1 \
    --schedulerC_milestones 3 \
    --test_optimizer sgd \
    --avoid_cls_reinit \
    --num-workers 2 \
    --seed "$SEED" \
    2>&1 | tee "$STAGE2_LOG"

  S2=${PIPESTATUS[0]}

  if [ "$S2" -eq 0 ]; then
    S2_STATUS="DONE"
  else
    S2_STATUS="FAILED"
  fi

  END=$(date "+%Y-%m-%d %H:%M:%S")
  echo "${DATASET},${MODEL},${KC},${PR},${SEED},${TARGET},${CLSMODEL},${S1_STATUS},${S2_STATUS},${STAGE1_LOG},${STAGE2_LOG},${START},${END}" >> "$STATUS"

  echo "END $TAG : stage1=$S1_STATUS stage2=$S2_STATUS"
}

# pilot 4: CIFAR10 only, PR=0.05, seed0
run_one cifar10 resnet18 4 0.05 0 0
run_one cifar10 vgg11    4 0.05 0 0
run_one cifar10 resnet18 8 0.05 0 0
run_one cifar10 vgg11    8 0.05 0 0

echo
echo "========== STATUS =========="
cat "$STATUS"
