#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):$PYTHONPATH"

DATASET=gtsrb
METHOD=full
POISON=0.1
SEARCH_GRID=4
SEARCH_SUBSET=2048

for MODEL in resnet18 vgg11
do
  CONFIG=configs/${DATASET}_${MODEL}.yaml

  for KC in 3 4
  do
    for SEED in 0 1 2
    do
      RUN_NAME="${DATASET}_${MODEL}_${METHOD}_dyn_pr${POISON}_kc${KC}_seed${SEED}"

      if [ -f "results/${RUN_NAME}/summary.json" ]; then
        echo "Skip existing run: ${RUN_NAME}"
        continue
      fi

      echo "============================================================"
      echo "Running ${RUN_NAME}"
      echo "============================================================"

      python -m src.train \
        --config ${CONFIG} \
        --method ${METHOD} \
        --poison-rate ${POISON} \
        --seed ${SEED} \
        --kc ${KC} \
        --position-mode dynamic \
        --search-grid ${SEARCH_GRID} \
        --search-subset ${SEARCH_SUBSET} \
        2>&1 | tee logs/kc_diag_${RUN_NAME}.log
    done
  done
done
