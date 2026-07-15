#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):$PYTHONPATH"

MANIFEST=results/tables/defense_manifest_seed0.csv
CKPT_NAME=last

tail -n +2 "$MANIFEST" | while IFS=, read -r DATASET MODEL METHOD POISON SEED KC POSITION_MODE DEFENSE
do
  DATASET=$(echo "$DATASET" | tr -d '\r' | xargs)
  MODEL=$(echo "$MODEL" | tr -d '\r' | xargs)
  METHOD=$(echo "$METHOD" | tr -d '\r' | xargs)
  POISON=$(echo "$POISON" | tr -d '\r' | xargs)
  SEED=$(echo "$SEED" | tr -d '\r' | xargs)
  KC=$(echo "$KC" | tr -d '\r' | xargs)
  POSITION_MODE=$(echo "$POSITION_MODE" | tr -d '\r' | xargs)
  DEFENSE=$(echo "$DEFENSE" | tr -d '\r' | xargs)

  RUN_NAME="${DATASET}_${MODEL}_${METHOD}_dyn_pr${POISON}_kc${KC}_seed${SEED}"
  OUT="results/defenses/${RUN_NAME}/${DEFENSE}/metrics.json"

  if [ -f "$OUT" ]; then
    echo "Skip existing defense result: $OUT"
    continue
  fi

  echo "======================================================================"
  echo "Running defense=${DEFENSE} run=${RUN_NAME}"
  echo "======================================================================"

  if [ "$DEFENSE" = "strip" ]; then
    python tools/run_defense_eval.py \
      --dataset "$DATASET" \
      --model "$MODEL" \
      --method "$METHOD" \
      --poison-rate "$POISON" \
      --seed "$SEED" \
      --kc "$KC" \
      --ckpt-name "$CKPT_NAME" \
      --defense strip \
      --strip-samples 500 \
      --strip-repeats 10 \
      2>&1 | tee "logs/defense_${RUN_NAME}_${DEFENSE}.log"

  elif [ "$DEFENSE" = "spectral_signatures" ]; then
    python tools/run_defense_eval.py \
      --dataset "$DATASET" \
      --model "$MODEL" \
      --method "$METHOD" \
      --poison-rate "$POISON" \
      --seed "$SEED" \
      --kc "$KC" \
      --ckpt-name "$CKPT_NAME" \
      --defense spectral_signatures \
      --ss-samples 5000 \
      --ss-remove-fraction 0.1 \
      2>&1 | tee "logs/defense_${RUN_NAME}_${DEFENSE}.log"

  elif [ "$DEFENSE" = "fine_pruning" ]; then
    python tools/run_defense_eval.py \
      --dataset "$DATASET" \
      --model "$MODEL" \
      --method "$METHOD" \
      --poison-rate "$POISON" \
      --seed "$SEED" \
      --kc "$KC" \
      --ckpt-name "$CKPT_NAME" \
      --defense fine_pruning \
      --fp-rank-batches 20 \
      --fp-fractions 0.0 0.1 0.2 0.3 0.5 \
      2>&1 | tee "logs/defense_${RUN_NAME}_${DEFENSE}.log"

  elif [ "$DEFENSE" = "neural_cleanse" ]; then
    python tools/run_defense_eval.py \
      --dataset "$DATASET" \
      --model "$MODEL" \
      --method "$METHOD" \
      --poison-rate "$POISON" \
      --seed "$SEED" \
      --kc "$KC" \
      --ckpt-name "$CKPT_NAME" \
      --defense neural_cleanse \
      --nc-samples 128 \
      --nc-steps 60 \
      --nc-lr 0.1 \
      --nc-l1 0.01 \
      2>&1 | tee "logs/defense_${RUN_NAME}_${DEFENSE}.log"
  else
    echo "Unknown defense: [$DEFENSE]"
    exit 1
  fi
done
