#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.."
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):$PYTHONPATH"

MANIFEST="results/tables/defense_manifest_seed012.csv"

tail -n +2 "$MANIFEST" | while IFS=, read -r DATASET MODEL METHOD POISON SEED KC POSITION_MODE DEFENSE
do
  DATASET=$(echo "$DATASET" | tr -d '\r' | xargs)
  MODEL=$(echo "$MODEL" | tr -d '\r' | xargs)
  METHOD=$(echo "$METHOD" | tr -d '\r' | xargs)
  POISON=$(echo "$POISON" | tr -d '\r' | xargs)
  SEED=$(echo "$SEED" | tr -d '\r' | xargs)
  KC=$(echo "$KC" | tr -d '\r' | xargs)
  DEFENSE=$(echo "$DEFENSE" | tr -d '\r' | xargs)

  RUN_NAME="${DATASET}_${MODEL}_${METHOD}_dyn_pr${POISON}_kc${KC}_seed${SEED}"
  OUT="results/defenses/${RUN_NAME}/${DEFENSE}/metrics.json"

  if [ -f "$OUT" ]; then
    echo "[SKIP] exists: $OUT"
    continue
  fi

  echo "===================================================================================================="
  echo "[RUN] defense=$DEFENSE dataset=$DATASET model=$MODEL method=$METHOD poison=$POISON seed=$SEED kc=$KC"

  EXTRA=()

  if [ "$DEFENSE" = "strip" ]; then
    EXTRA+=(--strip-samples 1000 --strip-repeats 10)
  fi

  if [ "$DEFENSE" = "fine_pruning" ]; then
    EXTRA+=(--fp-rank-batches 20 --fp-fractions 0.0 0.1 0.2 0.3 0.5)
  fi

  python tools/run_defense_eval.py \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --method "$METHOD" \
    --poison-rate "$POISON" \
    --seed "$SEED" \
    --kc "$KC" \
    --ckpt-name last \
    --defense "$DEFENSE" \
    "${EXTRA[@]}"

  STATUS=$?
  if [ "$STATUS" -ne 0 ]; then
    echo "[ERROR] failed with status=$STATUS: $RUN_NAME / $DEFENSE"
    exit "$STATUS"
  fi
done
