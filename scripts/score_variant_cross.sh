#!/usr/bin/env bash
# Cross-dataset variant of score_variant.sh: scores one trained LoRA
# checkpoint against a non-LEVIR-CC manifest (SECOND_CC/DUBAI_CC), same
# pipeline (lora_infer.py -> E08), so the SECOND-CC/Dubai-CC generalization
# table can use the *fixed* TCPO checkpoints (tcpo_v3/tcpo_random_severity/
# tcpo_no_severity) instead of the pre-fix collapsed tcpo_seed42/dpo_seed42
# checkpoints that score 0.0 there.
#
# Usage: scripts/score_variant_cross.sh <adapter_dir> <variant_seed_name> <DATASET>
set -uo pipefail
cd "$(dirname "$0")/.."
ADAPTER=$1
NAME=$2
DATASET=$3
M="artifacts/manifests/${DATASET}.jsonl"

if [ ! -d "$ADAPTER" ] || [ -z "$(ls -A "$ADAPTER" 2>/dev/null)" ]; then
  echo "No checkpoint at $ADAPTER -- nothing to score."
  exit 1
fi
if [ ! -f "$M" ]; then
  echo "No manifest at $M -- nothing to score."
  exit 1
fi

mkdir -p "results/baselines/${NAME}" "results/E08/${DATASET}/${NAME}"
echo "=== $(date -Is) generating predictions for ${NAME} on ${DATASET} from ${ADAPTER} ==="
python3 scripts/lora_infer.py --adapter "$ADAPTER" --manifest "$M" \
  --output "results/baselines/${NAME}/${DATASET}_test.jsonl" --split test \
  > "logs/infer_${NAME}_${DATASET}.log" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "Inference failed (rc=$rc) for ${NAME}/${DATASET}; see logs/infer_${NAME}_${DATASET}.log"
  exit "$rc"
fi

echo "=== $(date -Is) scoring ${NAME} on ${DATASET} via E08 ==="
python3 -m ricce.cli experiment E08 "$M" \
  --predictions "results/baselines/${NAME}/${DATASET}_test.jsonl" \
  --outdir "results/E08/${DATASET}/${NAME}" \
  > "logs/e08_${NAME}_${DATASET}.log" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "E08 scoring failed (rc=$rc) for ${NAME}/${DATASET}; see logs/e08_${NAME}_${DATASET}.log"
  exit "$rc"
fi
echo "=== $(date -Is) done ${NAME} on ${DATASET} ==="
