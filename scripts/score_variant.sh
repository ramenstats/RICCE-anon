#!/usr/bin/env bash
# Scores one trained checkpoint through the exact same pipeline used for the
# existing baselines/{sft,dpo,tcpo}_seed42: generate predictions with the
# LoRA adapter (scripts/lora_infer.py), then score them with the E08
# experiment (ricce experiment E08), writing metrics.json in the same
# schema as results/E08/LEVIR_CC/sft_seed42/metrics.json.
#
# Usage: scripts/score_variant.sh <adapter_dir> <variant_seed_name>
# e.g.   scripts/score_variant.sh checkpoints/LEVIR_CC/tcpo_v3_seed42 tcpo_v3_seed42
set -uo pipefail
cd "$(dirname "$0")/.."
ADAPTER=$1
NAME=$2
M="artifacts/manifests/LEVIR_CC.jsonl"

if [ ! -d "$ADAPTER" ] || [ -z "$(ls -A "$ADAPTER" 2>/dev/null)" ]; then
  echo "No checkpoint at $ADAPTER -- nothing to score."
  exit 1
fi

mkdir -p "results/baselines/${NAME}" "results/E08/LEVIR_CC/${NAME}"
echo "=== $(date -Is) generating predictions for ${NAME} from ${ADAPTER} ==="
python3 scripts/lora_infer.py --adapter "$ADAPTER" --manifest "$M" \
  --output "results/baselines/${NAME}/LEVIR_CC_test.jsonl" --split test \
  > "logs/infer_${NAME}.log" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "Inference failed (rc=$rc) for ${NAME}; see logs/infer_${NAME}.log"
  exit "$rc"
fi

echo "=== $(date -Is) scoring ${NAME} via E08 ==="
python3 -m ricce.cli experiment E08 "$M" \
  --predictions "results/baselines/${NAME}/LEVIR_CC_test.jsonl" \
  --outdir "results/E08/LEVIR_CC/${NAME}" \
  > "logs/e08_${NAME}.log" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "E08 scoring failed (rc=$rc) for ${NAME}; see logs/e08_${NAME}.log"
  exit "$rc"
fi
echo "Done: results/E08/LEVIR_CC/${NAME}/metrics.json"
