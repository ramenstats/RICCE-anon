#!/usr/bin/env bash
# Re-run ONLY the E08 (CE-F1) scoring step for every Branch-A checkpoint
# whose predictions already exist on disk -- the original run trained all
# 21 variants correctly but the scoring step silently failed for every one
# of them due to a --expensive_limit vs --expensive-limit CLI flag bug in
# scripts/score_variant_branch_a.py (now fixed). Inference is NOT redone,
# only the fast CPU-side CE-F1 scoring, which the original 11-day run never
# actually completed for these variants.
set -o pipefail
cd "$(dirname "$0")/.."
source /opt/anaconda/etc/profile.d/conda.sh
conda activate ricce
set -u

M="artifacts/manifests/LEVIR_CC.jsonl"
VARIANTS="dpo_reg_seed1 dpo_reg_seed123 dpo_stab_seed1 dpo_stab_seed123 dpo_stab_seed42 tcpo_clipped_severity_seed1 tcpo_clipped_severity_seed123 tcpo_clipped_severity_seed42 tcpo_curriculum_severity_seed42 tcpo_inverse_severity_seed1 tcpo_inverse_severity_seed123 tcpo_inverse_severity_seed42 tcpo_normalized_severity_seed1 tcpo_normalized_severity_seed123 tcpo_normalized_severity_seed42 tcpo_random_severity_seed1 tcpo_random_severity_seed123 tcpo_v3_seed1 tcpo_v3_seed123 tcpo_no_severity_seed1 tcpo_no_severity_seed123"

mkdir -p logs
for v in $VARIANTS; do
  pred="results/baselines/${v}/LEVIR_CC_test.jsonl"
  if [ ! -f "$pred" ]; then
    echo "=== $(date -Is) ${v}: no predictions found at ${pred}, skipping ===" | tee -a logs/rescore_branch_a.log
    continue
  fi
  outdir="results/E08/LEVIR_CC/${v}"
  mkdir -p "$outdir"
  echo "=== $(date -Is) scoring ${v} ===" | tee -a logs/rescore_branch_a.log
  python3 -m ricce.cli experiment E08 "$M" --predictions "$pred" --outdir "$outdir" --expensive-limit 200 \
    > "logs/rescore_e08_${v}.log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "=== $(date -Is) ${v}: FAILED rc=${rc}, see logs/rescore_e08_${v}.log ===" | tee -a logs/rescore_branch_a.log
  else
    cef1=$(python3 -c "import json;print(json.load(open('${outdir}/metrics.json')).get('ce_f1'))" 2>/dev/null)
    echo "=== $(date -Is) ${v}: done, ce_f1=${cef1} ===" | tee -a logs/rescore_branch_a.log
  fi
done
echo "=== $(date -Is) RESCORE_BRANCH_A_E08_DONE ===" | tee -a logs/rescore_branch_a.log
