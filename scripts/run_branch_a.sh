#!/usr/bin/env bash
# v2 Branch A (TCPO causal-severity study) master orchestrator.
# Trains the full required matrix (A2: 7 weighting variants; A6: + inverse-
# severity negative control; A4: + hard-negative-curriculum variant) across
# 3 seeds, SCORING each checkpoint with the independent-metric pipeline
# (scripts/score_variant_branch_a.py) immediately after it finishes, and
# regenerating results_r1/tcpo_branch_a/severity_causal_ablation_summary.csv
# after every completed run so partial progress is always a real, readable
# deliverable -- not just checkpoints sitting unscored.
#
# All variants share: --init_from checkpoints/LEVIR_CC/sft_seed<seed> (same
# SFT policy as BOTH start point and frozen reference, per A1), same manifest,
# same LoRA config (fixed in configs/project.yaml), --lr 2e-6 --grad_accum 16
# --batch_size 1 --warmup_steps 100 --max_grad_norm 1.0 --max_steps 200
# --kl_coef 0.05 (except the DPO-unweighted variant, which sets kl_coef 0 to
# isolate the KL-regularization effect from dpo_reg) -- i.e. only the
# weighting/negative-construction strategy differs between conditions,
# exactly matching the four already-completed seed42 variants
# (dpo_reg/tcpo_v3/tcpo_random_severity/tcpo_no_severity) so all 7+ variants
# are directly comparable.
#
# Does NOT use `set -e`: one run OOM-ing or destabilizing must not abort the
# rest of the matrix, and must be logged as a genuine outcome.
set -uo pipefail
cd "$(dirname "$0")/.."
M="artifacts/manifests/LEVIR_CC.jsonl"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE="results_r1/tcpo_branch_a"
mkdir -p logs checkpoints/LEVIR_CC results/training_diagnostics "$BASE"/{checkpoints,metrics,logs,diagnostics}

wait_for_gpu() {
  local need_mib=$1 max_wait_s=${2:-10800} waited=0 free free2
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "$free" ] && [ "$free" -ge "$need_mib" ]; then
      sleep 20
      free2=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
      if [ -n "$free2" ] && [ "$free2" -ge "$need_mib" ]; then return 0; fi
    fi
    if [ "$waited" -ge "$max_wait_s" ]; then return 1; fi
    sleep 45; waited=$((waited+65))
  done
}

run_job() {
  local name=$1; shift
  local attempt=1 rc=0
  echo "=== $(date -Is) QUEUEING ${name} ===" | tee -a logs/branch_a_orchestrator.log
  while [ "$attempt" -le 3 ]; do
    if ! wait_for_gpu 26000 10800; then
      echo "=== $(date -Is) ${name} attempt ${attempt}: GPU never freed up (blocked by contention), skipping ===" | tee -a logs/branch_a_orchestrator.log
      return 1
    fi
    echo "=== $(date -Is) ${name} attempt ${attempt}: starting ===" | tee -a logs/branch_a_orchestrator.log
    "$@"
    rc=$?
    if [ "$rc" -eq 0 ]; then
      echo "=== $(date -Is) ${name} attempt ${attempt}: finished OK ===" | tee -a logs/branch_a_orchestrator.log
      return 0
    fi
    echo "=== $(date -Is) ${name} attempt ${attempt}: FAILED rc=${rc} (likely OOM under shared-GPU contention), backing off ===" | tee -a logs/branch_a_orchestrator.log
    sleep 120
    attempt=$((attempt+1))
  done
  echo "=== $(date -Is) ${name}: exhausted retries, giving up ===" | tee -a logs/branch_a_orchestrator.log
  return 1
}

# Trains one variant, scores it, copies provenance into $BASE, then
# regenerates the causal-ablation summary CSV -- all as one job so a run
# that fails never leaves a scored-but-not-copied or trained-but-not-scored
# half-state blocking the next one.
train_and_score() {
  local name=$1 seed=$2; shift 2
  local ckpt="checkpoints/LEVIR_CC/${name}"
  if [ -f "${ckpt}/run_config.json" ] && [ -f "${BASE}/metrics/${name}/metrics.json" ]; then
    echo "=== $(date -Is) ${name}: already trained+scored, skipping ===" | tee -a logs/branch_a_orchestrator.log
    return 0
  fi
  if [ ! -f "${ckpt}/run_config.json" ]; then
    run_job "train_${name}" python3 -m ricce.training.tcpo --manifest "$M" \
      --output "$ckpt" --seed "$seed" --init_from "checkpoints/LEVIR_CC/sft_seed${seed}" \
      --lr 2e-6 --grad_accum 16 --batch_size 1 --warmup_steps 100 --max_grad_norm 1.0 \
      --max_steps 200 --ckpt_every 50 --log_every 10 \
      --diag_out "results/training_diagnostics/${name}.jsonl" \
      "$@" > "logs/${name}.log" 2>&1
    if [ $? -ne 0 ]; then
      echo "=== $(date -Is) ${name}: training did not complete, NOT scoring ===" | tee -a logs/branch_a_orchestrator.log
      return 1
    fi
  fi
  run_job "score_${name}" python3 scripts/score_variant_branch_a.py \
    --adapter "$ckpt" --name "$name" --manifest "$M" \
    --expensive_limit 200 --e10_limit 150 \
    > "${BASE}/logs/score_${name}.log" 2>&1
  cp -f "${ckpt}/run_config.json" "${BASE}/checkpoints/${name}.run_config.json" 2>/dev/null
  cp -f "results/training_diagnostics/${name}.jsonl" "${BASE}/diagnostics/${name}.jsonl" 2>/dev/null
  python3 scripts/build_causal_ablation_summary.py > "${BASE}/logs/summary_rebuild_$(date +%s).log" 2>&1
}

# ---- Round A: new required/ablation variants at seed42 (matched to the 4
#      already-completed seed42 variants: dpo_reg, tcpo_v3, tcpo_random_severity,
#      tcpo_no_severity) ----
train_and_score dpo_stab_seed42 42 --mode dpo --kl_coef 0.0 --neg_strategy hardest
train_and_score tcpo_clipped_severity_seed42 42 --mode tcpo --severity clipped --kl_coef 0.05 --neg_strategy hardest
train_and_score tcpo_normalized_severity_seed42 42 --mode tcpo --severity normalized --kl_coef 0.05 --neg_strategy hardest
train_and_score tcpo_inverse_severity_seed42 42 --mode tcpo --severity inverse --kl_coef 0.05 --neg_strategy hardest
train_and_score tcpo_curriculum_severity_seed42 42 --mode tcpo --severity taxonomy --kl_coef 0.05 --neg_strategy curriculum --curriculum_schedule linear

# ---- Round B: seed=1 replicate of the full 8-way matched set (A3) ----
train_and_score dpo_stab_seed1 1 --mode dpo --kl_coef 0.0 --neg_strategy hardest
train_and_score dpo_reg_seed1 1 --mode dpo --kl_coef 0.05 --neg_strategy hardest
train_and_score tcpo_v3_seed1 1 --mode tcpo --severity taxonomy --kl_coef 0.05 --neg_strategy hardest
train_and_score tcpo_random_severity_seed1 1 --mode tcpo --severity random --kl_coef 0.05 --neg_strategy hardest
train_and_score tcpo_no_severity_seed1 1 --mode tcpo --severity uniform --kl_coef 0.05 --neg_strategy hardest
train_and_score tcpo_clipped_severity_seed1 1 --mode tcpo --severity clipped --kl_coef 0.05 --neg_strategy hardest
train_and_score tcpo_normalized_severity_seed1 1 --mode tcpo --severity normalized --kl_coef 0.05 --neg_strategy hardest
train_and_score tcpo_inverse_severity_seed1 1 --mode tcpo --severity inverse --kl_coef 0.05 --neg_strategy hardest

# ---- Round C: seed=123 replicate of the same 8-way matched set (A3, third seed) ----
train_and_score dpo_stab_seed123 123 --mode dpo --kl_coef 0.0 --neg_strategy hardest
train_and_score dpo_reg_seed123 123 --mode dpo --kl_coef 0.05 --neg_strategy hardest
train_and_score tcpo_v3_seed123 123 --mode tcpo --severity taxonomy --kl_coef 0.05 --neg_strategy hardest
train_and_score tcpo_random_severity_seed123 123 --mode tcpo --severity random --kl_coef 0.05 --neg_strategy hardest
train_and_score tcpo_no_severity_seed123 123 --mode tcpo --severity uniform --kl_coef 0.05 --neg_strategy hardest
train_and_score tcpo_clipped_severity_seed123 123 --mode tcpo --severity clipped --kl_coef 0.05 --neg_strategy hardest
train_and_score tcpo_normalized_severity_seed123 123 --mode tcpo --severity normalized --kl_coef 0.05 --neg_strategy hardest
train_and_score tcpo_inverse_severity_seed123 123 --mode tcpo --severity inverse --kl_coef 0.05 --neg_strategy hardest

# ---- Also score the 4 pre-existing seed42 variants + baselines into the
#      Branch-A metrics tree if not already there, so the summary CSV is
#      self-contained ----
for pair in "dpo_reg_seed42 42" "tcpo_v3_seed42 42" "tcpo_random_severity_seed42 42" "tcpo_no_severity_seed42 42"; do
  set -- $pair
  n=$1
  if [ ! -f "${BASE}/metrics/${n}/metrics.json" ] && [ -f "checkpoints/LEVIR_CC/${n}/run_config.json" ]; then
    run_job "score_${n}" python3 scripts/score_variant_branch_a.py \
      --adapter "checkpoints/LEVIR_CC/${n}" --name "$n" --manifest "$M" \
      --expensive_limit 200 --e10_limit 150 > "${BASE}/logs/score_${n}.log" 2>&1
    cp -f "checkpoints/LEVIR_CC/${n}/run_config.json" "${BASE}/checkpoints/${n}.run_config.json" 2>/dev/null
    cp -f "results/training_diagnostics/${n}.jsonl" "${BASE}/diagnostics/${n}.jsonl" 2>/dev/null
    python3 scripts/build_causal_ablation_summary.py > "${BASE}/logs/summary_rebuild_$(date +%s).log" 2>&1
  fi
done

echo "=== $(date -Is) BRANCH A ORCHESTRATOR DONE ===" | tee -a logs/branch_a_orchestrator.log
