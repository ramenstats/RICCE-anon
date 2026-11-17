#!/usr/bin/env bash
# required check orchestrator: (1) 3-seed SFT, (2) regularized DPO,
# TCPO, TCPO-random-severity, TCPO-without-severity, each with the
# stability fixes (SFT init/reference, low LR, warmup, grad clipping, explicit
# KL penalty, harder-negative mining) via ricce/training/tcpo.py's new
# --kl_coef/--severity/--neg_strategy/--batch_size flags. Does NOT use `set -e`
# on purpose: a single run OOM-ing or collapsing must not abort the rest of the
# queue, and must be logged as a genuine outcome, not silently skipped.
set -uo pipefail
cd "$(dirname "$0")/.."
M="artifacts/manifests/LEVIR_CC.jsonl"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs checkpoints/LEVIR_CC results/training_diagnostics results/E08/LEVIR_CC results/baselines

# Waits until at least $1 MiB is free on GPU 0 on TWO consecutive checks
# 20s apart, polling every 45s, up to $2 seconds total (default 3h). This
# machine's GPU is explicitly shared with other concurrent agents' jobs
# (dataset-benchmark, SPICE/LLM-judge scoring), so contention is expected,
# not exceptional. A single-sample check is racy under this kind of
# multi-tenant load: an earlier run of this script observed free memory clear
# a 16000MiB bar, then OOM within ~20s anyway because another concurrent job
# grabbed memory in the interim -- so this requires the headroom to hold
# steady across two checks, not just an instant, before declaring it safe.
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

# Runs one training command up to 3 attempts, waiting for GPU headroom before
# each attempt and backing off after an OOM. Logs a clear, non-fabricated
# status line either way.
run_job() {
  local name=$1; shift
  local attempt=1 rc=0
  echo "=== $(date -Is) QUEUEING ${name} ===" | tee -a logs/training_variants.log
  while [ "$attempt" -le 3 ]; do
    if ! wait_for_gpu 26000 10800; then
      echo "=== $(date -Is) ${name} attempt ${attempt}: GPU never freed up (blocked by contention), skipping ===" | tee -a logs/training_variants.log
      return 1
    fi
    echo "=== $(date -Is) ${name} attempt ${attempt}: starting ===" | tee -a logs/training_variants.log
    "$@"
    rc=$?
    if [ "$rc" -eq 0 ]; then
      echo "=== $(date -Is) ${name} attempt ${attempt}: finished OK ===" | tee -a logs/training_variants.log
      return 0
    fi
    echo "=== $(date -Is) ${name} attempt ${attempt}: FAILED rc=${rc} (likely OOM under shared-GPU contention), backing off ===" | tee -a logs/training_variants.log
    sleep 120
    attempt=$((attempt+1))
  done
  echo "=== $(date -Is) ${name}: exhausted retries, giving up ===" | tee -a logs/training_variants.log
  return 1
}

# ---- (A) 3-seed SFT (seed 42 already exists) ----
for s in 1 123; do
  run_job "sft_seed${s}" python3 -m ricce.training.sft --manifest "$M" \
    --output "checkpoints/LEVIR_CC/sft_seed${s}" --seed "$s" \
    > "logs/sft_seed${s}.log" 2>&1
done

# ---- (B) regularized DPO: SFT init+reference, low LR, warmup, grad clip,
#          explicit KL-to-reference penalty, harder-negative mining ----
run_job "dpo_reg_seed42" python3 -m ricce.training.tcpo --manifest "$M" \
  --output checkpoints/LEVIR_CC/dpo_reg_seed42 --mode dpo --seed 42 \
  --init_from checkpoints/LEVIR_CC/sft_seed42 --lr 2e-6 --grad_accum 16 --batch_size 1 \
  --neg_strategy hardest --kl_coef 0.05 --warmup_steps 100 --max_grad_norm 1.0 \
  --max_steps 200 --ckpt_every 50 --log_every 10 \
  --diag_out results/training_diagnostics/dpo_reg_seed42.jsonl \
  > logs/dpo_reg_seed42.log 2>&1

# ---- (C) TCPO (taxonomy severity), same hardening as (B) ----
run_job "tcpo_v3_seed42" python3 -m ricce.training.tcpo --manifest "$M" \
  --output checkpoints/LEVIR_CC/tcpo_v3_seed42 --mode tcpo --severity taxonomy --seed 42 \
  --init_from checkpoints/LEVIR_CC/sft_seed42 --lr 2e-6 --grad_accum 16 --batch_size 1 \
  --neg_strategy hardest --kl_coef 0.05 --warmup_steps 100 --max_grad_norm 1.0 \
  --max_steps 200 --ckpt_every 50 --log_every 10 \
  --diag_out results/training_diagnostics/tcpo_v3_seed42.jsonl \
  > logs/tcpo_v3_seed42.log 2>&1

# ---- (D) TCPO-random-severity ablation ----
run_job "tcpo_random_severity_seed42" python3 -m ricce.training.tcpo --manifest "$M" \
  --output checkpoints/LEVIR_CC/tcpo_random_severity_seed42 --mode tcpo --severity random --seed 42 \
  --init_from checkpoints/LEVIR_CC/sft_seed42 --lr 2e-6 --grad_accum 16 --batch_size 1 \
  --neg_strategy hardest --kl_coef 0.05 --warmup_steps 100 --max_grad_norm 1.0 \
  --max_steps 200 --ckpt_every 50 --log_every 10 \
  --diag_out results/training_diagnostics/tcpo_random_severity_seed42.jsonl \
  > logs/tcpo_random_severity_seed42.log 2>&1

# ---- (E) TCPO-without-severity ablation ----
run_job "tcpo_no_severity_seed42" python3 -m ricce.training.tcpo --manifest "$M" \
  --output checkpoints/LEVIR_CC/tcpo_no_severity_seed42 --mode tcpo --severity uniform --seed 42 \
  --init_from checkpoints/LEVIR_CC/sft_seed42 --lr 2e-6 --grad_accum 16 --batch_size 1 \
  --neg_strategy hardest --kl_coef 0.05 --warmup_steps 100 --max_grad_norm 1.0 \
  --max_steps 200 --ckpt_every 50 --log_every 10 \
  --diag_out results/training_diagnostics/tcpo_no_severity_seed42.jsonl \
  > logs/tcpo_no_severity_seed42.log 2>&1

echo "=== $(date -Is) ORCHESTRATOR DONE ===" | tee -a logs/training_variants.log
