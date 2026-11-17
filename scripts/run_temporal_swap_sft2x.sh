#!/usr/bin/env bash
# STEP-MATCHED CONTROL sft2x: same 13038 rows/steps as swapsft, but the 6223 extra rows are the SAME images+captions in ORIGINAL order (dup_control_train.jsonl). Waits for the swapsft orchestrator to finish first.
# (derived from) TemporalSwap-SFT orchestrator (idempotent / resumable). For seeds 42,1,123:
#   train sft2x_seed<s> = SFT hyperparameters identical to sft_seed<s> (configs/project.yaml,
#   default `python -m ricce.training.sft`) + --aug_jsonl swapped data -> evaluate with
#   scripts/eval_temporal_swap.py. Then evaluates the matched sft_seed{1,42,123} controls
#   (E10 @300 pairs; existing inference/E08 reused) and aggregates.
# No `set -e`: one failed job must not abort the matrix. Finished stages are skipped.
cd "$(dirname "$0")/.."
source /opt/anaconda/etc/profile.d/conda.sh && conda activate ricce
set -uo pipefail  # (after activate: conda's openjdk hook uses unset JAVA_HOME)
M="artifacts/manifests/LEVIR_CC.jsonl"
AUG="artifacts_cvpr_final/temporal_equivariance/dup_control_train.jsonl"
R="results_cvpr_final/temporal_equivariance"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs "$R/logs" "$R/runs" checkpoints/LEVIR_CC
ORCH="logs/temporal_swap_sft2x_orchestrator.log"
say() { echo "=== $(date -Is) $* ===" | tee -a "$ORCH" "$R/logs/orchestrator.log"; }

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
  local name=$1; shift; local attempt=1 rc=0
  say "QUEUEING ${name}"
  while [ "$attempt" -le 3 ]; do
    if ! wait_for_gpu 30000 10800; then say "${name} attempt ${attempt}: GPU never freed up, skipping"; return 1; fi
    say "${name} attempt ${attempt}: starting"
    "$@"; rc=$?
    if [ "$rc" -eq 0 ]; then say "${name} attempt ${attempt}: finished OK"; return 0; fi
    say "${name} attempt ${attempt}: FAILED rc=${rc} (likely OOM / contention), backing off"
    sleep 120; attempt=$((attempt+1))
  done
  say "${name}: exhausted retries"; return 1
}
eval_run() {  # $1=name
  local name=$1
  if [ -f "$R/runs/${name}/eval.json" ]; then say "${name}: already evaluated, skipping"; return 0; fi
  run_job "eval_${name}" python3 scripts/eval_temporal_swap.py --name "$name" \
      --adapter "checkpoints/LEVIR_CC/${name}" --e10_limit 300 >> "$R/logs/eval_${name}.log" 2>&1
  # verify: the E08 step must have produced ce_f1 (guards against silent CLI-typo failures)
  if ! python3 -c "import json,sys; sys.exit(0 if 'ce_f1' in json.load(open('results/E08/LEVIR_CC/${name}/metrics.json')) else 1)" 2>/dev/null; then
    say "${name}: VERIFY FAILED, results/E08/LEVIR_CC/${name}/metrics.json lacks ce_f1"; rm -f "$R/runs/${name}/eval.json"; return 1
  fi
}
train_run() {  # $1=seed
  local s=$1 name="sft2x_seed$1" ckpt="checkpoints/LEVIR_CC/sft2x_seed$1"
  if [ -f "${ckpt}/run_config.json" ] && [ -f "${ckpt}/adapter_model.safetensors" ]; then say "${name}: already trained, skipping"; return 0; fi
  run_job "train_${name}" python3 -m ricce.training.sft --manifest "$M" --aug_jsonl "$AUG" \
      --output "$ckpt" --seed "$s" --log_every 50 >> "logs/${name}.log" 2>&1
  cp -f "logs/${name}.log" "$R/logs/train_${name}.log" 2>/dev/null
}

WAIT_PID=${1:-}
if [ -n "$WAIT_PID" ]; then while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 120; done; fi
[ -f "$AUG" ] || { say "missing $AUG"; exit 1; }
for s in 42 1 123; do
  train_run "$s" && eval_run "sft2x_seed${s}"
  cp -f "logs/sft2x_seed${s}.log" "$R/logs/train_sft2x_seed${s}.log" 2>/dev/null
done
python3 scripts/eval_temporal_swap.py --aggregate > "$R/logs/aggregate_final.log" 2>&1
say "SFT2X ORCHESTRATOR DONE"
