#!/usr/bin/env bash
set -euo pipefail
M="artifacts/manifests/LEVIR_CC.jsonl"
[ -f "$M" ] || { echo "Missing $M"; exit 1; }
python -m ricce.training.sft --manifest "$M" --output checkpoints/LEVIR_CC/sft_seed42 --seed 42
python -m ricce.training.tcpo --manifest "$M" --output checkpoints/LEVIR_CC/dpo_seed42 --mode dpo --seed 42
python -m ricce.training.tcpo --manifest "$M" --output checkpoints/LEVIR_CC/tcpo_seed42 --mode tcpo --seed 42
