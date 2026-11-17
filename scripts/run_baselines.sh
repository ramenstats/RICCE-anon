#!/usr/bin/env bash
set -euo pipefail
M="artifacts/manifests/LEVIR_CC.jsonl"
[ -f "$M" ] || { echo "Missing $M; run scripts/prepare_all.sh first"; exit 1; }
mkdir -p results/baselines/qwen
python3 -m ricce.cli vlm-infer "$M" results/baselines/qwen/LEVIR_CC_test.jsonl --split test
python3 -m ricce.cli experiment E08 "$M" --predictions results/baselines/qwen/LEVIR_CC_test.jsonl --outdir results/E08/LEVIR_CC/qwen
