#!/bin/bash
# Score every input JSONL (E04 perturbations + frozen system predictions) in parallel, CPU only.
source /opt/anaconda/etc/profile.d/conda.sh && conda activate ricce
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 CE_TMPDIR=${CE_TMPDIR:-/tmp}
IN=artifacts_cvpr_final/baselines/inputs; OUT=results_cvpr_final/baselines/per_sample
mkdir -p $OUT results_cvpr_final/baselines/logs
for f in $IN/*.jsonl; do b=$(basename $f .jsonl); [ -f $OUT/$b.csv ] || echo $f; done | xargs -P 18 -I{} bash -c 'b=$(basename {} .jsonl); python scripts/run_baselines_on_jsonl.py {} '$OUT'/$b.csv > results_cvpr_final/baselines/logs/$b.log 2>&1'
