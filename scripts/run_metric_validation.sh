#!/usr/bin/env bash
set -euo pipefail
for ds in LEVIR_CC SECOND_CC DUBAI_CC RSCC; do
  m="artifacts/manifests/${ds}.jsonl"
  [ -f "$m" ] || continue
  python3 -m ricce.cli experiment E02 "$m" --outdir "results/E02/${ds}"
  python3 -m ricce.cli experiment E04 "$m" --outdir "results/E04/${ds}" --limit 1000
  python3 -m ricce.cli experiment E05 "$m" --outdir "results/E05/${ds}" --limit 1000
  python3 -m ricce.cli experiment E06 "$m" --outdir "results/E06/${ds}" --limit 1000
  python3 -m ricce.cli experiment E14 "$m" --outdir "results/E14/${ds}" --limit 1000
done
