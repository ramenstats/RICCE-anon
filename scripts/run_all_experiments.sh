#!/usr/bin/env bash
set -euo pipefail
bash scripts/prepare_all.sh
bash scripts/run_metric_validation.sh
for ds in LEVIR_CC SECOND_CC DUBAI_CC RSCC; do
  m="artifacts/manifests/${ds}.jsonl"
  [ -f "$m" ] || continue
  for e in E01 E02 E03 E04 E05 E06 E07 E08 E09 E10 E11 E12 E13 E14 E15 E16 E17 E18 E19 E20; do
    [ -d "results/${e}/${ds}" ] || python3 -m ricce.cli experiment "$e" "$m" --outdir "results/${e}/${ds}" || true
  done
done
echo "Automatic stages complete. Pending artifacts explicitly identify experiments requiring human/external-model inputs."
