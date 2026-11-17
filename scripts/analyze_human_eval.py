#!/usr/bin/env python
"""Analyse exported annotator CSVs. Prints 'PENDING HUMAN ANNOTATION' and exits 0 (writing nothing) if no real ratings exist."""
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.evaluation.human_analysis import run_analysis, PENDING

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collected", default=str(ROOT / "artifacts_cvpr_final/human_eval/image_grounded/collected"))
    ap.add_argument("--key", default=str(ROOT / "artifacts_cvpr_final/human_eval/image_grounded/hidden/answer_key.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results_cvpr_final/human_eval"))
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--kendall-resamples", type=int, default=2000)
    a = ap.parse_args()
    s = run_analysis(a.collected, a.key, a.out, B=a.resamples, kendall_B=a.kendall_resamples)
    if s["status"] == PENDING:
        print(PENDING); return 0
    print(f"analysed: {s['study_label']}; results in {s['out_dir']}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
