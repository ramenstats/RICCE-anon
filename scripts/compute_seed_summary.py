"""Computes mean +/- std across seeds for SFT and any DPO/TCPO variant that
actually converged (required check: ">=3 seeds ... report mean +/-
std"). Reads only already-scored results/E08/LEVIR_CC/<variant>_seed<N>/
metrics.json files -- no new inference or scoring -- and never fabricates a
seed's numbers: a variant/seed combination with no metrics.json, or whose
predictions collapsed into degenerate repetition (same check as
a detect_collapse check, frac_top>0.9), is recorded as
missing/collapsed and excluded from the mean+/-std rather than silently
imputed or dropped without a trace.

Usage: python scripts/compute_seed_summary.py
Writes results/E08/LEVIR_CC/seed_summary.json
"""
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E08 = ROOT / "results" / "E08" / "LEVIR_CC"
PRED = ROOT / "results" / "baselines"

VARIANTS = {
    "sft": [42, 1, 123],
    "dpo": [42],
    "tcpo": [42],
    "dpo_reg": [42],
    "tcpo_v3": [42],
    "tcpo_random_severity": [42],
    "tcpo_no_severity": [42],
}

NUMERIC_KEYS = ["ce_precision", "ce_recall", "ce_f1", "hallucination_rate", "omission_rate",
                "bleu1", "bleu2", "bleu3", "bleu4", "meteor", "rougeL",
                "cider_diagnostic", "cider_d", "cider_d_corpus"]


def detect_collapse(name):
    pf = PRED / name / "LEVIR_CC_test.jsonl"
    if not pf.exists():
        return None  # unknown, no predictions available
    preds = [json.loads(l).get("prediction", "") for l in pf.read_text().splitlines() if l.strip()]
    if not preds:
        return None
    top, n = Counter(preds).most_common(1)[0]
    frac_top = Counter(preds).most_common(1)[0][1] / n
    return frac_top > 0.9


def load_metrics(name):
    p = E08 / name / "metrics.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if "status" in d:  # pending/not_auto_executable placeholder, not real metrics
        return None
    return d


summary = {}
for variant, seeds in VARIANTS.items():
    per_seed = {}
    values = {k: [] for k in NUMERIC_KEYS}
    for seed in seeds:
        name = f"{variant}_seed{seed}"
        collapsed = detect_collapse(name)
        metrics = load_metrics(name)
        status = "missing"
        if metrics is not None:
            status = "collapsed" if collapsed else "scored"
        elif collapsed:
            status = "collapsed_unscored"
        per_seed[str(seed)] = {"status": status, "metrics": metrics}
        if metrics is not None and not collapsed:
            for k in NUMERIC_KEYS:
                if k in metrics and metrics[k] is not None:
                    values[k].append(metrics[k])
    n_ok = max((len(v) for v in values.values()), default=0)
    agg = {}
    for k, vs in values.items():
        if not vs:
            continue
        agg[k] = {
            "mean": statistics.fmean(vs),
            "std": statistics.stdev(vs) if len(vs) >= 2 else None,
            "n": len(vs),
        }
    summary[variant] = {"seeds_configured": seeds, "n_converged": n_ok, "per_seed": per_seed, "mean_std": agg}

out = E08 / "seed_summary.json"
out.write_text(json.dumps(summary, indent=2))
print(f"Wrote {out}")
for variant, d in summary.items():
    print(variant, "converged seeds:", d["n_converged"], "/", len(d["seeds_configured"]))
