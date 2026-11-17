"""Paired bootstrap CI + Wilcoxon significance test for the paper's central
claim (LoRA SFT vs. the frozen zero-shot baseline on LEVIR-CC), plus a
Holm-corrected family for every planned pairwise comparison in the broad
benchmark (internal design review, Phase I). Reads only already-
generated per_sample_metrics.csv files -- no new inference, no new parsing.

Usage: python scripts/run_significance.py [DATASET]
DATASET defaults to LEVIR_CC (preserves the original output path
results/significance/significance.json for backward compatibility). Any
other dataset name (e.g. SECOND_CC, DUBAI_CC) reads
results/E08/<DATASET>/<system>/per_sample_metrics.csv and writes to
results/significance/<DATASET>/significance.json, following the same
per-dataset directory pattern used elsewhere under results/, so it never
overwrites the existing LEVIR-CC significance.json.
"""
import json
import sys
from itertools import combinations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.analysis.stats import bootstrap_ci, paired_bootstrap_difference, wilcoxon_signed_rank, holm_correction

DATASET = sys.argv[1] if len(sys.argv) > 1 else "LEVIR_CC"
SYSTEMS = ["qwen", "sft_seed42", "dpo_seed42", "tcpo_seed42", "chg2cap", "swmcc"]
OUT = ROOT / "results" / "significance" if DATASET == "LEVIR_CC" else ROOT / "results" / "significance" / DATASET
OUT.mkdir(parents=True, exist_ok=True)


def load(name):
    p = ROOT / "results" / "E08" / DATASET / name / "per_sample_metrics.csv"
    if not p.exists():
        return None
    return pd.read_csv(p).set_index("image_id")["ce_f1"]


series = {s: load(s) for s in SYSTEMS}
available = [s for s in SYSTEMS if series[s] is not None]
print("available systems:", available)

results = {}
for a, b in combinations(available, 2):
    common = series[a].index.intersection(series[b].index)
    if len(common) == 0:
        continue
    va, vb = series[a].loc[common].values, series[b].loc[common].values
    diff = paired_bootstrap_difference(va, vb)
    wil = wilcoxon_signed_rank(va, vb)
    results[f"{a}_vs_{b}"] = {"bootstrap": diff, "wilcoxon": wil, "n": int(len(common))}
    print(a, "vs", b, "mean_diff", round(diff["mean_diff"], 4), "ci95", diff["ci95"], "p", round(diff["p_value"], 4))

if results:
    pvals = [v["bootstrap"]["p_value"] for v in results.values()]
    holm = holm_correction(pvals)
    for (k, v), adj, rej in zip(results.items(), holm["adjusted_p"], holm["reject_at_0.05"]):
        v["holm_adjusted_p"] = adj
        v["holm_reject_at_0.05"] = rej

for s in available:
    lo, hi = bootstrap_ci(series[s].values)
    point = float(series[s].mean())
    results.setdefault("marginals", {})[s] = {"point": point, "lo": lo, "hi": hi}
    print(s, "CE-F 95% CI:", round(lo, 4), "-", round(hi, 4), "point", round(point, 4))

(OUT / "significance.json").write_text(json.dumps(results, indent=2))
print("wrote", OUT / "significance.json")
