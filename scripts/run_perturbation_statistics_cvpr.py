"""v2 perturbation uncertainty (README S12).

The existing E04 controlled-perturbation sanity check (results/E04/*/
per_sample_metrics.csv, paper Table tab:sanity / TblEzerofourFull) reports a
single mean CE-F1 per perturbation type per dataset with no uncertainty
quantification and no significance test against the unperturbed baseline.
This adds, per dataset, per perturbation type:

  - paired bootstrap 95% CI (>=5000 resamples) on mean(CE-F1_perturbed -
    CE-F1_original), paired on image_id (only items present for BOTH the
    perturbation type and P0_original are used -- most perturbation types
    (P1-P7) only apply to a subset of reference captions where that
    perturbation is well-defined, e.g. P1_ancestor only fires where the
    taxonomy has an ancestor to substitute)
  - a paired bootstrap two-sided p-value for that difference
  - Holm correction across the family of 9 non-trivial perturbation types
    (P1_ancestor .. P9_change_state_reversal) per dataset -- P0_paraphrase is
    reported separately as a semantic-preserving sanity check (it should NOT
    move the score much, so it is not part of the "does the metric detect
    corruption" family being multiplicity-corrected)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results_r1" / "perturbation_statistics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOT = 5000
SEED = 0
DATASETS = ["LEVIR_CC", "SECOND_CC", "DUBAI_CC"]
NON_TRIVIAL_TYPES = [
    "P1_ancestor", "P2_distant_ancestor", "P3_sibling", "P4_action_reversal",
    "P5_location_swap", "P6_count_corruption", "P7_event_omission",
    "P8_hallucination", "P9_change_state_reversal",
]


def holm_correction(pvals):
    pvals = np.asarray(pvals)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = pvals[idx] * (m - rank)
        running_max = max(running_max, val)
        adj[idx] = min(running_max, 1.0)
    return adj


def paired_bootstrap(diffs, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs)
    n = len(diffs)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = diffs[idx].mean()
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # two-sided bootstrap p-value: 2 * min(P(boot<=0), P(boot>=0))
    p_le = np.mean(boots <= 0)
    p_ge = np.mean(boots >= 0)
    p = min(1.0, 2 * min(p_le, p_ge))
    return float(lo), float(hi), float(p), float(diffs.mean())


report = {}
for ds in DATASETS:
    csv_path = ROOT / "results" / "E04" / ds / "per_sample_metrics.csv"
    df = pd.read_csv(csv_path)
    orig = df[df["type"] == "P0_original"][["image_id", "ce_f1"]].rename(columns={"ce_f1": "ce_f1_orig"})
    ds_report = {}

    # paraphrase, reported separately (sanity check, not corruption family)
    para = df[df["type"] == "P0_paraphrase"][["image_id", "ce_f1"]]
    merged = para.merge(orig, on="image_id", how="inner")
    diffs = (merged["ce_f1"] - merged["ce_f1_orig"]).values
    lo, hi, p, mean_diff = paired_bootstrap(diffs)
    ds_report["P0_paraphrase"] = {
        "n_paired": int(len(merged)), "mean_ce_f1": float(merged["ce_f1"].mean()),
        "mean_diff_vs_original": mean_diff, "ci95": [lo, hi], "bootstrap_p": p,
    }

    raw_p = []
    per_type = {}
    for ptype in NON_TRIVIAL_TYPES:
        sub = df[df["type"] == ptype][["image_id", "ce_f1"]]
        merged = sub.merge(orig, on="image_id", how="inner")
        if len(merged) == 0:
            per_type[ptype] = {"n_paired": 0, "note": "no items in this dataset for this perturbation type"}
            raw_p.append(1.0)
            continue
        diffs = (merged["ce_f1"] - merged["ce_f1_orig"]).values
        lo, hi, p, mean_diff = paired_bootstrap(diffs)
        per_type[ptype] = {
            "n_paired": int(len(merged)), "mean_ce_f1": float(merged["ce_f1"].mean()),
            "mean_ce_f1_original_matched": float(merged["ce_f1_orig"].mean()),
            "mean_diff_vs_original": mean_diff, "ci95": [lo, hi], "bootstrap_p": p,
        }
        raw_p.append(p)

    holm_adj = holm_correction(raw_p)
    for i, ptype in enumerate(NON_TRIVIAL_TYPES):
        per_type[ptype]["holm_adj_p"] = float(holm_adj[i])
        per_type[ptype]["holm_significant_at_0.05"] = bool(holm_adj[i] < 0.05)

    ds_report["non_trivial_perturbations"] = per_type
    report[ds] = ds_report

    print(f"\n=== {ds} ===")
    print(f"  P0_paraphrase: n={ds_report['P0_paraphrase']['n_paired']}, "
          f"mean diff={ds_report['P0_paraphrase']['mean_diff_vs_original']:.4f}, "
          f"CI={ds_report['P0_paraphrase']['ci95']}, p={ds_report['P0_paraphrase']['bootstrap_p']:.4f}")
    for ptype in NON_TRIVIAL_TYPES:
        d = per_type[ptype]
        if d.get("n_paired", 0) == 0:
            print(f"  {ptype}: no paired items")
            continue
        print(f"  {ptype}: n={d['n_paired']}, mean_ce_f1={d['mean_ce_f1']:.3f}, "
              f"diff={d['mean_diff_vs_original']:.3f}, CI95={[round(x,3) for x in d['ci95']]}, "
              f"raw_p={d['bootstrap_p']:.4f}, holm_p={d['holm_adj_p']:.4f}, "
              f"sig@0.05={d['holm_significant_at_0.05']}")

with open(OUT_DIR / "report.json", "w") as f:
    json.dump(report, f, indent=2)
print(f"\nWrote {OUT_DIR / 'report.json'}")
