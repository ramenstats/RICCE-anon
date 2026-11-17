"""v2 re-analysis of the E07 human-correlation study.

Addresses a flagged issue: the original correlation table (see
results/mapper_v1/human_metric_correlation/correlations_refined.json) reported
CE-F1 vs human_overall Spearman r=0.122, p=0.034 as the only "significant"
metric among 5 tested, WITHOUT correcting for the multiple-comparison family.
This script re-derives every number from the real per-item data
(results/E07_human_pred_validation/per_item.json, n=300) and adds:

  A. Multiple-comparison correction (Holm + BH-FDR across the 5-metric family,
     for Pearson/Spearman/Kendall each).
  B. 95% paired bootstrap CIs (>=5000 resamples) for Pearson/Spearman/Kendall.
  C. Within-dataset (LEVIR/SECOND/DUBAI) recomputation + macro-average, to
     check whether the pooled n=300 correlation is a real within-dataset
     relationship or a between-dataset confound (Simpson's-paradox-style).
  D. Within-model (per-system) recomputation + macro-average (n~=60 cells,
     explicitly flagged as thin).
  E. Controlled OLS regression: human_overall ~ ce_f1 + C(dataset) + C(system).
  F. System-ranking robustness: pairwise Spearman among CE-F1/CIDEr-D/
     BERTScore/SPICE/LLM-judge recomputed EXCLUDING the 2 collapsed/
     degenerate checkpoints (dpo_seed42, tcpo_seed42), compared side-by-side
     with the original all-7-systems numbers already in
     results/metric_comparison/ranking_correlation.json.

No number here is asserted without being computed from the underlying files
loaded below. Output: results_r1/human_statistics/report.json
"""
import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results_r1" / "human_statistics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 0
N_BOOT = 10000
METRICS = ["ce_f1", "bleu4", "rouge_l", "meteor", "bertscore"]

# ---------------------------------------------------------------------------
# Load real per-item data
# ---------------------------------------------------------------------------
per_item_path = ROOT / "results" / "E07_human_pred_validation" / "per_item.json"
items = json.load(open(per_item_path))
df = pd.DataFrame(items)
assert len(df) == 300, f"expected 300 rows, got {len(df)}"
print(f"Loaded {len(df)} rows from {per_item_path}")
print("Datasets:", df["dataset"].value_counts().to_dict())
print("Systems:", df["system"].value_counts().to_dict())


def paired_bootstrap_ci(x, y, stat_fn, n_boot=N_BOOT, seed=RNG_SEED, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(x)
    x = np.asarray(x)
    y = np.asarray(y)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        xb, yb = x[idx], y[idx]
        if np.std(xb) == 0 or np.std(yb) == 0:
            boots[b] = np.nan
        else:
            boots[b] = stat_fn(xb, yb)
    boots = boots[~np.isnan(boots)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), int(len(boots))


def pearson_stat(x, y):
    return stats.pearsonr(x, y)[0]


def spearman_stat(x, y):
    return stats.spearmanr(x, y)[0]


def kendall_stat(x, y):
    return stats.kendalltau(x, y)[0]


def holm_correction(pvals):
    """Holm-Bonferroni step-down correction. Returns adjusted p-values and
    reject flags at alpha=0.05, in the ORIGINAL order of `pvals`."""
    pvals = np.asarray(pvals)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        factor = m - rank
        val = pvals[idx] * factor
        running_max = max(running_max, val)
        adj[idx] = min(running_max, 1.0)
    reject = adj < 0.05
    return adj, reject


def bh_fdr_correction(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values (q-values)
    and reject flags in the ORIGINAL order of `pvals`."""
    pvals = np.asarray(pvals)
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * m / (np.arange(m) + 1)
    # enforce monotonicity from the largest p-value down
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)
    adj = np.empty(m)
    adj[order] = q
    reject = adj < alpha
    return adj, reject


report = {}

# ---------------------------------------------------------------------------
# A + B: correlations, raw p-values (Pearson/Spearman/Kendall), bootstrap CI,
#        Holm + BH-FDR correction across the 5-metric family
# ---------------------------------------------------------------------------
human = df["human_overall"].values
corr_table = {}
for m in METRICS:
    x = df[m].values
    pear_r, pear_p = stats.pearsonr(x, human)
    spear_r, spear_p = stats.spearmanr(x, human)
    kend_r, kend_p = stats.kendalltau(x, human)
    pear_lo, pear_hi, pear_nb = paired_bootstrap_ci(x, human, pearson_stat)
    spear_lo, spear_hi, spear_nb = paired_bootstrap_ci(x, human, spearman_stat)
    kend_lo, kend_hi, kend_nb = paired_bootstrap_ci(x, human, kendall_stat)
    corr_table[m] = {
        "pearson_r": float(pear_r), "pearson_p": float(pear_p),
        "pearson_ci95": [pear_lo, pear_hi], "pearson_n_boot_valid": pear_nb,
        "spearman_r": float(spear_r), "spearman_p": float(spear_p),
        "spearman_ci95": [spear_lo, spear_hi], "spearman_n_boot_valid": spear_nb,
        "kendall_tau": float(kend_r), "kendall_p": float(kend_p),
        "kendall_ci95": [kend_lo, kend_hi], "kendall_n_boot_valid": kend_nb,
        "n": int(len(x)),
    }

# multiple comparison correction, separately per correlation type, across the
# 5-metric family (order = METRICS, fixed)
mc_correction = {}
for corr_type, pkey in [("pearson", "pearson_p"), ("spearman", "spearman_p"), ("kendall", "kendall_p")]:
    raw_p = [corr_table[m][pkey] for m in METRICS]
    holm_adj, holm_rej = holm_correction(raw_p)
    bh_adj, bh_rej = bh_fdr_correction(raw_p)
    mc_correction[corr_type] = {}
    for i, m in enumerate(METRICS):
        mc_correction[corr_type][m] = {
            "raw_p": raw_p[i],
            "holm_adj_p": float(holm_adj[i]),
            "holm_significant_at_0.05": bool(holm_rej[i]),
            "bh_fdr_adj_p": float(bh_adj[i]),
            "bh_fdr_significant_at_0.05": bool(bh_rej[i]),
        }

report["A_B_correlations_and_correction"] = {
    "per_metric": corr_table,
    "multiple_comparison_correction": mc_correction,
    "n_bootstrap_resamples": N_BOOT,
}

cef1_spearman_p = corr_table["ce_f1"]["spearman_p"]
cef1_holm_adj = mc_correction["spearman"]["ce_f1"]["holm_adj_p"]
cef1_holm_sig = mc_correction["spearman"]["ce_f1"]["holm_significant_at_0.05"]
print(f"\n[A] CE-F1 vs human_overall Spearman: r={corr_table['ce_f1']['spearman_r']:.4f}, "
      f"raw p={cef1_spearman_p:.4f}, Holm-adj p={cef1_holm_adj:.4f}, "
      f"survives Holm @0.05: {cef1_holm_sig}")

# ---------------------------------------------------------------------------
# C. Within-dataset analysis (Simpson's-paradox check)
# ---------------------------------------------------------------------------
within_dataset = {}
for ds, sub in df.groupby("dataset"):
    x = sub["ce_f1"].values
    y = sub["human_overall"].values
    sp_r, sp_p = stats.spearmanr(x, y)
    pe_r, pe_p = stats.pearsonr(x, y)
    ke_r, ke_p = stats.kendalltau(x, y)
    within_dataset[ds] = {
        "n": int(len(sub)),
        "spearman_r": float(sp_r), "spearman_p": float(sp_p),
        "pearson_r": float(pe_r), "pearson_p": float(pe_p),
        "kendall_tau": float(ke_r), "kendall_p": float(ke_p),
        "mean_ce_f1": float(x.mean()), "mean_human_overall": float(y.mean()),
    }
macro_avg_spearman = float(np.mean([within_dataset[d]["spearman_r"] for d in within_dataset]))
pooled_spearman = corr_table["ce_f1"]["spearman_r"]
report["C_within_dataset"] = {
    "per_dataset": within_dataset,
    "macro_average_spearman_r": macro_avg_spearman,
    "pooled_n300_spearman_r": pooled_spearman,
    "between_dataset_means_ce_f1": {d: within_dataset[d]["mean_ce_f1"] for d in within_dataset},
    "between_dataset_means_human_overall": {d: within_dataset[d]["mean_human_overall"] for d in within_dataset},
    "interpretation": (
        "If macro-average within-dataset r is much smaller than (or opposite "
        "sign to) the pooled r, the pooled correlation is largely a between-"
        "dataset confound (datasets that get higher CE-F1 also happen to get "
        "higher human ratings, or vice versa) rather than a genuine item-level "
        "relationship -- computed and reported honestly below, not assumed."
    ),
}
print(f"\n[C] Pooled Spearman r={pooled_spearman:.4f} vs within-dataset macro-avg r={macro_avg_spearman:.4f}")
for d in within_dataset:
    print(f"    {d}: n={within_dataset[d]['n']}, spearman_r={within_dataset[d]['spearman_r']:.4f}, p={within_dataset[d]['spearman_p']:.4f}")

# ---------------------------------------------------------------------------
# D. Within-model (per-system) analysis
# ---------------------------------------------------------------------------
within_system = {}
for sysname, sub in df.groupby("system"):
    x = sub["ce_f1"].values
    y = sub["human_overall"].values
    n = len(sub)
    if n >= 3 and np.std(x) > 0 and np.std(y) > 0:
        sp_r, sp_p = stats.spearmanr(x, y)
        pe_r, pe_p = stats.pearsonr(x, y)
    else:
        sp_r = sp_p = pe_r = pe_p = float("nan")
    within_system[sysname] = {
        "n": int(n),
        "spearman_r": float(sp_r) if sp_r == sp_r else None,
        "spearman_p": float(sp_p) if sp_p == sp_p else None,
        "pearson_r": float(pe_r) if pe_r == pe_r else None,
        "pearson_p": float(pe_p) if pe_p == pe_p else None,
        "mean_ce_f1": float(x.mean()), "mean_human_overall": float(y.mean()),
    }
valid_sp = [within_system[s]["spearman_r"] for s in within_system if within_system[s]["spearman_r"] is not None]
macro_avg_system_spearman = float(np.mean(valid_sp)) if valid_sp else None
report["D_within_system"] = {
    "per_system": within_system,
    "macro_average_spearman_r": macro_avg_system_spearman,
    "n_systems_with_valid_stat": len(valid_sp),
    "limitation": (
        "Per-system n ranges 20-60 (see counts above); at this sample size "
        "individual within-system correlation estimates carry wide "
        "uncertainty and should not be over-interpreted individually -- only "
        "the macro-average across systems is reported as a summary, and even "
        "that is thin evidence given so few items per cell."
    ),
}
print(f"\n[D] Within-system macro-average Spearman r={macro_avg_system_spearman}")
for s in within_system:
    print(f"    {s}: n={within_system[s]['n']}, spearman_r={within_system[s]['spearman_r']}")

# ---------------------------------------------------------------------------
# E. Controlled OLS regression: human_overall ~ ce_f1 + C(dataset) + C(system)
# ---------------------------------------------------------------------------
reg_df = df.copy()
model_full = smf.ols("human_overall ~ ce_f1 + C(dataset) + C(system)", data=reg_df).fit()
model_bivariate = smf.ols("human_overall ~ ce_f1", data=reg_df).fit()

cef1_coef = float(model_full.params["ce_f1"])
cef1_se = float(model_full.bse["ce_f1"])
cef1_p = float(model_full.pvalues["ce_f1"])
cef1_ci = model_full.conf_int().loc["ce_f1"].tolist()

report["E_controlled_regression"] = {
    "formula": "human_overall ~ ce_f1 + C(dataset) + C(system)",
    "n": int(model_full.nobs),
    "r_squared": float(model_full.rsquared),
    "adj_r_squared": float(model_full.rsquared_adj),
    "ce_f1_coefficient": cef1_coef,
    "ce_f1_se": cef1_se,
    "ce_f1_p_value": cef1_p,
    "ce_f1_ci95": [float(cef1_ci[0]), float(cef1_ci[1])],
    "bivariate_comparison": {
        "formula": "human_overall ~ ce_f1",
        "r_squared": float(model_bivariate.rsquared),
        "ce_f1_coefficient": float(model_bivariate.params["ce_f1"]),
        "ce_f1_p_value": float(model_bivariate.pvalues["ce_f1"]),
    },
    "caveat": (
        "This is a controlled ASSOCIATION check (does CE-F1 retain a "
        "statistically detectable relationship with human_overall once "
        "dataset- and system-level mean differences are partialled out via "
        "fixed effects), NOT a causal claim. OLS on ordinal human ratings "
        "with dataset/system fixed effects does not establish that CE-F1 "
        "causes higher human ratings, only whether the within-group "
        "association survives controlling for these observed confounders."
    ),
}
print(f"\n[E] OLS human_overall ~ ce_f1 + C(dataset) + C(system): "
      f"ce_f1 coef={cef1_coef:.4f}, p={cef1_p:.4f}, R2={model_full.rsquared:.4f} "
      f"(bivariate ce_f1-only R2={model_bivariate.rsquared:.4f})")

# ---------------------------------------------------------------------------
# F. System-ranking robustness excluding collapsed/degenerate checkpoints
# ---------------------------------------------------------------------------
ranking_path = ROOT / "results" / "metric_comparison" / "ranking_correlation.json"
ranking_data = json.load(open(ranking_path))
raw_scores = ranking_data["raw_scores"]
all_systems = list(raw_scores.keys())
collapsed = ["dpo_seed42", "tcpo_seed42"]
non_collapsed = [s for s in all_systems if s not in collapsed]
assert set(non_collapsed) == {"qwen", "sft_seed42", "chg2cap", "swmcc"}, non_collapsed

metric_keys = ranking_data["metrics_available"]  # cider_d, ricce_f, bertscore, spice, llm_judge


def pairwise_spearman_for(systems):
    out = {}
    for a, b in itertools.combinations(metric_keys, 2):
        xa = [raw_scores[s][a] for s in systems]
        xb = [raw_scores[s][b] for s in systems]
        rho, p = stats.spearmanr(xa, xb)
        out[f"{a}_vs_{b}"] = {"spearman_rho": float(rho) if rho == rho else None,
                               "p_value": float(p) if p == p else None, "n": len(systems)}
    return out


all7 = pairwise_spearman_for(all_systems)
non_collapsed5 = pairwise_spearman_for(non_collapsed)

report["F_ranking_robustness"] = {
    "all_7_systems": {"systems": all_systems, "pairwise_spearman": all7},
    "5_non_collapsed_systems": {"systems": non_collapsed, "pairwise_spearman": non_collapsed5},
    "collapsed_systems_excluded": collapsed,
    "collapse_evidence": {s: raw_scores[s] for s in collapsed},
    "note": (
        "dpo_seed42 and tcpo_seed42 have ricce_f=0.0 and cider_d=0.0/near-0 "
        "(degenerate/constant-output checkpoints -- see raw_scores above), which "
        "places them as extreme outliers at the bottom of every metric's ranking "
        "simultaneously and can trivially inflate rank-correlation agreement "
        "between metrics. The 5-system numbers isolate whether the metrics still "
        "agree on ranking among genuinely-varying, non-degenerate systems."
    ),
}
print("\n[F] ricce_f vs cider_d Spearman rho: "
      f"all-7={all7['cider_d_vs_ricce_f']['spearman_rho']:.4f} (p={all7['cider_d_vs_ricce_f']['p_value']:.4f}) "
      f"vs 5-non-collapsed={non_collapsed5['cider_d_vs_ricce_f']['spearman_rho']}"
      f" (p={non_collapsed5['cider_d_vs_ricce_f']['p_value']})")
for k in ["ricce_f_vs_bertscore", "ricce_f_vs_spice", "ricce_f_vs_llm_judge"]:
    print(f"    {k}: all-7 rho={all7[k]['spearman_rho']:.4f} | 5-non-collapsed rho={non_collapsed5[k]['spearman_rho']}")

# ---------------------------------------------------------------------------
# Bottom-line verdict
# ---------------------------------------------------------------------------
report["bottom_line"] = {
    "original_claim": "CE-F1 is the only metric with a statistically significant correlation with human overall rating (Spearman r=0.122, p=0.034), among BLEU-4/ROUGE-L/METEOR/BERTScore/CE-F1.",
    "raw_p_reproduced": cef1_spearman_p,
    "holm_adjusted_p": cef1_holm_adj,
    "survives_holm_correction_at_0.05": cef1_holm_sig,
    "survives_bh_fdr_at_0.05": mc_correction["spearman"]["ce_f1"]["bh_fdr_significant_at_0.05"],
}

with open(OUT_DIR / "report.json", "w") as f:
    json.dump(report, f, indent=2)
print(f"\nWrote {OUT_DIR / 'report.json'}")
print("\nBOTTOM LINE:", json.dumps(report["bottom_line"], indent=2))
