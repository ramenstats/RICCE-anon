"""Analysis of image-grounded, multi-annotator human ratings.

Only real exported annotation CSVs (schema in human_protocol.CSV_COLUMNS) may be analysed.
When none exist, `run_analysis` returns status 'PENDING HUMAN ANNOTATION' and writes NOTHING.
"""
from __future__ import annotations
import glob, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from .human_protocol import AXES, CSV_COLUMNS

PENDING = "PENDING HUMAN ANNOTATION"
SINGLE_LABEL = "single-annotator, no reliability estimate"
META_KEYS = ("ce_p", "ce_r")

# ------------------------------------------------------------------ loading
def load_annotations(paths):
    frames = []
    for p in paths:
        df = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")  # utf-8-sig: Excel "CSV UTF-8" writes a BOM
        missing = [c for c in ("annotator_id", "task", "item_id") if c not in df.columns]
        if missing:
            raise ValueError(f"{p}: missing columns {missing}")
        for c in CSV_COLUMNS:
            if c not in df.columns: df[c] = ""
        frames.append(df[CSV_COLUMNS])
    if not frames: return pd.DataFrame(columns=CSV_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(["annotator_id", "task", "item_id"], keep="last")   # re-exports overwrite earlier partial exports
    for a in AXES:
        df[a] = pd.to_numeric(df[a].replace({"NA": np.nan, "": np.nan}), errors="coerce")
    df["seconds_spent"] = pd.to_numeric(df["seconds_spent"], errors="coerce")
    return df

# Captioning systems whose outputs enter the human-study analysis; constructed cases carry no system.
STUDY_SYSTEMS = ("qwen", "sft_seed42", "chg2cap", "swmcc")

def in_study(r, systems=STUDY_SYSTEMS):
    """True if every system a key row names (an item, or both sides of a pair) is in `systems`."""
    return all(r.get(k) is None or r.get(k) in systems for k in ("system", "system_a", "system_b"))

def load_key(path, systems=None):
    rows = [json.loads(l) for l in open(path)]
    if systems is not None:
        rows = [r for r in rows if in_study(r, systems)]
    items = {r["item_id"]: r for r in rows if r["kind"] == "item"}
    pairs = {r["item_id"]: r for r in rows if r["kind"] == "pair"}
    return items, pairs

def has_real_ratings(df):
    if df is None or len(df) == 0: return False
    a = df[df.task == "A"][AXES].notna().any(axis=1).any()
    b = df[df.task == "B"]["preference"].isin(["A", "B", "Tie", "CannotDetermine"]).any()
    return bool(a or b)

# ------------------------------------------------------------------ reliability
def matrix(df, col, task):
    d = df[df.task == task].pivot(index="annotator_id", columns="item_id", values=col)
    return d

def krippendorff_ordinal(m):
    """m: DataFrame annotators x items. Returns NaN with <2 annotators."""
    if m.shape[0] < 2: return float("nan")
    import krippendorff
    arr = m.to_numpy(dtype=float)
    keep = (~np.isnan(arr)).sum(axis=0) >= 2
    if keep.sum() < 2: return float("nan")
    arr = arr[:, keep]
    if len(np.unique(arr[~np.isnan(arr)])) < 2: return float("nan")   # no variation: alpha undefined
    return float(krippendorff.alpha(reliability_data=arr, level_of_measurement="ordinal"))

def fleiss_kappa_from_labels(m):
    """m: DataFrame annotators x items of category labels (NaN missing). Uses items rated by ALL annotators."""
    from statsmodels.stats.inter_rater import fleiss_kappa
    if m.shape[0] < 2: return float("nan"), 0
    full = m.dropna(axis=1, how="any")
    if full.shape[1] < 2: return float("nan"), full.shape[1]
    cats = sorted(pd.unique(full.to_numpy().ravel()))
    table = np.array([[int((full[c] == k).sum()) for k in cats] for c in full.columns])
    if len(cats) < 2: return float("nan"), full.shape[1]
    return float(fleiss_kappa(table, method="fleiss")), full.shape[1]

def icc_two_way(x):
    """x: items x raters complete matrix. Returns (ICC(2,1), ICC(2,k)) absolute agreement (Shrout & Fleiss)."""
    x = np.asarray(x, float); n, k = x.shape
    if n < 2 or k < 2: return float("nan"), float("nan")
    gm = x.mean()
    ssr = k * ((x.mean(1) - gm) ** 2).sum(); ssc = n * ((x.mean(0) - gm) ** 2).sum()
    sst = ((x - gm) ** 2).sum(); sse = sst - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    d1 = msr + (k - 1) * mse + k * (msc - mse) / n
    icc1 = (msr - mse) / d1 if d1 > 0 else float("nan")
    d2 = msr + (msc - mse) / n
    icck = (msr - mse) / d2 if d2 > 0 else float("nan")
    return float(icc1), float(icck)

def reliability_table(df):
    n_ann = df.annotator_id.nunique()
    rows = []
    if n_ann < 2:
        return pd.DataFrame([dict(measure="all", value=np.nan, note=SINGLE_LABEL)]), n_ann
    for a in AXES:
        m = matrix(df, a, "A")
        if m.shape[1] == 0: continue
        rows.append(dict(measure=f"krippendorff_alpha_ordinal:{a}", value=krippendorff_ordinal(m), n_items=int((m.notna().sum() >= 2).sum()), note=""))
    m = matrix(df, "overall", "A").dropna(axis=1, how="any")
    i1, ik = icc_two_way(m.T.to_numpy()) if m.shape[1] else (np.nan, np.nan)
    rows += [dict(measure="ICC2_1:overall", value=i1, n_items=m.shape[1], note="items rated by all annotators"),
             dict(measure="ICC2_k:overall(mean rating)", value=ik, n_items=m.shape[1], note="items rated by all annotators")]
    pm = matrix(df, "preference", "B").replace("", np.nan)
    if pm.shape[1]:
        k, n = fleiss_kappa_from_labels(pm)
        rows.append(dict(measure="fleiss_kappa:pairwise_preference", value=k, n_items=n, note="items rated by all annotators"))
        pm2 = pm.where(pm.isin(["A", "B"]))
        k2, n2 = fleiss_kappa_from_labels(pm2)
        rows.append(dict(measure="fleiss_kappa:pairwise_preference_AvsB_only", value=k2, n_items=n2, note="Tie/Cannot -> missing; items with all annotators A/B"))
    return pd.DataFrame(rows), n_ann

# ------------------------------------------------------------------ consensus / distributions
def consensus_items(df):
    a = df[df.task == "A"]
    g = a.groupby("item_id")
    out = g[AXES].mean().add_suffix("_mean")
    out = out.join(g[AXES].median().add_suffix("_median")).join(g.annotator_id.nunique().rename("n_annotators"))
    return out.reset_index()

def annotator_distributions(df):
    rows = []
    for aid, g in df.groupby("annotator_id"):
        a = g[g.task == "A"]
        for ax in AXES:
            v = a[ax].dropna()
            row = dict(annotator_id=aid, task="A", field=ax, n=len(v), mean=v.mean() if len(v) else np.nan, sd=v.std() if len(v) > 1 else np.nan,
                       n_NA=int(a[ax].isna().sum()))
            for s in range(1, 6): row[f"pct_{s}"] = float((v == s).mean()) if len(v) else np.nan
            rows.append(row)
        b = g[g.task == "B"]
        for lab in ["A", "B", "Tie", "CannotDetermine"]:
            rows.append(dict(annotator_id=aid, task="B", field=f"pref={lab}", n=len(b), mean=float((b.preference == lab).mean()) if len(b) else np.nan))
        rows.append(dict(annotator_id=aid, task="all", field="median_seconds_per_item", n=int(g.seconds_spent.notna().sum()), mean=g.seconds_spent.median()))
    return pd.DataFrame(rows)

# ------------------------------------------------------------------ bootstrap correlation
def _corr(x, y, kind):
    if kind == "pearson":
        if x.std() == 0 or y.std() == 0: return np.nan
        return float(np.corrcoef(x, y)[0, 1])
    if kind == "spearman":
        rx, ry = stats.rankdata(x), stats.rankdata(y)
        if rx.std() == 0 or ry.std() == 0: return np.nan
        return float(np.corrcoef(rx, ry)[0, 1])
    r = stats.kendalltau(x, y).statistic
    return float(r) if r == r else np.nan

def p_display(p, B):
    return f"< {1.0 / (B + 1):.1e}" if p <= 1.0 / (B + 1) + 1e-15 else f"{p:.4f}"

def cluster_bootstrap_corr(x, y, clusters, kind="spearman", B=10000, seed=0):
    """Percentile CI + two-sided bootstrap p (=2*min tail, floor 1/(B+1)) resampling whole clusters (image pairs)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    est = _corr(x, y, kind)
    cl = pd.factorize(np.asarray(clusters))[0]
    idx = [np.where(cl == c)[0] for c in range(cl.max() + 1)]
    rng = np.random.default_rng(seed); vals = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(idx), len(idx))
        ii = np.concatenate([idx[j] for j in pick])
        vals[b] = _corr(x[ii], y[ii], kind)
    v = vals[~np.isnan(vals)]
    if len(v) == 0: return est, np.nan, np.nan, np.nan, len(idx)
    lo, hi = np.percentile(v, [2.5, 97.5])
    tail = min((v <= 0).sum(), (v >= 0).sum())
    p = min(1.0, 2 * (tail + 1) / (len(v) + 1))
    p = max(p, 1.0 / (len(v) + 1))
    return est, float(lo), float(hi), float(p), len(idx)

def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj

def bh(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p)[::-1]; adj = np.empty(m); prev = 1.0
    for r, i in enumerate(order):
        rank = m - r; prev = min(prev, p[i] * m / rank); adj[i] = prev
    return adj

def build_frame(consensus, key_items, dataset_col=True):
    """consensus overall + axis means joined with hidden key (metrics, system, dataset, cluster id)."""
    rows = []
    c = consensus.set_index("item_id")
    for iid, r in c.iterrows():
        k = key_items.get(iid)
        if k is None: continue
        row = dict(item_id=iid, dataset=k["dataset"], image_id=k["image_id"], cluster=f"{k['dataset']}:{k['image_id']}",
                   system=k["system"] or "perturbed", subset="model_output" if k["source"] == "E08_model_output" else "hard_case",
                   perturbation_type=k.get("perturbation_type"))
        for a in AXES: row["h_" + a] = r.get(a + "_mean", np.nan)
        for m, v in (k.get("metrics") or {}).items(): row["m_" + m] = v
        rows.append(row)
    return pd.DataFrame(rows)

def metric_cols(fr): return [c for c in fr.columns if c.startswith("m_")]

def assoc_table(fr, pairs_metric_target, B, seed, kendall_B, min_n=20, group=None):
    """pairs_metric_target: list of (metric_col, human_col). Returns DataFrame; Holm/BH applied within this table."""
    rows = []
    for mc, hc in pairs_metric_target:
        d = fr[[mc, hc, "cluster"]].dropna()
        if len(d) < min_n or d[mc].nunique() < 2 or d[hc].nunique() < 2:
            rows.append(dict(metric=mc[2:], human=hc[2:], n=len(d), note="insufficient data/variation")); continue
        r = dict(metric=mc[2:], human=hc[2:], n=len(d), n_clusters=d.cluster.nunique())
        for kind in ("spearman", "pearson", "kendall"):
            Bk = kendall_B if kind == "kendall" else B
            est, lo, hi, p, _ = cluster_bootstrap_corr(d[mc], d[hc], d.cluster, kind, Bk, seed)
            r.update({f"{kind}": est, f"{kind}_ci_lo": lo, f"{kind}_ci_hi": hi, f"{kind}_p_boot": p,
                      f"{kind}_p_display": p_display(p, Bk), f"{kind}_resamples": Bk})
        rows.append(r)
    t = pd.DataFrame(rows)
    if "spearman_p_boot" in t:
        ok = t.spearman_p_boot.notna()
        t.loc[ok, "spearman_p_holm"] = holm(t.loc[ok, "spearman_p_boot"])
        t.loc[ok, "spearman_p_bh"] = bh(t.loc[ok, "spearman_p_boot"])
    return t

# ------------------------------------------------------------------ regression
def regression_table(fr, metric):
    import statsmodels.formula.api as smf
    d = fr[["h_overall", "m_" + metric, "dataset", "system", "cluster"]].dropna().copy()
    if len(d) < 30 or d["m_" + metric].nunique() < 2: return pd.DataFrame()
    d["metric_z"] = (d["m_" + metric] - d["m_" + metric].mean()) / d["m_" + metric].std()
    d["cl"] = pd.factorize(d.cluster)[0]
    rows = []
    try:
        f = smf.ols("h_overall ~ metric_z + C(dataset) + C(system)", d).fit(cov_type="cluster", cov_kwds={"groups": d.cl})
        rows.append(dict(metric=metric, model="fixed_effects_OLS_cluster_robust", coef=f.params["metric_z"], se=f.bse["metric_z"],
                         ci_lo=f.conf_int().loc["metric_z", 0], ci_hi=f.conf_int().loc["metric_z", 1], p=f.pvalues["metric_z"], n=len(d)))
    except Exception as e:
        rows.append(dict(metric=metric, model="fixed_effects_OLS_cluster_robust", note=repr(e)))
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mm = smf.mixedlm("h_overall ~ metric_z + C(dataset) + C(system)", d, groups=d["cl"]).fit(reml=True)
        rows.append(dict(metric=metric, model="mixed_random_intercept_image", coef=mm.params["metric_z"], se=mm.bse["metric_z"],
                         ci_lo=mm.conf_int().loc["metric_z", 0], ci_hi=mm.conf_int().loc["metric_z", 1], p=mm.pvalues["metric_z"], n=len(d)))
    except Exception as e:
        rows.append(dict(metric=metric, model="mixed_random_intercept_image", note=repr(e)))
    out = pd.DataFrame(rows)
    if "p" in out: out["p_display"] = out.p.map(lambda p: "< 1e-16" if p == p and p < 1e-16 else (f"{p:.4g}" if p == p else ""))
    return out

# ------------------------------------------------------------------ pairwise
def pairwise_consensus(df):
    b = df[(df.task == "B") & df.preference.isin(["A", "B", "Tie", "CannotDetermine"])]
    rows = []
    for pid, g in b.groupby("item_id"):
        votes = g.preference.value_counts()
        strict = votes.reindex(["A", "B"]).fillna(0)
        if len(g) == 1: cons = g.preference.iloc[0]
        elif strict.max() > len(g) / 2: cons = strict.idxmax()          # strict majority of all raters
        elif votes.get("Tie", 0) > len(g) / 2: cons = "Tie"
        else: cons = "Unresolved"
        rows.append(dict(item_id=pid, n_annotators=len(g), human_pref=cons, votes_A=int(strict["A"]), votes_B=int(strict["B"]),
                         votes_Tie=int(votes.get("Tie", 0)), votes_Cannot=int(votes.get("CannotDetermine", 0))))
    return pd.DataFrame(rows)

def pair_accuracy(pc, key_pairs, metric, policy, B, seed, subset=None):
    """policy 'excluded': drop metric ties and human Tie/Cannot/Unresolved. 'half_credit': metric tie on a strict human pair = 0.5."""
    rows = []
    for _, r in pc.iterrows():
        k = key_pairs.get(r["item_id"])
        if k is None or r["human_pref"] not in ("A", "B"): continue
        mp = k["metric_preferences"].get(metric)
        if mp is None: continue
        if subset and not subset(k): continue
        if mp == "tie":
            if policy == "excluded": continue
            score = 0.5
        else: score = float(mp == r["human_pref"])
        rows.append((score, f"{k['dataset']}:{k['image_id']}"))
    if not rows: return dict(n=0)
    s = np.array([x[0] for x in rows]); cl = pd.factorize(np.array([x[1] for x in rows]))[0]
    idx = [np.where(cl == c)[0] for c in range(cl.max() + 1)]
    rng = np.random.default_rng(seed); vals = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(idx), len(idx)); vals[b] = s[np.concatenate([idx[j] for j in pick])].mean()
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return dict(n=len(s), accuracy=float(s.mean()), ci_lo=float(lo), ci_hi=float(hi), n_clusters=len(idx))

# ------------------------------------------------------------------ driver
def run_analysis(collected_dir, key_path, out_dir, B=10000, kendall_B=2000, seed=20260923, allow_synthetic=False):
    """Returns dict(status=..., ...). Writes files only when real ratings are present."""
    paths = sorted(glob.glob(str(Path(collected_dir) / "*.csv")))
    df = load_annotations(paths) if paths else pd.DataFrame()
    if not has_real_ratings(df):
        return dict(status=PENDING)
    if not allow_synthetic and df.annotator_id.str.upper().str.startswith("SYNTHETIC").any():
        raise ValueError("refusing to analyse annotator ids starting with SYNTHETIC outside tests")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    key_items, key_pairs = load_key(key_path)
    n_ann = df.annotator_id.nunique()
    label = SINGLE_LABEL if n_ann == 1 else f"{n_ann} annotators"
    def w(name, t):
        t = t.copy(); t.insert(0, "study_label", label); t.to_csv(out / name, index=False)
    summary = dict(status="ANALYSED", study_label=label, n_annotators=int(n_ann), annotators=sorted(df.annotator_id.unique()),
                   bootstrap_resamples=B, kendall_resamples=kendall_B, bootstrap="cluster bootstrap by dataset:image_id",
                   p_value_note=f"bootstrap p floor is 1/(B+1) = {1.0 / (B + 1):.2e}; reported as '< floor'",
                   n_task_A_rows=int((df.task == "A").sum()), n_task_B_rows=int((df.task == "B").sum()))
    cons = consensus_items(df); w("consensus_items.csv", cons)
    w("annotator_distributions.csv", annotator_distributions(df))
    rel, _ = reliability_table(df); w("reliability.csv", rel)
    summary["reliability"] = rel.replace({np.nan: None}).to_dict("records")
    fr = build_frame(cons, key_items)
    if len(fr):
        mcols = metric_cols(fr)
        for sub in ("all", "model_output", "hard_case"):
            f = fr if sub == "all" else fr[fr.subset == sub]
            t = assoc_table(f, [(m, "h_overall") for m in mcols], B, seed, kendall_B)
            if len(t): t.insert(0, "subset", sub); w(f"association_overall_{sub}.csv", t)
            if sub == "all": summary["association_overall_all"] = t.replace({np.nan: None}).to_dict("records")
        tgt = [(m, h) for m, h in (("m_ce_p", "h_hallucination"), ("m_ce_p", "h_overall"), ("m_ce_r", "h_completeness"), ("m_ce_f1", "h_overall")) if m in fr]
        w("association_targeted.csv", assoc_table(fr, tgt, B, seed, kendall_B))
        for col, name in (("dataset", "association_by_dataset.csv"), ("system", "association_by_system.csv")):
            parts = []
            for gval, g in fr.groupby(col):
                t = assoc_table(g, [(m, "h_overall") for m in mcols], B, seed, 0, min_n=15)
                if len(t): t.insert(0, col, gval); parts.append(t)
            if parts: w(name, pd.concat(parts))
        reg = pd.concat([regression_table(fr, m[2:]) for m in mcols] or [pd.DataFrame()])
        if len(reg): w("regression.csv", reg)
    pc = pairwise_consensus(df)
    if len(pc):
        w("pairwise_consensus.csv", pc)
        metrics = sorted({m for k in key_pairs.values() for m, v in k["metric_preferences"].items() if v is not None})
        acc, bycat = [], []
        for m in metrics:
            for pol in ("excluded", "half_credit"):
                r = pair_accuracy(pc, key_pairs, m, pol, B, seed); acc.append(dict(metric=m, tie_policy=pol, **r))
                for grp in ("disagree", "agree", "all"):
                    cats = sorted({k["category"] for k in key_pairs.values()})
                    for cat in cats:
                        r = pair_accuracy(pc, key_pairs, m, pol, min(B, 2000), seed,
                                          subset=lambda k, cat=cat, grp=grp: k["category"] == cat and (grp == "all" or k["agreement_group"] == grp))
                        if r["n"]: bycat.append(dict(metric=m, tie_policy=pol, category=cat, agreement_group=grp, **r))
        w("pairwise_accuracy.csv", pd.DataFrame(acc)); w("pairwise_accuracy_by_category.csv", pd.DataFrame(bycat))
        summary["pairwise_human_pref_counts"] = pc.human_pref.value_counts().to_dict()
        summary["pairwise_accuracy"] = pd.DataFrame(acc).replace({np.nan: None}).to_dict("records")
        summary["pairwise_bycat_resamples"] = min(B, 2000)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    summary["out_dir"] = str(out)
    return summary
