"""CVPR'27, H1/B1/M1: re-analysis of the image-grounded human study
(3 annotators; files collected/annotations_ann*_2026-10-29.csv). The analysis keeps the 898 captions and 380 pairs
whose system outputs come from STUDY_SYSTEMS (ricce/evaluation/human_analysis.py) or are constructed cases.

Changes with respect to the v4 analysis (results_v3/human_eval/analysis_final_evaluator):
  * CIDEr-D is the official formula on every caption (scripts/cvpr27_03_extend_human_metrics.py);
    v4 pooled official CIDEr-D (365 real outputs) with a no-IDF proxy (612 hard cases).
    The v4 numbers are reproduced here from the proxy column (`cider_mixed_v4`).
  * BERTScore, SBERT, SPICE and the LLM text judge cover every caption, so every correlation
    and every paired difference is computed on one matched item set.
  * Every metric on a subset is evaluated on the SAME cluster-bootstrap resamples (image pairs
    as clusters, B = 10,000, seed 20260923), so differences are paired.
  * Kendall tau-b, per-annotator, leave-one-annotator-out and median-consensus sensitivity.
  * Krippendorff alpha with bootstrap CIs (items resampled, B = 1,000).
  * Pairwise preferences on a matched pair set (pairs where the annotators' strict majority
    chose A or B and BOTH compared metrics are untied), separately for the pairs sampled where
    metrics disagreed and the agreement-control pairs.
  * Evaluator variants (binding, gate, assignment, hierarchy ...) and prior hierarchical metrics
    on the human data; the hard gate's activations against the ratings.

Families (Holm within each):
  primary (stated 2026-10-26 in scripts/v3_human_eval_compare.py): CE-F vs CIDEr-D, BERTScore;
  secondary (same statement): CE-F vs BLEU-4, METEOR, ROUGE-L;
  exploratory (added 2026-10-29, after the ratings were analysed): every other comparison.
Outputs -> submission_cvpr27_new_version/03_HUMAN_REANALYSIS/"""
import glob, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.evaluation.human_analysis import (load_annotations, load_key, consensus_items, pairwise_consensus,
                                                 krippendorff_ordinal, matrix, holm, regression_table, STUDY_SYSTEMS)
from ricce.evaluation.human_protocol import AXES

H = ROOT / 'artifacts_cvpr_final' / 'human_eval' / 'image_grounded'
S = ROOT / 'submission_cvpr27_new_version'
OUT = S / '03_HUMAN_REANALYSIS'; OUT.mkdir(parents=True, exist_ok=True)
B, SEED, TIE = 10000, 20260923, 1e-6
ANN = ['ann1', 'ann2', 'ann3']

# ------------------------------------------------------------------ data
df = load_annotations(sorted(glob.glob(str(H / 'collected' / '*.csv'))))
assert sorted(df.annotator_id.unique()) == ANN
key_items, key_pairs = load_key(H / 'hidden' / 'answer_key_final_evaluator.jsonl', systems=STUDY_SYSTEMS)
df = df[df.item_id.isin(set(key_items) | set(key_pairs))]
TM = pd.read_csv(S / '07_BASELINES' / 'human_captions_text_metrics.csv').set_index('job')
VM = pd.read_csv(S / '07_BASELINES' / 'human_captions_variants.csv').set_index('job')
VLM = S / '07_BASELINES' / 'human_captions_vlm_judge.csv'
VJ = pd.read_csv(VLM).set_index('job') if VLM.exists() else None
CS = pd.read_csv(S / '07_BASELINES' / 'human_captions_csm.csv').set_index('job')

NAME = {'bleu4': 'BLEU-4', 'meteor': 'METEOR', 'rougeL': 'ROUGE-L', 'cider_d': 'CIDEr-D', 'cider_mixed_v4': 'CIDEr-D (v4 mix)',
        'bertscore_f': 'BERTScore', 'sbert': 'SBERT', 'spice_f': 'SPICE', 'llm_judge': 'LLM judge (text)', 'vlm_judge': 'VLM judge (image)',
        'llm_judge_ev': 'LLM judge (text, expected)', 'csm4': 'CSM-4', 'csm1': 'CSM-1',
        'b_kiritchenko_macro': 'hF (Kiritchenko)', 'b_lca_f': 'LCA-F', 'b_hcm': 'HCM', 'b_taxaware_single': 'TaxAware single',
        'b_taxaware_axisflat': 'TaxAware axis-flat', 'ce_p': 'CE-P', 'ce_r': 'CE-R', 'ce_f1': 'CE-F'}
HEAD = ['bleu4', 'meteor', 'rougeL', 'cider_d', 'csm4', 'bertscore_f', 'sbert', 'spice_f', 'llm_judge'] + (['vlm_judge'] if VJ is not None else []) + \
       ['b_kiritchenko_macro', 'b_lca_f', 'b_hcm', 'b_taxaware_single', 'b_taxaware_axisflat', 'ce_p', 'ce_r', 'ce_f1']
VAR = [c for c in VM.columns if c.startswith('v_')]
EXTRA = ['cider_mixed_v4', 'csm1'] + (['llm_judge_ev'] if VJ is not None else [])


def metric_row(job, km):
    r = {m: km.get(m, np.nan) for m in ('bleu4', 'meteor', 'rougeL', 'ce_p', 'ce_r', 'ce_f1')}
    t = TM.loc[job]
    r.update(cider_d=t.cider_d, cider_mixed_v4=km.get('cider_d', np.nan), bertscore_f=t.bertscore_f, sbert=t.sbert,
             spice_f=t.get('spice_f', np.nan), llm_judge=t.get('llm_judge', np.nan))
    if VJ is not None: r['vlm_judge'], r['llm_judge_ev'] = VJ.loc[job, 'vlm_judge'], VJ.loc[job, 'llm_judge_ev']
    r['csm4'], r['csm1'] = CS.loc[job, 'csm4'], CS.loc[job, 'csm1']
    v = VM.loc[job]
    r.update({c: v[c] for c in VAR}); r.update({c: v[c] for c in v.index if c.startswith('b_')})
    r['gate_refs_activated'], r['gate_kinds'] = v.gate_refs_activated, v.gate_kinds
    return r


cons = consensus_items(df).set_index('item_id')
A_ = df[df.task == 'A']
own = {a: A_[A_.annotator_id == a].set_index('item_id').overall for a in ANN}
rows = []
for iid, k in key_items.items():
    r = dict(item_id=iid, dataset=k['dataset'], image_id=k['image_id'], cluster=f"{k['dataset']}:{k['image_id']}",
             system=k['system'] or 'perturbed', subset='real' if k['source'] == 'E08_model_output' else 'constructed',
             perturbation_type=k.get('perturbation_type'))
    for a in AXES: r['h_' + a] = cons.loc[iid, a + '_mean']
    r['h_overall_median'] = cons.loc[iid, 'overall_median']
    for a in ANN:
        r[f'h_{a}'] = own[a].get(iid, np.nan)
    for a in ANN:
        r[f'h_loo_{a}'] = np.nanmean([own[b].get(iid, np.nan) for b in ANN if b != a])
    r.update(metric_row(iid, k['metrics']))
    rows.append(r)
F = pd.DataFrame(rows)
F.to_csv(OUT / 'item_frame.csv', index=False)


# ------------------------------------------------------------------ paired cluster bootstrap
def picks(clusters, B, seed):
    cl = pd.factorize(np.asarray(clusters))[0]
    idx = [np.where(cl == c)[0] for c in range(cl.max() + 1)]
    rng = np.random.default_rng(seed)
    return [np.concatenate([idx[j] for j in rng.integers(0, len(idx), len(idx))]) for _ in range(B)]


def spearman_matrix(X, y):
    """Spearman of each column of X with y (no NaN)."""
    rx = stats.rankdata(X, axis=0); ry = stats.rankdata(y)
    rx = rx - rx.mean(0); ry = ry - ry.mean()
    den = np.sqrt((rx ** 2).sum(0) * (ry ** 2).sum())
    with np.errstate(invalid='ignore', divide='ignore'):
        return (rx * ry[:, None]).sum(0) / den


def boot_corr(d, cols, y, B=B, seed=SEED):
    """Point Spearman, and B x k bootstrap matrix on shared resamples. d has no NaN in cols+y."""
    X, Y = d[cols].to_numpy(float), d[y].to_numpy(float)
    est = spearman_matrix(X, Y)
    bm = np.vstack([spearman_matrix(X[ii], Y[ii]) for ii in picks(d.cluster, B, seed)])
    return est, bm


def p_boot(v):
    v = v[~np.isnan(v)]
    tail = min((v <= 0).sum(), (v >= 0).sum())
    return max(min(1.0, 2 * (tail + 1) / (len(v) + 1)), 1.0 / (len(v) + 1))


def ci(v):
    v = v[~np.isnan(v)]
    return np.percentile(v, [2.5, 97.5]) if len(v) else (np.nan, np.nan)


SUBSETS = {'all': F, 'real': F[F.subset == 'real'], 'constructed': F[F.subset == 'constructed']}
for ds in ('LEVIR_CC', 'SECOND_CC', 'DUBAI_CC'):
    SUBSETS[f'{ds}:all'] = F[F.dataset == ds]; SUBSETS[f'{ds}:real'] = F[(F.dataset == ds) & (F.subset == 'real')]

t0 = time.time()
main_rows, diff_rows, matched_n = [], [], []
PRIMARY, SECONDARY = ['cider_d', 'bertscore_f'], ['bleu4', 'meteor', 'rougeL']
for sname, d0 in SUBSETS.items():
    d = d0.dropna(subset=HEAD + ['h_overall'])
    matched_n.append(dict(subset=sname, n_items_in_subset=len(d0), n_matched=len(d), n_clusters=d.cluster.nunique(),
                          dropped=len(d0) - len(d), dropped_by={m: int(d0[m].isna().sum()) for m in HEAD if d0[m].isna().any()}))
    B_ = B if sname in ('all', 'real', 'constructed') else 2000
    est, bm = boot_corr(d, HEAD + EXTRA, 'h_overall', B_)
    for j, m in enumerate(HEAD + EXTRA):
        lo, hi = ci(bm[:, j])
        kt = stats.kendalltau(d[m], d.h_overall).statistic
        main_rows.append(dict(subset=sname, metric=m, name=NAME[m], n=len(d), n_clusters=d.cluster.nunique(), spearman=est[j],
                              ci_lo=lo, ci_hi=hi, kendall_tau_b=kt, resamples=B_))
    i_ce = HEAD.index('ce_f1')
    for j, m in enumerate(HEAD + EXTRA):
        if m == 'ce_f1': continue
        dv = bm[:, i_ce] - bm[:, j]; lo, hi = ci(dv)
        fam = 'primary' if m in PRIMARY else 'secondary' if m in SECONDARY else 'exploratory'
        diff_rows.append(dict(subset=sname, comparison=f'ce_f1 - {m}', other=m, family=fam, n=len(d), diff=est[i_ce] - est[j],
                              ci_lo=lo, ci_hi=hi, p_boot=p_boot(dv), resamples=B_))
print(f'main correlations: {time.time() - t0:.0f}s', flush=True)
M = pd.DataFrame(main_rows); D = pd.DataFrame(diff_rows)
for (s, fam), g in D.groupby(['subset', 'family']):
    D.loc[g.index, 'p_holm'] = holm(g.p_boot.to_numpy())
M.to_csv(OUT / 'human_main_metrics.csv', index=False); D.to_csv(OUT / 'human_metric_differences.csv', index=False)
pd.DataFrame(matched_n).to_csv(OUT / 'human_matched_n.csv', index=False)

# Kendall tau-b CIs for the headline metrics (all / real), B = 2000, same cluster scheme
krows = []
for sname in ('all', 'real'):
    d = SUBSETS[sname].dropna(subset=HEAD + ['h_overall'])
    ps = picks(d.cluster, 2000, SEED)
    for m in ('ce_f1', 'cider_d', 'bertscore_f', 'llm_judge', 'spice_f', 'bleu4'):
        x, y = d[m].to_numpy(float), d.h_overall.to_numpy(float)
        v = np.array([stats.kendalltau(x[ii], y[ii]).statistic for ii in ps]); lo, hi = ci(v)
        krows.append(dict(subset=sname, metric=m, n=len(d), kendall_tau_b=stats.kendalltau(x, y).statistic, ci_lo=lo, ci_hi=hi, resamples=2000))
pd.DataFrame(krows).to_csv(OUT / 'human_kendall.csv', index=False)
print(f'kendall: {time.time() - t0:.0f}s', flush=True)

# ------------------------------------------------------------------ per-annotator, leave-one-out, median
rob = []
for sname in ('all', 'real'):
    d = SUBSETS[sname].dropna(subset=HEAD)
    for target in ['h_overall', 'h_overall_median'] + [f'h_{a}' for a in ANN] + [f'h_loo_{a}' for a in ANN]:
        dd = d.dropna(subset=[target])
        est, bm = boot_corr(dd, ['ce_f1', 'cider_d', 'bertscore_f', 'llm_judge'], target, 2000)
        for j, m in enumerate(['ce_f1', 'cider_d', 'bertscore_f', 'llm_judge']):
            lo, hi = ci(bm[:, j])
            row = dict(subset=sname, target=target, metric=m, n=len(dd), spearman=est[j], ci_lo=lo, ci_hi=hi)
            if m != 'ce_f1':
                dv = bm[:, 0] - bm[:, j]; row.update(diff_vs_cef=est[0] - est[j], diff_ci_lo=ci(dv)[0], diff_ci_hi=ci(dv)[1])
            rob.append(row)
pd.DataFrame(rob).to_csv(OUT / 'per_annotator_robustness.csv', index=False)
print(f'robustness: {time.time() - t0:.0f}s', flush=True)

# ------------------------------------------------------------------ within-system and regression
ws = []
for sy, g in F.groupby('system'):
    d = g.dropna(subset=HEAD + ['h_overall'])
    if len(d) < 15: continue
    est = spearman_matrix(d[HEAD].to_numpy(float), d.h_overall.to_numpy(float))
    ws += [dict(system=sy, metric=m, n=len(d), spearman=e) for m, e in zip(HEAD, est)]
pd.DataFrame(ws).to_csv(OUT / 'within_system.csv', index=False)
fr = F.rename(columns={m: 'm_' + m for m in HEAD})
reg = pd.concat([regression_table(fr.dropna(subset=['m_' + m for m in HEAD]), m) for m in HEAD])
reg.to_csv(OUT / 'regression.csv', index=False)

# ------------------------------------------------------------------ reliability with CIs
rel = []
rng = np.random.default_rng(SEED)
for ax in AXES:
    m = matrix(df, ax, 'A')
    a_all = krippendorff_ordinal(m)
    cols = m.columns.to_numpy()
    bs = [krippendorff_ordinal(m[cols[rng.integers(0, len(cols), len(cols))]]) for _ in range(1000)]
    lo, hi = ci(np.array(bs, float))
    row = dict(scale=ax, alpha_all=a_all, ci_lo=lo, ci_hi=hi, n_items=int((m.notna().sum() >= 2).sum()))
    for x, y in (('ann1', 'ann2'), ('ann1', 'ann3'), ('ann2', 'ann3')):
        row[f'alpha_{x}_{y}'] = krippendorff_ordinal(m.loc[[x, y]])
    rel.append(row)
pd.DataFrame(rel).to_csv(OUT / 'rating_agreement.csv', index=False)
print(f'reliability: {time.time() - t0:.0f}s', flush=True)

# ------------------------------------------------------------------ pairwise preferences (matched pairs)
pc = pairwise_consensus(df).set_index('item_id')
PM = ['ce_f1', 'cider_d', 'csm4', 'bertscore_f', 'sbert', 'spice_f', 'llm_judge', 'bleu4', 'meteor', 'rougeL'] + (['vlm_judge'] if VJ is not None else [])


def side(job, m):
    if m in ('bleu4', 'meteor', 'rougeL', 'ce_f1'):
        k = key_pairs[job[:-1]]; return k[f'metrics_{job[-1]}'].get(m, np.nan)
    if m == 'vlm_judge': return VJ.loc[job, 'vlm_judge']
    if m == 'csm4': return CS.loc[job, 'csm4']
    return TM.loc[job, m]


prow = []
for pid, k in key_pairs.items():
    r = dict(pair_id=pid, dataset=k['dataset'], image_id=k['image_id'], cluster=f"{k['dataset']}:{k['image_id']}", source=k['source'],
             category=k['category'], group=k['agreement_group'], human=pc.human_pref.get(pid, 'missing'))
    for m in PM:
        a, b = side(pid + 'a', m), side(pid + 'b', m)
        r[m] = np.nan if (a != a or b != b) else ('A' if a - b > TIE else 'B' if b - a > TIE else 'tie')
    prow.append(r)
P = pd.DataFrame(prow); P.to_csv(OUT / 'pairwise_frame.csv', index=False)


def acc_boot(s, cl, Bp=2000):
    ps = picks(cl, Bp, SEED); v = np.array([s[ii].mean() for ii in ps]); return s.mean(), *ci(v)


pw = []
strict = P[P.human.isin(['A', 'B'])]
for grp in ('all', 'disagree', 'agree'):
    for src in ('all', 'real', 'pert'):
        g = strict if grp == 'all' else strict[strict.group == grp]
        g = g if src == 'all' else g[g.source == src]
        for m in PM:
            if m == 'ce_f1': continue
            mm = g[(g.ce_f1.isin(['A', 'B'])) & (g[m].isin(['A', 'B']))]          # matched: both metrics untied
            if len(mm) < 10: continue
            sc = (mm.ce_f1 == mm.human).to_numpy(float); so = (mm[m] == mm.human).to_numpy(float)
            a1, l1, h1 = acc_boot(sc, mm.cluster); a2, l2, h2 = acc_boot(so, mm.cluster)
            dlt = sc - so; dd, dl, dh = acc_boot(dlt, mm.cluster)
            ta = g[g[m].notna() & g.ce_f1.notna()]                                 # tie-aware: metric tie = 0.5
            tc = np.where(ta.ce_f1 == 'tie', 0.5, (ta.ce_f1 == ta.human).astype(float)).mean()
            to = np.where(ta[m] == 'tie', 0.5, (ta[m] == ta.human).astype(float)).mean()
            pw.append(dict(group=grp, source=src, other=m, n_strict_pairs=len(g), n_matched=len(mm), cef_acc=a1, cef_lo=l1, cef_hi=h1,
                           other_acc=a2, other_lo=l2, other_hi=h2, diff=dd, diff_lo=dl, diff_hi=dh,
                           n_tie_aware=len(ta), cef_tie_aware=tc, other_tie_aware=to,
                           cef_ties=int((g.ce_f1 == 'tie').sum()), other_ties=int((g[m] == 'tie').sum())))
pd.DataFrame(pw).to_csv(OUT / 'pairwise_preferences.csv', index=False)
cnt = P.human.value_counts().to_dict()
print(f'pairwise: {time.time() - t0:.0f}s', flush=True)

# ------------------------------------------------------------------ evaluator variants on human data (exploratory)
vr = []
for sname in ('all', 'real'):
    d = SUBSETS[sname].dropna(subset=VAR + ['h_overall'])
    est, bm = boot_corr(d, VAR, 'h_overall', 2000)
    i0 = VAR.index('v_A0_full')
    for j, v in enumerate(VAR):
        lo, hi = ci(bm[:, j]); dv = bm[:, i0] - bm[:, j]
        vr.append(dict(subset=sname, variant=v[2:], n=len(d), spearman=est[j], ci_lo=lo, ci_hi=hi,
                       full_minus_variant=est[i0] - est[j], diff_ci_lo=ci(dv)[0], diff_ci_hi=ci(dv)[1]))
pd.DataFrame(vr).to_csv(S / '08_METHOD_ABLATIONS' / 'human_variant_agreement.csv', index=False)

# gate activations against the ratings
gate = []
for sname in ('real', 'constructed'):
    d = SUBSETS[sname]
    for act, g in d.groupby(d.gate_refs_activated > 0):
        gate.append(dict(subset=sname, gate_active=bool(act), n=len(g), mean_overall=g.h_overall.mean(), mean_action=g.h_action.mean(),
                         share_overall_le_2=(g.h_overall <= 2).mean(), mean_cef=g.ce_f1.mean(), mean_cef_no_gate=g.v_A4_no_gate.mean()))
pd.DataFrame(gate).to_csv(S / '08_METHOD_ABLATIONS' / 'gate_vs_human.csv', index=False)
F[(F.gate_refs_activated > 0) & (F.subset == 'real')][['item_id', 'dataset', 'image_id', 'system', 'gate_kinds', 'ce_f1', 'v_A4_no_gate',
                                                       'h_overall', 'h_action', 'h_ann1', 'h_ann2', 'h_ann3']].to_csv(
    S / '08_METHOD_ABLATIONS' / 'gate_activations_real_outputs.csv', index=False)
# dual vs joint assignment: how often they differ on the rated captions
dj = F[['item_id', 'subset', 'v_A0_full', 'v_A8_single_assignment']].dropna()
pd.DataFrame([dict(subset=s, n=len(g), n_differ=int(((g.v_A0_full - g.v_A8_single_assignment).abs() > 1e-9).sum()),
                   max_abs_diff=float((g.v_A0_full - g.v_A8_single_assignment).abs().max()))
              for s, g in [('all', dj)] + list(dj.groupby('subset'))]).to_csv(S / '08_METHOD_ABLATIONS' / 'dual_vs_joint_on_rated_captions.csv', index=False)

# blind error-audit sample: 50 items with the largest CE-F vs human rank disagreement among real outputs,
# chosen on CE-F's error only (not on whether another metric does better); for manual labelling
r = SUBSETS['real'].dropna(subset=['ce_f1']).copy()
r['rank_gap'] = (r.ce_f1.rank(pct=True) - r.h_overall.rank(pct=True)).abs()
aud = r.sort_values('rank_gap', ascending=False).head(50)
items = {json.loads(l)['item_id']: json.loads(l) for l in open(H / 'items.jsonl')}
aud = aud.assign(candidate=aud.item_id.map(lambda i: items[i]['candidate']), references=aud.item_id.map(lambda i: ' | '.join(items[i]['references'])))
aud[['item_id', 'dataset', 'image_id', 'system', 'candidate', 'references', 'ce_f1', 'h_overall', 'rank_gap']].assign(
    error_source_LABEL_BY_HUMAN='', note='').to_csv(OUT / 'error_audit_sample_FOR_HUMAN_LABELLING.csv', index=False)

summary = dict(n_items=len(F), n_real=int((F.subset == 'real').sum()), n_constructed=int((F.subset == 'constructed').sum()),
               n_pairs=len(P), pair_consensus=cnt, bootstrap=f'cluster (image pair) bootstrap, B={B}, seed={SEED}, shared resamples per subset',
               headline_metrics=HEAD, vlm_judge_included=VJ is not None)
(OUT / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
print(M[M.subset.isin(['all', 'real'])].pivot(index='name', columns='subset', values='spearman').round(3).to_string())
print(D[D.subset.isin(['all', 'real']) & D.family.isin(['primary', 'secondary'])][['subset', 'other', 'diff', 'ci_lo', 'ci_hi', 'p_holm']].round(4).to_string())
print(pd.DataFrame(matched_n)[['subset', 'n_items_in_subset', 'n_matched', 'dropped_by']].to_string())
