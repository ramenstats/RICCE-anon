"""CVPR'27, B1/M1: score every caption of the human study with
  * the evaluator variants of scripts/v3_ablations.py (A0 full ... A11), which remove
    one component each (binding, gate, hierarchy, dual assignment, ...);
  * the soft contradiction gate for a grid of gamma values and the joint-assignment
    objectives (harmonic, mean, min, geometric);
  * the prior hierarchical metrics of ricce/baselines (Kiritchenko hF, LCA-F,
    HCM, TaxAware single-node and axis-flat) on the same parses;
  * the hard gate's activations (ricce.metrics.final.gate_activations).
References and aggregation follow the key (scripts/v3_human_eval_rescore_key.py):
hard cases vs the first reference, real outputs vs every reference, mean over the
scorable references. Parses come from the human-study cache
(results_v3/human_eval/key_rescoring/parse_cache_human_eval.json); nothing is re-parsed.
No variant or gamma is selected here; the analysis reports all of them.

Check: A0_full reproduces the key's CE-F for every caption.
Output: submission_cvpr27_new_version/07_BASELINES/human_captions_variants.csv (+ _checks.json)"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.evaluation.parse_store import ParseStore
from ricce.metrics.final import gate_activations
from ricce.evaluation.human_analysis import in_study

# evaluator variants: the definition part of v3_ablations.py (everything before its data loading)
ABL = ROOT / 'scripts' / 'v3_ablations.py'
MARK = '# ---------------------------------------------------------------- data (loaded before fork)'
src = ABL.read_text(); assert src.count(MARK) == 1
ns = {'__file__': str(ABL), '__name__': 'v3_ablations_defs'}
exec(compile(src.split(MARK)[0], str(ABL), 'exec'), ns)
VARS, Variant, score, T, W = ns['VARS'], ns['Variant'], ns['score'], ns['T'], ns['W']

OUT = ROOT / 'submission_cvpr27_new_version' / '07_BASELINES'
import importlib.util
_spec = importlib.util.spec_from_file_location('cvpr27_03', ROOT / 'scripts' / 'cvpr27_03_extend_human_metrics.py')
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
J = _m.jobs()                                   # same captions and references as the text metrics
store = ParseStore(ROOT / 'results_v3' / 'human_eval' / 'key_rescoring' / 'parse_cache_human_eval.json')
ev = lambda t, mode: store.events(t, mode)

KEEP = ['A0_full', 'A1_no_hierarchy', 'A2_no_decomposition', 'A3_no_binding', 'A4_no_gate', 'A5_unordered_source_target',
        'A6_greedy', 'A7_no_nochange_state', 'A8_single_assignment', 'A9_root_counted', 'A10_root_excluded', 'A11_gate_any_object']
extra = []
for g in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
    extra.append(Variant(f'G_soft_{g:g}', f'soft gate, matched contradictions scaled by {g:g}', cfg=dict(gate_mode='soft', soft_gamma=g)))
for fn in ('harmonic', 'mean', 'min', 'geometric'):
    extra.append(Variant(f'J_{fn}', f'one shared assignment maximising {fn}(S_P,S_R)', cfg=dict(dual_matching=False, joint_fn=fn)))
V = {n: VARS[n] for n in KEEP} | {v.name: v for v in extra}


def mean_over_refs(v, cand, refs):
    mode = 'single' if v.single_parse else 'multi'
    x = [score(v, ev(cand, mode), ev(g, mode)) for g in refs]
    x = [y for y in x if y == y]
    return float(np.mean(x)) if x else np.nan


t0 = time.time()
rows = []
for r in J.itertuples():
    row = {'job': r.job}
    for n, v in V.items():
        row[f'v_{n}'] = mean_over_refs(v, r.candidate, r.references)
    acts = [gate_activations(ev(r.candidate, 'multi'), ev(g, 'multi'), T, W) for g in r.references]
    row['gate_refs_activated'] = sum(bool(a) for a in acts)
    row['gate_kinds'] = '|'.join(sorted({k for a in acts for _, _, k in a}))
    rows.append(row)
print(f'variants: {time.time() - t0:.0f}s', flush=True)
R = pd.DataFrame(rows)

# prior hierarchical metrics on the same parses
from ricce.baselines.registry import BaselineContext, score_pair
ctx = BaselineContext()
for t in {t for c, rs in zip(J.candidate, J.references) for t in [c, *rs]}:
    ctx._cache[t] = ev(t, 'multi')
BASE = ['kiritchenko_macro', 'lca_f', 'hcm', 'taxaware_single', 'taxaware_axisflat']
t0 = time.time()
for b in BASE:
    vals = []
    for r in J.itertuples():
        x = [score_pair(b, r.candidate, g, ctx)['F'] for g in r.references]
        x = [y for y in x if y == y]
        vals.append(float(np.mean(x)) if x else np.nan)
    R[f'b_{b}'] = vals
    print(b, f'{time.time() - t0:.0f}s', flush=True)

R.to_csv(OUT / 'human_captions_variants.csv', index=False)
key = {}
for l in open(ROOT / 'artifacts_cvpr_final/human_eval/image_grounded/hidden/answer_key_final_evaluator.jsonl'):
    k = json.loads(l)
    if not in_study(k): continue
    if k['kind'] == 'item': key[k['item_id']] = k['metrics'].get('ce_f1')
    else: key[k['item_id'] + 'a'], key[k['item_id'] + 'b'] = k['metrics_a'].get('ce_f1'), k['metrics_b'].get('ce_f1')
kv = R.job.map(key).astype(float); a0 = R.v_A0_full
both = kv.notna() & a0.notna()
chk = dict(a0_vs_key_n=int(both.sum()), a0_vs_key_max_abs_diff=float((a0[both] - kv[both]).abs().max()),
           nan_mismatch=int((kv.isna() != a0.isna()).sum()),
           g_soft_0_equals_hard=float((R['v_G_soft_0'] - a0).abs().max()),
           coverage={c: int(R[c].notna().sum()) for c in R.columns if c.startswith(('v_', 'b_'))},
           variants={n: v.desc for n, v in V.items()})
(OUT / 'human_captions_variants_checks.json').write_text(json.dumps(chk, indent=1))
print(json.dumps({k: v for k, v in chk.items() if k not in ('coverage', 'variants')}, indent=1))
