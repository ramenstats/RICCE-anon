"""Re-score the human study's analyst-only answer key with the FINAL evaluator.

The key (hidden/answer_key.jsonl, built 2026-09-23 by scripts/build_human_eval_items.py)
stores CE-P/CE-R/CE-F from the historical evaluator:
  * hard-case items and perturbation pairs: E04 protocol, candidate vs the image's FIRST
    reference (manifest references[0]), single-event parse, metrics.event.ricce_scores;
  * real model outputs and real pairs: E08 protocol, candidate vs every reference of the image,
    mean over references, single-event parse, same historical evaluator.

This script keeps exactly those references and replaces only the evaluator, using the
paper's final protocol: multi-event parse (as in results_v3/parse_cache.json) and
ricce.metrics.final.final_scores (FINAL_CONFIG); per-reference not-scorable pairs are
skipped (mean over the scorable references, as in v3_recompute_systems.py); an item with no
scorable reference gets no CE value. Every other metric in the key is copied unchanged.

Checks (written to summary.json):
  * source identification: every stored value equals its E04 / E08 per-sample row, and the E04 reference
    equals the manifest's references[0] (so the references used here are the ones the key used);
  * consistency with the paper: for real model outputs, the final-evaluator CE-F equals
    results_v3/systems/per_image/<DS>__<system>.csv exactly;
  * historical reproduction (informative only): the stored CE values came from the E04/E08 runs of
    2026-08-24/25, made with a code state that was never committed. Re-running the historical evaluator
    with the current code does not reproduce them; attempts with the 2026-08-21 commit are documented in
    results_v3/human_eval/key_rescoring/historical_reproduction/.

Outputs:
  artifacts_cvpr_final/human_eval/image_grounded/hidden/answer_key_final_evaluator.jsonl  (analyst-only)
  results_v3/human_eval/key_rescoring/{item_ce.csv, pair_ce.csv, summary.json}"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.utils import read_jsonl
from ricce.benchmarks.env import load_taxes, load_weights
from ricce.evaluation.parse_store import ParseStore
from ricce.metrics.final import FINAL_CONFIG, final_scores
from ricce.metrics.event import ricce_scores as historical_scores

H = ROOT / 'artifacts_cvpr_final' / 'human_eval' / 'image_grounded'
OUT = ROOT / 'results_v3' / 'human_eval' / 'key_rescoring'; OUT.mkdir(parents=True, exist_ok=True)
T, W = load_taxes(), load_weights()
TIE_EPS = 1e-6
key_rows = [json.loads(l) for l in open(H / 'hidden' / 'answer_key.jsonl')]
items = {r['item_id']: r for r in read_jsonl(H / 'items.jsonl')}
pairs = {r['pair_id']: r for r in read_jsonl(H / 'pairwise_items.jsonl')}
man = {ds: {r['image_id']: r for r in read_jsonl(ROOT / 'artifacts' / 'manifests' / f'{ds}.jsonl')} for ds in ('LEVIR_CC', 'SECOND_CC', 'DUBAI_CC')}


def refs_for(k):
    m = man[k['dataset']][k['image_id']]
    hard = k['source'] in ('E04_perturbation', 'pert')
    return m['references'][:1] if hard else list(m['references'])


# ------------------------------------------------------------------ parses (separate cache; the paper's cache is not modified)
jobs = []   # (row, side, candidate text, refs)
for k in key_rows:
    if k['kind'] == 'item':
        jobs.append((k, None, items[k['item_id']]['candidate'], refs_for(k)))
    else:
        p = pairs[k['item_id']]
        jobs.append((k, 'a', p['candidate_a'], refs_for(k)))
        jobs.append((k, 'b', p['candidate_b'], refs_for(k)))
texts = sorted({t for _, _, c, rs in jobs for t in [c, *rs]})
store = ParseStore(OUT / 'parse_cache_human_eval.json')
main = ParseStore(ROOT / 'results_v3' / 'parse_cache.json')
for mode in ('multi', 'single'):
    for t in texts:
        if t in main.d[mode]: store.seed(mode, t, main.d[mode][t])
    print(f'{mode}: parsed {store.ensure(texts, mode)} new texts', flush=True)
store.save()
ev = lambda t, mode: store.events(t, mode)


def ce_hist(cand, refs):
    s = [historical_scores(ev(cand, 'single'), ev(g, 'single'), T, W) for g in refs]
    return {k: float(np.mean([x[k] for x in s])) for k in ('ce_precision', 'ce_recall', 'ce_f1')}


def ce_final(cand, refs):
    vals = []
    for g in refs:
        s = final_scores(ev(cand, 'multi'), ev(g, 'multi'), T, W)
        if not s['not_scorable']:
            vals.append((s['ce_precision'], s['ce_recall'], s['ce_f1']))
    if not vals:
        return None, 0
    v = np.array(vals)
    return dict(ce_p=float(v[:, 0].mean()), ce_r=float(v[:, 1].mean()), ce_f1=float(v[:, 2].mean())), len(vals)


# ------------------------------------------------------------------ score
irows, prows, basis = [], [], []
new_rows = []
res = {}
for k, side, cand, refs in jobs:
    h = ce_hist(cand, refs); f, n_ok = ce_final(cand, refs)
    stored = k['metrics'] if side is None else k[f'metrics_{side}']
    basis.append(max(abs(h['ce_f1'] - stored['ce_f1']), abs(h['ce_precision'] - stored['ce_p']), abs(h['ce_recall'] - stored['ce_r'])))
    res[(k['item_id'], side)] = (f, n_ok, h, stored)
basis = np.array(basis)
print(f'historical evaluator, CURRENT code, vs stored key: max |diff| = {basis.max():.2e}; {int((basis > 1e-9).sum())} of {len(basis)} above 1e-9 '
      f'(expected: the stored values come from an uncommitted 2026-08-24/25 code state)', flush=True)

# source identification
clean = lambda x: ' '.join(str(x).split())
e04 = {ds: pd.read_csv(ROOT / 'results' / 'E04' / ds / 'per_sample_metrics.csv') for ds in man}
src_ok = gt_ok = n_hard = real_ok = n_real = 0
for k in key_rows:
    if k['kind'] != 'item': continue
    if k['source'] == 'E04_perturbation':
        df = e04[k['dataset']]
        r = df[(df.image_id == k['image_id']) & (df['type'] == k['perturbation_type']) & (df.prediction.map(clean) == items[k['item_id']]['candidate'])]
        n_hard += 1
        if len(r):
            r = r.iloc[0]
            src_ok += abs(r['ce_f1'] - k['metrics']['ce_f1']) < 1e-12
            gt_ok += clean(r['gt']) == clean(man[k['dataset']][k['image_id']]['references'][0]) == k['source_reference']
    else:
        df = pd.read_csv(ROOT / 'results' / 'E08' / k['dataset'] / k['system'] / 'per_sample_metrics.csv'); r = df[df.image_id == k['image_id']]
        n_real += 1; real_ok += bool(len(r)) and abs(r.iloc[0]['ce_f1'] - k['metrics']['ce_f1']) < 1e-12
print(f'source identification: hard items {src_ok}/{n_hard} equal their E04 row, reference = references[0] for {gt_ok}/{n_hard}; '
      f'real items {real_ok}/{n_real} equal their E08 row', flush=True)

for k in key_rows:
    k2 = json.loads(json.dumps(k))
    if k['kind'] == 'item':
        f, n_ok, h, stored = res[(k['item_id'], None)]
        for m in ('ce_f1', 'ce_p', 'ce_r'):
            k2['metrics'].pop(m, None)
            if f is not None: k2['metrics'][m] = f[m]
        irows.append(dict(item_id=k['item_id'], dataset=k['dataset'], source=k['source'], system=k['system'], perturbation_type=k.get('perturbation_type'),
                          n_refs=len(refs_for(k)), n_scorable_refs=n_ok, ce_f1_historical=stored['ce_f1'],
                          ce_f1_final=None if f is None else f['ce_f1'], ce_p_final=None if f is None else f['ce_p'], ce_r_final=None if f is None else f['ce_r']))
    else:
        fa, na, _, sa = res[(k['item_id'], 'a')]; fb, nb, _, sb = res[(k['item_id'], 'b')]
        for side, f in (('a', fa), ('b', fb)):
            for m in ('ce_f1', 'ce_p', 'ce_r'):
                k2[f'metrics_{side}'].pop(m, None)
                if f is not None: k2[f'metrics_{side}'][m] = f[m]
        if fa is None or fb is None:
            pref = None
        else:
            d = fa['ce_f1'] - fb['ce_f1']; pref = 'tie' if abs(d) <= TIE_EPS else ('A' if d > 0 else 'B')
        k2['metric_preferences']['ce_f1'] = pref
        prows.append(dict(pair_id=k['item_id'], dataset=k['dataset'], source=k['source'], category=k['category'], agreement_group_at_build=k['agreement_group'],
                          ce_f1_a_final=None if fa is None else fa['ce_f1'], ce_f1_b_final=None if fb is None else fb['ce_f1'],
                          ce_pref_historical=k['metric_preferences']['ce_f1'], ce_pref_final=pref))
    k2['ce_evaluator'] = 'final (ricce/metrics/final.py FINAL_CONFIG, multi-event parse); rescored by scripts/v3_human_eval_rescore_key.py'
    new_rows.append(k2)

with open(H / 'hidden' / 'answer_key_final_evaluator.jsonl', 'w') as fh:
    for r in new_rows: fh.write(json.dumps(r) + '\n')
I = pd.DataFrame(irows); P = pd.DataFrame(prows)
I.to_csv(OUT / 'item_ce.csv', index=False); P.to_csv(OUT / 'pair_ce.csv', index=False)
cons = []
for _, r in I[I.source == 'E08_model_output'].iterrows():
    p = pd.read_csv(ROOT / 'results_v3' / 'systems' / 'per_image' / f'{r.dataset}__{r.system}.csv')
    q = p[p.image_id == next(k['image_id'] for k in key_rows if k['item_id'] == r.item_id)]
    cons.append(abs(q.iloc[0]['ce_f'] - r.ce_f1_final) if len(q) and pd.notna(r.ce_f1_final) else np.nan)
cons = np.array(cons)
print(f'consistency with the paper: real items final CE-F vs systems/per_image: max |diff| {np.nanmax(cons):.2e} over {int(np.isfinite(cons).sum())}', flush=True)
summary = dict(final_config=FINAL_CONFIG, n_items=len(I), n_pairs=len(P),
               source_identification=dict(hard_items=n_hard, hard_items_equal_E04_row=int(src_ok), hard_items_reference_is_references0=int(gt_ok),
                                          real_items=n_real, real_items_equal_E08_row=int(real_ok)),
               consistency_with_paper_real_items=dict(n=int(np.isfinite(cons).sum()), max_abs_diff=float(np.nanmax(cons))),
               historical_evaluator_current_code_vs_stored=dict(n_captions=int(len(basis)), n_above_1e9=int((basis > 1e-9).sum()), max_abs_diff=float(basis.max()),
                                                                note='stored values come from an uncommitted 2026-08-24/25 code state; see historical_reproduction/'),
               items_without_scorable_reference=int(I.ce_f1_final.isna().sum()),
               pairs_without_ce_preference=int(P.ce_pref_final.isna().sum()),
               mean_ce_f1_items_historical=float(I.ce_f1_historical.mean()), mean_ce_f1_items_final=float(I.ce_f1_final.mean()),
               pair_ce_preference_changed=int((P.ce_pref_historical != P.ce_pref_final).sum()),
               pair_ce_pref_counts_final=P.ce_pref_final.value_counts(dropna=False).to_dict())
(OUT / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
print(json.dumps(summary, indent=1, default=str))
