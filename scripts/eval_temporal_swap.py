#!/usr/bin/env python3
"""Evaluate one checkpoint (or aggregate all) for the TemporalSwap-SFT study.

Per checkpoint  (--name NAME --adapter DIR):
  (i)   lora_infer on LEVIR-CC test + E08 (CE-F1 + conventional metrics); verifies ce_f1 present
        (existing predictions/E08 for that name are reused if valid)
  (ii)  E10 swap consistency on --e10_limit (default 300) test pairs with the adapter
  (iii) action-reversal detection on the swapped pairs (self-consistency via is_action_reversal, and
        swapped prediction vs. ORIGINAL-order ground-truth action)
  -> results_cvpr_final/temporal_equivariance/runs/NAME/eval.json
Aggregate (--aggregate): per_seed.csv, summary.json, README.md for sft_seed{1,42,123} vs swapsft_seed{1,42,123}
  (mean+-std over seeds, paired bootstrap CI, 10,000 resamples over test items, for CE-F1 and inversion accuracy).
"""
import argparse, json, subprocess, sys, time, math
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'results_cvpr_final' / 'temporal_equivariance'
SEEDS = [1, 42, 123]
MANIFEST = 'artifacts/manifests/LEVIR_CC.jsonl'


def sh(cmd, log):
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with open(log, 'w') as f:
        return subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode


def has_ce_f1(p):
    try:
        return 'ce_f1' in json.loads(Path(p).read_text())
    except Exception:
        return False


def parser_stack(cfg_path='configs/project.yaml'):
    from ricce.utils import load_yaml
    from ricce.ontology.tree import Taxonomy
    from ricce.parsing.hybrid import HybridParser
    cfg = load_yaml(cfg_path)
    t = {k: Taxonomy.from_yaml(v) for k, v in cfg['ontology'].items()}
    return t, HybridParser(t['object'], t['action'], t['location'], t['quantity'], use_embeddings=False)


def reversal_detection(raw_pred_path, manifest, taxes, parser):
    from ricce.utils import read_jsonl
    from ricce.metrics.event import is_action_reversal
    from ricce.training.temporal_swap import RHO_ACTION
    gt = {r['image_id']: r['references'] for r in read_jsonl(manifest) if r['split'] == 'test'}
    n_self = n_self_rev = n_gt = n_gt_rev = n_gt_copy = n_gt_rho = 0
    for l in open(raw_pred_path):
        p = json.loads(l)
        en = parser.parse(p['normal_order'])[0]; es = parser.parse(p['swapped_order'])[0]
        if en.action in RHO_ACTION and en.action != 'no_change':
            n_self += 1; n_self_rev += int(bool(is_action_reversal(en, es, taxes)))
        votes = Counter()
        evs = {}
        for ref in gt.get(p['image_id'], []):
            e = parser.parse(ref)[0]; votes[e.action] += 1; evs.setdefault(e.action, e)
        if not votes:
            continue
        ga = votes.most_common(1)[0][0]
        if ga in RHO_ACTION and ga != 'no_change':
            n_gt += 1
            n_gt_rev += int(bool(is_action_reversal(es, evs[ga], taxes)))
            n_gt_copy += int(es.action == ga)
            n_gt_rho += int(es.action == RHO_ACTION[ga])
    r = lambda a, b: (a / b) if b else float('nan')
    return {'n_self_directional': n_self, 'self_reversal_detection_rate': r(n_self_rev, n_self),
            'n_gt_directional_pairs': n_gt, 'swapped_pred_reverses_gt_rate': r(n_gt_rev, n_gt),
            'swapped_pred_copies_gt_action_rate': r(n_gt_copy, n_gt),
            'swapped_pred_equals_rho_gt_action_rate': r(n_gt_rho, n_gt)}


def eval_one(a):
    name, adapter = a.name, Path(a.adapter)
    if not adapter.is_dir() or not (adapter / 'adapter_model.safetensors').exists():
        sys.exit(f'no adapter at {adapter}')
    run = OUT / 'runs' / name; run.mkdir(parents=True, exist_ok=True)
    logs = OUT / 'logs'; logs.mkdir(parents=True, exist_ok=True)
    res = {'name': name, 'adapter': str(adapter)}; t0 = time.time()
    pred = ROOT / 'results' / 'baselines' / name / 'LEVIR_CC_test.jsonl'
    e08 = ROOT / 'results' / 'E08' / 'LEVIR_CC' / name
    if not (e08 / 'metrics.json').exists() or not has_ce_f1(e08 / 'metrics.json') or not (e08 / 'per_sample_metrics.csv').exists():
        if not pred.exists() or sum(1 for _ in open(pred)) < 1000:
            pred.parent.mkdir(parents=True, exist_ok=True)
            rc = sh([sys.executable, 'scripts/lora_infer.py', '--adapter', str(adapter), '--manifest', MANIFEST,
                     '--output', str(pred), '--split', 'test'], logs / f'infer_{name}.log')
            if rc: sys.exit(f'lora_infer failed rc={rc}')
        e08.mkdir(parents=True, exist_ok=True)
        rc = sh([sys.executable, '-m', 'ricce.cli', 'experiment', 'E08', MANIFEST, '--predictions', str(pred),
                 '--outdir', str(e08), '--expensive-limit', '200'], logs / f'e08_{name}.log')
        if rc or not has_ce_f1(e08 / 'metrics.json'):
            sys.exit(f'E08 scoring FAILED or ce_f1 missing for {name} (rc={rc}); see {logs}/e08_{name}.log')
    res['e08'] = json.loads((e08 / 'metrics.json').read_text())
    assert 'ce_f1' in res['e08']
    e10 = run / 'e10'
    if not has_metric(e10 / 'metrics.json', 'directional_inversion_accuracy') or not (e10 / 'per_sample_swap.jsonl').exists():
        rc = sh([sys.executable, 'scripts/run_e10_temporal_swap.py', '--manifest', MANIFEST, '--outdir', str(e10),
                 '--limit', str(a.e10_limit), '--adapter', str(adapter)], logs / f'e10_{name}.log')
        if rc or not has_metric(e10 / 'metrics.json', 'directional_inversion_accuracy'):
            sys.exit(f'E10 FAILED for {name} rc={rc}; see {logs}/e10_{name}.log')
    res['e10'] = json.loads((e10 / 'metrics.json').read_text())
    if res['e10']['n_samples'] < 300:
        print('WARNING: fewer than 300 E10 pairs')
    taxes, parser = parser_stack()
    res['reversal_detection'] = reversal_detection(e10 / 'raw_predictions.jsonl', MANIFEST, taxes, parser)
    res['elapsed_s'] = round(time.time() - t0, 1)
    (run / 'eval.json').write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ('e10', 'reversal_detection')}, indent=1))
    print('CE-F1', res['e08']['ce_f1'])


def has_metric(p, k):
    try:
        return k in json.loads(Path(p).read_text())
    except Exception:
        return False


# ---------------------------------------------------------------- aggregation
def boot_paired(fa, fb, n_items, B=10000, seed=0):
    """fa/fb: functions(idx array [B,n]) -> array [B] of the statistic for model group A / B.
    Returns point diff, 95% CI, P(diff<=0) for stat_A - stat_B resampling test items."""
    import numpy as np
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_items, size=(B, n_items))
    d = fa(idx) - fb(idx)
    full = np.arange(n_items)[None, :]
    pt = (fa(full) - fb(full))[0]
    return {'diff': float(pt), 'ci95': [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))],
            'boot_mean': float(np.nanmean(d)), 'frac_resamples_diff_le_0': float(np.nanmean(d <= 0))}


def aggregate():
    import numpy as np, pandas as pd
    groups = {'sft': [], 'sft2x': [], 'swapsft': []}
    rows = []
    for g in groups:
        for s in SEEDS:
            f = OUT / 'runs' / f'{g}_seed{s}' / 'eval.json'
            if not f.exists():
                continue
            d = json.loads(f.read_text()); e = d['e10']; rd = d['reversal_detection']; m = d['e08']
            rows.append({'group': g, 'seed': s, 'ce_f1': m['ce_f1'], 'ce_precision': m['ce_precision'], 'ce_recall': m['ce_recall'],
                         'bleu4': m.get('bleu4'), 'meteor': m.get('meteor'), 'rougeL': m.get('rougeL'), 'cider_d': m.get('cider_d'),
                         'e10_n': e['n_samples'], 'n_directional': e['n_directional_action'],
                         'inversion_acc': e['directional_inversion_accuracy'],
                         'n_no_change': e['n_no_change'], 'nochange_consistency': e['no_change_consistency_under_swap'],
                         'n_other_or_unparsed': e['n_other_or_unparsed'], **rd})
            groups[g].append(s)
    df = pd.DataFrame(rows); OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / 'per_seed.csv', index=False)
    summ = {'seeds_available': groups, 'per_metric': {}, 'paired_bootstrap': {}}
    metrics = ['ce_f1', 'bleu4', 'meteor', 'rougeL', 'cider_d', 'inversion_acc', 'nochange_consistency',
               'self_reversal_detection_rate', 'swapped_pred_reverses_gt_rate', 'swapped_pred_copies_gt_action_rate']
    for g in groups:
        sub = df[df.group == g] if len(df) else df
        summ['per_metric'][g] = {}
        for k in metrics:
            if len(sub) and k in sub:
                v = sub[k].astype(float).values
                summ['per_metric'][g][k] = {'mean': float(np.nanmean(v)), 'std': float(np.nanstd(v, ddof=1)) if len(v) > 1 else None, 'n_seeds': int(len(v))}
    verdict = []
    for ctl in ('sft', 'sft2x'):
     common = sorted(set(groups[ctl]) & set(groups['swapsft']))
     if common:
         # CE-F1: paired over test items, mean over matched seeds
         def load_ce(g, s):
             d = pd.read_csv(ROOT / 'results' / 'E08' / 'LEVIR_CC' / f'{g}_seed{s}' / 'per_sample_metrics.csv').set_index('image_id')['ce_f1']
             return d
         A = pd.concat([load_ce('swapsft', s) for s in common], axis=1).dropna()
         Bm = pd.concat([load_ce(ctl, s) for s in common], axis=1).reindex(A.index)
         a_ = A.values.mean(1); b_ = Bm.values.mean(1)
         summ['paired_bootstrap'][f'ce_f1_swapsft_minus_{ctl}'] = {**boot_paired(lambda i: a_[i].mean(1), lambda i: b_[i].mean(1), len(a_)),
                                                                'n_items': int(len(a_)), 'seeds': common, 'resamples': 10000}
         # inversion accuracy over E10 pairs; per seed accuracy over directional items in resample, then mean over seeds
         def load_inv(g, s):
             r = [json.loads(l) for l in open(OUT / 'runs' / f'{g}_seed{s}' / 'e10' / 'per_sample_swap.jsonl')]
             return {x['image_id']: (float(x['consistent']) if x['category'] == 'directional' else float('nan')) for x in r}
         ids = None
         M = {}
         for g in ('swapsft', ctl):
             arr = []
             for s in common:
                 d = load_inv(g, s); ids = ids or sorted(d); arr.append([d.get(i, float('nan')) for i in ids])
             M[g] = np.array(arr)   # [seeds, items]
         def inv(g):
             def f(idx):
                 out = np.zeros(idx.shape[0])
                 x = M[g]
                 vals = []
                 for si in range(x.shape[0]):
                     v = x[si][idx]; vals.append(np.nanmean(v, axis=1) if True else 0)
                 return np.mean(vals, axis=0)
             return f
         import warnings
         with warnings.catch_warnings():
             warnings.simplefilter('ignore')
             summ['paired_bootstrap'][f'inversion_acc_swapsft_minus_{ctl}'] = {**boot_paired(inv('swapsft'), inv(ctl), len(ids)),
                                                                           'n_items': len(ids), 'seeds': common, 'resamples': 10000}
         for k, lab in ((f'ce_f1_swapsft_minus_{ctl}', 'CE-F1'), (f'inversion_acc_swapsft_minus_{ctl}', 'directional inversion accuracy')):
             b = summ['paired_bootstrap'][k]
             lo, hi = b['ci95']
             if lo > 0: verdict.append(f'vs {ctl}, {lab}: swapsft > {ctl} (diff {b["diff"]:+.4f}, 95% CI [{lo:+.4f},{hi:+.4f}]).')
             elif hi < 0: verdict.append(f'vs {ctl}, {lab}: swapsft WORSE than {ctl} (diff {b["diff"]:+.4f}, 95% CI [{lo:+.4f},{hi:+.4f}]).')
             else: verdict.append(f'vs {ctl}, {lab}: no significant difference (diff {b["diff"]:+.4f}, 95% CI [{lo:+.4f},{hi:+.4f}] includes 0).')
         inv_b = summ['paired_bootstrap'][f'inversion_acc_swapsft_minus_{ctl}']
         helped = inv_b['ci95'][0] > 0
         summ[f'augmentation_helped_inversion_vs_{ctl}'] = bool(helped)
         verdict.append('' if helped else f'PLAINLY: vs {ctl} the augmentation did NOT demonstrably improve inversion accuracy.')
     else:
         verdict.append(f'INCOMPLETE: no matched {ctl}/swapsft seeds yet.')
    summ['verdict'] = ' '.join(v for v in verdict if v)
    (OUT / 'summary.json').write_text(json.dumps(summ, indent=2))
    lines = ['# TemporalSwap-SFT: temporal-equivariant augmentation', '',
             'Swapped training captions are RULE-GENERATED (word/verb substitution table, ricce/training/temporal_swap.py), not human-written.',
             'Controls: existing SFT seeds {1,42,123} (same hyperparameters/seeds) and sft2x seeds {1,42,123}. CONFOUND: swapsft sees 13038 rows vs 6815 for sft (~1.9x optimiser steps), so gains vs sft could be extra training. sft2x is the step-matched control: identical to swapsft (same 13038 rows/steps, seeds, hyperparameters) except its 6223 extra rows are the SAME images and captions in ORIGINAL order (dup_control_train.jsonl). Compare swapsft vs sft2x for the effect of the swap content alone.', '',
             '## Verdict', '', summ['verdict'], '', '## Per-seed', '', df.round(4).to_markdown(index=False) if len(df) else '(none yet)', '',
             '## Mean +- std over seeds', '']
    for g in groups:
        if g not in summ['per_metric']: continue
        lines.append(f'**{g}** (seeds {groups[g]}): ' + '; '.join(
            f"{k}={v['mean']:.4f}" + (f"+-{v['std']:.4f}" if v['std'] is not None else '') for k, v in summ['per_metric'][g].items()))
        lines.append('')
    lines += ['## Paired bootstrap (10,000 resamples of test items; statistic = mean over matched seeds)', '', '```', json.dumps(summ['paired_bootstrap'], indent=1), '```', '',
              'Notes: E10 uses the first 300 LEVIR-CC test pairs, greedy decoding; inversion accuracy is computed over the pairs whose normal-order prediction is directional, so the eligible set differs per model. '
              'With n=3 seeds the seed-level std is only indicative; the CIs above reflect test-item sampling, not training-seed variance.']
    (OUT / 'README.md').write_text('\n'.join(lines))
    print(summ['verdict'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--name'); ap.add_argument('--adapter'); ap.add_argument('--e10_limit', type=int, default=300)
    ap.add_argument('--aggregate', action='store_true')
    a = ap.parse_args()
    if a.aggregate:
        aggregate()
    else:
        assert a.name and a.adapter, '--name and --adapter required'
        eval_one(a)
