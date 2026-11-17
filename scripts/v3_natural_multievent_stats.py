"""v3: statistics for the natural multi-event analysis,
from results_v3/systems/natural_multievent_pairs.csv (759 real
(prediction, naturally multi-event reference) LEVIR-CC test pairs per system).
  * per metric: mean and 95% CI per system (cluster bootstrap over images);
  * per metric: Kendall tau between its 5-system order on these pairs and the
    order every metric (BLEU-4, CIDEr-D, CE-F) gives on the full test set;
  * per metric: paired difference fine-tuned (LoRA SFT) minus zero-shot Qwen on
    the same pairs, cluster bootstrap over images, two-sided p (floored);
  * how many predictions are themselves multi-event.
Outputs -> results_v3/natural_multievent/summary.json."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import kendalltau
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results_v3' / 'natural_multievent'; OUT.mkdir(parents=True, exist_ok=True)
N = pd.read_csv(ROOT / 'results_v3' / 'systems' / 'natural_multievent_pairs.csv', keep_default_na=False)
N['cef'] = pd.to_numeric(N.cef, errors='coerce')
MET = ['cef', 'kiritchenko_micro', 'hcm', 'taxaware_axisflat']
SYS = ['qwen', 'sft_seed42', 'swmcc', 'chg2cap']
full = pd.read_csv(ROOT / 'results_v3' / 'systems' / 'summary.csv')
full = full[full.dataset == 'LEVIR_CC'].set_index('system')
ref_order = {k: [full.loc[s, k] for s in SYS] for k in ('ce_f', 'bleu4', 'cider_d')}
assert min(kendalltau(ref_order['ce_f'], ref_order[k])[0] for k in ('bleu4', 'cider_d')) > 0.999, 'full-test-set orders disagree'
rng = np.random.default_rng(0); B = 10000
res = {'n_pairs_per_system': N.groupby('system').size().to_dict(), 'n_images': int(N.image_id.nunique()),
       'pred_multi_event_share': N.groupby('system').apply(lambda g: float((g.n_pred_events >= 2).mean())).to_dict(),
       'mean_ref_events': float(N.n_ref_events.mean()), 'metrics': {}}
imgs = np.array(sorted(N.image_id.unique())); idx = rng.integers(0, len(imgs), (B, len(imgs)))
for m in MET:
    per_img = N.groupby(['system', 'image_id'])[m].agg(['sum', 'count']).reset_index()
    d = {}
    means = {}
    for s in SYS:
        g = per_img[per_img.system == s].set_index('image_id').reindex(imgs).fillna(0)
        su, co = g['sum'].values, g['count'].values
        bm = su[idx].sum(1) / co[idx].sum(1)
        means[s] = float(su.sum() / co.sum())
        d[s] = dict(mean=means[s], lo=float(np.percentile(bm, 2.5)), hi=float(np.percentile(bm, 97.5)))
    a = N[N.system == 'sft_seed42'].reset_index(drop=True); b = N[N.system == 'qwen'].reset_index(drop=True)
    assert (a.image_id.values == b.image_id.values).all() and (a.reference.values == b.reference.values).all()
    diff = pd.DataFrame({'image_id': a.image_id, 'd': a[m].values - b[m].values}).dropna()
    gi = diff.groupby('image_id').d.agg(['sum', 'count']).reindex(imgs).fillna(0)
    bm = gi['sum'].values[idx].sum(1) / gi['count'].values[idx].sum(1)
    p = max(2 * min((bm <= 0).mean(), (bm >= 0).mean()), 1 / (B + 1))
    res['metrics'][m] = dict(per_system=d, tau_vs_full_test_order=float(kendalltau([means[s] for s in SYS], ref_order['ce_f'])[0]),
                             order=sorted(SYS, key=lambda s: -means[s]),
                             sft_minus_qwen=dict(diff=float(diff.d.mean()), lo=float(np.percentile(bm, 2.5)), hi=float(np.percentile(bm, 97.5)), p=float(p)))
json.dump(res, open(OUT / 'summary.json', 'w'), indent=1)
for m in MET:
    r = res['metrics'][m]
    print(m, 'tau', round(r['tau_vs_full_test_order'], 3), 'order', r['order'], 'sft-qwen', {k: round(v, 4) for k, v in r['sft_minus_qwen'].items()})
print('pred multi-event share', {k: round(v, 3) for k, v in res['pred_multi_event_share'].items()}, 'mean ref events', round(res['mean_ref_events'], 2))
