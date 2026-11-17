"""v2-final: resolve the dual-vs-joint matching interpretation question.
Compares V2-A (separate S_P- and S_R-optimal assignments, the formulation
used throughout results_metric_v2/) against V2-B joint-assignment variants
(one assignment maximising a single combined objective, so CE-P and CE-R
come from the same event correspondence): harmonic (=S_F, reproduces the
frozen v1 matcher's assignment rule), mean (alpha=0.5), min, geometric.
Reports, per variant: whether the CE-R non-monotonicity property (P1) still
holds, RICCE-Bench ordering accuracy, 8-system ranking (Kendall tau vs.
v1), and runtime. Chosen on this comparison alone, never on a downstream
result already reported elsewhere."""
import json, random, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import kendalltau
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from ricce.schema import Event
from ricce.benchmarks.env import load_taxes, load_weights
from ricce.metrics.event import ricce_scores as v1_scores
from ricce.metrics.event_v2 import ricce_scores_v2

TAXES = load_taxes(); W = load_weights()
VARIANTS = {
    'V2-A dual': dict(dual_matching=True),
    'V2-B joint-harmonic': dict(dual_matching=False, joint_fn='harmonic'),
    'V2-B joint-mean(0.5)': dict(dual_matching=False, joint_fn='mean', joint_alpha=0.5),
    'V2-B joint-min': dict(dual_matching=False, joint_fn='min'),
    'V2-B joint-geometric': dict(dual_matching=False, joint_fn='geometric'),
}
COMMON = dict(gate_mode='hard', exclude_root=True)


def sc(pred, gt, variant):
    return ricce_scores_v2(pred, gt, TAXES, W, **COMMON, **VARIANTS[variant])


# ---------------- 1. recall-monotonicity property, per variant ----------------
def ev(obj='building', act='constructed', loc='left', qty='several', st='change', src=None, tgt='__auto__', cnt=None):
    if tgt == '__auto__':
        tgt = obj if act in ('constructed', 'appearance') else None
        if act in ('demolished', 'disappearance') and src is None:
            src = obj
    return Event(object=obj, action=act, source=src, target=tgt, location=loc, quantity=qty, exact_count=cnt, change_state=st)


def no_change():
    return Event(change_state='no_change', action='no_change')


def rand_event(rng):
    T = TAXES
    if rng.random() < 0.15:
        return no_change()
    obj = rng.choice(sorted(T['object'].nodes))
    act = rng.choice([a for a in sorted(T['action'].nodes) if a not in ('no_change', 'temporal_state')])
    e = ev(obj, act, rng.choice([None] + sorted(T['location'].nodes)), rng.choice([None] + sorted(T['quantity'].nodes)))
    if act == 'conversion':
        e.source, e.target = rng.choice(sorted(T['object'].nodes)), rng.choice(sorted(T['object'].nodes))
    return e


def rand_events(rng, lo, hi):
    return [rand_event(rng) for _ in range(rng.randint(lo, hi))]


def recall_monotonicity_violation_rate(variant, n=400, seed=3):
    rng = random.Random(seed)
    checked = violated = 0
    for _ in range(n):
        pred, gt = rand_events(rng, 2, 3), rand_events(rng, 1, 3)
        base = sc(pred, gt, variant)
        if base['not_scorable']:
            continue
        for (i, j, sr) in base['matches_r']:
            if sr <= 0:
                continue
            new = sc([p for k, p in enumerate(pred) if k != i], gt, variant)
            if new['not_scorable']:
                continue
            checked += 1
            if new['ce_recall'] > base['ce_recall'] + 1e-9:
                violated += 1
    return dict(checked=checked, violated=violated, violation_rate=violated / max(1, checked))


def root_only_score(variant):
    gt = [ev('residential_building', 'constructed', 'upper_left', 'several')]
    root = Event(object='entity', action='temporal_state', location='image', quantity='quantity', change_state='change')
    return sc([root], gt, variant)['ce_f1']


def order_invariance_ok(variant, n=200, seed=4):
    rng = random.Random(seed)
    ok = True
    for _ in range(n):
        pred, gt = rand_events(rng, 1, 4), rand_events(rng, 1, 4)
        a = sc(pred, gt, variant)
        p2, g2 = pred[:], gt[:]
        rng.shuffle(p2); rng.shuffle(g2)
        b = sc(p2, g2, variant)
        if a['not_scorable'] or b['not_scorable']:
            continue
        if abs(a['ce_f1'] - b['ce_f1']) > 1e-9:
            ok = False
    return ok


results = {}
for name in VARIANTS:
    t0 = time.time()
    rm = recall_monotonicity_violation_rate(name)
    root_f1 = root_only_score(name)
    oi = order_invariance_ok(name)
    dt = time.time() - t0
    results[name] = dict(recall_monotonicity=rm, root_only_cef=root_f1, order_invariant=oi, wallclock_400x3_checks_s=round(dt, 2))
    print(name, results[name], flush=True)

# ---------------- 2. RICCE-Bench ordering, per variant ----------------
print('=== RICCE-Bench ordering per variant ===', flush=True)
items = []
with open(ROOT / 'artifacts_cvpr_final/riccebench/items.jsonl') as f:
    for l in f:
        d = json.loads(l)
        if not d['purity_checks']['pure']:
            continue
        items.append(dict(source_id=d['source_id'], ptype=d['perturbation_type'],
                           cand=[Event(**{k: v for k, v in e.items() if k in Event.__dataclass_fields__}) for e in d['candidate_event']],
                           ref=[Event(**{k: v for k, v in e.items() if k in Event.__dataclass_fields__}) for e in d['source_event']]))


def ordering_stats(scores, items):
    df = pd.DataFrame({'source_id': [it['source_id'] for it in items], 'ptype': [it['ptype'] for it in items], 'score': scores})
    piv = df.pivot_table(index='source_id', columns='ptype', values='score')
    CORRUPT = ['P3_sibling_substitution', 'P4_action_reversal', 'P5_location_swap', 'P6_count_corruption',
               'P7_event_omission', 'P8_event_hallucination', 'P9_change_state_reversal',
               'unsupported_descendant_specificity', 'source_target_swap', 'negation_flip']
    core_n = core_win = 0
    for bad in CORRUPT:
        if 'P0_identity' in piv and bad in piv:
            x = piv[['P0_identity', bad]].dropna()
            core_win += int((x['P0_identity'] > x[bad] + 1e-9).sum()); core_n += len(x)
    for a, b in [('P0_paraphrase', 'P1_near_ancestor'), ('P1_near_ancestor', 'P2_distant_ancestor')]:
        if a in piv and b in piv:
            x = piv[[a, b]].dropna()
            core_win += int((x[a] > x[b] + 1e-9).sum()); core_n += len(x)
    GOOD = ['P0_paraphrase', 'P1_near_ancestor']; BAD = ['P4_action_reversal', 'P9_change_state_reversal', 'negation_flip', 'source_target_swap']
    val_n = val_win = 0
    for gd in GOOD:
        for bd in BAD:
            if gd in piv and bd in piv:
                x = piv[[gd, bd]].dropna()
                val_win += int((x[gd] > x[bd] + 1e-9).sum()); val_n += len(x)
    return dict(core_n=core_n, core_acc=core_win / max(1, core_n), valid_n=val_n, valid_acc=val_win / max(1, val_n))


for name in VARIANTS:
    scores = []
    for it in items:
        r = ricce_scores_v2(it['cand'], it['ref'], TAXES, W, **COMMON, **VARIANTS[name])
        scores.append(None if r['not_scorable'] else r['ce_f1'])
    scores = [np.nan if s is None else s for s in scores]
    st = ordering_stats(scores, items)
    results[name]['bench_ordering'] = st
    print(name, st, flush=True)

# ---------------- 3. 8-system ranking, per variant, vs v1 ----------------
print('=== system ranking per variant ===', flush=True)
from ricce.utils import read_jsonl
from ricce.parsing.hybrid import HybridParser
import yaml
cfg = yaml.safe_load(open(ROOT / 'configs/project.yaml'))
parser = HybridParser(TAXES['object'], TAXES['action'], TAXES['location'], TAXES['quantity'],
                       use_embeddings=True, embedding_model=cfg['parser'].get('embedding_model'),
                       similarity_threshold=float(cfg['parser'].get('similarity_threshold', 0.58)))
man = {r['image_id']: r for r in read_jsonl(ROOT / 'artifacts/manifests/LEVIR_CC.jsonl')}
SYSTEMS = ['qwen', 'sft_seed42', 'chg2cap', 'swmcc', 'rsiccformer', 'promptcc', 'rscama']
parsed_preds = {}
parsed_refs = {}
v1_means = {}
per_variant_means = {name: {} for name in VARIANTS}
for sy in SYSTEMS:
    pf = ROOT / 'results/baselines' / sy / 'LEVIR_CC_test.jsonl'
    if not pf.exists():
        continue
    preds = [r for r in read_jsonl(pf) if r['image_id'] in man]
    v1_scores = {name: [] for name in VARIANTS}
    for r in preds:
        pe = parser.parse(r['prediction'], allow_multi_event=True)
        ge_list = [parser.parse(g, allow_multi_event=True) for g in man[r['image_id']]['references']]
        for name in VARIANTS:
            vals = []
            for ge in ge_list:
                rr = ricce_scores_v2(pe, ge, TAXES, W, **COMMON, **VARIANTS[name])
                if not rr['not_scorable']:
                    vals.append(rr['ce_f1'])
            v1_scores[name].append(np.mean(vals) if vals else np.nan)
    for name in VARIANTS:
        per_variant_means[name][sy] = float(np.nanmean(v1_scores[name]))
    print(sy, 'done', flush=True)

for name in VARIANTS:
    results[name]['system_cef'] = per_variant_means[name]

# Kendall tau between each variant's ranking and V2-A dual's ranking (the reference already in results_metric_v2/)
base = [per_variant_means['V2-A dual'][s] for s in SYSTEMS if s in per_variant_means['V2-A dual']]
for name in VARIANTS:
    vals = [per_variant_means[name][s] for s in SYSTEMS if s in per_variant_means[name]]
    tau, p = kendalltau(base, vals)
    results[name]['tau_vs_V2A'] = float(tau)

OUT = ROOT / 'artifacts_v2'; OUT.mkdir(parents=True, exist_ok=True)
json.dump(results, open(OUT / 'dual_vs_joint_matching.json', 'w'), indent=1, default=str)
print(json.dumps(results, indent=1, default=str))
print('DONE')
