"""Evaluation of the v2 evaluator variant (event_v2.py)
against the frozen v1 evaluator (event.py). Never modifies or re-derives any
existing headline number; writes only to results_metric_v2/. Sections:
  1. Property-test comparison (v1 vs v2-full) -- pytest, run separately, see
     tests/test_metric_properties_v2.py; this script does not duplicate it.
  2. RICCE-Bench ordering: core + semantic-validity suites, v1 vs v2-full,
     using the benchmark's own pre-parsed source_event/candidate_event (no
     re-parsing, so this isolates the SCORING formulation, not the parser).
  3. Gate ablation (G1 hard / G2 soft / G3 none), gate strength ISOLATED from
     the P1/P2 changes (exclude_root=False, dual_matching=False, i.e. G1 here
     reproduces v1 exactly): on RICCE-Bench ordering, on the gate stress
     sets (20 + 227 item), and on a freshly generated real same-image
     reference-pair false-contradiction test (first two references of every
     LEVIR-CC test image that has >=2), replacing the earlier, untraceable
     "400 real pairs" claim with a regenerated, fully-scripted one.
  4. System-ranking stability: v1 vs v2-full CE-F on the 8-system LEVIR-CC
     benchmark, Kendall tau between rankings.
"""
import json, os, sys, itertools
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.schema import Event
from ricce.utils import read_jsonl
from ricce.ontology.tree import Taxonomy
from ricce.parsing.hybrid import HybridParser
from ricce.metrics.event import ricce_scores
from ricce.metrics.event_v2 import ricce_scores_v2
import yaml

OUT = ROOT / 'results_metric_v2'
OUT.mkdir(parents=True, exist_ok=True)

cfg = yaml.safe_load(open(ROOT / 'configs/project.yaml'))
taxes = {k: Taxonomy.from_yaml(ROOT / v) for k, v in cfg['ontology'].items()}
W = cfg['metric']['weights']


def mk_events(dicts):
    return [Event(**{k: v for k, v in d.items() if k in Event.__dataclass_fields__}) for d in dicts]


def v1_f(cand, ref):
    return ricce_scores(cand, ref, taxes, W)['ce_f1']


def v2_f(cand, ref, gate_mode='hard', exclude_root=True, dual_matching=True):
    r = ricce_scores_v2(cand, ref, taxes, W, gate_mode=gate_mode, exclude_root=exclude_root, dual_matching=dual_matching)
    return None if r['not_scorable'] else r['ce_f1']


# ============================================================ 1. RICCE-Bench: v1 vs v2-full
print('=== RICCE-Bench: v1 vs v2-full scoring ===', flush=True)
items = []
with open(ROOT / 'artifacts_cvpr_final/riccebench/items.jsonl') as f:
    for l in f:
        d = json.loads(l)
        if not d['purity_checks']['pure']:
            continue
        items.append(dict(sample_id=d['sample_id'], source_id=d['source_id'], ptype=d['perturbation_type'],
                           dataset=d['dataset'],
                           cand=mk_events(d['candidate_event']), ref=mk_events(d['source_event'])))
print(f'{len(items)} pure items loaded', flush=True)

rows = []
for it in items:
    f1v1 = v1_f(it['cand'], it['ref'])
    f1v2 = v2_f(it['cand'], it['ref'], gate_mode='hard', exclude_root=True, dual_matching=True)
    rows.append(dict(sample_id=it['sample_id'], source_id=it['source_id'], ptype=it['ptype'], dataset=it['dataset'],
                      v1_cef=f1v1, v2_cef=f1v2))
bench = pd.DataFrame(rows)
bench.to_csv(OUT / 'bench_v1_vs_v2_scores.csv', index=False)
print('scored', len(bench), 'items', flush=True)


def ordering_stats(df, score_col):
    piv = df.pivot_table(index='source_id', columns='ptype', values=score_col)
    # core: identity/paraphrase/near > every corruption; paraphrase>near>distant
    CORRUPT = ['P3_sibling_substitution', 'P4_action_reversal', 'P5_location_swap', 'P6_count_corruption',
               'P7_event_omission', 'P8_event_hallucination', 'P9_change_state_reversal',
               'unsupported_descendant_specificity', 'source_target_swap', 'negation_flip']
    core_n, core_win = 0, 0
    for bad in CORRUPT:
        if 'P0_identity' in piv and bad in piv:
            x = piv[['P0_identity', bad]].dropna()
            core_win += int((x['P0_identity'] > x[bad] + 1e-9).sum()); core_n += len(x)
    for a, b in [('P0_paraphrase', 'P1_near_ancestor'), ('P1_near_ancestor', 'P2_distant_ancestor')]:
        if a in piv and b in piv:
            x = piv[[a, b]].dropna()
            core_win += int((x[a] > x[b] + 1e-9).sum()); core_n += len(x)
    # semantic-validity
    GOOD = ['P0_paraphrase', 'P1_near_ancestor']
    BAD = ['P4_action_reversal', 'P9_change_state_reversal', 'negation_flip', 'source_target_swap']
    val_n, val_win = 0, 0
    for gd in GOOD:
        for bd in BAD:
            if gd in piv and bd in piv:
                x = piv[[gd, bd]].dropna()
                val_win += int((x[gd] > x[bd] + 1e-9).sum()); val_n += len(x)
    return dict(core_n=core_n, core_acc=core_win / max(1, core_n), valid_n=val_n, valid_acc=val_win / max(1, val_n))


ord_v1 = ordering_stats(bench, 'v1_cef')
ord_v2 = ordering_stats(bench.dropna(subset=['v2_cef']), 'v2_cef')
print('v1 ordering:', ord_v1, flush=True)
print('v2-full ordering:', ord_v2, flush=True)
n_not_scorable = int(bench['v2_cef'].isna().sum())
print('v2 not_scorable items (both empty):', n_not_scorable, flush=True)

# ============================================================ 2. Gate ablation G1/G2/G3 (isolated: exclude_root=False, dual_matching=False)
print('=== Gate ablation G1 (hard) / G2 (soft) / G3 (none) on RICCE-Bench ===', flush=True)
gate_rows = {}
for gm in ('hard', 'soft', 'none'):
    col = f'gate_{gm}'
    bench[col] = [v2_f(it['cand'], it['ref'], gate_mode=gm, exclude_root=False, dual_matching=False) for it in items]
    gate_rows[gm] = ordering_stats(bench, col)
    print(gm, gate_rows[gm], flush=True)
bench.to_csv(OUT / 'bench_gate_ablation_scores.csv', index=False)
json.dump({'ordering': {'v1_frozen': ord_v1, 'v2_full': ord_v2, **{f'gate_{k}': v for k, v in gate_rows.items()}},
           'v2_not_scorable_items': n_not_scorable, 'n_items': len(bench)},
          open(OUT / 'bench_ordering_summary.json', 'w'), indent=1)
print('wrote', OUT / 'bench_ordering_summary.json', flush=True)

# ============================================================ 3. Gate stress sets (20 + 227) under G1/G2/G3
print('=== Gate stress test under G1/G2/G3 ===', flush=True)
parser = HybridParser(taxes['object'], taxes['action'], taxes['location'], taxes['quantity'],
                       use_embeddings=True, embedding_model=cfg['parser'].get('embedding_model'),
                       similarity_threshold=float(cfg['parser'].get('similarity_threshold', 0.58)))
stress_out = {}
for name, fn in [('original20', 'original_20_stress.jsonl'), ('enlarged227', 'action_reversal_stress.jsonl')]:
    stress_items = [json.loads(l) for l in open(ROOT / 'artifacts_r1' / fn)]
    for gm in ('hard', 'soft', 'none'):
        tp = fp = fn_ = tn = 0
        for it in stress_items:
            pe = parser.parse(it['pred'], allow_multi_event=True)
            ge = parser.parse(it['gt'], allow_multi_event=True)
            f1 = v2_f(pe, ge, gate_mode=gm, exclude_root=False, dual_matching=False)
            f1 = 0.0 if f1 is None else f1
            obs = 'gated' if f1 < 0.15 else 'not_gated'
            if it['expected'] == 'gated':
                tp += obs == 'gated'; fn_ += obs != 'gated'
            else:
                fp += obs == 'gated'; tn += obs != 'gated'
        n = tp + fp + fn_ + tn
        key = f'{name}_{gm}'
        stress_out[key] = dict(n=n, gated_recall=tp / max(1, tp + fn_), false_contradiction_rate=fp / max(1, fp + tn),
                                tp=int(tp), fp=int(fp), fn=int(fn_), tn=int(tn))
        print(key, stress_out[key], flush=True)
json.dump(stress_out, open(OUT / 'gate_stress_ablation.json', 'w'), indent=1)

# ============================================================ 4. Real same-image reference-pair false-contradiction (regenerates the
#    earlier, untraceable "400 real pairs" claim with a scripted, reproducible one)
print('=== Real same-image reference-pair false-contradiction rate (LEVIR-CC test) ===', flush=True)
man = list(read_jsonl(ROOT / 'artifacts/manifests/LEVIR_CC.jsonl'))
pair_texts = []
for r in man:
    refs = [x for x in r.get('references', []) if x and x.strip()]
    if len(refs) >= 2:
        pair_texts.append((r['image_id'], refs[0], refs[1]))
print(f'{len(pair_texts)} images with >=2 references; using first two references of each as a real reference pair', flush=True)
parsed_pairs = [(iid, parser.parse(a, allow_multi_event=True), parser.parse(b, allow_multi_event=True)) for iid, a, b in pair_texts]
refpair_out = {}
for gm in ('hard', 'soft', 'none'):
    fired = 0
    for iid, ea, eb in parsed_pairs:
        f1 = v2_f(ea, eb, gate_mode=gm, exclude_root=False, dual_matching=False)
        f1 = 0.0 if f1 is None else f1
        if f1 < 0.15:
            fired += 1
    refpair_out[gm] = dict(n=len(parsed_pairs), false_contradiction_rate=fired / max(1, len(parsed_pairs)), n_fired=fired)
    print(gm, refpair_out[gm], flush=True)
json.dump(refpair_out, open(OUT / 'real_reference_pair_false_contradiction.json', 'w'), indent=1)

# ============================================================ 5. System-ranking stability: v1 vs v2-full on the 8-system LEVIR-CC benchmark
print('=== System-ranking stability: v1 vs v2-full ===', flush=True)
SYSTEMS = ['qwen', 'sft_seed42', 'chg2cap', 'swmcc', 'rsiccformer', 'promptcc', 'rscama']
man_by_id = {r['image_id']: r for r in man}
sys_rows = []
for sy in SYSTEMS:
    pf = ROOT / 'results/baselines' / sy / 'LEVIR_CC_test.jsonl'
    if not pf.exists():
        print('missing', sy); continue
    preds = [r for r in read_jsonl(pf) if r['image_id'] in man_by_id]
    v1_scores, v2_scores = [], []
    for r in preds:
        pe = parser.parse(r['prediction'], allow_multi_event=True)
        refs = man_by_id[r['image_id']]['references']
        ge_list = [parser.parse(g, allow_multi_event=True) for g in refs]
        v1_scores.append(np.mean([v1_f(pe, ge) for ge in ge_list]))
        v2vals = [v2_f(pe, ge, gate_mode='hard', exclude_root=True, dual_matching=True) for ge in ge_list]
        v2vals = [x for x in v2vals if x is not None]
        v2_scores.append(np.mean(v2vals) if v2vals else np.nan)
    sys_rows.append(dict(system=sy, n=len(preds), v1_cef=float(np.mean(v1_scores)), v2_cef=float(np.nanmean(v2_scores))))
    print(sys_rows[-1], flush=True)
sysdf = pd.DataFrame(sys_rows)
sysdf.to_csv(OUT / 'system_ranking_v1_vs_v2.csv', index=False)
tau, p = kendalltau(sysdf.v1_cef, sysdf.v2_cef)
print('Kendall tau(v1 ranking, v2-full ranking) =', tau, 'p =', p, flush=True)
json.dump({'kendall_tau': float(tau), 'p': float(p), 'systems': sys_rows}, open(OUT / 'system_ranking_stability.json', 'w'), indent=1)

print('DONE', flush=True)
os._exit(0)
