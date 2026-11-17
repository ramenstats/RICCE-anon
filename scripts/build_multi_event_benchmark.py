"""Build the v2 multi-event benchmark.

Two, clearly separated, sources of evidence:

 (A) REAL multi-event captions. We now scan EVERY reference caption in
     LEVIR_CC / SECOND_CC / DUBAI_CC (not a 1500-caption sample -- the full
     manifests, ~21.4k caption rows) with the newly extended
     HybridParser.parse(text, allow_multi_event=True), which only splits a
     caption into >1 event when every resulting clause independently
     resolves its own action verb (see ricce/parsing/hybrid.py). Any
     caption that genuinely parses to >1 event this way is tagged
     "source": "real" in the output benchmark -- these are real reference
     captions from the datasets, not invented text.

 (B) SYNTHETIC matching-algorithm stress tests. Per the task brief, we
     cannot invent a second real event for an image pair that only has one
     annotated event -- that would be fabrication. Instead we combine two
     (or three) REAL single-event reference captions belonging to DIFFERENT
     real image pairs into one synthetic multi-clause item, purely to stress
     -test the matching algorithm (Hungarian vs greedy vs order-based) with
     known ground-truth correspondences. Every such item is tagged
     "source": "synthetic_stress_test" and is NEVER used to claim anything
     about real captioner behaviour -- only about matcher behaviour.

Outputs:
  artifacts_r1/multi_event_benchmark.jsonl
  results_r1/multi_event/matching_comparison.json
  results_r1/multi_event/README.md
"""
import sys, random, itertools, json
from pathlib import Path
from collections import Counter
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.utils import read_jsonl, write_jsonl, dump_json
from ricce.ontology.tree import Taxonomy
from ricce.parsing.hybrid import HybridParser
from ricce.schema import Event
from ricce.metrics.event import (
    match_events, match_events_greedy, match_events_order, ricce_scores,
)

random.seed(20260901)

ART_OUT = ROOT / 'artifacts_r1'
RES_OUT = ROOT / 'results_r1' / 'multi_event'
ART_OUT.mkdir(parents=True, exist_ok=True)
RES_OUT.mkdir(parents=True, exist_ok=True)

cfg = yaml.safe_load(open(ROOT / 'configs' / 'project.yaml'))
taxes = {k: Taxonomy.from_yaml(ROOT / v) for k, v in cfg['ontology'].items()}
weights = cfg['metric']['weights']
# CPU-only, no embedding model needed for exact/alias resolution which is
# what the compound-clause action check relies on; matches the T5/T6 script.
parser = HybridParser(taxes['object'], taxes['action'], taxes['location'], taxes['quantity'], use_embeddings=False)

DATASETS = ['LEVIR_CC', 'SECOND_CC', 'DUBAI_CC']

# ---------------------------------------------------------------------------
# (A) Full-dataset scan for genuine real multi-event captions
# ---------------------------------------------------------------------------
real_multi_event_items = []
scan_stats = {}
single_event_atoms = []  # (dataset, image_id, caption_text, [Event]) for real, single-event, non-no_change captions

for ds in DATASETS:
    mpath = ROOT / 'artifacts' / 'manifests' / f'{ds}.jsonl'
    if not mpath.exists():
        continue
    rows = read_jsonl(mpath)
    n_captions = 0
    n_multi = 0
    for r in rows:
        for cap in r['references']:
            cap = cap.strip()
            if not cap:
                continue
            n_captions += 1
            events = parser.parse(cap, allow_multi_event=True)
            if len(events) > 1:
                n_multi += 1
                real_multi_event_items.append({
                    'source': 'real',
                    'dataset': ds,
                    'image_id': r['image_id'],
                    'caption': cap,
                    'n_events': len(events),
                    'events': [e.to_dict() for e in events],
                })
            else:
                e = events[0]
                if e.change_state == 'change' and e.action and e.object:
                    single_event_atoms.append({'dataset': ds, 'image_id': r['image_id'], 'caption': cap, 'event': e})
    scan_stats[ds] = {'n_captions_scanned': n_captions, 'n_real_multi_event': n_multi,
                       'fraction_multi_event': n_multi / max(1, n_captions)}
    print(ds, scan_stats[ds])

print(f'TOTAL real multi-event captions found (full-dataset scan, allow_multi_event=True): {len(real_multi_event_items)}')
print(f'Single-event atom pool available for synthetic construction: {len(single_event_atoms)}')

# ---------------------------------------------------------------------------
# (B) Synthetic multi-event stress-test construction
# ---------------------------------------------------------------------------
# Keep only atoms whose object/action/location all resolved cleanly -- these
# make the cleanest, least-ambiguous building blocks for controlled stress
# tests (we want the *matching algorithm* under test, not parser noise).
clean_atoms = [a for a in single_event_atoms if a['event'].location is not None]
random.shuffle(clean_atoms)
print(f'Clean atoms (object+action+location all resolved): {len(clean_atoms)}')

OBJECT_TAX = taxes['object']
ACTION_TAX = taxes['action']
LOCATION_TAX = taxes['location']


def clone_event(e, **overrides):
    d = e.to_dict()
    d.update(overrides)
    return Event(**d)


def coarsen_object(obj_id, hops=2):
    """Walk up the object taxonomy to build a deliberately coarse label for
    the 'one coarse + one specific event' test category."""
    anc = OBJECT_TAX.ancestors(obj_id)
    idx = max(0, len(anc) - 1 - hops)
    return anc[idx]


def event_summary(e):
    return {'object': e.object, 'action': e.action, 'source': e.source, 'target': e.target,
            'location': e.location, 'change_state': e.change_state, 'text': e.text}


synthetic_items = []


def add_item(category, atoms, gt_events, pred_events, note):
    synthetic_items.append({
        'source': 'synthetic_stress_test',
        'category': category,
        'atoms': [{'dataset': a['dataset'], 'image_id': a['image_id'], 'caption': a['caption']} for a in atoms],
        'synthetic_combined_caption': ' and '.join(a['caption'].rstrip('. ') for a in atoms) + '.',
        'gt_events': [event_summary(e) for e in gt_events],
        'pred_events': [event_summary(e) for e in pred_events],
        'note': note,
    })


N_PAIRS_TARGET = 60  # per category (2-event categories) -> ~9 categories * 60 = 540, trimmed to budget below
atom_pairs = []
used = set()
i = 0
while len(atom_pairs) < 400 and i + 1 < len(clean_atoms):
    a1, a2 = clean_atoms[i], clean_atoms[i + 1]
    key = (a1['image_id'], a2['image_id'])
    if a1['event'].object != a2['event'].object or a1['event'].action != a2['event'].action:
        atom_pairs.append((a1, a2))
    i += 2

print(f'Constructed {len(atom_pairs)} distinct real-atom pairs for 2-event synthetic combinations')

CATS_PER_PAIR = [
    'both_events_correct', 'one_event_omitted', 'one_event_hallucinated',
    'objects_swapped', 'actions_swapped', 'locations_swapped',
    'source_target_swapped', 'duplicate_predicted_event', 'coarse_vs_specific',
]

n_per_cat_target = 40  # 9 categories * 40 = 360 two-event items
pair_idx = 0
cat_counts = Counter()
for cat in CATS_PER_PAIR:
    made = 0
    while made < n_per_cat_target and pair_idx < len(atom_pairs):
        a1, a2 = atom_pairs[pair_idx]
        pair_idx += 1
        e1, e2 = a1['event'], a2['event']
        gt = [e1, e2]

        if cat == 'both_events_correct':
            pred = [clone_event(e1), clone_event(e2)]
            note = 'Predicted caption reproduces both real events exactly; expected P=R=F=1.0 for a correct matcher.'
        elif cat == 'one_event_omitted':
            pred = [clone_event(e1)]
            note = 'Predicted caption reports only the first event; second real event is omitted -> expected recall < 1.0.'
        elif cat == 'one_event_hallucinated':
            extra = clean_atoms[(pair_idx + 137) % len(clean_atoms)]['event']
            pred = [clone_event(e1), clone_event(e2), clone_event(extra)]
            note = 'Predicted caption adds a third event not present in either real reference -> expected precision < 1.0.'
        elif cat == 'objects_swapped':
            pred = [clone_event(e1, object=e2.object, source=e2.source if e1.source else e1.source,
                                 target=e2.object if e1.target else e1.target),
                    clone_event(e2, object=e1.object, source=e1.source if e2.source else e2.source,
                                 target=e1.object if e2.target else e2.target)]
            note = 'Objects swapped between the two predicted events (action/location kept in place) -- tests whether matcher/scorer correctly penalizes cross-wired object attribution.'
        elif cat == 'actions_swapped':
            def retarget(e, new_action):
                src = e.object if new_action in ('demolished', 'disappearance') else None
                tgt = e.object if new_action in ('constructed', 'appearance') else None
                return clone_event(e, action=new_action, source=src, target=tgt)
            pred = [retarget(e1, e2.action), retarget(e2, e1.action)]
            note = 'Actions swapped between the two predicted events (object/location kept in place).'
        elif cat == 'locations_swapped':
            pred = [clone_event(e1, location=e2.location), clone_event(e2, location=e1.location)]
            note = 'Locations swapped between the two predicted events.'
        elif cat == 'source_target_swapped':
            pred = [clone_event(e1, source=e1.target, target=e1.source),
                    clone_event(e2, source=e2.target, target=e2.source)]
            note = 'Source/target swapped within each predicted event (e.g. a construction event mis-described as demolition-shaped source/target).'
        elif cat == 'duplicate_predicted_event':
            pred = [clone_event(e1), clone_event(e1)]
            note = 'Predicted caption reports the first event twice and omits the second real event entirely -- tests whether the matcher lets one predicted event double-count against multiple reference events (Hungarian/greedy must not; naive order-based coincidentally also cannot here since it is 1:1 positional, but the duplicate still can only match one gt event).'
        elif cat == 'coarse_vs_specific':
            coarse_obj = coarsen_object(e1.object, hops=2)
            pred = [clone_event(e1, object=coarse_obj), clone_event(e2)]
            note = f'First predicted event generalizes the object to a coarse ancestor ({coarse_obj}) while the second stays specific and correct -- tests graceful hierarchical degradation is preserved under multi-event matching.'
        else:
            continue

        add_item(cat, [a1, a2], gt, pred, note)
        made += 1
        cat_counts[cat] += 1

print('2-event category counts:', dict(cat_counts))

# --- three-event cases ---
n_three = 0
triple_target = 60
while n_three < triple_target and pair_idx + 1 < len(clean_atoms):
    a1, a2, a3 = clean_atoms[pair_idx % len(clean_atoms)], clean_atoms[(pair_idx + 1) % len(clean_atoms)], clean_atoms[(pair_idx + 2) % len(clean_atoms)]
    pair_idx += 3
    e1, e2, e3 = a1['event'], a2['event'], a3['event']
    objs = {e1.object, e2.object, e3.object}
    if len(objs) < 3:
        continue
    variant = n_three % 3
    if variant == 0:
        pred = [clone_event(e1), clone_event(e2), clone_event(e3)]
        note = 'Three real events, all reproduced correctly -> expected P=R=F=1.0.'
        cat = 'three_event_all_correct'
    elif variant == 1:
        pred = [clone_event(e1), clone_event(e2)]
        note = 'Three real events; predicted caption omits the third -> expected recall ~2/3.'
        cat = 'three_event_one_omitted'
    else:
        pred = [clone_event(e1, object=e3.object), clone_event(e2), clone_event(e3, object=e1.object)]
        note = 'Three real events; objects rotated between event 1 and event 3 (event 2 stays correct) -- tests whether Hungarian correctly untangles a 3x3 ambiguous assignment vs greedy/order baselines.'
        cat = 'three_event_objects_rotated'
    add_item(cat, [a1, a2, a3], [e1, e2, e3], pred, note)
    n_three += 1

# --- adversarial greedy-suboptimality search ---
# The 2-event "swap" categories above (objects/actions/locations swapped),
# built from real-atom pairs, turned out to ALWAYS have Hungarian and greedy
# agree (see results below) -- 2x2 greedy happens to find the optimal
# assignment whenever the single highest-scoring cell participates in some
# optimal assignment, which is common but not guaranteed. To check whether
# Hungarian's optimality guarantee is EVER actually exercised (i.e. whether
# a genuine Hungarian != greedy counterexample is even reachable under this
# scoring function), we do a small, explicitly-labelled brute-force search
# directly over taxonomy labels for the classic bipartite-matching
# counterexample pattern: greedy's locally-best first pick forces a bad
# second pick, while a different (globally optimal) assignment scores
# higher in total. These items are NOT built from real reference captions
# at all -- they are directly-constructed Event objects using real ontology
# labels, purely to test whether the Hungarian-vs-greedy divergence is
# reachable in principle. Tagged as a distinct sub-category so this is never
# confused with the real-atom-combination stress tests above.
adversarial_items = []
obj_candidates = list(OBJECT_TAX.nodes.keys())
loc_candidates = list(LOCATION_TAX.nodes.keys())
random.shuffle(obj_candidates)
random.shuffle(loc_candidates)


def make_event(obj, action, loc):
    tgt = obj if action in ('constructed', 'appearance') else None
    src = obj if action in ('demolished', 'disappearance') else None
    return Event(object=obj, action=action, source=src, target=tgt, location=loc, change_state='change')


N_ADVERSARIAL_TARGET = 20
seen_signatures = set()
for o1 in obj_candidates[:10]:
    for o2 in obj_candidates[:10]:
        if len(adversarial_items) >= N_ADVERSARIAL_TARGET:
            break
        for a1, a2 in itertools.product(['constructed', 'demolished'], repeat=2):
            if len(adversarial_items) >= N_ADVERSARIAL_TARGET:
                break
            for l1, l2 in itertools.product(loc_candidates[:6], repeat=2):
                gt1 = make_event(o1, a1, l1)
                gt2 = make_event(o2, a2, l2)
                pred1 = make_event(o1, a1, l1)
                pred2 = make_event(o2, a2, l2)
                # perturb pred1/pred2 across a small grid to look for the
                # greedy-trap pattern, mirroring the exploratory search that
                # first surfaced this counterexample.
                found = False
                for po1 in obj_candidates[:10]:
                    for pl1 in loc_candidates[:6]:
                        p1 = make_event(po1, a1, pl1)
                        for po2 in obj_candidates[:10]:
                            for pl2 in loc_candidates[:6]:
                                p2 = make_event(po2, a2, pl2)
                                pairs_h, _ = match_events([p1, p2], [gt1, gt2], taxes, weights)
                                pairs_g, _ = match_events_greedy([p1, p2], [gt1, gt2], taxes, weights)
                                mh = {(i, j) for i, j, *_ in pairs_h}
                                mg = {(i, j) for i, j, *_ in pairs_g}
                                if mh != mg:
                                    fh = sum(x[4] for x in pairs_h)
                                    fg = sum(x[4] for x in pairs_g)
                                    if fh > fg + 0.01:
                                        sig = (o1, a1, l1, o2, a2, l2, po1, pl1, po2, pl2)
                                        if sig in seen_signatures:
                                            continue
                                        seen_signatures.add(sig)
                                        adversarial_items.append({
                                            'source': 'synthetic_stress_test',
                                            'category': 'adversarial_greedy_suboptimal',
                                            'construction': 'direct_taxonomy_labels_not_real_captions',
                                            'gt_events': [event_summary(gt1), event_summary(gt2)],
                                            'pred_events': [event_summary(p1), event_summary(p2)],
                                            'note': (f'Directly-constructed adversarial case (brute-force search over '
                                                      f'taxonomy labels, NOT derived from any real reference caption): '
                                                      f'greedy picks the single highest-scoring cell first, which forces '
                                                      f'a worse second assignment; Hungarian finds the globally optimal '
                                                      f'assignment instead. Hungarian total S_F={fh:.4f} > greedy total '
                                                      f'S_F={fg:.4f}.'),
                                        })
                                        found = True
                                        break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if len(adversarial_items) >= N_ADVERSARIAL_TARGET:
                    break
            if len(adversarial_items) >= N_ADVERSARIAL_TARGET:
                break
        if len(adversarial_items) >= N_ADVERSARIAL_TARGET:
            break
    if len(adversarial_items) >= N_ADVERSARIAL_TARGET:
        break

print(f'Found {len(adversarial_items)} adversarial Hungarian != greedy counterexamples via direct taxonomy-label construction')
synthetic_items.extend(adversarial_items)

print(f'Constructed {n_three} three-event synthetic items')

print(f'Total synthetic stress-test items: {len(synthetic_items)}')
print(f'Total benchmark items (real + synthetic): {len(real_multi_event_items) + len(synthetic_items)}')

# ---------------------------------------------------------------------------
# Write benchmark jsonl
# ---------------------------------------------------------------------------
benchmark_rows = []
for it in real_multi_event_items:
    benchmark_rows.append({
        'source': 'real', 'category': 'real_multi_event_caption',
        'dataset': it['dataset'], 'image_id': it['image_id'], 'caption': it['caption'],
        'n_events': it['n_events'], 'events': it['events'],
    })
for it in synthetic_items:
    benchmark_rows.append(it)

write_jsonl(benchmark_rows, ART_OUT / 'multi_event_benchmark.jsonl')
print(f'Wrote {len(benchmark_rows)} rows to {ART_OUT / "multi_event_benchmark.jsonl"}')

# ---------------------------------------------------------------------------
# Score every item with Hungarian / greedy / order-based matchers and
# compare where they diverge.
# ---------------------------------------------------------------------------


def events_from_summaries(dicts):
    return [Event(**{k: v for k, v in d.items() if k in Event.__dataclass_fields__}) for d in dicts]


def score_item(pred_events, gt_events):
    out = {}
    for name, fn in [('hungarian', match_events), ('greedy', match_events_greedy), ('order', match_events_order)]:
        s = ricce_scores(pred_events, gt_events, taxes, weights, matcher=fn)
        out[name] = {'ce_precision': s['ce_precision'], 'ce_recall': s['ce_recall'], 'ce_f1': s['ce_f1'],
                     'matches': [(p, g) for p, g, *_ in s['matches']],
                     'hallucination_rate': s['hallucination_rate'], 'omission_rate': s['omission_rate']}
    return out


per_item_scores = []
divergent_cases = []

for it in real_multi_event_items:
    events = [Event(**{k: v for k, v in ev.items() if k in Event.__dataclass_fields__}) for ev in it['events']]
    # Self-match (pred==gt==parsed events) exercises the matcher on a real
    # multi-event caption's own parsed events; the informative comparison
    # here is the assignment stability across algorithms.
    scores = score_item(events, events)
    row = {'source': 'real', 'dataset': it['dataset'], 'image_id': it['image_id'], 'caption': it['caption'],
           'n_pred': len(events), 'n_gt': len(events), 'scores': scores}
    per_item_scores.append(row)
    if scores['hungarian']['matches'] != scores['greedy']['matches'] or scores['hungarian']['matches'] != scores['order']['matches']:
        divergent_cases.append(row)

for it in synthetic_items:
    pred_events = events_from_summaries(it['pred_events'])
    gt_events = events_from_summaries(it['gt_events'])
    scores = score_item(pred_events, gt_events)
    row = {'source': 'synthetic_stress_test', 'category': it['category'], 'atoms': it.get('atoms', []),
           'synthetic_combined_caption': it.get('synthetic_combined_caption'), 'note': it['note'],
           'n_pred': len(pred_events), 'n_gt': len(gt_events), 'scores': scores}
    per_item_scores.append(row)
    matches_diff = (scores['hungarian']['matches'] != scores['greedy']['matches'] or
                     scores['hungarian']['matches'] != scores['order']['matches'])
    f1_diff = (abs(scores['hungarian']['ce_f1'] - scores['greedy']['ce_f1']) > 1e-9 or
               abs(scores['hungarian']['ce_f1'] - scores['order']['ce_f1']) > 1e-9)
    if matches_diff or f1_diff:
        divergent_cases.append(row)

print(f'Items where assignment or score differs between matchers: {len(divergent_cases)} / {len(per_item_scores)}')

# --- aggregate CE-P/CE-R/CE-F per method, overall and per category ---


def agg(rows, key_fn=None):
    buckets = {}
    for r in rows:
        k = key_fn(r) if key_fn else 'all'
        buckets.setdefault(k, []).append(r)
    out = {}
    for k, rs in buckets.items():
        out[k] = {'n': len(rs)}
        for method in ('hungarian', 'greedy', 'order'):
            out[k][method] = {
                'ce_precision_mean': sum(r['scores'][method]['ce_precision'] for r in rs) / len(rs),
                'ce_recall_mean': sum(r['scores'][method]['ce_recall'] for r in rs) / len(rs),
                'ce_f1_mean': sum(r['scores'][method]['ce_f1'] for r in rs) / len(rs),
                'hallucination_rate_mean': sum(r['scores'][method]['hallucination_rate'] for r in rs) / len(rs),
                'omission_rate_mean': sum(r['scores'][method]['omission_rate'] for r in rs) / len(rs),
            }
    return out


synthetic_rows = [r for r in per_item_scores if r['source'] == 'synthetic_stress_test']
real_rows = [r for r in per_item_scores if r['source'] == 'real']

overall_synthetic = agg(synthetic_rows)
by_category = agg(synthetic_rows, key_fn=lambda r: r['category'])
overall_real = agg(real_rows) if real_rows else {}

def _divergence_priority(row):
    # Prefer showing synthetic (labelled, interpretable) divergences first,
    # and within those, prefer cases where Hungarian's *F1* actually beats
    # greedy's/order's F1 (a real quality difference) over cases that only
    # differ by assignment permutation with equal F1 (a tie-break, not a
    # quality difference).
    is_synth = 0 if row['source'] == 'synthetic_stress_test' else 1
    f1_gap = max(abs(row['scores']['hungarian']['ce_f1'] - row['scores']['greedy']['ce_f1']),
                 abs(row['scores']['hungarian']['ce_f1'] - row['scores']['order']['ce_f1']))
    return (is_synth, -f1_gap)


divergent_cases.sort(key=_divergence_priority)

concrete_divergent_examples = []
for row in divergent_cases[:25]:
    concrete_divergent_examples.append({
        'source': row['source'],
        'category': row.get('category', row.get('dataset')),
        'caption_or_combo': row.get('caption') or row.get('synthetic_combined_caption') or 'directly-constructed events (no caption text; see multi_event_benchmark.jsonl for gt_events/pred_events)',
        'note': row.get('note'),
        'hungarian': {'matches': row['scores']['hungarian']['matches'], 'ce_f1': row['scores']['hungarian']['ce_f1']},
        'greedy': {'matches': row['scores']['greedy']['matches'], 'ce_f1': row['scores']['greedy']['ce_f1']},
        'order': {'matches': row['scores']['order']['matches'], 'ce_f1': row['scores']['order']['ce_f1']},
    })

# --- explicit F1-quality divergence counts (distinct from assignment-tuple
# divergence, which can differ by a tied-score permutation with identical
# F1) -- this is the number that actually answers "does Hungarian's
# optimality guarantee ever matter" ---
f1_diverge_hg_by_cat = Counter()
f1_diverge_ho_by_cat = Counter()
f1_diverge_go_by_cat = Counter()
for r in synthetic_rows:
    cat = r['category']
    sh, sg, so = r['scores']['hungarian'], r['scores']['greedy'], r['scores']['order']
    if abs(sh['ce_f1'] - sg['ce_f1']) > 1e-9:
        f1_diverge_hg_by_cat[cat] += 1
    if abs(sh['ce_f1'] - so['ce_f1']) > 1e-9:
        f1_diverge_ho_by_cat[cat] += 1
    if abs(sg['ce_f1'] - so['ce_f1']) > 1e-9:
        f1_diverge_go_by_cat[cat] += 1

f1_divergence = {
    'hungarian_vs_greedy': {
        'n_items_with_different_ce_f1': sum(f1_diverge_hg_by_cat.values()),
        'n_synthetic_items_total': len(synthetic_rows),
        'by_category': dict(f1_diverge_hg_by_cat),
    },
    'hungarian_vs_order': {
        'n_items_with_different_ce_f1': sum(f1_diverge_ho_by_cat.values()),
        'n_synthetic_items_total': len(synthetic_rows),
        'by_category': dict(f1_diverge_ho_by_cat),
    },
    'greedy_vs_order': {
        'n_items_with_different_ce_f1': sum(f1_diverge_go_by_cat.values()),
        'n_synthetic_items_total': len(synthetic_rows),
        'by_category': dict(f1_diverge_go_by_cat),
    },
    'note': ('"n_items_with_different_ce_f1" counts a genuine SCORE quality difference '
             '(the two matchers disagree on the actual CE-F1 they output), which is the '
             'meaningful question -- distinct from "assignment differs" counts elsewhere '
             'in this file, which can include cases where two different assignments happen '
             'to tie in total score (a coincidental permutation, not a quality difference).'),
}

comparison = {
    'real_multi_event_scan': {
        'method': 'Full (non-sampled) scan of every reference caption in artifacts/manifests/{LEVIR_CC,SECOND_CC,DUBAI_CC}.jsonl using the newly extended HybridParser.parse(text, allow_multi_event=True), which splits a caption into >1 event only when every resulting clause independently resolves its own non-no_change action verb.',
        'per_dataset': scan_stats,
        'total_captions_scanned': sum(v['n_captions_scanned'] for v in scan_stats.values()),
        'total_real_multi_event_captions_found': len(real_multi_event_items),
    },
    'synthetic_stress_test': {
        'method': 'Synthetic multi-event items built by combining REAL single-event reference captions from DIFFERENT real image pairs; used only to stress-test the matching ALGORITHM with known ground-truth correspondences, never as a real-world accuracy claim.',
        'n_items': len(synthetic_items),
        'n_two_event_items': sum(1 for it in synthetic_items if len(it['gt_events']) == 2),
        'n_three_event_items': sum(1 for it in synthetic_items if len(it['gt_events']) == 3),
        'categories': dict(Counter(it['category'] for it in synthetic_items)),
    },
    'divergence_summary': {
        'n_items_scored_total': len(per_item_scores),
        'n_items_where_hungarian_differs_from_greedy_or_order': len(divergent_cases),
        'fraction_divergent': len(divergent_cases) / max(1, len(per_item_scores)),
        'concrete_divergent_examples': concrete_divergent_examples,
    },
    'f1_quality_divergence': f1_divergence,
    'aggregate_scores_synthetic_overall': overall_synthetic,
    'aggregate_scores_synthetic_by_category': by_category,
    'aggregate_scores_real_multi_event_self_match': overall_real,
}

dump_json(comparison, RES_OUT / 'matching_comparison.json')
print(f'Wrote comparison to {RES_OUT / "matching_comparison.json"}')

print('\n=== SUMMARY ===')
print('Real multi-event captions found (full scan):', len(real_multi_event_items))
print('Synthetic stress-test items:', len(synthetic_items))
print('Divergent (Hungarian != greedy or order) items:', len(divergent_cases), '/', len(per_item_scores))
