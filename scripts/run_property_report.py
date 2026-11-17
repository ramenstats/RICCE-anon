"""Generates results_cvpr_final/properties/{FAILURES.md,counterexample_separate_pr.md,summary.json,...}
from real metric calls: minimal reproducers for property failures + the
separate-P/R counterexample. Also runs the property test file."""
import sys, json, itertools, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
from ricce.schema import Event
from ricce.metrics.event import ricce_scores, match_events, event_similarity_pr
from test_metric_properties import TAXES, W, ev, no_change, sc

OUT = ROOT / 'results_cvpr_final' / 'properties'
OUT.mkdir(parents=True, exist_ok=True)


def fmt(e):
    d = {k: v for k, v in e.to_dict().items() if v not in (None, False, '') and k not in ('confidence', 'text')}
    return str(d)


def show(events):
    return '[' + '; '.join(fmt(e) for e in events) + ']'


# ---- minimal reproducer search over a small event pool (smallest total events first)
POOL = [ev('building', 'constructed', 'left', 'several'), ev('building', 'appearance', 'left', 'several'),
        ev('road', 'constructed', 'left', 'several'), ev('building', 'constructed', None, None),
        ev('residential_building', 'constructed', 'left', 'several'), ev('road', 'appearance', 'right', 'few'),
        ev('building', 'constructed', 'left', None), ev('vegetation', 'constructed', 'left', 'several')]


def find_omission_violation():
    best = None
    for npred in (2, 3):
        for ngt in (1, 2):
            for pred in itertools.product(POOL, repeat=npred):
                for gt in itertools.product(POOL, repeat=ngt):
                    base = sc(list(pred), list(gt))
                    for (i, j, sp, sr, sf) in base['matches']:
                        if sr <= 0:
                            continue
                        new = sc([p for k, p in enumerate(pred) if k != i], list(gt))
                        if new['ce_recall'] > base['ce_recall'] + 1e-9:
                            return list(pred), list(gt), i, base, new
    return None


def find_halluc_violation():
    for pred in itertools.product(POOL, repeat=2):
        for gt in itertools.product(POOL, repeat=2):
            base = sc(list(pred), list(gt))
            for extra in POOL:
                new = sc(list(pred) + [extra], list(gt))
                if len(pred) in {m[0] for m in new['matches']}:
                    continue
                if new['ce_precision'] > base['ce_precision'] + 1e-9:
                    return list(pred), list(gt), extra, base, new
    return None


fails = []
om = find_omission_violation()
md = ['# Property failures (real outputs of the current metric; tests NOT weakened)\n',
      'Tests live in `tests/test_metric_properties.py`; the failing ones are `xfail(strict=True)` so the suite stays green but any fix will surface.\n']
summary = {}
if om:
    pred, gt, i, base, new = om
    md += ['## F1. Omission monotonicity: removing a *matched* predicted event can RAISE CE-R\n',
           f'Minimal reproducer found by exhaustive search over an 8-event pool (smallest sizes first).\n',
           f'- prediction: `{show(pred)}`', f'- reference: `{show(gt)}`', f'- removed predicted event index: {i}\n',
           f'| | CE-P | CE-R | CE-F | matches (pred,gt,S_P,S_R,S_F) |', '|---|---|---|---|---|',
           f"| before | {base['ce_precision']:.4f} | {base['ce_recall']:.4f} | {base['ce_f1']:.4f} | {[tuple(round(x,3) for x in m) for m in base['matches']]} |",
           f"| after removal | {new['ce_precision']:.4f} | {new['ce_recall']:.4f} | {new['ce_f1']:.4f} | {[tuple(round(x,3) for x in m) for m in new['matches']]} |\n",
           'Cause (read off the matches above): pairs are chosen by maximising the harmonic S_F, but CE-R sums S_R of the chosen pairs. '
           'The removed prediction was the S_F-optimal match; the surviving prediction is a lower-S_F candidate that nevertheless has HIGHER S_R against the same reference event, '
           'so once the S_F-preferred match is deleted CE-R goes up. CE-R is therefore not monotone under deletion of a matched prediction whenever S_F and S_R rank candidates differently. '
           'Not fixed (core file untouched).\n']
    summary['omission_monotonicity'] = dict(violated=True, pred=show(pred), gt=show(gt), before_ce_r=base['ce_recall'], after_ce_r=new['ce_recall'])
else:
    summary['omission_monotonicity'] = dict(violated=False)
hv = find_halluc_violation()
summary['hallucination_monotonicity_exhaustive_pool'] = dict(violated=bool(hv))
if hv:
    pred, gt, extra, base, new = hv
    md += ['## F1b. Hallucination monotonicity (unmatched extra prediction raises CE-P)\n', f'- prediction {show(pred)}; extra {fmt(extra)}; reference {show(gt)}',
           f"- CE-P {base['ce_precision']:.4f} -> {new['ce_precision']:.4f}\n"]
# root-only
gt = [ev('residential_building', 'constructed', 'upper_left', 'several')]
root = Event(object='entity', action='temporal_state', location='image', quantity='quantity', change_state='change')
s = sc([root], gt)
md += ['## F2. Root-only prediction is not scored low\n',
       f'- reference: `{show(gt)}`', f'- prediction (every ontology field set to its taxonomy ROOT): `{show([root])}`',
       f"- CE-P={s['ce_precision']:.4f} CE-R={s['ce_recall']:.4f} CE-F={s['ce_f1']:.4f} (a-priori threshold in the test: CE-F < 0.5)\n",
       'Cause: a root node is an ancestor of everything, so hierarchical precision is exactly 1 for every root-only field, and change_state (weight 1.5) matches. '
       'CE-R is low (0.43) but CE-F is dominated by the perfect precision. Consequence: an uninformative prediction is not penalised on precision; '
       'report CE-P together with CE-R (which is what the paper does) and never CE-P alone. Not fixed (core file untouched).\n']
summary['root_only'] = dict(ce_p=s['ce_precision'], ce_r=s['ce_recall'], ce_f=s['ce_f1'])
e0 = sc([Event()], [Event()])
md += ['## F3. Identity = 1 fails for unparsed (all-None) events\n',
       'Real reproducers from RICCE-Bench identity items (candidate == reference text): "All is the same." and "All remained the same." both parse to an event with every field None (confidence 0.5).',
       f"- `Event()` vs `Event()`: CE-P={e0['ce_precision']:.4f} CE-R={e0['ce_recall']:.4f} CE-F={e0['ce_f1']:.4f} (expected 1.0)\n",
       'Cause: with no field present on either side, event_similarity_pr returns S_P=S_R=0 (empty numerator and denominator), so a perfectly identical caption that the parser cannot resolve scores 0. '
       'This is a parser-coverage x scorer-convention issue (2 of 3842 identity items in RICCE-Bench). Not fixed (core files untouched).\n']
summary['identity_unparsed'] = dict(ce_f=e0['ce_f1'], n_identity_items_affected=2, n_identity_items=3842)
r = subprocess.run([sys.executable, '-m', 'pytest', '-q', str(ROOT / 'tests' / 'test_metric_properties.py'), '-rxX'], capture_output=True, text=True, cwd=ROOT)
summary['pytest_tail'] = r.stdout.strip().splitlines()[-8:]
(OUT / 'FAILURES.md').write_text('\n'.join(md))

# ---- counterexample: separated vs collapsed
def collapsed(pred, gt):
    pairs, _ = match_events(pred, gt, TAXES, W)
    m = sum(p[4] for p in pairs)
    P, R = m / max(1, len(pred)), m / max(1, len(gt))
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)


def row(name, pred, gt):
    a = sc(pred, gt)
    P, R, F = collapsed(pred, gt)
    return dict(name=name, SP=a['ce_precision'], SR=a['ce_recall'], SF=a['ce_f1'], collapsed_F=F)


ref = [ev('building', 'constructed', 'left', 'several')]
coarse = [ev('built_environment', 'constructed', 'left', 'several')]
over = [ev('residential_building', 'constructed', 'left', 'several')]
A = [row('coarse-but-correct (pred=built_environment)', coarse, ref), row('unsupported over-specific (pred=residential_building)', over, ref)]
refR = [ev('residential_building', 'constructed', 'left', 'several')]
refB = [ev('building', 'constructed', 'left', 'several')]
B = [row('coarse-but-correct: ref=residential_building, pred=building', [ev('building', 'constructed', 'left', 'several')], refR),
     row('unsupported over-specific: ref=building, pred=residential_building', [ev('residential_building', 'constructed', 'left', 'several')], refB)]
def tbl(rows):
    o = ['| system | CE-P (SP) | CE-R (SR) | separated CE-F | collapsed-harmonic score (P=R=mean S_F) |', '|---|---|---|---|---|']
    for x in rows:
        o.append(f"| {x['name']} | {x['SP']:.4f} | {x['SR']:.4f} | {x['SF']:.4f} | {x['collapsed_F']:.4f} |")
    return '\n'.join(o)
anc = lambda n: TAXES['object'].ancestors(n)
cx = ['# Counterexample: why CE-P and CE-R must stay separate\n',
      'If per-event hierarchical precision/recall were first collapsed into one harmonic scalar S_F and only THEN aggregated, '
      'a correct-but-coarse prediction and an unsupported over-specific prediction lose the information that distinguishes them. '
      'All numbers below are computed by `scripts/run_property_report.py` with the real ontology/weights/matcher (test: `tests/test_counterexample_separate_pr.py`).\n',
      f"Object taxonomy path: building = {anc('building')}; residential_building = {anc('residential_building')}; built_environment = {anc('built_environment')}.\n",
      '## Scenario A -- same reference (`building`, constructed, left, several); both predictions differ only in object depth (target follows object)\n', tbl(A),
      f"\nSeparated: coarse SP={A[0]['SP']:.4f} SR={A[0]['SR']:.4f} vs over-specific SP={A[1]['SP']:.4f} SR={A[1]['SR']:.4f}: the coarse prediction asserts nothing false (SP=1, SR<1), the over-specific one asserts unsupported detail (SP<1, SR=1). "
      f"Collapsed: {A[0]['collapsed_F']:.4f} vs {A[1]['collapsed_F']:.4f} -> the over-specific (unsupported) prediction is ranked {'ABOVE' if A[1]['collapsed_F']>A[0]['collapsed_F'] else 'below/equal to'} the correct-but-coarse one: {'REVERSED' if A[1]['collapsed_F']>A[0]['collapsed_F'] else 'not reversed'}.\n",
      '## Scenario B -- mirrored pair (each system evaluated against its own reference)\n', tbl(B),
      f"\nSeparated: SP/SR are (1, {B[0]['SR']:.4f}) vs ({B[1]['SP']:.4f}, 1): distinguishable. Collapsed scores are {B[0]['collapsed_F']:.6f} vs {B[1]['collapsed_F']:.6f} -> "
      f"{'INDISTINGUISHABLE (equal to 1e-12)' if abs(B[0]['collapsed_F']-B[1]['collapsed_F'])<1e-12 else 'different'}.\n"]
(OUT / 'counterexample_separate_pr.md').write_text('\n'.join(cx))
summary['counterexample'] = dict(scenario_A=A, scenario_B=B)
json.dump(summary, open(OUT / 'summary.json', 'w'), indent=1)
json.dump(dict(pool=[fmt(e) for e in POOL], weights=W, note='exhaustive search over pool product; matcher=hungarian'), open(OUT / 'config.json', 'w'), indent=1)
(OUT / 'command.txt').write_text('python scripts/run_property_report.py\npython -m pytest -q tests/test_metric_properties.py tests/test_counterexample_separate_pr.py\n')
print(json.dumps(summary, indent=1)[:3000])
