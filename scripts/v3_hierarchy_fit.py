"""v3 analysis: how closely do captions (system predictions, and references
for comparison) match the RICCE hierarchy (configs/ontology/*.yaml)?

Three measurements, all with the frozen parser / final evaluator:

  A. caption -> hierarchy similarity (object, action, location axes): the
     parser's own MiniLM cosine between the normalized caption and every
     node label of the axis (HybridParser._semantic); we report the best-node
     cosine, the share at or above the frozen acceptance threshold (0.58),
     and which resolution stage resolves the axis on the whole caption
     (exact alias -> embedding -> unresolved; resolve_field order). Whole
     caption = the parser's single-event path; multi-event clause splitting
     is not re-run here.
  B. node usage, from the shared v3 parse cache (multi-event mode, exactly the
     events every v3 number is computed from): per axis, share of events with
     the field present, share at the root ('unspecified'), mean depth of a
     non-root node, and top-level branch distribution.
  C. prediction vs reference per axis: hierarchical precision/recall per axis
     on the final evaluator's alignment (FINAL_CONFIG; hP from the
     S_P-optimal assignment, hR from the S_R-optimal one), field scores are
     ungated (event_field_scores_v2); mean over scored event pairs, then over
     references of an image, then over images (same per-image averaging as
     v3_recompute_systems.py).
  D. reference-side gaps (follow-up to A): share of references with a
     location word whose parse has no location node, and object resolution on
     references parsed as 'change' vs 'no change'.

Outputs -> results_v3/hierarchy_fit/."""
import collections, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.utils import read_jsonl
from ricce.benchmarks.env import load_taxes, load_weights, load_cfg
from ricce.evaluation.parse_store import ParseStore
from ricce.metrics.final import FINAL_CONFIG
from ricce.metrics.event_v2 import match_events_v2, event_field_scores_v2, ricce_scores_v2, is_empty_event
from ricce.parsing.hybrid import HybridParser

OUT = ROOT / 'results_v3' / 'hierarchy_fit'; OUT.mkdir(parents=True, exist_ok=True)
T, W = load_taxes(), load_weights(); cfg = load_cfg()
store = ParseStore(ROOT / 'results_v3' / 'parse_cache.json')
MAIN = ['qwen', 'sft_seed42', 'swmcc', 'chg2cap', 'rsiccformer', 'promptcc', 'rscama']
DS = ['LEVIR_CC', 'SECOND_CC', 'DUBAI_CC']
AX_EMB = ['object', 'action', 'location']
AX_ALL = ['object', 'action', 'source', 'target', 'location', 'quantity', 'change_state']
TAX_OF = {'object': 'object', 'source': 'object', 'target': 'object', 'action': 'action', 'location': 'location', 'quantity': 'quantity'}
THR = float(cfg['parser']['similarity_threshold'])
man = {ds: {r['image_id']: r for r in read_jsonl(ROOT / 'artifacts' / 'manifests' / f'{ds}.jsonl') if r.get('split') == 'test'} for ds in DS}

# ------------------------------------------------------------------ caption sets
sources = {}  # (ds, source) -> {image_id: [captions]}
for ds in DS:
    for sy in MAIN:
        pf = ROOT / 'results' / 'baselines' / sy / f'{ds}_test.jsonl'
        if pf.exists():
            sources[(ds, sy)] = {r['image_id']: [str(r['prediction'])] for r in read_jsonl(pf) if r['image_id'] in man[ds]}
    ids = set().union(*[set(v) for (d, _), v in sources.items() if d == ds])  # images any system was run on
    sources[(ds, 'references')] = {i: [g for g in man[ds][i]['references'] if g and g.strip()] for i in sorted(ids)}

# ------------------------------------------------------------------ A. caption -> hierarchy similarity
t0 = time.time()
parser = HybridParser(T['object'], T['action'], T['location'], T['quantity'], use_embeddings=True,
                      embedding_model=cfg['parser']['embedding_model'], similarity_threshold=THR)
assert parser.embedder is not None, 'embedding model failed to load; similarity cannot be computed'
texts = sorted({c for v in sources.values() for caps in v.values() for c in caps})
norm = [HybridParser.normalize(t) for t in texts]
E = parser.embedder.encode(norm, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
cos, best, stage = {}, {}, {}
for ax in AX_EMB:
    ids, L = parser._label_embeddings(T[ax])
    S = E @ L.T
    cos[ax] = S.max(1); best[ax] = [ids[i] for i in S.argmax(1)]
    st = []
    for k, t in enumerate(norm):
        if parser._exact(t, T[ax]) is not None:
            st.append('alias')
        elif cos[ax][k] >= THR:
            st.append('embedding')
        else:
            st.append('unresolved')
    stage[ax] = st
tidx = {t: k for k, t in enumerate(texts)}
print(f'A: {len(texts)} unique captions embedded in {time.time()-t0:.0f}s', flush=True)

rowsA = []
for (ds, src), d in sources.items():
    ks = [tidx[c] for caps in d.values() for c in caps]
    r = dict(dataset=ds, source=src, n_captions=len(ks))
    for ax in AX_EMB:
        c = cos[ax][ks]; s = collections.Counter(stage[ax][k] for k in ks)
        r[f'{ax}_cos_mean'] = float(c.mean()); r[f'{ax}_cos_median'] = float(np.median(c))
        r[f'{ax}_cos_ge_thr'] = float((c >= THR).mean())
        for z in ('alias', 'embedding', 'unresolved'):
            r[f'{ax}_{z}'] = s[z] / len(ks)
    rowsA.append(r)
A = pd.DataFrame(rowsA); A.to_csv(OUT / 'A_caption_to_hierarchy.csv', index=False)

# ------------------------------------------------------------------ B. node usage (parse cache)
def top_branch(tax, n):
    a = tax.ancestors(n); return a[1] if len(a) > 1 else a[0]

rowsB, branches = [], {}
for (ds, src), d in sources.items():
    evs = [e for caps in d.values() for c in caps for e in store.events(c, 'multi')]
    caps = [c for caps in d.values() for c in caps]
    r = dict(dataset=ds, source=src, n_captions=len(caps), n_events=len(evs),
             events_per_caption=len(evs) / max(1, len(caps)),
             all_empty_caption_share=float(np.mean([all(is_empty_event(e, T, True) for e in store.events(c, 'multi')) for c in caps])))
    for ax in AX_ALL:
        vals = [getattr(e, ax) for e in evs]
        pres = [v for v in vals if v]
        r[f'{ax}_present'] = len(pres) / max(1, len(evs))
        if ax in TAX_OF:
            tax = T[TAX_OF[ax]]
            r[f'{ax}_root_share_of_present'] = float(np.mean([v == tax.root for v in pres])) if pres else np.nan
            nr = [v for v in pres if v != tax.root and v in tax.nodes]
            r[f'{ax}_mean_depth_nonroot'] = float(np.mean([tax.depth(v) for v in nr])) if nr else np.nan
            if ax in ('object', 'action', 'location'):
                cnt = collections.Counter(top_branch(tax, v) for v in nr)
                branches[f'{ds}|{src}|{ax}'] = {k: v / max(1, len(nr)) for k, v in cnt.most_common()}
    rowsB.append(r)
B = pd.DataFrame(rowsB); B.to_csv(OUT / 'B_node_usage.csv', index=False)
json.dump(branches, open(OUT / 'B_top_branch_distribution.json', 'w'), indent=1)
print('B done', flush=True)

# ------------------------------------------------------------------ C. prediction vs reference per axis
def axis_pr(pe, ge):
    """Per-axis (mean hP over S_P-optimal pairs, mean hR over S_R-optimal pairs) for one pred/ref pair."""
    s = ricce_scores_v2(pe, ge, T, W, **FINAL_CONFIG)
    if s['not_scorable']:
        return None
    hp = {a: [] for a in AX_ALL}; hr = {a: [] for a in AX_ALL}
    for i, j, _ in s['matches_p']:
        for a, (p, _r) in event_field_scores_v2(pe[i], ge[j], T, root_mode='unspecified').items():
            if p is not None: hp[a].append(p)
    for i, j, _ in s['matches_r']:
        for a, (_p, r) in event_field_scores_v2(pe[i], ge[j], T, root_mode='unspecified').items():
            if r is not None: hr[a].append(r)
    return ({a: np.mean(v) if v else np.nan for a, v in hp.items()},
            {a: np.mean(v) if v else np.nan for a, v in hr.items()}, s['ce_f1'])

rowsC = []
for (ds, src), d in sources.items():
    if src == 'references':
        continue
    per_img = []
    for iid, caps in d.items():
        pe = store.events(caps[0], 'multi')
        outs = [axis_pr(pe, store.events(g, 'multi')) for g in sources[(ds, 'references')][iid]]
        outs = [o for o in outs if o]
        if not outs:
            continue
        row = {'ce_f': np.mean([o[2] for o in outs])}
        for a in AX_ALL:
            ps = [o[0][a] for o in outs if np.isfinite(o[0][a])]; rs = [o[1][a] for o in outs if np.isfinite(o[1][a])]
            row[f'{a}_hP'] = np.mean(ps) if ps else np.nan; row[f'{a}_hR'] = np.mean(rs) if rs else np.nan
        per_img.append(row)
    P = pd.DataFrame(per_img)
    r = dict(dataset=ds, system=src, n_images=len(P), ce_f=float(P['ce_f'].mean()))
    for a in AX_ALL:
        for m in ('hP', 'hR'):
            col = P[f'{a}_{m}']; r[f'{a}_{m}'] = float(col.mean()); r[f'{a}_{m}_n_images'] = int(col.notna().sum())
    rowsC.append(r)
    print('C', ds, src, len(P), flush=True)
C = pd.DataFrame(rowsC); C.to_csv(OUT / 'C_pred_vs_ref_per_axis.csv', index=False)

json.dump({'threshold': THR, 'embedding_model': cfg['parser']['embedding_model'], 'ontology': cfg['ontology'],
           'final_config': FINAL_CONFIG, 'n_unique_captions': len(texts),
           'A': rowsA, 'B': rowsB, 'C': rowsC}, open(OUT / 'summary.json', 'w'), indent=1, default=float)
(OUT / 'README.md').write_text(
    '# Hierarchy fit (v3)\n\nCommand: `python scripts/v3_hierarchy_fit.py` (conda env ricce).\n\n'
    + (__doc__ or '') + '\n\nFiles: A_caption_to_hierarchy.csv, B_node_usage.csv, B_top_branch_distribution.json, '
    'C_pred_vs_ref_per_axis.csv, D_reference_gaps.csv, D_location_gap_examples.json, summary.json.\n')
print('done', time.time() - t0)

# ------------------------------------------------------------------ D. reference-side gaps (follow-up to A)
# D1: location words present in the caption but no location node resolved
# (any event of the cached parse); D2: object resolution on references whose
# parse says 'change' vs 'no change' (no-change captions name no object).
import re
LOCW = re.compile(r'\b(top|bottom|upper|lower|left|right|middle|center|centre|corner|side|north|south|east|west|edge|border)\b')
rowsD, gaps = [], {}
for ds in DS:
    refs = [g for caps in sources[(ds, 'references')].values() for g in caps]
    ev = {g: store.events(g, 'multi') for g in refs}
    has_w = [g for g in refs if LOCW.search(HybridParser.normalize(g))]
    no_loc = [g for g in has_w if not any(e.location for e in ev[g])]
    ch = [g for g in refs if any(e.change_state == 'change' for e in ev[g])]
    nc = [g for g in refs if g not in set(ch)]
    rowsD.append(dict(dataset=ds, n_refs=len(refs), share_with_location_word=len(has_w) / len(refs),
                      share_location_word_but_unresolved=len(no_loc) / max(1, len(has_w)),
                      n_changed_refs=len(ch), object_resolved_on_changed=float(np.mean([any(e.object for e in ev[g]) for g in ch])) if ch else np.nan,
                      n_nochange_or_unparsed_refs=len(nc), object_resolved_on_nochange=float(np.mean([any(e.object for e in ev[g]) for g in nc])) if nc else np.nan))
    cnt = collections.Counter(m.group(0) for g in no_loc for m in LOCW.finditer(HybridParser.normalize(g)))
    gaps[ds] = {'top_location_words_in_unresolved': cnt.most_common(12), 'examples': no_loc[:8]}
pd.DataFrame(rowsD).to_csv(OUT / 'D_reference_gaps.csv', index=False)
json.dump(gaps, open(OUT / 'D_location_gap_examples.json', 'w'), indent=1)
print(pd.DataFrame(rowsD).to_string()); print(json.dumps(gaps, indent=1)[:3000])
