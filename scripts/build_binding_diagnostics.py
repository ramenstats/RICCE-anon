"""Construct controlled DIAGNOSTIC pairs from REAL reference captions (not E04):
  bind_swap  : a real two-event reference whose clauses have different resolved
               objects and actions; candidate = same caption with the two
               clauses' object phrases exchanged (event-binding corruption:
               the bag of labels is unchanged, the event structure is not).
  st_swap    : a real directional conversion reference ("X converted into Y");
               candidate = X and Y exchanged (source/target swap).
  identity   : reference vs itself (sanity control, all metrics must be 1).
  clause_order: same two clauses in reversed order (must stay 1 for all).
Everything is edited from real reference captions by string surgery on the
matched taxonomy alias spans; nothing is generated."""
import json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.baselines.flatten import FlatContext
from ricce.parsing.hybrid import HybridParser

OUT = ROOT / 'artifacts_cvpr_final' / 'baselines' / 'inputs'
ctx = FlatContext(use_embeddings=False)
P = ctx.parser
norm = HybridParser.normalize
otax = ctx.taxes['object']


def obj_span(text):
    """(start,end,node) of the longest object alias in (raw) clause text (regex on lowercase text)."""
    best = None
    low = text.lower()
    for nid, ph in otax.labels_and_aliases():
        ph = ph.lower()
        if not ph or nid == otax.root: continue
        m = re.search(rf'\b{re.escape(ph)}\b', low)
        if m and (best is None or (m.end() - m.start()) > (best[1] - best[0])):
            best = (m.start(), m.end(), nid)
    return best


rows = {'bind_swap': [], 'st_swap': [], 'identity': [], 'clause_order': []}
seen = set()
for d in ['LEVIR_CC', 'SECOND_CC', 'DUBAI_CC']:
    for l in open(ROOT / 'artifacts' / 'manifests' / f'{d}.jsonl'):
        x = json.loads(l)
        if x['split'] != 'test': continue
        for k, ref in enumerate(x['references']):
            if ref in seen: continue
            seen.add(ref)
            cl = P._split_clauses(ref)
            evs = ctx.parse(ref)
            if len(evs) == 2 and cl and len(cl) == 2:
                e1, e2 = evs
                s1, s2 = obj_span(cl[0]), obj_span(cl[1])
                if s1 and s2 and e1.object != e2.object and e1.action != e2.action and e1.object and e2.object:
                    c1, c2 = cl
                    n1 = c1[:s1[0]] + c2[s2[0]:s2[1]] + c1[s1[1]:]
                    n2 = c2[:s2[0]] + c1[s1[0]:s1[1]] + c2[s2[1]:]
                    cand = f'{n1} and {n2}'
                    rows['bind_swap'].append((d, x['image_id'], k, ref, cand))
                    rows['clause_order'].append((d, x['image_id'], k, ref, f'{c2} and {c1}'))
                    rows['identity'].append((d, x['image_id'], k, ref, ref))
            m = HybridParser._CONVERSION_DIR_RE.search(norm(ref))
            if m and len(evs) == 1 and evs[0].action == 'conversion' and evs[0].source and evs[0].target and evs[0].source != evs[0].target:
                nref = norm(ref)
                a, b = m.group(1).strip(), m.group(2).strip()
                # rebuild candidate on normalized text: swap the two sides around the marker
                marker = nref[len(m.group(1)):nref.index(m.group(2))] if m.group(2) in nref else None
                if marker:
                    rows['st_swap'].append((d, x['image_id'], k, ref, f'{b} {marker.strip()} {a}'))
for g, rs in rows.items():
    with open(OUT / f'diag_{g}.jsonl', 'w') as f:
        for d, i, k, ref, cand in rs:
            f.write(json.dumps({'id': f'{i}|r{k}', 'dataset': d, 'group': g, 'reference': ref, 'candidate': cand}) + '\n')
    print(g, len(rs))
    for r in rs[:2]: print('   REF:', r[3], '\n   CAND:', r[4])
