"""v3 analysis: WordNet hierarchy vs the manual RICCE ontology as the
hierarchy against which prediction/reference captions are compared.

Both hierarchies are used through the SAME bag procedure, so only the
hierarchy and its token->node mapping differ:
  * words of a caption -> nodes (WordNet: synsets; manual: the frozen parser's
    object / action nodes from the shared v3 parse cache);
  * symmetric node similarity (WordNet: Wu-Palmer; manual: Dice of ancestor
    sets with the root removed = Kiritchenko hF, tax.hierarchical_scores);
  * precision = mean over predicted nodes of the best match among reference
    nodes, recall = the reverse; an axis empty on one side gives no
    precision (or recall) on that side, empty on both = not scored;
    mean over scored references -> image -> dataset (as in the main evaluation).

WordNet side, no part-of-speech tagger is available offline here, so:
  * candidate WORDS: lower-cased alphabetic tokens not in the stop / location /
    quantity / generic lists below;
  * a word is a NOUN candidate if its first (most frequent) WordNet sense is a
    noun, a VERB candidate if the first sense is a verb;
  * word similarity: 'first' = first sense only; 'max' = max over all senses
    of the right part of speech (generous).
Wikipedia/Wikidata hierarchies are NOT evaluated here (see README).

Run with argument `changed_only` to keep only images whose prediction and all references parse as a change
(removes the change/no-change confound of noun candidates such as 'nothing').
Tests:
  T1 coverage: share of captions with >=1 candidate / manual node.
  T2 discrimination: for each system prediction, similarity to its own
     reference(s) vs to the reference(s) of a different random image
     (seed 0); AUC over images and mean gap. Higher similarity alone is not
     better; a hierarchy must separate matched from mismatched content.
  T3 direction: similarity of opposite action pairs vs identical pairs.
  T4 object granularity: similarities of fixed object pairs.
Outputs -> results_v3/wordnet_hierarchy/."""
import collections, json, sys, itertools
from pathlib import Path
import numpy as np, pandas as pd
from nltk.corpus import wordnet as wn
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.utils import read_jsonl
from ricce.benchmarks.env import load_taxes
from ricce.evaluation.parse_store import ParseStore
from ricce.parsing.hybrid import HybridParser

CHANGED_ONLY = len(sys.argv) > 1 and sys.argv[1] == 'changed_only'  # keep only images whose prediction AND every reference parse as 'change'
OUT = ROOT / 'results_v3' / ('wordnet_hierarchy_changed_only' if CHANGED_ONLY else 'wordnet_hierarchy'); OUT.mkdir(parents=True, exist_ok=True)
T = load_taxes(); store = ParseStore(ROOT / 'results_v3' / 'parse_cache.json')
DS = ['LEVIR_CC', 'SECOND_CC', 'DUBAI_CC']
MAIN = ['qwen', 'sft_seed42', 'swmcc', 'chg2cap', 'rsiccformer', 'promptcc', 'rscama']

STOP = set('a an the of in on at to and or is are was were be been being by with from for as its it this that there these those near some into onto over under between while than then also has have had do does did not no nor can could would should will may might more most many much several few other another each both all any one two three four five six seven eight nine ten lot lots number numbers'.split())
LOC = set('top bottom left right side sides corner corners edge edges center centre middle upper lower part parts region direction western eastern northern southern north south east west northwest northeast southwest southeast'.split())
GENERIC = set('scene scenes image images picture area areas place places difference differences change changes thing things way kind type time times same different new old whole entire overall between two seem seems seemed remain remains remained appear appears appeared look looks looked'.split()) - {'appear', 'appears', 'appeared'}
EXCL_NOUN = STOP | LOC | GENERIC
AUX = set('is are was were be been being has have had do does did seem seems seemed remain remains remained get gets got become becomes became look looks looked take takes took place placed make made'.split())
EXCL_VERB = STOP | LOC | AUX
TOK = lambda s: [t for t in HybridParser.normalize(s).split() if t.isalpha()]

_first = {}
def first_pos(w):
    if w not in _first:
        s = wn.synsets(w)
        _first[w] = s[0].pos() if s else None
    return _first[w]

def words(caption):
    toks = TOK(caption)
    nouns = [t for t in toks if t not in EXCL_NOUN and first_pos(t) == 'n']
    verbs = [t for t in toks if t not in EXCL_VERB and first_pos(t) == 'v']
    return nouns, verbs

_syn = {}
def syns(w, pos):
    k = (w, pos)
    if k not in _syn:
        _syn[k] = wn.synsets(w, pos=pos)
    return _syn[k]

_ws = {}
def wsim(a, b, pos, mode):
    k = (a, b, pos, mode)
    if k in _ws:
        return _ws[k]
    A, B = syns(a, pos), syns(b, pos)
    if mode == 'first':
        A, B = A[:1], B[:1]
    best = 0.0
    for x in A:
        for y in B:
            v = x.wup_similarity(y)
            if v is not None and v > best:
                best = v
    _ws[k] = best; _ws[(b, a, pos, mode)] = best
    return best

_ms = {}
def msim(tax_key, a, b):
    k = (tax_key, a, b)
    if k not in _ms:
        t = T[tax_key]
        _ms[k] = t.hierarchical_scores(a, b, exclude_root=True)[2] if (a in t.nodes and b in t.nodes) else 0.0
    return _ms[k]

def bag_pr(P, R, sim):
    """(precision, recall); None for an undefined side. Not scorable -> (None, None, True)."""
    if not P and not R:
        return None, None
    pr = (float(np.mean([max((sim(p, r) for r in R), default=0.0) for p in P])) if P else None)
    rc = (float(np.mean([max((sim(r, p) for p in P), default=0.0) for r in R])) if R else None)
    return pr, rc

def f(pr, rc):
    if pr is None or rc is None:
        return None
    return 0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc)

# ------------------------------------------------------------------ caption representations
man = {ds: {r['image_id']: r for r in read_jsonl(ROOT / 'artifacts' / 'manifests' / f'{ds}.jsonl') if r.get('split') == 'test'} for ds in DS}
rep = {}  # caption -> dict(wn_n, wn_v, m_obj, m_act)
def R(c):
    if c not in rep:
        n, v = words(c)
        ev = store.events(c, 'multi')
        mo = [e.object for e in ev if e.object and e.object != T['object'].root]
        ma = [e.action for e in ev if e.action and e.action != T['action'].root]
        rep[c] = dict(n=n, v=v, mo=mo, ma=ma)
    return rep[c]

VARIANTS = {  # name -> (axis, kind, sim fn)
    'obj_manual':      ('mo', lambda a, b: msim('object', a, b)),
    'obj_wn_first':    ('n',  lambda a, b: wsim(a, b, 'n', 'first')),
    'obj_wn_max':      ('n',  lambda a, b: wsim(a, b, 'n', 'max')),
    'act_manual':      ('ma', lambda a, b: msim('action', a, b)),
    'act_wn_first':    ('v',  lambda a, b: wsim(a, b, 'v', 'first')),
    'act_wn_max':      ('v',  lambda a, b: wsim(a, b, 'v', 'max')),
}

def pair_scores(pc, rc):
    a, b = R(pc), R(rc)
    out = {}
    for name, (key, sim) in VARIANTS.items():
        p, r = bag_pr(a[key], b[key], sim)
        out[name] = (p, r)
    return out

def mean_ignore(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None

def image_score(pc, refs):
    """per variant (P, R, F) averaged over scored references."""
    per = [pair_scores(pc, g) for g in refs]
    res = {}
    for name in VARIANTS:
        ps = [x[name][0] for x in per]; rs = [x[name][1] for x in per]
        fs = [f(*x[name]) for x in per]
        res[name] = (mean_ignore(ps), mean_ignore(rs), mean_ignore(fs))
    return res

# ------------------------------------------------------------------ sources
sources = {}
for ds in DS:
    for sy in MAIN:
        pf = ROOT / 'results' / 'baselines' / sy / f'{ds}_test.jsonl'
        if pf.exists():
            sources[(ds, sy)] = {r['image_id']: str(r['prediction']) for r in read_jsonl(pf) if r['image_id'] in man[ds]}
refs_of = {ds: {i: [g for g in man[ds][i]['references'] if g and g.strip()] for i in man[ds]} for ds in DS}

# ------------------------------------------------------------------ T1 coverage
rows1 = []
for ds in DS:
    capsets = {'references': [g for i in {i for (d, _), v in sources.items() if d == ds for i in v} for g in refs_of[ds][i]]}
    for (d, sy), v in sources.items():
        if d == ds: capsets[sy] = list(v.values())
    for name, caps in capsets.items():
        rs = [R(c) for c in caps]
        rows1.append(dict(dataset=ds, source=name, n_captions=len(caps),
                          has_wn_noun=float(np.mean([bool(r['n']) for r in rs])), has_manual_object=float(np.mean([bool(r['mo']) for r in rs])),
                          has_wn_verb=float(np.mean([bool(r['v']) for r in rs])), has_manual_action=float(np.mean([bool(r['ma']) for r in rs])),
                          mean_wn_nouns=float(np.mean([len(r['n']) for r in rs])), mean_manual_objects=float(np.mean([len(r['mo']) for r in rs]))))
pd.DataFrame(rows1).to_csv(OUT / 'T1_coverage.csv', index=False)
print('T1 done', flush=True)

# ------------------------------------------------------------------ matched scores + T2 discrimination
rng = np.random.default_rng(0)
rows_match, rows2 = [], []
def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    gt = (pos[:, None] > neg[None, :]).mean(); eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)

for (ds, sy), preds in sources.items():
    ids = sorted(preds)
    if CHANGED_ONLY:
        isch = lambda c: any(e.change_state == 'change' for e in store.events(c, 'multi'))
        ids = [i for i in ids if isch(preds[i]) and all(isch(g) for g in refs_of[ds][i])]
    if len(ids) < 20:  # too few images for a shuffle test; recorded, not scored
        rows2.append(dict(dataset=ds, system=sy, n_images=len(ids), skipped='fewer than 20 images')); print('T2 skip', ds, sy, len(ids), flush=True); continue
    idset = ids
    perm = {i: idset[(k + 1 + rng.integers(0, len(idset) - 1)) % len(idset)] for k, i in enumerate(idset)}
    matched = {n: [] for n in VARIANTS}; shuffled = {n: [] for n in VARIANTS}
    for i in ids:
        m = image_score(preds[i], refs_of[ds][i])
        s = image_score(preds[i], refs_of[ds][perm[i]])
        for n in VARIANTS:
            matched[n].append(m[n]); shuffled[n].append(s[n])
    rec = dict(dataset=ds, system=sy, n_images=len(ids))
    for n in VARIANTS:
        mF = np.array([np.nan if x[2] is None else x[2] for x in matched[n]]); sF = np.array([np.nan if x[2] is None else x[2] for x in shuffled[n]])
        ok = np.isfinite(mF) & np.isfinite(sF)
        rec[f'{n}_P'] = mean_ignore([x[0] for x in matched[n]]); rec[f'{n}_R'] = mean_ignore([x[1] for x in matched[n]])
        rec[f'{n}_F_matched'] = float(np.nanmean(mF)); rec[f'{n}_F_shuffled'] = float(np.nanmean(sF))
        rec[f'{n}_gap'] = rec[f'{n}_F_matched'] - rec[f'{n}_F_shuffled']
        rec[f'{n}_auc'] = auc(mF[ok], sF[ok]) if ok.sum() > 1 else np.nan
        rec[f'{n}_n_scored'] = int(ok.sum())
    rows2.append(rec)
    print('T2', ds, sy, flush=True)
D2 = pd.DataFrame(rows2); D2.to_csv(OUT / 'T2_matched_vs_shuffled.csv', index=False)

# ------------------------------------------------------------------ T3 direction / T4 granularity
def best_man(tax_key, node, other):
    return msim(tax_key, node, other)
rows3 = []
for a, b, ma, mb in [('appear', 'disappear', 'appearance', 'disappearance'), ('construct', 'demolish', 'constructed', 'demolished'),
                     ('expand', 'contract', 'expansion', 'contraction'), ('appear', 'appear', 'appearance', 'appearance'),
                     ('construct', 'build', 'constructed', 'constructed'), ('demolish', 'destroy', 'demolished', 'demolished'),
                     ('appear', 'construct', 'appearance', 'constructed'), ('disappear', 'demolish', 'disappearance', 'demolished')]:
    rows3.append(dict(pair=f'{a} / {b}', manual_pair=f'{ma} / {mb}', wordnet_first=wsim(a, b, 'v', 'first'), wordnet_max=wsim(a, b, 'v', 'max'),
                      manual=msim('action', ma, mb)))
pd.DataFrame(rows3).to_csv(OUT / 'T3_action_pairs.csv', index=False)
rows4 = []
for a, b, ma, mb in [('building', 'house', 'building', 'residential_building'), ('house', 'factory', 'residential_building', 'industrial_building'),
                     ('building', 'road', 'building', 'road'), ('building', 'tree', 'building', 'tree'), ('tree', 'forest', 'tree', 'forest'),
                     ('tree', 'vegetation', 'tree', 'vegetation'), ('road', 'bridge', 'road', 'bridge'), ('river', 'lake', 'river', 'lake'),
                     ('building', 'water', 'building', 'water'), ('tree', 'road', 'tree', 'road'), ('soil', 'sand', 'soil', 'sand'),
                     ('building', 'building', 'building', 'building')]:
    rows4.append(dict(pair=f'{a} / {b}', manual_pair=f'{ma} / {mb}', wordnet_first=wsim(a, b, 'n', 'first'), wordnet_max=wsim(a, b, 'n', 'max'),
                      manual=msim('object', ma, mb)))
pd.DataFrame(rows4).to_csv(OUT / 'T4_object_pairs.csv', index=False)

# most frequent words WordNet takes as nouns/verbs in LEVIR references (what is being compared)
cn, cv = collections.Counter(), collections.Counter()
for g in (g for i in refs_of['LEVIR_CC'] for g in refs_of['LEVIR_CC'][i]):
    r = R(g); cn.update(r['n']); cv.update(r['v'])
json.dump({'top_nouns': cn.most_common(30), 'top_verbs': cv.most_common(30)}, open(OUT / 'levir_ref_top_words.json', 'w'), indent=1)
json.dump({'excluded_noun_words': sorted(EXCL_NOUN), 'excluded_verb_words': sorted(EXCL_VERB)}, open(OUT / 'word_lists.json', 'w'), indent=1)
(OUT / 'README.md').write_text('# WordNet vs manual hierarchy (v3)\n\n`python scripts/v3_wordnet_hierarchy.py` (env ricce; NLTK WordNet).\n\n' + __doc__ +
    '\n\nFiles: T1_coverage.csv, T2_matched_vs_shuffled.csv, T3_action_pairs.csv, T4_object_pairs.csv, levir_ref_top_words.json, word_lists.json.\n')
print('done')
