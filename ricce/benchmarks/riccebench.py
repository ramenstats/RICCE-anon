"""RICCE-Bench builder: controlled semantic minimal pairs from real reference captions.
Usage: python -m ricce.benchmarks.riccebench [--n-change 1000 --n-nochange 250 --n-multi 400 --seed 20240611]
"""
import argparse, json, random, collections, sys, time
from pathlib import Path
from ..utils import read_jsonl
from .env import ROOT, load_taxes, make_parser, CachedParser
from .perturbations import FAMILIES, FAMILY_FN, SEVERITY
from .purity import check_purity

DATASETS = ['LEVIR_CC', 'SECOND_CC', 'DUBAI_CC']
ART = ROOT / 'artifacts_cvpr_final' / 'riccebench'


def load_captions(ds):
    seen, out = set(), []
    for r in read_jsonl(ROOT / 'artifacts' / 'manifests' / f'{ds}.jsonl'):
        for i, c in enumerate(r['references']):
            key = ' '.join(c.lower().split())
            if not key or key in seen:
                continue
            seen.add(key)
            out.append((str(r['image_id']), i, c))
    return out


def sample_sources(ds, parser, n_change, n_nochange, n_multi, seed):
    caps = load_captions(ds)
    random.Random(f'{seed}|{ds}|sources').shuffle(caps)
    quota = {'change': n_change, 'nochange': n_nochange, 'multi': n_multi, 'other': 10 ** 9}
    got = collections.defaultdict(list)
    for iid, i, c in caps:
        if all(len(got[k]) >= v for k, v in quota.items() if k != 'other') and sum(len(got[k]) for k in ('change', 'nochange', 'multi')) >= n_change:
            break
        ev = parser.parse(c, allow_multi_event=True)
        if len(ev) >= 2:
            k = 'multi'
        elif ev and ev[0].change_state is None:
            k = 'other'
        elif ev and ev[0].change_state == 'no_change':
            k = 'nochange'
        elif ev and ev[0].change_state == 'change':
            k = 'change'
        else:
            continue
        if len(got[k]) < quota[k]:
            got[k].append((iid, i, c, k))
    main = [x for k in ('change', 'nochange', 'multi') for x in got[k]]
    # 'other' = real captions whose parse resolves no change state (unresolved); used ONLY to top up to >= n_change sources
    need = max(0, n_change - len(main))
    return main + got['other'][:need], len(caps)


def build_dataset(ds, parser, taxes, seed, n_change, n_nochange, n_multi):
    sources, n_unique = sample_sources(ds, parser, n_change, n_nochange, n_multi, seed)
    items = []
    for iid, ri, ref, stratum in sources:
        sid = f'{ds}:{iid}:{ri}'
        ref_ev = parser.parse(ref, allow_multi_event=True)
        ctx = dict(taxes=taxes, parser=parser, ref_events=ref_ev)
        base = dict(source_id=sid, dataset=ds, reference=ref, source_stratum=stratum,
                    source_event=[e.to_dict() for e in ref_ev])
        items.append(dict(base, sample_id=f'{sid}:P0_identity', candidate=ref, perturbation_type='P0_identity',
                          severity_level=SEVERITY['P0_identity'], changed_axis='none', expected_relation='identical',
                          candidate_event=[e.to_dict() for e in ref_ev],
                          purity_checks=check_purity('P0_identity', ref_ev, ref_ev, {}, taxes)))
        for fam in FAMILIES[1:]:
            rng = random.Random(f'{seed}|{sid}|{fam}')
            res = FAMILY_FN[fam](ref, ctx, rng)
            if not res or ' '.join(res['candidate'].lower().split()) == ' '.join(ref.lower().split()):
                continue
            cand_ev = parser.parse(res['candidate'], allow_multi_event=True)
            pc = check_purity(fam, ref_ev, cand_ev, res.get('meta'), taxes)
            items.append(dict(base, sample_id=f'{sid}:{fam}', candidate=res['candidate'], perturbation_type=fam,
                              severity_level=SEVERITY[fam], changed_axis=res['changed_axis'],
                              expected_relation=res['expected_relation'], perturbation_meta=res.get('meta', {}),
                              candidate_event=[e.to_dict() for e in cand_ev], purity_checks=pc))
    return items, len(sources), n_unique


def _fam_stats(allitems, datasets):
    fam_stats = {}
    for fam in FAMILIES:
        for ds in list(datasets) + ['ALL']:
            sel = [x for x in allitems if x['perturbation_type'] == fam and (ds == 'ALL' or x['dataset'] == ds)]
            n = len(sel)
            npure = sum(x['purity_checks']['pure'] for x in sel)
            fam_stats.setdefault(fam, {})[ds] = dict(n_generated=n, n_pure=npure, purity_rate=(npure / n if n else None))
    return fam_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=20240611)
    ap.add_argument('--n-change', type=int, default=1000)
    ap.add_argument('--n-nochange', type=int, default=250)
    ap.add_argument('--n-multi', type=int, default=400)
    ap.add_argument('--datasets', nargs='+', default=DATASETS)
    ap.add_argument('--merge', action='store_true', help='merge per-dataset parts into items.jsonl/impure.jsonl')
    a = ap.parse_args()
    parts = ART / 'parts'
    parts.mkdir(parents=True, exist_ok=True)
    if a.merge:
        allitems, stats = [], {}
        for ds in DATASETS:
            allitems += [json.loads(l) for l in open(parts / f'{ds}.jsonl')]
            stats[ds] = json.load(open(parts / f'{ds}.stats.json'))
        pure = [x for x in allitems if x['purity_checks']['pure']]
        impure = [x for x in allitems if not x['purity_checks']['pure']]
        with open(ART / 'items.jsonl', 'w') as f:
            for x in pure:
                f.write(json.dumps(x) + '\n')
        with open(ART / 'impure.jsonl', 'w') as f:
            for x in impure:
                f.write(json.dumps(x) + '\n')
        json.dump(dict(seed=a.seed, args=vars(a), datasets=stats, n_pure=len(pure), n_impure=len(impure),
                       purity_by_family=_fam_stats(allitems, DATASETS)), open(ART / 'build_stats.json', 'w'), indent=1)
        print('merged', len(pure), len(impure))
        return
    taxes = load_taxes()
    parser = CachedParser(make_parser(taxes))
    for ds in a.datasets:
        t = time.time()
        items, nsrc, nuniq = build_dataset(ds, parser, taxes, a.seed, a.n_change, a.n_nochange, a.n_multi)
        with open(parts / f'{ds}.jsonl', 'w') as f:
            for x in items:
                f.write(json.dumps(x) + '\n')
        json.dump(dict(n_sources=nsrc, n_unique_reference_captions=nuniq, n_items=len(items), seconds=time.time() - t),
                  open(parts / f'{ds}.stats.json', 'w'))
        print(ds, nsrc, nuniq, len(items), f'{time.time()-t:.0f}s', flush=True)


if __name__ == '__main__':
    main()
