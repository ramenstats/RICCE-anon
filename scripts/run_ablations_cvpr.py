"""Task II: ablations on RICCE-Bench + real systems (LEVIR_CC test, mean over refs).
Outputs -> results_cvpr_final/ablations/. See README.md there."""
import sys, json, time, glob, os, itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.utils import read_jsonl
from ricce.schema import Event
from ricce.benchmarks.env import load_taxes, load_weights
from ricce.evaluation.ordering_report import load_items
from ricce.evaluation.ordering import build_constraints, pairwise_ordering_accuracy, flatten
from ricce.evaluation.parse_store import ParseStore
from ricce.baselines.ablations import build_variants, score_events
from ricce.metrics.event import ricce_scores

OUT = ROOT / 'results_cvpr_final' / 'ablations'
OUT.mkdir(parents=True, exist_ok=True)
ITEMS = ROOT / 'artifacts_cvpr_final' / 'riccebench' / 'items.jsonl'
SYSTEMS = ['qwen', 'sft_seed42', 'chg2cap', 'swmcc']
TAXES, W = load_taxes(), load_weights()
G = {}


def load_system_data():
    refs = {r['image_id']: r['references'] for r in read_jsonl(ROOT / 'artifacts' / 'manifests' / 'LEVIR_CC.jsonl')}
    data = {}
    for s in SYSTEMS:
        df = pd.read_csv(ROOT / 'results' / 'E08' / 'LEVIR_CC' / s / 'per_sample_metrics.csv', keep_default_na=False)
        df['prediction'] = df['prediction'].astype(str)
        data[s] = [(r.image_id, r.prediction, refs[r.image_id]) for r in df.itertuples()]
    return data


def run_variant(name):
    v = G['variants'][name]
    store, items, sysd = G['store'], G['items'], G['sys']
    mode = 'multi' if v.parse_multi else 'single'
    ev = lambda t: store.events(t, mode)
    t0 = time.time()
    n = len(items)
    P, R, F = np.zeros(n, np.float32), np.zeros(n, np.float32), np.zeros(n, np.float32)
    for i, (c, r) in enumerate(zip(items['candidate'], items['reference'])):
        s = score_events(v, ev(c), ev(r), TAXES, W)
        P[i], R[i], F[i] = s['ce_precision'], s['ce_recall'], s['ce_f1']
    sysres = {}
    for sname, rows in sysd.items():
        acc = []
        for iid, pred, refs in rows:
            pe = ev(pred)
            ss = [score_events(v, pe, ev(g), TAXES, W) for g in refs]
            acc.append([np.mean([x[k] for x in ss]) for k in ('ce_precision', 'ce_recall', 'ce_f1')])
        sysres[sname] = np.mean(acc, 0).tolist()
    return name, P, R, F, sysres, time.time() - t0


def main():
    t0 = time.time()
    items = load_items(ITEMS)
    sysd = load_system_data()
    store = ParseStore(ROOT / 'results_cvpr_final' / 'ablations' / 'parse_cache.json')
    for r in items.itertuples():
        store.seed('multi', r.reference, r.source_event)
        store.seed('multi', r.candidate, r.candidate_event)
    texts = set(items['candidate']) | set(items['reference'])
    for rows in sysd.values():
        for _, p, refs in rows:
            texts |= {p, *refs}
    print('texts', len(texts), flush=True)
    for mode in ('multi', 'single'):
        print('parsing', mode, store.ensure(texts, mode), flush=True)
    store.save()
    variants = build_variants(TAXES)
    G.update(variants=variants, store=store, items=items, sys=sysd)
    import multiprocessing as mp
    names = list(variants)
    with mp.get_context('fork').Pool(min(40, len(names))) as pool:
        res = {}
        for name, P, R, F, sr, dt in pool.imap_unordered(run_variant, names):
            res[name] = (P, R, F, sr)
            print('variant', name, f'{dt:.0f}s', flush=True)
    # ---------------- tables
    fam_rows, item_scores = [], {'sample_id': items['sample_id'].values}
    for nme in names:
        P, R, F, _ = res[nme]
        item_scores[f'{nme}|F'] = F
        if nme.startswith('A'):
            item_scores[f'{nme}|P'], item_scores[f'{nme}|R'] = P, R
        d = items.assign(P=P, R=R, F=F)
        for ds in ['ALL'] + sorted(items['dataset'].unique()):
            dd = d if ds == 'ALL' else d[d['dataset'] == ds]
            g = dd.groupby('perturbation_type')[['P', 'R', 'F']].mean()
            for fam, r in g.iterrows():
                fam_rows.append(dict(variant=nme, dataset=ds, family=fam, ce_p=r.P, ce_r=r.R, ce_f=r.F, n=int((dd['perturbation_type'] == fam).sum())))
    pd.DataFrame(fam_rows).to_csv(OUT / 'family_means_by_variant.csv', index=False)
    pd.DataFrame(item_scores).to_csv(OUT / 'per_item_scores.csv.gz', index=False)
    cons = build_constraints(items)
    core = cons[cons['suite'] == 'core']
    a0 = res['A0_full']
    summ = []
    ref_rank = {k: [a0[3][s][i] for s in SYSTEMS] for i, k in enumerate(('P', 'R', 'F'))}
    sysrows = []
    for nme in names:
        P, R, F, sr = res[nme]
        row = dict(variant=nme, group=variants[nme].family, desc=variants[nme].desc, mean_ce_f_all_items=float(F.mean()),
                   mean_abs_item_shift_vs_A0=float(np.abs(F - a0[2]).mean()))
        for ds in ['ALL'] + sorted(items['dataset'].unique()):
            ci = core if ds == 'ALL' else core[core['dataset'] == ds]
            r = pairwise_ordering_accuracy(pd.Series(F, index=items['sample_id']), ci, n_boot=500, seed=0)
            row[f'ord_strict_{ds}'] = r['accuracy_strict']
            row[f'ord_strict_{ds}_ci_lo'], row[f'ord_strict_{ds}_ci_hi'] = r['accuracy_strict_ci95']
            row[f'ord_tiehalf_{ds}'] = r['accuracy_tiehalf']
            row[f'tie_rate_{ds}'] = r['tie_rate']
        for i, k in enumerate(('P', 'R', 'F')):
            vals = [sr[s][i] for s in SYSTEMS]
            row[f'sys_tau_vs_A0_{k}'] = float(kendalltau(vals, ref_rank[k])[0])
            for s, x in zip(SYSTEMS, vals):
                sysrows.append(dict(variant=nme, system=s, metric=f'CE-{k}', score=x))
        row['sys_rank_F'] = ' > '.join(sorted(SYSTEMS, key=lambda s: -sr[s][2]))
        summ.append(row)
    S = pd.DataFrame(summ)
    S.to_csv(OUT / 'ablation_summary.csv', index=False)
    pd.DataFrame(sysrows).to_csv(OUT / 'system_scores_by_variant.csv', index=False)
    # ---------------- A6 matching study
    a6 = matching_study(store, items, sysd)
    json.dump(a6, open(OUT / 'a6_hungarian_vs_greedy.json', 'w'), indent=1)
    # depth robustness table
    dep = S[S['variant'].isin(['A0_full', 'DEPTH_collapse1', 'DEPTH_collapse2'])]
    dep.to_csv(OUT / 'depth_robustness.csv', index=False)
    fm = pd.DataFrame(fam_rows)
    dm = fm[(fm.dataset == 'ALL') & fm.variant.isin(['A0_full', 'DEPTH_collapse1', 'DEPTH_collapse2'])].pivot(index='family', columns='variant', values='ce_f')
    dm.to_csv(OUT / 'depth_family_score_movement.csv')
    json.dump(dict(n_items=int(len(items)), n_variants=len(names), seconds=time.time() - t0, systems=SYSTEMS,
                   headline=S.set_index('variant')[['ord_strict_ALL', 'ord_tiehalf_ALL', 'sys_tau_vs_A0_F', 'sys_tau_vs_A0_P', 'sys_tau_vs_A0_R', 'mean_ce_f_all_items']].to_dict('index'),
                   a6=a6), open(OUT / 'summary.json', 'w'), indent=1, default=float)
    (OUT / 'command.txt').write_text('python scripts/run_ablations_cvpr.py\n')
    json.dump(dict(items=str(ITEMS.relative_to(ROOT)), systems=SYSTEMS, weights=W, variants={k: v.desc for k, v in variants.items()},
                   system_protocol='LEVIR_CC test, per-image mean over 5 references, multi-event parse (except A2)'), open(OUT / 'config.json', 'w'), indent=1)
    print(S[['variant', 'ord_strict_ALL', 'ord_tiehalf_ALL', 'sys_tau_vs_A0_F']].to_string())


def matching_study(store, items, sysd):
    import timeit
    res = {}
    # synthetic
    syn = [json.loads(l) for l in open(ROOT / 'artifacts_r1' / 'multi_event_benchmark.jsonl')]
    syn = [r for r in syn if r['source'] == 'synthetic_stress_test']
    E_ = lambda L: [Event(**{k: v for k, v in e.items() if k in Event.__dataclass_fields__}) for e in L]
    def study(pairs, label):
        h = [ricce_scores(p, g, TAXES, W, 'hungarian')['ce_f1'] for p, g in pairs]
        g_ = [ricce_scores(p, g, TAXES, W, 'greedy')['ce_f1'] for p, g in pairs]
        o = [ricce_scores(p, g, TAXES, W, 'order')['ce_f1'] for p, g in pairs]
        h, g_, o = map(np.array, (h, g_, o))
        th = timeit.timeit(lambda: [ricce_scores(p, g, TAXES, W, 'hungarian') for p, g in pairs], number=3) / 3
        tg = timeit.timeit(lambda: [ricce_scores(p, g, TAXES, W, 'greedy') for p, g in pairs], number=3) / 3
        return dict(n_pairs=len(pairs), n_hung_ne_greedy=int((np.abs(h - g_) > 1e-9).sum()), n_hung_ne_order=int((np.abs(h - o) > 1e-9).sum()),
                    frac_hung_ne_greedy=float((np.abs(h - g_) > 1e-9).mean()), mean_abs_diff_greedy=float(np.abs(h - g_).mean()),
                    max_abs_diff_greedy=float(np.abs(h - g_).max()), mean_ce_f_hungarian=float(h.mean()), mean_ce_f_greedy=float(g_.mean()),
                    mean_ce_f_order=float(o.mean()), n_hung_gt_greedy=int((h - g_ > 1e-9).sum()), n_greedy_gt_hung=int((g_ - h > 1e-9).sum()),
                    runtime_hungarian_s=th, runtime_greedy_s=tg, avg_events_pred=float(np.mean([len(p) for p, _ in pairs])), avg_events_gt=float(np.mean([len(g) for _, g in pairs])))
    res['synthetic_stress_440'] = study([(E_(r['pred_events']), E_(r['gt_events'])) for r in syn], 'syn')
    # real multi-event: RICCE-Bench multi-event families (ref has >=2 events) + system predictions vs multi-event refs
    real = [(store.events(r.candidate, 'multi'), store.events(r.reference, 'multi')) for r in items.itertuples() if len(r.source_event) >= 2]
    res['real_multi_event_benchmark_items'] = study(real, 'bench')
    sysreal = []
    refs_all = {r['image_id']: r['references'] for r in read_jsonl(ROOT / 'artifacts' / 'manifests' / 'LEVIR_CC.jsonl')}
    for s, rows in sysd.items():
        for iid, pred, refs in rows:
            pe = store.events(pred, 'multi')
            for g in refs:
                ge = store.events(g, 'multi')
                if len(ge) >= 2 or len(pe) >= 2:
                    sysreal.append((pe, ge))
    res['real_system_pred_vs_ref_any_side_multi_event'] = study(sysreal, 'sys')
    return res


if __name__ == '__main__':
    main()
