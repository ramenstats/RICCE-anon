"""Cross-dataset decomposition v2: parser conditional resolution, parse strata, OOV, style shift, score shift."""
import sys, json, re, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from ricce.evaluation.mapper_v2 import *

OUT = ROOT / 'results_cvpr_final/cross_dataset_v2'; OUT.mkdir(parents=True, exist_ok=True)
cfg = load_cfg(); taxes = load_taxes(cfg); parser = build_parser(taxes, cfg)
SYS = ['sft_seed42', 'chg2cap', 'swmcc', 'qwen']
STOP = set('a an the of in on at to and is are was were be been by with from for as its it this that there near some'.split())
rng = np.random.default_rng(0)
def boot(x, B=1000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if not len(x): return None
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]; return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

def stratum(ev, mode):
    if ev.change_state == 'no_change': return 'no_change'
    core = int(ev.object is not None) + int(ev.action is not None)
    if mode == 'core':  return {2: 'full', 1: 'partial', 0: 'failed'}[core]
    n = core + int(ev.location is not None) + int(ev.quantity is not None)
    return 'full' if n == 4 else ('failed' if n == 0 else 'partial')

manifests = {ds: [r for r in read_jsonl(ROOT / f'artifacts/manifests/{ds}.jsonl')] for ds in DATASETS}
tokz = lambda t: norm(t).split()
levir_train_vocab = collections.Counter(w for r in manifests['LEVIR_CC'] if r['split'] == 'train' for c in r['references'] for w in tokz(c))
alias_vocab = set(w for t in taxes.values() for _, ph in t.labels_and_aliases() for w in tokz(ph))
res = {'per_dataset': {}, 'score_shift': {}, 'strata': {}}

ref_parse, pred_parse, clean_ref, clean_ref_change, ref_nc = {}, {}, {}, {}, {}
for ds in DATASETS:
    test = [r for r in manifests[ds] if r['split'] == 'test']
    refs = [(r['image_id'], c) for r in test for c in r['references']]
    evs = {}
    for iid, c in refs: evs.setdefault(iid, []).append(parser.parse(c)[0])
    ref_parse[ds] = evs
    clean_ref[ds] = {iid: all(stratum(e, 'core') in ('full', 'no_change') for e in es) for iid, es in evs.items()}
    clean_ref_change[ds] = {iid: all(stratum(e, 'core') == 'full' for e in es) for iid, es in evs.items()}
    ref_nc[ds] = {iid: float(np.mean([e.change_state == 'no_change' for e in es])) for iid, es in evs.items()}
    allev = [e for es in evs.values() for e in es]; ch = [e for e in allev if e.change_state != 'no_change']
    toks = [w for _, c in refs for w in tokz(c)]; types = set(toks); content = [w for w in toks if w not in STOP]
    multi = [len(parser.parse(c, allow_multi_event=True)) for _, c in refs]
    d = {'n_test_images': len(test), 'n_reference_captions': len(refs), 'n_change_refs': len(ch),
         'unconditional_axis_coverage_on_change_refs': {ax: float(np.mean([getattr(e, ax) is not None for e in ch])) for ax in ['object', 'action', 'source', 'target', 'location', 'quantity']},
         'ref_core_stratum_share': {k: float(np.mean([stratum(e, 'core') == k for e in allev])) for k in ['full', 'partial', 'failed', 'no_change']},
         'ref_all4_stratum_share': {k: float(np.mean([stratum(e, 'all4') == k for e in allev])) for k in ['full', 'partial', 'failed', 'no_change']},
         'vocab_oov_vs_LEVIR_train': {'token_rate': float(np.mean([w not in levir_train_vocab for w in toks])), 'type_rate': float(np.mean([w not in levir_train_vocab for w in types])),
                                      'content_token_rate': float(np.mean([w not in levir_train_vocab for w in content])), 'n_types': len(types)},
         'vocab_oov_vs_taxonomy_alias_tokens': {'token_rate': float(np.mean([w not in alias_vocab for w in toks])), 'content_token_rate': float(np.mean([w not in alias_vocab for w in content]))},
         'style': {'mean_caption_tokens': float(np.mean([len(tokz(c)) for _, c in refs])), 'mean_event_count_multi_event_split': float(np.mean(multi)), 'frac_multi_event_refs': float(np.mean(np.array(multi) > 1)),
                   'no_change_ref_fraction': float(np.mean([e.change_state == 'no_change' for e in allev])), 'frac_images_all_refs_core_parsed': float(np.mean(list(clean_ref[ds].values())))}}
    res['per_dataset'][ds] = d
    print(ds, 'refs parsed', len(refs), flush=True)

# conditional resolution (given human-labelled field present) from mapper_v2 per_axis.csv (n=160/dataset)
pa = pd.read_csv(ROOT / 'results_cvpr_final/mapper_v2/per_axis.csv')
for ds in DATASETS:
    res['per_dataset'][ds]['conditional_resolution_given_present_human_n160'] = {r.axis: {'P_resolved': r.P_resolved_given_present, 'n_present': int(r.n_present)} for r in pa[pa.dataset == ds].itertuples()}

rows = []
for ds in DATASETS:
    test_ids = {r['image_id'] for r in manifests[ds] if r['split'] == 'test'}
    for s in SYS:
        e = pd.read_csv(ROOT / f'results/E08/{ds}/{s}/per_sample_metrics.csv'); e = e[e.image_id.isin(test_ids)].drop_duplicates('image_id')
        for r in e.itertuples():
            pe = parser.parse(str(r.prediction))[0]
            rows.append(dict(dataset=ds, system=s, image_id=r.image_id, ce_f1=r.ce_f1, rougeL=r.rougeL, meteor=r.meteor, bleu4=r.bleu4,
                             pred_core=stratum(pe, 'core'), pred_all4=stratum(pe, 'all4'), ref_clean=clean_ref[ds].get(r.image_id, False), ref_change_clean=clean_ref_change[ds].get(r.image_id, False), ref_nc_frac=ref_nc[ds].get(r.image_id, np.nan), pred_tokens=len(tokz(str(r.prediction)))))
        print(ds, s, 'scored', flush=True)
P = pd.DataFrame(rows); P.to_csv(OUT / 'per_image_systems.csv', index=False)

for ds in DATASETS:
    for s in SYS:
        g = P[(P.dataset == ds) & (P.system == s)]
        for mode in ['pred_core', 'pred_all4']:
            res['strata'].setdefault(ds, {}).setdefault(s, {})[mode] = {k: {'n': int((g[mode] == k).sum()), 'share': float((g[mode] == k).mean()), 'mean_ce_f1': float(g[g[mode] == k].ce_f1.mean()) if (g[mode] == k).any() else None} for k in ['full', 'partial', 'failed', 'no_change']}
        res['strata'][ds][s]['ref_side_clean_vs_not'] = {str(k): {'n': int((g.ref_clean == k).sum()), 'mean_ce_f1': float(g[g.ref_clean == k].ce_f1.mean()) if (g.ref_clean == k).any() else None} for k in [True, False]}
        both = g[g.ref_clean & g.pred_core.isin(['full', 'no_change'])]
        bc = g[g.ref_change_clean & (g.pred_core == 'full')]
        res['strata'][ds][s]['parse_clean_change_only_both_sides'] = {'n': len(bc), 'mean_ce_f1': float(bc.ce_f1.mean()) if len(bc) else None, 'note': 'all 5 refs resolve object+action (no no_change refs) and prediction resolves object+action'}
        res['strata'][ds][s]['parse_clean_both_sides'] = {'n': len(both), 'mean_ce_f1': float(both.ce_f1.mean()) if len(both) else None}
for s in SYS:
    lv = P[(P.dataset == 'LEVIR_CC') & (P.system == s)]
    lvc = lv[lv.ref_clean & lv.pred_core.isin(['full', 'no_change'])]; lvb = lv[lv.ref_change_clean & (lv.pred_core == 'full')]
    res['score_shift'][s] = {'LEVIR_CC': {'ce_f1': float(lv.ce_f1.mean()), 'ce_f1_CI95': boot(lv.ce_f1), 'rougeL': float(lv.rougeL.mean()), 'meteor': float(lv.meteor.mean()), 'n': len(lv), 'ce_f1_parse_clean': float(lvc.ce_f1.mean()), 'n_parse_clean': len(lvc), 'ce_f1_change_clean': float(lvb.ce_f1.mean()), 'n_change_clean': len(lvb), 'pred_no_change_share': float((lv.pred_core == 'no_change').mean()), 'mean_ref_no_change_frac': float(lv.ref_nc_frac.mean())}}
    for ds in ['SECOND_CC', 'DUBAI_CC']:
        g = P[(P.dataset == ds) & (P.system == s)]; gc = g[g.ref_clean & g.pred_core.isin(['full', 'no_change'])]; gb = g[g.ref_change_clean & (g.pred_core == 'full')]
        res['score_shift'][s][ds] = {'ce_f1': float(g.ce_f1.mean()), 'ce_f1_CI95': boot(g.ce_f1), 'rougeL': float(g.rougeL.mean()), 'meteor': float(g.meteor.mean()), 'n': len(g),
            'delta_ce_f1_vs_LEVIR': float(g.ce_f1.mean() - lv.ce_f1.mean()), 'delta_rougeL_vs_LEVIR': float(g.rougeL.mean() - lv.rougeL.mean()), 'delta_meteor_vs_LEVIR': float(g.meteor.mean() - lv.meteor.mean()),
            'ce_f1_parse_clean': float(gc.ce_f1.mean()) if len(gc) else None, 'n_parse_clean': len(gc), 'delta_ce_f1_parse_clean_vs_LEVIR_parse_clean': float(gc.ce_f1.mean() - lvc.ce_f1.mean()) if len(gc) and len(lvc) else None,
            'ce_f1_change_clean': float(gb.ce_f1.mean()) if len(gb) else None, 'n_change_clean': len(gb), 'delta_ce_f1_change_clean_vs_LEVIR': float(gb.ce_f1.mean() - lvb.ce_f1.mean()) if len(gb) and len(lvb) else None, 'pred_no_change_share': float((g.pred_core == 'no_change').mean()), 'mean_ref_no_change_frac': float(g.ref_nc_frac.mean()), 'rel_drop_ce_f1': float(1 - g.ce_f1.mean() / lv.ce_f1.mean()), 'rel_drop_rougeL': float(1 - g.rougeL.mean() / lv.rougeL.mean())}
res['conclusion_notes'] = [
 'CONTROLLED: parser coverage on the prediction and reference side (parse-clean subset restricts to images whose references and prediction all resolve object+action or are no_change); parser-independent metrics (ROUGE-L, METEOR) reported alongside CE-F for the same images.',
 'NOT CONTROLLED: reference style/ontology-granularity differences (caption length, event count, no-change share, vocabulary) between datasets; image content difficulty; number of test items (Dubai n=150); the parse-clean subset is itself selected on parser success and thus biased to canonical phrasings; zero-shot vs in-domain training of each system (SFT/Chg2Cap/SWMCC LEVIR-trained transfer, Qwen zero-shot) is not disentangled.',
 'A CE-F drop that persists inside the parse-clean subset and co-occurs with drops in ROUGE-L/METEOR is consistent with (but does not prove) model/domain shift; a drop that vanishes only in the parse-clean subset would indicate parser/ontology shift. No causal decomposition into shares is claimed.']
json.dump(res, open(OUT / 'summary.json', 'w'), indent=1)
json.dump({'parser': cfg['parser'], 'systems': SYS, 'splits': 'test only for scores/refs; vocab baseline = LEVIR-CC train refs', 'stop_words_for_content_tokens': sorted(STOP)}, open(OUT / 'config.json', 'w'), indent=1)
print(json.dumps({s: res['score_shift'][s] for s in SYS}, indent=1)[:3000])
