"""Build results/metric_comparison/sensitivity_table.json: for each of
BLEU-4, CIDEr-D (official, corpus), BERTScore, SPICE (if available),
LLM-Judge (if available) and RICCE-F, check whether the metric actually
moves the way a semantically-aware metric should under four controlled
perturbations already generated for E04/E05/E06 (paraphrase, action-reversal,
hallucination) and E10 (T1<->T2 swap) -- rather than assuming it does.

Source data (all pre-existing, produced by earlier experiments; this script
does not regenerate perturbations, it only scores the already-perturbed
(gt, prediction) / (normal_order, swapped_order) pairs with metrics that had
not yet been applied to them):
  - results/E04/LEVIR_CC/per_sample_metrics.csv  (gt, prediction, type, ce_f1, bleu4, ...)
  - results/E10/LEVIR_CC/raw_predictions.jsonl + per_sample_swap.jsonl (normal_order, swapped_order, category)

Definitions (documented here, not just implied by the numbers):
  - "invariant" to paraphrase: ratio = mean_score(P0_paraphrase) / mean_score(P0_original) is HIGH
    (metric correctly does not punish a same-meaning reword).
  - "sensitive" to action-reversal / hallucination: ratio = mean_score(P_x) / mean_score(P0_original)
    is LOW (metric correctly punishes a semantically-broken but lexically-similar caption).
  - "sensitive" to T1/T2 swap: normal_order-vs-swapped_order similarity is markedly LOWER for the
    'directional' category (captions whose correct meaning genuinely flips under a T1/T2 swap,
    e.g. appearance<->disappearance) than for the 'no_change' category (where nothing should
    flip) -- i.e. the metric notices that only the directional captions' content should differ
    after the swap.
Thresholds used to turn a ratio/gap into a boolean are explicit constants below (not tuned to
produce a particular headline finding) and the raw numbers are kept in the output so a reviewer
can re-derive the calls with a different threshold.
"""
import sys, json, math
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ricce.metrics.conventional import cider_d_corpus, bertscore as bertscore_single
from ricce.metrics.spice_llm_judge import spice_available, spice_corpus, llm_judge_available, llm_judge_batch, sample_indices

PARAPHRASE_INVARIANT_RATIO_THRESH = 0.85   # ratio >= this => called invariant
REVERSAL_SENSITIVE_RATIO_THRESH = 0.75     # ratio <= this => called sensitive
SWAP_SENSITIVE_GAP_THRESH = 0.05           # (no_change_sim - directional_sim) >= this => called sensitive
SUBSAMPLE_N = 120
SEED = 42

OUT_DIR = ROOT / 'results' / 'metric_comparison'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def bertscore_corpus(preds, refs_list):
    from bert_score import BERTScorer
    sc = BERTScorer(lang='en', rescale_with_baseline=False, device='cuda')
    try:
        _, _, F = sc.score(preds, refs_list, batch_size=64)
    except Exception:
        sc = BERTScorer(lang='en', rescale_with_baseline=False, device='cpu')
        _, _, F = sc.score(preds, refs_list, batch_size=32)
    return [float(x) for x in F]


def score_group(tag, ids, gts, preds, spice_ok, judge_ok, subsample_n=SUBSAMPLE_N):
    """gts/preds: parallel lists of single strings (one reference each -- these
    perturbation pairs are 1:1, not multi-reference)."""
    n = len(ids)
    out = {'n': n}
    # BLEU-4: reuse the already-computed nltk sentence_bleu4 (recomputed here
    # fresh, not copy-pasted from the CSV, so this script is self-contained).
    from ricce.metrics.conventional import sentence_bleu_scores
    b4 = [sentence_bleu_scores(p, [g])['bleu4'] for p, g in zip(preds, gts)]
    out['bleu4_mean'] = float(sum(b4) / len(b4))

    uids = [f'{tag}_{i}' for i in ids]
    cider_mean, _ = cider_d_corpus(uids, preds, {u: [g] for u, g in zip(uids, gts)})
    out['cider_d_mean'] = cider_mean

    bs = bertscore_corpus(preds, [[g] for g in gts])
    out['bertscore_f_mean'] = float(sum(bs) / len(bs))

    idx = sample_indices(n, min(subsample_n, n), seed=SEED)
    sub_uids = [uids[i] for i in idx]
    sub_preds = [preds[i] for i in idx]
    sub_gts = [gts[i] for i in idx]
    out['subsample_n'] = len(idx)

    if spice_ok:
        sp_mean, _, sp_reason = spice_corpus(sub_uids, sub_preds, {u: [g] for u, g in zip(sub_uids, sub_gts)})
        out['spice_f_mean'] = sp_mean
        out['spice_reason'] = sp_reason
    else:
        out['spice_f_mean'] = None

    if judge_ok:
        scores, j_reason = llm_judge_batch(sub_preds, [[g] for g in sub_gts], batch_size=16)
        valid = [s for s in scores if s == s]
        out['llm_judge_mean_1to5'] = float(sum(valid) / len(valid)) if valid else None
        out['llm_judge_n_parsed'] = len(valid)
    else:
        out['llm_judge_mean_1to5'] = None

    return out


def main():
    spice_ok, spice_reason = spice_available()
    judge_ok, judge_reason = llm_judge_available()
    print(f'SPICE available: {spice_ok} ({spice_reason})')
    print(f'LLM-judge available: {judge_ok} ({judge_reason})')

    df = pd.read_csv(ROOT / 'results' / 'E04' / 'LEVIR_CC' / 'per_sample_metrics.csv')
    groups = {}
    ce_f1 = {}
    type_map = {
        'P0_original': 'original',
        'P0_paraphrase': 'paraphrase',
        'P4_action_reversal': 'action_reversal',
        'P8_hallucination': 'hallucination',
    }
    for ptype, key in type_map.items():
        sub = df[df['type'] == ptype]
        ids = sub['image_id'].tolist()
        gts = sub['gt'].astype(str).tolist()
        preds = sub['prediction'].astype(str).tolist()
        print(f'[E04 {ptype}] n={len(ids)} scoring ...')
        groups[key] = score_group(key, ids, gts, preds, spice_ok, judge_ok)
        ce_f1[key] = float(sub['ce_f1'].mean())
        groups[key]['ricce_f_mean'] = ce_f1[key]

    # T1/T2-swap sensitivity from E10.
    preds_map = {}
    with open(ROOT / 'results' / 'E10' / 'LEVIR_CC' / 'raw_predictions.jsonl') as f:
        for line in f:
            r = json.loads(line)
            preds_map[r['image_id']] = r
    swap_rows = []
    with open(ROOT / 'results' / 'E10' / 'LEVIR_CC' / 'per_sample_swap.jsonl') as f:
        for line in f:
            swap_rows.append(json.loads(line))

    swap_groups = {}
    for cat in ('directional', 'no_change'):
        rows = [r for r in swap_rows if r['category'] == cat and r['image_id'] in preds_map]
        ids = [r['image_id'] for r in rows]
        normal = [preds_map[i]['normal_order'] for i in ids]
        swapped = [preds_map[i]['swapped_order'] for i in ids]
        print(f'[E10 {cat}] n={len(ids)} scoring normal_order-vs-swapped_order similarity ...')
        # normal_order treated as the reference, swapped_order as the candidate:
        # a metric "notices" the swap if this cross-similarity drops for the
        # directional category (whose correct caption content should flip)
        # but stays high for the no_change category (should not flip).
        swap_groups[cat] = score_group(f'e10_{cat}', ids, normal, swapped, spice_ok, judge_ok, subsample_n=SUBSAMPLE_N)

    # ---- assemble calls -------------------------------------------------
    def ratio(key, base='original'):
        return {m: (groups[key].get(m) / groups[base][m])
                 if isinstance(groups[key].get(m), (int, float)) and isinstance(groups[base].get(m), (int, float)) and groups[base].get(m)
                 else None
                for m in ('bleu4_mean', 'cider_d_mean', 'bertscore_f_mean', 'spice_f_mean', 'llm_judge_mean_1to5', 'ricce_f_mean')}

    paraphrase_ratio = ratio('paraphrase')
    action_reversal_ratio = ratio('action_reversal')
    hallucination_ratio = ratio('hallucination')

    metric_names = {
        'bleu4_mean': 'BLEU-4',
        'cider_d_mean': 'CIDEr-D',
        'bertscore_f_mean': 'BERTScore',
        'spice_f_mean': 'SPICE',
        'llm_judge_mean_1to5': 'LLM-Judge',
        'ricce_f_mean': 'RICCE-F',
    }

    table = {}
    for mkey, mname in metric_names.items():
        p_ratio = paraphrase_ratio.get(mkey)
        a_ratio = action_reversal_ratio.get(mkey)
        h_ratio = hallucination_ratio.get(mkey)
        dc = swap_groups['directional'].get(mkey)
        nc = swap_groups['no_change'].get(mkey)
        swap_gap = (nc - dc) if isinstance(dc, (int, float)) and isinstance(nc, (int, float)) else None
        row = {
            'paraphrase_ratio_vs_original': p_ratio,
            'paraphrase_invariant': (p_ratio >= PARAPHRASE_INVARIANT_RATIO_THRESH) if p_ratio is not None else None,
            'action_reversal_ratio_vs_original': a_ratio,
            'action_reversal_sensitive': (a_ratio <= REVERSAL_SENSITIVE_RATIO_THRESH) if a_ratio is not None else None,
            'hallucination_ratio_vs_original': h_ratio,
            'hallucination_sensitive': (h_ratio <= REVERSAL_SENSITIVE_RATIO_THRESH) if h_ratio is not None else None,
            't1t2_swap_no_change_score': nc,
            't1t2_swap_directional_score': dc,
            't1t2_swap_gap_no_change_minus_directional': swap_gap,
            't1t2_swap_sensitive': (swap_gap >= SWAP_SENSITIVE_GAP_THRESH) if swap_gap is not None else None,
        }
        if mkey == 'spice_f_mean' and not spice_ok:
            row['unavailable_reason'] = spice_reason
        if mkey == 'llm_judge_mean_1to5' and not judge_ok:
            row['unavailable_reason'] = judge_reason
        table[mname] = row

    out = {
        'thresholds': {
            'paraphrase_invariant_ratio_thresh': PARAPHRASE_INVARIANT_RATIO_THRESH,
            'reversal_sensitive_ratio_thresh': REVERSAL_SENSITIVE_RATIO_THRESH,
            'swap_sensitive_gap_thresh': SWAP_SENSITIVE_GAP_THRESH,
        },
        'spice_available': spice_ok,
        'spice_unavailable_reason': None if spice_ok else spice_reason,
        'llm_judge_available': judge_ok,
        'llm_judge_unavailable_reason': None if judge_ok else judge_reason,
        'raw_group_scores_E04_perturbations': groups,
        'raw_group_scores_E10_t1t2_swap': swap_groups,
        'sensitivity_table': table,
    }
    out_path = OUT_DIR / 'sensitivity_table.json'
    out_path.write_text(json.dumps(out, indent=2))
    print(f'wrote {out_path}')
    print(json.dumps(table, indent=2))


if __name__ == '__main__':
    main()
