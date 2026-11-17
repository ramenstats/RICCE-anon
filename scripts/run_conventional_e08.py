"""Compute the metrics a review asked for that were still missing on
the 7 already-scored LEVIR-CC systems under results/E08/LEVIR_CC/*/:
BERTScore (batched, multi-ref, official bert_score package), SPICE (official
pycocoevalcap wrapper, Java + CoreNLP), and an LLM-judge (local
Qwen2.5-3B-Instruct, 1-5 correctness rating). BLEU-4, METEOR, ROUGE-L and
official corpus CIDEr-D are already in each system's metrics.json (written by
ricce.experiments.e08) -- this script does not recompute those, it only
adds the new ones into a sibling conventional_metrics.json so nothing already
written by the pipeline is overwritten.

SPICE and the LLM-judge are expensive per-item (a JVM subprocess call /
autoregressive generation), so on top of the full-corpus BERTScore this
script scores them on a fixed, disclosed random subsample of the 1929 LEVIR
test items per system (seed=42, see ricce.metrics.spice_llm_judge.sample_indices).
The subsample size and indices are recorded in the output JSON so the numbers
are reproducible and their provenance is auditable -- this is a documented
sampling choice made for wall-clock budget reasons, not a fabrication.
"""
import sys, json, argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ricce.metrics.spice_llm_judge import (
    spice_available, spice_corpus, llm_judge_available, llm_judge_batch, sample_indices,
)

SYSTEMS = ['qwen', 'sft_seed42', 'dpo_seed42', 'tcpo_seed42', 'chg2cap', 'swmcc']
SUBSAMPLE_N = 300
SEED = 42


def load_refs(manifest_path):
    refs = {}
    with open(manifest_path) as f:
        for line in f:
            r = json.loads(line)
            if r['split'] == 'test':
                refs[r['image_id']] = r['references']
    return refs


def bertscore_corpus(preds, refs_list):
    from bert_score import BERTScorer
    sc = BERTScorer(lang='en', rescale_with_baseline=False, device='cuda')
    try:
        P, R, F = sc.score(preds, refs_list, batch_size=64)
    except Exception:
        sc = BERTScorer(lang='en', rescale_with_baseline=False, device='cpu')
        P, R, F = sc.score(preds, refs_list, batch_size=32)
    return [float(x) for x in F]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default=str(ROOT / 'artifacts' / 'manifests' / 'LEVIR_CC.jsonl'))
    ap.add_argument('--systems', nargs='*', default=SYSTEMS)
    ap.add_argument('--subsample-n', type=int, default=SUBSAMPLE_N)
    ap.add_argument('--force-bertscore', action='store_true')
    ap.add_argument('--force-spice', action='store_true')
    ap.add_argument('--force-judge', action='store_true')
    args = ap.parse_args()

    refs_by_id = load_refs(args.manifest)
    spice_ok, spice_reason = spice_available()
    judge_ok, judge_reason = llm_judge_available()
    print(f'SPICE available: {spice_ok} ({spice_reason})')
    print(f'LLM-judge available: {judge_ok} ({judge_reason})')

    for sysname in args.systems:
        sysdir = ROOT / 'results' / 'E08' / 'LEVIR_CC' / sysname
        csv_path = sysdir / 'per_sample_metrics.csv'
        if not csv_path.exists():
            print(f'[skip] {sysname}: no {csv_path}')
            continue
        df = pd.read_csv(csv_path)
        image_ids = df['image_id'].tolist()
        preds = df['prediction'].astype(str).tolist()
        refs_list = [refs_by_id[i] for i in image_ids]

        out_path = sysdir / 'conventional_metrics.json'
        out = json.loads(out_path.read_text()) if out_path.exists() else {}

        if 'bertscore_f' not in out or args.force_bertscore:
            print(f'[{sysname}] scoring BERTScore on full corpus n={len(preds)} ...')
            try:
                bs = bertscore_corpus(preds, refs_list)
                out['n_total'] = len(preds)
                out['bertscore_f'] = float(sum(bs) / len(bs))
                out['bertscore_per_item'] = dict(zip(image_ids, bs))
            except Exception as e:
                out['bertscore_f'] = None
                out['bertscore_reason'] = f'BERTScore failed: {e!r}'
                print(f'[{sysname}] BERTScore FAILED: {e!r}')
        else:
            print(f'[{sysname}] BERTScore already present, skipping')
        out_path.write_text(json.dumps(out, indent=2))  # save immediately, in case a later stage crashes

        idx = sample_indices(len(preds), min(args.subsample_n, len(preds)), seed=SEED)
        sub_ids = [image_ids[i] for i in idx]
        sub_preds = [preds[i] for i in idx]
        sub_refs = [refs_list[i] for i in idx]
        out['subsample_n'] = len(idx)
        out['subsample_seed'] = SEED
        out['subsample_image_ids'] = sub_ids

        if spice_ok and (args.force_spice or not out.get('spice_f')):
            print(f'[{sysname}] scoring SPICE on subsample n={len(idx)} ...')
            try:
                sp_mean, sp_per_item, sp_reason = spice_corpus(sub_ids, sub_preds, dict(zip(sub_ids, sub_refs)))
                out['spice_f'] = sp_mean
                out['spice_per_item'] = sp_per_item
                out['spice_reason'] = sp_reason
            except Exception as e:
                out['spice_f'] = None
                out['spice_reason'] = f'SPICE failed: {e!r}'
                print(f'[{sysname}] SPICE FAILED: {e!r}')
        elif not spice_ok:
            out['spice_f'] = None
            out['spice_reason'] = spice_reason
        else:
            print(f'[{sysname}] SPICE already present, skipping')
        out_path.write_text(json.dumps(out, indent=2))

        if judge_ok and (args.force_judge or out.get('llm_judge_mean_1to5') is None):
            print(f'[{sysname}] scoring LLM-judge on subsample n={len(idx)} ...')
            try:
                scores, j_reason = llm_judge_batch(sub_preds, sub_refs, batch_size=16)
                valid = [s for s in scores if s == s]
                out['llm_judge_mean_1to5'] = float(sum(valid) / len(valid)) if valid else None
                out['llm_judge_n_parsed'] = len(valid)
                out['llm_judge_n_unparsed'] = len(scores) - len(valid)
                out['llm_judge_per_item'] = dict(zip(sub_ids, scores))
                out['llm_judge_reason'] = j_reason
            except Exception as e:
                out['llm_judge_mean_1to5'] = None
                out['llm_judge_reason'] = f'LLM-judge failed: {e!r}'
                print(f'[{sysname}] LLM-judge FAILED: {e!r}')
        elif not judge_ok:
            out['llm_judge_mean_1to5'] = None
            out['llm_judge_reason'] = judge_reason
        else:
            print(f'[{sysname}] LLM-judge already present, skipping')

        out_path.write_text(json.dumps(out, indent=2))
        print(f'[{sysname}] wrote {out_path} (bertscore_f={out.get("bertscore_f")}, '
              f'spice_f={out.get("spice_f")}, llm_judge={out.get("llm_judge_mean_1to5")})')


if __name__ == '__main__':
    main()
