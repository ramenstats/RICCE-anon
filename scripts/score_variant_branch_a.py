#!/usr/bin/env python3
"""v2 Branch A (TCPO causal-severity study) independent evaluation
pipeline (A5). For one trained LoRA checkpoint, runs the FULL required metric
suite -- not just CE-F1, since severity weights were derived from the same
CE-F1/taxonomy machinery being evaluated:

  - CE-F1/precision/recall, hallucination_rate, omission_rate, BLEU-1..4,
    METEOR, ROUGE-L, CIDEr-D            (via ricce experiment E08, full
                                          test set)
  - BERTScore                            (via E08 --expensive_limit, capped
                                          subset -- heavy CPU model)
  - SPICE                                (via ricce.metrics.spice_llm_judge,
                                          only if java+pycocoevalcap are
                                          actually available -- reported as
                                          unavailable otherwise, never faked)
  - change/no-change accuracy            (predicted vs reference change_state,
                                          parsed by the same frozen
                                          HybridParser used everywhere else)
  - temporal-reversal (E10 swap) directional-inversion accuracy and
    no-change consistency under T1/T2 swap, evaluated on THIS checkpoint's
    adapter (not just the zero-shot base model), capped subset

Writes one combined metrics.json per variant into
results_r1/tcpo_branch_a/metrics/<name>/metrics.json, and copies
the training run_config.json + diagnostics.jsonl alongside it so every
completed checkpoint's full provenance lives in one place.

Never fabricates: any stage that fails or is unavailable is recorded as
such (status/error field), not silently dropped or estimated.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sh(cmd, log_path):
    print(f'  $ {" ".join(cmd)}  (log: {log_path})')
    with open(log_path, 'w') as f:
        p = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    return p.returncode


def change_state_accuracy(manifest_path, per_sample_csv, cfg_path):
    import pandas as pd
    from ricce.utils import read_jsonl, load_yaml
    from ricce.experiments import build_stack
    cfg = load_yaml(cfg_path)
    _, parser = build_stack(cfg)
    rows = {r['image_id']: r for r in read_jsonl(manifest_path)}
    df = pd.read_csv(per_sample_csv)
    n = 0; correct = 0
    n_change_ref = 0; correct_change = 0
    n_nochange_ref = 0; correct_nochange = 0
    for _, row in df.iterrows():
        iid = row['image_id']
        if iid not in rows:
            continue
        refs = rows[iid]['references']
        from collections import Counter
        votes = Counter()
        for ref in refs:
            ev = parser.parse(ref)[0]
            votes[ev.change_state or 'unknown'] += 1
        ref_state = votes.most_common(1)[0][0] if votes else 'unknown'
        pred_state = parser.parse(str(row['prediction']))[0].change_state or 'unknown'
        n += 1
        match = (pred_state == ref_state)
        correct += int(match)
        if ref_state == 'change':
            n_change_ref += 1; correct_change += int(match)
        elif ref_state == 'no_change':
            n_nochange_ref += 1; correct_nochange += int(match)
    return {
        'n': n, 'change_state_accuracy': (correct / n) if n else float('nan'),
        'n_change_ref': n_change_ref,
        'change_recall_accuracy': (correct_change / n_change_ref) if n_change_ref else float('nan'),
        'n_nochange_ref': n_nochange_ref,
        'nochange_recall_accuracy': (correct_nochange / n_nochange_ref) if n_nochange_ref else float('nan'),
    }


def spice_score(manifest_path, per_sample_csv):
    sys.path.insert(0, str(ROOT))
    from ricce.metrics.spice_llm_judge import spice_available, spice_corpus
    ok, reason = spice_available()
    if not ok:
        return {'status': 'unavailable', 'reason': reason}
    import pandas as pd
    from ricce.utils import read_jsonl
    rows = {r['image_id']: r for r in read_jsonl(manifest_path)}
    df = pd.read_csv(per_sample_csv)
    ids = [r['image_id'] for _, r in df.iterrows() if r['image_id'] in rows]
    preds = [str(r['prediction']) for _, r in df.iterrows() if r['image_id'] in rows]
    refs_by_id = {iid: rows[iid]['references'] for iid in ids}
    try:
        overall, _per = spice_corpus(ids, preds, refs_by_id)
        return {'status': 'ok', 'spice': overall}
    except Exception as e:
        return {'status': 'error', 'reason': repr(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adapter', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--manifest', default='artifacts/manifests/LEVIR_CC.jsonl')
    ap.add_argument('--config', default='configs/project.yaml')
    ap.add_argument('--expensive_limit', type=int, default=200)
    ap.add_argument('--e10_limit', type=int, default=150)
    ap.add_argument('--skip_e10', action='store_true')
    a = ap.parse_args()

    adapter = Path(a.adapter)
    if not adapter.is_dir() or not any(adapter.iterdir()):
        print(f'No checkpoint at {adapter} -- nothing to score.')
        sys.exit(1)

    out_dir = ROOT / 'results_r1' / 'tcpo_branch_a' / 'metrics' / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = ROOT / 'results_r1' / 'tcpo_branch_a' / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    combined = {'name': a.name, 'adapter': str(adapter)}

    # 1) inference
    pred_path = ROOT / 'results' / 'baselines' / a.name / 'LEVIR_CC_test.jsonl'
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    rc = sh(['python3', 'scripts/lora_infer.py', '--adapter', str(adapter), '--manifest', a.manifest,
             '--output', str(pred_path), '--split', 'test'], logs_dir / f'infer_{a.name}.log')
    combined['inference_rc'] = rc
    if rc != 0:
        combined['status'] = 'inference_failed'
        (out_dir / 'metrics.json').write_text(json.dumps(combined, indent=2))
        print(f'Inference failed for {a.name}; see {logs_dir}/infer_{a.name}.log')
        sys.exit(rc)

    # 2) E08 (full set, BERTScore on first --expensive_limit)
    e08_dir = ROOT / 'results' / 'E08' / 'LEVIR_CC' / a.name
    e08_dir.mkdir(parents=True, exist_ok=True)
    rc = sh([sys.executable, '-m', 'ricce.cli', 'experiment', 'E08', a.manifest, '--predictions', str(pred_path),
             '--outdir', str(e08_dir), '--expensive-limit', str(a.expensive_limit)],
            logs_dir / f'e08_{a.name}.log')
    combined['e08_rc'] = rc
    if rc == 0 and (e08_dir / 'metrics.json').exists():
        combined['e08'] = json.loads((e08_dir / 'metrics.json').read_text())
    else:
        combined['e08'] = {'status': 'failed'}

    per_sample_csv = e08_dir / 'per_sample_metrics.csv'

    # 3) SPICE (best-effort, corpus-level, honestly marked unavailable if it is)
    if per_sample_csv.exists():
        try:
            combined['spice'] = spice_score(a.manifest, per_sample_csv)
        except Exception as e:
            combined['spice'] = {'status': 'error', 'reason': repr(e)}
    else:
        combined['spice'] = {'status': 'skipped_no_per_sample_csv'}

    # 4) change/no-change accuracy
    if per_sample_csv.exists():
        try:
            combined['change_state'] = change_state_accuracy(a.manifest, per_sample_csv, a.config)
        except Exception as e:
            combined['change_state'] = {'status': 'error', 'reason': repr(e)}
    else:
        combined['change_state'] = {'status': 'skipped_no_per_sample_csv'}

    # 5) E10 temporal-reversal swap, on THIS checkpoint
    if not a.skip_e10:
        e10_dir = ROOT / 'results_r1' / 'tcpo_branch_a' / 'e10' / a.name
        e10_dir.mkdir(parents=True, exist_ok=True)
        rc = sh(['python3', 'scripts/run_e10_temporal_swap.py', '--manifest', a.manifest,
                 '--outdir', str(e10_dir), '--limit', str(a.e10_limit), '--adapter', str(adapter)],
                logs_dir / f'e10_{a.name}.log')
        combined['e10_rc'] = rc
        if rc == 0 and (e10_dir / 'metrics.json').exists():
            combined['e10'] = json.loads((e10_dir / 'metrics.json').read_text())
        else:
            combined['e10'] = {'status': 'failed'}
    else:
        combined['e10'] = {'status': 'skipped'}

    # 6) copy training provenance alongside, if present
    run_cfg = ROOT / 'checkpoints' / 'LEVIR_CC' / a.name / 'run_config.json'
    if run_cfg.exists():
        combined['training_run_config'] = json.loads(run_cfg.read_text())
    diag = ROOT / 'results' / 'training_diagnostics' / f'{a.name}.jsonl'
    if diag.exists():
        combined['training_diagnostics_path'] = str(diag)

    combined['status'] = 'complete'
    combined['scoring_elapsed_s'] = round(time.time() - t0, 1)
    (out_dir / 'metrics.json').write_text(json.dumps(combined, indent=2))
    print(f'Done: {out_dir}/metrics.json')


if __name__ == '__main__':
    main()
