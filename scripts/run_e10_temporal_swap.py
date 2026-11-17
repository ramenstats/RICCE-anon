"""E10: temporal T1/T2 swap validation (internal design review,
Sec. 7.5 F1/F2 -- "arguably the most task-specific validation for a
temporal caption metric"). For a sample of real LEVIR-CC test pairs, we
generate a SECOND zero-shot Qwen2.5-VL caption with the image order
reversed (t2 presented as "before", t1 as "after") and check whether the
model's own directional-action word inverts consistently with the
taxonomy's reversal map (constructed<->demolished, appearance<->
disappearance, expansion<->contraction) -- and that a no-change pair
stays no-change under swap, since swapping a symmetric "nothing changed"
pair should not manufacture a change.

This does NOT require new ground truth: it checks self-consistency of the
model's own directional judgment against time reversal, which is exactly
what was requested, using the same frozen parser as everywhere
else in the paper.
"""
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ricce.utils import read_jsonl, load_yaml, write_jsonl, dump_json
from ricce.ontology.tree import Taxonomy
from ricce.parsing.hybrid import HybridParser

REVERSALS = {'constructed': 'demolished', 'demolished': 'constructed',
             'appearance': 'disappearance', 'disappearance': 'appearance',
             'expansion': 'contraction', 'contraction': 'expansion'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/project.yaml')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--limit', type=int, default=300)
    ap.add_argument('--adapter', default=None, help='v2 A5: LoRA checkpoint dir (e.g. checkpoints/LEVIR_CC/tcpo_v3_seed42) to evaluate instead of the frozen zero-shot base model, so temporal-reversal consistency can be measured per trained variant/seed, not just once for the base model.')
    a = ap.parse_args()
    cfg = load_yaml(a.config)
    vc = cfg['vlm']
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    rows = [r for r in read_jsonl(a.manifest) if r['split'] == 'test']
    rows = rows[:a.limit]

    if a.adapter:
        from ricce.vlm.qwen import QwenPairCaptioner
        import torch
        from peft import PeftModel
        cap = QwenPairCaptioner(vc['model_name_or_path'])
        cap.model = PeftModel.from_pretrained(cap.model, a.adapter)
        cap.model.eval()
    else:
        from ricce.vlm.qwen import QwenPairCaptioner
        cap = QwenPairCaptioner(vc['model_name_or_path'])

    preds = []
    for i, r in enumerate(rows, 1):
        normal = cap.generate(r['t1_path'], r['t2_path'], vc['prompt'], vc['max_new_tokens'], vc['temperature'])
        swapped = cap.generate(r['t2_path'], r['t1_path'], vc['prompt'], vc['max_new_tokens'], vc['temperature'])
        preds.append({'image_id': r['image_id'], 'normal_order': normal, 'swapped_order': swapped})
        if i % 20 == 0:
            write_jsonl(preds, out / 'raw_predictions.jsonl')
            print(f'{i}/{len(rows)}')
    write_jsonl(preds, out / 'raw_predictions.jsonl')
    print(f'Generated {len(preds)} normal+swapped pairs')

    taxes = {k: Taxonomy.from_yaml(v) for k, v in cfg['ontology'].items()}
    parser = HybridParser(taxes['object'], taxes['action'], taxes['location'], taxes['quantity'], use_embeddings=False)

    rows_out = []
    n_directional = n_directional_inverted = 0
    n_nochange = n_nochange_consistent = 0
    n_other = 0
    for p in preds:
        en = parser.parse(p['normal_order'])[0]
        es = parser.parse(p['swapped_order'])[0]
        row = {'image_id': p['image_id'], 'normal_action': en.action, 'swapped_action': es.action,
               'normal_change_state': en.change_state, 'swapped_change_state': es.change_state}
        if en.action == 'no_change':
            n_nochange += 1
            consistent = (es.action == 'no_change')
            n_nochange_consistent += int(consistent)
            row['category'] = 'no_change'; row['consistent'] = consistent
        elif en.action in REVERSALS:
            n_directional += 1
            inverted = (es.action == REVERSALS[en.action])
            n_directional_inverted += int(inverted)
            row['category'] = 'directional'; row['consistent'] = inverted
        else:
            n_other += 1
            row['category'] = 'other'; row['consistent'] = None
        rows_out.append(row)

    metrics = {
        'n_samples': len(preds),
        'n_directional_action': n_directional,
        'directional_inversion_accuracy': (n_directional_inverted / n_directional) if n_directional else float('nan'),
        'n_no_change': n_nochange,
        'no_change_consistency_under_swap': (n_nochange_consistent / n_nochange) if n_nochange else float('nan'),
        'n_other_or_unparsed': n_other,
    }
    write_jsonl(rows_out, out / 'per_sample_swap.jsonl')
    dump_json(metrics, out / 'metrics.json')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
