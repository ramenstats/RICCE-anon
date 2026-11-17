#!/usr/bin/env python3
"""Build TemporalSwap-SFT augmentation data from the LEVIR-CC TRAIN split.

Swapped captions are RULE-GENERATED (ricce/training/temporal_swap.py), not
human-written. Per train image:
  * every reference is parsed (HybridParser, multi-event allowed);
  * a reference is 'directional-eligible' iff it parses to exactly ONE event,
    non-negated, action in the reversal families, and reverse_caption() covers
    every directional word with a single direction group; the reversed caption
    must re-parse to exactly rho(original action) and be an is_action_reversal;
  * an image is kept if it has >=1 eligible ref and all eligible refs share
    the same action family (else dropped as conflicting). Its swapped sample =
    images (T2,T1) + all purity-checked reversed refs;
  * no-change images (ALL refs parse to no_change) are added swapped-order with
    the same references, in the natural no-change:directional proportion
    (n_nc_aug = round(n_dir_kept * n_nc_images / n_dir_eligible_images), capped).
"""
import argparse, json, random, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ricce.utils import read_jsonl, write_jsonl, load_yaml
from ricce.ontology.tree import Taxonomy
from ricce.parsing.hybrid import HybridParser
from ricce.training.temporal_swap import reverse_caption, check_purity, DIRECTIONAL_ACTIONS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/project.yaml')
    ap.add_argument('--manifest', default='artifacts/manifests/LEVIR_CC.jsonl')
    ap.add_argument('--out', default='artifacts_cvpr_final/temporal_equivariance/swap_train.jsonl')
    ap.add_argument('--examples', default='artifacts_cvpr_final/temporal_equivariance/examples.jsonl')
    ap.add_argument('--stats', default='artifacts_cvpr_final/temporal_equivariance/stats.json')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    cfg = load_yaml(a.config)
    taxes = {k: Taxonomy.from_yaml(v) for k, v in cfg['ontology'].items()}
    parser = HybridParser(taxes['object'], taxes['action'], taxes['location'], taxes['quantity'], use_embeddings=False)
    rng = random.Random(a.seed)
    rows = [r for r in read_jsonl(a.manifest) if r['split'] == 'train' and r['references']]
    cap_drop = collections.Counter(); cap_total = 0; cap_ok = 0
    dir_rows, nc_rows, pairs = [], [], []
    img_drop = collections.Counter(); n_dir_elig_img = 0
    for r in rows:
        kept, actions, all_nc = [], set(), True
        for ref in r['references']:
            cap_total += 1
            ref = ref.strip()
            ev = parser.parse(ref, allow_multi_event=True)
            if not (len(ev) == 1 and ev[0].action == 'no_change'):
                all_nc = False
            if len(ev) != 1:
                cap_drop['not_single_event'] += 1; continue
            e = ev[0]
            if e.action == 'no_change':
                cap_drop['no_change(handled separately)'] += 1; continue
            if e.negated:
                cap_drop['negated'] += 1; continue
            if e.action is None:
                cap_drop['unresolved_action'] += 1; continue
            if e.action not in DIRECTIONAL_ACTIONS:
                cap_drop[f'non_reversal_action:{e.action}'] += 1; continue
            rev, info = reverse_caption(ref)
            if rev is None:
                cap_drop[info.split(':')[0] if info.startswith('uncovered') else info] += 1
                if info.startswith('uncovered'):
                    cap_drop['  uncovered word ' + info.split(':')[1]] += 0  # counted below
                continue
            ok, why, ao, as_ = check_purity(ref, rev, parser, taxes)
            if not ok:
                cap_drop[why] += 1; continue
            cap_ok += 1
            kept.append((ref, rev, ao, as_))
            actions.add(ao)
        if all_nc:
            nc_rows.append(r); continue
        if not kept:
            img_drop['no_eligible_reference'] += 1; continue
        n_dir_elig_img += 1
        fams = {('app' if x in ('constructed', 'appearance') else 'dis' if x in ('disappearance', 'demolished')
                 else x) for x in actions}
        if len(fams) != 1:
            img_drop['conflicting_reference_families'] += 1; continue
        row = {'dataset': 'LEVIR_CC', 'image_id': r['image_id'] + '__swap', 'orig_image_id': r['image_id'],
               'split': 'train', 't1_path': r['t2_path'], 't2_path': r['t1_path'],
               'references': [k[1] for k in kept], 'orig_references': [k[0] for k in kept],
               'orig_actions': [k[2] for k in kept], 'swapped_actions': [k[3] for k in kept],
               'kind': 'directional', 'rule_generated': True}
        dir_rows.append(row)
        for k in kept:
            pairs.append({'image_id': r['image_id'], 'original': k[0], 'swapped': k[1],
                          'orig_action': k[2], 'swapped_action': k[3], 'kind': 'directional'})
    n_nc_target = min(len(nc_rows), round(len(dir_rows) * len(nc_rows) / max(1, n_dir_elig_img)))
    nc_sel = rng.sample(nc_rows, n_nc_target)
    out_rows = list(dir_rows)
    for r in nc_sel:
        out_rows.append({'dataset': 'LEVIR_CC', 'image_id': r['image_id'] + '__swap', 'orig_image_id': r['image_id'],
                         'split': 'train', 't1_path': r['t2_path'], 't2_path': r['t1_path'],
                         'references': [x.strip() for x in r['references']],
                         'orig_references': [x.strip() for x in r['references']],
                         'orig_actions': ['no_change'] * len(r['references']),
                         'swapped_actions': ['no_change'] * len(r['references']),
                         'kind': 'no_change', 'rule_generated': False})
        pairs.append({'image_id': r['image_id'], 'original': r['references'][0].strip(),
                      'swapped': r['references'][0].strip(), 'orig_action': 'no_change',
                      'swapped_action': 'no_change', 'kind': 'no_change'})
    # purity of kept: recheck 100% (already enforced per caption)
    bad = 0
    for p in pairs:
        ok, *_ = check_purity(p['original'], p['swapped'], parser, taxes)
        bad += (not ok)
    assert bad == 0, bad
    write_jsonl(out_rows, a.out)
    # 50-example sample: 40 directional (spread over actions) + 10 no-change
    dpairs = [p for p in pairs if p['kind'] == 'directional']
    rng.shuffle(dpairs)
    byact = collections.defaultdict(list)
    for p in dpairs: byact[p['orig_action']].append(p)
    sample = []
    while len(sample) < 40 and any(byact.values()):
        for k in list(byact):
            if byact[k] and len(sample) < 40: sample.append(byact[k].pop())
    ncp = [p for p in pairs if p['kind'] == 'no_change']; rng.shuffle(ncp)
    sample += ncp[:10]
    write_jsonl(sample, a.examples)
    stats = {'train_images': len(rows), 'train_captions': cap_total,
             'captions_directional_eligible_and_pure': cap_ok,
             'caption_drop_reasons': {k: v for k, v in sorted(cap_drop.items(), key=lambda x: -x[1]) if v},
             'images_with_eligible_directional_ref': n_dir_elig_img,
             'image_drop_reasons': dict(img_drop),
             'directional_swapped_samples_kept': len(dir_rows),
             'no_change_train_images': len(nc_rows), 'no_change_swapped_samples_included': len(nc_sel),
             'total_aug_samples': len(out_rows),
             'directional_action_counts_kept_refs': dict(collections.Counter(p['orig_action'] for p in pairs if p['kind'] == 'directional')),
             'purity_recheck_failures_among_kept': bad, 'purity_pass_rate_kept': 1.0}
    Path(a.stats).write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
