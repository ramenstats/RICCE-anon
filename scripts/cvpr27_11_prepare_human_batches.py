"""CVPR'27: annotation batches for the work that needs NEW human labels.
Every batch is built from real data by a fixed random rule (seed 20261029) and every label column
is EMPTY. Nothing here produces, imputes or simulates a label. Each batch has an annotator sheet
(no hidden information), an analyst-only key where needed (*_KEY_DO_NOT_SHARE.*) and images.

  04_CONFIRMATORY_HUMAN  H2: image-only confirmatory study. Random stratified sample of natural
                         system outputs on test image pairs NOT used in the first study; sheet shows
                         T1, T2 and the caption only (no references, no system, no scores). Plus
                         same-image pairs of two systems for preferences.
  05_PARSER_ORACLE       P1: gold event annotation of captions (references and predictions).
  09_DOMAIN_TRANSFER     X1: changed / unchanged / ambiguous labels from the images alone
                         (all 150 Dubai-CC test pairs, 150 SECOND-CC, 150 LEVIR-CC).
  08_METHOD_ABLATIONS    M1: random sample of gate activations and non-activations on real
                         same-image reference pairs, to be labelled against the images.
  10_TEMPORAL            T1: stratified sample of swapped-order outputs to verify against the images."""
import json, random, sys
from pathlib import Path
import pandas as pd
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.utils import read_jsonl
S = ROOT / 'submission_cvpr27_new_version'
H = ROOT / 'artifacts_cvpr_final' / 'human_eval' / 'image_grounded'
SEED = 20261029
DS = ('LEVIR_CC', 'SECOND_CC', 'DUBAI_CC')
man = {ds: {r['image_id']: r for r in read_jsonl(ROOT / 'artifacts' / 'manifests' / f'{ds}.jsonl')} for ds in DS}
key = [json.loads(l) for l in open(H / 'hidden' / 'answer_key_final_evaluator.jsonl')]
used = {(k['dataset'], k['image_id']) for k in key}                        # images of the first study


def thumbs(ds, iid, out_dir):
    m = man[ds][iid]; out_dir.mkdir(parents=True, exist_ok=True)
    rel = []
    for tag, p in (('A', m['t1_path']), ('B', m['t2_path'])):
        dst = out_dir / f'{ds}_{iid}_{tag}.png'
        if not dst.exists():
            Image.open(p).convert('RGB').save(dst)
        rel.append(f'images/{dst.name}')
    return rel


def write_sheets(df, d, name, n_ann=3):
    for a in range(1, n_ann + 1):
        df.assign(annotator_id=f'ann{a}').to_csv(d / f'{name}_ann{a}.csv', index=False)


# ------------------------------------------------------------------ H2 image-only confirmatory
d = S / '04_CONFIRMATORY_HUMAN'
SYS = ['qwen', 'sft_seed42', 'chg2cap', 'swmcc']
pool = []
for ds in DS:
    for sy in SYS:
        p = ROOT / 'results' / 'E08' / ds / sy / 'per_sample_metrics.csv'
        e = pd.read_csv(p, usecols=['image_id', 'prediction']); e['prediction'] = e.prediction.fillna('').astype(str)
        for r in e.itertuples():
            if (ds, r.image_id) not in used and r.prediction.strip():
                pool.append(dict(dataset=ds, system=sy, image_id=r.image_id, candidate=' '.join(r.prediction.split())))
pool = pd.DataFrame(pool)
rng = random.Random(SEED)
PER = {'LEVIR_CC': 30, 'SECOND_CC': 30, 'DUBAI_CC': 20}             # per system: 5 x (30+30+20) = 400 items
rows = []
for (ds, sy), g in pool.groupby(['dataset', 'system']):
    idx = sorted(g.index); pick = rng.sample(idx, PER[ds])
    for i in pick:
        rows.append({**pool.loc[i].to_dict(), 'inclusion_probability': PER[ds] / len(g), 'stratum_size': len(g)})
smp = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
smp.insert(0, 'item_id', [f'C{i + 1:04d}' for i in range(len(smp))])
smp.to_csv(d / 'random_sampling_manifest_KEY_DO_NOT_SHARE.csv', index=False)
# pairs: 120 images (40 per dataset) with two different systems, random A/B order
prs = []
for ds in DS:
    imgs = sorted({i for i in pool[pool.dataset == ds].image_id})
    for iid in rng.sample(imgs, min(40, len(imgs))):
        g = pool[(pool.dataset == ds) & (pool.image_id == iid)]
        a, b = rng.sample(list(g.index), 2)
        prs.append(dict(dataset=ds, image_id=iid, system_a=pool.loc[a, 'system'], system_b=pool.loc[b, 'system'],
                        candidate_a=pool.loc[a, 'candidate'], candidate_b=pool.loc[b, 'candidate']))
prs = pd.DataFrame(prs); prs.insert(0, 'pair_id', [f'Q{i + 1:04d}' for i in range(len(prs))])
prs.to_csv(d / 'pairs_manifest_KEY_DO_NOT_SHARE.csv', index=False)
AX = ['object', 'action', 'source_target', 'location', 'quantity', 'change_nochange', 'specificity', 'completeness', 'hallucination', 'overall']
sheet = []
for r in smp.itertuples():
    t1, t2 = thumbs(r.dataset, r.image_id, d / 'images')
    sheet.append(dict(task='A', item_id=r.item_id, dataset=r.dataset, T1_image=t1, T2_image=t2, candidate_or_A=r.candidate, candidate_B='',
                      **{a: '' for a in AX}, preference='', notes=''))
for r in prs.itertuples():
    t1, t2 = thumbs(r.dataset, r.image_id, d / 'images')
    sheet.append(dict(task='B', item_id=r.pair_id, dataset=r.dataset, T1_image=t1, T2_image=t2, candidate_or_A=r.candidate_a,
                      candidate_B=r.candidate_b, **{a: '' for a in AX}, preference='', notes=''))
write_sheets(pd.DataFrame(sheet), d, 'image_only_annotation_sheet')

# ------------------------------------------------------------------ P1 gold events
d = S / '05_PARSER_ORACLE' / 'gold_annotation_templates'; d.mkdir(parents=True, exist_ok=True)
caps = []
for ds in DS:
    test = [m for m in man[ds].values() if m['split'] == 'test' and (ds, m['image_id']) not in used]
    for m in rng.sample(test, min(100, len(test))):
        caps.append(dict(dataset=ds, image_id=m['image_id'], kind='reference', caption=' '.join(rng.choice(m['references']).split())))
    pr = pool[pool.dataset == ds]
    for i in rng.sample(list(pr.index), 67):
        caps.append(dict(dataset=ds, image_id=pr.loc[i, 'image_id'], kind='prediction', caption=pr.loc[i, 'candidate']))
caps = pd.DataFrame(caps).sample(frac=1, random_state=SEED).reset_index(drop=True)
caps.insert(0, 'caption_id', [f'G{i + 1:04d}' for i in range(len(caps))])
caps.to_csv(d / 'holdout_ids.csv', index=False)
EV = ['event_id', 'object', 'action', 'source', 'target', 'location', 'quantity_class', 'exact_count', 'change_state', 'polarity',
      'clause_span', 'uncertainty', 'notes']
caps[['caption_id', 'dataset', 'caption']].assign(**{c: '' for c in EV}).to_csv(d / 'parse_gold_template.csv', index=False)

# ------------------------------------------------------------------ X1 change labels
d = S / '09_DOMAIN_TRANSFER'
lab = []
for ds, n in (('DUBAI_CC', 150), ('SECOND_CC', 150), ('LEVIR_CC', 150)):
    test = sorted(i for i, m in man[ds].items() if m['split'] == 'test')
    for iid in (test if n >= len(test) else rng.sample(test, n)):
        t1, t2 = thumbs(ds, iid, d / 'images')
        lab.append(dict(dataset=ds, image_id=iid, T1_image=t1, T2_image=t2, label_changed_unchanged_ambiguous='', notes=''))
lab = pd.DataFrame(lab).sample(frac=1, random_state=SEED).reset_index(drop=True)
lab.insert(0, 'pair_id', [f'X{i + 1:04d}' for i in range(len(lab))])
write_sheets(lab, d, 'independent_change_labels_sheet')

# ------------------------------------------------------------------ M1 gate activations
d = S / '08_METHOD_ABLATIONS'
rp = pd.read_csv(ROOT / 'results_v3' / 'gate' / 'reference_pairs.csv.gz')
rp = rp[rp.split == 'test']
act = rp[rp.activated].sample(n=min(100, int(rp.activated.sum())), random_state=SEED)
non = rp[~rp.activated].sample(n=100, random_state=SEED)
g = pd.concat([act, non]).sample(frac=1, random_state=SEED).reset_index(drop=True)
g.insert(0, 'audit_id', [f'M{i + 1:04d}' for i in range(len(g))])
g[['audit_id', 'image_id', 'activated', 'types', 'cef_hard', 'cef_none']].to_csv(d / 'gate_audit_KEY_DO_NOT_SHARE.csv', index=False)
sh = []
for r in g.itertuples():
    t1, t2 = thumbs('LEVIR_CC', r.image_id, d / 'images')
    sh.append(dict(audit_id=r.audit_id, T1_image=t1, T2_image=t2, caption_1=' '.join(r.ref_a.split()), caption_2=' '.join(r.ref_b.split()),
                   label_same_object_opposite_change_YES_NO='', label_caption_1_correct_YES_NO_UNSURE='', label_caption_2_correct_YES_NO_UNSURE='', notes=''))
write_sheets(pd.DataFrame(sh), d, 'gate_activation_labels_sheet', n_ann=2)

# ------------------------------------------------------------------ T1 swapped outputs
d = S / '10_TEMPORAL'
R = ROOT / 'results_cvpr_final' / 'temporal_equivariance' / 'runs'
tw = []
for run in ('swapsft_seed42', 'sft2x_seed42'):
    raw = {json.loads(l)['image_id']: json.loads(l) for l in open(R / run / 'e10' / 'raw_predictions.jsonl')}
    ps = [json.loads(l) for l in open(R / run / 'e10' / 'per_sample_swap.jsonl')]
    for cat in ('directional', 'no_change', 'other'):
        c = [p for p in ps if p['category'] == cat]
        for p in rng.sample(c, min(len(c), 15 if cat != 'other' else 5)):
            tw.append(dict(run=run, image_id=p['image_id'], category=cat, normal=raw[p['image_id']]['normal_order'],
                           swapped=raw[p['image_id']]['swapped_order'], parser_says_consistent=p['consistent']))
tw = pd.DataFrame(tw).sample(frac=1, random_state=SEED).reset_index(drop=True)
tw.insert(0, 'audit_id', [f'T{i + 1:04d}' for i in range(len(tw))])
tw.to_csv(d / 'manual_swap_audit_KEY_DO_NOT_SHARE.csv', index=False)
sh = []
for r in tw.itertuples():
    t1, t2 = thumbs('LEVIR_CC', r.image_id, d / 'images')
    sh.append(dict(audit_id=r.audit_id, T1_image=t1, T2_image=t2, caption_for_T1_to_T2=r.normal, caption_for_T2_to_T1=r.swapped,
                   label_first_caption_correct_YES_NO_UNSURE='', label_second_caption_correct_YES_NO_UNSURE='', notes=''))
write_sheets(pd.DataFrame(sh), d, 'manual_swap_audit_sheet', n_ann=2)
print(dict(H2_items=len(smp), H2_pairs=len(prs), P1_captions=len(caps), X1_pairs=len(lab), M1_gate=len(g), T1_swaps=len(tw)))
