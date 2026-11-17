"""Builds the data bundle (JSON) for the human-annotation artifact: a
stratified real sample of (a) generated captions with T1/T2 image
thumbnails for E07 human-correlation rating, and (b) real reference
captions (text only) for E03 mapper-validation. Every item is real data
already on disk -- nothing here is synthesized.
"""
import base64, io, json, random
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from ricce.utils import read_jsonl

random.seed(42)


def thumb_b64(path, size=(160, 160), quality=65):
    try:
        im = Image.open(path).convert('RGB').resize(size)
        buf = io.BytesIO()
        im.save(buf, format='JPEG', quality=quality)
        return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')
    except Exception as e:
        return None


def load_manifest(ds):
    return {r['image_id']: r for r in read_jsonl(ROOT / 'artifacts' / 'manifests' / f'{ds}.jsonl')}


e07_items = []
sources = [
    ('LEVIR_CC', 'qwen', 40), ('LEVIR_CC', 'sft_seed42', 40),
    ('SECOND_CC', 'qwen', 30), ('DUBAI_CC', 'qwen', 30),
]
for ds, model, k in sources:
    pred_path = ROOT / 'results' / 'baselines' / model / f'{ds}_test.jsonl'
    if not pred_path.exists():
        print('MISSING', pred_path); continue
    m = load_manifest(ds)
    preds = [r for r in read_jsonl(pred_path) if r['image_id'] in m]
    sample = random.sample(preds, min(k, len(preds)))
    for r in sample:
        row = m[r['image_id']]
        t1 = thumb_b64(row['t1_path']); t2 = thumb_b64(row['t2_path'])
        if not (t1 and t2):
            continue
        e07_items.append({
            'id': f'{ds}__{model}__{r["image_id"]}', 'dataset': ds, 'model': model,
            'image_id': r['image_id'], 'prediction': r['prediction'], 't1': t1, 't2': t2,
        })
random.shuffle(e07_items)
print('E07 items:', len(e07_items))

e03_items = []
# plan requires ~300-500 manually labeled captions for E03 mapper
# validation (was 120); 160/dataset x 3 datasets = 480, within that range.
for ds in ['LEVIR_CC', 'SECOND_CC', 'DUBAI_CC']:
    m = load_manifest(ds)
    rows = [r for r in m.values() if r['references']]
    sample = random.sample(rows, min(160, len(rows)))
    for r in sample:
        cap = random.choice(r['references'])
        e03_items.append({'id': f'{ds}__{r["image_id"]}', 'dataset': ds, 'image_id': r['image_id'], 'caption': cap})
# Single-annotator test-retest reliability proxy: with one annotator (you),
# real inter-annotator agreement isn't obtainable, but blind duplicates of
# ~30 already-sampled captions (re-labeled under a different id, shuffled
# far from the original) give an honest self-consistency check instead of
# a fabricated multi-rater agreement number.
retest_source = random.sample(e03_items, min(30, len(e03_items)))
for r in retest_source:
    e03_items.append({**r, 'id': r['id'] + '__retest', 'retest_of': r['id']})
random.shuffle(e03_items)
print('E03 items:', len(e03_items), f'({len(retest_source)} blind retest duplicates included)')

bundle = {'e07': e07_items, 'e03': e03_items}
out = ROOT / 'results' / 'E07' / 'annotation_bundle.json'
out.write_text(json.dumps(bundle))
size_mb = out.stat().st_size / 1e6
print('wrote', out, f'{size_mb:.2f} MB')
