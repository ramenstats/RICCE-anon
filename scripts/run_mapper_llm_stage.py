"""LLM-fallback stage under the frozen evaluator on a fixed random subset (seed 0, 250 LEVIR-CC test images):
frozen alias+embedding mapper vs. the same mapper + constrained-JSON LLM fallback (Qwen2.5-VL-7B as a text LLM).
Same images for both configs; multi-event parse; mean over 5 refs. Reports CE-P/R/F, unresolved share and
the share of field resolutions handled by the LLM stage."""
import json, os, random, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from ricce.utils import read_jsonl
from ricce.ontology.tree import Taxonomy
from ricce.parsing.hybrid import HybridParser
from ricce.metrics.event import ricce_scores
cfg = yaml.safe_load(open(ROOT / 'configs/project.yaml')); taxes = {k: Taxonomy.from_yaml(ROOT / v) for k, v in cfg['ontology'].items()}
W = cfg['metric']['weights']; emb = cfg['parser'].get('embedding_model'); thr = float(cfg['parser'].get('similarity_threshold', 0.58))
llm = cfg.get('vlm', {}).get('model_name_or_path', 'Qwen/Qwen2.5-VL-7B-Instruct')
man = {r['image_id']: r for r in read_jsonl(ROOT / 'artifacts/manifests/LEVIR_CC.jsonl')}
ids = sorted(i for i in man if man[i]['split'] == 'test'); ids = random.Random(0).sample(ids, 250)
rows = []
for cname, kw in [('alias_embed (frozen)', {}), ('+LLM fallback', dict(use_llm_fallback=True, llm_model=llm))]:
    P = HybridParser(taxes['object'], taxes['action'], taxes['location'], taxes['quantity'], use_embeddings=True, embedding_model=emb, similarity_threshold=thr, **kw); cache = {}
    def parse(t):
        if t not in cache: cache[t] = P.parse(t, allow_multi_event=True)
        return cache[t]
    for sy in ['qwen', 'rscama']:
        pr = {r['image_id']: r['prediction'] for r in read_jsonl(ROOT / f'results/baselines/{sy}/LEVIR_CC_test.jsonl')}
        f_, p_, r_ = [], [], []
        for i in ids:
            pe = parse(pr[i]); ces = [ricce_scores(pe, parse(g), taxes, W) for g in man[i]['references']]
            f_.append(np.mean([c['ce_f1'] for c in ces])); p_.append(np.mean([c['ce_precision'] for c in ces])); r_.append(np.mean([c['ce_recall'] for c in ces]))
        rows.append(dict(config=cname, system=sy, n=len(f_), ce_p=np.mean(p_), ce_r=np.mean(r_), ce_f=np.mean(f_), stage_counts=json.dumps(P.stage_counts)))
        print(cname, sy, 'CE-F %.4f' % np.mean(f_), P.stage_counts, flush=True)
out = ROOT / 'results_cvpr_final/mapper_cascade'; out.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(out / 'llm_stage_subset250.csv', index=False)
print('done'); os._exit(0)
