"""Mapper-cascade ablation under the frozen evaluator (multi-event parse, whole-event gates, mean over 5 refs).
Configs: alias-only | +n-gram | +embeddings (frozen default) | +n-gram+embeddings | +ancestor fallback (embeddings).
The constrained-LLM stage is not evaluated here (cost); it is not part of the frozen configuration.
Systems: sft_seed42 and rscama on LEVIR-CC test. Reports mean CE-P/CE-R/CE-F and the share of field
resolutions left unresolved (HybridParser.stage_counts)."""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from ricce.utils import read_jsonl
from ricce.ontology.tree import Taxonomy
from ricce.parsing.hybrid import HybridParser
from ricce.metrics.event import ricce_scores
cfg = yaml.safe_load(open(ROOT / 'configs/project.yaml')); taxes = {k: Taxonomy.from_yaml(ROOT / v) for k, v in cfg['ontology'].items()}
W = cfg['metric']['weights']; emb = cfg['parser'].get('embedding_model'); thr = float(cfg['parser'].get('similarity_threshold', 0.58))
man = {r['image_id']: r for r in read_jsonl(ROOT / 'artifacts/manifests/LEVIR_CC.jsonl')}
CONF = {'alias_only': dict(use_embeddings=False), 'alias_ngram': dict(use_embeddings=False, use_ngram=True),
        'alias_embed (frozen)': dict(use_embeddings=True, embedding_model=emb), 'alias_ngram_embed': dict(use_embeddings=True, embedding_model=emb, use_ngram=True),
        'alias_embed_ancestor': dict(use_embeddings=True, embedding_model=emb, use_ancestor_fallback=True)}
rows = []
for cname, kw in CONF.items():
    P = HybridParser(taxes['object'], taxes['action'], taxes['location'], taxes['quantity'], similarity_threshold=thr, **kw); cache = {}
    def parse(t):
        if t not in cache: cache[t] = P.parse(t, allow_multi_event=True)
        return cache[t]
    for sy in ['qwen']:
        preds = [r for r in read_jsonl(ROOT / f'results/baselines/{sy}/LEVIR_CC_test.jsonl') if r['image_id'] in man]
        p_, r_, f_ = [], [], []
        for r in preds:
            pe = parse(r['prediction']); ces = [ricce_scores(pe, parse(g), taxes, W) for g in man[r['image_id']]['references']]
            p_.append(np.mean([c['ce_precision'] for c in ces])); r_.append(np.mean([c['ce_recall'] for c in ces])); f_.append(np.mean([c['ce_f1'] for c in ces]))
        sc = P.stage_counts; tot = sum(sc.values())
        rows.append(dict(config=cname, system=sy, n=len(f_), ce_p=np.mean(p_), ce_r=np.mean(r_), ce_f=np.mean(f_), unresolved_share=sc['unresolved'] / max(1, tot), stage_counts=json.dumps(sc)))
        print(cname, sy, 'CE-F %.4f unresolved %.3f' % (np.mean(f_), sc['unresolved'] / max(1, tot)), flush=True)
out = ROOT / 'results_cvpr_final/mapper_cascade'; out.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(out / 'summary_qwen.csv', index=False)
json.dump({'command': ' '.join(sys.argv), 'note': 'stage_counts accumulate over all parses of the config (refs+preds), so unresolved_share is per field-resolution attempt'}, open(out / 'config_qwen.json', 'w'))
print('done'); os._exit(0)
