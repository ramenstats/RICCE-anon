"""Deterministic qualitative examples from real LEVIR-CC test predictions under the frozen evaluator.
Categories: correct, over-generation (low CE-P), omission (low CE-R), false no-change, false change, direction error.
Picks the first (by image_id) matching image per category; reference shown = first reference."""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from ricce.utils import read_jsonl
from ricce.ontology.tree import Taxonomy
from ricce.parsing.hybrid import HybridParser
from ricce.metrics.event import ricce_scores, is_action_reversal
cfg = yaml.safe_load(open(ROOT / 'configs/project.yaml')); taxes = {k: Taxonomy.from_yaml(ROOT / v) for k, v in cfg['ontology'].items()}; W = cfg['metric']['weights']
P = HybridParser(taxes['object'], taxes['action'], taxes['location'], taxes['quantity'], use_embeddings=True, embedding_model=cfg['parser'].get('embedding_model'), similarity_threshold=float(cfg['parser'].get('similarity_threshold', 0.58)))
man = {r['image_id']: r for r in read_jsonl(ROOT / 'artifacts/manifests/LEVIR_CC.jsonl')}
def ev(e): return f"({e.object or '-'}, {e.action or '-'}, {e.location or '-'})"
def evs(t): return ' + '.join(ev(e) for e in P.parse(t, allow_multi_event=True))
rows = []
for sy in ['sft_seed42', 'rscama']:
    pf = {r['image_id']: r['prediction'] for r in read_jsonl(ROOT / f'results/baselines/{sy}/LEVIR_CC_test.jsonl')}
    pi = pd.read_csv(ROOT / f'results_cvpr_final/system_benchmark/per_image/LEVIR_CC__{sy}__multi.csv').sort_values('image_id')
    picked = {}
    for _, r in pi.iterrows():
        i = r.image_id; pred = pf[i]; ref = man[i]['references'][0]
        pe = P.parse(pred, allow_multi_event=True); ge = P.parse(ref, allow_multi_event=True)
        pn = all(e.change_state == 'no_change' for e in pe); gn = all(e.change_state == 'no_change' for e in ge)
        cat = None
        if r.ce_f >= 0.99 and not gn: cat = 'correct'
        elif pn and not gn: cat = 'false no-change'
        elif gn and not pn: cat = 'false change'
        elif not pn and not gn and any(is_action_reversal(p, g, taxes) and p.object and p.object == g.object for p in pe for g in ge): cat = 'direction error'
        elif r.ce_p < r.ce_r - 0.2 and not pn: cat = 'over-generation'
        elif r.ce_r < r.ce_p - 0.2 and not pn: cat = 'omission'
        if cat and cat not in picked:
            picked[cat] = dict(system=sy, image_id=i, category=cat, reference=ref.strip(), prediction=pred.strip(), ref_events=evs(ref), pred_events=evs(pred), ce_p=r.ce_p, ce_r=r.ce_r, ce_f=r.ce_f)
        if len(picked) == 6: break
    rows += list(picked.values())
out = ROOT / 'results_cvpr_final/error_examples'; out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out / 'examples.csv', index=False)
def tx(s): return s.replace('_', '\\_').replace('&', '\\&').replace('%', '\\%')
L = ['\\begin{tabular}{@{}p{1.6cm}p{3.2cm}p{3.2cm}p{4.6cm}c@{}}\\toprule', 'Case (system) & Reference (first) & Prediction & Parsed events (ref $\\to$ pred) & \\cep/\\cer\\\\\\midrule']
for r in rows:
    L.append(f"{r['category']} ({'SFT' if r['system'].startswith('sft') else 'RSCaMa'}) & {tx(r['reference'])} & {tx(r['prediction'])} & {tx(r['ref_events'])} $\\to$ {tx(r['pred_events'])} & {r['ce_p']:.2f}/{r['ce_r']:.2f}\\\\")
L.append('\\bottomrule\\end{tabular}')
T = ROOT / 'paper_cvpr_submission/tables'; T.mkdir(exist_ok=True); (T / 'error_examples.tex').write_text('\n'.join(L))
print(len(rows), 'examples'); print(pd.DataFrame(rows)[['system','category','ce_p','ce_r']].to_string()); os._exit(0)
