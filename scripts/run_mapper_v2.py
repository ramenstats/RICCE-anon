"""Mapper v2: per-axis conditional resolution/consistency vs single-annotator free-text
labels (PROXY, via conservative phrase->node mapping) + node-label annotation package."""
import sys, json, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from ricce.evaluation.mapper_v2 import *

OUT = ROOT / 'results_cvpr_final/mapper_v2'; ART = ROOT / 'artifacts_cvpr_final/mapper_v2'
OUT.mkdir(parents=True, exist_ok=True); ART.mkdir(parents=True, exist_ok=True)
cfg = load_cfg(); taxes = load_taxes(cfg); parser = build_parser(taxes, cfg); hm = HumanMapper(taxes)
lab = load_labels()

per_sample, cells, confusions = [], [], []
for row in label_records(lab):
    m = hm.map_row(row)
    ev = parser.parse(row['caption'])[0]
    nochg = row['change_state'] == 'no_change'
    rec = {'id': row['id'], 'dataset': row['dataset'], 'image_id': row['image_id'], 'caption': row['caption'],
           'fully_mappable': hm.fully_mappable(m)}
    for ax in AXES:
        pn = parser_field(ev, ax)
        cat = classify_axis(ax, row[ax], m[ax]['node'], m[ax]['status'], pn, nochg)
        tax = taxes[AXIS_TAX[ax]] if ax in AXIS_TAX else None
        rel = dist = hp = hr = hf = None
        if cat in ('b_consistent', 'c_different'):
            rel = relation(tax, pn, m[ax]['node'])
            if tax is not None:
                dist = tax.tree_distance(pn, m[ax]['node']); hp, hr, hf = tax.hierarchical_scores(pn, m[ax]['node'])
            else:
                dist = 0 if rel == 'exact' else None; hp = hr = hf = 1.0 if rel == 'exact' else 0.0
        rec.update({f'{ax}_human': row[ax], f'{ax}_human_node': m[ax]['node'], f'{ax}_human_status': m[ax]['status'],
                    f'{ax}_parser': pn, f'{ax}_cat': cat, f'{ax}_rel': rel})
        cells.append(dict(dataset=row['dataset'], axis=ax, cat=cat, present=row[ax] is not None, mappable=m[ax]['status'] == 'mapped',
                          rel=rel, dist=dist, hp=hp, hr=hr, hf=hf))
        if cat in ('c_different', 'e_spurious', 'd_unresolved', 'f_resolved_human_unmappable') :
            confusions.append({'id': row['id'], 'dataset': row['dataset'], 'axis': ax, 'category': cat, 'caption': row['caption'],
                               'human_phrase': row[ax], 'human_node': m[ax]['node'], 'human_status': m[ax]['status'], 'parser_node': pn, 'relation': rel})
    per_sample.append(rec)
pd.DataFrame(per_sample).to_csv(OUT / 'per_sample.csv', index=False)
with open(OUT / 'confusion_examples.jsonl', 'w') as f:
    by = {}
    for c in confusions: by.setdefault((c['axis'], c['category']), []).append(c)
    rnd = random.Random(0)
    for k in sorted(by):
        for c in rnd.sample(by[k], min(15, len(by[k]))): f.write(json.dumps(c) + '\n')

C = pd.DataFrame(cells)
def agg(g):
    n = len(g); cat = g['cat']; pres = g[g.present]; np_ = len(pres)
    res = pres[pres.cat != 'd_unresolved']; mp = pres[pres.mappable]; mpr = mp[mp.cat.isin(['b_consistent', 'c_different'])]
    d = {'n_captions': n, 'n_present': np_}
    for c in ['a_null_correct', 'b_consistent', 'c_different', 'd_unresolved', 'e_spurious', 'f_resolved_human_unmappable']: d['n_' + c] = int((cat == c).sum())
    d['P_resolved_given_present'] = len(res) / np_ if np_ else None
    lo, hi = wilson(len(res), np_); d['P_resolved_CI95'] = [lo, hi]
    d['spurious_rate_given_absent'] = d['n_e_spurious'] / (n - np_) if n > np_ else None
    d['n_present_mappable'] = len(mp); d['mappable_rate_given_present'] = len(mp) / np_ if np_ else None
    d['n_mappable_resolved'] = len(mpr)
    d['unresolved_rate_mappable'] = float((mp.cat == 'd_unresolved').mean()) if len(mp) else None
    for rel in ['exact', 'ancestor', 'descendant', 'sibling', 'other']:
        d[f'{rel}_rate'] = float((mpr.rel == rel).mean()) if len(mpr) else None
    d['exact_acc_strict_over_mappable_present'] = float((mp.cat == 'b_consistent').mean()) if len(mp) else None  # unresolved counts wrong
    d['mean_tree_distance'] = float(mpr.dist.dropna().mean()) if len(mpr) and mpr.dist.notna().any() else None
    for k in ['hp', 'hr', 'hf']: d['h' + k[1].upper() + '_mean'] = float(mpr[k].astype(float).mean()) if len(mpr) else None
    return d
rows = []
for ax in AXES:
    for ds in DATASETS + ['ALL']:
        g = C[(C.axis == ax) & ((C.dataset == ds) | (ds == 'ALL'))]
        rows.append({'axis': ax, 'dataset': ds, **agg(g)})
pa = pd.DataFrame(rows); pa['P_resolved_CI95'] = pa['P_resolved_CI95'].map(lambda x: json.dumps(x)); pa.to_csv(OUT / 'per_axis.csv', index=False)

ps = pd.DataFrame(per_sample)
# schema-convention note: parser copies object into target/source; human leaves them empty unless conversion
conv = sum(1 for r in per_sample if r['target_cat'] == 'e_spurious' and r['target_parser'] == r['object_parser'])
summary = {
    'proxy_warning': 'Human labels are single-annotator free text; node comparisons are a PROXY through a conservative exact-alias phrase->node mapping. True node-id labels: artifacts_cvpr_final/mapper_v2/node_label_template.csv (PENDING HUMAN ANNOTATION).',
    'n_unique_captions': len(lab), 'per_dataset_n': lab.groupby('dataset').size().to_dict(),
    'fully_mappable_captions': {ds: int(ps[ps.dataset == ds].fully_mappable.sum()) for ds in DATASETS} | {'ALL': int(ps.fully_mappable.sum())},
    'parser_config': dict(cfg['parser']), 'parser_stage_counts': parser.stage_counts,
    'target_spurious_that_copy_parser_object': conv,
    'per_axis_ALL': {r['axis']: {k: r[k] for k in ['n_present', 'P_resolved_given_present', 'P_resolved_CI95', 'n_present_mappable', 'mappable_rate_given_present', 'n_mappable_resolved', 'exact_rate', 'ancestor_rate', 'descendant_rate', 'sibling_rate', 'other_rate', 'unresolved_rate_mappable', 'exact_acc_strict_over_mappable_present', 'mean_tree_distance', 'hP_mean', 'hR_mean', 'hF_mean', 'n_spurious' if False else 'n_e_spurious']} for r in rows if r['dataset'] == 'ALL'},
    'per_axis_P_resolved_by_dataset': {ax: {r['dataset']: r['P_resolved_given_present'] for r in rows if r['axis'] == ax and r['dataset'] != 'ALL'} for ax in AXES},
}
json.dump(summary, open(OUT / 'summary.json', 'w'), indent=1, default=str)
json.dump({'parser': cfg['parser'], 'metric_weights': cfg['metric']['weights'], 'labels_csv': str(LABEL_CSV.relative_to(ROOT)),
           'parse_mode': 'HybridParser.parse(caption)[0], production defaults (exact/alias + semantic, no ngram/LLM/ancestor)', 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')},
          open(OUT / 'config.json', 'w'), indent=1)

# ---------------- annotation package ----------------
rows_ref = []
for k, tax in [('object', taxes['object']), ('action', taxes['action']), ('location', taxes['location']), ('quantity', taxes['quantity'])]:
    for nid, n in tax.nodes.items():
        rows_ref.append({'axis_taxonomy': k, 'node_id': nid, 'label': n.label, 'parent': n.parent, 'depth': tax.depth(nid),
                         'aliases': '; '.join(n.aliases), 'description': n.description, 'used_for_axes': ','.join(a for a, t in AXIS_TAX.items() if t == k)})
rows_ref.append({'axis_taxonomy': 'change_state', 'node_id': 'change', 'label': 'change', 'parent': '', 'depth': 0, 'aliases': '', 'description': 'a meaningful change occurred', 'used_for_axes': 'change_state'})
rows_ref.append({'axis_taxonomy': 'change_state', 'node_id': 'no_change', 'label': 'no_change', 'parent': '', 'depth': 0, 'aliases': '', 'description': 'caption states nothing changed', 'used_for_axes': 'change_state'})
ref = pd.DataFrame(rows_ref); ref.to_csv(ART / 'taxonomy_reference.csv', index=False)
md = ['# Taxonomy reference (valid node ids)\n', 'Generated from configs/ontology/*.yaml. Use the `node_id` column values only. Full table: taxonomy_reference.csv.\n']
for k, g in ref.groupby('axis_taxonomy', sort=False):
    md.append(f'\n## {k} taxonomy (axes: {g.used_for_axes.iloc[0]})\n\n| node_id | label | parent | aliases |\n|---|---|---|---|')
    for _, x in g.iterrows(): md.append(f"| {x.node_id} | {x.label} | {x.parent} | {x.aliases} |")
(ART / 'taxonomy_reference.md').write_text('\n'.join(md) + '\n')

# stratified 300: 100 per dataset, strata = human change_state x fully_mappable (proportional, seed 0, min 1)
rnd = random.Random(0); sel = []
for ds in DATASETS:
    g = ps[ps.dataset == ds].merge(lab[['id', 'change_state']], on='id')
    strata = {k: list(v.index) for k, v in g.groupby(['change_state', 'fully_mappable'])}
    alloc = {k: max(1, round(100 * len(v) / len(g))) for k, v in strata.items()}
    while sum(alloc.values()) > 100: alloc[max(alloc, key=alloc.get)] -= 1
    while sum(alloc.values()) < 100: alloc[max(alloc, key=alloc.get)] += 1
    for k, idxs in strata.items(): sel += [g.loc[i, 'id'] for i in rnd.sample(idxs, min(alloc[k], len(idxs)))]
tpl = lab[lab.id.isin(sel)][['id', 'dataset', 'image_id', 'caption']].copy()
tpl = tpl.sort_values(['dataset', 'id']).reset_index(drop=True)
for c in AXES + ['exact_count', 'notes', 'annotator_id']: tpl[c] = ''
tpl['status'] = 'PENDING HUMAN ANNOTATION'
tpl.to_csv(ART / 'node_label_template.csv', index=False)
(ART / 'INSTRUCTIONS.md').write_text(f"""# Node-level mapper labels: annotation instructions

Status: PENDING HUMAN ANNOTATION (no cell in node_label_template.csv has been filled by any tool).

File: node_label_template.csv, {len(tpl)} captions (100 per dataset; drawn from the 480 already free-text-labelled captions,
stratified by human change_state x whether the free-text labels were fully mappable to the taxonomy; seed 0).
The earlier free-text labels are intentionally NOT shown, so this is an independent annotation.

For each caption fill, per axis column, ONE node_id from taxonomy_reference.md / taxonomy_reference.csv:
- object, source, target: object taxonomy node ids. source/target only for conversion/replacement ("X replaced by Y": source=X, target=Y); leave empty otherwise.
- action: action taxonomy node id (choose the most specific node that is clearly supported; use a parent such as `change` if the caption is vague).
- location: location taxonomy node id (positions like left/center/upper_right; `image` if only "the scene/area").
- quantity: quantity node id; also write the integer in exact_count if the caption states one.
- change_state: `change` or `no_change`.
- Leave a cell EMPTY if the caption does not mention that field. If a mentioned concept has no fitting node, write NOT_IN_TAXONOMY:<your phrase> (this measures ontology gaps; do not force-fit).
- Choose the most specific node you are confident about; do not use the parser output. Fill annotator_id; ideally two annotators label the same rows independently so agreement can be computed.
- When done set status to ANNOTATED.
""")
print(json.dumps(summary['fully_mappable_captions']))
print(pa[pa.dataset == 'ALL'][['axis', 'n_present', 'P_resolved_given_present', 'n_present_mappable', 'exact_rate', 'n_e_spurious']].to_string())
