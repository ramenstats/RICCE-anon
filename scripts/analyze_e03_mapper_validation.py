"""E03 mapper-validation analysis (required check #2). Consumes a
ratings JSON exported from the annotation tool's "export" textarea
("Parser check" mode) -- real human labels, nothing simulated -- and
compares them against the actual HybridParser output on the same
captions to compute:
  - exact-node accuracy per axis (object/action/location/quantity/
    source/target/change_state)
  - ancestor-level accuracy per axis (credit if the parser's node is an
    ancestor of the human's gold node, i.e. "correct but coarse")
  - confusion matrices for action and location (item #9)
  - single-annotator test-retest agreement, using the blind duplicate
    items build_annotation_bundle.py inserted (id suffix "__retest")
    as an honest reliability proxy in place of unavailable multi-rater
    inter-annotator agreement

Usage: python scripts/analyze_e03_mapper_validation.py <ratings.json>
Until real ratings exist this prints a clear "no data yet" status
rather than fabricating any accuracy number.
"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.utils import read_jsonl
from ricce.ontology.tree import Taxonomy
from ricce.parsing.hybrid import HybridParser
import yaml

AXES = ['object', 'action', 'location', 'quantity', 'source', 'target', 'change_state']

def main(ratings_path):
    ratings_path = Path(ratings_path)
    if not ratings_path.exists():
        print(f'No ratings file at {ratings_path} yet -- export ratings from the annotation '
              f'tool (Parser check mode, export textarea) and pass that file here. '
              f'No accuracy numbers are computed or estimated until real labels exist.')
        return

    ratings = json.loads(ratings_path.read_text())
    e03 = {k.split('::', 1)[1]: v for k, v in ratings.items() if k.startswith('e03::')}
    if not e03:
        print('Ratings file has zero E03 ("Parser check") entries -- nothing to analyze yet.')
        return

    bundle = json.loads((ROOT / 'results' / 'E07' / 'annotation_bundle.json').read_text())
    items_by_id = {it['id']: it for it in bundle['e03']}

    cfg = yaml.safe_load(open(ROOT / 'configs' / 'project.yaml'))
    taxes = {k: Taxonomy.from_yaml(ROOT / v) for k, v in cfg['ontology'].items()}
    p = cfg['parser']
    parser = HybridParser(taxes['object'], taxes['action'], taxes['location'], taxes['quantity'],
                           use_embeddings=p.get('use_embeddings', True), embedding_model=p.get('embedding_model'),
                           similarity_threshold=float(p.get('similarity_threshold', 0.58)))

    exact_correct = Counter(); ancestor_correct = Counter(); total = Counter()
    confusion = {'action': defaultdict(Counter), 'location': defaultdict(Counter)}
    retest_pairs = []
    n_labeled = 0

    for item_id, gold in e03.items():
        if not any(gold.get(ax) for ax in AXES):
            continue  # rated=key exists but nothing filled in yet
        it = items_by_id.get(item_id)
        if it is None:
            continue
        if item_id.endswith('__retest'):
            orig_id = it.get('retest_of')
            if orig_id in e03:
                retest_pairs.append((orig_id, item_id))
            continue
        n_labeled += 1
        ev = parser.parse(it['caption'])[0]
        for ax in AXES:
            gold_val = (gold.get(ax) or '').strip().lower()
            if not gold_val:
                continue
            total[ax] += 1
            pred_val = getattr(ev, ax, None)
            pred_norm = (pred_val or '').strip().lower()
            if pred_norm == gold_val:
                exact_correct[ax] += 1; ancestor_correct[ax] += 1
            elif ax in ('object', 'action', 'location') and pred_val and gold_val in taxes.get(ax, taxes['object']).nodes:
                tax = taxes[ax] if ax in taxes else None
                if tax and pred_val in tax.nodes and gold_val in tax.nodes:
                    if pred_val in tax.ancestors(gold_val) or gold_val in tax.ancestors(pred_val):
                        ancestor_correct[ax] += 1
            if ax in confusion:
                confusion[ax][gold_val][pred_norm or '<none>'] += 1

    print(f'{n_labeled} fully-labeled E03 items found (of {len([k for k in e03 if not k.endswith("__retest")])} rated keys)')
    if n_labeled == 0:
        print('No axis fields filled in on any rated item -- nothing to compute yet.')
        return

    print('\nPer-axis accuracy (exact-node / ancestor-credited):')
    for ax in AXES:
        if total[ax] == 0:
            continue
        print(f'  {ax:14s} exact={exact_correct[ax]/total[ax]:.3f}  ancestor={ancestor_correct[ax]/total[ax]:.3f}  (n={total[ax]})')

    for ax in ('action', 'location'):
        if confusion[ax]:
            print(f'\nConfusion matrix ({ax}, gold -> predicted counts):')
            for gold_val, preds in confusion[ax].items():
                print(f'  {gold_val}: {dict(preds)}')

    if retest_pairs:
        agree = 0
        for orig_id, retest_id in retest_pairs:
            g1, g2 = e03[orig_id], e03[retest_id]
            match = all((g1.get(ax) or '').strip().lower() == (g2.get(ax) or '').strip().lower() for ax in AXES)
            agree += match
        print(f'\nSingle-annotator test-retest agreement: {agree}/{len(retest_pairs)} = {agree/len(retest_pairs):.3f} '
              f'(exact match across all axes on blind duplicate items)')
    else:
        print('\nNo retest-duplicate pairs rated yet -- test-retest reliability not yet computable.')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(__doc__); sys.exit(1)
    main(sys.argv[1])
