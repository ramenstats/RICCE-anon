"""v3: one shared parse cache (frozen alias+embedding HybridParser,
configs/project.yaml) for every caption the v3 recomputation scores, so that
every evaluator variant compared in this round sees byte-identical events.

Texts: all LEVIR-CC references (all splits; reference-pair analyses), all
SECOND-CC / Dubai-CC test references, every prediction file under
results/baselines/*/<DS>_test.jsonl. Modes: 'multi' for everything; 'single'
additionally for LEVIR-CC test references and predictions (no-decomposition
ablation). Output: results_v3/parse_cache.json."""
import glob, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.utils import read_jsonl
from ricce.evaluation.parse_store import ParseStore

OUT = ROOT / 'results_v3'; OUT.mkdir(parents=True, exist_ok=True)
store = ParseStore(OUT / 'parse_cache.json')
multi, single = set(), set()
for ds in ('LEVIR_CC', 'SECOND_CC', 'DUBAI_CC'):
    for r in read_jsonl(ROOT / 'artifacts' / 'manifests' / f'{ds}.jsonl'):
        refs = [g for g in r.get('references', []) if g and g.strip()]
        if ds == 'LEVIR_CC' or r.get('split') == 'test':
            multi.update(refs)
        if ds == 'LEVIR_CC' and r.get('split') == 'test':
            single.update(refs)
    for f in glob.glob(str(ROOT / 'results' / 'baselines' / '*' / f'{ds}_test.jsonl')):
        for r in read_jsonl(f):
            t = str(r['prediction'])
            multi.add(t)
            if ds == 'LEVIR_CC':
                single.add(t)
print('texts multi', len(multi), 'single', len(single), flush=True)
print('parsed multi', store.ensure(multi, 'multi', n_proc=48), flush=True)
store.save()
print('parsed single', store.ensure(single, 'single', n_proc=48), flush=True)
store.save()
print('DONE')
