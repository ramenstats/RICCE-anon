from collections import Counter
from pathlib import Path
import re
from ..utils import dump_json

def audit_samples(samples,out_json=None):
    splits=Counter(s.split for s in samples); refs=[r for s in samples for r in s.references]; words=Counter()
    for r in refs: words.update(re.findall(r'[a-zA-Z]+',r.lower()))
    report={
      'num_samples':len(samples),'splits':dict(splits),
      'samples_with_references':sum(bool(s.references) for s in samples),
      'caption_count':len(refs),'mean_references_per_pair':len(refs)/max(1,len(samples)),
      'missing_t1':sum(not Path(s.t1_path).exists() for s in samples),
      'missing_t2':sum(not Path(s.t2_path).exists() for s in samples),
      'top_words':words.most_common(100)}
    if out_json: dump_json(report,out_json)
    return report
