"""Sanity check: score a predictions JSONL with the upstream RSICC repo's own
eval_func (Bleu/Cider, pycocoevalcap-style) against LevirCCcaptions.json test
tokens, to separate 'reproduction is off' from 'RICCE CIDEr-D differs'."""
import sys, json, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'RSICC'))
from eval_func.bleu.bleu import Bleu
from eval_func.cider.cider import Cider
J = json.load(open(os.path.join(os.environ.get('RICCE_DATASET_ROOT', 'data'), 'LEVIR_CCaption', 'LevirCCcaptions.json')))
refs = {}
for im in J["images"]:
    if im["split"] == "test":
        refs[os.path.splitext(im["filename"])[0]] = [" ".join(s["tokens"]) for s in im["sentences"]]
def score(path):
    hyp = {json.loads(l)["image_id"]: json.loads(l)["prediction"].replace(".", " ").split() for l in open(path)}
    ids = sorted(hyp)
    gts = {k: refs[i] for k, i in enumerate(ids)}
    res = {k: [" ".join(hyp[i])] for k, i in enumerate(ids)}
    b, _ = Bleu(4).compute_score(gts, res); c, _ = Cider().compute_score(gts, res)
    return {"bleu4": b[3], "cider": c, "n": len(ids)}
for p in sys.argv[1:]:
    print(p, score(p))
