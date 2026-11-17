"""RSICCFormer (Chen-Yang-Liu/RSICC) batch inference over a RICCE manifest.
Uses the official caption.evaluate_transformer() unmodified (beam decoding, official
released checkpoint); only save_captions is monkeypatched to capture the token ids
of each pair instead of writing eval_results/*.json. Images are read with the
official imread (native 256x256 LEVIR-CC, /255, ImageNet normalize)."""
import argparse, json, os, sys, time, types
from pathlib import Path
REPO = Path(__file__).resolve().parents[1] / 'third_party' / 'RSICC'
os.chdir(REPO); sys.path.insert(0, str(REPO))
import torch
import caption as C

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True); ap.add_argument("--output", required=True)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--data_folder", required=True)
ap.add_argument("--data_name", default="LEVIR_CC_5_cap_per_img_5_min_word_freq")
ap.add_argument("--beam_size", type=int, default=1)
ap.add_argument("--split", default="test"); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.manifest)]
rows = [r for r in rows if r["split"] == a.split]
if a.limit: rows = rows[:a.limit]
ck = torch.load(a.checkpoint, map_location=C.device, weights_only=False)
enc_i, enc_f, dec = ck["encoder_image"], ck["encoder_feat"], ck["decoder"]
# torch>=2.0 nn.TransformerDecoder passes tgt_is_causal/memory_is_causal kwargs that the
# 2022 custom layer does not accept; drop them (masks are still passed as before).
for m in dec.modules():
    cls = type(m)
    if cls.__name__ == "Mesh_TransformerDecoderLayer" and not getattr(cls, "_compat", False):
        _f = cls.forward
        cls.forward = (lambda f: (lambda self, *args, **kw: f(self, *args, **{k: v for k, v in kw.items() if not k.endswith("_is_causal")})))(_f)
        cls._compat = True
word_map = json.load(open(os.path.join(a.data_folder, "WORDMAP_%s.json" % a.data_name)))
rev = {v: k for k, v in word_map.items()}
captured = []
C.save_captions = lambda args, wm, hyps: captured.extend(hyps)
args = types.SimpleNamespace(img_A=None, img_B=None, data_folder=a.data_folder, data_name=a.data_name,
                             beam_size=a.beam_size)
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
raw = open(a.output + ".rawids.jsonl", "w"); out = open(a.output, "w"); t0 = time.time()
for i, r in enumerate(rows, 1):
    args.img_A, args.img_B = r["t1_path"], r["t2_path"]
    captured.clear()
    C.evaluate_transformer(args, enc_i, enc_f, dec)
    ids = captured[0]
    pred = " ".join(rev[w] for w in ids)
    raw.write(json.dumps({"image_id": r["image_id"], "ids": ids}) + "\n")
    out.write(json.dumps({"image_id": r["image_id"], "prediction": pred}) + "\n")
    if i % 100 == 0 or i == len(rows):
        out.flush(); raw.flush(); print(i, len(rows), time.time() - t0, flush=True)
