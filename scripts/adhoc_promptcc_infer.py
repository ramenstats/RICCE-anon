"""PromptCC (Chen-Yang-Liu/PromptCC) batch inference over a RICCE manifest.
Uses the official caption_beams.evaluate() unmodified (predicted-changeflag path,
beam=3 -- the repo demo defaults) with the released BEST_checkpoint_ViT-B_32 model.
The only patch is the one the README requires: CLIP VisionTransformer.forward must
also return the patch feature map (README 'NOTE' / upstream issue #3); applied here
as a runtime monkeypatch instead of editing site-packages."""
import argparse, json, os, sys, time, types
from pathlib import Path
REPO = Path(__file__).resolve().parents[1] / 'third_party' / 'PromptCC'
os.chdir(REPO); sys.path.insert(0, str(REPO))
import torch, clip
import clip.model as CM

def vit_forward(self, x):
    x = self.conv1(x)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
    x = x + self.positional_embedding.to(x.dtype)
    x = self.ln_pre(x)
    x = x.permute(1, 0, 2); x = self.transformer(x); x = x.permute(1, 0, 2)
    featuremap = x[:, 1:, :]          # patch tokens, before ln_post/proj (issue #3)
    x = self.ln_post(x[:, 0, :])
    if self.proj is not None:
        x = x @ self.proj
    return x, featuremap
CM.VisionTransformer.forward = vit_forward

import caption_beams as CB
from transformers import GPT2Tokenizer
import skimage.io as io
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True); ap.add_argument("--output", required=True)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--beam_size", type=int, default=3)
ap.add_argument("--split", default="test"); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
rows = [json.loads(l) for l in open(a.manifest)]
rows = [r for r in rows if r["split"] == a.split]
if a.limit: rows = rows[:a.limit]
dev = CB.device
tok = GPT2Tokenizer.from_pretrained("gpt2")
ck = torch.load(a.checkpoint, map_location=dev, weights_only=False)
model = ck["model_GPT"]; model.eval()
# checkpoint was pickled with transformers 4.10; newer versions expect this attribute
for m in model.modules():
    if type(m).__name__ == "GPT2Model" and not hasattr(m, "gradient_checkpointing"):
        m.gradient_checkpointing = False
    if type(m).__name__ == "GPT2Attention":
        for k, v in (("reorder_and_upcast_attn", False), ("scale_attn_by_inverse_layer_idx", False)):
            if not hasattr(m, k):
                setattr(m, k, v)
_, preprocess = clip.load("ViT-B/32", device="cpu", jit=False)
args = types.SimpleNamespace(dataset_name="LEVIR_CC", beam_size=a.beam_size, device=dev)
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
raw = open(a.output + ".raw.jsonl", "w"); out = open(a.output, "w"); t0 = time.time()
for i, r in enumerate(rows, 1):
    A = preprocess(Image.fromarray(io.imread(r["t1_path"]))).unsqueeze(0)
    B = preprocess(Image.fromarray(io.imread(r["t2_path"]))).unsqueeze(0)
    ori = torch.cat([A, B], 0).unsqueeze(0).to(dev)
    sent = CB.evaluate(args, ori, tok, model)
    pred = (sent or "").replace("Ġ", " ").strip()   # same detok as upstream demo, + strip
    raw.write(json.dumps({"image_id": r["image_id"], "raw": sent}) + "\n")
    out.write(json.dumps({"image_id": r["image_id"], "prediction": pred}) + "\n")
    if i % 100 == 0 or i == len(rows):
        out.flush(); raw.flush(); print(i, len(rows), time.time() - t0, flush=True)
