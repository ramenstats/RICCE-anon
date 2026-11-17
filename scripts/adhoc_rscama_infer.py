"""RSCaMa (Chen-Yang-Liu/RSCaMa) batch inference over a RICCE manifest.
Model construction, checkpoint loading, and decoding (decoder.sample(feat, k=1),
greedy as in the official test.py) are the official modules; the per-pair loop
replaces test.py's LEVIRCCDataset loader with the same CLIP preprocessing
(skimage.io.imread -> PIL -> clip preprocess). CLIP VisionTransformer.forward is
monkeypatched to also return patch features, as required by the README NOTE
(upstream PromptCC issue #3), instead of editing site-packages."""
import argparse, json, os, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1] / 'third_party' / 'RSCaMa'
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
    featuremap = x[:, 1:, :]
    x = self.ln_post(x[:, 0, :])
    if self.proj is not None:
        x = x @ self.proj
    return x, featuremap
CM.VisionTransformer.forward = vit_forward

from PIL import Image
import skimage.io as io
from model.model_encoder_attMamba import Encoder, AttentiveEncoder
from model.model_decoder import DecoderTransformer

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True); ap.add_argument("--output", required=True)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--vocab", required=True)  # produced by official preprocess_data.py (threshold 5) from LevirCCcaptions.json
ap.add_argument("--split", default="test"); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
rows = [json.loads(l) for l in open(a.manifest)]
rows = [r for r in rows if r["split"] == a.split]
if a.limit: rows = rows[:a.limit]

word_vocab = json.load(open(a.vocab))
ck = torch.load(a.checkpoint, map_location="cpu")
# defaults of official test.py for --network CLIP-ViT-B/32
network, encoder_dim, feat_size, n_heads, n_layers, dec_layers, dropout, max_length = "CLIP-ViT-B/32", 768, 7, 8, 3, 1, 0.1, 42
encoder = Encoder(network)
encoder_trans = AttentiveEncoder(n_layers=n_layers, feature_size=[feat_size, feat_size, encoder_dim], heads=n_heads, dropout=dropout)
decoder = DecoderTransformer(decoder_type="transformer_decoder", embed_dim=encoder_dim, vocab_size=len(word_vocab),
                             max_lengths=max_length, word_vocab=word_vocab, n_head=n_heads, n_layers=dec_layers, dropout=dropout)
encoder.load_state_dict(ck["encoder_dict"]); encoder_trans.load_state_dict(ck["encoder_trans_dict"]); decoder.load_state_dict(ck["decoder_dict"])
encoder, encoder_trans, decoder = [m.eval().cuda() for m in (encoder, encoder_trans, decoder)]
_, preprocess = clip.load("ViT-B/32", device="cpu", jit=False)
idx2w = {v: k for k, v in word_vocab.items()}
except_tokens = {word_vocab["<START>"], word_vocab["<END>"], word_vocab["<NULL>"]}
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
raw = open(a.output + ".rawids.jsonl", "w"); out = open(a.output, "w"); t0 = time.time()
with torch.no_grad():
    for i, r in enumerate(rows, 1):
        A = preprocess(Image.fromarray(io.imread(r["t1_path"]))).unsqueeze(0).cuda()
        B = preprocess(Image.fromarray(io.imread(r["t2_path"]))).unsqueeze(0).cuda()
        f1, f2 = encoder(A, B); feat = encoder_trans(f1, f2)
        seq = decoder.sample(feat, k=1)
        pred_seq = [w for w in seq if w not in except_tokens]
        pred = " ".join(idx2w[w] for w in pred_seq)
        raw.write(json.dumps({"image_id": r["image_id"], "ids": [int(w) for w in seq]}) + "\n")
        out.write(json.dumps({"image_id": r["image_id"], "prediction": pred}) + "\n")
        if i % 100 == 0 or i == len(rows):
            out.flush(); raw.flush(); print(i, len(rows), time.time() - t0, flush=True)
