"""Batch inference wrapper around Chg2Cap's own infer.py code path
(build_models_from_checkpoint + decoder.generate + ids_to_sentence), looping
over a RICCE manifest's test split instead of one image pair at a time.
No model/generation code is reimplemented -- it is imported directly from
the Chg2Cap repo. Reused to extend the LEVIR-CC-trained checkpoint to new
datasets (SECOND_CC, DUBAI_CC) for the requested multi-dataset
benchmark. This is genuine cross-dataset zero-shot inference: the checkpoint
was trained only on LEVIR-CC, so quality may be poor on other datasets --
that is expected and reported honestly, not papered over.
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CHG2CAP_REPO = Path(os.environ.get('CHG2CAP_REPO', ROOT / 'third_party' / 'Chg2Cap'))
sys.path.insert(0, str(CHG2CAP_REPO))

from data import LEVIR_MEAN, LEVIR_STD  # noqa: E402
from test import build_models_from_checkpoint  # noqa: E402
from utils import ids_to_sentence, invert_vocabulary, load_checkpoint, strip_special_tokens  # noqa: E402


def load_image(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32)
    array = (array - LEVIR_MEAN[None, None, :]) / LEVIR_STD[None, None, :]
    return torch.from_numpy(np.transpose(array, (2, 0, 1)).copy()).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--checkpoint", default=str(CHG2CAP_REPO / "models_checkpoint" / "BEST_LEVIR_CC.pth"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--decoding-method", choices=["greedy", "beam"], default="beam")
    ap.add_argument("--beam-size", type=int, default=3)
    ap.add_argument("--error-log", default=None)
    a = ap.parse_args()

    device = torch.device(a.device if a.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = load_checkpoint(a.checkpoint, device)
    encoder, attentive_encoder, decoder, word_vocab, config = build_models_from_checkpoint(checkpoint, device)
    index_to_word = invert_vocabulary(word_vocab)

    rows = [json.loads(l) for l in open(a.manifest)]
    rows = [r for r in rows if a.split == "all" or r["split"] == a.split]

    out = []
    errors = []
    t0 = time.time()
    with torch.no_grad():
        for i, r in enumerate(rows, 1):
            try:
                image_a = load_image(Path(r["t1_path"]), a.image_size).to(device)
                image_b = load_image(Path(r["t2_path"]), a.image_size).to(device)
                feature_a, feature_b = encoder(image_a, image_b)
                feature_a, feature_b = attentive_encoder(feature_a, feature_b)
                generated = decoder.generate(feature_a, feature_b, method=a.decoding_method, beam_size=a.beam_size)[0].tolist()
                token_ids = strip_special_tokens(generated, word_vocab)
                pred = ids_to_sentence(token_ids, index_to_word)
                out.append({"image_id": r["image_id"], "prediction": pred})
            except Exception as e:
                errors.append({"image_id": r.get("image_id"), "t1_path": r.get("t1_path"), "t2_path": r.get("t2_path"), "error": repr(e), "traceback": traceback.format_exc()})
            if i % 100 == 0 or i == len(rows):
                Path(a.output).parent.mkdir(parents=True, exist_ok=True)
                with open(a.output, "w") as f:
                    for row in out:
                        f.write(json.dumps(row) + "\n")
                print(f"{i}/{len(rows)} ok={len(out)} err={len(errors)} elapsed={time.time()-t0:.1f}s", flush=True)

    with open(a.output, "w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    if a.error_log:
        Path(a.error_log).parent.mkdir(parents=True, exist_ok=True)
        with open(a.error_log, "w") as f:
            json.dump(errors, f, indent=2)
    print(f"Wrote {len(out)} predictions to {a.output}; {len(errors)} errors")
    if len(out) == 0 and len(errors) > 0:
        print("FATAL: zero successful predictions, all samples errored")
        sys.exit(1)


if __name__ == "__main__":
    main()
