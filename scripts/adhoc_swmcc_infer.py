"""Batch inference wrapper around SWMCC's own infer.py code path
(build_model + model.generate), looping over a RICCE manifest's test
split instead of one image pair at a time. No model/generation code is
reimplemented -- imported directly from the SWMCC repo. Used to extend the
LEVIR-CC-trained checkpoint to new datasets (SECOND_CC, DUBAI_CC) for the
requested multi-dataset benchmark. Genuine cross-dataset
zero-shot inference -- the checkpoint was trained only on LEVIR-CC.
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SWMCC_REPO = Path(os.environ.get('SWMCC_REPO', ROOT / 'third_party' / 'SWMCC'))
sys.path.insert(0, str(SWMCC_REPO))

from swmcc.config import _to_namespace, load_config  # noqa: E402
from swmcc.data import PairedImageTransform  # noqa: E402
from swmcc.model import build_model  # noqa: E402
from swmcc.utils import select_device  # noqa: E402
from swmcc.vocabulary import Vocabulary  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--checkpoint", default=str(SWMCC_REPO / "outputs" / "swmcc_levir" / "best.pth"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--error-log", default=None)
    a = ap.parse_args()

    checkpoint = torch.load(a.checkpoint, map_location="cpu")
    config = _to_namespace(checkpoint["config"])
    vocabulary = Vocabulary.from_dict(checkpoint["vocabulary"])
    device = torch.device(a.device) if a.device else select_device(config.training.device)
    model = build_model(config, vocabulary).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    transform = PairedImageTransform(
        config.dataset.image_size, augment=False,
        mean=tuple(config.dataset.mean), std=tuple(config.dataset.std),
    )

    rows = [json.loads(l) for l in open(a.manifest)]
    rows = [r for r in rows if a.split == "all" or r["split"] == a.split]

    out = []
    errors = []
    t0 = time.time()
    with torch.no_grad():
        for i, r in enumerate(rows, 1):
            try:
                with Image.open(r["t1_path"]) as image_a, Image.open(r["t2_path"]) as image_b:
                    tensor_a, tensor_b = transform(image_a, image_b)
                generated = model.generate(
                    tensor_a.unsqueeze(0).to(device),
                    tensor_b.unsqueeze(0).to(device),
                    vocabulary.bos_id, vocabulary.eos_id,
                    config.dataset.max_caption_length,
                )
                pred = vocabulary.decode(generated[0].cpu().tolist())
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
