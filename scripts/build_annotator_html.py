#!/usr/bin/env python
"""Build the self-contained annotator HTML from items/pairs (no metrics, no system identity).
  --mode full|pilot   which item set   --embed   inline base64 JPEG thumbnails (single file, larger) instead of relative images/ files
"""
import argparse, base64, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts_cvpr_final/human_eval/image_grounded"
FORBID = {"system", "perturbation_type", "metrics", "answer", "source_reference"}

def rj(p): return [json.loads(l) for l in open(p)]

def src(rel, embed):
    if not embed: return rel
    return "data:image/jpeg;base64," + base64.b64encode((BASE / rel).read_bytes()).decode()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "pilot"], default="full")
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed-label", default="ce-human-v1")
    a = ap.parse_args()
    pre = "pilot_" if a.mode == "pilot" else ""
    items = rj(BASE / f"{pre}items.jsonl"); pairs = rj(BASE / f"{pre}pairwise_items.jsonl")
    for r in items + pairs:
        assert not (FORBID & set(r)), "annotator-facing record leaks hidden fields"
        r["t1_src"] = src(r["t1_thumb"], a.embed); r["t2_src"] = src(r["t2_thumb"], a.embed)
        for k in ("t1_path", "t2_path", "t1_thumb", "t2_thumb", "stratum", "dataset", "image_id"): r.pop(k, None)
    for r in items: r["references"] = r["references"][:2]
    for r in pairs: r["references"] = r["references"][:2]
    study = dict(study_id=f"{a.seed_label}-{a.mode}", seed=a.seed_label, mode=a.mode, items=items, pairs=pairs)
    html = (ROOT / "ricce/evaluation/annotator_template.html").read_text()
    html = html.replace("/*__DATA__*/null", json.dumps(study, ensure_ascii=False))
    out = Path(a.out) if a.out else BASE / ("annotator_pilot.html" if a.mode == "pilot" else "annotator.html")
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size/1e6:.2f} MB): {len(items)} Task-A items, {len(pairs)} Task-B pairs, embed={a.embed}")
if __name__ == "__main__":
    main()
