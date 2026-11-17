"""E09: split a frozen E08 benchmark by predicted vs reference change-state.

The canonical Sample.change_flag field is never populated by the dataset
adapters (no adapter infers it), so an ALL/CHANGE/NO-CHANGE split cannot be
read off the manifest directly. This script derives the reference change
state the same way the rest of the pipeline does: by parsing the reference
captions with the same frozen HybridParser/ontology used everywhere else.
No label is invented; the ontology's own "no_change" node is authoritative.
"""
import argparse
from pathlib import Path
from collections import Counter
import pandas as pd
from ricce.utils import read_jsonl, load_yaml, dump_json
from ricce.experiments import build_stack


def reference_change_state(parser, references):
    votes = Counter()
    for ref in references:
        ev = parser.parse(ref)[0]
        votes[ev.change_state or "unknown"] += 1
    if not votes:
        return "unknown"
    return votes.most_common(1)[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/project.yaml")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--per-sample", required=True, help="results/E08/<ds>/qwen/per_sample_metrics.csv")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    cfg = load_yaml(a.config)
    _, parser = build_stack(cfg)
    rows = {r["image_id"]: r for r in read_jsonl(a.manifest)}
    df = pd.read_csv(a.per_sample)
    df["ref_change_state"] = df["image_id"].map(
        lambda iid: reference_change_state(parser, rows[iid]["references"]) if iid in rows else "unknown"
    )

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "per_sample_with_state.csv", index=False)

    metric_cols = ["ce_precision", "ce_recall", "ce_f1"]
    out = {}
    for subset_name, sub in [
        ("ALL", df),
        ("CHANGE", df[df["ref_change_state"] == "change"]),
        ("NO_CHANGE", df[df["ref_change_state"] == "no_change"]),
        ("UNKNOWN", df[~df["ref_change_state"].isin(["change", "no_change"])]),
    ]:
        out[subset_name] = {
            "n": int(len(sub)),
            **{c: (float(sub[c].mean()) if len(sub) else float("nan")) for c in metric_cols},
        }
    dump_json(out, outdir / "metrics.json")
    print(f"Wrote {outdir/'metrics.json'}")


if __name__ == "__main__":
    main()
