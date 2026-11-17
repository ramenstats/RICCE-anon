"""Cross-dataset parser/model-shift decomposition (v2, README S11).

Question: SFT's CE-F1 drops from 0.703 on LEVIR-CC (in-domain) to 0.349 on
SECOND-CC and 0.308 on Dubai-CC (cross-dataset zero-shot transfer, see
results/E08/{SECOND_CC,DUBAI_CC}/sft_seed42/metrics.json and
paper/tables_all.tex TblCrossDataset). How much of that drop is genuine
captioner domain-shift (the model describing the scene worse) vs. an
artifact of the parser/ontology's coverage being worse on SECOND-CC/Dubai-CC's
reference vocabulary than on LEVIR-CC's (a metric-side confound, not a
model-side one)?

Two-part measurement, both computed from real files (no estimation):

  Part 1 (parser/ontology coverage, isolated from captioner quality): run
  HybridParser (production config: exact/alias + semantic-retrieval stages,
  same as every other E-series result) over each dataset's own REFERENCE
  captions (not predictions) in artifacts/manifests/{LEVIR_CC,SECOND_CC,
  DUBAI_CC}.jsonl, and measure per-axis (object/action/location/quantity)
  resolution coverage plus overall "fully resolved" rate.

  Part 2 (stratify existing cross-dataset CE-F1 by parse quality): for
  SFT/Chg2Cap's SECOND-CC/Dubai-CC per_sample_metrics.csv, parse each
  prediction with the same parser and bucket into
  fully-parsed / partially-parsed / parser-failure, then report mean CE-F1
  per bucket per dataset. If most items are fully-parsed and CE-F1 is still
  low even in the fully-parsed bucket, the drop is captioner-side, not
  parser-side; if unresolved-heavy items dominate and drag the mean down,
  parser/ontology coverage is a real confound.
"""
import json
from pathlib import Path
from collections import Counter

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from ricce.utils import read_jsonl
from ricce.ontology.tree import Taxonomy
from ricce.parsing.hybrid import HybridParser

OUT_DIR = ROOT / "results_r1" / "cross_dataset_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

cfg = yaml.safe_load(open(ROOT / "configs" / "project.yaml"))
taxes = {k: Taxonomy.from_yaml(ROOT / v) for k, v in cfg["ontology"].items()}
parser_kwargs = dict(cfg["parser"])  # use_embeddings, embedding_model, similarity_threshold (production defaults)

DATASETS = ["LEVIR_CC", "SECOND_CC", "DUBAI_CC"]

# ---------------------------------------------------------------------------
# Part 1: reference-caption coverage per dataset (parser/ontology isolation)
# ---------------------------------------------------------------------------
print("Building HybridParser (production config: exact/alias + semantic retrieval)...")
parser = HybridParser(taxes["object"], taxes["action"], taxes["location"], taxes["quantity"], **parser_kwargs)

coverage_report = {}
for ds in DATASETS:
    manifest_path = ROOT / "artifacts" / "manifests" / f"{ds}.jsonl"
    rows = list(read_jsonl(manifest_path))
    # unique reference captions across the whole manifest (train+test), since
    # this measures ontology/parser coverage of the dataset's caption
    # vocabulary, not any particular split
    refs = set()
    for r in rows:
        for ref in (r.get("references") or []):
            refs.add(ref.strip())
    refs = sorted(refs)
    n = len(refs)
    counts = {"object": 0, "action": 0, "location": 0, "quantity": 0, "fully_resolved": 0, "no_change": 0}
    parser.stage_counts = {"exact": 0, "ngram": 0, "semantic": 0, "llm_fallback": 0, "ancestor_fallback": 0, "unresolved": 0}
    for cap in refs:
        ev = parser.parse(cap)[0]
        if ev.change_state == "no_change":
            counts["no_change"] += 1
            # no_change events intentionally leave object/action/location/quantity
            # unset by design (see hybrid.py _parse_single) -- treat separately,
            # not as unresolved coverage failures
            continue
        if ev.object is not None:
            counts["object"] += 1
        if ev.action is not None:
            counts["action"] += 1
        if ev.location is not None:
            counts["location"] += 1
        if ev.quantity is not None:
            counts["quantity"] += 1
        if ev.object is not None and ev.action is not None and ev.location is not None and ev.quantity is not None:
            counts["fully_resolved"] += 1
    n_non_nochange = n - counts["no_change"]
    coverage_report[ds] = {
        "n_unique_references": n,
        "n_no_change": counts["no_change"],
        "n_change_references": n_non_nochange,
        "object_coverage_rate": counts["object"] / n_non_nochange if n_non_nochange else None,
        "action_coverage_rate": counts["action"] / n_non_nochange if n_non_nochange else None,
        "location_coverage_rate": counts["location"] / n_non_nochange if n_non_nochange else None,
        "quantity_coverage_rate": counts["quantity"] / n_non_nochange if n_non_nochange else None,
        "fully_resolved_rate": counts["fully_resolved"] / n_non_nochange if n_non_nochange else None,
        "unresolved_token_rate": 1 - (counts["fully_resolved"] / n_non_nochange) if n_non_nochange else None,
        "resolution_stage_counts": dict(parser.stage_counts),
    }
    print(f"{ds}: n_unique_refs={n} (no_change={counts['no_change']}, change={n_non_nochange}); "
          f"fully_resolved_rate(of change refs)={coverage_report[ds]['fully_resolved_rate']:.3f}, "
          f"object={coverage_report[ds]['object_coverage_rate']:.3f}, "
          f"action={coverage_report[ds]['action_coverage_rate']:.3f}, "
          f"location={coverage_report[ds]['location_coverage_rate']:.3f}, "
          f"quantity={coverage_report[ds]['quantity_coverage_rate']:.3f}")

# ---------------------------------------------------------------------------
# Part 2: stratify existing cross-dataset CE-F1 by prediction parse quality
# ---------------------------------------------------------------------------
STRAT_SYSTEMS = ["sft_seed42", "chg2cap"]
strat_report = {}
for sysname in STRAT_SYSTEMS:
    strat_report[sysname] = {}
    for ds in ["SECOND_CC", "DUBAI_CC", "LEVIR_CC"]:
        csv_path = ROOT / "results" / "E08" / ds / sysname / "per_sample_metrics.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        bucket_f1 = {"fully_parsed": [], "partially_parsed": [], "parser_failure": []}
        for _, row in df.iterrows():
            pred = str(row["prediction"])
            ev = parser.parse(pred)[0]
            if ev.change_state == "no_change":
                bucket = "fully_parsed"  # no_change is a clean, deliberate resolution, not a failure
            else:
                n_resolved = sum(x is not None for x in [ev.object, ev.action, ev.location, ev.quantity])
                if n_resolved == 4:
                    bucket = "fully_parsed"
                elif n_resolved == 0:
                    bucket = "parser_failure"
                else:
                    bucket = "partially_parsed"
            bucket_f1[bucket].append(row["ce_f1"])
        summary = {}
        for b, vals in bucket_f1.items():
            summary[b] = {
                "n": len(vals),
                "mean_ce_f1": float(pd.Series(vals).mean()) if vals else None,
            }
        n_total = len(df)
        summary["overall"] = {"n": n_total, "mean_ce_f1": float(df["ce_f1"].mean())}
        strat_report[sysname][ds] = summary
        print(f"\n{sysname} / {ds} (n={n_total}, overall mean CE-F1={summary['overall']['mean_ce_f1']:.3f}):")
        for b in ["fully_parsed", "partially_parsed", "parser_failure"]:
            s = summary[b]
            print(f"    {b}: n={s['n']} ({s['n']/n_total:.1%}), mean CE-F1={s['mean_ce_f1']}")

# ---------------------------------------------------------------------------
# Decomposition verdict
# ---------------------------------------------------------------------------
verdict = {}
for sysname in STRAT_SYSTEMS:
    levir = strat_report[sysname].get("LEVIR_CC", {}).get("overall", {}).get("mean_ce_f1")
    for ds in ["SECOND_CC", "DUBAI_CC"]:
        if ds not in strat_report[sysname]:
            continue
        cross = strat_report[sysname][ds]
        overall = cross["overall"]["mean_ce_f1"]
        fp = cross["fully_parsed"]
        pct_fully_parsed = fp["n"] / cross["overall"]["n"] if cross["overall"]["n"] else None
        verdict[f"{sysname}_{ds}"] = {
            "levir_cc_ce_f1_in_domain": levir,
            "cross_dataset_ce_f1_overall": overall,
            "cross_dataset_ce_f1_fully_parsed_only": fp["mean_ce_f1"],
            "pct_predictions_fully_parsed": pct_fully_parsed,
            "interpretation": (
                "If fully_parsed-only mean CE-F1 is still far below the LEVIR-CC in-domain "
                "score, the drop is predominantly captioner/domain-shift, not a parser-coverage "
                "artifact (the parser is resolving these predictions fine; they are just scored "
                "as worse matches to the references). If fully_parsed-only CE-F1 is much closer "
                "to the LEVIR-CC score than the overall mean is, a meaningful share of the drop "
                "is attributable to parser/ontology coverage shift on this dataset's vocabulary."
            ),
        }
        print(f"\n{sysname} {ds}: LEVIR in-domain={levir}, cross-dataset overall={overall}, "
              f"cross-dataset fully-parsed-only={fp['mean_ce_f1']} "
              f"({pct_fully_parsed:.1%} of predictions fully parsed)" if pct_fully_parsed is not None else "")

report = {
    "part1_reference_coverage_by_dataset": coverage_report,
    "part2_prediction_stratification": strat_report,
    "decomposition_verdict": verdict,
    "methodology_note": (
        "Part 1 parses each dataset's own unique reference captions (manifest-wide, "
        "train+test) with the production HybridParser config (exact/alias + semantic "
        "retrieval, no LLM/ancestor fallback -- identical to every other E-series result "
        "in this project) to isolate ontology/parser coverage from captioner quality. "
        "no_change references are excluded from the coverage denominator since the "
        "parser deliberately leaves object/action/location/quantity unset for them by "
        "design (see ricce/parsing/hybrid.py _parse_single), not as a coverage gap. "
        "Part 2 re-parses the actual SFT/Chg2Cap SECOND-CC/Dubai-CC predictions from "
        "results/E08/*/*/per_sample_metrics.csv and stratifies their already-computed "
        "CE-F1 by whether the prediction itself fully resolved, partially resolved, or "
        "failed to resolve under the same parser."
    ),
}
with open(OUT_DIR / "report.json", "w") as f:
    json.dump(report, f, indent=2)
print(f"\nWrote {OUT_DIR / 'report.json'}")
