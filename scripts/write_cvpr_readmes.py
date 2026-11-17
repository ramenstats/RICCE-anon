"""Writes README.md / command.txt / config.json / summary.json for results_cvpr_final/{riccebench,ordering,ablations,properties}
and artifacts_cvpr_final/riccebench/README.md purely from files produced by the pipeline scripts (no hand-typed numbers)."""
import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results_cvpr_final'
ART = ROOT / 'artifacts_cvpr_final' / 'riccebench'
pd.set_option('display.width', 250)


def md_table(df, floatfmt='{:.4f}'):
    cols = list(df.columns)
    out = ['| ' + ' | '.join(map(str, cols)) + ' |', '|' + '---|' * len(cols)]
    for _, r in df.iterrows():
        out.append('| ' + ' | '.join(floatfmt.format(x) if isinstance(x, float) else str(x) for x in r) + ' |')
    return '\n'.join(out)


# ------------------------------------------------------------ benchmark
bs = json.load(open(ART / 'build_stats.json'))
items = [json.loads(l) for l in open(ART / 'items.jsonl')]
imp = [json.loads(l) for l in open(ART / 'impure.jsonl')]
rows = []
for fam, v in bs['purity_by_family'].items():
    a = v['ALL']
    rows.append(dict(family=fam, n_generated=a['n_generated'], n_pure=a['n_pure'], purity_rate=a['purity_rate'],
                     **{f'purity_{d}': (v[d]['purity_rate'] if v[d]['purity_rate'] is not None else float('nan')) for d in ('LEVIR_CC', 'SECOND_CC', 'DUBAI_CC')}))
pt = pd.DataFrame(rows)
pt.to_csv(RES / 'riccebench' / 'purity_by_family.csv', index=False)
src = {d: v['n_sources'] for d, v in bs['datasets'].items()}
strata = {}
for x in items:
    if x['perturbation_type'] == 'P0_identity':
        strata.setdefault(x['dataset'], {}).setdefault(x['source_stratum'], 0)
        strata[x['dataset']][x['source_stratum']] += 1
readme = f"""# RICCE-Bench

Controlled semantic minimal pairs built from REAL reference captions of LEVIR_CC, SECOND_CC and DUBAI_CC
(deterministic seed {bs['seed']}). Each item = (reference, candidate) where the candidate differs from the reference along ONE
intended semantic axis. Built by `python -m ricce.benchmarks.riccebench` (per dataset) then `--merge`.

Files: `items.jsonl` (PURE items, primary analysis: {bs['n_pure']}), `impure.jsonl` (items whose re-parse showed an unintended axis change or the
intended change was not realised: {bs['n_impure']}; excluded from primary analysis), `build_stats.json`, `parts/` (raw per-dataset output incl. impure).

## Item schema
`sample_id, source_id, dataset, source_stratum, reference, candidate, perturbation_type, severity_level, changed_axis, expected_relation,
source_event, candidate_event, purity_checks, perturbation_meta`. `source_id` (= dataset:image_id:ref_idx) is the cluster id used for bootstrap CIs.

## Sources
Unique (case/whitespace-normalised) reference captions are shuffled with the seed and parsed with `HybridParser.parse(..., allow_multi_event=True)`;
strata: `change` (single event, change), `nochange`, `multi` (>=2 events, real multi-event captions), `other` (parse resolves no change state; used only to top up to >=1000 sources).
Sources per dataset: {src}. Identity items per stratum: {strata}.
NOTE: LEVIR_CC has only a limited number of UNIQUE single-event change captions, so its `change` stratum is below the 1000 quota (all available were used) while total sources exceed 1000;
DUBAI_CC has only 1414 unique reference captions, hence 1000 sources incl. the `other` top-up.

## Families
P0_identity, P0_paraphrase (alias substitution), P1_near_ancestor / P2_distant_ancestor (object -> 1/2-hop ancestor; root uses alias 'land cover'),
P3_sibling_substitution (taxonomic sibling), P4_action_reversal, P5_location_swap (mirror location), P6_count_corruption, P7_event_omission (multi-event only),
P8_event_hallucination (append an unsupported event; change captions only), P9_change_state_reversal, unsupported_descendant_specificity (object -> child),
source_target_swap (conversion captions with distinct parsed source/target), negation_flip, multi_event_reorder (order-invariance control),
multi_event_source_target_ambiguity (swap the objects between the first two events: same bag of objects, wrong event binding; this is our operationalisation of source/target-role ambiguity at the event-binding level).
`severity_level` is an a-priori ordinal design assumption (0 identity/paraphrase/reorder, 1 near ancestor & descendant, 2 distant ancestor, 3 sibling/location/count, 4 omission/hallucination/binding, 5 reversal/contradiction), not a measured quantity.

## Purity validation
Reference and candidate are re-parsed; a perturbation is PURE iff every parsed field outside the family's allowed set is unchanged AND the intended change is realised (e.g. the object equals the intended ancestor, the reversal is detected by is_action_reversal). Set-level families check event multisets.
Purity is measured with the same parser the metric uses, so impurity also reflects parser noise (e.g. an alias collision such as 'built environment' containing the action alias 'built').

{md_table(pt)}

## Caveats
Perturbations are template-level text edits (fluency of edited captions is not guaranteed, e.g. plural agreement after object substitution). Purity is parser-relative, not human-verified.
"""
(ART / 'README.md').write_text(readme)
d = RES / 'riccebench'
d.mkdir(exist_ok=True, parents=True)
(d / 'README.md').write_text(readme.replace('Files:', 'Artifacts live in artifacts_cvpr_final/riccebench/. Files:'))
(d / 'command.txt').write_text('for d in LEVIR_CC SECOND_CC DUBAI_CC; do OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= python -m ricce.benchmarks.riccebench --datasets $d; done\npython -m ricce.benchmarks.riccebench --merge\npython scripts/write_cvpr_readmes.py\n')
json.dump(bs['args'], open(d / 'config.json', 'w'), indent=1)
json.dump(dict(n_pure=bs['n_pure'], n_impure=bs['n_impure'], sources=src, purity_by_family={r['family']: r['purity_rate'] for r in rows}, identity_items_by_stratum=strata), open(d / 'summary.json', 'w'), indent=1)

# ------------------------------------------------------------ ordering
o = RES / 'ordering'
m = pd.read_csv(o / 'ordering_accuracy_by_metric.csv')
core = m[(m.suites == 'core')]
allc = core[core.dataset == 'ALL'][['metric_label', 'n_constraints', 'n_source_captions', 'accuracy_strict', 'accuracy_strict_ci95_lo', 'accuracy_strict_ci95_hi', 'accuracy_tiehalf', 'tie_rate', 'monotonicity_violation_rate', 'kendall_tau_severity_ladder_mean', 'kendall_tau_severity_all_mean']].sort_values('accuracy_strict', ascending=False)
byds = core.pivot(index='metric_label', columns='dataset', values='accuracy_strict').reset_index()
inv = m[(m.suites == 'invariance') & (m.dataset == 'ALL')][['metric_label', 'n_constraints', 'accuracy_strict']]
fam = pd.read_csv(o / 'ordering_accuracy_by_family.csv')
famp = fam[fam.suite == 'core'].pivot(index='group', columns='metric_label', values='accuracy_strict').reset_index()
famp.to_csv(o / 'ordering_accuracy_by_family_pivot.csv', index=False)
cfg = dict(items='artifacts_cvpr_final/riccebench/items.jsonl', bootstrap='1000 resamples of source captions (cluster bootstrap), seed 0',
           subset_metrics=dict(spice='300 random source captions per dataset (seed 42), same ids for llm_judge; ids in subset_sample_ids.csv', llm_judge='Qwen2.5-3B-Instruct local, 1-5 rating'),
           cider_d='official pycocoevalcap CIDEr-D; IDF from the per-dataset multiset of benchmark references (each item = one id, one reference); reported per-item scores of that corpus-level run (label: CIDEr-D corpus-IDF per-item)',
           bertscore='roberta-large default, F1, no baseline rescaling, CPU', sbert='all-MiniLM-L6-v2 cosine', meteor='nltk meteor, single reference', bleu4='nltk sentence bleu4 smoothing method1, single reference',
           tie_semantics="'gt' constraints: strict counts ties as violations, tiehalf gives 0.5; 'ge' (identity>=paraphrase) ties satisfy in both; 'eq' (reorder invariance) tol 1e-9",
           extra_metric_hook='python scripts/merge_extra_metrics_into_ordering.py --extra CSV(sample_id,metric,score)')
json.dump(cfg, open(o / 'config.json', 'w'), indent=1)
(o / 'command.txt').write_text('python scripts/run_ordering_eval.py --metrics bleu4 meteor rougeL ricce sbert cider_d\npython scripts/run_ordering_eval.py --metrics bertscore\npython scripts/run_ordering_eval.py --metrics spice --chunk 500   # subset, checkpointed\npython scripts/run_ordering_eval.py --metrics llm_judge --chunk 256   # subset, checkpointed\npython scripts/run_ordering_eval.py --report-only\n')
json.dump({r.metric_label: dict(n_constraints=int(r.n_constraints), acc_strict=r.accuracy_strict, acc_tiehalf=r.accuracy_tiehalf, tie_rate=r.tie_rate) for r in allc.itertuples()}, open(o / 'summary.json', 'w'), indent=1)
(o / 'README.md').write_text(f"""# Pairwise semantic-ordering accuracy on RICCE-Bench (pure items)

Core constraints per source caption: identity >= paraphrase > near ancestor > distant ancestor; identity > each corrupted family
(near/distant ancestor, sibling, action reversal, location swap, count, omission, hallucination, change-state reversal, unsupported descendant, source/target swap, negation, binding swap).
Higher score = better for every metric. CI = 95% cluster bootstrap over source captions. ALL = LEVIR_CC + SECOND_CC + DUBAI_CC pooled.
SPICE and LLM-judge are computed on a fixed random subset (300 source captions per dataset), so their n_constraints is smaller; compare them with the full-data metrics with care.
CE-P alone and CE-R alone are shown separately on purpose: they are one-sided metrics and are expected to fail the constraints they are blind to (e.g. CE-P cannot see omission, CE-R cannot see hallucination).
Note that `identity > X` constraints are easy for any metric that gives identity its maximum; the discriminating constraints are the ancestor chain and the families where surface overlap is high
(action reversal, change-state reversal, negation, source/target swap, binding swap) -- see the per-family table.

## Headline (core suite, ALL datasets)
{md_table(allc)}

## Same-subset comparison (all metrics restricted to the SPICE/LLM-judge subset items; core suite)
{md_table(pd.read_csv(o / 'ordering_accuracy_subset_matched.csv')[['metric_label', 'n_constraints', 'accuracy_strict', 'accuracy_strict_ci95_lo', 'accuracy_strict_ci95_hi', 'accuracy_tiehalf']].sort_values('accuracy_strict', ascending=False))}

## Strict accuracy by dataset
{md_table(byds)}

## Order-invariance control (identity == reorder within 1e-9)
{md_table(inv)}

## Per-family strict accuracy (constraint group x metric); full CSV: ordering_accuracy_by_family.csv
{md_table(famp)}

## Caveats (read before quoting numbers)
- CIRCULARITY: the benchmark is generated on the RICCE ontology and purity is validated with the RICCE parser, so RICCE accuracy here measures internal consistency of the metric with its own ontology/parser, not independent validity. Independent validity needs the human study.
- The `identity > X` constraints are trivial for any metric that maps identity to its maximum (BLEU/ROUGE/SBERT/BERTScore/CIDEr-D all reach ~1.0 on identity); ties and the hard families are where metrics differ.
- CIDEr-D is the official corpus-level implementation with IDF from the benchmark reference multiset; the per-item numbers are NOT comparable in scale to CIDEr-D on a captioning test set.
- SPICE (pycocoevalcap): the existing `ricce.metrics.spice_llm_judge.spice_corpus` zips per-item scores in INPUT order while pycocoevalcap returns them in lexicographically SORTED id order; passing unsorted string ids silently misaligns scores. scripts/run_ordering_eval.py uses zero-padded ids to avoid this (the core file was not modified). Any earlier SPICE per-item numbers produced through spice_corpus with non-sorted ids should be re-checked.
- SPICE and LLM-judge use a random subset (see config.json); the LLM judge is a 3B local model and is noisy/discretised (1-5), which produces many ties.
- Two identity items (real captions 'All is the same.' / 'All remained the same.') parse to an all-None event and score 0 under RICCE (properties/FAILURES.md F3).

Files: ordering_accuracy_by_metric.csv (per dataset, per suite), ordering_accuracy_by_family.csv, per_item_scores_wide.csv, scores/*.csv (long: sample_id,metric,score), constraint_counts.csv.
Kendall tau columns: mean per-source Kendall tau-b between design severity and -score (+1 = score falls monotonically with severity); `ladder` uses identity/paraphrase/near/distant only.
Sibling-agent metrics: merge with scripts/merge_extra_metrics_into_ordering.py.
""")

# ------------------------------------------------------------ ablations
a = RES / 'ablations'
S = pd.read_csv(a / 'ablation_summary.csv')
sm = S[['variant', 'group', 'ord_strict_ALL', 'ord_strict_ALL_ci_lo', 'ord_strict_ALL_ci_hi', 'ord_tiehalf_ALL', 'tie_rate_ALL', 'mean_abs_item_shift_vs_A0', 'sys_tau_vs_A0_F', 'sys_tau_vs_A0_P', 'sys_tau_vs_A0_R', 'sys_rank_F']]
fm = pd.read_csv(a / 'family_means_by_variant.csv')
fm = fm[fm.dataset == 'ALL']
main_v = ['A0_full', 'A1_no_hierarchy', 'A2_no_event_decomposition', 'A3_no_event_binding', 'A4_no_gates', 'A5_no_source_target_distinction', 'A6_greedy_matching', 'A7_no_nochange_state', 'DEPTH_collapse1', 'DEPTH_collapse2']
fp = fm[fm.variant.isin(main_v)].pivot(index='family', columns='variant', values='ce_f')[main_v].reset_index()
a6 = json.load(open(a / 'a6_hungarian_vs_greedy.json'))
a6t = pd.DataFrame([dict(set=k, **{x: y for x, y in v.items() if x in ('n_pairs', 'n_hung_ne_greedy', 'frac_hung_ne_greedy', 'mean_abs_diff_greedy', 'max_abs_diff_greedy', 'mean_ce_f_hungarian', 'mean_ce_f_greedy', 'mean_ce_f_order', 'runtime_hungarian_s', 'runtime_greedy_s')}) for k, v in a6.items()])
ss = pd.read_csv(a / 'system_scores_by_variant.csv')
sst = ss[(ss.metric == 'CE-F') & ss.variant.isin(main_v)].pivot(index='variant', columns='system', values='score').reset_index()
(a / 'README.md').write_text(f"""# Ablations (Task II)

Scored on (i) RICCE-Bench pure items (n={len(items)}) and (ii) 5 real systems on LEVIR_CC test (per-image mean over 5 references).
A0 = full RICCE with multi-event decomposition. All variants are wrappers/monkeypatches in ricce/baselines/ablations.py (core files untouched).
Ordering accuracy = core-suite strict accuracy of per-item CE-F (95% cluster-bootstrap CI). tau = Kendall tau of the 5-system CE-F/CE-P/CE-R ranking vs A0 (5 systems -> coarse: one swap of adjacent systems = 0.8).

## Summary per variant
{md_table(sm)}

## Mean CE-F by perturbation family (ALL datasets)
{md_table(fp, '{:.3f}')}

## System CE-F (LEVIR_CC test)
{md_table(sst)}

## A6 Hungarian vs greedy vs order (real multi-event data + synthetic stress set; runtime = seconds for the whole set, mean of 3 repeats)
{md_table(a6t, '{:.5f}')}

Interpretation notes are in summary.json/CSV; files: ablation_summary.csv, family_means_by_variant.csv (per dataset), system_scores_by_variant.csv, depth_robustness.csv, depth_family_score_movement.csv, per_item_scores.csv.gz, a6_hungarian_vs_greedy.json, parse_cache.json.
Variant definitions: config.json. Weight variants: leave-one-axis-out (weight 0, gates unchanged), equal weights, action-/object-heavy (x3), each weight +-20%, 10 random joint +-20% draws (seed 0).
Depth: taxonomies collapsed programmatically (cap = max(1, maxdepth-k)); events mapped to surviving ancestors; hard gates use the uncollapsed ids so that the ablation isolates hierarchy resolution.
""")

# ------------------------------------------------------------ properties
p = RES / 'properties'
(p / 'README.md').write_text("""# Property tests (Task III)

- tests/test_metric_properties.py: bounds, identity, ancestor/descendant/sibling P/R behaviour, hallucination and omission monotonicity, order invariance, one-to-one matching, change/no-change and directional-reversal gates, missing-field asymmetry, root-only prediction, determinism.
- tests/test_counterexample_separate_pr.py + counterexample_separate_pr.md: collapsed harmonic scalar vs separate CE-P/CE-R.
- FAILURES.md: properties the CURRENT metric violates, with minimal reproducers (tests are xfail(strict=True), not weakened).
- Regenerate: python scripts/run_property_report.py
""")
print('ok')
