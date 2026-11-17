#!/usr/bin/env python
"""RICCE: one entry point for every experiment of the paper.

Run from anywhere (paths are resolved relative to the RICCE project root,
the parent of this REPRODUCE/ folder), inside the `ricce` conda env:

    python REPRODUCE/reproduce_ricce.py list              # every stage, in order
    python REPRODUCE/reproduce_ricce.py check             # are inputs/tools present?
    python REPRODUCE/reproduce_ricce.py run               # DEFAULT: final evaluator, tables and figures (phases v3, paper)
    python REPRODUCE/reproduce_ricce.py run --phase upstream --allow-gpu
    python REPRODUCE/reproduce_ricce.py run --stages v3_systems v3_tables
    python REPRODUCE/reproduce_ricce.py run --from v3_tables
    python REPRODUCE/reproduce_ricce.py run --dry-run     # print commands only

Phases, in pipeline order (see REPRODUCE/README.md):
  setup     unit and property tests
  data      dataset discovery + manifests (needs the raw datasets)
  models    caption predictions: Qwen zero-shot, LoRA SFT/DPO/TCPO training,
            swap augmentation, external RSICC baselines (GPU / external envs)
  upstream  canonical benchmark, ordering suites, flat baselines, mapper,
            earlier evaluator rounds whose files the final analysis reads
  v3        every number of the paper under the FINAL evaluator
  paper     tables and figures generated from the result files
  extra     analyses not in the manuscript (hierarchy fit, WordNet comparison)
  human     human-annotation studies: annotation itself needs real annotators (never simulated);
            quality control, key re-scoring and analysis of the collected files run on CPU

Stage kinds:
  cpu       executed by `run`
  gpu       executed only with --allow-gpu (uses the GPU or a local LLM)
  external  needs another conda env / third-party repo / hand action: commands
            are printed, never executed by this script
  human     needs real human annotations: instructions are printed, never executed

Re-running a phase overwrites its outputs. The default `run` re-derives the
paper's numbers from the existing model predictions and upstream results; it
does not retrain models or re-run upstream stages.
"""
import argparse, datetime, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]           # RICCE project root
PY = sys.executable
TECT = os.environ.get('TECTONIC') or shutil.which('tectonic') or os.path.expanduser('~/tectonic_env/bin/tectonic')
DATASET_ROOT = os.environ.get('RICCE_DATASET_ROOT', 'data')
DS = ['LEVIR_CC', 'SECOND_CC', 'DUBAI_CC']
SEEDS = [1, 42, 123]
CPU_ENV = 'CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 '


def S(id, phase, kind, title, cmds=(), needs=(), makes=(), paper='', note=''):
    return dict(id=id, phase=phase, kind=kind, title=title, cmds=list(cmds), needs=list(needs), makes=list(makes), paper=paper, note=note)


STAGES = [
    # ------------------------------------------------------------------ setup
    S('tests', 'setup', 'cpu', 'Unit, property and regression tests (evaluator, parser, baselines, human-analysis code)',
      [f'{PY} -m pytest -q tests'], needs=['tests', 'ricce', 'third_party']),

    # ------------------------------------------------------------------ data
    S('data_discover', 'data', 'cpu', 'Locate the raw datasets (LEVIR-CC, SECOND-CC, Dubai-CC; RSCC has no captions)',
      [f'{PY} -m ricce.cli discover --dataset-root {DATASET_ROOT}']),
    S('data_prepare', 'data', 'cpu', 'Build artifacts/manifests/<DS>.jsonl (image paths + reference captions) and audits',
      [f'{PY} -m ricce.cli prepare --config configs/project.yaml'], needs=['configs/project.yaml'],
      makes=[f'artifacts/manifests/{d}.jsonl' for d in DS]),

    # ------------------------------------------------------------------ models (predictions)
    S('qwen_zero_shot', 'models', 'gpu', 'Frozen Qwen2.5-VL-7B-Instruct zero-shot captions on the three test sets',
      [f'{PY} -m ricce.cli vlm-infer artifacts/manifests/{d}.jsonl results/baselines/qwen/{d}_test.jsonl --split test' for d in DS],
      needs=['artifacts/manifests/LEVIR_CC.jsonl'], makes=[f'results/baselines/qwen/{d}_test.jsonl' for d in DS]),
    S('sft_train', 'models', 'gpu', 'LoRA SFT of Qwen2.5-VL on LEVIR-CC, seeds 1/42/123, then test-set predictions',
      [c for s in SEEDS for c in (f'{PY} -m ricce.training.sft --manifest artifacts/manifests/LEVIR_CC.jsonl --output checkpoints/LEVIR_CC/sft_seed{s} --seed {s}',
                                   f'bash scripts/score_variant.sh checkpoints/LEVIR_CC/sft_seed{s} sft_seed{s}')],
      makes=[f'results/baselines/sft_seed{s}/LEVIR_CC_test.jsonl' for s in SEEDS], note='Hours per seed on one H100.'),
    S('sft_cross_dataset', 'models', 'gpu', 'SFT (seed 42) zero-shot transfer to SECOND-CC and Dubai-CC',
      [f'bash scripts/score_variant_cross.sh checkpoints/LEVIR_CC/sft_seed42 sft_seed42 {d}' for d in ('SECOND_CC', 'DUBAI_CC')],
      makes=['results/baselines/sft_seed42/SECOND_CC_test.jsonl', 'results/baselines/sft_seed42/DUBAI_CC_test.jsonl']),
    S('swap_augmentation', 'models', 'gpu', 'Swap-augmented SFT and the step-matched control (sft2x), 3 seeds each',
      ['bash scripts/run_temporal_swap_sft.sh', 'bash scripts/run_temporal_swap_sft2x.sh'],
      makes=[f'results/baselines/{v}_seed{s}/LEVIR_CC_test.jsonl' for v in ('swapsft', 'sft2x') for s in SEEDS]),
    S('preference_optimization', 'models', 'gpu', 'DPO / TCPO: original, stabilized and matched-control severity variants (25 stabilized runs)',
      ['bash scripts/run_training_variants.sh', 'bash scripts/run_branch_a.sh'],
      makes=['results/baselines/tcpo_v3_seed42/LEVIR_CC_test.jsonl'], note='Multi-day; resumable.'),
    S('temporal_swap_e10', 'models', 'gpu', 'T1/T2 swap test of the frozen model (E10)',
      [f'{PY} scripts/run_e10_temporal_swap.py --manifest artifacts/manifests/LEVIR_CC.jsonl --outdir results/E10/LEVIR_CC'],
      makes=['results/E10/LEVIR_CC/metrics.json']),
    S('external_baselines', 'models', 'external', 'Chg2Cap, SWMCC, RSICCformer, PromptCC, RSCaMa predictions (official code + checkpoints)',
      ['python scripts/adhoc_chg2cap_infer.py --manifest artifacts/manifests/<DS>.jsonl --output results/baselines/chg2cap/<DS>_test.jsonl',
       'python scripts/adhoc_swmcc_infer.py   --manifest artifacts/manifests/<DS>.jsonl --output results/baselines/swmcc/<DS>_test.jsonl',
       'python scripts/adhoc_rsiccformer_infer.py --manifest artifacts/manifests/LEVIR_CC.jsonl --output results/baselines/rsiccformer/LEVIR_CC_test.jsonl --checkpoint <ckpt> --data_folder <dir>',
       'python scripts/adhoc_promptcc_infer.py --manifest artifacts/manifests/LEVIR_CC.jsonl --output results/baselines/promptcc/LEVIR_CC_test.jsonl --checkpoint <ckpt>',
       'python scripts/adhoc_rscama_infer.py   --manifest artifacts/manifests/LEVIR_CC.jsonl --output results/baselines/rscama/LEVIR_CC_test.jsonl --checkpoint <ckpt> --vocab <vocab>'],
      makes=[f'results/baselines/{s}/LEVIR_CC_test.jsonl' for s in ('chg2cap', 'swmcc', 'rsiccformer', 'promptcc', 'rscama')],
      note='Each runs inside its own environment (third_party/ repos; conda envs rsicc_fmr / mamba; checkpoints in checkpoints_external/). '
           'Exact settings: results/baselines/BASELINE_GENERATION_REPORT*.md and results/baselines/<system>/run.log.'),

    # ------------------------------------------------------------------ upstream analyses
    S('bench_build', 'upstream', 'cpu', 'RICCE-Bench: 16 perturbation families, purity validation (35,514 items)',
      [CPU_ENV.replace('OMP_NUM_THREADS=4', 'OMP_NUM_THREADS=1') + f'{PY} -m ricce.benchmarks.riccebench --datasets {d}' for d in DS]
      + [f'{PY} -m ricce.benchmarks.riccebench --merge', f'{PY} scripts/write_cvpr_readmes.py'],
      makes=['artifacts_cvpr_final/riccebench/items.jsonl']),
    S('ordering_string_metrics', 'upstream', 'cpu', 'Ordering scores for BLEU-4, METEOR, ROUGE-L, CIDEr-D, SBERT (and the canonical RICCE)',
      [f'{PY} scripts/run_ordering_eval.py --metrics bleu4 meteor rougeL ricce sbert cider_d'],
      needs=['artifacts_cvpr_final/riccebench/items.jsonl'], makes=['results_cvpr_final/ordering/scores']),
    S('ordering_model_metrics', 'upstream', 'gpu', 'Ordering scores for BERTScore, SPICE (needs Java) and the LLM judge (subsets)',
      [f'{PY} scripts/run_ordering_eval.py --metrics bertscore', f'{PY} scripts/run_ordering_eval.py --metrics spice --chunk 500',
       f'{PY} scripts/run_ordering_eval.py --metrics llm_judge --chunk 256']),
    S('ordering_report', 'upstream', 'cpu', 'Ordering accuracy tables and the semantic-validity suite',
      [f'{PY} scripts/run_ordering_eval.py --report-only', f'{PY} scripts/run_semantic_validity_ordering.py'],
      makes=['results_cvpr_final/ordering/ordering_accuracy_by_metric.csv', 'results_cvpr_final/ordering/semantic_validity_ordering.csv']),
    S('flat_baselines', 'upstream', 'cpu', 'Kiritchenko hP/hR, LCA-F (HEMKit), HCM, TaxAware on identical parsed labels',
      [f'{PY} scripts/build_baseline_inputs.py', f'{PY} scripts/build_binding_diagnostics.py', 'bash scripts/run_baselines_all.sh',
       f'{PY} scripts/validate_lca_python.py', f'{PY} scripts/analyze_baselines.py'],
      makes=['results_cvpr_final/baselines/diagnostics_summary.csv']),
    S('mapper', 'upstream', 'cpu', 'Mapper (parser) validation and cascade ablations',
      [CPU_ENV + f'{PY} scripts/run_mapper_v2.py', CPU_ENV + f'{PY} scripts/run_mapper_cascade_ablation.py',
       CPU_ENV + f'{PY} scripts/run_mapper_revised_validation.py'],
      makes=['results_cvpr_final/mapper_v2/per_axis.csv', 'results_metric_v2/mapper_revised/before_after_comparison.csv']),
    S('mapper_llm_stage', 'upstream', 'gpu', 'Constrained-JSON LLM fallback of the mapper on a 250-image subset',
      [f'{PY} scripts/run_mapper_llm_stage.py'], makes=['results_cvpr_final/mapper_cascade/llm_stage_subset250.csv']),
    S('parser_error', 'upstream', 'cpu', 'Parser-error propagation against human-corrected labels (480 captions)',
      [CPU_ENV + f'{PY} scripts/run_parser_error_propagation.py'], makes=['results_cvpr_final/parser_error/summary.json']),
    S('multievent_and_gate_stress', 'upstream', 'cpu', 'Real multi-event caption scan and contradiction-gate stress items',
      [f'{PY} scripts/build_multi_event_benchmark.py', f'{PY} scripts/run_gate_stress_final.py'],
      makes=['artifacts_r1/multi_event_benchmark.jsonl']),
    S('canonical_systems', 'upstream', 'cpu', 'Frozen (earlier) evaluator system scores and swap statistics',
      [f'{PY} scripts/run_canonical_system_scores.py --systems qwen,sft_seed42,chg2cap,swmcc,rsiccformer,promptcc,rscama',
       f'{PY} scripts/run_swap_canonical_stats.py', f'{PY} scripts/run_swap_action_family_breakdown.py'],
      makes=['results_cvpr_final/system_benchmark/summary.csv', 'results_cvpr_final/temporal_equivariance/summary.json',
             'results_metric_v2/swap_action_family_breakdown.json']),
    S('previous_evaluator_round', 'upstream', 'cpu', 'Previous evaluator round (v2): files the final analysis uses for basis checks',
      [f'{PY} scripts/run_v2_system_benchmark.py', f'{PY} scripts/run_v2_cross_dataset.py', f'{PY} scripts/run_v2_rescore_baselines.py',
       f'{PY} scripts/run_v2_human_pilot.py', f'{PY} scripts/run_v2_parser_error_propagation.py', CPU_ENV + f'{PY} scripts/run_cross_dataset_v2.py'],
      makes=['results_v2/system_benchmark_v2/summary.csv', 'results_v2/cross_dataset_v2/summary_v2.json']),
    S('properties_and_runtime', 'upstream', 'cpu', 'Property report, canonical ablations, CPU runtime',
      [f'{PY} scripts/run_property_report.py', f'{PY} scripts/run_ablations_cvpr.py',
       'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ' + f'{PY} scripts/run_runtime.py --skip-llm'],
      makes=['results_cvpr_final/runtime/summary_cpu.json']),
    S('runtime_llm', 'upstream', 'gpu', 'Runtime of the LLM components',
      ['OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ' + f'{PY} scripts/run_runtime.py'], makes=['results_cvpr_final/runtime/summary_llm.json']),
    S('teaser_inputs', 'upstream', 'external', 'Teaser (Fig. 1) conventional-metric scores',
      [], makes=['results_cvpr_final/teaser/teaser_scores.json'],
      note='No producing script exists in the repository; the saved file is used as-is. The RICCE scores of the teaser are recomputed by v3_bench_ordering_teaser.'),

    # ------------------------------------------------------------------ v3: final evaluator
    S('v3_parse_cache', 'v3', 'cpu', 'Parse every caption once with the frozen parser (shared by every v3 analysis)',
      [CPU_ENV + f'{PY} scripts/v3_build_parse_cache.py'], note='Incremental: parses only captions not yet in the cache. Delete results_v3/parse_cache.json to re-parse everything.', needs=['results/baselines', 'artifacts/manifests/LEVIR_CC.jsonl'],
      makes=['results_v3/parse_cache.json']),
    S('v3_systems', 'v3', 'cpu', 'System benchmark, learning runs, cross-dataset strata, natural multi-event pairs (final evaluator)',
      [CPU_ENV + f'{PY} scripts/v3_recompute_systems.py'], needs=['results_v3/parse_cache.json', 'results_v2/system_benchmark_v2/summary.csv', 'third_party'],
      makes=['results_v3/systems/summary.csv', 'results_v3/systems/summary.json']),
    S('v3_ordering_teaser', 'v3', 'cpu', 'RICCE-Bench ordering numbers, heatmap column and teaser scores (final evaluator)',
      [CPU_ENV + f'{PY} scripts/v3_bench_ordering_teaser.py'],
      needs=['artifacts_cvpr_final/riccebench/items.jsonl', 'results_cvpr_final/ordering/scores/ricce.csv', 'results_cvpr_final/teaser/teaser_scores.json'],
      makes=['results_v3/ordering/summary.json', 'results_v3/teaser/teaser_final.json']),
    S('v3_root_mode', 'v3', 'cpu', 'Design choice 2: how a root-level label is scored (properties R1-R4)',
      [CPU_ENV + f'{PY} scripts/v3_root_mode_comparison.py'], makes=['results_v3/root_mode/summary_bench_properties.json']),
    S('v3_gate_scope', 'v3', 'cpu', 'Design choice 3: object-scoped contradiction gate (criteria C1-C4)',
      [CPU_ENV + f'{PY} scripts/v3_gate_object_match.py'], makes=['results_v3/gate_object_match']),
    S('v3_gate', 'v3', 'cpu', 'Gate activations measured directly; activation breakdown',
      [CPU_ENV + f'{PY} scripts/v3_gate_analysis.py', CPU_ENV + f'{PY} scripts/v3_gate_activation_breakdown.py'],
      makes=['results_v3/gate/summary.json']),
    S('v3_ablations', 'v3', 'cpu', 'Design ablations A0-A11 under the final evaluator',
      [CPU_ENV + f'{PY} scripts/v3_ablations.py'], makes=['results_v3/ablations/ablation_summary.csv']),
    S('v3_dual_vs_joint', 'v3', 'cpu', 'Design choice 1: dual vs joint event assignment (monotonicity violations)',
      [CPU_ENV + f'{PY} scripts/v3_dual_vs_joint_matching.py'], makes=['results_v3/dual_vs_joint']),
    S('v3_flat_baselines', 'v3', 'cpu', 'Flat-vs-event comparison rescored with the final evaluator',
      [CPU_ENV + f'{PY} scripts/v3_rescore_baselines.py'], needs=['artifacts_cvpr_final/baselines/inputs'],
      makes=['results_v3/baselines/diagnostics_summary_v2.csv']),
    S('v3_human_pilot', 'v3', 'cpu', 'Single-annotator text-only pilot correlations (existing real ratings)',
      [CPU_ENV + f'{PY} scripts/v3_human_pilot.py'], needs=['results/E07_human_pred_validation/per_item.json'],
      makes=['results_v3/human_pilot/per_item.csv']),
    S('v3_parser_error', 'v3', 'cpu', 'Parser-error propagation under the final evaluator',
      [CPU_ENV + f'{PY} scripts/v3_parser_error_propagation.py'], makes=['results_v3/parser_error/summary.json']),
    S('v3_refagg_cascade', 'v3', 'cpu', 'Reference-aggregation protocols and the non-LLM mapper cascade',
      [CPU_ENV + f'{PY} scripts/v3_refagg_cascade.py'], makes=['results_v3/refagg_cascade']),
    S('v3_natural_multievent', 'v3', 'cpu', 'Statistics of the 759 real multi-event pairs per system',
      [CPU_ENV + f'{PY} scripts/v3_natural_multievent_stats.py'], needs=['results_v3/systems/natural_multievent_pairs.csv'],
      makes=['results_v3/natural_multievent/summary.json']),

    # ------------------------------------------------------------------ paper
    S('figures', 'paper', 'cpu', 'Fig. 2 (ordering heatmap) and Fig. 3 (cross-dataset) from result files',
      [f'{PY} scripts/v3_make_figures.py'], makes=['paper_tables/fig/ordering_heatmap_v3.pdf', 'paper_tables/fig/cross_dataset_v3.pdf']),
    S('tables', 'paper', 'cpu', 'Every table of the main paper and supplement, from result files only',
      [CPU_ENV + f'{PY} scripts/v3_build_tables.py'], makes=['paper_tables/tables_v3']),
    # ------------------------------------------------------------------ extra (not in the manuscript)
    S('hierarchy_fit', 'extra', 'cpu', 'How captions map onto the hierarchy; per-axis prediction-vs-reference scores',
      [CPU_ENV + f'{PY} scripts/v3_hierarchy_fit.py'], makes=['results_v3/hierarchy_fit/summary.json']),
    S('wordnet_comparison', 'extra', 'cpu', 'WordNet hierarchy vs the RICCE ontology (all scenes; changed scenes only)',
      [CPU_ENV + f'{PY} scripts/v3_wordnet_hierarchy.py', CPU_ENV + f'{PY} scripts/v3_wordnet_hierarchy.py changed_only'],
      makes=['results_v3/wordnet_hierarchy/T2_matched_vs_shuffled.csv'], note='Needs NLTK WordNet data.'),

    # ------------------------------------------------------------------ human annotation
    S('human_image_grounded', 'human', 'human', 'Image-grounded study, 3 independent annotators (977 items + 477 pairs, or the pilot subset)',
      ['python scripts/make_annotation_sheets.py --check artifacts_cvpr_final/human_eval/image_grounded/annotation_sheets/<file>.csv',
       'cp <validated files> artifacts_cvpr_final/human_eval/image_grounded/collected/',
       'python scripts/analyze_human_eval.py'],
      note='Protocol: artifacts_cvpr_final/human_eval/image_grounded/README.md and annotation_sheets/ANNOTATOR_BRIEFING.md. '
           'Done for the 2026-10-29 files; the analysis stages below use them.'),
    S('human_eval_qc', 'human', 'cpu', 'Quality control of the three annotation files (format, repeated patterns, face validity, agreement, copy test)',
      [f'{PY} scripts/v3_human_eval_qc.py'], needs=[f'artifacts_cvpr_final/human_eval/image_grounded/annotation_sheets/annotations_ann{k}_full.csv' for k in (1, 2, 3)],
      makes=['results_v3/human_eval/qc/qc_summary.json']),
    S('human_eval_rescore_key', 'human', 'cpu', 'Re-score the analyst-only answer key with the final evaluator (same captions and references)',
      [CPU_ENV + f'{PY} scripts/v3_human_eval_rescore_key.py'], needs=['artifacts_cvpr_final/human_eval/image_grounded/hidden/answer_key.jsonl'],
      makes=['artifacts_cvpr_final/human_eval/image_grounded/hidden/answer_key_final_evaluator.jsonl', 'results_v3/human_eval/key_rescoring/summary.json']),
    S('human_eval_analysis', 'human', 'cpu', 'Reliability, metric-human correlations, pairwise accuracy (final key; historical key as sensitivity); paired metric comparison',
      [CPU_ENV + f'{PY} scripts/analyze_human_eval.py --key artifacts_cvpr_final/human_eval/image_grounded/hidden/answer_key_final_evaluator.jsonl --out results_v3/human_eval/analysis_final_evaluator',
       CPU_ENV + f'{PY} scripts/analyze_human_eval.py --key artifacts_cvpr_final/human_eval/image_grounded/hidden/answer_key.jsonl --out results_v3/human_eval/analysis_historical_key',
       CPU_ENV + f'{PY} scripts/v3_human_eval_compare.py'],
      needs=['artifacts_cvpr_final/human_eval/image_grounded/collected', 'artifacts_cvpr_final/human_eval/image_grounded/hidden/answer_key_final_evaluator.jsonl'],
      makes=['results_v3/human_eval/analysis_final_evaluator/summary.json', 'results_v3/human_eval/analysis_final_evaluator/metric_differences.csv'], note='Reads every CSV in collected/; only validated files belong there.'),
    S('human_parser_labels', 'human', 'human', 'Node-level parser validation labels (300 captions, 2+ annotators)',
      ['fill artifacts_cvpr_final/mapper_v2/node_label_template.csv per artifacts_cvpr_final/mapper_v2/INSTRUCTIONS.md']),

]
PHASES = ['setup', 'data', 'models', 'upstream', 'v3', 'paper', 'extra', 'human']
DEFAULT_PHASES = ['v3', 'paper']
BY_ID = {s['id']: s for s in STAGES}
assert len(BY_ID) == len(STAGES), 'duplicate stage id'


def exists(p):
    return Path(p).exists() if Path(p).is_absolute() else (ROOT / p).exists()


def cmd_list(_):
    for ph in PHASES:
        print(f'\n[{ph}]' + ('   <- default run' if ph in DEFAULT_PHASES else ''))
        for s in (x for x in STAGES if x['phase'] == ph):
            print(f"  {s['id']:<28} {s['kind']:<9} {s['title']}" + (f"  [{s['paper']}]" if s['paper'] else ''))


def cmd_check(_):
    bad = 0
    print(f'project root : {ROOT}\npython       : {PY}')
    for mod in ('ricce', 'numpy', 'pandas', 'scipy', 'yaml', 'sentence_transformers'):
        try:
            sys.path.insert(0, str(ROOT)); __import__(mod); print(f'  import {mod:<22} ok')
        except Exception as e:
            bad += 1; print(f'  import {mod:<22} FAILED ({e.__class__.__name__}: {e})')
    for tool in (TECT, 'pdfinfo', 'pdftotext', 'pdfunite'):
        ok = Path(tool).exists() if '/' in tool else shutil.which(tool)
        print(f'  tool {tool:<40} {"ok" if ok else "MISSING (needed by the paper phase)"}')
    print('\ninputs of every stage (missing inputs are produced by an earlier stage or phase):')
    for s in STAGES:
        missing = [n for n in s['needs'] if not exists(n)]
        absent = [m for m in s['makes'] if not exists(m)]
        state = 'ready' if not missing else 'MISSING ' + ', '.join(missing)
        print(f"  {s['phase']:<9} {s['id']:<28} {state}" + (f"   (outputs not yet present: {len(absent)})" if absent else ''))
    return 1 if bad else 0


def select(a):
    if a.stages:
        unknown = [x for x in a.stages if x not in BY_ID]
        if unknown: sys.exit(f'unknown stage(s): {unknown}; see `list`')
        return [BY_ID[x] for x in a.stages]
    phases = a.phase or DEFAULT_PHASES
    sel = [s for s in STAGES if s['phase'] in phases]
    ids = [s['id'] for s in sel]
    if a.from_:
        if a.from_ not in ids: sys.exit(f'--from {a.from_} is not in the selected phases {phases}')
        sel = sel[ids.index(a.from_):]
    if a.to:
        ids = [s['id'] for s in sel]
        if a.to not in ids: sys.exit(f'--to {a.to} is not in the selection')
        sel = sel[:ids.index(a.to) + 1]
    return sel


def cmd_run(a):
    sel = select(a)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    logdir = ROOT / 'REPRODUCE' / 'logs' / stamp
    if not a.dry_run: logdir.mkdir(parents=True, exist_ok=True)
    print(f'{len(sel)} stage(s); project root {ROOT}' + ('' if a.dry_run else f'; logs in {logdir}'))
    report = []
    for s in sel:
        head = f"\n=== [{s['phase']}] {s['id']} ({s['kind']}): {s['title']}"
        print(head)
        if s['note']: print(f"    note: {s['note']}")
        if s['kind'] in ('external', 'human') or (s['kind'] == 'gpu' and not a.allow_gpu):
            why = {'external': 'external environment / no script: run by hand', 'human': 'needs real human annotation',
                   'gpu': 'GPU stage: add --allow-gpu to execute'}[s['kind']]
            print(f'    NOT EXECUTED ({why}). Commands:')
            for c in s['cmds']: print(f'      {c}')
            report.append((s['id'], 'not executed: ' + s['kind'], 0.0)); continue
        missing = [n for n in s['needs'] if not exists(n)]
        if missing and not a.dry_run:
            print(f'    MISSING INPUTS: {missing}')
            report.append((s['id'], 'FAILED: missing inputs', 0.0))
            if a.keep_going: continue
            break
        t0 = time.time(); status = 'ok'
        for i, c in enumerate(s['cmds']):
            print(f'    $ {c}')
            if a.dry_run: continue
            log = logdir / f"{s['id']}_{i}.log"
            with open(log, 'w') as fh:
                fh.write(f'# {c}\n# cwd {ROOT}\n'); fh.flush()
                r = subprocess.run(c, shell=True, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, executable='/bin/bash')
            tail = log.read_text(errors='replace').strip().splitlines()[-3:]
            for line in tail: print(f'      | {line[:200]}')
            if r.returncode != 0:
                status = f'FAILED (exit {r.returncode}, see {log.relative_to(ROOT)})'; break
        dt = time.time() - t0
        if not a.dry_run:
            absent = [m for m in s['makes'] if not exists(m)]
            if status == 'ok' and absent: status = f'FAILED: expected outputs missing {absent}'
            print(f'    -> {status}  [{dt/60:.1f} min]')
        report.append((s['id'], status if not a.dry_run else 'dry-run', dt))
        if status != 'ok' and not a.dry_run and not a.keep_going: break
    print('\nSummary')
    for sid, st, dt in report: print(f'  {sid:<28} {st:<40} {dt/60:6.1f} min')
    if not a.dry_run:
        (logdir / 'summary.tsv').write_text('stage\tstatus\tminutes\n' + ''.join(f'{i}\t{st}\t{dt/60:.2f}\n' for i, st, dt in report))
    return 0 if all(st in ('ok', 'dry-run') or st.startswith('not executed') for _, st, _ in report) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('list', help='list every stage in pipeline order')
    sub.add_parser('check', help='check imports, tools and stage inputs')
    r = sub.add_parser('run', help='run stages (default: phases v3 and paper)')
    r.add_argument('--phase', nargs='+', choices=PHASES, help=f'phases to run (default: {" ".join(DEFAULT_PHASES)})')
    r.add_argument('--stages', nargs='+', help='run exactly these stage ids, in the given order')
    r.add_argument('--from', dest='from_', help='start at this stage (within the selected phases)')
    r.add_argument('--to', help='stop after this stage')
    r.add_argument('--allow-gpu', action='store_true', help='also execute GPU / local-LLM stages')
    r.add_argument('--keep-going', action='store_true', help='continue after a failed stage')
    r.add_argument('--dry-run', action='store_true', help='print the commands without running them')
    a = ap.parse_args()
    sys.exit({'list': cmd_list, 'check': cmd_check, 'run': cmd_run}[a.cmd](a) or 0)


if __name__ == '__main__':
    main()
