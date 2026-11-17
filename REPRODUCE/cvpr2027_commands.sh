#!/usr/bin/env bash
# Rebuild every result, table and figure of the CVPR 2027 paper, in order (verified 2026-10-29).
# Run from the project root. CPU is enough except stage 3 (BERTScore runs faster on CPU too) and stage 6 (judges: GPU, ~25 min).
set -euo pipefail
# Tools (override as needed):
#   TECTONIC  LaTeX engine used for every PDF
#   RSICC_PY  python of the conda env with the RSICCformer/PromptCC/RSCaMa dependencies (image corruption)
TECTONIC=${TECTONIC:-tectonic}
RSICC_PY=${RSICC_PY:-$(conda run -n rsicc_fmr which python)}
eval "$(conda shell.bash hook)" && conda activate ricce
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/cvpr27_02_annotation_provenance.py           # H0: outcome-blind checks of both annotation rounds
python scripts/cvpr27_03_extend_human_metrics.py            # CIDEr-D, BERTScore, SBERT, SPICE, LLM judge on every rated caption (+ checks)
python scripts/cvpr27_04_variants_human.py                  # evaluator variants, soft gate, joint assignment, prior hierarchical metrics
python scripts/cvpr27_10_csm_human.py                       # CSM (re-implementation of the authors' scorer)
python scripts/cvpr27_06_judges.py                          # VLM judge (images) and text-judge expected score  [GPU]
python scripts/cvpr27_05_human_reanalysis.py                # H1 re-analysis (paired cluster bootstrap, B=10,000)
python scripts/cvpr27_07_temporal_fixed_cohort.py           # T1 fixed-cohort temporal endpoints
python scripts/cvpr27_09_bertscore_canonical.py             # canonical BERTScore for RICCE-Bench, teaser, pilot
python scripts/cvpr27_11_prepare_human_batches.py           # empty annotation batches (H2, P1, X1, gate, swap)
python scripts/cvpr27_12_tables_tokens.py                   # every new table and number
python scripts/cvpr27_17_make_figures.py                    # heatmap (canonical BERTScore), forest plot, examples
python scripts/cvpr27_22_make_overview_figures.py          # CVPR2027_Figures/: teaser (Fig. 1) and method overview, with LaTeX snippets
# acquisition interference (Real-CD protocol): corrupted test images -> captions -> scores -> tables
PYTHONPATH=third_party/imagecorruptions_pkg:third_party/pkg_resources_compat "$RSICC_PY" scripts/cvpr27_23_robustness_corrupt.py
PYTHONPATH=third_party/imagecorruptions_pkg:third_party/pkg_resources_compat "$RSICC_PY" scripts/cvpr27_23_robustness_corrupt.py --datasets LEVIR_CC --severities 1,5
#   impulse noise is not reproducible from the seed (skimage draws it unseeded): the stored images are the reference;
#   verify them against artifacts_cvpr27_robustness/image_sha256.tsv (sha256 of the list: 78324954c01765c5dd4e5e5fc606956509e9e7502d0be57740282b5a74a076b6)
PYTHONPATH=third_party/imagecorruptions_pkg:third_party/pkg_resources_compat "$RSICC_PY" scripts/cvpr27_23_robustness_corrupt.py --hashes
python scripts/cvpr27_24_robustness_infer.py --cpu-workers 20 --gpu-workers 4 --threads 4   # resumable; ~8.5 h on the shared machine  [GPU]
python scripts/cvpr27_25_robustness_score.py                # CE-P/R/F, BLEU-4, CIDEr-D, diagnostics, bootstrap CIs (+ clean-vs-frozen checks)
python scripts/cvpr27_25b_robustness_same_vs_mixed.py       # same- vs mixed-type and T1- vs T2-interference contrasts (paired bootstrap)
python scripts/cvpr27_26_robustness_tables.py               # CVPR2027_Robustness/: tables, example figure, paper text (templates/*.in)
python -m pytest -q tests                                   # unit tests

# Field-weight study (2026-11-09, exploratory; no weight changed) -> submission_cvpr27_new_version/14_WEIGHT_STUDY/
python scripts/cvpr27_38_weight_study.py > submission_cvpr27_new_version/14_WEIGHT_STUDY_run.log 2>&1

# Framework figure (Fig. 2 style of Snaebjarnarson et al. 2025) and taxonomy figure (2026-11-10) -> CVPR2027_Framework_Figures/
python scripts/cvpr27_39_framework_figures.py     # fig_framework.pdf, fig_taxonomies.pdf, figure_values.json (all values from the evaluator)

# Framework/taxonomy figures, version 2 (2026-11-11: minimal text, all text black, no weight table) -> CVPR2027_Framework_Figures_v2/
python scripts/cvpr27_41_framework_figures_v2.py

# Fig. 1 (motivation), version 2 (2026-11-11: less text, black text, no LLM judge, clearer (b)/(c)) -> CVPR2027_Motivation_Figure_v2/
python scripts/cvpr27_42_motivation_v2.py

# Table of the properties desired of an RSCC metric -> CVPR2027_Schematic_and_Properties/
python scripts/cvpr27_43_properties_table.py
