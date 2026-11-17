# Reproducing the experiments

Two entry points, run from the repository root inside the `ricce` conda env:

| File | What it runs |
|---|---|
| `reproduce_ricce.py` | The pipeline from the raw datasets to the result files: manifests, caption predictions, RICCE-Bench, ordering suites, baselines, the final evaluator on every system, tables and figures. |
| `cvpr2027_commands.sh` | The analyses of the paper on top of those results, in order: human-study re-analysis, judges, CSM, tables, figures, acquisition interference, field-weight study, properties table. |

## Stage kinds

| Kind | Meaning | Executed by `run`? |
|---|---|---|
| `cpu` | Runs in the `ricce` env on CPU | Yes |
| `gpu` | Needs the GPU or a local LLM (training, Qwen inference, BERTScore, LLM/VLM judges) | Only with `--allow-gpu` |
| `external` | Needs another conda env or a third-party repository with its own checkpoints | No; the commands are printed |
| `human` | Needs real human annotations | No; the instructions are printed. Results are never simulated. |

## Quick start

```bash
python REPRODUCE/reproduce_ricce.py list                       # every stage, in order, with its kind
python REPRODUCE/reproduce_ricce.py check                      # are the inputs and tools present?
python REPRODUCE/reproduce_ricce.py run                        # final evaluator, tables and figures (phases v3, paper)
python REPRODUCE/reproduce_ricce.py run --phase upstream --allow-gpu
python REPRODUCE/reproduce_ricce.py run --stages v3_systems v3_ablations
python REPRODUCE/reproduce_ricce.py run --from v3_gate         # resume after fixing a failure
python REPRODUCE/reproduce_ricce.py run --dry-run              # print the commands only
bash REPRODUCE/cvpr2027_commands.sh
```

Phases, in pipeline order: `setup` (tests), `data` (dataset discovery and manifests), `models` (caption predictions: Qwen2.5-VL zero-shot, LoRA SFT, swap augmentation, preference optimization, external RSICC baselines), `upstream` (RICCE-Bench, ordering suites, flat hierarchical baselines, parser validation), `v3` (every number under the final evaluator), `paper` (tables and figures), `extra` (analyses not in the paper), `human` (annotation, quality control, answer-key re-scoring and analysis).

Every run writes `REPRODUCE/logs/<date_time>/` with one log per command and a `summary.tsv` of each stage's status and time. The runner stops at the first failed stage unless `--keep-going` is given. Re-running a stage overwrites its outputs.

## What is not in this repository

- The datasets (LEVIR-CC, SECOND-CC, Dubai-CC): put them under `data/` or set `RICCE_DATASET_ROOT`.
- Model checkpoints and the third-party captioning and hierarchical-metric code (`third_party/`, see the top-level README).
- Result folders (`results*/`, `artifacts*/`, `submission_cvpr27_new_version/`): every script writes them.
- The human-study annotation files and the analyst-only answer key.

## Where results go

| Folder | Contents |
|---|---|
| `results/baselines/` | caption predictions of every system |
| `artifacts_cvpr_final/`, `results_cvpr_final/` | RICCE-Bench, ordering scores, flat baselines, parser validation |
| `results_v2/`, `results_metric_v2/`, `results_r1/` | earlier evaluator rounds, read for basis checks |
| `results_v3/` | every final-evaluator result |
| `submission_cvpr27_new_version/` | human-study re-analysis, baselines on the rated captions, method ablations, field-weight study, tables and numbers of the paper |
| `results_cvpr27_robustness/` | acquisition interference: predictions, scores, contrasts |

## Troubleshooting

- **A stage failed.** Open the log the runner names, fix the cause, then resume with `run --from <stage>`.
- **`ModuleNotFoundError: hierarchical_confusion_matrix`.** `third_party/` is missing; it holds the official HCM and HEMKit code.
- **A bare `ricce` command fails with an import error.** Use `python -m ricce.cli ...`.
- **WordNet missing.** `python -c "import nltk; nltk.download('wordnet')"`.
