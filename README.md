# RICCE: Robust Interpretable Change Caption Evaluation



A change caption is parsed into events (object, action, source, target, location, quantity, change state). Each field is compared with the reference on its taxonomy, and a contradiction gate gives zero credit to a caption that reverses the change. 

## Installation

```bash
conda create -n ricce python=3.11 -y
conda activate ricce
pip install -e .
# only for captioning models, training and the LLM/VLM judges:
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 torchvision==0.20.1
pip install -e '.[vlm]'
pip install statsmodels
```

Use `python -m ricce.cli ...` rather than the `ricce` console script, so the active environment is always the one that runs.

## Scoring a caption

```python
from ricce.benchmarks.env import load_taxes, load_weights, make_parser
from ricce.metrics.final import final_prf

taxes, weights = load_taxes(), load_weights()
parser = make_parser(taxes)

ref = parser.parse('a new building is constructed in the upper right', allow_multi_event=True)
cand = parser.parse('some buildings appear', allow_multi_event=True)
print(final_prf(cand, ref, taxes, weights))   # (CE-P, CE-R, CE-F) = (0.91, 0.73, 0.81)

cand = parser.parse('a building is demolished in the upper right', allow_multi_event=True)
print(final_prf(cand, ref, taxes, weights))   # (0.0, 0.0, 0.0): the gate
```

For several references, score against each and average, as in the paper. `ricce/metrics/final.py` fixes the evaluator configuration used for every number in the paper.

## Repository layout

```
ricce/
  ontology/         the four taxonomies and hierarchical precision/recall
  parsing/          caption -> events (alias match, n-grams, MiniLM retrieval, multi-event split)
  metrics/          event_v2.py (evaluator), final.py (paper configuration), conventional metrics, SPICE, LLM judge
  benchmarks/       RICCE-Bench: perturbation families and re-parse validation
  baselines/        flat hierarchical baselines (Kiritchenko hP/hR, LCA-F, HCM) and evaluator ablations
  evaluation/       ordering accuracy, parse cache, parser validation, human-study protocol and analysis
  training/, vlm/   Qwen2.5-VL captioner, LoRA SFT, DPO, swap augmentation
  data/             dataset discovery and manifests
configs/
  project.yaml      dataset root, parser settings, field weights
  ontology/         object, action, location and quantity taxonomies
scripts/            one script per experiment, table or figure
tests/              unit and property tests
REPRODUCE/          experiment runner and the command list for the paper
```

## Data

The experiments use LEVIR-CC, SECOND-CC and Dubai-CC. Put them, or a symlink to them, under `data/` (or change `dataset_root` in `configs/project.yaml`), then build the manifests:

```bash
python -m ricce.cli discover --dataset-root data
bash scripts/prepare_all.sh
```

The prior hierarchical metrics use the authors' code, cloned into `third_party/` (their tests are skipped without it):

```bash
git clone https://github.com/DerKevinRiehl/HierarchicalConfusionMatrix third_party/HierarchicalConfusionMatrix
git -C third_party/HierarchicalConfusionMatrix checkout 2256f4c
git clone https://github.com/BioASQ/Evaluation-Measures third_party/Evaluation-Measures
git -C third_party/Evaluation-Measures checkout 8c8e689
```

The captioning baselines (Chg2Cap, SWMCC, RSICCformer, PromptCC, RSCaMa) are expected under `third_party/` as well. Each `scripts/adhoc_*_infer.py` reads its repository from there, or from `CHG2CAP_REPO` and `SWMCC_REPO`.

## Reproducing the paper

- `REPRODUCE/cvpr2027_commands.sh` lists every command behind the paper's results, tables and figures, in order.
- `REPRODUCE/reproduce_ricce.py` runs the evaluator pipeline stage by stage (`list`, `check`, `run`); see `REPRODUCE/README.md`.

Most analyses run on CPU. Model training and inference, BERTScore and the LLM/VLM judges need a GPU. PDFs are built with [Tectonic](https://tectonic-typesetting.github.io/), found through `TECTONIC` or on `PATH`.

Run the tests with:

```bash
python -m pytest -q tests
```
