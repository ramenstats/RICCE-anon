"""v2-final: artifacts_v2/baseline_registry.csv -- one row per
captioning system used in the benchmark, with reproduction provenance.
Commit hashes and checkpoint sources are as recorded in Appendix H of
paper_tables/sec/H_official_baselines.tex; this file is the
machine-readable form of that appendix, not a re-derivation."""
import csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

rows = [
 ('RSICCFormer', 'Liu et al. 2022 (IEEE TGRS)', 'official (Google Drive checkpoint)', 'd1505e5', 'official released checkpoint', 'results/baselines/rsiccformer/LEVIR_CC_test.jsonl', 'YES', 'BLEU-4/CIDEr 63.1/134.1 (paper 62.8/134.1); LEVIR-CC only, no SECOND-CC/Dubai-CC checkpoint available'),
 ('PromptCC', 'Liu et al. 2023 (IEEE TGRS)', 'official (HuggingFace checkpoint)', '082d055', 'official released checkpoint', 'results/baselines/promptcc/LEVIR_CC_test.jsonl', 'YES', 'BLEU-4/CIDEr 65.1/135.7 (paper 63.5/136.4); LEVIR-CC only'),
 ('RSCaMa', 'Liu et al. 2024 (IEEE GRSL)', 'official (HuggingFace checkpoint)', 'aa68e4c', 'official released checkpoint', 'results/baselines/rscama/LEVIR_CC_test.jsonl', 'YES', 'BLEU-4/CIDEr 65.6/138.9 (paper 65.2/136.6); LEVIR-CC only'),
 ('Chg2Cap', 'Chang & Ghamisi 2023 (IEEE TIP)', 'our reproduction (no official checkpoint release used)', 'n/a (retrained)', 'trained in this project', 'results/baselines/chg2cap/{LEVIR_CC,SECOND_CC,DUBAI_CC}_test.jsonl', 'PARTIAL (retrained, not loaded from an official checkpoint)', 'architecture reproduced from the paper; zero-shot transfer to SECOND-CC/Dubai-CC from the LEVIR-CC-trained checkpoint'),
 ('SWMCC', 'not independently published as a standalone paper in this project\'s reference list; internal reproduction', 'our reproduction', 'n/a (retrained)', 'trained in this project', 'results/baselines/swmcc/{LEVIR_CC,SECOND_CC,DUBAI_CC}_test.jsonl', 'PARTIAL (retrained)', 'zero-shot transfer to SECOND-CC/Dubai-CC from the LEVIR-CC-trained checkpoint'),
 ('Qwen2.5-VL-7B (zero-shot)', 'Bai et al. 2025 (Qwen2.5-VL technical report)', 'official (HuggingFace, Qwen/Qwen2.5-VL-7B-Instruct)', 'n/a (inference only, no fine-tuning)', 'official pretrained checkpoint', 'results/baselines/qwen/{LEVIR_CC,SECOND_CC,DUBAI_CC}_test.jsonl', 'YES (official weights; no captioning-specific reproduction claimed)', 'general-purpose VLM used zero-shot as a lower-bound baseline'),
 ('Qwen2.5-VL-7B + LoRA SFT', 'this paper (baseline training recipe, not a prior-work reproduction)', 'n/a', 'n/a (trained in this project)', 'LoRA fine-tuned in this project, seeds 1/42/123', 'results/baselines/sft_seed{1,42,123}/{LEVIR_CC,SECOND_CC,DUBAI_CC}_test.jsonl', 'N/A (not a reproduction; this is our own baseline)', 'zero-shot transfer to SECOND-CC/Dubai-CC from the LEVIR-CC-trained checkpoint except where noted'),
]

OUT = ROOT / 'artifacts_v2' / 'baseline_registry.csv'
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['baseline', 'paper', 'official_repo', 'commit', 'checkpoint', 'prediction_file', 'reproduced', 'notes'])
    w.writerows(rows)
print(f'wrote {OUT} ({len(rows)} rows)')
