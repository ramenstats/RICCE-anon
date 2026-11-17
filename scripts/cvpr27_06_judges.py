"""CVPR'27, B1: LLM / VLM judges on every caption of the human study.

  vlm_judge   image-aware judge, local Qwen2.5-VL-3B-Instruct. Input: the T1 and T2
              thumbnails the annotators saw (images/<DS>/*_A.jpg, *_B.jpg) and the
              candidate; NO reference captions. Score = expected rating sum_k k * p(k)
              over the next-token probabilities of the digits 1-5 (G-Eval style);
              vlm_judge_digit = the most probable digit.
  llm_judge_ev  text judge with the prompt of ricce/metrics/spice_llm_judge.py
              (Qwen2.5-3B-Instruct, references, no images), scored the same way.
Both use greedy single-step scoring, no sampling, a fixed prompt written before any
judge output was read, and are run once. Different inputs: the VLM judge sees pixels,
the text judge and every other metric see references only.

Output: submission_cvpr27_new_version/07_BASELINES/human_captions_vlm_judge.csv"""
import json, os, sys, time
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np, pandas as pd, torch
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.utils import read_jsonl
from ricce.metrics.spice_llm_judge import _JUDGE_PROMPT
H = ROOT / 'artifacts_cvpr_final' / 'human_eval' / 'image_grounded'
OUT = ROOT / 'submission_cvpr27_new_version' / '07_BASELINES' / 'human_captions_vlm_judge.csv'
import importlib.util
_spec = importlib.util.spec_from_file_location('cvpr27_03', ROOT / 'scripts' / 'cvpr27_03_extend_human_metrics.py')
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
J = _m.jobs()
thumbs = {r['item_id']: (r['t1_thumb'], r['t2_thumb']) for r in read_jsonl(H / 'items.jsonl')}
for r in read_jsonl(H / 'pairwise_items.jsonl'):
    thumbs[r['pair_id']] = (r.get('t1_thumb'), r.get('t2_thumb'))

VLM_PROMPT = (
    "The first image was taken at time T1 (before) and the second image at time T2 (after); both show the same area.\n"
    "Candidate caption describing the change from T1 to T2:\n{pred}\n\n"
    "Rate how correctly and completely the candidate describes what changed between T1 and T2 "
    "(or correctly states that nothing changed), on this scale:\n"
    "1 = completely wrong (describes a different/opposite change, or hallucinates changes that did not happen)\n"
    "2 = mostly wrong / major omissions or errors\n"
    "3 = partially correct (right general idea, but missing or wrong details)\n"
    "4 = mostly correct (minor detail differences only)\n"
    "5 = fully correct\n\n"
    "Respond with ONLY a single digit 1-5 on its own, nothing else.")


def digit_ids(tok):
    ids = [tok.convert_tokens_to_ids(str(k)) for k in range(1, 6)]
    assert all(isinstance(i, int) and i >= 0 for i in ids), ids
    return ids


def ev_from_logits(logits, ids):
    p = torch.softmax(logits[:, ids].float(), dim=-1).cpu().numpy()
    return (p * np.arange(1, 6)).sum(1), p.argmax(1) + 1


@torch.no_grad()
def run_vlm(jobs, bs=8):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    name = 'Qwen/Qwen2.5-VL-3B-Instruct'
    proc = AutoProcessor.from_pretrained(name); proc.tokenizer.padding_side = 'left'
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(name, dtype=torch.bfloat16, device_map='cuda').eval()
    ids = digit_ids(proc.tokenizer); ev, dg = [], []
    i = 0
    while i < len(jobs):
        chunk = jobs[i:i + bs]
        texts, imgs = [], []
        for r in chunk:
            t1, t2 = thumbs[r.item_id]
            msg = [{'role': 'user', 'content': [{'type': 'text', 'text': 'Image at T1 (before):'}, {'type': 'image'},
                                                 {'type': 'text', 'text': 'Image at T2 (after):'}, {'type': 'image'},
                                                 {'type': 'text', 'text': VLM_PROMPT.format(pred=r.candidate.strip())}]}]
            texts.append(proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
            imgs += [Image.open(H / t1).convert('RGB'), Image.open(H / t2).convert('RGB')]
        try:
            enc = proc(text=texts, images=imgs, return_tensors='pt', padding=True).to('cuda')
            lg = model(**enc).logits[:, -1, :]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); bs = max(1, bs // 2); print('OOM -> batch', bs, flush=True); continue
        e, d = ev_from_logits(lg, ids); ev += list(e); dg += list(d); i += len(chunk)
        if i % 200 < bs: print('vlm', i, flush=True)
    del model; torch.cuda.empty_cache()
    return ev, dg


@torch.no_grad()
def run_text(jobs, bs=16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = 'Qwen/Qwen2.5-3B-Instruct'
    tok = AutoTokenizer.from_pretrained(name); tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map='cuda').eval()
    ids = digit_ids(tok); ev = []
    for i in range(0, len(jobs), bs):
        chunk = jobs[i:i + bs]
        ps = [_JUDGE_PROMPT.format(refs='\n'.join(f'- {g.strip()}' for g in r.references), pred=r.candidate.strip()) for r in chunk]
        chats = [tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True) for p in ps]
        enc = tok(chats, return_tensors='pt', padding=True).to('cuda')
        e, _ = ev_from_logits(model(**enc).logits[:, -1, :], ids); ev += list(e)
    del model; torch.cuda.empty_cache()
    return ev


if __name__ == '__main__':
    rows = list(J.itertuples()); t = time.time()
    v, d = run_vlm(rows); print(f'vlm {time.time() - t:.0f}s', flush=True)
    pd.DataFrame(dict(job=J.job, vlm_judge=v, vlm_judge_digit=d)).to_csv(OUT, index=False)
    tx = run_text(rows); print(f'text {time.time() - t:.0f}s', flush=True)
    pd.DataFrame(dict(job=J.job, vlm_judge=v, vlm_judge_digit=d, llm_judge_ev=tx)).to_csv(OUT, index=False)
    print(pd.read_csv(OUT).describe().round(3).to_string())
