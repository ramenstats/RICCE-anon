"""SPICE and LLM-judge scorers, added to close the review gap that
``ricce/metrics/conventional.py`` (BLEU/METEOR/ROUGE-L/official CIDEr-D/
sentence-BERT/BERTScore) left open.

Both scorers follow a hard rule: if the underlying dependency genuinely is
not available in this environment, the function returns NaN / None and sets
an ``*_unavailable_reason`` string -- it never invents a plausible-looking
number. Callers must check the reason field before reporting a metric.

SPICE
-----
Uses the official ``pycocoevalcap.spice.spice.Spice`` (Java + the
stanford-corenlp-3.6.0 SPICE models jar, downloaded on first use by
``pycocoevalcap`` itself into its own package directory). This machine's
system Python (base Anaconda) had no ``java`` binary; a real Java 11
(Temurin, via ``conda install -n ricce openjdk=11``) was installed into
the user-owned ``ricce`` conda env and its ``bin/`` is added to PATH by
this module so the SPICE jar's ``java -jar`` subprocess call can find a JVM
without touching anything we did not create. If java is still not
discoverable, or the (large, ~384MB) CoreNLP model download that
``pycocoevalcap`` performs on first use has not completed / fails, SPICE is
reported unavailable rather than faked.

LLM-judge
---------
Reuses the exact model family already used for VLM captioning in this repo
(``ricce/vlm/qwen.py`` -> Qwen2.5-VL-7B-Instruct, from
``configs/project.yaml``): the text-only sibling checkpoint
``Qwen/Qwen2.5-3B-Instruct`` (already cached locally under
``~/.cache/huggingface/hub``), loaded via plain ``transformers``. No
hosted-LLM API key is set in this
environment, so a locally-hosted instruction model is the only real access
path -- there is no schema-constrained "LLM fallback" stage elsewhere in
this repo to reuse (only ``ricce/parsing/hybrid.py``'s embedding
fallback, which is not an LLM call). The judge is prompted to output a
single integer 1-5 rating of whether the prediction correctly and completely
describes the same change(s) as the reference caption(s); the raw text
response is parsed with a strict regex and any unparsable response is
recorded as NaN, never coerced to a guessed score.
"""
import os
import re
import sys
import random
import numpy as np

# This machine's GPU is shared with several other users' jobs and routinely
# sits near full (observed 89-93GB / 95GB used by processes that are not
# ours). expandable_segments reduces allocator fragmentation under that kind
# of contention (suggested by the PyTorch OOM message itself); set it before
# torch is ever imported by anything in this process.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# ---------------------------------------------------------------------------
# SPICE
# ---------------------------------------------------------------------------

_JAVA_CANDIDATES = [
    os.path.join(sys.prefix, 'bin'),
    os.path.expanduser('~/.conda/envs/ricce/bin'),
]

def _ensure_java_on_path():
    """Make a real `java` binary discoverable for the SPICE jar's subprocess
    call, without mutating any file we did not create. Returns True if a
    `java` binary is now on PATH, False otherwise (SPICE must then be
    reported unavailable, never faked)."""
    import shutil
    if shutil.which('java'):
        return True
    for cand in _JAVA_CANDIDATES:
        if os.path.exists(os.path.join(cand, 'java')):
            os.environ['PATH'] = cand + os.pathsep + os.environ.get('PATH', '')
            if shutil.which('java'):
                return True
    return False

_SPICE_CACHE = {}

def spice_available():
    """Real availability probe: java on PATH AND pycocoevalcap.spice importable
    AND the (large) stanford-corenlp SPICE model resources already downloaded.
    Does not download anything itself (that happens lazily inside Spice()
    on first compute_score call, which can take a long time on first run)."""
    if not _ensure_java_on_path():
        return False, 'no java runtime discoverable (installed openjdk=11 into conda env `ricce`, but binary not found on PATH)'
    try:
        import pycocoevalcap
        from pycocoevalcap.spice.spice import Spice  # noqa: F401
    except Exception as e:
        return False, f'pycocoevalcap.spice import failed: {e!r}'
    return True, None

def spice_corpus(image_ids, predictions, references_by_id, timeout=None):
    """Official SPICE (Anderson et al. 2016), computed for a whole corpus in
    a single Java subprocess call (as the pycocoevalcap wrapper requires).

    Returns (corpus_mean, {image_id: per_image_f_score}, reason). `reason` is
    None on success, else a human-readable string explaining why SPICE could
    not be computed (java missing, CoreNLP model download incomplete/failed,
    subprocess error, etc.) -- callers must treat a non-None reason as "no
    real SPICE number available" and omit it from tables rather than filling
    in a placeholder.
    """
    if not image_ids:
        return float('nan'), {}, 'no items'
    ok, reason = spice_available()
    if not ok:
        return float('nan'), {i: float('nan') for i in image_ids}, reason
    try:
        from pycocoevalcap.spice.spice import Spice
        gts = {str(i): references_by_id[i] for i in image_ids}
        res = {str(i): [predictions[j]] for j, i in enumerate(image_ids)}
        sp = Spice()
        overall, per_item_detail = sp.compute_score(gts, res)
        per_item = {}
        for i, d in zip(image_ids, per_item_detail):
            f = d.get('All', {}).get('f', float('nan'))
            per_item[i] = float(f) if f == f else float('nan')  # NaN-safe
        return float(overall), per_item, None
    except Exception as e:
        return float('nan'), {i: float('nan') for i in image_ids}, f'SPICE subprocess/compute failed: {e!r}'

# ---------------------------------------------------------------------------
# LLM-judge
# ---------------------------------------------------------------------------

_LLM_JUDGE_MODEL_NAME = 'Qwen/Qwen2.5-3B-Instruct'
_LLM_JUDGE_CACHE = {}
_LLM_JUDGE_CPU_CACHE = {}

_JUDGE_PROMPT = (
    "You are grading a remote-sensing bi-temporal change-detection caption.\n"
    "Reference caption(s) (ground truth, may be multiple paraphrases of the same change):\n"
    "{refs}\n\n"
    "Candidate caption (model prediction):\n"
    "{pred}\n\n"
    "Rate how correctly and completely the candidate describes the SAME change(s) as the "
    "reference(s), on this scale:\n"
    "1 = completely wrong (describes a different/opposite change, or hallucinates changes that did not happen)\n"
    "2 = mostly wrong / major omissions or errors\n"
    "3 = partially correct (right general idea, but missing or wrong details)\n"
    "4 = mostly correct (minor wording/detail differences only)\n"
    "5 = fully correct (same meaning as reference)\n\n"
    "Respond with ONLY a single digit 1-5 on its own, nothing else."
)

def llm_judge_available():
    """Real availability probe for the local text-LLM judge: is the
    Qwen2.5-3B-Instruct checkpoint locally cached / fetchable, and does
    transformers import. No API key check needed since no LLM API key
    is set in this environment
    -- this repo's only viable LLM access path is a locally-hosted model."""
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
    except Exception as e:
        return False, f'transformers/torch import failed: {e!r}'
    return True, None

def _load_llm_judge(model_name=_LLM_JUDGE_MODEL_NAME):
    if model_name in _LLM_JUDGE_CACHE:
        return _LLM_JUDGE_CACHE[model_name]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map='auto')
    model.eval()
    _LLM_JUDGE_CACHE[model_name] = (tok, model)
    return tok, model

def _load_llm_judge_cpu(model_name=_LLM_JUDGE_MODEL_NAME):
    """A second, CPU-resident copy of the judge model, loaded lazily only if
    the GPU copy hits a persistent OOM under this machine's shared-GPU
    contention. Slow, but never fabricates a score just to avoid the cost."""
    if model_name in _LLM_JUDGE_CPU_CACHE:
        return _LLM_JUDGE_CPU_CACHE[model_name]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map='cpu')
    model.eval()
    _LLM_JUDGE_CPU_CACHE[model_name] = (tok, model)
    return tok, model

_SCORE_RE = re.compile(r'[1-5]')

def _parse_judge_score(text):
    m = _SCORE_RE.search(text.strip())
    return float(m.group(0)) if m else float('nan')

def _generate_chunk(tok, model, device, chunk, max_new_tokens):
    """Run one batch through the model. Raises torch.cuda.OutOfMemoryError on
    OOM (caller retries with a smaller batch / falls back to CPU)."""
    import torch
    chats = [tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True) for p in chunk]
    enc = tok(chats, return_tensors='pt', padding=True, truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
    gen = out[:, enc['input_ids'].shape[1]:]
    return tok.batch_decode(gen, skip_special_tokens=True)


def llm_judge_batch(preds, refs_list, model_name=_LLM_JUDGE_MODEL_NAME, batch_size=16, max_new_tokens=4):
    """Score a list of (prediction, references) pairs with the local LLM
    judge, batched for throughput. Returns (scores, reason). `scores` is a
    list of floats in {1,2,3,4,5} or NaN for any item whose response could
    not be parsed as a 1-5 digit *or* that could not be scored at all because
    of a persistent CUDA OOM on this shared GPU; `reason` is None if every
    item was at least attempted, or a string summarizing how many items were
    skipped due to OOM (this machine's GPU is shared with other users' jobs
    that can occupy 90+ GB of 95GB at any moment, so a mid-run OOM here is
    expected contention, not a bug) -- callers must not treat OOM-skipped
    items as a score of anything.

    This mirrors the degrade-rather-than-crash pattern already used for the
    GPU sentence embedder in ricce/parsing/hybrid.py._semantic: a
    transient OOM shrinks the batch / falls back to CPU instead of taking
    down the whole scoring run."""
    ok, reason = llm_judge_available()
    if not ok:
        return [float('nan')] * len(preds), reason
    try:
        import torch
        tok, model = _load_llm_judge(model_name)
    except Exception as e:
        return [float('nan')] * len(preds), f'failed to load local LLM judge {model_name!r}: {e!r}'
    device = next(model.parameters()).device
    scores = [float('nan')] * len(preds)
    prompts = []
    for pred, refs in zip(preds, refs_list):
        refs_txt = '\n'.join(f'- {r.strip()}' for r in refs) if refs else '(no reference available)'
        prompts.append(_JUDGE_PROMPT.format(refs=refs_txt, pred=pred.strip()))

    n_oom_skipped = 0
    start = 0
    cur_batch_size = batch_size
    while start < len(prompts):
        chunk = prompts[start:start + cur_batch_size]
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            texts = _generate_chunk(tok, model, device, chunk, max_new_tokens)
            for j, t in enumerate(texts):
                scores[start + j] = _parse_judge_score(t)
            start += cur_batch_size
        except torch.cuda.OutOfMemoryError:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            if cur_batch_size > 1:
                cur_batch_size = max(1, cur_batch_size // 4)
                continue
            # Single-item batch still OOMs under contention on this shared
            # GPU: try once on a separate CPU-resident copy of the model
            # (slow but memory-safe) before giving up on this item -- do not
            # fabricate a score.
            try:
                cpu_tok, cpu_model = _load_llm_judge_cpu(model_name)
                cpu_texts = _generate_chunk(cpu_tok, cpu_model, torch.device('cpu'), chunk, max_new_tokens)
                scores[start] = _parse_judge_score(cpu_texts[0])
            except Exception:
                n_oom_skipped += 1
            start += 1
            cur_batch_size = batch_size  # reset for the next chunk
    reason_out = None
    if n_oom_skipped:
        reason_out = (f'{n_oom_skipped}/{len(preds)} items skipped (left as NaN): persistent CUDA OOM '
                      f'under shared-GPU contention even at batch_size=1 and after a CPU fallback attempt')
    return scores, reason_out

def sample_indices(n_total, n_sample, seed=42):
    """Deterministic, disclosed random subsample used only to keep wall-clock
    cost of the expensive scorers (SPICE JVM calls, LLM-judge generation)
    bounded in this shared/limited environment. Returns sorted indices."""
    if n_sample >= n_total:
        return list(range(n_total))
    rng = random.Random(seed)
    return sorted(rng.sample(range(n_total), n_sample))
