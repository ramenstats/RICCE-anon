"""Robustness to simulated acquisition interference, step 2: captions.

Runs the unmodified inference wrappers used for the frozen benchmark (Tab. 5)
on the clean pair and on the 25 corrupted scenarios of
scripts/cvpr27_23_robustness_corrupt.py, one subprocess per
(system, dataset, condition). Settings are those that reproduce the frozen
predictions (checked on 20 pairs per cell before this run, and on the full
test set by step 3):
  * chg2cap: greedy decoding on LEVIR-CC (the frozen LEVIR-CC run used the
    repo's test.py, whose default is greedy), beam 3 on SECOND-CC / Dubai-CC
    (the frozen runs used this wrapper's default);
  * swmcc, rsiccformer, promptcc, rscama: wrapper defaults, official/retrained
    checkpoints as in results/baselines/BASELINE_GENERATION_REPORT*.md.
Device: one device per (system, dataset), the same for clean and corrupted
inputs. GPU (shared, ~15 GB free) only where needed: rscama (CUDA-only code),
swmcc (its CPU deform_conv2d kernel segfaults: on
SECOND-CC pairs, as documented for the frozen run, and, in this run, at the same
clean LEVIR-CC pair between #500 and #600 on two attempts; swmcc therefore runs on
GPU for all three datasets). PromptCC runs on CPU: on the contended shared GPU it
took 1.7 s per pair (over 50 min per LEVIR-CC scenario). Re-running a system on a
different device or entry script than its frozen run can flip a few beam-search
near-ties, so clean captions are compared with the frozen ones in step 3 and the
clean column is always this run's own clean pass. Everything else runs on CPU. The Qwen2.5-VL systems (zero-shot, LoRA SFT) are not run: the 7B model
does not fit in the GPU memory left by other users' jobs (BLOCKED).
Outputs: results_cvpr27_robustness/predictions/<system>/<DS>/<condition>.jsonl
(+ .done marker once the file has exactly one prediction per test pair).
Usage: python scripts/cvpr27_24_robustness_infer.py [--cpu-workers 12] [--gpu-workers 3] [--only sys,...]
"""
import argparse, json, os, subprocess, sys, threading, time, queue
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = Path(os.environ.get('CONDA_ENVS', Path.home() / '.conda' / 'envs'))
CKX = ROOT / 'checkpoints_external'
COMPAT = str(ROOT / 'third_party' / 'pkg_resources_compat')
MAN = ROOT / 'artifacts_cvpr27_robustness' / 'manifests'
OUT = ROOT / 'results_cvpr27_robustness' / 'predictions'
LOG = ROOT / 'results_cvpr27_robustness' / 'infer_logs'
CONDS = ['clean'] + [f's3/{a}-{b}' for a in 'FGIMS' for b in 'FGIMS']
# severity sweep (Real-CD Fig. 5 uses levels 1/3/5): same type on both images, LEVIR-CC only, queued last
SWEEP = [f's{s}/{a}-{a}' for s in (1, 5) for a in 'FGIMS']


def spec(system, ds):
    """(python, script args without --manifest/--output, device, extra env) or None."""
    if system == 'chg2cap':
        dec = ['--decoding-method', 'greedy'] if ds == 'LEVIR_CC' else []
        return ENV / 'swmcc/bin/python', ['scripts/adhoc_chg2cap_infer.py', '--device', 'cpu', *dec], 'cpu', {}
    if system == 'swmcc':
        return ENV / 'swmcc/bin/python', ['scripts/adhoc_swmcc_infer.py', '--device', 'cuda'], 'cuda', {}
    if ds != 'LEVIR_CC':
        return None  # official checkpoints are evaluated on LEVIR-CC only (as in Tab. 5)
    if system == 'rsiccformer':
        return ENV / 'rsicc_fmr/bin/python', ['scripts/adhoc_rsiccformer_infer.py', '--checkpoint',
                                              str(CKX / 'rsiccformer/RSICC/BEST_checkpoint_resnet101_MCCFormers_diff_as_Q_trans.pth.tar'),
                                              '--data_folder', str(CKX / 'rsiccformer/data')], 'cpu', {}
    if system == 'promptcc':
        return ENV / 'rsicc_fmr/bin/python', ['scripts/adhoc_promptcc_infer.py', '--checkpoint',
                                              str(CKX / 'promptcc/BEST_checkpoint_ViT-B_32.pth.tar')], 'cpu', {'PYTHONPATH': COMPAT}
    if system == 'rscama':
        return ENV / 'rsicc_fmr/bin/python', ['scripts/adhoc_rscama_infer.py', '--checkpoint', str(CKX / 'rscama/RSCaMa.pth'),
                                              '--vocab', str(CKX / 'rscama/data_LEVIR_CC/vocab.json')], 'cuda', {'PYTHONPATH': COMPAT}
    raise KeyError(system)


SYSTEMS = ['chg2cap', 'swmcc', 'rsiccformer', 'promptcc', 'rscama']
DATASETS = ['LEVIR_CC', 'SECOND_CC', 'DUBAI_CC']


def manifest(ds, cond):
    return MAN / ds / ('clean.jsonl' if cond == 'clean' else f'{cond}.jsonl')


def outpath(sy, ds, cond):
    return OUT / sy / ds / f"{cond.replace('/', '_')}.jsonl"


def complete(sy, ds, cond):
    o = outpath(sy, ds, cond)
    if not o.with_suffix('.done').exists():
        return False
    return True


def verify(sy, ds, cond):
    want = [json.loads(l)['image_id'] for l in open(manifest(ds, cond))]
    o = outpath(sy, ds, cond)
    if not o.exists():
        return 'missing output'
    got = [json.loads(l)['image_id'] for l in open(o) if l.strip()]
    if sorted(got) != sorted(want):
        return f'{len(got)} predictions for {len(want)} pairs'
    return None


def run(job, threads, dev):
    sy, ds, cond = job
    py, args, _, extra = spec(sy, ds)
    o = outpath(sy, ds, cond); o.parent.mkdir(parents=True, exist_ok=True)
    lg = LOG / sy / ds / f"{cond.replace('/', '_')}.log"; lg.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', **extra,
           'OMP_NUM_THREADS': str(threads), 'MKL_NUM_THREADS': str(threads)}
    if dev == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = ''
    cmd = [str(py), *args[:1], '--manifest', str(manifest(ds, cond)), '--output', str(o), *args[1:]]
    t0 = time.time()
    with open(lg, 'w') as f:
        f.write(' '.join(cmd) + '\n'); f.flush()
        rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    err = f'exit {rc}' if rc else verify(sy, ds, cond)
    if err is None:
        o.with_suffix('.done').write_text(json.dumps(dict(device=dev, seconds=round(time.time() - t0, 1), cmd=cmd)))
    return err, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cpu-workers', type=int, default=12)
    ap.add_argument('--gpu-workers', type=int, default=3)
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--only', default='')
    ap.add_argument('--retries', type=int, default=3)
    a = ap.parse_args()
    systems = a.only.split(',') if a.only else SYSTEMS
    qs = {'cpu': queue.Queue(), 'cuda': queue.Queue()}
    n = 0
    for cond in CONDS + SWEEP:  # clean first, then scenarios in a fixed order, then the sweep
        for ds in (DATASETS if cond in CONDS else ['LEVIR_CC']):
            for sy in systems:
                s = spec(sy, ds)
                if s is None or complete(sy, ds, cond):
                    continue
                qs[s[2]].put((sy, ds, cond)); n += 1
    print(f'{n} jobs: cpu {qs["cpu"].qsize()}, gpu {qs["cuda"].qsize()}', flush=True)
    lock = threading.Lock(); failed = []

    def worker(dev):
        while True:
            try:
                job = qs[dev].get_nowait()
            except queue.Empty:
                return
            for attempt in range(1, a.retries + 1):
                err, sec = run(job, a.threads, dev)
                with lock:
                    print(time.strftime('%H:%M:%S'), dev, *job, 'ok' if err is None else f'FAIL({err}) attempt {attempt}', f'{sec:.0f}s', flush=True)
                if err is None:
                    break
                time.sleep(60)
            else:
                with lock:
                    failed.append(job)

    th = [threading.Thread(target=worker, args=('cpu',)) for _ in range(a.cpu_workers)]
    th += [threading.Thread(target=worker, args=('cuda',)) for _ in range(a.gpu_workers)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    print('FAILED:', failed if failed else 'none', flush=True)


if __name__ == '__main__':
    main()
