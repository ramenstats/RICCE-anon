"""CVPR'27, B1: the Change-Sensitive Metric (CSM) of the HCFNet authors on every caption
of the human study, against the same references as every other metric (hard cases: first
reference; real outputs: all references).

This is OUR re-implementation of CSM-k as defined in the authors' released csm_batch.py
(github.com/Hope21maker/HCFNet, commit 817975c3); running the downloaded script itself was not
permitted in this environment. Only the authors' two word-level rule files (JSON data) are used
directly. Definition (per sample): whitespace-lowercase tokens; for each n <= k, a candidate
n-gram that matches a reference n-gram exactly (clipped by the maximum reference count) scores 1,
any other n-gram scores min(0, best soft match), where the soft match of two n-grams is the mean
over aligned tokens of 1 (equal) or 0.5*(p_scale + p_extent) with p = -1/-2/-3 for a level gap of
0/1/>=2; precision_n = mean contribution, clipped at 0; CSM-k = brevity penalty (closest reference
length) x geometric mean of the clipped precisions (floor 1e-12). We report CSM-4 and, as a sensitivity check, CSM-1.
Output: submission_cvpr27_new_version/07_BASELINES/human_captions_csm.csv"""
import importlib.util, json, math, sys
from collections import Counter
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
B = ROOT / 'submission_cvpr27_new_version' / '07_BASELINES'
RULES = B / 'third_party_csm' / 'CSM Level Rules'
SS = {k: int(v) for k, v in json.load(open(RULES / 'Spatial Scale.json', encoding='utf-8-sig')).items()}
CE = {k: int(v) for k, v in json.load(open(RULES / 'Change Extent.json', encoding='utf-8-sig')).items()}
EPS = 1e-12


def tok(s):
    return s.lower().strip().split()


def grams(t, n):
    return [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]


def pen(gap):
    gap = abs(gap)
    return -1.0 if gap == 0 else -2.0 if gap == 1 else -3.0


def tsim(r, c):
    if r == c:
        return 1.0
    return 0.5 * (pen(SS.get(c, 0) - SS.get(r, 0)) + pen(CE.get(c, 0) - CE.get(r, 0)))


def precision(refs, cand, n):
    cg = grams(cand, n)
    if not cg:
        return 0.0
    allr, mx = [], Counter()
    for r in refs:
        g = grams(r, n); allr += g
        for k, v in Counter(g).items():
            mx[k] = max(mx[k], v)
    used, contrib = Counter(), []
    for c in cg:
        if used[c] < mx.get(c, 0):
            contrib.append(1.0); used[c] += 1; continue
        soft = [sum(tsim(a, b) for a, b in zip(r, c)) / len(c) for r in allr if r != c]
        contrib.append(min(0.0, max(soft) if soft else pen(0)))
    return max(0.0, sum(contrib) / len(cg))


def csm(refs, cand, k=4):
    if not refs or not cand:
        return 0.0
    ps = [precision(refs, cand, n) for n in range(1, k + 1)]
    rl = min((len(r) for r in refs), key=lambda L: abs(L - len(cand)))
    bp = 1.0 if len(cand) > rl else math.exp(1.0 - rl / len(cand))
    return bp * math.exp(sum(math.log(max(EPS, p)) for p in ps) / k)


if __name__ == '__main__':
    _spec = importlib.util.spec_from_file_location('cvpr27_03', ROOT / 'scripts' / 'cvpr27_03_extend_human_metrics.py')
    _m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
    J = _m.jobs()
    rows = [dict(job=r.job, csm1=csm([tok(g) for g in r.references], tok(r.candidate), 1),
                 csm4=csm([tok(g) for g in r.references], tok(r.candidate), 4)) for r in J.itertuples()]
    pd.DataFrame(rows).to_csv(B / 'human_captions_csm.csv', index=False)
    print(pd.DataFrame(rows).describe().round(3).to_string())
