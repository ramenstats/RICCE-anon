"""Robustness to simulated acquisition interference, step 1: corrupted test images.

Protocol (fixed before any caption was generated, following Real-CD,
Fang et al., IEEE TIP 35:4295-4309, 2026, Sec. III-A):
  * five interference types: fog (F), Gaussian noise (G), impulse noise (I),
    motion blur (M), snow (S);
  * each type is applied independently to the T1 and to the T2 image of every
    test pair; every degraded T1 is paired with every degraded T2, giving
    5 x 5 = 25 bi-temporal scenarios per dataset (label "F-G" = fog on T1,
    Gaussian noise on T2), plus the clean pair;
  * implementation: the ImageNet-C corruption functions (Hendrycks & Dietterich,
    ICLR 2019) as packaged in `imagecorruptions` 1.1.2 (Michaelis et al. 2019),
    the package behind imgaug's imgcorruptlike augmenters; Real-CD used imgaug
    but does not report its settings, so its exact images are NOT reproduced;
  * severity 3 of 5 for all types (the middle level; Real-CD's Fig. 5 uses
    levels 1/3/5); optional extra severities via --severities;
  * corruption is applied at each dataset's native resolution (LEVIR-CC and
    SECOND-CC 256x256, Dubai-CC 50x50 tiles) before the captioner's own
    preprocessing, as interference would be present in the stored image;
  * each degraded image is written once and the same file is used in all five
    scenarios that contain it (existing files are never overwritten);
  * every call is seeded with np.random.seed(sha256(dataset|image_id|time|type|severity)).
    This fixes fog, Gaussian noise, motion blur and snow (regeneration is
    byte-identical), but NOT impulse noise: imagecorruptions calls
    skimage.util.random_noise, which in skimage >= 0.19 (0.25.2 here) draws from
    its own unseeded generator. The stored impulse-noise images are therefore the
    reference artifacts; --hashes records the SHA-256 of every stored image.
Outputs (artifacts_cvpr27_robustness/):
  images/<DS>/s<sev>/<type>/<t1|t2>/<image_id>.png
  manifests/<DS>/clean.jsonl and manifests/<DS>/s<sev>/<A>-<B>.jsonl
  corruption_log.json  (settings, library version, counts, manifest hashes)
  image_sha256.tsv     (--hashes: path, sha256 of every stored image)
Run in the rsicc_fmr env:
  PYTHONPATH=third_party/imagecorruptions_pkg:third_party/pkg_resources_compat \
    conda run -n rsicc_fmr python scripts/cvpr27_23_robustness_corrupt.py [--hashes]
"""
import argparse, hashlib, json, multiprocessing as mp, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'third_party' / 'imagecorruptions_pkg'))
sys.path.insert(0, str(ROOT / 'third_party' / 'pkg_resources_compat'))
import imagecorruptions as IC  # noqa: E402

TYPES = {'F': 'fog', 'G': 'gaussian_noise', 'I': 'impulse_noise', 'M': 'motion_blur', 'S': 'snow'}
DATASETS = ['LEVIR_CC', 'SECOND_CC', 'DUBAI_CC']
OUT = ROOT / 'artifacts_cvpr27_robustness'


def seed_of(*parts):
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:8], 16)


def work(job):
    ds, iid, t, src, code, sev = job
    dst = OUT / 'images' / ds / f's{sev}' / code / t / f'{iid}.png'
    if dst.exists():
        return str(dst), False
    x = np.asarray(Image.open(src).convert('RGB'), dtype=np.uint8)
    np.random.seed(seed_of(ds, iid, t, TYPES[code], sev))
    y = IC.corrupt(x, severity=sev, corruption_name=TYPES[code])
    assert y.shape == x.shape and y.dtype == np.uint8, (src, code, y.shape, y.dtype)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(y).save(dst)
    return str(dst), True


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--severities', default='3')
    ap.add_argument('--datasets', default=','.join(DATASETS))
    ap.add_argument('--procs', type=int, default=32)
    ap.add_argument('--hashes', action='store_true', help='only write image_sha256.tsv for the stored images')
    a = ap.parse_args()
    if a.hashes:
        imgs = sorted(OUT.glob('images/*/s*/*/t*/*.png'))
        with mp.get_context('fork').Pool(a.procs) as pool:
            hs = pool.map(sha, imgs, chunksize=256)
        with open(OUT / 'image_sha256.tsv', 'w') as f:
            f.writelines(f'{p.relative_to(OUT)}\t{h}\n' for p, h in zip(imgs, hs))
        print(len(imgs), 'image hashes written')
        return
    sevs = [int(s) for s in a.severities.split(',')]
    t0 = time.time()
    logp = OUT / 'corruption_log.json'
    log = json.load(open(logp)) if logp.exists() else {}
    log.update(dict(library='imagecorruptions', library_version=IC.__dict__.get('__version__', '1.1.2'),
                    types=TYPES, protocol='Real-CD 5x5 bi-temporal scenarios', seeding='sha256(dataset|image_id|t|type|severity)[:8]'))
    for ds in a.datasets.split(','):
        rows = [json.loads(l) for l in open(ROOT / 'artifacts' / 'manifests' / f'{ds}.jsonl')]
        rows = [r for r in rows if r['split'] == 'test']
        (OUT / 'manifests' / ds).mkdir(parents=True, exist_ok=True)
        with open(OUT / 'manifests' / ds / 'clean.jsonl', 'w') as f:
            for r in rows:
                f.write(json.dumps({**r, 'condition': 'clean'}) + '\n')
        for sev in sevs:
            jobs = [(ds, r['image_id'], t, r[f'{t}_path'], c, sev) for r in rows for t in ('t1', 't2') for c in TYPES]
            with mp.get_context('fork').Pool(a.procs) as pool:
                res = pool.map(work, jobs, chunksize=16)
            made = sum(n for _, n in res)
            img = {(j[1], j[2], j[4]): p for j, (p, _) in zip(jobs, res)}
            mdir = OUT / 'manifests' / ds / f's{sev}'; mdir.mkdir(parents=True, exist_ok=True)
            hashes = {}
            for c1 in TYPES:
                for c2 in TYPES:
                    p = mdir / f'{c1}-{c2}.jsonl'
                    with open(p, 'w') as f:
                        for r in rows:
                            f.write(json.dumps({**r, 't1_path': img[(r['image_id'], 't1', c1)], 't2_path': img[(r['image_id'], 't2', c2)],
                                                'condition': f'{c1}-{c2}', 'severity': sev, 'clean_t1_path': r['t1_path'],
                                                'clean_t2_path': r['t2_path']}) + '\n')
                    hashes[f'{c1}-{c2}'] = sha(p)
            log.setdefault('datasets', {}).setdefault(ds, {})[f's{sev}'] = dict(
                n_pairs=len(rows), n_images=len(jobs), n_newly_written=made, manifest_sha256=hashes)
            print(ds, f's{sev}', len(rows), 'pairs', len(jobs), 'images', made, 'new', f'{time.time() - t0:.0f}s', flush=True)
        log['datasets'][ds]['clean_manifest_sha256'] = sha(OUT / 'manifests' / ds / 'clean.jsonl')
    json.dump(log, open(logp, 'w'), indent=1)


if __name__ == '__main__':
    main()
