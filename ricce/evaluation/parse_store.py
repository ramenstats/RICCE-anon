"""Disk-backed parse cache: {mode: {text: [event dict, ...]}} with parallel bulk parsing."""
import json, os
from pathlib import Path

_W = {}


def _init():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    torch.set_num_threads(1)
    from ..benchmarks.env import make_parser
    _W['p'] = make_parser()


def _parse(args):
    text, multi = args
    return text, [e.to_dict() for e in _W['p'].parse(text, allow_multi_event=multi)]


class ParseStore:
    def __init__(self, path):
        self.path = Path(path)
        self.d = {'multi': {}, 'single': {}}
        if self.path.exists():
            self.d.update(json.load(open(self.path)))

    def save(self):
        json.dump(self.d, open(self.path, 'w'))

    def seed(self, mode, text, events):
        self.d[mode].setdefault(text, events)

    def ensure(self, texts, mode, n_proc=32):
        need = sorted({t for t in texts if t not in self.d[mode]})
        if need:
            import multiprocessing as mp
            with mp.get_context('fork').Pool(n_proc, initializer=_init) as pool:
                for t, ev in pool.imap_unordered(_parse, [(t, mode == 'multi') for t in need], chunksize=64):
                    self.d[mode][t] = ev
        return len(need)

    def events(self, text, mode):
        from ..schema import Event
        return [Event(**e) for e in self.d[mode][text]]
