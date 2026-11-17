#!/usr/bin/env python3
"""Rebuilds results_r1/tcpo_branch_a/severity_causal_ablation_summary.csv
from whatever results_r1/tcpo_branch_a/metrics/<name>/metrics.json
files currently exist on disk. Safe to call at any point mid-matrix -- it
only ever reads what has genuinely finished; it never estimates or
backfills a row for a variant/seed that hasn't been trained+scored yet.
"""
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / 'results_r1' / 'tcpo_branch_a' / 'metrics'
OUT_CSV = ROOT / 'results_r1' / 'tcpo_branch_a' / 'severity_causal_ablation_summary.csv'

# name pattern -> (variant_family, severity_source, seed)
import re
PATTERN = re.compile(r'^(?P<fam>.+)_seed(?P<seed>\d+)$')


def variant_family(name):
    m = PATTERN.match(name)
    fam = m.group('fam') if m else name
    seed = int(m.group('seed')) if m else None
    return fam, seed


SEVERITY_SOURCE = {
    'tcpo_v3': 'true_taxonomy', 'tcpo_severity': 'true_taxonomy',
    'tcpo_random_severity': 'random', 'tcpo_no_severity': 'uniform',
    'tcpo_inverse_severity': 'inverse', 'tcpo_clipped_severity': 'clipped_taxonomy',
    'tcpo_normalized_severity': 'normalized_taxonomy',
    'tcpo_curriculum_severity': 'true_taxonomy_curriculum',
    'dpo_reg': 'n/a_kl_regularized_dpo', 'dpo_stab': 'n/a_unweighted_dpo',
}


def main():
    rows = []
    if not METRICS_DIR.exists():
        print('No metrics yet.'); return
    for d in sorted(METRICS_DIR.iterdir()):
        mf = d / 'metrics.json'
        if not mf.exists():
            continue
        m = json.loads(mf.read_text())
        if m.get('status') != 'complete':
            continue
        name = m['name']
        fam, seed = variant_family(name)
        e08 = m.get('e08', {}) or {}
        e10 = m.get('e10', {}) or {}
        cs = m.get('change_state', {}) or {}
        spice = m.get('spice', {}) or {}
        trc = m.get('training_run_config', {}) or {}
        rows.append({
            'name': name, 'variant_family': fam, 'severity_source': SEVERITY_SOURCE.get(fam, fam), 'seed': seed,
            'mode': trc.get('mode'), 'severity_flag': trc.get('severity'), 'neg_strategy': trc.get('neg_strategy'),
            'kl_coef': trc.get('kl_coef'), 'opt_steps': trc.get('opt_steps'),
            'ce_f1': e08.get('ce_f1'), 'ce_precision': e08.get('ce_precision'), 'ce_recall': e08.get('ce_recall'),
            'hallucination_rate': e08.get('hallucination_rate'), 'omission_rate': e08.get('omission_rate'),
            'bleu4': e08.get('bleu4'), 'meteor': e08.get('meteor'), 'rougeL': e08.get('rougeL'),
            'cider_d_corpus': e08.get('cider_d_corpus'),
            'bertscore_f1': e08.get('bertscore') if isinstance(e08.get('bertscore'), (int, float)) else e08.get('bertscore_f1'),
            'spice_status': spice.get('status'), 'spice': spice.get('spice'),
            'change_state_accuracy': cs.get('change_state_accuracy'),
            'change_recall_accuracy': cs.get('change_recall_accuracy'),
            'nochange_recall_accuracy': cs.get('nochange_recall_accuracy'),
            'e10_directional_inversion_accuracy': e10.get('directional_inversion_accuracy'),
            'e10_no_change_consistency_under_swap': e10.get('no_change_consistency_under_swap'),
            'e10_n_directional_action': e10.get('n_directional_action'),
        })
    if not rows:
        print('No completed variant/seed results yet -- nothing to summarize.')
        return
    fields = list(rows[0].keys())
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'Wrote {len(rows)} rows to {OUT_CSV}')


if __name__ == '__main__':
    main()
