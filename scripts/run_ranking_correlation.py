"""results/metric_comparison/ranking_correlation.json: Spearman rank
correlation between the 7 LEVIR-CC systems' rankings under CIDEr-D (official,
corpus-level, from each system's existing results/E08/LEVIR_CC/<sys>/metrics.json),
BERTScore (from the sibling conventional_metrics.json written by
scripts/run_conventional_e08.py), and RICCE-F (ce_f1, also already in
metrics.json). SPICE and LLM-Judge system-level means are included too when
available (from the same conventional_metrics.json), but the primary 3-way
comparison was requested -- CIDEr vs BERTScore vs RICCE-F -- does
not depend on them.

Human ranking is NOT included: no human annotation exists yet in this repo
(results/E07 is a template for collecting it, not data), so a human leg of
this correlation is left out entirely rather than invented.
"""
import json
from pathlib import Path
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
SYSTEMS = ['qwen', 'sft_seed42', 'dpo_seed42', 'tcpo_seed42', 'chg2cap', 'swmcc']
OUT_DIR = ROOT / 'results' / 'metric_comparison'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    rows = {}
    for sysname in SYSTEMS:
        sysdir = ROOT / 'results' / 'E08' / 'LEVIR_CC' / sysname
        m = json.loads((sysdir / 'metrics.json').read_text())
        row = {'cider_d': m.get('cider_d_corpus', m.get('cider_d')), 'ricce_f': m.get('ce_f1')}
        conv_path = sysdir / 'conventional_metrics.json'
        if conv_path.exists():
            c = json.loads(conv_path.read_text())
            row['bertscore'] = c.get('bertscore_f')
            row['spice'] = c.get('spice_f')
            row['llm_judge'] = c.get('llm_judge_mean_1to5')
        rows[sysname] = row

    have_bertscore = all(rows[s].get('bertscore') is not None for s in SYSTEMS)
    have_spice = all(rows[s].get('spice') is not None for s in SYSTEMS)
    have_judge = all(rows[s].get('llm_judge') is not None for s in SYSTEMS)

    metrics_present = ['cider_d', 'ricce_f']
    if have_bertscore:
        metrics_present.append('bertscore')
    if have_spice:
        metrics_present.append('spice')
    if have_judge:
        metrics_present.append('llm_judge')

    rankings = {}
    for metric in metrics_present:
        vals = [(s, rows[s][metric]) for s in SYSTEMS]
        vals.sort(key=lambda x: -x[1])
        rankings[metric] = [s for s, _ in vals]

    pairwise = {}
    for i in range(len(metrics_present)):
        for j in range(i + 1, len(metrics_present)):
            a, b = metrics_present[i], metrics_present[j]
            xa = [rows[s][a] for s in SYSTEMS]
            xb = [rows[s][b] for s in SYSTEMS]
            rho, p = spearmanr(xa, xb)
            pairwise[f'{a}_vs_{b}'] = {'spearman_rho': float(rho), 'p_value': float(p), 'n': len(SYSTEMS)}

    out = {
        'systems': SYSTEMS,
        'raw_scores': rows,
        'metrics_available': metrics_present,
        'bertscore_available_for_all_systems': have_bertscore,
        'spice_available_for_all_systems': have_spice,
        'llm_judge_available_for_all_systems': have_judge,
        'human_ranking': None,
        'human_ranking_note': 'No human annotation data exists in this repo yet (results/E07 is an unfilled annotation template); the human leg of this correlation is intentionally omitted rather than fabricated.',
        'rankings_high_to_low': rankings,
        'pairwise_spearman': pairwise,
    }
    out_path = OUT_DIR / 'ranking_correlation.json'
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
