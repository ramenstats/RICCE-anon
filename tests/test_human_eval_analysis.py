"""Tests for ricce.evaluation.human_analysis.
ALL ratings here are SYNTHETIC (generated in-test from a seeded RNG, annotator ids start with SYNTHETIC_) and exist only to
exercise the statistics code. They are never written outside pytest tmp dirs."""
import json, os, shutil, subprocess
from pathlib import Path
import numpy as np, pandas as pd, pytest
from ricce.evaluation import human_analysis as ha
from ricce.evaluation.human_protocol import AXES, CSV_COLUMNS

ROOT = Path(__file__).resolve().parents[1]

def synth_key(n_items=60, n_pairs=40, seed=0):
    rng = np.random.default_rng(seed); rows = []; truth = {}
    for i in range(n_items):
        q = int(rng.integers(1, 6)); iid = f"A{i:04d}"; truth[iid] = q
        real = i % 2 == 0
        rows.append(dict(kind="item", item_id=iid, dataset=["LEVIR_CC", "SECOND_CC"][i % 2], image_id=f"img{i // 2}",
                         system="sysX" if real else None, source="E08_model_output" if real else "E04_perturbation", perturbation_type=None,
                         metrics=dict(ce_f1=q / 5 + rng.normal(0, .05), ce_p=q / 5 + rng.normal(0, .1), ce_r=q / 5 + rng.normal(0, .1),
                                      bleu4=float(rng.random()))))
    ptruth = {}
    for j in range(n_pairs):
        pid = f"P{j:04d}"; h = ["A", "B"][j % 2]; ptruth[pid] = h
        rows.append(dict(kind="pair", item_id=pid, dataset="LEVIR_CC", image_id=f"img{j}", source="pert", category=["omission", "wrong_location"][j % 2],
                         agreement_group=["agree", "disagree"][(j // 2) % 2],
                         metric_preferences=dict(ce_f1=h if j % 5 else "tie", bleu4=["A", "B"][(j + 1) % 2])))
    return rows, truth, ptruth

def write_csv(path, aid, truth, ptruth, noise, seed):
    rng = np.random.default_rng(seed); rows = []
    for iid, q in truth.items():
        r = {c: "" for c in CSV_COLUMNS}; r.update(annotator_id=aid, task="A", item_id=iid)
        for a in AXES:
            r[a] = int(np.clip(round(q + rng.normal(0, noise)), 1, 5)) if noise >= 0 else int(rng.integers(1, 6))
        r["location"] = "NA"
        rows.append(r)
    for pid, h in ptruth.items():
        r = {c: "" for c in CSV_COLUMNS}; r.update(annotator_id=aid, task="B", item_id=pid, preference=h if noise >= 0 else str(rng.choice(["A", "B"])))
        rows.append(r)
    pd.DataFrame(rows).to_csv(path, index=False)

@pytest.fixture
def study(tmp_path):
    rows, truth, ptruth = synth_key()
    key = tmp_path / "key.jsonl"; key.write_text("\n".join(json.dumps(r) for r in rows))
    return tmp_path, key, truth, ptruth

def test_alpha_identical_is_one_random_near_zero():
    rng = np.random.default_rng(1); x = rng.integers(1, 6, 300).astype(float)
    same = pd.DataFrame([x, x, x]); assert ha.krippendorff_ordinal(same) == pytest.approx(1.0)
    rnd = pd.DataFrame([rng.integers(1, 6, 300), rng.integers(1, 6, 300), rng.integers(1, 6, 300)]).astype(float)
    assert abs(ha.krippendorff_ordinal(rnd)) < 0.2
    assert np.isnan(ha.krippendorff_ordinal(pd.DataFrame([x])))       # <2 annotators

def test_fleiss_and_icc():
    rng = np.random.default_rng(2); lab = rng.choice(["A", "B", "Tie"], 100)
    k, n = ha.fleiss_kappa_from_labels(pd.DataFrame([lab, lab, lab])); assert k == pytest.approx(1.0) and n == 100
    k2, _ = ha.fleiss_kappa_from_labels(pd.DataFrame([rng.choice(["A", "B"], 400) for _ in range(3)])); assert abs(k2) < 0.15
    x = rng.integers(1, 6, 50).astype(float)
    i1, ik = ha.icc_two_way(np.column_stack([x, x, x])); assert i1 == pytest.approx(1.0) and ik == pytest.approx(1.0)
    i1r, _ = ha.icc_two_way(rng.normal(size=(200, 3))); assert abs(i1r) < 0.2

def test_multiple_testing_and_p_display():
    p = [0.01, 0.04, 0.03, 0.5]
    assert list(ha.holm(p)) == pytest.approx([0.04, 0.09, 0.09, 0.5])
    assert list(ha.bh(p)) == pytest.approx([0.04, 0.0533333, 0.0533333, 0.5], abs=1e-5)
    assert ha.p_display(1 / 10001, 10000).startswith("<") and "0.0000" not in ha.p_display(1e-9, 10000)

def test_cluster_bootstrap_detects_association_and_null():
    rng = np.random.default_rng(3); cl = np.repeat(np.arange(40), 3); z = rng.normal(size=120)
    est, lo, hi, p, ncl = ha.cluster_bootstrap_corr(z, z + rng.normal(0, .3, 120), cl, "spearman", 500, 0)
    assert est > 0.8 and lo > 0.5 and p < 0.01 and ncl == 40
    est, lo, hi, p, _ = ha.cluster_bootstrap_corr(z, rng.normal(size=120), cl, "spearman", 500, 0)
    assert lo < 0 < hi and p > 0.05
    assert p >= 1 / 501

def test_pending_writes_nothing(tmp_path):
    coll = tmp_path / "collected"; coll.mkdir(); out = tmp_path / "res"
    assert ha.run_analysis(coll, tmp_path / "nokey.jsonl", out)["status"] == ha.PENDING
    assert not out.exists()
    # header-only CSV (template, no ratings) is also 'pending'
    pd.DataFrame(columns=CSV_COLUMNS).to_csv(coll / "empty.csv", index=False)
    assert ha.run_analysis(coll, tmp_path / "nokey.jsonl", out)["status"] == ha.PENDING and not out.exists()
    # ids with SYNTHETIC prefix refused without the explicit test-only flag
    rows, truth, pt = synth_key(); write_csv(coll / "s.csv", "SYNTHETIC_1", truth, pt, 0.3, 1)
    key = tmp_path / "k.jsonl"; key.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError): ha.run_analysis(coll, key, out)
    assert not out.exists()

def test_three_annotator_pipeline(study):
    tmp, key, truth, pt = study; coll = tmp / "collected"; coll.mkdir()
    for i in range(3): write_csv(coll / f"a{i}.csv", f"SYNTHETIC_{i}", truth, pt, 0.4, 10 + i)
    s = ha.run_analysis(coll, key, tmp / "res", B=200, kendall_B=100, allow_synthetic=True)
    assert s["status"] == "ANALYSED" and s["n_annotators"] == 3
    for f in ["summary.json", "consensus_items.csv", "reliability.csv", "association_overall_all.csv", "association_targeted.csv",
              "association_by_dataset.csv", "pairwise_accuracy.csv", "pairwise_accuracy_by_category.csv", "regression.csv"]:
        assert (tmp / "res" / f).exists(), f
    rel = pd.read_csv(tmp / "res/reliability.csv"); assert rel[rel.measure == "krippendorff_alpha_ordinal:overall"].value.iloc[0] > 0.5
    assoc = pd.read_csv(tmp / "res/association_overall_all.csv")
    assert assoc[assoc.metric == "ce_f1"].spearman.iloc[0] > assoc[assoc.metric == "bleu4"].spearman.iloc[0]
    assert {"spearman_p_holm", "spearman_p_bh"} <= set(assoc.columns)
    pa = pd.read_csv(tmp / "res/pairwise_accuracy.csv")
    assert pa[(pa.metric == "ce_f1") & (pa.tie_policy == "excluded")].accuracy.iloc[0] == pytest.approx(1.0)   # ce_f1 built to equal truth
    hp = pa[(pa.metric == "ce_f1") & (pa.tie_policy == "half_credit")].accuracy.iloc[0]; assert hp < 1.0       # ties get 0.5
    assert "0.0000" not in open(tmp / "res/association_overall_all.csv").read()

def test_single_annotator_labelled(study):
    tmp, key, truth, pt = study; coll = tmp / "collected"; coll.mkdir()
    write_csv(coll / "a.csv", "SYNTHETIC_solo", truth, pt, 0.3, 5)
    s = ha.run_analysis(coll, key, tmp / "res", B=100, kendall_B=50, allow_synthetic=True)
    assert s["study_label"] == "single-annotator, no reliability estimate"
    rel = pd.read_csv(tmp / "res/reliability.csv"); assert rel.value.isna().all() and "single-annotator" in rel.note.iloc[0]
    assert (pd.read_csv(tmp / "res/consensus_items.csv").study_label == s["study_label"]).all()

def test_random_annotators_low_reliability(study):
    tmp, key, truth, pt = study; coll = tmp / "collected"; coll.mkdir()
    for i in range(3): write_csv(coll / f"r{i}.csv", f"SYNTHETIC_r{i}", truth, pt, -1, 20 + i)
    ha.run_analysis(coll, key, tmp / "res", B=50, kendall_B=20, allow_synthetic=True)
    rel = pd.read_csv(tmp / "res/reliability.csv"); assert abs(rel[rel.measure == "krippendorff_alpha_ordinal:overall"].value.iloc[0]) < 0.35

def test_html_tool_js_logic():
    node = shutil.which("node") or str(Path.home() / ".local/lib/python3.12/site-packages/nodejs_wheel/bin/node")
    html = ROOT / "artifacts_cvpr_final/human_eval/image_grounded/annotator.html"
    if not (os.path.exists(node) and html.exists()): pytest.skip("node or built annotator.html unavailable")
    r = subprocess.run([node, str(ROOT / "tests/js_export_check.js"), str(html)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
