#!/usr/bin/env python
"""Build Task-A items and Task-B pairwise items for the image-grounded human study.

Outputs (artifacts_cvpr_final/human_eval/image_grounded/):
  items.jsonl, pairwise_items.jsonl            annotator-facing (no system identity / metrics / error type)
  pilot_items.jsonl, pilot_pairwise_items.jsonl  single-annotator pilot subset (150 items + 60 pairs)
  hidden/answer_key.jsonl                      system, perturbation type, metric scores + metric preferences
  build_manifest.json                          seed + sampling procedure + counts
  images/<dataset>/*.jpg                       downscaled (<=448px) T1/T2 thumbnails
NO human ratings are created here. Everything is real data already on disk.
"""
import argparse, json, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ricce.evaluation.human_protocol import PAIR_CATEGORIES

SEED = 20260923
DATASETS = ["LEVIR_CC", "SECOND_CC", "DUBAI_CC"]
SYSTEMS = ["qwen", "sft_seed42", "chg2cap", "swmcc"]
MIN_UNIQUE_RATIO = 0.05      # systems with < 5% unique outputs on a dataset are treated as collapsed and excluded
MAX_SAME_TEXT = 3            # at most 3 real items with identical candidate text per (dataset, system)
MAX_CAND_CHARS = 700
THUMB = 448
TIE_EPS = 1e-6

# per-dataset hard-case quotas (perturbation-derived; E04) -> total 204 per dataset
HARD_QUOTA = {
    "correct": (["P0_original"], 18), "paraphrase": (["P0_paraphrase"], 18),
    "action_reversal": (["P4_action_reversal"], 22), "location_swap": (["P5_location_swap"], 20),
    "count_corruption": (["P6_count_corruption"], 20), "omission": (["P7_event_omission"], 20),
    "hallucination": (["P8_hallucination"], 22), "no_change_flip": (["P9_change_state_reversal"], 22),
    "coarse_ancestor": (["P1_ancestor", "P2_distant_ancestor"], 22),
    "sibling_substitution": (["P3_sibling"], 20),
}
REAL_PER_DATASET = 136
TYPE2CAT = {"P4_action_reversal": "action_reversal", "P5_location_swap": "wrong_location",
            "P6_count_corruption": "wrong_quantity", "P7_event_omission": "omission",
            "P8_hallucination": "hallucination", "P9_change_state_reversal": "no_change_error"}
COARSE = {"P1_ancestor", "P2_distant_ancestor"}

def load_manifest(ds):
    return {r["image_id"]: r for r in map(json.loads, open(ROOT / f"artifacts/manifests/{ds}.jsonl"))}

def clean(s):
    return re.sub(r"\s+", " ", str(s)).strip()

def pick_refs(man, exclude=None, k=2, rng=None):
    refs = [clean(r) for r in man["references"] if clean(r)]
    pool = [r for r in refs if exclude is None or r != clean(exclude)] or refs
    pool = list(dict.fromkeys(pool))
    return pool[:k]

def load_extra_metrics(ds, system):
    """BERTScore / SPICE / LLM-judge per item if a conventional_metrics.json exists."""
    p = ROOT / f"results/E08/{ds}/{system}/conventional_metrics.json"
    out = {}
    if p.exists():
        d = json.load(open(p))
        for k, name in [("bertscore_per_item", "bertscore_f"), ("spice_per_item", "spice_f"),
                        ("llm_judge_per_item", "llm_judge")]:
            if isinstance(d.get(k), dict):
                out[name] = d[k]
    return out

def real_frame(ds, system):
    p = ROOT / f"results/E08/{ds}/{system}/per_sample_metrics.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["prediction"] = df["prediction"].fillna("").astype(str)
    if df["prediction"].nunique() / max(len(df), 1) < MIN_UNIQUE_RATIO:
        return None
    ex = load_extra_metrics(ds, system)
    for name, m in ex.items():
        df[name] = df["image_id"].map(m)
    return df

METRIC_COLS = {"ce_f1": "ce_f1", "ce_p": "ce_precision", "ce_r": "ce_recall", "bleu4": "bleu4", "meteor": "meteor",
               "rougeL": "rougeL", "cider_d": "cider_d", "bertscore_f": "bertscore_f",
               "spice_f": "spice_f", "llm_judge": "llm_judge"}

def metrics_of(row):
    o = {}
    for k, c in METRIC_COLS.items():
        if k == "cider_d" and c not in row and "cider_diagnostic" in row:
            c = "cider_diagnostic"   # E04 has only the single-item CIDEr diagnostic
        if c in row and pd.notna(row[c]):
            o[k] = float(row[c])
    return o

def build_items(rng):
    items, keys = [], []
    used_hard = set()
    for ds in DATASETS:
        man = load_manifest(ds)
        e04 = pd.read_csv(ROOT / f"results/E04/{ds}/per_sample_metrics.csv")
        e04 = e04[e04["prediction"].notna()]
        for stratum, (types, n) in HARD_QUOTA.items():
            sub = e04[e04["type"].isin(types)].copy()
            if stratum not in ("correct", "paraphrase"):
                sub = sub[sub["prediction"].map(clean) != sub["gt"].map(clean)]   # perturbation must alter the text
            sub = sub[sub["prediction"].str.len() > 0]
            recs = sub.to_dict("records"); rng.shuffle(recs)
            chosen, per_type = [], Counter()
            recs.sort(key=lambda r: r["image_id"] in used_hard)      # prefer images not yet used (stable)
            if len(types) > 1:  # balance sub-types
                for r in recs:
                    if per_type[r["type"]] < n // len(types) and r["image_id"] in man:
                        chosen.append(r); per_type[r["type"]] += 1
            for r in recs:
                if len(chosen) >= n: break
                if r not in chosen and r["image_id"] in man: chosen.append(r)
            for r in chosen[:n]:
                used_hard.add(r["image_id"])
                m = man[r["image_id"]]
                items.append(dict(dataset=ds, image_id=r["image_id"], t1_path=m["t1_path"], t2_path=m["t2_path"],
                                  candidate=clean(r["prediction"]), references=pick_refs(m, exclude=r["gt"]),
                                  stratum="hard_case"))
                keys.append(dict(kind="item", dataset=ds, image_id=r["image_id"], system=None,
                                 source="E04_perturbation", perturbation_type=r["type"], stratum=f"{ds}/hard/{stratum}",
                                 source_reference=clean(r["gt"]), metrics=metrics_of(r)))
        # real model outputs
        frames = {s: real_frame(ds, s) for s in SYSTEMS}
        active = [s for s, f in frames.items() if f is not None]
        excluded = [s for s in SYSTEMS if s not in active]
        n_each = REAL_PER_DATASET // len(active)
        extra = REAL_PER_DATASET - n_each * len(active)
        for i, s in enumerate(active):
            need = n_each + (1 if i < extra else 0)
            recs = frames[s].to_dict("records"); rng.shuffle(recs)
            seen, cnt = Counter(), 0
            for r in recs:
                if cnt >= need: break
                t = clean(r["prediction"])
                if not t or len(t) > MAX_CAND_CHARS or seen[t] >= MAX_SAME_TEXT or r["image_id"] not in man: continue
                seen[t] += 1; cnt += 1
                m = man[r["image_id"]]
                items.append(dict(dataset=ds, image_id=r["image_id"], t1_path=m["t1_path"], t2_path=m["t2_path"],
                                  candidate=t, references=pick_refs(m), stratum="model_output"))
                keys.append(dict(kind="item", dataset=ds, image_id=r["image_id"], system=s, source="E08_model_output",
                                 perturbation_type=None, stratum=f"{ds}/real/{s}", metrics=metrics_of(r)))
        print(f"[items] {ds}: real systems active={active} excluded_collapsed={excluded}")
    order = list(range(len(items))); rng.shuffle(order)
    out_i, out_k = [], []
    for n, j in enumerate(order):
        iid = f"A{n+1:04d}"
        it, k = dict(items[j]), dict(keys[j])
        it = {"item_id": iid, **it}; k["item_id"] = iid
        out_i.append(it); out_k.append(k)
    return out_i, out_k

# ------------------------------------------------------------------ pairs
NOCHG = re.compile(r"\b(no (significant |noticeable |visible )?(change|difference)s?|nothing (has )?changed|identical|the same|unchanged|no changes)\b", re.I)
REV = [("built|construct|constructed|added|appear|new|erected|replace", "demolish|demolished|destroy|destroyed|removed|disappear|cleared|gone|torn")]
LOCW = re.compile(r"\b(left|right|top|bottom|upper|lower|north|south|east|west|center|centre|corner|middle)\b", re.I)
NUMW = re.compile(r"\b(one|two|three|four|five|six|several|many|few|some|a|single|\d+)\b", re.I)
SPEC = re.compile(r"\b(red|blue|green|white|black|yellow|grey|gray|brown|orange|\d+|two|three|four|five|six|seven|eight|nine|ten|circular|l-shaped|rectangular)\b", re.I)
def toks(s): return set(re.findall(r"[a-z0-9\-]+", s.lower()))
def act_dir(s):
    a = bool(re.search(REV[0][0], s, re.I)); b = bool(re.search(REV[0][1], s, re.I))
    return (a, b)

def heuristic_tags(cand, refs):
    """Lexical heuristics used ONLY to stratify real-model pairs (documented as heuristic, not ground truth)."""
    tags = set(); rt = set().union(*[toks(r) for r in refs]) if refs else set()
    rj = " ".join(refs); ce = toks(cand)
    if bool(NOCHG.search(cand)) != bool(NOCHG.search(rj)): tags.add("no_change_error")
    ca, cb = act_dir(cand); ra, rb = act_dir(rj)
    if (ca and not cb and rb and not ra) or (cb and not ca and ra and not rb): tags.add("action_reversal")
    if set(x.lower() for x in LOCW.findall(cand)) and set(x.lower() for x in LOCW.findall(cand)).isdisjoint(x.lower() for x in LOCW.findall(rj)) and LOCW.search(rj): tags.add("wrong_location")
    cn = {x.lower() for x in NUMW.findall(cand)} - {"a"}; rn = {x.lower() for x in NUMW.findall(rj)} - {"a"}
    if cn and rn and cn.isdisjoint(rn): tags.add("wrong_quantity")
    sp = {x.lower() for x in SPEC.findall(cand)} - rt
    if sp: tags.add("unsupported_specificity")
    if len(ce) < 0.5 * min(len(toks(r)) for r in refs) if refs else False: tags.add("omission")
    if len(ce - rt) >= 6 and len(ce) > 1.8 * max(len(toks(r)) for r in refs): tags.add("hallucination")
    if len(re.findall(r"\b(and|while|,)\b", rj)) >= 2 and re.search(r"\band\b", rj): tags.add("multi_event_binding")
    return tags

PREF_METRICS = ["ce_f1", "bleu4", "meteor", "rougeL", "cider_d", "bertscore_f", "spice_f", "llm_judge"]

def metric_prefs(ma, mb):
    """Preference of each metric for the displayed A/B. 'tie' if |diff| <= 1e-6 ; None if unavailable."""
    out = {}
    for k in PREF_METRICS:
        if k in ma and k in mb:
            d = ma[k] - mb[k]
            out[k] = "tie" if abs(d) <= TIE_EPS else ("A" if d > 0 else "B")
        else:
            out[k] = None
    return out

def agreement_group(p):
    cf = p.get("ce_f1")
    if cf in (None, "tie"): return None
    opp = [k for k in ("cider_d", "bertscore_f") if p.get(k) not in (None, "tie") and p[k] != cf]
    if opp: return "disagree"
    ref = [k for k in ("cider_d", "bleu4", "rougeL", "bertscore_f") if p.get(k) is not None]
    if ref and all(p[k] == cf for k in ref): return "agree"
    return None

def pert_pair_category(ta, tb):
    ts = {ta, tb}
    errs = [t for t in ts if t in TYPE2CAT]
    if len(errs) == 2 or not errs:
        return None
    e = errs[0]; other = (ts - {e}).pop()
    if other in COARSE: return "coarse_but_correct"
    return TYPE2CAT[e]

def build_pairs(rng, items_key):
    cands = []   # dicts
    for ds in DATASETS:
        man = load_manifest(ds)
        e04 = pd.read_csv(ROOT / f"results/E04/{ds}/per_sample_metrics.csv")
        e04 = e04[e04["prediction"].notna()]
        for iid, g in e04.groupby("image_id"):
            if iid not in man: continue
            rows = g.to_dict("records")
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    a, b = rows[i], rows[j]
                    cat = pert_pair_category(a["type"], b["type"])
                    if cat is None or clean(a["prediction"]) == clean(b["prediction"]): continue
                    cands.append(dict(dataset=ds, image_id=iid, src="pert", a=a, b=b, category=cat,
                                      ta=a["type"], tb=b["type"], gt=a["gt"]))
        frames = {s: real_frame(ds, s) for s in SYSTEMS}
        idx = {s: f.set_index("image_id") for s, f in frames.items() if f is not None}
        sysl = list(idx)
        common = set.intersection(*[set(v.index) for v in idx.values()]) if idx else set()
        for iid in sorted(common):
            if iid not in man: continue
            refs = pick_refs(man[iid])
            for i in range(len(sysl)):
                for j in range(i + 1, len(sysl)):
                    ra, rb = idx[sysl[i]].loc[iid].to_dict(), idx[sysl[j]].loc[iid].to_dict()
                    ta_, tb_ = clean(ra["prediction"]), clean(rb["prediction"])
                    if ta_ == tb_ or not ta_ or not tb_ or len(ta_) > MAX_CAND_CHARS or len(tb_) > MAX_CAND_CHARS: continue
                    tg = heuristic_tags(ta_, refs) | heuristic_tags(tb_, refs)
                    if not tg: continue
                    cands.append(dict(dataset=ds, image_id=iid, src="real", a=dict(ra, prediction=ta_), b=dict(rb, prediction=tb_),
                                      sa=sysl[i], sb=sysl[j], tags=sorted(tg), category=None))
    return cands

def select_pairs(cands, rng, n_dis_per_cat=40, n_agree_per_cat=13):
    """Stratified selection: (category x agreement group), <=1 pair per (dataset,image_id), datasets balanced."""
    for c in cands:
        ma, mb = metrics_of(c["a"]), metrics_of(c["b"])
        c["ma"], c["mb"] = ma, mb
        c["prefs"] = metric_prefs(ma, mb); c["group"] = agreement_group(c["prefs"])
    cands = [c for c in cands if c["group"]]
    buckets = defaultdict(list)
    for c in cands:
        cats = [c["category"]] if c["category"] else [t for t in c["tags"] if t in PAIR_CATEGORIES]
        for cat in cats:
            buckets[(cat, c["group"])].append((cat, c))
    chosen, used = [], set()
    for grp, quota in (("disagree", n_dis_per_cat), ("agree", n_agree_per_cat)):
        for cat in PAIR_CATEGORIES:
            pool = buckets.get((cat, grp), [])[:]
            rng.shuffle(pool)
            # balance datasets round-robin; prefer perturbation-derived (known error type) up to 60%
            byds = defaultdict(list)
            for cat_, c in pool: byds[c["dataset"]].append((cat_, c))
            got = 0; dss = list(DATASETS); k = 0; stalls = 0
            while got < quota and stalls < 3:
                ds = dss[k % 3]; k += 1
                lst = byds[ds]; took = False
                while lst:
                    cat_, c = lst.pop()
                    key = (c["dataset"], c["image_id"])
                    if key in used: continue
                    used.add(key); chosen.append((cat_, c)); got += 1; took = True; break
                stalls = 0 if took else stalls + 1
    return chosen

def make_pair_records(chosen, rng, man_cache):
    recs, keys = [], []
    order = list(range(len(chosen))); rng.shuffle(order)
    for n, j in enumerate(order):
        cat, c = chosen[j]
        man = man_cache[c["dataset"]][c["image_id"]]
        swap = rng.random() < 0.5
        A, B = (c["b"], c["a"]) if swap else (c["a"], c["b"])
        ma, mb = (c["mb"], c["ma"]) if swap else (c["ma"], c["mb"])
        if c["src"] == "pert":
            ida, idb = (c["tb"], c["ta"]) if swap else (c["ta"], c["tb"])
            refs = pick_refs(man, exclude=c["gt"]); sysA = sysB = None
        else:
            sa, sb = (c["sb"], c["sa"]) if swap else (c["sa"], c["sb"])
            ida = idb = None; sysA, sysB = sa, sb; refs = pick_refs(man)
        pid = f"P{n+1:04d}"
        recs.append(dict(pair_id=pid, dataset=c["dataset"], image_id=c["image_id"], t1_path=man["t1_path"], t2_path=man["t2_path"],
                         candidate_a=clean(A["prediction"]), candidate_b=clean(B["prediction"]), references=refs, stratum="pair"))
        keys.append(dict(kind="pair", item_id=pid, dataset=c["dataset"], image_id=c["image_id"], source=c["src"],
                         category=cat, category_is_heuristic=(c["src"] == "real"), agreement_group=c["group"],
                         swapped_ab=swap, system_a=sysA, system_b=sysB, perturbation_type_a=ida, perturbation_type_b=idb,
                         metrics_a=ma, metrics_b=mb, metric_preferences=metric_prefs(ma, mb),
                         tie_policy=f"|metric_A - metric_B| <= {TIE_EPS} counts as 'tie'"))
    return recs, keys

# ------------------------------------------------------------------ images
def thumb(src, dst):
    if dst.exists(): return
    dst.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(src)
    if im.mode not in ("RGB",):
        im = im.convert("RGB")
    im.thumbnail((THUMB, THUMB), Image.LANCZOS)
    im.save(dst, "JPEG", quality=72)

def attach_thumbs(recs, out):
    for r in recs:
        base = re.sub(r"[^A-Za-z0-9_\-]", "_", r["image_id"])
        for side, k in (("A", "t1_path"), ("B", "t2_path")):
            rel = f"images/{r['dataset']}/{base}_{side}.jpg"
            thumb(r[k], out / rel); r["t1_thumb" if side == "A" else "t2_thumb"] = rel

def pilot_subset(items, keys, pairs, pkeys, rng, n_items=150, n_pairs=60):
    kmap = {k["item_id"]: k for k in keys}
    groups = defaultdict(list)
    for it in items: groups[kmap[it["item_id"]]["stratum"].rsplit("/", 1)[0] + "/" + kmap[it["item_id"]]["stratum"].split("/")[-1] if False else (it["dataset"], kmap[it["item_id"]]["perturbation_type"] or "real")].append(it)
    sel, gl = [], list(groups.values())
    for g in gl: rng.shuffle(g)
    while len(sel) < n_items and any(gl):
        for g in gl:
            if g and len(sel) < n_items: sel.append(g.pop())
    pk = {k["item_id"]: k for k in pkeys}
    pg = defaultdict(list)
    for p in pairs: pg[(pk[p["pair_id"]]["category"], pk[p["pair_id"]]["agreement_group"])].append(p)
    psel, gl = [], list(pg.values())
    for g in gl: rng.shuffle(g)
    while len(psel) < n_pairs and any(gl):
        for g in gl:
            if g and len(psel) < n_pairs: psel.append(g.pop())
    return sorted(sel, key=lambda x: x["item_id"]), sorted(psel, key=lambda x: x["pair_id"])

def wjsonl(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "artifacts_cvpr_final/human_eval/image_grounded"))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-images", action="store_true", help="skip thumbnail generation")
    a = ap.parse_args()
    out = Path(a.out); rng = random.Random(a.seed)
    items, ikeys = build_items(rng)
    cands = build_pairs(rng, ikeys)
    chosen = select_pairs(cands, rng)
    man_cache = {ds: load_manifest(ds) for ds in DATASETS}
    pairs, pkeys = make_pair_records(chosen, rng, man_cache)
    pi, pp = pilot_subset(items, ikeys, pairs, pkeys, random.Random(a.seed + 1))
    if not a.no_images:
        seen = {}
        for r in items + pairs: pass
        attach_thumbs(items, out); attach_thumbs(pairs, out)
        tmap = {r["item_id"]: (r["t1_thumb"], r["t2_thumb"]) for r in items}
        tmap.update({r["pair_id"]: (r["t1_thumb"], r["t2_thumb"]) for r in pairs})
    wjsonl(out / "items.jsonl", items); wjsonl(out / "pairwise_items.jsonl", pairs)
    wjsonl(out / "pilot_items.jsonl", pi); wjsonl(out / "pilot_pairwise_items.jsonl", pp)
    wjsonl(out / "hidden/answer_key.jsonl", ikeys + pkeys)
    strata = Counter(k["stratum"] for k in ikeys)
    pc = Counter((k["category"], k["agreement_group"], k["source"]) for k in pkeys)
    manifest = dict(seed=a.seed, n_items=len(items), n_pairs=len(pairs), n_pilot_items=len(pi), n_pilot_pairs=len(pp),
        items_per_dataset=dict(Counter(i["dataset"] for i in items)),
        items_per_stratum=dict(sorted(strata.items())),
        pairs_per_category_group_source={f"{c}|{g}|{s}": n for (c, g, s), n in sorted(pc.items())},
        pairs_per_dataset=dict(Counter(p["dataset"] for p in pairs)),
        procedure=[
         "rng = random.Random(seed); all sampling uses this single stream (pilot subset uses seed+1).",
         "Task A hard cases: per dataset, per perturbation stratum quota drawn from results/E04/<DS>/per_sample_metrics.csv (train-split images), "
         "candidate must differ from its source reference (except correct/paraphrase); images not yet used by other hard strata preferred.",
         f"Task A real outputs: 136 per dataset, split equally over systems {SYSTEMS}; a system is excluded on a dataset if unique-output ratio < {MIN_UNIQUE_RATIO} "
         f"(collapsed); max {MAX_SAME_TEXT} identical texts per (dataset,system); candidates > {MAX_CAND_CHARS} chars dropped. TCPO/DPO variants are not used.",
         "Item ids are assigned after a global shuffle; annotator-facing files contain no system, perturbation type, metric value.",
         "Task B: candidate pairs on the same (dataset,image): (i) two E04 perturbation variants (error type vs. correct/paraphrase/sibling/coarse), "
         "(ii) two real systems. Metric preference per metric from per-sample scores; tie if |diff|<=1e-6 (tie policy, recorded).",
         "Group 'disagree': CE-F has a strict preference and CIDEr-D and/or BERTScore has the strict opposite preference. Group 'agree' (control): CE-F strict and every available "
         "of {CIDEr-D, BLEU-4, ROUGE-L, BERTScore} strict-same. Quota 40 disagree + 13 agree per category, round-robin over datasets, one pair per image.",
         "Categories for perturbation pairs come from the known error type (coarse_but_correct = coarse ancestor vs an error type). Categories for real-model pairs "
         "are lexical HEURISTICS (regex on candidate vs references) and flagged category_is_heuristic=true.",
         "A/B order randomised per pair with the recorded rng; swapped_ab stored in the hidden key.",
         "No E04 perturbation for 'unsupported_specificity' exists; that category is populated only by heuristic real-model pairs (may be under quota).",
         "BERTScore/SPICE/LLM-judge are available only where results/E08 conventional_metrics.json exist (LEVIR_CC real outputs); missing = absent from metrics dict."])
    (out / "build_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "procedure"}, indent=1))

if __name__ == "__main__":
    main()
