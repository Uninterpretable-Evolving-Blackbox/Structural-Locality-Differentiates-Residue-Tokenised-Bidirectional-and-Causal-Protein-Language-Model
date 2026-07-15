#!/usr/bin/env python3
"""
summarize_multiseed_extensions.py
=================================
Aggregate the headline scalar(s) of each extension experiment across SAE init
seeds {42, 43, 44} at the headline cells and report mean + cross-seed SD.

seed 42 lives in the canonical results_<stem>/<cell>/ dirs; seeds 43/44 live in
results_<stem>_seed<seed>/<cell>/ (written by run_multiseed_extensions_*.sh).

Reports mean and sample SD (ddof=1) across the seeds present, plus the raw
values so anything can be recomputed. Missing seeds are skipped and counted.
Writes results_multiseed_extensions/summary.json + prints a table.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_multiseed_extensions"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 43, 44]


def sdir(stem, cell, seed):
    return ROOT / (f"results_{stem}/{cell}" if seed == 42
                   else f"results_{stem}_seed{seed}/{cell}")


def load(stem, cell, seed, fname="summary.json"):
    p = sdir(stem, cell, seed) / fname
    if not p.exists():
        return None
    return json.loads(p.read_text())


def probe_auroc(d, task, feat):
    for r in d.get("results", []):
        if r.get("task") == task and r.get("features") == feat:
            return r.get("auroc")
    return None


# (label, stem, cell, fname, extractor) -> one scalar per seed
METRICS = [
    ("null: real struct-delta mean",       "null", "esm2_l16", "summary.json", lambda d: d["real"]["mean"]),
    ("null: real/global-shuffle ratio",    "null", "esm2_l16", "summary.json", lambda d: d["global_shuffle"]["real_to_null_ratio"]),
    ("null(RITA): real struct-delta mean", "null", "rita_l18", "summary.json", lambda d: d["real"]["mean"]),
    ("diag: val_EV",                        "sae_diagnostics", "esm2_l16", "diagnostics.json", lambda d: d["val_EV_meta"]),
    ("diag: median firing rate",            "sae_diagnostics", "esm2_l16", "diagnostics.json", lambda d: d["firing"]["median_firing_rate"]),
    ("diag: mean max decoder cosine",       "sae_diagnostics", "esm2_l16", "diagnostics.json", lambda d: d["geometry"]["mean_max_cosine"]),
    ("diag(RITA): val_EV",                  "sae_diagnostics", "rita_l18", "diagnostics.json", lambda d: d["val_EV_meta"]),
    ("interp: rho(struct_delta,conceptF1)", "interp_comparison", "esm2_l16", "summary.json", lambda d: d["cross_metric_spearman"]["struct_delta"]["concept_valF1"]),
    ("interp: rho(RSA,coord)",              "interp_comparison", "esm2_l16", "summary.json", lambda d: d["cross_metric_spearman"]["|spearman_rsa|"]["|spearman_coord|"]),
    ("probe: helix raw AUROC",              "probe", "esm2_l16", "summary.json", lambda d: probe_auroc(d, "helix", "raw")),
    ("probe: helix SAE AUROC",              "probe", "esm2_l16", "summary.json", lambda d: probe_auroc(d, "helix", "sae")),
    ("probe(RITA): helix raw AUROC",        "probe", "rita_l18", "summary.json", lambda d: probe_auroc(d, "helix", "raw")),
    ("E2 causal: p(struct more harmful)",   "causal", "esm2_l16", "summary.json", lambda d: d["p_struct_more_harmful"]),
    ("E2 causal: mean delta struct",        "causal", "esm2_l16", "summary.json", lambda d: d["mean_delta_struct"]),
    ("E2 causal: mean delta control",       "causal", "esm2_l16", "summary.json", lambda d: d["mean_delta_control"]),
    ("E3 steer: structural slope",          "steering_sweep", "esm2_l16", "summary.json", lambda d: d["slopes_contact_specificity"]["structural"]),
    ("E3 steer: random slope",              "steering_sweep", "esm2_l16", "summary.json", lambda d: d["slopes_contact_specificity"]["random"]),
    ("E3 steer: p(struct>random @4x)",      "steering_sweep", "esm2_l16", "summary.json", lambda d: d["bootstrap"]["structural_vs_random@4.0x"]["p_diff_le_0"]),
    ("faith: mean delta CE",                "faithfulness", "esm2_l16", "summary.json", lambda d: d["mean_delta_ce"]),
    ("faith: mean loss_recovered",          "faithfulness", "esm2_l16", "summary.json", lambda d: d["mean_loss_recovered"]),
]


def main():
    rows = []
    for label, stem, cell, fname, extract in METRICS:
        vals, seeds_present = [], []
        for s in SEEDS:
            d = load(stem, cell, s, fname)
            if d is None:
                continue
            try:
                v = extract(d)
            except (KeyError, TypeError, IndexError):
                v = None
            if v is not None and np.isfinite(v):
                vals.append(float(v)); seeds_present.append(s)
        arr = np.array(vals, dtype=float)
        row = {
            "metric": label, "stem": stem, "cell": cell,
            "seeds_present": seeds_present, "n": len(arr),
            "values": [float(x) for x in arr],
            "mean": float(arr.mean()) if len(arr) else None,
            "sd_ddof1": float(arr.std(ddof=1)) if len(arr) > 1 else (0.0 if len(arr) == 1 else None),
        }
        rows.append(row)

    (OUT / "summary.json").write_text(json.dumps({"seeds": SEEDS, "rows": rows}, indent=2))

    w = max(len(r["metric"]) for r in rows)
    print(f"{'metric':<{w}}  n  {'mean':>10}  {'SD(ddof1)':>10}   values")
    print("-" * (w + 40))
    for r in rows:
        m = f"{r['mean']:.4g}" if r["mean"] is not None else "--"
        sd = f"{r['sd_ddof1']:.4g}" if r["sd_ddof1"] is not None else "--"
        vv = ", ".join(f"{v:.4g}" for v in r["values"]) if r["values"] else "(none yet)"
        flag = "" if r["n"] == 3 else f"  [<-- {r['n']}/3 seeds]"
        print(f"{r['metric']:<{w}}  {r['n']}  {m:>10}  {sd:>10}   {vv}{flag}")
    print(f"\nWrote {OUT/'summary.json'}")


if __name__ == "__main__":
    main()
