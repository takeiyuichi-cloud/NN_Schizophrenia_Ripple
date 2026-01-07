#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig4b: Inter-event intervals (IEI) panel (0–500 ms)
Single-table summary version (NO N columns)

Input:
  data/IEI_pairs_subject_medians_0to500ms.csv
Required columns:
  subject, group, pair, med

Output:
  outputs/tables/Fig4b_IEI_summary_0to500ms.csv
  outputs/figures/Fig4b_IEI_barplot.pdf
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
import openpyxl

PAIR_ORDER = [
    "HIP→HIP (same)",
    "HIP→HIP (opposite)",
    "CTX→HIP",
    "HIP→CTX",
    "CTX→CTX",
]

PAIR_ALIASES = {
    "HIP→HIP(same)": "HIP→HIP (same)",
    "HIP→HIP (same hemisphere)": "HIP→HIP (same)",
    "HIP→HIP(opposite)": "HIP→HIP (opposite)",
    "HIP→HIP (opposite hemisphere)": "HIP→HIP (opposite)",
    "CTX→HIP": "CTX→HIP",
    "HIP→CTX": "HIP→CTX",
    "CTX→CTX": "CTX→CTX",
}

COLOR = {"HC": "#4C72B0", "SZ": "#C44E52"}

def norm_group(g):
    s = str(g).strip().upper()
    if s in ("SC", "SCHIZOPHRENIA"):
        return "SZ"
    if s in ("HEALTHY", "CONTROL"):
        return "HC"
    return s

def norm_pair(p):
    return PAIR_ALIASES.get(str(p).strip(), str(p).strip())

def mean_se(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    m = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(x.size)) if x.size >= 2 else np.nan
    return m, se

def perm_test(hc, sz, n_perm=10000, seed=0):
    rng = np.random.default_rng(seed)
    hc = np.asarray(hc, float); hc = hc[np.isfinite(hc)]
    sz = np.asarray(sz, float); sz = sz[np.isfinite(sz)]
    if hc.size < 2 or sz.size < 2:
        return np.nan, np.nan
    obs = float(np.mean(sz) - np.mean(hc))
    pooled = np.concatenate([hc, sz])
    nx = hc.size
    diffs = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(pooled)
        diffs[i] = pooled[nx:].mean() - pooled[:nx].mean()
    p = (np.abs(diffs) >= abs(obs)).mean()
    p = max(p, 1.0 / n_perm)
    return obs, p

def stars(q):
    if not np.isfinite(q):
        return ""
    if q < 1e-4:
        return "****"
    if q < 1e-3:
        return "***"
    if q < 1e-2:
        return "**"
    if q < 0.05:
        return "*"
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None)
    ap.add_argument("--input", type=str, default=None)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve() if args.root else Path(__file__).resolve().parents[1]
    in_csv = Path(args.input).expanduser().resolve() if args.input else root / "data" / "IEI_pairs_subject_medians_0to500ms.csv"
    out_tab = root / "outputs" / "tables"
    out_fig = root / "outputs" / "figures"
    out_tab.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    df["group"] = df["group"].map(norm_group)
    df["pair"] = df["pair"].map(norm_pair)
    df["med"] = pd.to_numeric(df["med"], errors="coerce")
    df = df[df["group"].isin(["HC", "SZ"]) & df["pair"].isin(PAIR_ORDER)].dropna(subset=["med"])

    rows = []
    tests = []

    for pair in PAIR_ORDER:
        hc = df.loc[(df["pair"] == pair) & (df["group"] == "HC"), "med"]
        sz = df.loc[(df["pair"] == pair) & (df["group"] == "SZ"), "med"]

        mean_hc, se_hc = mean_se(hc)
        mean_sz, se_sz = mean_se(sz)
        diff, p = perm_test(hc, sz, args.n_perm, args.seed + abs(hash(pair)) % 10000)

        rows.append(dict(
            pair=pair,
            mean_HC=mean_hc,
            se_HC=se_hc,
            mean_SZ=mean_sz,
            se_SZ=se_sz,
            diff_SZ_minus_HC=diff,
            p_perm=p
        ))
        tests.append(p)

    df_out = pd.DataFrame(rows)
    df_out["q_fdr"] = multipletests(df_out["p_perm"], method="fdr_bh")[1]
    df_out["q_fdr_star"] = df_out["q_fdr"].apply(stars)

    df_out.to_csv(out_tab / "Table_S9_IEI_summary_0to500ms.csv", index=False)
    
    # ---- plot ----
    x = np.arange(len(PAIR_ORDER))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.6, 4.8))

    for i, grp in enumerate(["HC", "SZ"]):
        sub = df_out.set_index("pair").loc[PAIR_ORDER]
        y = sub[f"mean_{grp}"]
        e = sub[f"se_{grp}"]
        ax.bar(x + (i - 0.5) * w, y, w, yerr=e, capsize=3,
               color=COLOR[grp], edgecolor="black", linewidth=0.5, label=grp)

    ymax = float((df_out["mean_HC"] + df_out["se_HC"]).max() * 1.2)
    ax.set_ylim(0, ymax)

    for i, r in df_out.iterrows():
        if r["q_fdr"] < 0.05:
            ax.text(PAIR_ORDER.index(r["pair"]), ymax * 0.97,
                    "**" if r["q_fdr"] < 0.01 else "*",
                    ha="center", va="top", fontsize=12)

    ax.set_xticks(x)
    ax.set_xticklabels(PAIR_ORDER)
    ax.set_ylabel("Inter-event intervals (ms)")
    ax.set_title("Inter-event intervals (0–500 ms), subject median")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_fig / "Fig4b_IEI_barplot.pdf", dpi=300)
    plt.close(fig)

    print("[OK] Fig4b IEI summary written")

if __name__ == "__main__":
    main()