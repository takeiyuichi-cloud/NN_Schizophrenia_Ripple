#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig3c: Peak rate within temporally clustered SWRs
(= max_epoch_rate_hz vs frequency, HC vs SZ)

Inputs (NN_open_code/data/):
  - rate_epoch_subject_level.csv
  - df_clean_expanded.csv   (for mapping site+subject -> group)

Outputs (NN_open_code/outputs/):
  - figures/Fig3c_max_epoch_rate_real_vs_freq.pdf

Stats:
  - permutation test on mean difference (SZ - HC), two-sided
  - Cohen's d (SZ - HC)  ★added
  - BH-FDR across frequencies (within this metric)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests


# -------------------------
# Settings
# -------------------------
FREQ_ORDER = [80, 120, 160, 200, 240]
N_PERM = 10000
SEED = 0

COLOR = {"HC": "#4C72B0", "SZ": "#C44E52"}  # blue / red


# -------------------------
# Helpers
# -------------------------
def canon_subject(x) -> str:
    import re
    s = str(x).strip()
    m = re.search(r"(\d+)", s)
    return f"NB_subject_{int(m.group(1))}" if m else s

def norm_site(x) -> str:
    return str(x).strip().lower()

def norm_group(x) -> str:
    s = str(x).strip().upper()
    if s in ("SC",):
        return "SZ"
    if s in ("SCHIZOPHRENIA",):
        return "SZ"
    if s in ("CONTROL", "HEALTHY"):
        return "HC"
    return s

def mean_sem(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, 0
    m = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(x.size)) if x.size >= 2 else np.nan
    return m, se, int(x.size)

def perm_test_two_sided_mean_diff(hc_vals, sz_vals, n_perm=N_PERM, seed=0):
    """
    Two-sided permutation test on mean difference:
      obs = mean(SZ) - mean(HC)
    p = (count(|perm|>=|obs|)+1)/(n_perm+1)
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(hc_vals, float); x = x[np.isfinite(x)]
    y = np.asarray(sz_vals, float); y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return np.nan, np.nan
    obs = float(np.mean(y) - np.mean(x))
    comb = np.concatenate([x, y])
    nx = x.size
    diffs = np.empty(n_perm, float)
    for i in range(n_perm):
        rng.shuffle(comb)
        diffs[i] = float(np.mean(comb[nx:]) - np.mean(comb[:nx]))
    p = float((np.sum(np.abs(diffs) >= abs(obs)) + 1) / (n_perm + 1))
    return obs, p

def cohens_d_sz_minus_hc(hc_vals, sz_vals):
    """
    Cohen's d (SZ - HC) using pooled SD.
    """
    x = np.asarray(hc_vals, float); x = x[np.isfinite(x)]
    y = np.asarray(sz_vals, float); y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return np.nan
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    denom = (x.size + y.size - 2)
    if denom <= 0:
        return np.nan
    pooled = np.sqrt(((x.size - 1) * vx + (y.size - 1) * vy) / denom)
    if not np.isfinite(pooled) or pooled <= 0:
        return np.nan
    return float((y.mean() - x.mean()) / pooled)


# -------------------------
# Loaders
# -------------------------
def load_rate_epoch_subject(root: Path) -> pd.DataFrame:
    p = root / "data" / "rate_epoch_subject_level.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)

    need = {"site", "subject", "freq", "max_epoch_rate_hz"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"rate_epoch_subject_level.csv missing columns: {sorted(miss)}")

    df = df.copy()
    df["site"] = df["site"].map(norm_site)
    df["subject"] = df["subject"].map(canon_subject)
    df["freq"] = pd.to_numeric(df["freq"], errors="coerce").astype("Int64")
    df["max_epoch_rate_hz"] = pd.to_numeric(df["max_epoch_rate_hz"], errors="coerce")
    return df

def load_group_map(root: Path) -> pd.DataFrame:
    p = root / "data" / "df_clean_expanded.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)

    need = {"site", "subject", "group"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"df_clean_expanded.csv missing columns: {sorted(miss)}")

    g = df.copy()
    g["site"] = g["site"].map(norm_site)
    g["subject"] = g["subject"].map(canon_subject)
    g["group"] = g["group"].map(norm_group)
    g = g[g["group"].isin(["HC", "SZ"])].drop_duplicates(subset=["site", "subject"])
    return g[["site", "subject", "group"]].copy()


# -------------------------
# Stats + plot
# -------------------------
def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in FREQ_ORDER:
        sub = df[df["freq"] == f]
        hc = pd.to_numeric(sub.loc[sub["group"] == "HC", "max_epoch_rate_hz"], errors="coerce").to_numpy(float)
        sz = pd.to_numeric(sub.loc[sub["group"] == "SZ", "max_epoch_rate_hz"], errors="coerce").to_numpy(float)
        hc = hc[np.isfinite(hc)]
        sz = sz[np.isfinite(sz)]

        m_hc, se_hc, n_hc = mean_sem(hc)
        m_sz, se_sz, n_sz = mean_sem(sz)

        obs, p = perm_test_two_sided_mean_diff(hc, sz, n_perm=N_PERM, seed=SEED + int(f))
        d = cohens_d_sz_minus_hc(hc, sz)

        rows.append(dict(
            freq=int(f),
            n_HC=int(n_hc), mean_HC=m_hc, sem_HC=se_hc,
            n_SZ=int(n_sz), mean_SZ=m_sz, sem_SZ=se_sz,
            diff_SZ_minus_HC=float(obs) if np.isfinite(obs) else np.nan,
            cohens_d=float(d) if np.isfinite(d) else np.nan,
            p_perm=float(p) if np.isfinite(p) else np.nan,
        ))

    out = pd.DataFrame(rows)
    out["q_fdr"] = np.nan
    m = out["p_perm"].notna()
    if m.any():
        out.loc[m, "q_fdr"] = multipletests(out.loc[m, "p_perm"].values, method="fdr_bh")[1]
    return out

def plot_fig(stats: pd.DataFrame, out_pdf: Path):
    stats = stats.set_index("freq").reindex(FREQ_ORDER).reset_index()
    x = np.arange(len(FREQ_ORDER))

    fig, ax = plt.subplots(figsize=(6.6, 4.4))

    ax.errorbar(x, stats["mean_HC"], yerr=stats["sem_HC"], fmt="-o", capsize=3,
                label="HC", color=COLOR["HC"])
    ax.errorbar(x, stats["mean_SZ"], yerr=stats["sem_SZ"], fmt="-o", capsize=3,
                label="SZ", color=COLOR["SZ"])

    ax.set_xticks(x)
    ax.set_xticklabels([str(f) for f in FREQ_ORDER])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Max clustered SWRs rate (events/s)")
    ax.set_title("Peak rate within temporally clustered SWRs", loc="left")
    ax.legend(frameon=False)

    # stars based on q_fdr
    for i, f in enumerate(FREQ_ORDER):
        hit = stats[stats["freq"] == f]
        if len(hit) == 1:
            q = float(hit["q_fdr"].iloc[0]) if pd.notna(hit["q_fdr"].iloc[0]) else np.nan
            if np.isfinite(q) and q < 0.05:
                star = "***" if q < 0.001 else ("**" if q < 0.01 else "*")
                ymax = ax.get_ylim()[1]
                ax.text(i, ymax * 1.02, star, ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None)
    args = ap.parse_args()

    if args.root is not None:
        root = Path(args.root).expanduser().resolve()
    else:
        try:
            root = Path(__file__).resolve().parents[1]  # NN_open_code
        except NameError:
            root = Path.cwd().resolve()

    out_tab = root / "outputs" / "tables"
    out_fig = root / "outputs" / "figures"
    out_tab.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)

    real = load_rate_epoch_subject(root)
    gmap = load_group_map(root)

    merged = real.merge(gmap, on=["site", "subject"], how="left")
    merged = merged[merged["group"].isin(["HC", "SZ"])].copy()
    merged = merged[merged["freq"].isin(FREQ_ORDER)].copy()

    stats = compute_stats(merged)


    out_pdf = out_fig / "Fig3c_max_epoch_rate_real_vs_freq.pdf"
    plot_fig(stats, out_pdf)

    print("[OK] Written:")
    print(" -", out_pdf)


if __name__ == "__main__":
    main()
