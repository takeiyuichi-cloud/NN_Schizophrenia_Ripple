#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig.1 Panel B: Fold-change in ripple counts (SZ / HC)
- One figure per site (Gunma, Kumagaya)
- Hippocampus: yellow
- Cortex: purple

Input:
  NN_open_code/data/df_clean_expanded.csv
Required columns:
  site, subject, group, cond, freq, event_count

Outputs:
  NN_open_code/outputs/figures/Fig1b_fold_change_counts_Gunma.pdf
  NN_open_code/outputs/figures/Fig1b_fold_change_counts_Kumagaya.pdf
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Normalization
# -------------------------

COND_MAP = {
    "hippocampus": "Hippocampus",
    "extrahippocampal": "Cortex",
    "extrahippocampus": "Cortex",
    "extrahippocapus": "Cortex",
    "cortex": "Cortex",
    "cortical": "Cortex",
    "ctx": "Cortex",
    # if present, treat as TOTAL (not used for site-wise plots by default)
    "filtered_ged_meg": "TOTAL",
    "original_meg": "TOTAL",
}

def norm_site(x: str) -> str:
    return str(x).strip().lower()

def display_site(site_raw: str) -> str:
    s = str(site_raw).lower()
    if "gundai" in s or "gunma" in s:
        return "Gunma"
    if "kumasou" in s or "kumagaya" in s:
        return "Kumagaya"
    if s == "total":
        return "TOTAL"
    return site_raw

def norm_group(x: str) -> str:
    s = str(x).strip().upper()
    if s in ("HC", "CONTROL", "HEALTHY"):
        return "HC"
    if s in ("SZ", "SCHIZOPHRENIA", "SC"):
        return "SZ"
    return s

def norm_cond(x: str) -> str:
    s = str(x).strip()
    # handle already-capitalized Extrahippocampal
    if s == "Extrahippocampal":
        return "Cortex"
    return COND_MAP.get(s.lower(), s)


# -------------------------
# Summary + fold change
# -------------------------

def summarize_event_counts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (site, cond, freq, group), sub in df.groupby(["site","cond","freq","group"], dropna=False):
        vals = pd.to_numeric(sub["event_count"], errors="coerce").dropna().to_numpy(float)
        if vals.size == 0:
            continue
        mean = float(vals.mean())
        se = float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size >= 2 else np.nan
        rows.append(dict(site=site, cond=cond, freq=float(freq), group=group, n=int(vals.size), mean=mean, se=se))
    return pd.DataFrame(rows)

def compute_fold_change_sz_over_hc(df_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (site, cond, freq), g in df_summary.groupby(["site","cond","freq"], dropna=False):
        hc = g[g["group"]=="HC"]
        sz = g[g["group"]=="SZ"]
        if hc.empty or sz.empty:
            continue

        mu_hc = float(hc["mean"].iloc[0]); se_hc = float(hc["se"].iloc[0]) if len(hc) else np.nan
        mu_sz = float(sz["mean"].iloc[0]); se_sz = float(sz["se"].iloc[0]) if len(sz) else np.nan

        if (not np.isfinite(mu_hc)) or mu_hc <= 0 or (not np.isfinite(mu_sz)):
            fold, se_fold = np.nan, np.nan
        else:
            fold = mu_sz / mu_hc
            var_hc = (se_hc**2) if np.isfinite(se_hc) else 0.0
            var_sz = (se_sz**2) if np.isfinite(se_sz) else 0.0
            # delta method
            var_fold = (1.0/mu_hc)**2 * var_sz + (mu_sz/(mu_hc**2))**2 * var_hc
            se_fold = float(np.sqrt(var_fold)) if var_fold > 0 else np.nan

        rows.append(dict(site=site, cond=cond, freq=float(freq), fold=fold, se_fold=se_fold,
                         mean_HC=mu_hc, mean_SZ=mu_sz))
    return pd.DataFrame(rows).sort_values(["site","cond","freq"]).reset_index(drop=True)


# -------------------------
# Plot (one figure per site)
# -------------------------

def plot_site_foldchange(df_fold: pd.DataFrame, site: str, out_pdf: Path):
    """
    One figure per site:
      two lines: Hippocampus (yellow), Cortex (purple)
    """
    color_map = {
        "Hippocampus": "#FFD700",  # yellow (gold)
        "Cortex": "#7B2CBF",       # purple
    }

    plt.figure(figsize=(6.8, 4.6))

    for cond in ["Hippocampus", "Cortex"]:
        sub = df_fold[(df_fold["site"] == site) & (df_fold["cond"] == cond)].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("freq")
        x = sub["freq"].to_numpy(float)
        y = sub["fold"].to_numpy(float)
        yerr = sub["se_fold"].to_numpy(float)

        plt.errorbar(
            x, y, yerr=yerr,
            fmt="-o",
            capsize=3,
            label=cond,
            color=color_map[cond],
            ecolor=color_map[cond],
        )

    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Fold-change (SZ / HC)")
    plt.title(f"Fold-change in ripple counts — {display_site(site)}")
    plt.legend(frameon=False)
    plt.tight_layout()

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close()


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None)
    ap.add_argument("--input", type=str, default=None)
    args = ap.parse_args()

    if args.root is not None:
        root = Path(args.root).expanduser().resolve()
    else:
        try:
            root = Path(__file__).resolve().parents[1]
        except NameError:
            root = Path.cwd().resolve()

    input_csv = Path(args.input).expanduser().resolve() if args.input else (root / "data" / "df_clean_expanded.csv")
    out_fig = root / "outputs" / "figures"
    out_tab = root / "outputs" / "tables"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_tab.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    df = df.copy()
    df["site"] = df["site"].map(norm_site)
    df["subject"] = df["subject"].astype(str).str.strip()
    df["group"] = df["group"].map(norm_group)
    df["cond"] = df["cond"].map(norm_cond)
    df["freq"] = pd.to_numeric(df["freq"], errors="coerce")
    df["event_count"] = pd.to_numeric(df["event_count"], errors="coerce")

    df = df.dropna(subset=["site","cond","freq","group","event_count"])
    df = df[df["group"].isin(["HC","SZ"])].copy()
    df = df[df["cond"].isin(["Hippocampus","Cortex"])].copy()

    summary = summarize_event_counts(df)
  

    df_fold = compute_fold_change_sz_over_hc(summary)
   
    # one figure per site
    for site in sorted(df_fold["site"].unique()):
        # restrict to known sites only
        if not (("gundai" in site) or ("kumasou" in site) or (site in ("gunma","kumagaya"))):
            # skip unknown / aggregated labels unless you want them
            continue
        out_pdf = out_fig / f"Fig1b_fold_change_counts_{display_site(site)}.pdf"
        plot_site_foldchange(df_fold, site, out_pdf)

    print("[OK] Saved:")
    print(" -", out_tab / "Fig1b_fold_change_summary_means.csv")
    print(" -", out_tab / "Fig1b_fold_change_sz_over_hc.csv")
    print(" -", out_fig)

if __name__ == "__main__":
    main()
