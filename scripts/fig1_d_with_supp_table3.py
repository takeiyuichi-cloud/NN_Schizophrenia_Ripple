#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig.1 Panel D: Event-controlled spectral power (raw / adjusted)

Input (default):
  source_fig5_predicted_share_by_load_symptom_tertiles_d.csv

Required columns:
  site, cond, freq, group, subject,
  log10_psd, n_events_used, concat_duration_s
Optional:
  has_power_csv (boolean)

Outputs:
  Tables:
    outputs/tables/Fig1D_cell_summary_log10_adj.csv
    outputs/tables/Supplementary_Table_S3_psd_raw_vs_event_normalized.csv

  Figures (one per site):
    outputs/figures/Fig1D_power_adj_{Site}.pdf
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

# -------------------------
# Normalization
# -------------------------

COND_MAP = {
    "hippocampus": "Hippocampus",
    "extrahippocampal": "Cortex",
    "extrahippocampal": "Cortex",
    "extrahippocapus": "Cortex",
    "cortex": "Cortex",
    "cortical": "Cortex",
    "ctx": "Cortex",
}

def norm_site(x: str) -> str:
    return str(x).strip().lower()

def display_site(site_raw: str) -> str:
    s = str(site_raw).lower()
    if "gundai" in s or "gunma" in s:
        return "Gunma"
    if "kumasou" in s or "kumagaya" in s:
        return "Kumagaya"
    return site_raw

def norm_group(x: str) -> str:
    s = str(x).strip().upper()
    if s in ("SC",):
        return "SZ"
    if s in ("SCHIZOPHRENIA",):
        return "SZ"
    if s in ("CONTROL", "HEALTHY"):
        return "HC"
    return s

def norm_cond(x: str) -> str:
    s = str(x).strip()
    if s == "Extrahippocampal":
        return "Cortex"
    return COND_MAP.get(s.lower(), s)

def _safe_log10_pos(x):
    x = pd.to_numeric(x, errors="coerce").astype(float)
    return np.where(x > 0, np.log10(x), np.nan)


# -------------------------
# Core: recompute log10_psd_adj (group NOT included)
# -------------------------

def recompute_log10_psd_adj(df_PSD: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute log10_psd_adj as residualized log10_psd:
      log10_psd ~ log10_n_events + log10_dur_s
    within each (site, cond, freq).
    """
    df = df_PSD.copy()

    if ("n_events_used" not in df.columns) or ("concat_duration_s" not in df.columns):
        raise ValueError("Input must contain columns: n_events_used, concat_duration_s")

    if "log10_psd" not in df.columns:
        raise ValueError("Input must contain column: log10_psd")

    df["log10_n_events"] = _safe_log10_pos(df["n_events_used"])
    df["log10_dur_s"]    = _safe_log10_pos(df["concat_duration_s"])
    df["log10_psd_adj"]  = np.nan

    for (site, cond, freq), sub in df.groupby(["site","cond","freq"], dropna=False):
        tmp = sub[["log10_psd","log10_n_events","log10_dur_s"]].copy()
        tmp = tmp.dropna(subset=["log10_psd"])
        if tmp.shape[0] < 4:
            continue

        X = pd.DataFrame({
            "log10_n_events": tmp["log10_n_events"],
            "log10_dur_s":    tmp["log10_dur_s"],
        })

        # 共変量がほぼ使えない場合はスキップ（参照スクリプト準拠の思想）
        if X.dropna(how="all").shape[0] < 4:
            continue

        X = sm.add_constant(X, has_constant="add")
        y = tmp["log10_psd"]

        try:
            model = sm.OLS(y, X, missing="drop").fit()
            y_adj = model.resid + model.params.get("const", 0.0)
        except Exception:
            continue

        idx = sub.index.intersection(tmp.index)
        df.loc[idx, "log10_psd_adj"] = y_adj.loc[idx]

    return df



def cohens_d(x_hc: np.ndarray, x_sz: np.ndarray) -> float:
    x_hc = x_hc[np.isfinite(x_hc)]
    x_sz = x_sz[np.isfinite(x_sz)]
    if x_hc.size < 2 or x_sz.size < 2:
        return np.nan
    v1 = np.var(x_hc, ddof=1)
    v2 = np.var(x_sz, ddof=1)
    pooled = ((x_hc.size - 1) * v1 + (x_sz.size - 1) * v2) / max(1, (x_hc.size + x_sz.size - 2))
    if not np.isfinite(pooled) or pooled <= 0:
        return np.nan
    return float((np.mean(x_sz) - np.mean(x_hc)) / np.sqrt(pooled))

def perm_mean_diff(x_hc: np.ndarray, x_sz: np.ndarray, n_perm: int = 10000, seed: int = 0):
    """
    Two-sided permutation test on mean difference (SZ - HC)
    returns: (diff, p_perm) with p floored at 1/n_perm
    """
    rng = np.random.default_rng(seed)
    x_hc = x_hc[np.isfinite(x_hc)]
    x_sz = x_sz[np.isfinite(x_sz)]
    if x_hc.size < 2 or x_sz.size < 2:
        return (np.nan, np.nan)

    obs = float(np.mean(x_sz) - np.mean(x_hc))
    pooled = np.concatenate([x_hc, x_sz])
    n_hc = x_hc.size

    diffs = np.empty(n_perm, float)
    for i in range(n_perm):
        rng.shuffle(pooled)
        diffs[i] = float(np.mean(pooled[n_hc:]) - np.mean(pooled[:n_hc]))

    p = float(np.mean(np.abs(diffs) >= abs(obs)))
    p = max(p, 1.0 / float(n_perm))
    return (obs, p)

def build_table_s3(
    df: pd.DataFrame,
    value_col: str,
    table_type: str,
    n_perm: int = 10000,
    seed: int = 0,
) -> pd.DataFrame:
    """
    site×cond×freq ごとに HC vs SZ の統計を1行で出す（画像のTable S3形式）
    """
    rows = []
    # ★ group を含めずに groupby するのが重要
    for (site, cond, freq), sub in df.groupby(["site", "cond", "freq"], dropna=False):
        x_hc = pd.to_numeric(sub.loc[sub["group"] == "HC", value_col], errors="coerce").to_numpy(float)
        x_sz = pd.to_numeric(sub.loc[sub["group"] == "SZ", value_col], errors="coerce").to_numpy(float)

        x_hc = x_hc[np.isfinite(x_hc)]
        x_sz = x_sz[np.isfinite(x_sz)]

        n_hc = int(x_hc.size)
        n_sz = int(x_sz.size)

        mean_hc = float(np.mean(x_hc)) if n_hc else np.nan
        mean_sz = float(np.mean(x_sz)) if n_sz else np.nan
        sd_hc   = float(np.std(x_hc, ddof=1)) if n_hc >= 2 else np.nan
        sd_sz   = float(np.std(x_sz, ddof=1)) if n_sz >= 2 else np.nan

        diff, p = perm_mean_diff(x_hc, x_sz, n_perm=n_perm, seed=seed + abs(hash((table_type, site, cond, int(freq)))) % 100000)
        d = cohens_d(x_hc, x_sz)

        rows.append(dict(
            Type=table_type,
            Site=display_site(site),
            Condition=str(cond),
            Frequency_Hz=int(freq),
            Mean_log10_PSD_HC=mean_hc,
            Mean_log10_PSD_SZ=mean_sz,
            SD_log10_PSD_HC=sd_hc,
            SD_log10_PSD_SZ=sd_sz,
            Mean_diff_SZ_minus_HC=diff,
            Cohens_d=d,
            p_value=p,
        ))

    out = pd.DataFrame(rows)
    if out.empty:
        out["q_FDR"] = []
        return out

    # ★ BH-FDR：Type×Site×Condition ごとに frequency 方向で補正
    out["q_FDR"] = np.nan
    for (typ, site, cond), g in out.groupby(["Type", "Site", "Condition"], sort=False):
        m = g["p_value"].notna()
        if m.any():
            out.loc[g.index[m], "q_FDR"] = multipletests(g.loc[m, "p_value"].values, method="fdr_bh")[1]

    return out.sort_values(["Type", "Site", "Condition", "Frequency_Hz"]).reset_index(drop=True)


def summarize_cells_power(df: pd.DataFrame, metrics=("log10_psd",), n_perm=10000, seed=0) -> pd.DataFrame:
    rows = []
    for (site, cond, freq, group), sub in df.groupby(["site", "cond", "freq", "group"], dropna=False):
        for m in metrics:
            if m not in sub.columns:
                continue
            vals = pd.to_numeric(sub[m], errors="coerce").astype(float)
            vals = vals[np.isfinite(vals)]
            n = int(len(vals))
            if n == 0:
                continue
            mean = float(vals.mean())
            se = float(np.sqrt(np.var(vals, ddof=1) / n)) if n >= 2 else np.nan

            # Cohen's d calculation
            group_hc = sub[sub["group"] == "HC"][m]
            group_sz = sub[sub["group"] == "SZ"][m]
            mean_hc = group_hc.mean()
            mean_sz = group_sz.mean()
            std_hc = group_hc.std()
            std_sz = group_sz.std()
            pooled_std = np.sqrt(((std_hc ** 2) + (std_sz ** 2)) / 2)
            cohen_d = (mean_sz - mean_hc) / pooled_std if pooled_std > 0 else np.nan

            # p-value calculation via permutation test
            p_value = np.nan
            fdr_p_value = np.nan

            if len(group_hc) > 1 and len(group_sz) > 1:
                diff, p_value = perm_mean_diff(group_hc.values, group_sz.values, n_perm, seed)

            # FDR correction for p-values
            if not np.isnan(p_value):
                fdr_p_value = multipletests([p_value], method='fdr_bh')[1][0]

            rows.append(dict(
                site=site,
                cond=cond,
                freq=int(freq),
                group=group,
                metric=m,
                n=n,
                mean=mean,
                se=se,
                cohen_d=cohen_d,
                p_value=p_value,
                fdr_p_value=fdr_p_value
            ))

    return pd.DataFrame(rows)
def plot_site_two_panels(cell_df: pd.DataFrame, metric: str, ylabel: str, out_pdf: Path, site: str):
    """
    For a given site, make a single figure with two panels:
      left: Cortex, right: Hippocampus
    Each panel: HC vs SZ bars per frequency with mean±SE.
    """
    group_order = ["HC", "SZ"]
    color_map = {"HC": "#4C72B0", "SZ": "#C44E52"}  # blue / red

    site_df = cell_df[(cell_df["site"] == site) & (cell_df["metric"] == metric)].copy()
    if site_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, cond in zip(axes, ["Cortex", "Hippocampus"]):
        sub = site_df[site_df["cond"] == cond].copy()
        if sub.empty:
            ax.axis("off")
            ax.set_title(f"{cond} (no data)")
            continue

        sub = sub.sort_values("freq")
        freqs = sorted(sub["freq"].unique())
        x = np.arange(len(freqs))
        w = 0.36

        for gi, g in enumerate(group_order):
            gsub = sub[sub["group"] == g]
            mean_map = {int(r["freq"]): float(r["mean"]) for _, r in gsub.iterrows()}
            se_map   = {int(r["freq"]): float(r["se"])   for _, r in gsub.iterrows()}
            means = np.array([mean_map.get(int(f), np.nan) for f in freqs], float)
            ses   = np.array([se_map.get(int(f), np.nan) for f in freqs], float)

            offset = (-w/2) if gi == 0 else (w/2)
            ax.bar(
                x + offset, means, width=w,
                yerr=ses, capsize=3,
                color=color_map.get(g, None),
                edgecolor="black", linewidth=0.6,
                label=g
            )

        ax.set_xticks(x)
        ax.set_xticklabels([str(int(f)) for f in freqs])
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{display_site(site)} · {cond}", loc="left")
        ax.grid(True, axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None)
    ap.add_argument("--input", type=str, default=None,
                    help="CSV path for df_PSD (default: <root>/data/df_PSD.csv)")
    args = ap.parse_args()

    if args.root is not None:
        root = Path(args.root).expanduser().resolve()
    else:
        try:
            root = Path(__file__).resolve().parents[1]
        except NameError:
            root = Path.cwd().resolve()

    in_csv = Path(args.input).expanduser().resolve() if args.input else (root / "data" / "df_PSD.csv")
    out_tables = root / "outputs" / "tables"
    out_figs   = root / "outputs" / "figures"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    if not in_csv.exists():
        raise FileNotFoundError(f"Input not found: {in_csv}")

    df = pd.read_csv(in_csv)

    # normalize
    df = df.copy()
    df["site"] = df["site"].map(norm_site)
    df["cond"] = df["cond"].map(norm_cond)
    df["group"] = df["group"].map(norm_group)
    df["freq"] = pd.to_numeric(df["freq"], errors="coerce")
    df["log10_psd"] = pd.to_numeric(df["log10_psd"], errors="coerce")
    df["n_events_used"] = pd.to_numeric(df["n_events_used"], errors="coerce")
    df["concat_duration_s"] = pd.to_numeric(df["concat_duration_s"], errors="coerce")
    df["subject"] = df["subject"].astype(str).str.strip()

    # keep only rows with power (optional flag)
    if "has_power_csv" in df.columns:
        df = df[df["has_power_csv"].astype(bool)]

    # keep only target conditions/groups
    df = df[df["cond"].isin(["Hippocampus","Cortex"])].copy()
    df = df[df["group"].isin(["HC","SZ"])].copy()

    # (1) Recompute adjusted power
    df_adj = recompute_log10_psd_adj(df)

    # (2) Summarize adjusted data
    cell_adj = summarize_cells_power(df_adj, metrics=("log10_psd_adj",))
    # (3) Table S3: raw と event_controlled を統計付きで作る
    t3_adj = build_table_s3(df_adj, value_col="log10_psd_adj", table_type="event_controlled", n_perm=10000, seed=0)
    
 
    out_s3_xlsx = out_tables / "TableS3_psd_log10_adj.xlsx"
    with pd.ExcelWriter(out_s3_xlsx, engine="openpyxl") as w:
        t3_adj.to_excel(w, sheet_name="Table_S3", index=False)
    


    # (4) Create figures: one per site
    for site in sorted(df_adj["site"].unique()):
        # Adjusted power figure
        outpdf_adj = out_figs / f"Fig1d_power_adj_{display_site(site)}.pdf"
        plot_site_two_panels(
            cell_adj, metric="log10_psd_adj",
            ylabel="log10 power (fT²/Hz) — event-controlled",
            out_pdf=outpdf_adj, site=site
        )

    print("[OK] Written tables:")
   
    print("[OK] Written figures to:", out_figs)

if __name__ == "__main__":
    main()
