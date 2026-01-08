#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig5b (LOCAL for manuscript; minimal outputs):
Clustered SWRs and positive symptoms (SZ only)

Inputs (NN_open_code/data/):
  - fig5a_source_public.csv        (SAFE; anon_id, site_public, PANSS_positive, covars, minutes_sum)
  - fig5a_id_map_private.csv       (PRIVATE; subject <-> anon_id; DO NOT SHARE)
  - rate_epoch_subject_level.csv   (subject×freq; from sliding-window pipeline)

Output (ONLY ONE FIGURE):
  - NN_open_code/outputs/figures/Fig5b_clustered_SWRS_and_positive_symptoms.pdf

Pooled (80–240 Hz) definitions (fixed):
  - clustered_swr_counts  = sum(n_epochs) across freqs
  - outside_ripple_counts = sum(n_events_outside_epochs) across freqs
      (fallback: n_events - n_events_in_epochs if needed)
  - peak_rate_within_clustered_swrs = mean(max_epoch_rate_hz) across freqs
      (AGG_PEAK can be changed to 'max' if you prefer)

Models (PANSS_positive):
  - Count outcomes: NB-GLM log link + covars + C(site_public), offset=log(minutes_sum), HC3
    annotate IRR and FDR-p (BH across 3 tests)
  - Peak rate: OLS + covars + C(site_public), HC3
    annotate beta and FDR-p

Note:
  - This script does NOT write any intermediate CSVs (to keep files minimal).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


# -------------------------
# Settings
# -------------------------
FREQS_USE = [80, 120, 160, 200, 240]
COVARS = ["age", "sex", "JART", "sleepiness_pre", "antipsychotics"]

AGG_PEAK = "mean"   # "mean" (default) or "max"

OUTFIG_NAME  = "Fig5b_clustered_SWRS_and_positive_symptoms.pdf"


# -------------------------
# Helpers
# -------------------------
def _safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _dropna(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols, how="any").copy()

def mode_or_median(s: pd.Series):
    s2 = s.dropna()
    if s2.empty:
        return np.nan
    if s2.nunique() <= 3:
        return s2.mode().iloc[0]
    return float(s2.median())

def fit_nb(D: pd.DataFrame, ycol: str, xcol: str, covars: list[str], site_col: str, offset_col: str):
    fam = sm.families.NegativeBinomial(link=sm.families.links.Log())
    formula = f"{ycol} ~ {xcol}"
    if covars:
        formula += " + " + " + ".join(covars)
    formula += f" + C({site_col})"
    off = np.log(pd.to_numeric(D[offset_col], errors="coerce").astype(float))
    fit = smf.glm(formula=formula, data=D, family=fam, offset=off).fit(cov_type="HC3")
    beta = float(fit.params[xcol])
    p = float(fit.pvalues[xcol])
    ci = fit.conf_int().loc[xcol].to_numpy(float)
    irr = float(np.exp(beta))
    irr_lo, irr_hi = float(np.exp(ci[0])), float(np.exp(ci[1]))
    return fit, dict(p=p, IRR=irr, IRR_lo=irr_lo, IRR_hi=irr_hi)

def fit_ols(D: pd.DataFrame, ycol: str, xcol: str, covars: list[str], site_col: str):
    formula = f"{ycol} ~ {xcol}"
    if covars:
        formula += " + " + " + ".join(covars)
    formula += f" + C({site_col})"
    fit = smf.ols(formula=formula, data=D).fit(cov_type="HC3")
    beta = float(fit.params[xcol])
    p = float(fit.pvalues[xcol])
    return fit, dict(p=p, beta=beta)

def pred_count(fit, xname: str, xgrid: np.ndarray, fixed: dict, minutes_val: float):
    P = pd.DataFrame({xname: xgrid})
    for k, v in fixed.items():
        P[k] = v
    sf = fit.get_prediction(P, offset=np.log(np.full_like(xgrid, minutes_val))).summary_frame(alpha=0.05)
    return sf["mean"].to_numpy(float), sf["mean_ci_lower"].to_numpy(float), sf["mean_ci_upper"].to_numpy(float)

def pred_ols(fit, xname: str, xgrid: np.ndarray, fixed: dict):
    P = pd.DataFrame({xname: xgrid})
    for k, v in fixed.items():
        P[k] = v
    sf = fit.get_prediction(P).summary_frame(alpha=0.05)
    return sf["mean"].to_numpy(float), sf["mean_ci_lower"].to_numpy(float), sf["mean_ci_upper"].to_numpy(float)


# -------------------------
# Main
# -------------------------
def main():
    root = Path(__file__).resolve().parents[1]  # NN_open_code
    data_dir = root / "data"
    out_fig_dir = root / "outputs" / "figures"
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    p_fig5a = data_dir / "fig5a_source_public.csv"
    p_map   = data_dir / "fig5a_id_map_private.csv"      # private
    p_epoch = data_dir / "rate_epoch_subject_level.csv"

    if not p_fig5a.exists():
        raise FileNotFoundError(p_fig5a)
    if not p_map.exists():
        raise FileNotFoundError(p_map)
    if not p_epoch.exists():
        raise FileNotFoundError(p_epoch)

    # --- fig5a public ---
    df5a = pd.read_csv(p_fig5a)
    req5a = {"anon_id", "site_public", "PANSS_positive", "minutes_sum"}
    miss = req5a - set(df5a.columns)
    if miss:
        raise ValueError(f"fig5a_source_public.csv missing: {sorted(miss)}")

    df5a = _safe_numeric(df5a, ["PANSS_positive", "minutes_sum"] + [c for c in COVARS if c in df5a.columns])
    df5a["site_public"] = df5a["site_public"].astype(str)

    # --- private id map ---
    idmap = pd.read_csv(p_map)
    if not {"subject", "anon_id"}.issubset(idmap.columns):
        raise ValueError("fig5a_id_map_private.csv must have columns: subject, anon_id")
    idmap["subject"] = idmap["subject"].astype(str)
    idmap["anon_id"] = idmap["anon_id"].astype(str)

    # --- rate_epoch_subject_level (subject×freq) ---
    ep = pd.read_csv(p_epoch)

    need_ep = {"site", "subject", "freq", "n_epochs", "max_epoch_rate_hz"}
    miss = need_ep - set(ep.columns)
    if miss:
        raise ValueError(f"rate_epoch_subject_level.csv missing: {sorted(miss)}")

    ep = ep.copy()
    ep["subject"] = ep["subject"].astype(str)
    ep["freq"] = pd.to_numeric(ep["freq"], errors="coerce")
    ep = ep[ep["freq"].isin(FREQS_USE)].copy()

    # outside counts
    if "n_events_outside_epochs" in ep.columns:
        ep["n_events_outside_epochs"] = pd.to_numeric(ep["n_events_outside_epochs"], errors="coerce")
    else:
        if {"n_events", "n_events_in_epochs"}.issubset(ep.columns):
            ep["n_events"] = pd.to_numeric(ep["n_events"], errors="coerce")
            ep["n_events_in_epochs"] = pd.to_numeric(ep["n_events_in_epochs"], errors="coerce").fillna(0)
            ep["n_events_outside_epochs"] = ep["n_events"] - ep["n_events_in_epochs"]
        else:
            raise ValueError("Need outside count: n_events_outside_epochs OR (n_events & n_events_in_epochs).")

    ep["n_epochs"] = pd.to_numeric(ep["n_epochs"], errors="coerce").fillna(0)
    ep["max_epoch_rate_hz"] = pd.to_numeric(ep["max_epoch_rate_hz"], errors="coerce")

    # pooled outcomes across freqs
    if AGG_PEAK == "max":
        peak_agg = ("max_epoch_rate_hz", "max")
    else:
        peak_agg = ("max_epoch_rate_hz", "mean")

    pooled = ep.groupby(["site", "subject"], as_index=False).agg(
        clustered_swr_counts=("n_epochs", "sum"),
        outside_ripple_counts=("n_events_outside_epochs", "sum"),
        peak_rate_within_clustered_swrs=peak_agg,
    )

    # attach anon_id
    pooled = pooled.merge(idmap, on="subject", how="inner")

    # join with fig5a public by anon_id
    df = df5a.merge(
        pooled[["anon_id", "clustered_swr_counts", "outside_ripple_counts", "peak_rate_within_clustered_swrs"]],
        on="anon_id", how="inner"
    )

    df["minutes_sum"] = pd.to_numeric(df["minutes_sum"], errors="coerce")
    df = df[df["minutes_sum"] > 0].copy()

    covars = [c for c in COVARS if c in df.columns]

    # ---------- models ----------
    specs = [
        ("Clustered SWRs counts", "clustered_swr_counts", "count"),
        ("Ripple counts outside clustered SWRs", "outside_ripple_counts", "count"),
        ("Peak rate within clustered SWRs", "peak_rate_within_clustered_swrs", "ols"),
    ]

    infos = []
    fits = []
    for title, ycol, mtype in specs:
        cols_need = ["site_public", "PANSS_positive", "minutes_sum", ycol] + covars
        D = _dropna(df, cols_need)
        if mtype == "count":
            fit, info = fit_nb(D, ycol=ycol, xcol="PANSS_positive", covars=covars,
                               site_col="site_public", offset_col="minutes_sum")
            label = f"IRR={info['IRR']:.3f}"
        else:
            fit, info = fit_ols(D, ycol=ycol, xcol="PANSS_positive", covars=covars, site_col="site_public")
            label = f"β={info['beta']:.3f}"
        infos.append({"p": float(info["p"]), "label": label})
        fits.append((fit, D, mtype, title, ycol))

    # BH-FDR across 3 tests
    pvals = np.array([i["p"] for i in infos], float)
    q = np.full_like(pvals, np.nan)
    m = np.isfinite(pvals)
    if m.any():
        q[m] = multipletests(pvals[m], method="fdr_bh")[1]
    for i, qi in zip(infos, q):
        i["q_FDR"] = float(qi) if np.isfinite(qi) else np.nan

    # ---------- plot ----------
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), dpi=160)

    for ax, (fit, D, mtype, title, ycol), info in zip(axes, fits, infos):
        x = pd.to_numeric(D["PANSS_positive"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(D[ycol], errors="coerce").to_numpy(float)
        ax.scatter(x, y, s=18, alpha=0.85, color="black")

        x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
        x_grid = np.linspace(x_min, x_max, 180) if np.isfinite(x_min) and np.isfinite(x_max) and x_min != x_max else np.array([x_min])

        site_mode = D["site_public"].mode().iloc[0]
        fixed = {"site_public": site_mode}
        for c in covars:
            fixed[c] = mode_or_median(D[c])

        if mtype == "count":
            minutes_fixed = float(mode_or_median(D["minutes_sum"]))
            yhat, ylo, yhi = pred_count(fit, "PANSS_positive", x_grid, fixed, minutes_val=minutes_fixed)
        else:
            yhat, ylo, yhi = pred_ols(fit, "PANSS_positive", x_grid, fixed)

        ax.plot(x_grid, yhat, lw=2, color="black")
        ax.fill_between(x_grid, ylo, yhi, color="black", alpha=0.12, linewidth=0)

        qtxt = f"FDR-p={info['q_FDR']:.4g}" if np.isfinite(info["q_FDR"]) else "FDR-p=NA"
        ax.text(0.02, 0.98, f"{info['label']}, {qtxt}", transform=ax.transAxes,
                ha="left", va="top", fontsize=10)

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("PANSS positive")
        ax.set_ylabel(title)
        ax.grid(alpha=0.25)

    fig.suptitle("(b) Clustered SWRs and positive symptoms", y=1.04, fontsize=14)
    out_pdf = out_fig_dir / OUTFIG_NAME
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("[OK] Saved figure:", out_pdf)


if __name__ == "__main__":
    main()