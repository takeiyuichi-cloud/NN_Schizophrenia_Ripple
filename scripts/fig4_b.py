#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig4b (FINAL; aligned to S10 framework + S11 adjustments; SZ only):
Clustered ripple dynamics and clinical measures (plot: PANSS_positive)

ALIGNMENT GOALS
---------------
This script is designed to be fully consistent with:
- Supplementary Table S10 framework:
    * Negative Binomial GLM (log link) for count outcomes
    * offset = log(minutes_sum)
    * covariates + C(site_public)
    * HC3 robust SE
    * one model per clinical predictor
    * BH-FDR across clinical predictors within each outcome family
- S11-style adjustment for clustered metrics:
    * additionally adjust for total ripple load (events_sum)
    * outcomes are pooled across 80–240 Hz (sum across freqs for counts)
    * intensity outcome uses PEAK RATE (consistent with Fig.3c definition)
      - Peak rate within clustered ripple events:
          mean(max_epoch_rate_hz) across freqs (default) or max(...) if AGG_PEAK="max"

INPUTS (NN_open_code/data/)
---------------------------
- fig4a_source_public.csv        (SAFE; anon_id, site_public, clinical, covars, minutes_sum, events_sum, group(optional))
- fig4a_id_map_private.csv       (PRIVATE; subject <-> anon_id; DO NOT SHARE)
- rate_epoch_subject_level.csv   (subject×freq; from sliding-window pipeline)

REQUIRED columns:
- fig4a_source_public.csv: anon_id, site_public, minutes_sum, events_sum, PANSS_positive (+ covars)
- rate_epoch_subject_level.csv: subject, freq, n_epochs, n_events, n_events_in_epochs, max_epoch_rate_hz

OUTPUT (ONLY ONE FIGURE)
------------------------
- NN_open_code/outputs/figures/fig4b_clustered_SWRS_and_positive_symptoms.pdf

NOTES
-----
- We fit models for multiple clinical predictors (as in S10), apply BH-FDR across predictors
  within each outcome, and then plot ONLY PANSS_positive with its corresponding FDR q.
- If a predictor column is missing, it is skipped (same behavior as S10 scripts).
"""

from __future__ import annotations
from pathlib import Path
import re
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

# Clinical predictors to fit (S10-aligned; any missing will be skipped)
# NOTE: some codebases use "PANSS_pasological" as GEN.
# We also support "PANSS_general" as an alias if present.
PRED_MAP = [
    ("PANSS_positive", "PANSS POS"),
    ("PANSS_negative", "PANSS NEG"),
    ("PANSS_pasological", "PANSS GEN"),
    ("GAF", "GAF"),
]

# Plot predictor
PLOT_PRED = "PANSS_positive"

# Peak aggregation across frequencies for the intensity outcome
AGG_PEAK = "mean"  # "mean" (default) or "max"

OUTFIG_NAME = "fig4b_clustered_SWRS_and_positive_symptoms.pdf"


# -------------------------
# Helpers
# -------------------------
def canon_subject(x) -> str:
    """Match subject canonicalization (avoid merge loss across scripts)."""
    s = str(x).strip()
    m = re.search(r"(\d+)", s)
    return f"NB_subject_{int(m.group(1))}" if m else s

def _safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _dropna(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols, how="any").copy()

def mode_or_median(s: pd.Series):
    s2 = pd.to_numeric(s, errors="coerce").dropna()
    if s2.empty:
        return np.nan
    if s2.nunique() <= 3:
        try:
            return s2.mode().iloc[0]
        except Exception:
            return float(s2.median())
    return float(s2.median())

def _stars(q: float) -> str:
    if q is None or (isinstance(q, float) and (not np.isfinite(q))):
        return ""
    if q < 1e-4: return "****"
    if q < 1e-3: return "***"
    if q < 1e-2: return "**"
    if q < 5e-2: return "*"
    return ""

def fit_nb_one_pred(D: pd.DataFrame, ycol: str, pred_col: str, covars: list[str], site_col: str, offset_col: str):
    fam = sm.families.NegativeBinomial(link=sm.families.links.Log())
    formula = f"{ycol} ~ {pred_col} + events_sum"
    if covars:
        formula += " + " + " + ".join(covars)
    formula += f" + C({site_col})"
    off = np.log(pd.to_numeric(D[offset_col], errors="coerce").astype(float))
    fit = smf.glm(formula=formula, data=D, family=fam, offset=off).fit(cov_type="HC3")

    beta = float(fit.params[pred_col])
    p = float(fit.pvalues[pred_col])
    se = float(fit.bse[pred_col])

    irr = float(np.exp(beta))
    irr_lo = float(np.exp(beta - 1.96 * se))
    irr_hi = float(np.exp(beta + 1.96 * se))
    return fit, dict(p=p, IRR=irr, IRR_lo=irr_lo, IRR_hi=irr_hi)

def fit_ols_one_pred(D: pd.DataFrame, ycol: str, pred_col: str, covars: list[str], site_col: str):
    formula = f"{ycol} ~ {pred_col} + events_sum"
    if covars:
        formula += " + " + " + ".join(covars)
    formula += f" + C({site_col})"
    fit = smf.ols(formula=formula, data=D).fit(cov_type="HC3")

    beta = float(fit.params[pred_col])
    p = float(fit.pvalues[pred_col])
    se = float(fit.bse[pred_col])
    ci_lo = beta - 1.96 * se
    ci_hi = beta + 1.96 * se
    return fit, dict(p=p, beta=beta, beta_lo=ci_lo, beta_hi=ci_hi)

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

def _resolve_panss_gen(df: pd.DataFrame) -> pd.DataFrame:
    """Support PANSS_general as alias for PANSS_pasological if needed."""
    df = df.copy()
    if "PANSS_pasological" not in df.columns and "PANSS_general" in df.columns:
        df["PANSS_pasological"] = df["PANSS_general"]
    return df


# -------------------------
# Main
# -------------------------
def main():
    root = Path(__file__).resolve().parents[1]  # NN_open_code
    data_dir = root / "data"
    out_fig_dir = root / "outputs" / "figures"
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    p_clin = data_dir / "fig4a_source_public.csv"
    p_map  = data_dir / "fig4a_id_map_private.csv"      # private
    p_rate = data_dir / "rate_epoch_subject_level.csv"

    for p in [p_clin, p_map, p_rate]:
        if not p.exists():
            raise FileNotFoundError(p)

    # --- clinical (SAFE) ---
    clin = pd.read_csv(p_clin)
    clin = _resolve_panss_gen(clin)

    req = {"anon_id", "site_public", "minutes_sum", "events_sum"}
    miss = req - set(clin.columns)
    if miss:
        raise ValueError(f"fig4a_source_public.csv missing: {sorted(miss)}")

    num_cols = ["minutes_sum", "events_sum"] + COVARS + [c for c, _ in PRED_MAP]
    clin = _safe_numeric(clin, [c for c in num_cols if c in clin.columns])
    clin["site_public"] = clin["site_public"].astype(str)
    clin["anon_id"] = clin["anon_id"].astype(str)

    # SZ-only filter if group is present
    if "group" in clin.columns:
        clin["group"] = clin["group"].astype(str).str.upper().replace({"SC": "SZ"})
        clin = clin[clin["group"].isin(["SZ"])].copy()

    # --- private id map ---
    mp = pd.read_csv(p_map)
    if not {"subject", "anon_id"}.issubset(mp.columns):
        raise ValueError("fig4a_id_map_private.csv must have columns: subject, anon_id")
    mp = mp.copy()
    mp["subject"] = mp["subject"].map(canon_subject)
    mp["anon_id"] = mp["anon_id"].astype(str)

    # --- rate_epoch_subject_level (subject×freq) ---
    rate = pd.read_csv(p_rate)
    need_rate = {"subject", "freq", "n_epochs", "n_events", "n_events_in_epochs", "max_epoch_rate_hz"}
    miss = need_rate - set(rate.columns)
    if miss:
        raise ValueError(f"rate_epoch_subject_level.csv missing: {sorted(miss)}")

    rate = rate.copy()
    rate["subject"] = rate["subject"].map(canon_subject)
    rate["freq"] = pd.to_numeric(rate["freq"], errors="coerce").astype(int)

    for c in ["n_epochs", "n_events", "n_events_in_epochs", "max_epoch_rate_hz"]:
        rate[c] = pd.to_numeric(rate[c], errors="coerce")

    rate = rate[rate["freq"].isin(FREQS_USE)].copy()

    # outside counts
    rate["n_events_outside_epochs"] = rate["n_events"] - rate["n_events_in_epochs"]

    # pooled outcomes across freqs
    peak_agg = "max" if AGG_PEAK == "max" else "mean"
    agg = rate.groupby(["subject"], as_index=False).agg(
        clustered_epochs=("n_epochs", "sum"),
        outside_ripple_count=("n_events_outside_epochs", "sum"),
        peak_rate_within_clustered_swrs=("max_epoch_rate_hz", peak_agg),
    )

    # attach anon_id
    agg = agg.merge(mp, on="subject", how="left").dropna(subset=["anon_id"]).copy()
    agg["anon_id"] = agg["anon_id"].astype(str)

    # merge to clinical
    df = clin.merge(agg, on="anon_id", how="inner")

    # basic sanity filters
    df = df[pd.to_numeric(df["minutes_sum"], errors="coerce") > 0].copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    covars = [c for c in COVARS if c in df.columns]
    preds = [(c, lab) for c, lab in PRED_MAP if c in df.columns]

    if PLOT_PRED not in df.columns:
        raise ValueError(f"Required plot predictor '{PLOT_PRED}' not found in fig4a_source_public.csv")

    # outcomes: (col, title, type)
    outcomes = [
        ("clustered_epochs", "Number of high-rate ripple epochs", "count"),
        ("outside_ripple_count", "Ripple events outside high-rate epochs", "count"),
        ("peak_rate_within_clustered_swrs", "Peak rate within clustered ripple events", "cont"),
    ]

    # -------------------------
    # Fit models: one per predictor within each outcome,
    # BH-FDR across predictors within each outcome.
    # -------------------------
    fitted = {}  # outcome -> pred -> dict(fit, D, info)

    for out_col, out_title, out_type in outcomes:
        tmp_infos = []
        tmp_fits = []

        for pred_col, pred_label in preds:
            if out_type == "count":
                cols_need = ["site_public", "minutes_sum", "events_sum", out_col, pred_col] + covars
            else:
                cols_need = ["site_public", "events_sum", out_col, pred_col] + covars

            D = _dropna(df, cols_need)
            if out_type == "count":
                D = D[D["minutes_sum"] > 0].copy()
            if D.empty:
                continue

            if out_type == "count":
                fit, info = fit_nb_one_pred(
                    D, ycol=out_col, pred_col=pred_col,
                    covars=covars, site_col="site_public", offset_col="minutes_sum"
                )
                info["label"] = f"IRR={info['IRR']:.3f}"
            else:
                fit, info = fit_ols_one_pred(
                    D, ycol=out_col, pred_col=pred_col,
                    covars=covars, site_col="site_public"
                )
                info["label"] = f"β={info['beta']:.3f}"

            tmp_infos.append((pred_col, info))
            tmp_fits.append((pred_col, fit, D, pred_label))

        if not tmp_infos:
            continue

        # BH-FDR across predictors for this outcome
        pvals = np.array([info["p"] for _, info in tmp_infos], float)
        q = multipletests(pvals, method="fdr_bh")[1]

        fitted[out_col] = {}
        for (pred_col, info), qi, (pred_col2, fit, D, pred_label) in zip(tmp_infos, q, tmp_fits):
            info2 = info.copy()
            info2["q_FDR"] = float(qi)
            info2["pred_label"] = pred_label
            fitted[out_col][pred_col] = dict(fit=fit, D=D, info=info2)

    # ensure plot predictor exists in each outcome
    for out_col, _, _ in outcomes:
        if out_col not in fitted or PLOT_PRED not in fitted[out_col]:
            raise RuntimeError(
                f"Could not fit required model for outcome='{out_col}' with predictor='{PLOT_PRED}'. "
                f"Check missing data / columns."
            )

    # -------------------------
    # Plot (PANSS_positive only)
    # -------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.8), dpi=160)

    for ax, (out_col, out_title, out_type) in zip(axes, outcomes):
        pack = fitted[out_col][PLOT_PRED]
        fit = pack["fit"]
        D = pack["D"]
        info = pack["info"]

        x = pd.to_numeric(D[PLOT_PRED], errors="coerce").to_numpy(float)
        y = pd.to_numeric(D[out_col], errors="coerce").to_numpy(float)

        ax.scatter(x, y, s=18, alpha=0.85, color="black")

        x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
        x_grid = np.linspace(x_min, x_max, 180) if np.isfinite(x_min) and np.isfinite(x_max) and x_min != x_max else np.array([x_min])

        # fixed covariates for prediction
        site_mode = D["site_public"].mode().iloc[0]
        fixed = {"site_public": site_mode}
        fixed["events_sum"] = float(mode_or_median(D["events_sum"]))
        for c in covars:
            fixed[c] = mode_or_median(D[c])

        if out_type == "count":
            minutes_fixed = float(mode_or_median(D["minutes_sum"]))
            yhat, ylo, yhi = pred_count(fit, PLOT_PRED, x_grid, fixed, minutes_val=minutes_fixed)
        else:
            yhat, ylo, yhi = pred_ols(fit, PLOT_PRED, x_grid, fixed)

        ax.plot(x_grid, yhat, lw=2, color="black")
        ax.fill_between(x_grid, ylo, yhi, color="black", alpha=0.12, linewidth=0)

        qv = info.get("q_FDR", np.nan)
        qtxt = f"FDR-p={qv:.4g}{_stars(qv)}" if np.isfinite(qv) else "FDR-p=NA"
        ax.text(0.02, 0.98, f"{info['label']}, {qtxt}", transform=ax.transAxes,
                ha="left", va="top", fontsize=10)

        ax.set_title(out_title, fontsize=11)
        ax.set_xlabel("PANSS positive")
        ax.set_ylabel(out_title)
        ax.grid(alpha=0.25)

    fig.suptitle("(b) Clustered ripple dynamics and positive symptoms (SZ only)", y=1.04, fontsize=14)

    out_pdf = out_fig_dir / OUTFIG_NAME
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("[OK] Saved figure:", out_pdf)


if __name__ == "__main__":
    main()
