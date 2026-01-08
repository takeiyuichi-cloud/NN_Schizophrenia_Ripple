#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig5 Panel D (SZ only): Predicted HIP↔CTX share by ripple-load, stratified by PANSSpos tertiles
REPRODUCIBLE VERSION using the FINAL table `d` that actually generated the target figure.

Input (single file):
  - NN_open_code/data/source_fig5_predicted_share_by_load_symptom_tertiles_d.csv

Required columns in the input:
  - subject : subject identifier (string)
  - site    : site label (string; used only to create dummy covariates)
  - n       : numerator (HIP↔CTX pooled count; integer or float)
  - total   : denominator (total transitions; integer or float; must be >0)
  - total_ripples_80_240_z : z-scored ripple load (float)
  - PANSS_pos : PANSS positive score (float)
  - symptom_c : centered PANSS_pos (float)  (if missing, computed inside)

Model (fixed, to match the "target" script style):
  - Binomial GEE (logit), Exchangeable, cluster=subject, weights=total
  - Exog (minimal): const + total_z + symptom_c + total_z*symptom_c + site dummies

Figure:
  - Curves: marginal mean predicted share (%) vs total_z at 3 PANSSpos tertiles
  - CI: subject-level bootstrap (cluster bootstrap), refit GEE each replicate
        and recompute marginal curves; 2.5/97.5 percentiles
  - CI is drawn as whiskers at sparse grid points (matches your target style)

Outputs:
  - NN_open_code/outputs/figures/Fig5d_predicted_HIPCTX_share_by_load_PANSSpos_tertiles.pdf


Notes:
  - This script uses ONLY the provided "final d" CSV. No clin.csv and no other merges.
  - If your earlier results changed, this script should lock them.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable


# -------------------------
# Plot config
# -------------------------
COLOR_MAP = {
    "Low PANSSpos":  "#1f77b4",
    "Mid PANSSpos":  "#ff7f0e",
    "High PANSSpos": "#d62728",
}


def expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_site_dummies(df: pd.DataFrame) -> pd.DataFrame:
    # drop_first=True to avoid collinearity
    return pd.get_dummies(df["site"].astype(str), prefix="site", drop_first=True).astype(float)


def build_design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal design matrix consistent with the target script:
      const + total_z + symptom_c + total_z*symptom_c + site dummies
    """
    X = pd.DataFrame({
        "total_z": df["total_ripples_80_240_z"].astype(float),
        "symptom_c": df["symptom_c"].astype(float),
    })
    X["totalZ_x_symptom"] = X["total_z"] * X["symptom_c"]

    site_d = build_site_dummies(df)
    X = pd.concat([X, site_d], axis=1)

    X = X.replace([np.inf, -np.inf], np.nan)
    # drop all-NaN columns
    X = X.loc[:, X.notna().any(axis=0)]
    # fill remaining NaN with column means (match your prior scripts)
    if X.isna().any().any():
        X = X.fillna(X.mean(numeric_only=True))

    X = sm.add_constant(X, has_constant="add").astype(float)
    return X


def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    need = ["subject", "site", "n", "total", "total_ripples_80_240_z", "PANSS_pos"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Input CSV missing required columns: {miss}")

    df = df.copy()
    df["subject"] = df["subject"].astype(str)
    df["site"] = df["site"].astype(str)

    df = safe_numeric(df, ["n", "total", "total_ripples_80_240_z", "PANSS_pos", "symptom_c"])
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["subject", "site", "n", "total", "total_ripples_80_240_z", "PANSS_pos"], how="any").copy()

    df = df[df["total"] > 0].copy()
    # enforce bounds for binomial proportion
    df["n"] = np.minimum(df["n"].astype(float), df["total"].astype(float))
    df["share"] = (df["n"] / df["total"]).astype(float)

    # if symptom_c missing, compute centered PANSS_pos (within the current df)
    if "symptom_c" not in df.columns or df["symptom_c"].isna().all():
        m = float(np.nanmean(df["PANSS_pos"].to_numpy(float)))
        df["symptom_c"] = df["PANSS_pos"] - m

    return df


def fit_gee(df: pd.DataFrame):
    """
    Binomial GEE:
      y = n/total
      weights = total
      cluster = subject
    """
    X = build_design_matrix(df)
    y = (df["n"].to_numpy(float) / df["total"].to_numpy(float)).astype(float)
    w = df["total"].to_numpy(float)
    groups = pd.Categorical(df["subject"]).codes

    gee = GEE(y, X, groups=groups, family=Binomial(), cov_struct=Exchangeable(), weights=w)
    res = gee.fit()
    return res, X


def symptom_tertile_centers(df: pd.DataFrame):
    """
    Return representative symptom_c values for tertiles:
      low/mid/high computed as mean within each tertile, then centered (subtract overall mean)
    """
    s = df["PANSS_pos"].to_numpy(float)
    s = s[np.isfinite(s)]
    if s.size < 12:
        return None

    q1, q2 = np.nanquantile(s, [1/3, 2/3])
    s_mean = float(np.nanmean(s))

    def _mean_or_nan(a):
        a = a[np.isfinite(a)]
        return float(np.nanmean(a)) if a.size else np.nan

    s_low = _mean_or_nan(s[s <= q1])
    s_mid = _mean_or_nan(s[(s > q1) & (s <= q2)])
    s_high = _mean_or_nan(s[s > q2])

    return [
        ("Low PANSSpos",  s_low - s_mean),
        ("Mid PANSSpos",  s_mid - s_mean),
        ("High PANSSpos", s_high - s_mean),
    ]


def predict_marginal_curve(df_template: pd.DataFrame, X_cols: list[str], beta: np.ndarray,
                           grid: np.ndarray, symptom_c: float) -> np.ndarray:
    """
    Marginal mean prediction:
      - take template rows (subjects/covars/site)
      - overwrite total_z and symptom_c and interaction
      - compute mu=expit(X beta)
      - return mean(mu) across rows (marginal mean)
    """
    base = df_template.copy().reset_index(drop=True)
    X0 = build_design_matrix(base)
    # align columns exactly
    for c in X_cols:
        if c not in X0.columns:
            X0[c] = 0.0
    X0 = X0[X_cols].astype(float)

    out = np.empty_like(grid, dtype=float)
    for i, g in enumerate(grid):
        Xg = X0.copy()
        if "total_z" in Xg.columns:
            Xg["total_z"] = float(g)
        if "symptom_c" in Xg.columns:
            Xg["symptom_c"] = float(symptom_c)
        if "totalZ_x_symptom" in Xg.columns:
            Xg["totalZ_x_symptom"] = float(g) * float(symptom_c)

        eta = Xg.to_numpy(dtype=float) @ beta
        out[i] = float(np.nanmean(expit(eta)))
    return out


def bootstrap_ci(df: pd.DataFrame, grid: np.ndarray, levels, *,
                 n_boot: int = 800, seed: int = 0):
    """
    Cluster bootstrap by subject:
      resample subjects with replacement -> refit -> marginal curve
    Return dict label -> (lo, hi)
    """
    rng = np.random.default_rng(seed)
    subs = df["subject"].astype(str).unique().tolist()
    if len(subs) < 10:
        return {lab: (np.full_like(grid, np.nan), np.full_like(grid, np.nan)) for lab, _ in levels}

    # reference columns for stable alignment
    res0, X0 = fit_gee(df)
    X_cols = X0.columns.tolist()

    curves = {lab: [] for lab, _ in levels}

    for b in range(n_boot):
        samp = rng.choice(subs, size=len(subs), replace=True)
        db = pd.concat([df[df["subject"].astype(str) == s] for s in samp], ignore_index=True)
        db = db.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["subject", "site", "n", "total", "total_ripples_80_240_z", "symptom_c"], how="any"
        )
        if db["subject"].nunique() < 10:
            continue
        try:
            resb, Xb = fit_gee(db)
            beta = resb.params.reindex(X_cols).to_numpy(dtype=float)

            for lab, sc in levels:
                mu = predict_marginal_curve(db, X_cols, beta, grid, sc)
                if np.all(np.isfinite(mu)):
                    curves[lab].append(mu)
        except Exception:
            continue

    out = {}
    for lab, _ in levels:
        if len(curves[lab]) < max(80, int(0.2 * n_boot)):
            out[lab] = (np.full_like(grid, np.nan), np.full_like(grid, np.nan))
            continue
        M = np.vstack(curves[lab])
        out[lab] = (np.quantile(M, 0.025, axis=0), np.quantile(M, 0.975, axis=0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None, help="NN_open_code root (default: auto)")
    ap.add_argument("--input", type=str, default=None, help="input CSV (default: <root>/data/source_fig5_predicted_share_by_load_symptom_tertiles_d.csv)")
    ap.add_argument("--boot-n", type=int, default=800, help="bootstrap replicates")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    ap.add_argument("--grid-n", type=int, default=120, help="grid points")
    ap.add_argument("--ci-every", type=int, default=10, help="draw CI whiskers every N grid points")
    args = ap.parse_args()

    if args.root is not None:
        root = Path(args.root).expanduser().resolve()
    else:
        try:
            root = Path(__file__).resolve().parents[1]
        except NameError:
            root = Path.cwd().resolve()

    in_csv = Path(args.input).expanduser().resolve() if args.input else (root / "data" / "source_fig5_predicted_share_by_load_symptom_tertiles_d.csv")
    if not in_csv.exists():
        raise FileNotFoundError(in_csv)

    out_fig_dir = root / "outputs" / "figures"
    out_tab_dir = root / "outputs" / "tables"
    out_fig_dir.mkdir(parents=True, exist_ok=True)
    out_tab_dir.mkdir(parents=True, exist_ok=True)

    df0 = pd.read_csv(in_csv)
    df = sanitize_df(df0)

    # Fit once (point estimate)
    res, X = fit_gee(df)
    X_cols = X.columns.tolist()

    # Save parameter table
    params = []
    for term in res.params.index:
        beta = float(res.params[term])
        se = float(res.bse[term]) if term in res.bse.index else np.nan
        p = float(res.pvalues[term]) if term in res.pvalues.index else np.nan
        params.append(dict(term=term, beta=beta, SE=se, z=beta/se if (np.isfinite(beta) and np.isfinite(se) and se != 0) else np.nan,
                           p=p, OR=float(np.exp(beta)) if np.isfinite(beta) else np.nan))
    df_params = pd.DataFrame(params)
    
    # Build grid and levels
    tz = df["total_ripples_80_240_z"].to_numpy(float)
    tz = tz[np.isfinite(tz)]
    if tz.size < 10:
        raise RuntimeError("Too few finite total_ripples_80_240_z values.")
    grid = np.linspace(float(np.nanmin(tz)), float(np.nanmax(tz)), int(args.grid_n))

    levels = symptom_tertile_centers(df)
    if levels is None:
        raise RuntimeError("Too few PANSS_pos values for tertiles.")

    # Point estimate curves
    beta0 = res.params.reindex(X_cols).to_numpy(dtype=float)
    mu = {lab: predict_marginal_curve(df, X_cols, beta0, grid, sc) for lab, sc in levels}

    # Bootstrap CI
    ci = bootstrap_ci(df, grid, levels, n_boot=int(args.boot_n), seed=int(args.seed))

    # Export pred grid table
    pred_rows = []
    for lab, _sc in levels:
        for g, mval in zip(grid, mu[lab]):
            lo, hi = ci[lab]
            # find index
            i = int(np.where(grid == g)[0][0])
            pred_rows.append(dict(
                symptom_band=lab,
                total_z=float(g),
                pred_share_pct=float(100.0 * mval),
                ci_lo_pct=float(100.0 * lo[i]) if np.isfinite(lo[i]) else np.nan,
                ci_hi_pct=float(100.0 * hi[i]) if np.isfinite(hi[i]) else np.nan,
            ))
    
    # Plot
    out_pdf = out_fig_dir / "Fig5d_predicted_HIPCTX_share_by_load_PANSSpos_tertiles.pdf"

    plt.figure(figsize=(6.2, 4.6), dpi=180)

    idx = np.arange(0, len(grid), max(1, int(args.ci_every)))

    for lab, _sc in levels:
        plt.plot(grid, 100.0 * mu[lab], lw=2.2, label=lab, color=COLOR_MAP.get(lab, None))

        lo, hi = ci[lab]
        y = 100.0 * mu[lab][idx]
        ylo = 100.0 * lo[idx]
        yhi = 100.0 * hi[idx]
        m = np.isfinite(y) & np.isfinite(ylo) & np.isfinite(yhi)
        if m.any():
            yerr = np.vstack([y[m] - ylo[m], yhi[m] - y[m]])
            plt.errorbar(grid[idx][m], y[m], yerr=yerr, fmt="none",
                         elinewidth=0.9, capsize=2, color=COLOR_MAP.get(lab, None))

    plt.xlabel("Total ripple load (z, 80–240 Hz)")
    plt.ylabel("Predicted HIP↔CTX transition share (%)")
    plt.title("SZ only: predicted HIP↔CTX share by load, stratified by PANSSpos\n(marginal mean with bootstrap 95% CI)")
    plt.grid(alpha=0.25, ls="--", lw=0.6)
    plt.legend(frameon=False)
    plt.tight_layout()

    plt.savefig(out_pdf)
    plt.close()

    print("[OK] Written:")
    print(" -", out_pdf)


if __name__ == "__main__":
    main()
