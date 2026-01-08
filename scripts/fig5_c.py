#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig5 Panel C (SZ only): Transition residual vs PANSS positive (adj_total scatter) + OR annotation (GEE)

Inputs (NN_open_code/data/):
  - transition_share_master.csv        (subject, pair, pct, n, total)
  - fig5a_source_public.csv            (SAFE: anon_id, site_public, events_sum, PANSS_positive, covars; SZ only)
  - fig5a_id_map_private.csv           (PRIVATE: subject <-> anon_id)

Figure (outputs/figures/):
  - Fig5c_transition_residual_vs_PANSSpos_with_OR.pdf


Faithful points:
  - Scatter uses adj_total residualization (y_res from Binomial GLM with PSEUDO_TOTAL weight=100; PANSS not in y-model)
  - X-axis uses covariate-adjusted PANSS residual (OLS), then z-scored (as in manuscript label)
  - OR annotation uses Binomial GEE (exchangeable), cluster=subject (anon_id), weights=total (from master; typically 100)
  - BH-FDR across pairs within this PANSS panel
"""

from __future__ import annotations
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
from statsmodels.stats.multitest import multipletests


# -------------------------
# Settings
# -------------------------
PAIRS_PLOT = ["CTX→HIP", "HIP→CTX", "CTX→CTX"]   # match manuscript panel C
PAIR_ORDER_FDR = PAIRS_PLOT[:]                  # FDR across these pairs only
PSEUDO_TOTAL = 100.0

COVARS = ["age", "sex", "JART", "sleepiness_pre", "antipsychotics"]
SYM = "PANSS_positive"

ADD_CI_BAND = True
CI_N_BOOT = 2000
CI_SEED = 0


# -------------------------
# Helpers
# -------------------------
def _safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def sex01(x):
    s = str(x).strip().upper()
    if s in {"M","MALE","1"}:
        return 1.0
    if s in {"F","FEMALE","0"}:
        return 0.0
    try:
        return float(s)
    except Exception:
        return np.nan

def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    m = np.nanmean(x)
    sd = np.nanstd(x, ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        sd = 1.0
    return (x - m) / sd

def panel_symmetric_ylim(v, q=99, pad=1.15, min_half=0.06):
    vv = np.asarray(v, float)
    vv = vv[np.isfinite(vv)]
    if vv.size < 5:
        half = min_half
    else:
        qv = np.nanpercentile(np.abs(vv), q)
        half = max(float(qv) * pad, min_half)
    return (-half, half)

def _bootstrap_line_ci(x, y, xs, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 8:
        return None
    coef = np.polyfit(x, y, 1)
    yhat = coef[0] * xs + coef[1]
    preds = np.empty((n_boot, xs.size), float)
    n = x.size
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        cb = np.polyfit(x[idx], y[idx], 1)
        preds[b] = cb[0] * xs + cb[1]
    lo = np.quantile(preds, alpha/2, axis=0)
    hi = np.quantile(preds, 1-alpha/2, axis=0)
    return yhat, lo, hi


# -------------------------
# Load & merge
# -------------------------
def load_base(root: Path) -> pd.DataFrame:
    data = root / "data"
    p_master = data / "transition_share_master.csv"
    p_pub = data / "fig5a_source_public.csv"
    p_map = data / "fig5a_id_map_private.csv"

    for p in [p_master, p_pub, p_map]:
        if not p.exists():
            raise FileNotFoundError(p)

    master = pd.read_csv(p_master)
    need_m = {"subject","pair","pct","n","total"}
    miss = need_m - set(master.columns)
    if miss:
        raise ValueError(f"transition_share_master.csv missing: {sorted(miss)}")

    master["subject"] = master["subject"].astype(str)
    master["pair"] = master["pair"].astype(str)
    master = _safe_numeric(master, ["pct","n","total"])

    idmap = pd.read_csv(p_map)
    if not {"subject","anon_id"}.issubset(idmap.columns):
        raise ValueError("fig5a_id_map_private.csv must have columns: subject, anon_id")
    idmap["subject"] = idmap["subject"].astype(str)
    idmap["anon_id"] = idmap["anon_id"].astype(str)

    pub = pd.read_csv(p_pub)
    need_p = {"anon_id","site_public","events_sum",SYM} | set(COVARS)
    miss = need_p - set(pub.columns)
    if miss:
        raise ValueError(f"fig5a_source_public.csv missing: {sorted(miss)}")

    pub["site_public"] = pub["site_public"].astype(str)
    pub["sex"] = pub["sex"].map(sex01)
    pub = _safe_numeric(pub, ["events_sum", SYM] + COVARS)

    # treat PANSS==0 as NA (original convention)
    pub.loc[pub[SYM] == 0, SYM] = np.nan

    # total_z from events_sum (SZ-only public table)
    pub["total_z"] = _zscore(pub["events_sum"].to_numpy(float))

    # merge master -> anon -> public
    base = master.merge(idmap, on="subject", how="inner").merge(pub, on="anon_id", how="inner")
    base = base[base["pair"].isin(PAIRS_PLOT)].copy()
    return base


def design_matrix_covars(df: pd.DataFrame) -> pd.DataFrame:
    """
    X = covars + total_z + site dummies (+ const)
    NOTE: exact match of adj_total residualization (PANSS NOT included here).
    """
    X = pd.DataFrame({
        "age": df["age"],
        "sex": df["sex"],
        "JART": df["JART"],
        "sleep": df["sleepiness_pre"],
        "ap_dose": df["antipsychotics"],
        "total_z": df["total_z"],
    })
    site_d = pd.get_dummies(df["site_public"].astype(str), prefix="site", drop_first=True)
    X = pd.concat([X, site_d], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.notna().any(axis=0)]
    if X.isna().any().any():
        X = X.fillna(X.mean(numeric_only=True))
    return sm.add_constant(X, has_constant="add").astype(float)


# -------------------------
# Residualization (for scatter)
# -------------------------
def compute_partial_residuals_for_pair(dfp: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Faithful adj_total:
      y = pct/100, weights = PSEUDO_TOTAL (constant), y_res = y - yhat from Binomial GLM with covars only
      x = PANSS_positive, x_res = x - xhat from OLS with same covars
      x_res is z-scored (as in manuscript x-axis label)
    """
    df = dfp.copy().reset_index(drop=True)
    df = _safe_numeric(df, ["pct","n","total","age","sex","JART","sleepiness_pre","antipsychotics","total_z",SYM])

    # y
    y = np.clip(df["pct"].to_numpy(float) / 100.0, 0.0, 1.0)
    w = np.full_like(y, float(PSEUDO_TOTAL), dtype=float)

    # x
    x = df[SYM].to_numpy(float)

    X0 = design_matrix_covars(df)

    m_ok = np.isfinite(y) & np.isfinite(x) & np.isfinite(X0.to_numpy(float)).all(axis=1)
    if m_ok.sum() < 8:
        return np.array([]), np.array([]), int(m_ok.sum())

    y2 = y[m_ok]; w2 = w[m_ok]; x2 = x[m_ok]
    X2 = X0.loc[m_ok].copy()

    # Binomial GLM (covars only) -> response residual
    fit_y = sm.GLM(y2, X2, family=sm.families.Binomial(), freq_weights=w2).fit()
    y_res = (y2 - fit_y.fittedvalues).astype(float)

    # OLS for symptom (covars only) -> residual -> z-score
    fit_x = sm.OLS(x2, X2).fit()
    x_res = (x2 - fit_x.fittedvalues).astype(float)
    x_res_z = _zscore(x_res)

    return x_res_z, y_res, int(m_ok.sum())


# -------------------------
# OR (GEE) per pair
# -------------------------
def gee_or_for_pair(dfp: pd.DataFrame) -> dict:
    """
    Binomial GEE:
      y = n/total, weights = total (from master; usually 100)
      logit(y) ~ PANSS_positive + covars + total_z + site dummies
      cluster = anon_id
    """
    df = dfp.copy().reset_index(drop=True)
    df = _safe_numeric(df, ["n","total","age","sex","JART","sleepiness_pre","antipsychotics","total_z",SYM])

    need_cols = ["anon_id","site_public","n","total",SYM,"age","sex","JART","sleepiness_pre","antipsychotics","total_z"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=need_cols).copy()
    if df["anon_id"].nunique() < 8:
        return dict(N_subjects=int(df["anon_id"].nunique()), OR=np.nan, OR_LCL=np.nan, OR_UCL=np.nan, p=np.nan)

    y = (df["n"].to_numpy(float) / df["total"].to_numpy(float)).astype(float)
    w = df["total"].to_numpy(float)

    X = pd.DataFrame({
        "PANSS": df[SYM],
        "age": df["age"],
        "sex": df["sex"],
        "JART": df["JART"],
        "sleep": df["sleepiness_pre"],
        "ap_dose": df["antipsychotics"],
        "total_z": df["total_z"],
    })
    site_d = pd.get_dummies(df["site_public"].astype(str), prefix="site", drop_first=True)
    X = pd.concat([X, site_d], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.notna().any(axis=0)]
    if X.isna().any().any():
        X = X.fillna(X.mean(numeric_only=True))
    X = sm.add_constant(X, has_constant="add").astype(float)

    groups = pd.Categorical(df["anon_id"].astype(str)).codes
    gee = GEE(y, X, groups=groups, family=Binomial(), cov_struct=Exchangeable(), weights=w)
    res = gee.fit()

    if "PANSS" not in res.params.index:
        return dict(N_subjects=int(df["anon_id"].nunique()), OR=np.nan, OR_LCL=np.nan, OR_UCL=np.nan, p=np.nan)

    beta = float(res.params["PANSS"])
    se = float(res.bse["PANSS"])
    p = float(res.pvalues["PANSS"])

    OR = float(np.exp(beta))
    OR_LCL = float(np.exp(beta - 1.96*se))
    OR_UCL = float(np.exp(beta + 1.96*se))

    return dict(
        N_subjects=int(df["anon_id"].nunique()),
        beta=beta, se=se, p=p,
        OR=OR, OR_LCL=OR_LCL, OR_UCL=OR_UCL
    )


# -------------------------
# Main
# -------------------------
def main():
    root = Path(__file__).resolve().parents[1]  # NN_open_code
    out_fig = root / "outputs" / "figures"
    out_tab = root / "outputs" / "tables"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_tab.mkdir(parents=True, exist_ok=True)

    base = load_base(root)

    # OR stats per pair
    stat_rows = []
    for pair in PAIRS_PLOT:
        dfp = base[base["pair"] == pair].copy()
        r = gee_or_for_pair(dfp)
        r["pair"] = pair
        stat_rows.append(r)

    df_or = pd.DataFrame(stat_rows)
    # BH-FDR across pairs within this panel (PANSS only)
    df_or["q_FDR_pair"] = np.nan
    m = df_or["p"].notna()
    if m.any():
        df_or.loc[m, "q_FDR_pair"] = multipletests(df_or.loc[m, "p"].values, method="fdr_bh")[1]


    # ---- Plot residual scatter with OR annotation ----
    fig, axes = plt.subplots(1, len(PAIRS_PLOT), figsize=(4.2*len(PAIRS_PLOT), 3.8), dpi=160)
    axes = np.atleast_1d(axes)

    for ax, pair in zip(axes, PAIRS_PLOT):
        dfp = base[base["pair"] == pair].copy()
        x_res_z, y_res, n_ok = compute_partial_residuals_for_pair(dfp)

        if x_res_z.size == 0:
            ax.axis("off")
            ax.set_title(pair)
            continue

        ax.scatter(x_res_z, y_res, s=18, alpha=0.75, color="black")

        # regression line + CI band (visual)
        xs = np.linspace(np.nanmin(x_res_z), np.nanmax(x_res_z), 120)
        out_ci = _bootstrap_line_ci(x_res_z, y_res, xs, n_boot=CI_N_BOOT, alpha=0.05, seed=CI_SEED) if ADD_CI_BAND else None
        if out_ci is None:
            coef = np.polyfit(x_res_z, y_res, 1)
            ax.plot(xs, coef[0]*xs + coef[1], lw=1.1, color="black")
        else:
            yhat, ylo, yhi = out_ci
            ax.plot(xs, yhat, lw=1.1, color="black")
            ax.fill_between(xs, ylo, yhi, color="black", alpha=0.15, linewidth=0)

        # OR annotation from df_or
        hit = df_or[df_or["pair"] == pair]
        if len(hit) == 1:
            r = hit.iloc[0]
            qtxt = r["q_FDR_pair"]
            txt = (
                f"OR={r['OR']:.3f} [{r['OR_LCL']:.3f},{r['OR_UCL']:.3f}]\n"
                f"FDR-p={qtxt:.4g}" if pd.notna(qtxt) else
                f"OR={r['OR']:.3f} [{r['OR_LCL']:.3f},{r['OR_UCL']:.3f}]\n"
                f"p={r['p']:.3g}"
            )
            ax.text(0.02, 0.98, txt, transform=ax.transAxes, ha="left", va="top", fontsize=9)

        ax.set_title(pair)
        ax.set_xlabel("Symptom score (covariate-adjusted, z-scored)")
        ax.set_ylabel("Transition share (GLM residual)")
        ax.set_xlim(*panel_symmetric_ylim(x_res_z, min_half=0.05))
        ax.set_ylim(*panel_symmetric_ylim(y_res, min_half=0.06))
        ax.grid(alpha=0.3, ls="--", lw=0.6)

    fig.suptitle("(c) Transition residual and positive symptoms (SZ only)", y=1.03, fontsize=13)
    fig.tight_layout()

    out_pdf = out_fig / "Fig5c_transition_residual_vs_PANSSpos_with_OR.pdf"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("[OK] Saved figure:", out_pdf)
    print("[OK] Saved stats :", out_csv)


if __name__ == "__main__":
    main()
