#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig4d (PUBLIC): log10 ripple load vs HIP↔CTX transition residual (HC + SZ)

This script does NOT use clin.csv.
It expects a precomputed, safe-to-share source table:
  NN_open_code/data/fig4d_source_public.csv

Required columns in fig4d_source_public.csv:
  - anon_id
  - group              (HC / SZ)
  - site               (string; optional but recommended)
  - log10_ripple       (= log10(total_ripples_80_240 + 1))
  - hipctx_residual    (covariate-adjusted residual; computed offline)
  - total_ripples_80_240

Outputs (NN_open_code/outputs/):
  - figures/Fig4d_scatter_log10_ripple_vs_HIPCTX_residual_HC_SZ.pdf
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm


# -------------------------
# Settings
# -------------------------
N_BOOT_RHO = 20000
BOOT_SEED = 0
COLOR = {"HC": "#4C72B0", "SZ": "#C44E52"}  # blue/red


# -------------------------
# Helpers
# -------------------------
def boot_spearman_ci(x, y, n_boot=20000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for Spearman rho."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    n = x.size
    if n < 6:
        return (np.nan, np.nan, np.nan, int(n))
    rho_hat, _ = stats.spearmanr(x, y)

    rhos = np.empty(n_boot, float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rb, _ = stats.spearmanr(x[idx], y[idx])
        rhos[b] = rb

    lo = float(np.nanquantile(rhos, alpha/2))
    hi = float(np.nanquantile(rhos, 1 - alpha/2))
    return (float(rho_hat), lo, hi, int(n))

def fisher_z_test(r1, n1, r2, n2):
    """Two-sided Fisher z test for difference in correlations."""
    if not (np.isfinite(r1) and np.isfinite(r2)) or n1 < 4 or n2 < 4:
        return (np.nan, np.nan)
    r1 = np.clip(r1, -0.999999, 0.999999)
    r2 = np.clip(r2, -0.999999, 0.999999)
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)
    se = np.sqrt(1/(n1-3) + 1/(n2-3))
    z = (z1 - z2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return (float(z), float(p))


def main():
    root = Path(__file__).resolve().parents[1]  # NN_open_code
    in_csv = root / "data" / "fig4d_source_public.csv"

    out_fig = root / "outputs" / "figures"
    out_tab = root / "outputs" / "tables"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_tab.mkdir(parents=True, exist_ok=True)

    if not in_csv.exists():
        raise FileNotFoundError(in_csv)

    df = pd.read_csv(in_csv)
    required = {"anon_id", "group", "log10_ripple", "hipctx_residual", "total_ripples_80_240"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"fig4d_source_public.csv missing columns: {sorted(miss)}")

    df = df.copy()
    df["group"] = df["group"].astype(str).str.upper().replace({"SC": "SZ"})
    df = df[df["group"].isin(["HC", "SZ"])].copy()

    x_log = pd.to_numeric(df["log10_ripple"], errors="coerce").to_numpy(float)
    y_res = pd.to_numeric(df["hipctx_residual"], errors="coerce").to_numpy(float)

    mHC = (df["group"] == "HC").to_numpy(bool)
    mSZ = (df["group"] == "SZ").to_numpy(bool)

    # Spearman within groups + bootstrap CI
    rho_HC, lo_HC, hi_HC, nHC = boot_spearman_ci(x_log[mHC], y_res[mHC], n_boot=N_BOOT_RHO, seed=BOOT_SEED+1)
    rho_SZ, lo_SZ, hi_SZ, nSZ = boot_spearman_ci(x_log[mSZ], y_res[mSZ], n_boot=N_BOOT_RHO, seed=BOOT_SEED+2)

    _, p_HC = stats.spearmanr(x_log[mHC], y_res[mHC]) if nHC >= 6 else (np.nan, np.nan)
    _, p_SZ = stats.spearmanr(x_log[mSZ], y_res[mSZ]) if nSZ >= 6 else (np.nan, np.nan)

    # Fisher z difference (rho_HC vs rho_SZ)
    z_diff, p_diff = fisher_z_test(rho_HC, nHC, rho_SZ, nSZ)

    # OLS interaction on residuals (HC3)
    D = df.copy()
    D["is_SZ"] = (D["group"] == "SZ").astype(int)
    X = pd.DataFrame({
        "log10_ripple": pd.to_numeric(D["log10_ripple"], errors="coerce"),
        "is_SZ": D["is_SZ"].astype(float),
    })
    X["log10_x_isSZ"] = X["log10_ripple"] * X["is_SZ"]
    X = sm.add_constant(X, has_constant="add")
    y = pd.to_numeric(D["hipctx_residual"], errors="coerce")
    mm = np.isfinite(y.to_numpy(float)) & np.isfinite(X.to_numpy(float)).all(axis=1)
    fit_int = sm.OLS(y.to_numpy(float)[mm], X.to_numpy(float)[mm]).fit(cov_type="HC3") if mm.sum() >= 8 else None

    # Save stats CSV
    rows = []
    rows.append(dict(analysis="Spearman within HC", N=int(nHC), rho=rho_HC, rho_CI_lo=lo_HC, rho_CI_hi=hi_HC, p=float(p_HC)))
    rows.append(dict(analysis="Spearman within SZ", N=int(nSZ), rho=rho_SZ, rho_CI_lo=lo_SZ, rho_CI_hi=hi_SZ, p=float(p_SZ)))
    rows.append(dict(analysis="Fisher z (HC vs SZ)", N=int(nHC+nSZ), z=z_diff, p=float(p_diff)))

    # ---- OLS interaction stats ----
    if fit_int is not None:
        names = list(fit_int.model.exog_names)
        params = fit_int.params
        bse = fit_int.bse
        pvals = fit_int.pvalues
    
        name_to_idx = {n: i for i, n in enumerate(names)}
    
        for term in ["const", "log10_ripple", "is_SZ", "log10_x_isSZ"]:
            if term in name_to_idx:
                i = name_to_idx[term]
                rows.append(dict(
                    analysis="OLS interaction (HC3)",
                    term=term,
                    beta=float(params[i]),
                    se=float(bse[i]),
                    p=float(pvals[i]),
                ))
            else:
                rows.append(dict(
                    analysis="OLS interaction (HC3)",
                    term=term,
                    beta=np.nan,
                    se=np.nan,
                    p=np.nan,
                ))
  


    # Save report md
    lines = []
    lines.append("# Fig4d stats (public)\n")
    lines.append(f"- HC: N={nHC}, Spearman rho={rho_HC:.3f} (95% CI {lo_HC:.3f}–{hi_HC:.3f}), p={p_HC:.3g}")
    lines.append(f"- SZ: N={nSZ}, Spearman rho={rho_SZ:.3f} (95% CI {lo_SZ:.3f}–{hi_SZ:.3f}), p={p_SZ:.3g}")
    lines.append(f"- Difference (Fisher z): z={z_diff:.3g}, p={p_diff:.3g}\n")
    if fit_int is not None:
        lines.append("## OLS interaction on residuals (HC3)\n")
        lines.append(fit_int.summary().as_text())
    out_report.write_text("\n".join(lines), encoding="utf-8")

    # Plot (match your reference style)
    plt.figure(figsize=(6.8, 4.8))

    for grp, mask in [("HC", mHC), ("SZ", mSZ)]:
        plt.scatter(x_log[mask], y_res[mask], alpha=0.85, edgecolor="k", linewidth=0.4,
                    color=COLOR[grp], label=f"{grp} (N={int(mask.sum())})")

    # group-wise fit lines (visual)
    for grp, mask in [("HC", mHC), ("SZ", mSZ)]:
        mm2 = mask & np.isfinite(x_log) & np.isfinite(y_res)
        if mm2.sum() >= 2:
            coef = np.polyfit(x_log[mm2], y_res[mm2], 1)
            xs = np.linspace(np.nanmin(x_log[mm2]), np.nanmax(x_log[mm2]), 100)
            plt.plot(xs, coef[0]*xs + coef[1], color=COLOR[grp],
                     lw=1.4, linestyle="--" if grp == "SZ" else "-")

    plt.xlabel("log10(total ripple count 80–240 Hz + 1)")
    plt.ylabel("HIP↔CTX transition (covariate-adjusted residual)")
    plt.title(f"HC: ρ={rho_HC:.2f}, p={p_HC:.3g}    SZ: ρ={rho_SZ:.2f}, p={p_SZ:.3g}", fontsize=10)
    plt.grid(alpha=0.3, ls="--", lw=0.6)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()

    out_pdf = out_fig / "Fig4d_scatter_log10_ripple_vs_HIPCTX_residual_HC_SZ.pdf"
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close()

    print("[OK] Written:")
    print(" -", out_stats)
    print(" -", out_report)
    print(" -", out_pdf)


if __name__ == "__main__":
    main()
