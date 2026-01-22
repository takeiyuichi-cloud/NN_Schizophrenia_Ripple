#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PANSS_positive vs pooled ripple counts (SZ only) — single figure output

Input:
  NN_open_code/data/fig4a_source_public.csv

Output:
  NN_open_code/outputs/figures/fig4a_PANSS_positive_vs_ripple_counts.pdf

Model (for adjusted curve):
  events_sum ~ PANSS_positive + age + sex + JART + sleepiness_pre + antipsychotics + C(site_public)
  family: Negative Binomial (log link)
  offset: log(minutes_sum)

Plot:
  - scatter: raw points (PANSS_positive vs events_sum)
  - line: adjusted prediction with 95% CI
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf


def mode_or_median(s: pd.Series):
    s2 = s.dropna()
    if s2.empty:
        return np.nan
    if s2.nunique() <= 3:
        return s2.mode().iloc[0]
    return float(s2.median())


def main():
    root = Path(__file__).resolve().parents[1]  # NN_open_code
    in_csv = root / "data" / "fig4a_source_public.csv"
    out_pdf = root / "outputs" / "figures" / "fig4a_PANSS_positive_vs_ripple_counts.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    if not in_csv.exists():
        raise FileNotFoundError(in_csv)

    df = pd.read_csv(in_csv)

    required = {
        "site_public", "events_sum", "minutes_sum",
        "PANSS_positive",
        "age", "sex", "JART", "sleepiness_pre", "antipsychotics",
    }
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"fig4a_source_public.csv missing columns: {sorted(miss)}")

    # numeric coercion
    num_cols = ["events_sum", "minutes_sum", "PANSS_positive", "age", "sex", "JART", "sleepiness_pre", "antipsychotics"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["site_public"] = df["site_public"].astype(str)

    # drop NA
    cols_need = ["site_public"] + num_cols
    D = df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols_need).copy()
    D = D[D["minutes_sum"] > 0].copy()
    if D.empty:
        raise RuntimeError("No usable rows after dropna/filter.")

    # Fit NB-GLM (alpha fixed=1.0; acceptable for plotting; matches earlier script behavior)
    # Use non-deprecated Log link name
    fam = sm.families.NegativeBinomial(link=sm.families.links.Log())
    formula = "events_sum ~ PANSS_positive + age + sex + JART + sleepiness_pre + antipsychotics + C(site_public)"
    offset = np.log(D["minutes_sum"].astype(float))
    fit = smf.glm(formula=formula, data=D, family=fam, offset=offset).fit(cov_type="HC3")

    # Prediction grid
    x_min, x_max = float(D["PANSS_positive"].min()), float(D["PANSS_positive"].max())
    x_grid = np.linspace(x_min, x_max, 200) if np.isfinite(x_min) and np.isfinite(x_max) and x_min != x_max else np.array([x_min])

    site_mode = D["site_public"].mode().iloc[0]
    fixed = {
        "age": mode_or_median(D["age"]),
        "sex": mode_or_median(D["sex"]),
        "JART": mode_or_median(D["JART"]),
        "sleepiness_pre": mode_or_median(D["sleepiness_pre"]),
        "antipsychotics": mode_or_median(D["antipsychotics"]),
    }
    minutes_fixed = float(mode_or_median(D["minutes_sum"]))

    P = pd.DataFrame({
        "PANSS_positive": x_grid,
        "site_public": site_mode,
        "age": fixed["age"],
        "sex": fixed["sex"],
        "JART": fixed["JART"],
        "sleepiness_pre": fixed["sleepiness_pre"],
        "antipsychotics": fixed["antipsychotics"],
    })

    sf = fit.get_prediction(P, offset=np.log(np.full_like(x_grid, minutes_fixed))).summary_frame(alpha=0.05)
    yhat = sf["mean"].to_numpy(float)
    ylo = sf["mean_ci_lower"].to_numpy(float)
    yhi = sf["mean_ci_upper"].to_numpy(float)

    # Plot
    fig, ax = plt.subplots(figsize=(6.6, 5.0), dpi=160)
    ax.scatter(D["PANSS_positive"], D["events_sum"], s=18, alpha=0.75)

    ax.plot(x_grid, yhat, lw=2)
    ax.fill_between(x_grid, ylo, yhi, alpha=0.2, linewidth=0)

    ax.set_xlabel("PANSS positive")
    ax.set_ylabel("Pooled ripple count (80–240 Hz, per 5 min)")
    ax.set_title("SZ only: ripple count vs PANSS positive", loc="left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("[OK] Saved:", out_pdf)


if __name__ == "__main__":
    main()
