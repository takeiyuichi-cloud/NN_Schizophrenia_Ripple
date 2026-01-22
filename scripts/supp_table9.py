#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary Table S9 (FINAL; S8 framework + S9 adjustments; SZ only):
Clinical associations with ripple clustering outcomes (SZ only)

ALIGNMENT
---------
- Matches Supplementary Table S8 framework:
    * One model per clinical predictor (PANSS POS / NEG / GEN / GAF)
    * Count outcomes: Negative Binomial GLM (log link) + C(site_public)
      offset = log(minutes_sum), HC3 robust SE
    * Continuous outcome: OLS + C(site_public), HC3 robust SE
    * BH-FDR across clinical predictors WITHIN each outcome
- S9 adjustments:
    * Additionally adjust for total ripple load (events_sum) + covariates
    * Outcomes pooled across 80–240 Hz (sum across freqs for counts)

OUTCOMES (pooled 80–240 Hz)
--------------------------
(i)  Number of high-rate ripple epochs (count)
     = sum(n_epochs) across freqs
(ii) Ripple events outside high-rate epochs (count)
     = sum(n_events_outside_epochs) across freqs
(iii) Peak rate within clustered ripple events (counts/s)
     = mean(max_epoch_rate_hz) across freqs (default) or max(...) if AGG_PEAK="max"

NOTE
----
Previously S9 used "Mean ripple density within high-rate epochs (events/s)" defined as:
    sum(n_events_in_epochs) / sum(sum_epoch_dur_s)
If you still want to keep that density metric in the table, set INCLUDE_DENSITY=True
to add it as an additional outcome (optional).

Inputs:
  <root>/data/fig5a_source_public.csv      (SAFE)
  <root>/data/fig5a_id_map_private.csv     (PRIVATE; subject <-> anon_id)
  <root>/data/rate_epoch_subject_level.csv (from sliding-window pipeline)

Outputs (<root>/outputs/tables/):
  - Table_S9_clinical_associations_with_cluster_metrics.csv
"""

from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


# -------------------------
# Settings
# -------------------------
FREQS_USE = [80, 120, 160, 200, 240]
COVARS = ["age", "sex", "JART", "sleepiness_pre", "antipsychotics"]

PRED_MAP = [
    ("PANSS_positive", "PANSS POS"),
    ("PANSS_negative", "PANSS NEG"),
    ("PANSS_pasological", "PANSS GEN"),  # legacy column name in this project
    ("GAF", "GAF"),
]

# Peak aggregation across freqs for the intensity outcome
AGG_PEAK = "mean"          # "mean" (default) or "max"
INCLUDE_DENSITY = False    # set True to also include mean_within_epoch_density as extra outcome


# -------------------------
# Helpers
# -------------------------
def canon_subject(x) -> str:
    s = str(x).strip()
    m = re.search(r"(\d+)", s)
    return f"NB_subject_{int(m.group(1))}" if m else s

def _resolve_panss_gen(df: pd.DataFrame) -> pd.DataFrame:
    """Support PANSS_general as alias for PANSS_pasological if needed."""
    df = df.copy()
    if "PANSS_pasological" not in df.columns and "PANSS_general" in df.columns:
        df["PANSS_pasological"] = df["PANSS_general"]
    return df

def _dropna(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols, how="any").copy()


# -------------------------
# Main
# -------------------------
def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    out_dir = root / "outputs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    clin_csv = data_dir / "fig5a_source_public.csv"
    id_map  = data_dir / "fig5a_id_map_private.csv"
    rate_csv = data_dir / "rate_epoch_subject_level.csv"

    for p in [clin_csv, id_map, rate_csv]:
        if not p.exists():
            raise FileNotFoundError(p)

    # ---- clinical ----
    clin = pd.read_csv(clin_csv)
    clin = _resolve_panss_gen(clin)

    # required base columns
    req = {"anon_id", "site_public", "events_sum", "minutes_sum"}
    miss = req - set(clin.columns)
    if miss:
        raise ValueError(f"fig5a_source_public.csv missing: {sorted(miss)}")

    num_cols = ["events_sum", "minutes_sum"] + COVARS + [c for c, _ in PRED_MAP]
    for c in num_cols:
        if c in clin.columns:
            clin[c] = pd.to_numeric(clin[c], errors="coerce")
    clin["site_public"] = clin["site_public"].astype(str)
    clin["anon_id"] = clin["anon_id"].astype(str)

    # SZ-only if group exists
    if "group" in clin.columns:
        clin["group"] = clin["group"].astype(str).str.upper().replace({"SC": "SZ"})
        clin = clin[clin["group"].isin(["SZ"])].copy()

    # ---- id map (private) ----
    mp = pd.read_csv(id_map)
    if not {"subject", "anon_id"}.issubset(mp.columns):
        raise ValueError("fig5a_id_map_private.csv must have columns: subject, anon_id")
    mp = mp.copy()
    mp["subject"] = mp["subject"].map(canon_subject)
    mp["anon_id"] = mp["anon_id"].astype(str)

    # ---- rate_epoch_subject_level ----
    rate = pd.read_csv(rate_csv)

    need_rate = {"subject", "freq", "n_epochs", "n_events", "n_events_in_epochs", "max_epoch_rate_hz"}
    if INCLUDE_DENSITY:
        need_rate |= {"sum_epoch_dur_s"}
    miss = need_rate - set(rate.columns)
    if miss:
        raise ValueError(f"rate_epoch_subject_level missing columns: {sorted(miss)}")

    rate = rate.copy()
    rate["subject"] = rate["subject"].map(canon_subject)
    rate["freq"] = pd.to_numeric(rate["freq"], errors="coerce").astype(int)

    num_rate_cols = ["n_epochs", "n_events", "n_events_in_epochs", "max_epoch_rate_hz"]
    if INCLUDE_DENSITY:
        num_rate_cols.append("sum_epoch_dur_s")
    for c in num_rate_cols:
        rate[c] = pd.to_numeric(rate[c], errors="coerce")

    rate = rate[rate["freq"].isin(FREQS_USE)].copy()

    rate["n_events_outside_epochs"] = rate["n_events"] - rate["n_events_in_epochs"]

    peak_agg = "max" if AGG_PEAK == "max" else "mean"

    # pooled outcomes (subject level)
    agg_dict = dict(
        clustered_epochs=("n_epochs", "sum"),
        outside_ripple_count=("n_events_outside_epochs", "sum"),
        peak_rate_within_clustered_swrs=("max_epoch_rate_hz", peak_agg),
    )

    if INCLUDE_DENSITY:
        # optional: also compute mean density within epochs
        # density = sum(in_events) / sum(in_time_s)
        agg_dict.update(
            in_events=("n_events_in_epochs", "sum"),
            in_time_s=("sum_epoch_dur_s", "sum"),
        )

    agg = rate.groupby(["subject"], as_index=False).agg(**agg_dict)

    if INCLUDE_DENSITY:
        agg["mean_within_epoch_density"] = agg["in_events"] / agg["in_time_s"]
        bad = (agg["in_time_s"] <= 0) | (~np.isfinite(agg["mean_within_epoch_density"]))
        agg.loc[bad, "mean_within_epoch_density"] = np.nan

    # attach anon_id
    agg = agg.merge(mp, on="subject", how="left").dropna(subset=["anon_id"]).copy()
    agg["anon_id"] = agg["anon_id"].astype(str)

    # merge
    D = clin.merge(agg, on="anon_id", how="inner")
    D = D.replace([np.inf, -np.inf], np.nan).copy()

    # outcomes list
    outcomes = [
        ("clustered_epochs", "Number of high-rate ripple epochs", "count", "IRR"),
        ("outside_ripple_count", "Ripple events outside high-rate epochs", "count", "IRR"),
        ("peak_rate_within_clustered_swrs", "Peak rate within clustered ripple events (counts/s)", "cont", "beta"),
    ]
    if INCLUDE_DENSITY:
        outcomes.append(
            ("mean_within_epoch_density", "Mean ripple density within high-rate epochs (events/s)", "cont", "beta")
        )

    fam = sm.families.NegativeBinomial(link=sm.families.links.Log())

    rows = []
    for out_col, out_label, out_type, effect_type in outcomes:
        tmp = []
        for pred_col, pred_label in PRED_MAP:
            if pred_col not in D.columns:
                continue

            if out_type == "count":
                cols_need = ["site_public", "minutes_sum", "events_sum", out_col, pred_col] + COVARS
            else:
                cols_need = ["site_public", "events_sum", out_col, pred_col] + COVARS

            X = _dropna(D, cols_need)
            if X.empty:
                continue

            if out_type == "count":
                X = X[X["minutes_sum"] > 0].copy()
                if X.empty:
                    continue
                formula = f"{out_col} ~ {pred_col} + events_sum + " + " + ".join(COVARS) + " + C(site_public)"
                fit = smf.glm(
                    formula=formula,
                    data=X,
                    family=fam,
                    offset=np.log(X["minutes_sum"].astype(float))
                ).fit(cov_type="HC3")

                beta = float(fit.params[pred_col])
                se = float(fit.bse[pred_col])
                p = float(fit.pvalues[pred_col])

                eff = float(np.exp(beta))
                ci_low = float(np.exp(beta - 1.96 * se))
                ci_high = float(np.exp(beta + 1.96 * se))

            else:
                formula = f"{out_col} ~ {pred_col} + events_sum + " + " + ".join(COVARS) + " + C(site_public)"
                fit = smf.ols(formula=formula, data=X).fit(cov_type="HC3")

                beta = float(fit.params[pred_col])
                se = float(fit.bse[pred_col])
                p = float(fit.pvalues[pred_col])

                eff = beta
                ci_low = beta - 1.96 * se
                ci_high = beta + 1.96 * se

            tmp.append(dict(
                Outcome=out_label,
                Predictor=pred_label,
                Effect=eff,
                Effect_type=effect_type,
                CI_low=ci_low,
                CI_high=ci_high,
                p_value=p,
            ))

        if tmp:
            df_tmp = pd.DataFrame(tmp)
            df_tmp["q_fdr"] = multipletests(df_tmp["p_value"].values, method="fdr_bh")[1]
            rows.append(df_tmp)

    if not rows:
        raise RuntimeError("No models were fit. Check predictors/inputs and NA filtering.")

    out = pd.concat(rows, ignore_index=True)

    out_csv = out_dir / "Table_S9_clinical_associations_with_cluster_metrics.csv"
    out.to_csv(out_csv, index=False)

    print("[OK] Written:")
    print(" -", out_csv)
    if INCLUDE_DENSITY:
        print("[INFO] INCLUDE_DENSITY=True: density outcome included in addition to peak rate.")
    print(f"[INFO] Peak aggregation across freqs: AGG_PEAK='{AGG_PEAK}'")


if __name__ == "__main__":
    main()
