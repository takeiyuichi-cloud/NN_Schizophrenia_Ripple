#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary Table S12 builder (RECOMPUTE; privacy-forward, NO clin.csv, NO modelB_OR_long.csv)

This script reproduces the originally adopted Model B GEE for direction-specific transition shares,
but uses an already-anonymized subject-level table as the source of covariates/symptoms/total_z.

Key idea
- Use transition counts (subject × pair × n × total) for the dependent variable.
- Use the provided anonymized subject-level table:
    source_fig5_predicted_share_by_load_symptom_tertiles_d.csv
  as the source of:
    site, age, sex01, JART, sleep, ap_dose, total_ripples_80_240_z,
    PANSS_pos, PANSS_neg, PANSS_gen, GAF
  (one row per subject; we drop duplicates).

Model (per Transition Type × Clinical Measure; SZ only)
- Binomial GEE (logit), exchangeable correlation
- y = n / total
- weights = total (trials)
- cluster = subject
- exog = const + SYMPTOM + age + sex01 + JART + sleep + ap_dose + total_ripples_80_240_z + site dummies

Multiple comparisons
- BH-FDR across 5 transition types within each clinical measure

Inputs (auto-detected; can be overridden):
- transition_share_counts_subject_level.csv
- source_fig5_predicted_share_by_load_symptom_tertiles_d.csv

Outputs (<root>/outputs/tables/):
- Table_S12_transition_ORs.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import re

import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.stats.multitest import multipletests

PAIR_ORDER_RAW = ["CTX→CTX", "CTX→HIP", "HIP→CTX", "HIP→HIP(opposite)", "HIP→HIP(same)"]
PAIR_ORDER_OUT = ["CTX→CTX", "CTX→HIP", "HIP→CTX", "HIP→HIP (opposite)", "HIP→HIP (same)"]

SYM_SPECS = [
    ("PANSS_pos", "PANSS POS"),
    ("PANSS_neg", "PANSS NEG"),
    ("PANSS_gen", "PANSS GEN"),
    ("GAF", "GAF"),
]

COVAR_COLS = ["age", "sex01", "JART", "sleep", "ap_dose", "total_ripples_80_240_z"]

def find_root() -> Path:
    try:
        return Path(__file__).resolve().parents[1]
    except Exception:
        return Path.cwd().resolve()

def canon_subject(x) -> str:
    s = str(x).strip()
    m = re.search(r"(\d+)", s)
    return f"NB_subject_{int(m.group(1))}" if m else s

def norm_pair(p: str) -> str:
    s = str(p).strip()
    # unify arrows/labels
    s = s.replace("<->", "↔").replace("->", "→").replace("=>", "→")
    s = s.replace("HIP→HIP(opposite)", "HIP→HIP (opposite)")
    s = s.replace("HIP→HIP(same)", "HIP→HIP (same)")
    return s

def autodetect_counts(root: Path) -> Path:
    for c in [
        root / "data" / "transition_share_counts_subject_level.csv",
        root / "results_ronbun" / "fig4" / "transition_share_counts_subject_level.csv",
        root / "results_ronbun" / "fig3" / "transition_share_counts_subject_level.csv",
    ]:
        if c.exists():
            return c
    raise FileNotFoundError("transition_share_counts_subject_level.csv not found.")

def autodetect_subject_table(root: Path) -> Path:
    # Prefer project copy, else user-provided location next to script
    for c in [
        root / "data" / "source_fig5_predicted_share_by_load_symptom_tertiles_d.csv",
        root / "results_ronbun" / "fig5" / "panel_c_transversal_SZ" / "source_fig5_predicted_share_by_load_symptom_tertiles_d.csv",
        root / "results_ronbun" / "fig5" / "panel_c_transition_adj_total_spearman" / "source_fig5_predicted_share_by_load_symptom_tertiles_d.csv",
        Path(__file__).resolve().parent / "source_fig5_predicted_share_by_load_symptom_tertiles_d.csv",
    ]:
        if c.exists():
            return c
    raise FileNotFoundError("source_fig5_predicted_share_by_load_symptom_tertiles_d.csv not found.")

def build_site_dummies(site_series: pd.Series) -> pd.DataFrame:
    return pd.get_dummies(site_series.astype(str), prefix="site", drop_first=True).astype(float)

def design_matrix(df: pd.DataFrame, symptom_col: str) -> pd.DataFrame:
    X = pd.DataFrame({
        "SYMPTOM": pd.to_numeric(df[symptom_col], errors="coerce"),
        "age": pd.to_numeric(df["age"], errors="coerce"),
        "sex01": pd.to_numeric(df["sex01"], errors="coerce"),
        "JART": pd.to_numeric(df["JART"], errors="coerce"),
        "sleep": pd.to_numeric(df["sleep"], errors="coerce"),
        "ap_dose": pd.to_numeric(df["ap_dose"], errors="coerce"),
        "total_ripples_80_240_z": pd.to_numeric(df["total_ripples_80_240_z"], errors="coerce"),
    })
    X = pd.concat([X, build_site_dummies(df["site"])], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.notna().any(axis=0)]
    if X.isna().any().any():
        X = X.fillna(X.mean(numeric_only=True))
    return sm.add_constant(X, has_constant="add").astype(float)

def fit_gee_one(df_pair: pd.DataFrame, symptom_col: str) -> dict | None:
    need = ["subject", "n", "total", "site"] + COVAR_COLS + [symptom_col]
    df = df_pair.replace([np.inf, -np.inf], np.nan).dropna(subset=need).copy()
    if len(df) < 8:
        return None
    df = df.reset_index(drop=True)

    y = (df["n"].to_numpy(float) / df["total"].to_numpy(float)).astype(float)
    w = df["total"].to_numpy(float)

    X = design_matrix(df, symptom_col=symptom_col).reset_index(drop=True)
    groups = pd.Categorical(df["subject"].astype(str)).codes

    gee = GEE(endog=y, exog=X.to_numpy(float), groups=groups,
              family=Binomial(), cov_struct=Exchangeable(), weights=w)
    res = gee.fit()

    # SYMPTOM coefficient index
    cols = list(X.columns)
    j = cols.index("SYMPTOM")
    beta = float(res.params[j])
    se = float(res.bse[j])
    p = float(res.pvalues[j])
    OR = float(np.exp(beta))
    lo = float(np.exp(beta - 1.96 * se))
    hi = float(np.exp(beta + 1.96 * se))
    return {"OR": OR, "CI_low": lo, "CI_high": hi, "p": p, "N": int(df["subject"].nunique())}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None)
    ap.add_argument("--counts", type=str, default=None)
    ap.add_argument("--subject-table", type=str, default=None)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve() if args.root else find_root()
    out_dir = root / "outputs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    path_counts = Path(args.counts).expanduser().resolve() if args.counts else autodetect_counts(root)
    path_subj = Path(args.subject_table).expanduser().resolve() if args.subject_table else autodetect_subject_table(root)

    # counts
    cnt = pd.read_csv(path_counts).copy()
    need = {"subject","pair","n","total","group"}
    if not need.issubset(cnt.columns):
        raise ValueError(f"counts missing columns: {sorted(need-set(cnt.columns))}")
    cnt["subject"] = cnt["subject"].astype(str).map(canon_subject)
    cnt["pair"] = cnt["pair"].astype(str).map(norm_pair)
    cnt["group"] = cnt["group"].astype(str).str.upper().replace({"SC":"SZ"})
    cnt["n"] = pd.to_numeric(cnt["n"], errors="coerce")
    cnt["total"] = pd.to_numeric(cnt["total"], errors="coerce")
    cnt = cnt.replace([np.inf,-np.inf], np.nan).dropna(subset=["n","total"])
    cnt = cnt[(cnt["total"] > 0) & (cnt["n"] >= 0)].copy()
    cnt = cnt[cnt["pair"].isin(PAIR_ORDER_OUT)].copy()
    cnt = cnt[cnt["group"].isin(["SZ"])].copy()

    # anonymized subject table (contains covars + symptoms + total_z)
    subj = pd.read_csv(path_subj).copy()
    if "subject" not in subj.columns:
        raise ValueError("subject table must contain 'subject'")
    subj["subject"] = subj["subject"].astype(str).map(canon_subject)

    # enforce one row per subject
    keep_cols = ["subject","site"] + COVAR_COLS + [c for c,_ in SYM_SPECS]
    miss = [c for c in keep_cols if c not in subj.columns]
    if miss:
        raise ValueError(f"subject table missing columns: {miss}")
    subj = subj[keep_cols].drop_duplicates(subset=["subject"]).copy()

    # merge
    base = cnt.merge(subj, on="subject", how="inner")

    # run models
    rows = []
    for pair in PAIR_ORDER_OUT:
        dfp = base[base["pair"] == pair].copy()
        for sym_col, sym_lab in SYM_SPECS:
            r = fit_gee_one(dfp, symptom_col=sym_col)
            if r is None:
                continue
            rows.append({
                "Transition Type": pair,
                "Clinical Measure": sym_lab,
                "Adjusted Odds Ratio (OR)": r["OR"],
                "95% CI (Lower)": r["CI_low"],
                "95% CI (Upper)": r["CI_high"],
                "p-value": r["p"],
                "N_subjects": r["N"],
            })

    res = pd.DataFrame(rows)
    if res.empty:
        raise RuntimeError("No models were fit. Check that subjects overlap between counts and subject table.")

    # FDR within clinical measure across transition types
    res["FDR-corrected p-value"] = np.nan
    for cm, g in res.groupby("Clinical Measure", dropna=False):
        m = g["p-value"].notna()
        if m.sum():
            res.loc[g.index[m], "FDR-corrected p-value"] = multipletests(g.loc[m, "p-value"], method="fdr_bh")[1]

    def star(q):
        if q is None or (isinstance(q,float) and not np.isfinite(q)):
            return ""
        if q < 1e-3: return "***"
        if q < 1e-2: return "**"
        if q < 5e-2: return "*"
        return ""
    res["sig"] = res["FDR-corrected p-value"].apply(star)

    res["Transition Type"] = pd.Categorical(res["Transition Type"], categories=PAIR_ORDER_OUT, ordered=True)
    res["Clinical Measure"] = pd.Categorical(res["Clinical Measure"], categories=[lab for _,lab in SYM_SPECS], ordered=True)
    res = res.sort_values(["Clinical Measure","Transition Type"]).reset_index(drop=True)

    out_csv = out_dir/"Table_S12_transition_ORs.csv"
   
    res.to_csv(out_csv, index=False)

    print("[OK] Recomputed S12 using anonymized subject table (no clin.csv).")
    print("[OK] Inputs:")
    print(" - counts:", path_counts)
    print(" - subject table:", path_subj)
    print("[OK] Written:")
    print(" -", out_csv)

if __name__ == "__main__":
    main()
