#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 18:51:29 2026

@author: takeiyuuichi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make a PUBLIC (de-identified) source CSV for Fig4d:
  - x: log10(total ripple count 80–240 Hz + 1)
  - y: HIP↔CTX transition residual (covariate-adjusted; group NOT included)

This script is meant to be run LOCALLY where clin.csv is available.
The output CSV contains NO clinical covariates.

Inputs (default under NN_open_code/data/):
  - transition_share_master.csv
      required columns: subject, pair, pct
  - hippo_cortex_count_ratio_by_subject_pooled_80_240.csv
      required columns: subject, total_ripples_80_240 OR (n_hip, n_ctx)
  - clin.csv   (PRIVATE; not for sharing)
      required columns (configurable mapping):
        ID, site, S_ID, age, sex, JART, sleepiness_pre, antipsychotics

Outputs (NN_open_code/data/):
  - fig4d_source_public.csv   (SAFE TO SHARE)
  - (optional) fig4d_subject_id_map_private.csv (DO NOT SHARE; for your internal traceability)

Public CSV columns:
  - anon_id, group, site, log10_ripple, hipctx_residual, total_ripples_80_240
"""

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import statsmodels.api as sm


# -------------------------
# Settings
# -------------------------
PAIRS_HIPCTX = ["HIP→CTX", "CTX→HIP"]

# Map clin.csv column names (edit if needed)
COVAR_COLMAP = dict(
    subject="ID",
    site="site",
    S_ID="S_ID",  # 1=HC, 2=SZ
    age="age",
    sex="sex",
    JART="JART",
    sleep="sleepiness_pre",
    apdose="antipsychotics",
)

# How to anonymize IDs
ANON_PREFIX = "subj_"
ANON_DIGITS = 4  # subj_0001 style
# Set to True if you want stable hashing without a private mapping file
USE_HASH_IDS = False

# -------------------------
# Helpers
# -------------------------
def sex01(x):
    s = str(x).strip().upper()
    if s in {"M", "MALE", "1"}:
        return 1.0
    if s in {"F", "FEMALE", "0"}:
        return 0.0
    try:
        return 1.0 if float(s) >= 0.5 else 0.0
    except Exception:
        return np.nan

def canon_subject(x) -> str:
    # Keep original string; do not rewrite here (your clin/transition/rate should already match)
    return str(x).strip()

def hash_id(s: str) -> str:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()[:10]
    return f"{ANON_PREFIX}{h}"

def build_anon_map(subjects: list[str]) -> pd.DataFrame:
    subs = sorted(set([canon_subject(s) for s in subjects]))
    if USE_HASH_IDS:
        anon = [hash_id(s) for s in subs]
    else:
        anon = [f"{ANON_PREFIX}{i+1:0{ANON_DIGITS}d}" for i in range(len(subs))]
    return pd.DataFrame({"subject": subs, "anon_id": anon})


def main():
    root = Path(__file__).resolve().parents[1]  # NN_open_code
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    path_trans = data_dir / "transition_share_master.csv"
    path_rate  = data_dir / "hippo_cortex_count_ratio_by_subject_pooled_80_240.csv"
    path_clin  = data_dir / "clin.csv"  # PRIVATE input (local only)

    if not path_trans.exists():
        raise FileNotFoundError(path_trans)
    if not path_rate.exists():
        raise FileNotFoundError(path_rate)
    if not path_clin.exists():
        raise FileNotFoundError(
            f"{path_clin} not found.\n"
            "This script must be run locally with clin.csv present. "
            "The OUTPUT will be safe to share, but clin.csv itself should not be shared."
        )

    # ---- transition share: HIP↔CTX = HIP→CTX + CTX→HIP ----
    tr = pd.read_csv(path_trans)
    need_tr = {"subject", "pair", "pct"}
    miss = need_tr - set(tr.columns)
    if miss:
        raise ValueError(f"transition_share_master.csv missing columns: {sorted(miss)}")

    tr = tr.copy()
    tr["subject"] = tr["subject"].astype(str).map(canon_subject)
    tr["pair"] = tr["pair"].astype(str).str.strip()
    tr["pct"] = pd.to_numeric(tr["pct"], errors="coerce")

    hipctx = (
        tr[tr["pair"].isin(PAIRS_HIPCTX)]
        .groupby("subject", as_index=False)["pct"]
        .sum(min_count=1)
        .rename(columns={"pct": "hipctx_share_pct"})
    )

    # ---- ripple load (pooled 80–240) ----
    rr = pd.read_csv(path_rate)
    if "subject" not in rr.columns:
        raise ValueError("hippo_cortex_count_ratio_by_subject_pooled_80_240.csv must have 'subject' column.")

    rr = rr.copy()
    rr["subject"] = rr["subject"].astype(str).map(canon_subject)

    if "total_ripples_80_240" in rr.columns:
        rr["total_ripples_80_240"] = pd.to_numeric(rr["total_ripples_80_240"], errors="coerce")
    else:
        rr["n_hip"] = pd.to_numeric(rr.get("n_hip"), errors="coerce").fillna(0)
        rr["n_ctx"] = pd.to_numeric(rr.get("n_ctx"), errors="coerce").fillna(0)
        rr["total_ripples_80_240"] = rr["n_hip"] + rr["n_ctx"]

    rr = rr[["subject", "total_ripples_80_240"]].copy()

    # ---- clin (PRIVATE; for residualization only) ----
    clin_raw = pd.read_csv(path_clin)
    must = [COVAR_COLMAP[k] for k in ["subject","site","S_ID","age","sex","JART","sleep","apdose"]]
    miss = [c for c in must if c not in clin_raw.columns]
    if miss:
        raise ValueError(f"clin.csv missing columns: {miss}")

    clin = pd.DataFrame({
        "subject": clin_raw[COVAR_COLMAP["subject"]].astype(str).map(canon_subject),
        "site": clin_raw[COVAR_COLMAP["site"]].astype(str),
        "S_ID": pd.to_numeric(clin_raw[COVAR_COLMAP["S_ID"]], errors="coerce"),
        "age": pd.to_numeric(clin_raw[COVAR_COLMAP["age"]], errors="coerce"),
        "sex01": clin_raw[COVAR_COLMAP["sex"]].map(sex01),
        "JART": pd.to_numeric(clin_raw[COVAR_COLMAP["JART"]], errors="coerce"),
        "sleep": pd.to_numeric(clin_raw[COVAR_COLMAP["sleep"]], errors="coerce"),
        "ap_dose": pd.to_numeric(clin_raw[COVAR_COLMAP["apdose"]], errors="coerce"),
    })

    # ---- merge ----
    df = hipctx.merge(clin, on="subject", how="inner").merge(rr, on="subject", how="inner")

    df["group"] = df["S_ID"].map({1: "HC", 2: "SZ"})
    df = df[df["group"].isin(["HC", "SZ"])].copy()

    df["total_ripples_80_240"] = pd.to_numeric(df["total_ripples_80_240"], errors="coerce")
    df["log10_ripple"] = np.log10(df["total_ripples_80_240"] + 1.0)

    # ---- residualization: hipctx_share_pct ~ covars + site dummies (NO group) ----
    y = pd.to_numeric(df["hipctx_share_pct"], errors="coerce").to_numpy(float)

    X_base = pd.DataFrame({
        "age": df["age"],
        "sex01": df["sex01"],
        "JART": df["JART"],
        "sleep": df["sleep"],
        "ap_dose": df["ap_dose"],
    })
    site_d = pd.get_dummies(df["site"].astype(str), prefix="site", drop_first=True)
    X = pd.concat([X_base, site_d], axis=1).replace([np.inf, -np.inf], np.nan)

    # simple imputation like your reference script
    if X.isna().any().any():
        X = X.fillna(X.mean(numeric_only=True))

    X = sm.add_constant(X, has_constant="add").astype(float)

    m = np.isfinite(y) & np.isfinite(X.to_numpy()).all(axis=1)
    df = df.loc[m].copy()
    y = y[m]
    X = X.loc[m].copy()

    fit = sm.OLS(y, X).fit()
    df["hipctx_residual"] = y - fit.fittedvalues

    # ---- anonymize subject IDs ----
    anon_map = build_anon_map(df["subject"].tolist())
    df = df.merge(anon_map, on="subject", how="left")

    # ---- public output (SAFE) ----
    public = df[[
        "anon_id", "group", "site", "log10_ripple", "hipctx_residual", "total_ripples_80_240"
    ]].copy()

    out_public = data_dir / "fig4d_source_public.csv"
    public.to_csv(out_public, index=False)

    # ---- private map (DO NOT SHARE) ----
    if not USE_HASH_IDS:
        out_private = data_dir / "fig4d_subject_id_map_private.csv"
        anon_map.to_csv(out_private, index=False)
    else:
        out_private = None

    print("[OK] Public CSV written:", out_public)
    if out_private is not None:
        print("[NOTE] Private ID map written (do NOT share):", out_private)
    print("[OK] Residual model fitted. nobs =", int(fit.nobs))


if __name__ == "__main__":
    main()
