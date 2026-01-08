#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 20:24:47 2026

@author: takeiyuuichi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make fig5a_source_public.csv (LOCAL ONLY)

Goal:
- Create a SAFE-TO-SHARE dataset for Fig5a (SZ only), without distributing raw clinical Excel files.
- Join:
  (A) SZ clinical covariates/symptoms from gundai_subdata.xlsx + kumasou_subdata.xlsx
  (B) pooled ripple counts (80–240 Hz) from NN_open_code/data/df_clean_expanded.csv

Outputs:
- NN_open_code/data/fig5a_source_public.csv         (SAFE TO SHARE)
- NN_open_code/data/fig5a_id_map_private.csv        (DO NOT SHARE)

Assumptions:
- df_clean_expanded.csv contains: site, subject, group, cond, freq, event_count
  and includes either:
   - cond == 'filtered_GED_MEG' (preferred), OR
   - cond in {'Hippocampus','Cortex'/'Extrahippocampal'} which can be summed.
- Clinical Excel sheets contain columns: ID, S_ID, and (TARGETS/COVARS as available).
"""

from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd


# -------------------------
# User settings (edit if needed)
# -------------------------
BASE_DIR = Path("/Users/takeiyuuichi/MEG Dropbox/takei yuichi/RIPPLE_BIG_DATA")
NN_OPEN_CODE = Path("/Users/takeiyuuichi/Desktop/NN_open_code")  # adjust if needed

XLSX_GUNDAI  = BASE_DIR / "parameter" / "gundai_subdata.xlsx"
XLSX_KUMASOU = BASE_DIR / "parameter" / "kumasou_subdata.xlsx"

DF_CLEAN = NN_OPEN_CODE / "data" / "df_clean_expanded.csv"

OUT_PUBLIC  = NN_OPEN_CODE / "data" / "fig5a_source_public.csv"
OUT_PRIVATE = NN_OPEN_CODE / "data" / "fig5a_id_map_private.csv"

FREQS_USE = [80, 120, 160, 200, 240]
SZ_SID_VALUE = 2
MINUTES_SUM = 5.0  # 300s fixed in your pipeline

TARGETS = ["PANSS_positive", "PANSS_negative", "PANSS_pasological", "GAF"]
COVARS  = ["age", "sex", "JART", "sleepiness_pre", "antipsychotics"]

# anonymization
ANON_PREFIX = "subj_"
ANON_DIGITS = 4
SITE_REMAP = {"gundai": "SiteA", "kumasou": "SiteB"}  # set to None to keep original site labels


# -------------------------
# Helpers
# -------------------------
def canon_subject(x) -> str:
    """ID -> NB_subject_<int>"""
    s = str(x)
    m = re.search(r"(\d+)", s)
    if not m:
        return s.strip()
    return f"NB_subject_{int(m.group(1))}"

def norm_site(x) -> str:
    return str(x).strip().lower()

def norm_group(x) -> str:
    s = str(x).strip().upper()
    if s == "SC":
        return "SZ"
    if s in ("CONTROL", "HEALTHY"):
        return "HC"
    return s

def norm_cond(x) -> str:
    s = str(x).strip()
    s0 = s.lower()
    if s0 == "filtered_ged_meg":
        return "filtered_GED_MEG"
    if s0 in ("extrahippocampal", "extrahippocampus", "extrahippocapus", "cortex", "cortical", "ctx") or s == "Extrahippocampal":
        return "Cortex"
    if s0 == "hippocampus":
        return "Hippocampus"
    return s

def sex01(x):
    """Convert sex to 0/1 if possible; keeps numeric as-is."""
    s = str(x).strip().upper()
    if s in {"M", "MALE", "1"}:
        return 1.0
    if s in {"F", "FEMALE", "0"}:
        return 0.0
    try:
        return float(s)
    except Exception:
        return np.nan

def to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def make_anon_map(subjects: list[str]) -> pd.DataFrame:
    subs = sorted(set(subjects))
    anon = [f"{ANON_PREFIX}{i+1:0{ANON_DIGITS}d}" for i in range(len(subs))]
    return pd.DataFrame({"subject": subs, "anon_id": anon})


# -------------------------
# Load clinical (SZ only) from Excel (multi-sheet tolerant)
# -------------------------
def load_clin_one(xlsx_path: Path, site: str) -> pd.DataFrame:
    if not xlsx_path.exists():
        raise FileNotFoundError(xlsx_path)

    xls = pd.ExcelFile(xlsx_path)
    dfs = []
    for sh in xls.sheet_names:
        tmp = pd.read_excel(xlsx_path, sheet_name=sh)
        if ("ID" not in tmp.columns) or ("S_ID" not in tmp.columns):
            continue
        tmp = tmp.copy()
        tmp["site"] = site
        tmp["subject"] = tmp["ID"].map(canon_subject).astype(str)
        dfs.append(tmp)

    if not dfs:
        tmp = pd.read_excel(xlsx_path).copy()
        if ("ID" not in tmp.columns) or ("S_ID" not in tmp.columns):
            raise ValueError(f"{xlsx_path} に ID / S_ID が見当たらない。列={list(tmp.columns)}")
        tmp["site"] = site
        tmp["subject"] = tmp["ID"].map(canon_subject).astype(str)
        dfs = [tmp]

    df = pd.concat(dfs, ignore_index=True)

    # SZ only
    df = df[df["S_ID"] == SZ_SID_VALUE].copy()

    # standardize sex if present
    if "sex" in df.columns:
        df["sex"] = df["sex"].map(sex01)

    df = to_numeric(df, TARGETS + COVARS)

    keep = ["site", "subject", "S_ID"] + [c for c in (TARGETS + COVARS) if c in df.columns]
    return df[keep].copy()

def load_clin_sz_only() -> pd.DataFrame:
    g = load_clin_one(XLSX_GUNDAI, "gundai")
    k = load_clin_one(XLSX_KUMASOU, "kumasou")
    df = pd.concat([g, k], ignore_index=True)
    return df


# -------------------------
# Build pooled ripple counts (80–240 Hz) from df_clean_expanded.csv
# -------------------------
def load_and_build_events_sum() -> pd.DataFrame:
    if not DF_CLEAN.exists():
        raise FileNotFoundError(DF_CLEAN)

    df = pd.read_csv(DF_CLEAN)
    need = {"site", "subject", "cond", "freq", "event_count"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"df_clean_expanded.csv missing columns: {sorted(miss)}")

    df = df.copy()
    df["site"] = df["site"].map(norm_site)
    df["subject"] = df["subject"].map(canon_subject)
    df["cond"] = df["cond"].map(norm_cond)
    df["freq"] = pd.to_numeric(df["freq"], errors="coerce")
    df["event_count"] = pd.to_numeric(df["event_count"], errors="coerce")

    df = df[df["freq"].isin(FREQS_USE)].copy()

    # Prefer filtered_GED_MEG if available
    if (df["cond"] == "filtered_GED_MEG").any():
        base = df[df["cond"] == "filtered_GED_MEG"].copy()
        g = (base.groupby(["site", "subject"], as_index=False)["event_count"]
                 .sum(min_count=1)
                 .rename(columns={"event_count": "events_sum"}))
    else:
        # fallback: Hippocampus + Cortex
        base = df[df["cond"].isin(["Hippocampus", "Cortex"])].copy()
        g = (base.groupby(["site", "subject"], as_index=False)["event_count"]
                 .sum(min_count=1)
                 .rename(columns={"event_count": "events_sum"}))

    g["minutes_sum"] = float(MINUTES_SUM)
    return g


# -------------------------
# Main
# -------------------------
def main():
    # 1) load SZ clinical
    clin = load_clin_sz_only()
    if clin.empty:
        raise RuntimeError("SZ clinical table is empty after filtering S_ID==2.")

    # 2) build ripple counts (80–240 pooled)
    ev = load_and_build_events_sum()
    if ev.empty:
        raise RuntimeError("events_sum table is empty (check df_clean_expanded.csv and cond/freq).")

    # 3) merge on site+subject
    df = clin.merge(ev, on=["site", "subject"], how="inner")
    if df.empty:
        raise RuntimeError("Merge clin × events_sum returned empty. Check subject ID formatting and site labels.")

    # 4) anonymize IDs
    anon_map = make_anon_map(df["subject"].tolist())
    df = df.merge(anon_map, on="subject", how="left")

    # 5) site remap (optional)
    if SITE_REMAP is not None:
        df["site_public"] = df["site"].map(lambda s: SITE_REMAP.get(str(s).lower(), str(s)))
    else:
        df["site_public"] = df["site"].astype(str)

    # 6) build PUBLIC table (SAFE)
    cols_public = ["anon_id", "site_public", "events_sum", "minutes_sum"] \
                  + [c for c in TARGETS if c in df.columns] \
                  + [c for c in COVARS if c in df.columns]

    public = df[cols_public].copy()
    # ensure numeric columns are numeric
    public = to_numeric(public, [c for c in cols_public if c not in ("anon_id", "site_public")])

    OUT_PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    public.to_csv(OUT_PUBLIC, index=False)

    # 7) private mapping (DO NOT SHARE)
    private = df[["subject", "anon_id"]].drop_duplicates().sort_values("anon_id")
    private.to_csv(OUT_PRIVATE, index=False)

    print("[OK] Written public dataset:", OUT_PUBLIC)
    print("[OK] Written PRIVATE ID map (do NOT share):", OUT_PRIVATE)
    print("[INFO] Public rows:", len(public), "subjects:", public["anon_id"].nunique())


if __name__ == "__main__":
    main()