#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 09:56:32 2026

@author: takeiyuuichi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a single master table for transition shares (recommended stable version)

Inputs (place in NN_open_code/data/):
  - transition_share_subject_level.csv   (subject, pair, pct)
  - transition_share_counts_subject_level.csv (subject, pair, n, total)  [optional but recommended for audit]

Output:
  - NN_open_code/data/transition_share_master.csv

Policy:
  - pct is taken from subject_level as the source of truth.
  - For Binomial/GEE stability, (n,total) used for modeling are reconstructed from pct
    using a fixed PSEUDO_TOTAL (default=100), mimicking your Fig4b approach.
  - counts-version (n,total) are preserved as *_raw columns for debugging only.
"""

from pathlib import Path
import numpy as np
import pandas as pd

PSEUDO_TOTAL = 100  # fixed denominator used for modeling


def canon_subject(x) -> str:
    s = str(x).strip()
    return s

def main():
    root = Path(__file__).resolve().parents[1]  # NN_open_code
    data = root / "data"

    p_pct = data / "transition_share_subject_level.csv"
    p_cnt = data / "transition_share_counts_subject_level.csv"
    outp = data / "transition_share_master.csv"

    if not p_pct.exists():
        raise FileNotFoundError(p_pct)

    df_pct = pd.read_csv(p_pct)
    need_pct = {"subject", "pair", "pct"}
    miss = need_pct - set(df_pct.columns)
    if miss:
        raise ValueError(f"{p_pct.name} missing columns: {sorted(miss)}")

    df_pct = df_pct.copy()
    df_pct["subject"] = df_pct["subject"].astype(str).map(canon_subject)
    df_pct["pair"] = df_pct["pair"].astype(str)
    df_pct["pct"] = pd.to_numeric(df_pct["pct"], errors="coerce")
    df_pct = df_pct.dropna(subset=["pct"]).copy()

    # optional counts (audit only)
    if p_cnt.exists():
        df_cnt = pd.read_csv(p_cnt)
        need_cnt = {"subject", "pair", "n", "total"}
        miss2 = need_cnt - set(df_cnt.columns)
        if miss2:
            raise ValueError(f"{p_cnt.name} missing columns: {sorted(miss2)}")

        df_cnt = df_cnt.copy()
        df_cnt["subject"] = df_cnt["subject"].astype(str).map(canon_subject)
        df_cnt["pair"] = df_cnt["pair"].astype(str)
        df_cnt["n_raw"] = pd.to_numeric(df_cnt["n"], errors="coerce")
        df_cnt["total_raw"] = pd.to_numeric(df_cnt["total"], errors="coerce")
        df_cnt = df_cnt[["subject", "pair", "n_raw", "total_raw"]].copy()

        base = df_pct.merge(df_cnt, on=["subject", "pair"], how="left")
    else:
        base = df_pct.copy()
        base["n_raw"] = np.nan
        base["total_raw"] = np.nan

    # stable reconstruction for Binomial/GEE
    base["total"] = float(PSEUDO_TOTAL)
    base["n"] = np.rint((base["pct"] / 100.0) * base["total"]).astype(float)

    # clip to [0,total]
    base["n"] = np.clip(base["n"], 0.0, base["total"])

    # diagnostic: how far raw pct is from raw counts pct (if available)
    base["pct_from_raw"] = np.where(
        (base["total_raw"].notna()) & (base["total_raw"] > 0),
        100.0 * (base["n_raw"] / base["total_raw"]),
        np.nan
    )
    base["pct_diff_subject_minus_raw"] = base["pct"] - base["pct_from_raw"]

    # tidy
    base = base.sort_values(["pair", "subject"]).reset_index(drop=True)

    outp.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(outp, index=False)
    print("[OK] saved:", outp)
    print("[INFO] rows:", len(base), "subjects:", base["subject"].nunique(), "pairs:", base["pair"].nunique())
    if base["pct_from_raw"].notna().any():
        md = np.nanmedian(np.abs(base["pct_diff_subject_minus_raw"].to_numpy(float)))
        mx = np.nanmax(np.abs(base["pct_diff_subject_minus_raw"].to_numpy(float)))
        print(f"[INFO] median |pct(subject)-pct(rawcounts)| = {md:.3g} pp ; max = {mx:.3g} pp")

if __name__ == "__main__":
    main()