#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build transition_share_master_with_site.csv

Inputs (NN_open_code/data/):
  - transition_share_master.csv
      required: subject, group, pair, pct, n_raw, total_raw, total, n
  - fig4d_subject_id_map_private.csv
      required: subject, anon_id
  - fig5a_source_public.csv
      required: anon_id, site_public, PANSS_positive
      optional: events_sum (used as total_ripples_80_240 surrogate), age, sex, JART, sleepiness_pre, antipsychotics

Outputs (NN_open_code/data/):
  - transition_share_master_with_site.csv

Notes:
  - "site" is taken from fig5a_source_public.csv: site_public -> site
  - "PANSS_pos" is PANSS_positive renamed
  - "total_ripples_80_240" is approximated by events_sum if available (public-safe)
  - "total_ripples_80_240_z" is z-score of total_ripples_80_240 within the merged dataset (ignoring NaNs)
  - IMPORTANT: For Binomial/GEE, use n_raw/total_raw (real counts). Do NOT use n/total (pseudo counts).
"""

from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def canon_subject(x) -> str:
    import re
    s = str(x).strip()
    m = re.search(r"(\d+)", s)
    return f"NB_subject_{int(m.group(1))}" if m else s


def zscore_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series([np.nan] * len(x), index=s.index)
    return (x - mu) / sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None, help="NN_open_code root (default: parents[1] of this script)")
    ap.add_argument("--in-master", type=str, default="data/transition_share_master.csv")
    ap.add_argument("--in-idmap", type=str, default="data/fig4d_subject_id_map_private.csv")
    ap.add_argument("--in-public", type=str, default="data/fig5a_source_public.csv")
    ap.add_argument("--out", type=str, default="data/transition_share_master_with_site.csv")
    args = ap.parse_args()

    if args.root is not None:
        root = Path(args.root).expanduser().resolve()
    else:
        try:
            root = Path(__file__).resolve().parents[1]
        except NameError:
            root = Path.cwd().resolve()

    p_master = (root / args.in_master).resolve()
    p_idmap  = (root / args.in_idmap).resolve()
    p_pub    = (root / args.in_public).resolve()
    p_out    = (root / args.out).resolve()

    for p in [p_master, p_idmap, p_pub]:
        if not p.exists():
            raise FileNotFoundError(p)

    # -------------------------
    # Load transition master
    # -------------------------
    df = pd.read_csv(p_master)
    need_master = {"subject", "group", "pair", "pct", "n_raw", "total_raw"}
    miss = sorted(list(need_master - set(df.columns)))
    if miss:
        raise ValueError(f"transition_share_master.csv missing required columns: {miss}")

    df = df.copy()
    df["subject"] = df["subject"].map(canon_subject).astype(str)

    # group normalize (SC -> SZ)
    df["group"] = df["group"].astype(str).str.upper().replace({"SC": "SZ"})
    df["group"] = df["group"].where(df["group"].isin(["HC", "SZ"]), df["group"])

    # numeric
    df["pct"] = pd.to_numeric(df["pct"], errors="coerce")
    df["n_raw"] = pd.to_numeric(df["n_raw"], errors="coerce")
    df["total_raw"] = pd.to_numeric(df["total_raw"], errors="coerce")

    # -------------------------
    # Load private id map (subject -> anon_id)
    # -------------------------
    idmap = pd.read_csv(p_idmap)
    need_idmap = {"subject", "anon_id"}
    miss = sorted(list(need_idmap - set(idmap.columns)))
    if miss:
        raise ValueError(f"fig4d_subject_id_map_private.csv missing required columns: {miss}")

    idmap = idmap.copy()
    idmap["subject"] = idmap["subject"].map(canon_subject).astype(str)
    idmap["anon_id"] = idmap["anon_id"].astype(str)

    # -------------------------
    # Load public table (anon_id -> site_public, PANSS_positive, events_sum...)
    # -------------------------
    pub = pd.read_csv(p_pub)
    need_pub = {"anon_id", "site_public", "PANSS_positive"}
    miss = sorted(list(need_pub - set(pub.columns)))
    if miss:
        raise ValueError(f"fig5a_source_public.csv missing required columns: {miss}")

    pub = pub.copy()
    pub["anon_id"] = pub["anon_id"].astype(str)

    # Keep only necessary cols (but pass through common covariates if present)
    keep_pub = ["anon_id", "site_public", "PANSS_positive"]
    for c in ["events_sum", "age", "sex", "JART", "sleepiness_pre", "antipsychotics", "GAF",
              "PANSS_negative", "PANSS_pasological"]:
        if c in pub.columns:
            keep_pub.append(c)
    pub = pub[keep_pub].copy()

    # Rename to analysis-friendly columns
    pub = pub.rename(columns={
        "site_public": "site",
        "PANSS_positive": "PANSS_pos",
    })

    # numeric
    pub["PANSS_pos"] = pd.to_numeric(pub["PANSS_pos"], errors="coerce")
    if "events_sum" in pub.columns:
        pub["events_sum"] = pd.to_numeric(pub["events_sum"], errors="coerce")

    # -------------------------
    # Merge: master -> idmap -> pub
    # -------------------------
    merged = df.merge(idmap, on="subject", how="left", validate="m:1")
    merged = merged.merge(pub, on="anon_id", how="left", validate="m:1")

    # total ripple load surrogate from public-safe events_sum
    if "events_sum" in merged.columns:
        merged["total_ripples_80_240"] = pd.to_numeric(merged["events_sum"], errors="coerce")
    else:
        merged["total_ripples_80_240"] = np.nan

    merged["total_ripples_80_240_z"] = zscore_series(merged["total_ripples_80_240"])

    # basic sanity flags
    merged["_has_anon_id"] = merged["anon_id"].notna()
    merged["_has_site"] = merged["site"].notna()
    merged["_has_PANSS_pos"] = merged["PANSS_pos"].notna()

    # sort
    merged = merged.sort_values(["pair", "group", "subject"]).reset_index(drop=True)

    p_out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(p_out, index=False)

    print("[OK] saved:", p_out)
    print("[INFO] rows:", len(merged), "subjects:", merged["subject"].nunique(), "pairs:", merged["pair"].nunique())
    print("[INFO] has anon_id:", int(merged["_has_anon_id"].sum()), "/", len(merged))
    print("[INFO] has site:", int(merged["_has_site"].sum()), "/", len(merged))
    print("[INFO] has PANSS_pos:", int(merged["_has_PANSS_pos"].sum()), "/", len(merged))
    print("[INFO] NOTE: Use n_raw/total_raw for Binomial/GEE. Do NOT use n/total (pseudo).")


if __name__ == "__main__":
    main()