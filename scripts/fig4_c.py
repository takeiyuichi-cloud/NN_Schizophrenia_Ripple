#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 16:38:48 2026

@author: takeiyuuichi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig4c: Co-occurrence fraction vs window half-width (all pooled)

Input (NN_open_code/data/):
  - cooc_global_by_window_allpooled.csv
    required columns: pair, window_ms, frac

Output (NN_open_code/outputs/figures/):
  - Fig4c_cooccurrence_fraction_vs_window.pdf

Notes:
  - y-axis is plotted as percent (%).
  - Supports frac either in [0,1] or already in [0,100].
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PAIR_ORDER = ["HIP→HIP(same)", "HIP→HIP(opposite)", "CTX→CTX", "HIP→CTX", "CTX→HIP"]

PAIR_COLOR = {
    "HIP→HIP(same)":     "#4C78A8",
    "HIP→HIP(opposite)": "#9EC1E6",
    "CTX→CTX":           "#B79F00",
    "HIP→CTX":           "#E45756",
    "CTX→HIP":           "#54A24B",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None)
    ap.add_argument("--input", type=str, default=None,
                    help="default: <root>/data/cooc_global_by_window_allpooled.csv")
    args = ap.parse_args()

    if args.root is not None:
        root = Path(args.root).expanduser().resolve()
    else:
        try:
            root = Path(__file__).resolve().parents[1]  # NN_open_code
        except NameError:
            root = Path.cwd().resolve()

    in_csv = Path(args.input).expanduser().resolve() if args.input else (root / "data" / "cooc_global_by_window_allpooled.csv")
    out_fig = root / "outputs" / "figures"
    out_fig.mkdir(parents=True, exist_ok=True)

    if not in_csv.exists():
        raise FileNotFoundError(f"Missing input: {in_csv}")

    df = pd.read_csv(in_csv)

    need = {"pair", "window_ms", "frac"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"cooc CSV missing columns: {sorted(miss)}")

    df = df.copy()
    df["pair"] = df["pair"].astype(str).str.strip()
    df["window_ms"] = pd.to_numeric(df["window_ms"], errors="coerce")
    df["frac"] = pd.to_numeric(df["frac"], errors="coerce")
    df = df.dropna(subset=["window_ms", "frac"])

    # Keep only the expected pairs if present
    present_pairs = [p for p in PAIR_ORDER if p in set(df["pair"])]
    if not present_pairs:
        present_pairs = sorted(df["pair"].unique().tolist())

    # auto-detect whether frac is proportion or already percent
    fmax = float(np.nanmax(df["frac"].to_numpy(float)))
    as_percent = df["frac"].to_numpy(float) * 100.0 if fmax <= 1.5 else df["frac"].to_numpy(float)

    df["frac_pct"] = as_percent

    # plot
    plt.figure(figsize=(7.8, 5.0))

    for p in present_pairs:
        sub = df[df["pair"] == p].sort_values("window_ms")
        x = sub["window_ms"].to_numpy(float)
        y = sub["frac_pct"].to_numpy(float)
        plt.plot(
            x, y, "-o",
            label=p,
            color=PAIR_COLOR.get(p, "#666666"),
            alpha=0.95,
            markersize=4,
            linewidth=1.6,
        )

    plt.xlabel("Window half-width (ms)")
    plt.ylabel("Co-occurrence fraction (%)  [A has ≥1 B within ±W]")
    plt.title("Co-occurrence vs. window (all sites/subjects/frequencies pooled)")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()

    out_pdf = out_fig / "Fig4c_cooccurrence_fraction_vs_window.pdf"
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close()

    print("[OK] Saved:", out_pdf)


if __name__ == "__main__":
    main()