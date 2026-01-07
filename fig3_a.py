#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig3a: Example of temporally clustered SWRs (single example)
- top: sliding-window rate(t) + surrogate threshold + high-rate epoch shading
- bottom: event raster + same epoch shading

Input:
  NN_open_code/data/sample_candidate_120Hz.npz

NPZ accepted formats:
  (1) Full-prepared:
      t (T,), rate (T,), threshold (scalar), event_times (N,), epochs (K,2)
  (2) Minimal:
      event_times (N,)   OR   candidate (N,>=3) with [?, start, end, ...]
      -> This script computes t/rate/threshold/epochs using the same logic as your S3x pipeline.

Output:
  NN_open_code/outputs/figures/Fig3a_rate_example_subject82_120Hz.pdf  (filename is generic)
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------
# Settings (match your S3x pipeline)
# -------------------------
RECORDING_LEN_S = 300.0
WIN_S  = 5.0
STEP_S = 1.0
SURR_Q_FOR_THRESHOLD = 0.95
N_SURR = 500
RNG_SEED = 0
MIN_EPOCH_DUR_S = 5.0

EPOCH_SHADE_COLOR = "tab:orange"
LINE_COLOR = "black"


# -------------------------
# Core computations (ported from your S3x logic)
# -------------------------
def sliding_rate(times_s: np.ndarray, T: float, win_s: float, step_s: float):
    starts = np.arange(0.0, max(0.0, T - win_s) + 1e-9, step_s)
    centers = starts + win_s / 2.0
    rate = np.zeros_like(centers, dtype=float)
    if times_s.size == 0:
        return centers, rate
    ts = np.clip(np.asarray(times_s, float), 0.0, T)
    ts = ts[np.isfinite(ts)]
    ts.sort()
    for i, st in enumerate(starts):
        en = st + win_s
        n = np.searchsorted(ts, en, side="left") - np.searchsorted(ts, st, side="left")
        rate[i] = n / win_s
    return centers, rate

def poisson_surrogate_times(n_events: int, T: float, rng: np.random.Generator) -> np.ndarray:
    if n_events <= 0:
        return np.array([], dtype=float)
    return np.sort(rng.uniform(0.0, T, size=int(n_events)).astype(float))

def extract_high_rate_epochs(t_centers: np.ndarray, rate: np.ndarray, win_s: float, thr: float, min_dur_s: float):
    if t_centers.size == 0 or (not np.isfinite(thr)):
        return np.empty((0, 2), float)
    mask = np.isfinite(rate) & (rate >= thr)
    if not mask.any():
        return np.empty((0, 2), float)

    idx = np.where(mask)[0]
    runs = []
    run_s = idx[0]
    prev = idx[0]
    for k in idx[1:]:
        if k == prev + 1:
            prev = k
        else:
            runs.append((run_s, prev))
            run_s = k
            prev = k
    runs.append((run_s, prev))

    half = win_s / 2.0
    epochs = []
    for a, b in runs:
        st = float(t_centers[a] - half)
        en = float(t_centers[b] + half)
        if (en - st) >= min_dur_s:
            epochs.append([st, en])
    return np.asarray(epochs, float)

def _shade_epochs(ax, epochs):
    for st, en in np.asarray(epochs, float):
        if np.isfinite(st) and np.isfinite(en) and en > st:
            ax.axvspan(st, en, color=EPOCH_SHADE_COLOR, alpha=0.18, lw=0)

def _event_raster(ax, times_s: np.ndarray, T: float):
    ts = np.asarray(times_s, float)
    ts = ts[np.isfinite(ts)]
    ts = ts[(ts >= 0.0) & (ts <= T)]
    if ts.size:
        ax.vlines(ts, 0.0, 1.0, color="black", alpha=0.65, linewidth=0.7)
    else:
        ax.text(0.5, 0.5, "No events", transform=ax.transAxes, ha="center", va="center")
    ax.set_xlim(0, T)
    ax.set_ylim(0, 1)
    ax.set_yticks([])

def _get_event_times_from_npz(z) -> np.ndarray:
    if "event_times" in z.files:
        ev = np.asarray(z["event_times"], float)
        return ev[np.isfinite(ev)]
    if "candidate" in z.files:
        cand = np.asarray(z["candidate"])
        cand = np.squeeze(cand)
        if cand.size == 0:
            return np.array([], float)
        if cand.ndim == 1:
            cand = cand.reshape(1, -1)
        # assume cols: [?, start, end, ...] like your MAT
        if cand.shape[1] < 3:
            return np.array([], float)
        t0 = cand[:, 1].astype(float)
        t1 = cand[:, 2].astype(float)
        ev = 0.5 * (t0 + t1)
        ev = ev[np.isfinite(ev)]
        return np.sort(ev)
    raise KeyError("NPZ must contain either 'event_times' or 'candidate' (or full-prepared keys).")

def _load_or_compute(npz_path: Path):
    z = np.load(npz_path, allow_pickle=False)

    # Full-prepared path
    full_keys = {"t", "rate", "threshold", "event_times", "epochs"}
    if full_keys.issubset(set(z.files)):
        t = np.asarray(z["t"], float)
        rate = np.asarray(z["rate"], float)
        thr = float(np.asarray(z["threshold"]).reshape(-1)[0])
        ev = np.asarray(z["event_times"], float)
        epochs = np.asarray(z["epochs"], float)
        return t, rate, thr, ev, epochs

    # Otherwise compute from event_times/candidate
    ev = _get_event_times_from_npz(z)
    ev = ev[(ev >= 0.0) & (ev <= RECORDING_LEN_S)]
    ev.sort()

    # rate
    t, rate = sliding_rate(ev, RECORDING_LEN_S, WIN_S, STEP_S)

    # surrogate threshold (max-rate distribution)
    rng = np.random.default_rng(RNG_SEED)
    surr_max = []
    for _ in range(N_SURR):
        st = poisson_surrogate_times(ev.size, RECORDING_LEN_S, rng)
        _, sr = sliding_rate(st, RECORDING_LEN_S, WIN_S, STEP_S)
        if sr.size:
            surr_max.append(float(np.nanmax(sr)))
    surr_max = np.asarray(surr_max, float)
    thr = float(np.nanquantile(surr_max, SURR_Q_FOR_THRESHOLD)) if surr_max.size else np.nan

    # epochs
    epochs = extract_high_rate_epochs(t, rate, WIN_S, thr, MIN_EPOCH_DUR_S)
    return t, rate, thr, ev, epochs


# -------------------------
# Plot
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None)
    ap.add_argument("--input", type=str, default=None,
                    help="NPZ path (default: <root>/data/sample_candidate_120Hz.npz)")
    ap.add_argument("--out", type=str, default=None,
                    help="Output PDF path (default: <root>/outputs/figures/Fig3a_rate_example.pdf)")
    args = ap.parse_args()

    if args.root is not None:
        root = Path(args.root).expanduser().resolve()
    else:
        try:
            root = Path(__file__).resolve().parents[1]
        except NameError:
            root = Path.cwd().resolve()

    in_npz = Path(args.input).expanduser().resolve() if args.input else (root / "data" / "sample_candidate_120Hz.npz")
    out_dir = root / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = Path(args.out).expanduser().resolve() if args.out else (out_dir / "Fig3a_rate_example.pdf")

    if not in_npz.exists():
        raise FileNotFoundError(in_npz)

    t, rate, thr, ev, epochs = _load_or_compute(in_npz)

    fig = plt.figure(figsize=(10.5, 4.6), dpi=160)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.0], hspace=0.08)

    ax_rate = fig.add_subplot(gs[0, 0])
    ax_ras  = fig.add_subplot(gs[1, 0], sharex=ax_rate)

    # rate panel
    _shade_epochs(ax_rate, epochs)
    ax_rate.plot(t, rate, lw=1.6, color=LINE_COLOR)
    if np.isfinite(thr):
        ax_rate.axhline(thr, ls="--", lw=1.0, color="gray")
        ax_rate.text(0.99, 0.95, "Surrogate threshold", transform=ax_rate.transAxes,
                     ha="right", va="top", fontsize=9)
    ax_rate.set_xlim(0, RECORDING_LEN_S)
    ax_rate.set_ylabel("Rate (events/s)")
    ax_rate.grid(alpha=0.25)
    ax_rate.tick_params(labelbottom=False)

    # raster
    _shade_epochs(ax_ras, epochs)
    _event_raster(ax_ras, ev, RECORDING_LEN_S)
    ax_ras.set_xlabel("Time (s)")
    ax_ras.set_ylabel("Ripple event\noccurrences")

    fig.suptitle("(a) Example of temporally clustered SWRs", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("[OK] Saved:", out_pdf)


if __name__ == "__main__":
    main()