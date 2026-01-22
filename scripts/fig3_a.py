#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig3 panel-a (public sample): Example of temporally clustered SWRs
- input: NPZ file with ripple event times (seconds) for one subject/freq
- output: PDF figure

Update (2026-01):
- High-rate epoch threshold is taken from NPZ if available (surrogate-derived).
- Epoch intervals are taken from NPZ if available.
- Bottom raster color-codes events by region (Hippocampus vs Cortex) if labels exist in NPZ.
- If NPZ lacks labels, optionally fill labels from stc/event_peaks_120Hz.csv (or user-specified CSV).

NPZ patterns supported:
1) Separate arrays (preferred):
   - hip_event_times  (or hippocampus_event_times / hip_times)
   - ctx_event_times  (or cortex_event_times / outside_event_times / ctx_times)
2) Single array + labels:
   - event_times
   - event_region
3) Candidate style:
   - candidate (Nx>=3) with t0,t1 in columns 1,2 (seconds)
   - is_hip (N,) boolean/int (1=HIP,0=CTX)

Optional NPZ keys used if present:
- t          : sliding-window centers (s)
- rate       : sliding-window rate (events/s)
- threshold  : surrogate threshold (preferred)
- epochs     : (K,2) array of [start_s, end_s]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------
# Defaults
# -------------------------
DEFAULT_NPZ = "data/sample_candidate_120Hz.npz"  # relative to NN_open_code root
OUTNAME = "Fig3a_rate_example_subject82_120Hz.pdf"

RECORDING_LEN_S = 300.0
WIN_S  = 5.0
STEP_S = 1.0
MIN_EPOCH_DUR_S = 5.0

# fallback only if NPZ lacks threshold
RATE_Q = 0.95

# raster colors
COLOR_HIP = "tab:red"
COLOR_CTX = "tab:blue"


# -------------------------
# Helpers
# -------------------------
def _canon_region(x) -> str:
    if x is None:
        return "UNK"
    s = str(x).strip().upper()
    if s in ("HIP", "HPC", "HIPP", "HIPPOCAMPUS"):
        return "HIP"
    if s in ("CTX", "CORTEX", "OUTSIDE", "EXTRAHIPPOCAMPAL", "EXTRAHIPPOCAMPUS"):
        return "CTX"
    try:
        v = float(s)
        if v == 1:
            return "HIP"
        if v == 0:
            return "CTX"
    except Exception:
        pass
    return "UNK"


def _resolve_npz_path(root: Path, npz_arg: str | None) -> Path:
    if npz_arg:
        p = Path(npz_arg).expanduser()
        if p.is_absolute():
            return p.resolve()
        cand1 = (root / p).resolve()
        if cand1.exists():
            return cand1
        cand2 = (Path.cwd() / p).resolve()
        return cand2
    return (root / DEFAULT_NPZ).resolve()

def _load_events_from_npz(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (times_s, labels) where:
      - times_s: 1D float seconds, sorted
      - labels:  1D str of same length ('HIP'/'CTX'/'UNK')
    """
    d = np.load(npz_path, allow_pickle=True)
    keys = set(d.files)

    # ============================================================
    # (NEW) Pattern 0: event_times + is_hippocampus  ←あなたのNPZ用
    # ============================================================
    if "event_times" in keys and "is_hippocampus" in keys:
        times = np.asarray(d["event_times"]).astype(float).ravel()
        iship = np.asarray(d["is_hippocampus"]).ravel()

        n = min(times.size, iship.size)
        times = times[:n]
        iship = iship[:n]

        # iship が 0/1, True/False, "HIP"/"CTX" などでも吸収
        labels = []
        for v in iship:
            if _canon_region(v) in ("HIP", "CTX"):
                labels.append(_canon_region(v))
            else:
                labels.append("HIP" if bool(v) else "CTX")
        labels = np.asarray(labels, dtype=object)

        m = np.isfinite(times)
        times = times[m]
        labels = labels[m]

        order = np.argsort(times)
        return times[order], labels[order]

    # ============================================================
    # (NEW) Pattern 0b: event_times + event_network（保険）
    #   event_network が "Hippocampus" 系なら HIP、それ以外は CTX に寄せる
    # ============================================================
    if "event_times" in keys and "event_network" in keys:
        times = np.asarray(d["event_times"]).astype(float).ravel()
        net = np.asarray(d["event_network"]).ravel()
        n = min(times.size, net.size)
        times = times[:n]
        net = net[:n]

        labels = []
        for x in net:
            s = str(x).strip().upper()
            if "HIP" in s or "HPC" in s or "HIPP" in s:
                labels.append("HIP")
            else:
                # ここは「海馬以外は皮質」とする（あなたの Fig3a の目的に合う）
                labels.append("CTX")
        labels = np.asarray(labels, dtype=object)

        m = np.isfinite(times)
        times = times[m]
        labels = labels[m]

        order = np.argsort(times)
        return times[order], labels[order]

    # ---- Pattern 1: separate arrays (most reliable) ----
    hip_keys = ["hip_event_times", "hippocampus_event_times", "hip_times", "hip_event_centers"]
    ctx_keys = ["ctx_event_times", "cortex_event_times", "outside_event_times", "ctx_times", "ctx_event_centers"]

    hip = None
    for k in hip_keys:
        if k in keys:
            hip = np.asarray(d[k]).astype(float).ravel()
            break

    ctx = None
    for k in ctx_keys:
        if k in keys:
            ctx = np.asarray(d[k]).astype(float).ravel()
            break

    if hip is not None or ctx is not None:
        t_list = []
        lab_list = []
        if hip is not None and hip.size:
            t_list.append(hip)
            lab_list.append(np.array(["HIP"] * hip.size, dtype=object))
        if ctx is not None and ctx.size:
            t_list.append(ctx)
            lab_list.append(np.array(["CTX"] * ctx.size, dtype=object))

        if not t_list:
            return np.array([], float), np.array([], dtype=object)

        times = np.concatenate(t_list)
        labels = np.concatenate(lab_list)

        m = np.isfinite(times)
        times = times[m]
        labels = labels[m]

        order = np.argsort(times)
        return times[order], labels[order]

    # ---- Pattern 2: event_times + event_region ----
    if "event_times" in keys:
        times = np.asarray(d["event_times"]).astype(float).ravel()
        if "event_region" in keys:
            reg = np.asarray(d["event_region"]).ravel()
            labels = np.array([_canon_region(x) for x in reg], dtype=object)
            n = min(times.size, labels.size)
            times = times[:n]
            labels = labels[:n]
        else:
            labels = np.array(["UNK"] * times.size, dtype=object)

        m = np.isfinite(times)
        times = times[m]
        labels = labels[m]

        order = np.argsort(times)
        return times[order], labels[order]

    # ---- Pattern 3: candidate + is_hip ----
    if "candidate" in keys:
        cand = np.asarray(d["candidate"])
        cand = np.squeeze(cand)
        if cand.size == 0:
            return np.array([], float), np.array([], dtype=object)
        if cand.ndim == 1:
            cand = cand.reshape(1, -1)
        if cand.shape[1] < 3:
            return np.array([], float), np.array([], dtype=object)

        t0 = cand[:, 1].astype(float)
        t1 = cand[:, 2].astype(float)
        times = 0.5 * (t0 + t1)

        if "is_hip" in keys:
            is_hip = np.asarray(d["is_hip"]).ravel()
            n = min(times.size, is_hip.size)
            times = times[:n]
            is_hip = is_hip[:n]
            labels = np.array(["HIP" if bool(v) else "CTX" for v in is_hip], dtype=object)
        else:
            labels = np.array(["UNK"] * times.size, dtype=object)

        m = np.isfinite(times)
        times = times[m]
        labels = labels[m]

        order = np.argsort(times)
        return times[order], labels[order]

    raise ValueError(f"NPZ keys not recognized. Got keys={sorted(list(keys))}")

def sliding_rate(times_s: np.ndarray, T: float, win_s: float, step_s: float) -> tuple[np.ndarray, np.ndarray]:
    if T <= 0:
        return np.array([]), np.array([])
    starts = np.arange(0.0, max(0.0, T - win_s) + 1e-9, step_s)
    centers = starts + win_s / 2.0
    rate = np.zeros_like(centers, dtype=float)
    if times_s.size == 0:
        return centers, rate

    ts = np.clip(times_s, 0.0, T)
    for i, st in enumerate(starts):
        en = st + win_s
        n = np.searchsorted(ts, en, side="left") - np.searchsorted(ts, st, side="left")
        rate[i] = n / win_s
    return centers, rate


def extract_high_rate_epochs(t_centers: np.ndarray, rate: np.ndarray, win_s: float, thr: float, min_dur_s: float) -> list[tuple[float, float]]:
    if t_centers.size == 0 or (not np.isfinite(thr)):
        return []
    mask = np.isfinite(rate) & (rate >= thr)
    if not mask.any():
        return []

    idx = np.where(mask)[0]
    runs = []
    run_s = idx[0]
    prev = idx[0]
    for k in idx[1:]:
        if k == prev + 1:
            prev = k
            continue
        runs.append((run_s, prev))
        run_s = k
        prev = k
    runs.append((run_s, prev))

    out = []
    half = win_s / 2.0
    for a, b in runs:
        st = float(t_centers[a] - half)
        en = float(t_centers[b] + half)
        if (en - st) < min_dur_s:
            continue
        out.append((st, en))
    return out


def _shade_epochs(ax, epochs: list[tuple[float, float]]):
    for st, en in epochs:
        ax.axvspan(st, en, alpha=0.18, color="tab:orange", linewidth=0)


def _event_raster(ax, times_s: np.ndarray, labels: np.ndarray, T: float, *, alpha=0.75, lw=0.9):
    ts = np.asarray(times_s, float)
    lab = np.asarray(labels, dtype=object)

    m = np.isfinite(ts)
    ts = ts[m]
    lab = lab[m]

    inwin = (ts >= 0.0) & (ts <= T)
    ts = ts[inwin]
    lab = lab[inwin]

    if ts.size == 0:
        ax.set_xlim(0, T); ax.set_ylim(0, 1); ax.set_yticks([])
        ax.text(0.5, 0.5, "No events", transform=ax.transAxes, ha="center", va="center")
        return False

    lab_norm = np.array([_canon_region(x) for x in lab], dtype=object)
    has_any_label = np.any(lab_norm != "UNK")

    for tag, color in [("CTX", COLOR_CTX), ("HIP", COLOR_HIP), ("UNK", "black")]:
        idx = np.where(lab_norm == tag)[0]
        if idx.size == 0:
            continue
        ax.vlines(ts[idx], 0.0, 1.0, color=color, alpha=alpha, linewidth=lw, label=tag)

    ax.set_xlim(0, T); ax.set_ylim(0, 1); ax.set_yticks([])
    return bool(has_any_label)


def _load_labels_from_event_peaks_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)

    time_cands = ["time", "t", "time_s", "event_time", "peak_time", "peak_t"]
    tcol = next((c for c in time_cands if c in df.columns), None)
    if tcol is None:
        raise ValueError(f"{csv_path} lacks time column. columns={list(df.columns)}")

    reg_cands = ["region", "label", "source", "roi", "area", "is_hip", "is_hippocampus"]
    rcol = next((c for c in reg_cands if c in df.columns), None)
    if rcol is None:
        raise ValueError(f"{csv_path} lacks region/label column. columns={list(df.columns)}")

    times = pd.to_numeric(df[tcol], errors="coerce").to_numpy(float)
    raw = df[rcol].to_numpy()
    labels = np.array([_canon_region(x) for x in raw], dtype=object)

    m = np.isfinite(times) & (labels != "UNK")
    times = times[m]
    labels = labels[m]

    order = np.argsort(times)
    return times[order], labels[order]


def _assign_labels_by_nearest_time(npz_times: np.ndarray, csv_times: np.ndarray, csv_labels: np.ndarray, *, tol_s: float = 0.010) -> np.ndarray:
    npz_times = np.asarray(npz_times, float)
    out = np.array(["UNK"] * npz_times.size, dtype=object)
    if npz_times.size == 0 or csv_times.size == 0:
        return out

    csv_times = np.asarray(csv_times, float)
    csv_labels = np.asarray(csv_labels, dtype=object)

    idx = np.searchsorted(csv_times, npz_times, side="left")
    idx0 = np.clip(idx - 1, 0, csv_times.size - 1)
    idx1 = np.clip(idx,     0, csv_times.size - 1)

    d0 = np.abs(csv_times[idx0] - npz_times)
    d1 = np.abs(csv_times[idx1] - npz_times)
    pick = np.where(d1 < d0, idx1, idx0)

    ok = np.minimum(d0, d1) <= tol_s
    out[ok] = csv_labels[pick[ok]]
    return out


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None, help="NN_open_code root (default: auto)")
    ap.add_argument("--npz", type=str, default=None, help="NPZ path (absolute or relative)")
    ap.add_argument("--out", type=str, default=None, help=f"Output PDF (default: <root>/outputs/figures/{OUTNAME})")

    # ★ここを parse_args より前に置く
    ap.add_argument("--event-peaks", type=str, default=None,
                    help="Optional CSV (e.g., stc/event_peaks_120Hz.csv) to label events as HIP/CTX")
    ap.add_argument("--tol", type=float, default=0.010,
                    help="Time matching tolerance (sec) when assigning labels from CSV (default=0.010)")

    args = ap.parse_args()

    # root
    if args.root is not None:
        root = Path(args.root).expanduser().resolve()
    else:
        try:
            root = Path(__file__).resolve().parents[1]
        except NameError:
            root = Path.cwd().resolve()

    # npz
    npz_path = _resolve_npz_path(root, args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    # out
    out_pdf = Path(args.out).expanduser().resolve() if args.out else (root / "outputs" / "figures" / OUTNAME)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    # load npz
    d = np.load(npz_path, allow_pickle=True)
    keys = set(d.files)

    # events
    ev_t, ev_lab = _load_events_from_npz(npz_path)

    # label fill (only if all UNK)
    if ev_lab.size and np.all(np.asarray(ev_lab, dtype=object) == "UNK"):
        csv_path = None
        if args.event_peaks:
            csv_path = Path(args.event_peaks).expanduser().resolve()
        else:
            cand = npz_path.parent / "stc" / "event_peaks_120Hz.csv"
            if cand.exists():
                csv_path = cand

        if csv_path is not None and csv_path.exists():
            try:
                csv_times, csv_labels = _load_labels_from_event_peaks_csv(csv_path)
                ev_lab = _assign_labels_by_nearest_time(ev_t, csv_times, csv_labels, tol_s=float(args.tol))
                print(f"[INFO] labels filled from: {csv_path} (tol={args.tol}s)")
                print(f"[INFO] labeled events: HIP={(ev_lab=='HIP').sum()} CTX={(ev_lab=='CTX').sum()} UNK={(ev_lab=='UNK').sum()}")
            except Exception as e:
                print(f"[WARN] failed to use event_peaks CSV: {csv_path} -> {e}")

    # rate
    if ("t" in keys) and ("rate" in keys):
        tc = np.asarray(d["t"]).astype(float).ravel()
        rr = np.asarray(d["rate"]).astype(float).ravel()
    else:
        tc, rr = sliding_rate(ev_t, RECORDING_LEN_S, WIN_S, STEP_S)

    # threshold
    if "threshold" in keys:
        thr_arr = np.asarray(d["threshold"]).astype(float).ravel()
        thr = float(thr_arr[0]) if thr_arr.size and np.isfinite(thr_arr[0]) else np.nan
    else:
        thr = float(np.nanquantile(rr, RATE_Q)) if rr.size else np.nan

    # epochs
    epochs = []
    if "epochs" in keys:
        ep = np.asarray(d["epochs"])
        if ep.ndim == 2 and ep.shape[1] == 2 and ep.size:
            epochs = [(float(a), float(b)) for a, b in ep]
    if not epochs:
        epochs = extract_high_rate_epochs(tc, rr, WIN_S, thr, MIN_EPOCH_DUR_S)

    # plot
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(10.0, 6.2), dpi=150)
    gs = GridSpec(nrows=2, ncols=1, height_ratios=[2, 1], hspace=0.15)

    ax_rate = fig.add_subplot(gs[0, 0])
    ax_rate.plot(tc, rr, lw=1.6, color="black")
    if np.isfinite(thr):
        ax_rate.axhline(thr, ls="--", lw=1.0, color="gray")
    _shade_epochs(ax_rate, epochs)
    ax_rate.set_xlim(0, RECORDING_LEN_S)
    ax_rate.set_ylabel("Rate (events/s)")
    ax_rate.set_title("Sliding-window ripple rate with high-rate epochs (example)")
    ax_rate.grid(alpha=0.25)
    ax_rate.tick_params(labelbottom=False)

    ax_ras = fig.add_subplot(gs[1, 0], sharex=ax_rate)
    _shade_epochs(ax_ras, epochs)
    has_legend = _event_raster(ax_ras, ev_t, ev_lab, RECORDING_LEN_S)
    ax_ras.set_xlabel("Time (s)")
    ax_ras.set_ylabel("Ripple events")

    if has_legend:
        handles, labels = ax_ras.get_legend_handles_labels()
        uniq = {}
        for h, l in zip(handles, labels):
            if l not in uniq:
                uniq[l] = h
        order = ["CTX", "HIP", "UNK"]
        handles2 = [uniq[o] for o in order if o in uniq]
        labels2  = [o for o in order if o in uniq]
        ax_ras.legend(handles2, labels2, frameon=False, loc="upper right", fontsize=9)

    fig.suptitle("(a) Example of temporally clustered SWRs", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("[OK] Saved:", out_pdf)
    print("[INFO] NPZ:", npz_path)
    print("[INFO] keys:", sorted(list(keys)))
    
    print("[INFO] label counts:", {"HIP": int((ev_lab=="HIP").sum()),
                                  "CTX": int((ev_lab=="CTX").sum()),
                                  "UNK": int((ev_lab=="UNK").sum())})
if __name__ == "__main__":
    main()