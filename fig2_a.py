#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 06:17:54 2026

@author: takeiyuuichi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig.2a: Spatial distribution maps of voxel-normalized ripple events
(Hippocampus vs Cortex; HC vs SZ; 80–240 Hz)

Input (expected NIfTI files):
  NN_open_code/data/fig2a/
    Hippocampus_HC_80Hz.nii.gz
    Hippocampus_SZ_80Hz.nii.gz
    ...
    Cortex_HC_240Hz.nii.gz
    Cortex_SZ_240Hz.nii.gz

Output:
  NN_open_code/outputs/figures/Fig2a_spatial_distribution.pdf

Notes:
  - Each cell shows two views (sagittal + axial) similar to Fig2a.
  - Colorbar is shared per region (Hippocampus/Cortex) for visual consistency.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# nilearn is used for neuroimaging plots
from nilearn import plotting
from nilearn import image as nimg
import matplotlib as mpl
from nilearn.datasets import load_mni152_template


FREQS = [80, 120, 160, 200, 240]
REGIONS = ["Hippocampus", "Cortex"]
GROUPS = ["HC", "SZ"]  # row order within each region
THRESH_ABS = {"Hippocampus": 1.0, "Cortex": 0.6}

def expected_path(root: Path, region: str, group: str, freq: int) -> Path:
    return root / "data" / "fig2a" / f"{region}_{group}_{freq}Hz.nii.gz"

def _add_stat_map(ax, img, thr,bg_img, display_mode: str, vmax: float, title: str | None = None):
    """
    Plot into a given matplotlib Axes by forcing nilearn to use that axes.
    """
    disp = plotting.plot_stat_map(
        img,
        bg_img=bg_img,
        display_mode=display_mode,
        cut_coords=1,
        black_bg=True,
        cmap="hot",          # ★変更（または "turbo"）
        threshold= thr,       # ★低値も表示
        vmin=0,
        vmax=vmax,
        colorbar=False,
        annotate=False,
        axes=ax,
    )

def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # check input existence
    missing = []
    for region in REGIONS:
        for group in GROUPS:
            for f in FREQS:
                p = expected_path(root, region, group, f)
                if not p.exists():
                    missing.append(str(p))
    if missing:
        msg = "\n".join(missing[:20])
        raise FileNotFoundError(
            "Missing Fig2a input NIfTI files. Place them under NN_open_code/data/fig2a/.\n"
            "Example filenames: Hippocampus_HC_80Hz.nii.gz\n\n"
            f"First missing entries:\n{msg}\n"
            f"... ({len(missing)} missing total)"
        )

    # background template (MNI152)
    bg = load_mni152_template()

    # Decide shared vmax per region to match Fig2a style (separate scaling Hippocampus vs Cortex)
    # We scan max values across all freqs and groups within each region.
    region_vmax = {}
    for region in REGIONS:
        vals_all = []
        for group in GROUPS:
            for f in FREQS:
                img_path = expected_path(root, region, group, f)
                img = nimg.load_img(str(img_path))
                data = img.get_fdata()
                data = data[np.isfinite(data)]
                data = data[data > 0]  # 0は背景として除外（任意）
                if data.size:
                    vals_all.append(data)
        if vals_all:
            data_cat = np.concatenate(vals_all)
            region_vmax[region] = float(np.percentile(data_cat, 99.5))
        else:
            region_vmax[region] = None

    # Layout: 4 block-rows (Hippo HC/SZ, Cortex HC/SZ) × 5 freqs
    # Each cell contains 2 sub-axes (sagittal + axial).
    n_freq = len(FREQS)

    fig = plt.figure(figsize=(3.2 * n_freq, 10.5))
    outer = fig.add_gridspec(
        nrows=4, ncols=n_freq,
        height_ratios=[1, 1, 1, 1],
        wspace=0.05, hspace=0.22
    )

    # Row mapping
    row_specs = [
        ("Hippocampus", "HC"),
        ("Hippocampus", "SZ"),
        ("Cortex", "HC"),
        ("Cortex", "SZ"),
    ]

    # Titles (top)
    for j, f in enumerate(FREQS):
        ax_t = fig.add_subplot(outer[0, j])
        ax_t.axis("off")
        ax_t.set_title(f"{f} Hz", fontsize=12, pad=12)

    for i, (region, group) in enumerate(row_specs):
        for j, f in enumerate(FREQS):
            inner = outer[i, j].subgridspec(1, 2, wspace=0.02)
    
            img_path = expected_path(root, region, group, f)
            img = nimg.load_img(str(img_path))
            vmax = region_vmax[region]
    
            ax_x = fig.add_subplot(inner[0, 0])
            ax_z = fig.add_subplot(inner[0, 1])
            thr = THRESH_ABS[region]
            _add_stat_map(ax_x, img, thr,bg_img=bg, display_mode="x", vmax=vmax)
            _add_stat_map(ax_z, img, thr,bg_img=bg, display_mode="z", vmax=vmax)
            ...

            # left-most labels
            if j == 0:
                ax_x.text(
                    -0.18, 0.5,
                    f"{group}",
                    transform=ax_x.transAxes,
                    va="center", ha="right",
                    fontsize=12, fontweight="bold"
                )
                if group == "HC":
                    ax_x.text(
                        -0.18, 1.18,
                        f"(1) {region}" if region == "Hippocampus" else f"(2) {region}",
                        transform=ax_x.transAxes,
                        va="bottom", ha="right",
                        fontsize=12
                    )

            # clean axis frames
            for ax in (ax_x, ax_z):
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

    # Add colorbars per region (Hippocampus/Cortex)
    # We add two small colorbars using the last plotted display objects by re-plotting invisible.
    # Simple approach: create dummy scalar mappables.
    import matplotlib as mpl
    for region, y0 in [("Hippocampus", 0.535), ("Cortex", 0.085)]:
        vmax = region_vmax[region]
        if vmax is None:
            continue
        cax = fig.add_axes([0.92, y0, 0.015, 0.33])  # [left, bottom, width, height]
        norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap="hot")
        cb = fig.colorbar(sm, cax=cax)
        cb.ax.tick_params(labelsize=8)
        cb.set_label("Voxel-normalized ripple events (%)", fontsize=9)

    fig.suptitle("(a) Spatial distribution of ripple events in hippocampus and cerebral cortex", fontsize=14, y=0.995)
    out_pdf = out_dir / "Fig2a_spatial_distribution.pdf"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("[OK] Saved:", out_pdf)

if __name__ == "__main__":
    main()
