"""
stack_qc.py
------------------------------------------------------------------
Z-continuity QC for the mini-pig small-intestine label stack.

Reads the per-slice label masks exported from QuPath
(export_labeled_masks.groovy), orders them by slice index, detects the
missing slices, and measures how jumpy the stack is slice-to-slice using
the MUSCULARIS (label 4) as the geometric anchor.

Why this step first: registration so far is RIGID only. This tells us
whether the stack is smooth enough to mesh directly, or whether we need
deformable slice-to-slice refinement (ANTs SyN / VALIS) before 3D.

Streams one mask at a time (does NOT hold the whole volume in RAM) so it
runs comfortably even with 184 full-res slices.

Outputs (next to this script by default):
  - qc_zcontinuity.png   multi-panel diagnostic plot
  - zqc_report.txt       missing slices + flagged jumpy slices
"""

import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import tifffile
    def _imread(p): return tifffile.imread(p)
except ImportError:
    from skimage.io import imread as _imread

# ===== CONFIG ======================================================
MASK_DIR   = r"C:\small_intestine\qupath_si\labeled_masks"
OUT_DIR    = os.path.dirname(os.path.abspath(__file__))
LABELS = {1: "Lumen", 2: "Mucosa", 3: "Submucosa", 4: "Muscularis", 5: "Serosa"}
ANCHOR = 4                 # muscularis = the registration anchor
JUMP_SIGMA = 3.0           # flag slices whose step exceeds mean + N*std
# ===================================================================


def slice_index(fname):
    m = re.search(r"(\d+)", os.path.basename(fname))
    return int(m.group(1)) if m else None


def main():
    files = sorted(glob.glob(os.path.join(MASK_DIR, "*_labels.tif*")),
                   key=slice_index)
    if not files:
        raise SystemExit(f"No *_labels.tif found in {MASK_DIR!r} — run the "
                         f"export Groovy script first.")

    idx = [slice_index(f) for f in files]
    present = set(idx)
    full_range = range(min(idx), max(idx) + 1)
    missing = [z for z in full_range if z not in present]
    print(f"Found {len(files)} masks, indices {min(idx)}..{max(idx)}. "
          f"Missing: {missing}")

    # Per-slice metrics, streamed
    zs, cx, cy, anchor_area = [], [], [], []
    layer_area = {lab: [] for lab in LABELS}

    empty_slices = []
    for f, z in zip(files, idx):
        # 0-byte / unreadable exports (tapered ends) -> treat as all-background
        try:
            if os.path.getsize(f) == 0:
                raise ValueError("0-byte file")
            m = _imread(f)
        except Exception as e:
            empty_slices.append(z)
            for lab in LABELS:
                layer_area[lab].append(0)
            anchor_area.append(0)
            cx.append(np.nan); cy.append(np.nan); zs.append(z)
            print(f"  z={z:>3}  EMPTY/unreadable ({e})")
            continue
        if m.ndim == 3:                     # (H,W,C) -> take first channel
            m = m[..., 0]
        for lab in LABELS:
            layer_area[lab].append(int(np.count_nonzero(m == lab)))
        amask = (m == ANCHOR)
        a = int(amask.sum())
        anchor_area.append(a)
        if a > 0:
            ys, xs = np.nonzero(amask)
            cx.append(xs.mean()); cy.append(ys.mean())
        else:
            cx.append(np.nan); cy.append(np.nan)
        zs.append(z)
        print(f"  z={z:>3}  muscularis px={a}")

    zs = np.array(zs)
    cx = np.array(cx); cy = np.array(cy)
    anchor_area = np.array(anchor_area, float)

    # Frame-to-frame jump = centroid displacement between consecutive
    # PRESENT slices, normalized per unit z-gap (so a gap doesn't false-flag).
    dz = np.diff(zs)
    disp = np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2) / dz
    darea = np.abs(np.diff(anchor_area)) / np.maximum(anchor_area[:-1], 1) / dz

    def flags(metric):
        good = metric[np.isfinite(metric)]
        if good.size == 0:
            return set()
        thr = good.mean() + JUMP_SIGMA * good.std()
        return {int(zs[i + 1]) for i, v in enumerate(metric)
                if np.isfinite(v) and v > thr}

    jump_slices = sorted(flags(disp) | flags(darea))

    # ---- Report ----
    rep = os.path.join(OUT_DIR, "zqc_report.txt")
    with open(rep, "w") as fh:
        fh.write("Z-continuity QC — mini-pig small intestine\n")
        fh.write("=" * 44 + "\n\n")
        fh.write(f"masks: {len(files)}  indices {min(idx)}..{max(idx)}\n")
        fh.write(f"missing slices (interpolate at volume step): {missing}\n")
        fh.write(f"empty/unreadable masks (tapered ends): {empty_slices}\n\n")
        fh.write(f"anchor = {LABELS[ANCHOR]} (label {ANCHOR})\n")
        fh.write(f"centroid displacement  mean={np.nanmean(disp):.2f} "
                 f"max={np.nanmax(disp):.2f} px/slice\n")
        fh.write(f"slices with empty muscularis: "
                 f"{[int(z) for z, a in zip(zs, anchor_area) if a == 0]}\n\n")
        fh.write(f"JUMPY slices (>{JUMP_SIGMA}sigma in centroid or area, "
                 f"candidates for manual review / deformable refine):\n")
        fh.write(f"  {jump_slices}\n")
    print(f"\nwrote {rep}")
    print(f"jumpy slices: {jump_slices}")

    # ---- Plot ----
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0, 0].plot(zs, cx, ".-"); ax[0, 0].set_title("Muscularis centroid X vs Z")
    ax[0, 1].plot(zs, cy, ".-"); ax[0, 1].set_title("Muscularis centroid Y vs Z")
    ax[0, 0].set_xlabel("slice"); ax[0, 0].set_ylabel("px")
    ax[0, 1].set_xlabel("slice"); ax[0, 1].set_ylabel("px")
    for lab, name in LABELS.items():
        ax[1, 0].plot(zs, layer_area[lab], ".-", label=name)
    ax[1, 0].set_title("Layer area vs Z"); ax[1, 0].set_xlabel("slice")
    ax[1, 0].set_ylabel("px^2"); ax[1, 0].legend(fontsize=8)
    ax[1, 1].plot(zs[1:], disp, ".-", label="centroid disp (px/slice)")
    for z in jump_slices:
        ax[1, 1].axvline(z, color="r", alpha=0.3)
    ax[1, 1].set_title(f"Slice-to-slice jump (red = >{JUMP_SIGMA}sigma)")
    ax[1, 1].set_xlabel("slice"); ax[1, 1].set_ylabel("px/slice")
    for z in missing:
        for a in ax.ravel():
            a.axvspan(z - 0.5, z + 0.5, color="gray", alpha=0.15)
    fig.suptitle("Mini-pig small intestine — Z-continuity QC "
                 "(gray = missing slice, red = jumpy)")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "qc_zcontinuity.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
