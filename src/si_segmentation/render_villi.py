"""Renders of the finished male jejunum villous model, for eyeball QC.

Reads volume_villi.npz (the add_villi.py output) and writes four figures:

  RENDER_overview.png  full transverse slice, all 7 labels, with zoom insets
  RENDER_detail.png    six high-magnification windows: villus shape, crypt
                       placement, and whether stem cells sit at the crypt BASE
  RENDER_ortho.png     the three orthogonal planes through one point
  RENDER_3d.png        surface render of a block, plus a cutaway showing crypts
                       and the stem compartment inside the wall

Colours extend the convention already in visualize_model.py so these figures sit
next to the female ones. Stem cells are deliberately the loudest colour: they are
the radiosensitive target the whole model exists to compute.

Usage:  python -u render_villi.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, Rectangle

import sys
import json
# Shared script: the MODEL DIRECTORY is sys.argv[1], else the cwd. It supplies
# volume_villi.npz and receives the four RENDER_*.png files.
HERE = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.getcwd())
IN_NPZ = os.path.join(HERE, "volume_villi.npz")

_pf = os.path.join(HERE, "params.json")
if os.path.exists(_pf):
    with open(_pf) as _fh:
        MODEL_LABEL = json.load(_fh).get("_model", "villous model")
else:
    MODEL_LABEL = "male jejunum villous model"   # original default, no params.json
SPACING = (5.0, 8.7616, 8.7616)          # (z, y, x) um
dz, dy, dx = SPACING

# visualize_model.py's palette for 1-4, extended for serosa/crypt/stem.
NAMES = {0: "background", 1: "lumen", 2: "mucosa", 3: "submucosa",
         4: "muscularis", 5: "serosa", 6: "crypt", 7: "stem cells"}
COLORS = {0: "#ffffff", 1: "#1b2a4a", 2: "#e8743b", 3: "#ffd166",
          4: "#2a9d8f", 5: "#6a4c93", 6: "#c1121f", 7: "#f72585"}
CMAP = ListedColormap([COLORS[i] for i in range(8)])
NORM = BoundaryNorm(np.arange(-0.5, 8.5, 1.0), CMAP.N)


def scalebar(ax, um, px_per_um, label=None, color="black"):
    n = um * px_per_um
    x0, y0 = ax.get_xlim()[0], ax.get_ylim()[0]
    xr = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
    yr = abs(ax.get_ylim()[1] - ax.get_ylim()[0])
    x = x0 + 0.05 * xr
    y = y0 - 0.07 * yr if ax.get_ylim()[0] > ax.get_ylim()[1] else y0 + 0.07 * yr
    ax.plot([x, x + n], [y, y], color=color, lw=3, solid_capstyle="butt")
    ax.text(x + n / 2, y, label or f"{um:.0f} um", color=color, ha="center",
            va="bottom", fontsize=8)


def legend_handles(labels):
    return [Patch(facecolor=COLORS[k], edgecolor="#444444", lw=0.4,
                  label=f"{k}  {NAMES[k]}") for k in labels]


def pick_windows(vol, z, n=6, half=140):
    """Windows centred on stem-cell clusters -- the structures worth inspecting."""
    sl = vol[z]
    stem = (sl == 7)
    if not stem.any():                     # fall back to crypts, then mucosa
        stem = (sl == 6)
    if not stem.any():
        stem = (sl == 2)
    ys, xs = np.nonzero(stem)
    if len(ys) == 0:
        return []
    idx = np.linspace(0, len(ys) - 1, min(n * 40, len(ys))).astype(int)
    cand = list(zip(ys[idx], xs[idx]))
    picked = []
    for cy, cx in cand:
        if all((cy - py) ** 2 + (cx - px) ** 2 > (2.5 * half) ** 2 for py, px in picked):
            picked.append((cy, cx))
        if len(picked) == n:
            break
    out = []
    for cy, cx in picked:
        y0 = int(np.clip(cy - half, 0, vol.shape[1] - 2 * half))
        x0 = int(np.clip(cx - half, 0, vol.shape[2] - 2 * half))
        out.append((y0, x0, 2 * half))
    return out


def fig_overview(vol, z, wins):
    sl = vol[z]
    fig, ax = plt.subplots(figsize=(16, 8.5))
    ax.imshow(sl, cmap=CMAP, norm=NORM, interpolation="nearest")
    for i, (y0, x0, s) in enumerate(wins[:3]):
        ax.add_patch(Rectangle((x0, y0), s, s, fill=False, ec="#00e5ff", lw=1.6))
        ax.text(x0, y0 - 12, f"{i+1}", color="#00e5ff", fontsize=13, weight="bold")
    ax.set_title(f"{MODEL_LABEL} — transverse slice z={z} "
                 f"({z*dz:.0f} um into the 1 mm block)", fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    scalebar(ax, 1000, 1.0 / dx, "1 mm")
    ax.legend(handles=legend_handles([1, 2, 3, 4, 5, 6, 7]), loc="lower right",
              fontsize=9, framealpha=0.92, ncol=2)
    fig.tight_layout()
    p = os.path.join(HERE, "RENDER_overview.png")
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    return p


def fig_detail(vol, z, wins):
    n = len(wins)
    rows, cols = (2, 3) if n > 3 else (1, max(n, 1))
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 5.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (y0, x0, s) in zip(axes, wins):
        crop = vol[z, y0:y0 + s, x0:x0 + s]
        ax.imshow(crop, cmap=CMAP, norm=NORM, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"y {y0}-{y0+s}, x {x0}-{x0+s}   "
                     f"({s*dy/1000:.2f} x {s*dx/1000:.2f} mm)", fontsize=9)
        scalebar(ax, 200, 1.0 / dx, "200 um")
    for ax in axes[len(wins):]:
        ax.axis("off")
    fig.suptitle("Villus shape, crypt placement, and stem cells at the crypt base "
                 f"(z={z})", fontsize=13)
    fig.legend(handles=legend_handles([1, 2, 6, 7, 3, 4]), loc="lower center",
               ncol=6, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    p = os.path.join(HERE, "RENDER_detail.png")
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    return p


def fig_ortho(vol):
    nz, ny, nx = vol.shape
    cz, cy, cx = nz // 2, ny // 2, nx // 2
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.4, 1])
    a = fig.add_subplot(gs[0, :])
    a.imshow(vol[cz], cmap=CMAP, norm=NORM, interpolation="nearest")
    a.set_title(f"transverse  z={cz}", fontsize=11); a.set_xticks([]); a.set_yticks([])
    scalebar(a, 1000, 1.0 / dx, "1 mm")
    b = fig.add_subplot(gs[1, 0])
    b.imshow(vol[:, cy, :], cmap=CMAP, norm=NORM, interpolation="nearest",
             aspect=dz / dx)
    b.set_title(f"longitudinal  y={cy}   (z stretched: dz=5 um vs dx=8.76 um)",
                fontsize=10)
    b.set_xticks([]); b.set_yticks([])
    c = fig.add_subplot(gs[1, 1])
    c.imshow(vol[:, :, cx], cmap=CMAP, norm=NORM, interpolation="nearest",
             aspect=dz / dy)
    c.set_title(f"longitudinal  x={cx}", fontsize=10)
    c.set_xticks([]); c.set_yticks([])
    fig.suptitle("Orthogonal planes — the block is only 1 mm long in z",
                 fontsize=13)
    fig.legend(handles=legend_handles([1, 2, 3, 4, 5, 6, 7]), loc="lower center",
               ncol=7, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    p = os.path.join(HERE, "RENDER_ortho.png")
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    return p


def fig_3d(vol, wins):
    """Surface of the villous field, and a cutaway showing the wall interior."""
    try:
        import pyvista as pv
        from skimage.measure import marching_cubes
    except ImportError as e:
        print(f"  3D skipped: {e}")
        return None
    pv.OFF_SCREEN = True
    y0, x0, s = wins[0] if wins else (vol.shape[1] // 2 - 140,
                                      vol.shape[2] // 2 - 140, 280)
    blk = vol[:, y0:y0 + s, x0:x0 + s]

    def surf(mask):
        if mask.sum() < 100:
            return None
        v, f, _, _ = marching_cubes(mask.astype(np.float32), level=0.5,
                                    spacing=SPACING)
        return pv.PolyData(v, np.hstack([np.full((len(f), 1), 3), f]).ravel())

    p = pv.Plotter(shape=(1, 2), window_size=(1900, 950), off_screen=True)
    p.subplot(0, 0)
    tis = surf(blk >= 2)
    if tis is not None:
        p.add_mesh(tis, color=COLORS[2], smooth_shading=True)
    p.add_text("villous surface, seen from the lumen", font_size=11)
    p.camera_position = "yz"
    p.subplot(0, 1)
    half = blk[:, :, : blk.shape[2] // 2]
    for lab in (2, 3, 4, 5, 6, 7):
        m = surf(half == lab)
        if m is not None:
            p.add_mesh(m, color=COLORS[lab], smooth_shading=True,
                       opacity=1.0 if lab in (6, 7) else 0.92)
    p.add_text("cutaway: crypts (red) and stem cells (pink) inside the wall",
               font_size=11)
    p.camera_position = "yz"
    out = os.path.join(HERE, "RENDER_3d.png")
    p.screenshot(out); p.close()
    return out


def main():
    if not os.path.exists(IN_NPZ):
        raise SystemExit(f"missing {IN_NPZ} -- scp it down from HiPerGator first")
    vol = np.load(IN_NPZ)["volume"]
    print(f"volume {vol.shape}  spacing {SPACING} um")
    vx = dz * dy * dx / 1e9
    for k in range(8):
        n = int((vol == k).sum())
        if n:
            print(f"  label {k} {NAMES[k]:11s}: {n:>12,}  ({n*vx:8.3f} mm^3)")

    z = vol.shape[0] // 2
    wins = pick_windows(vol, z)
    print(f"\ninspection windows at z={z}: {len(wins)}")
    for f in (fig_overview(vol, z, wins), fig_detail(vol, z, wins),
              fig_ortho(vol), fig_3d(vol, wins)):
        if f:
            print(f"  wrote {os.path.basename(f)}")


if __name__ == "__main__":
    main()
