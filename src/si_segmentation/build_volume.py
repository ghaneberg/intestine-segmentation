"""
build_volume.py
------------------------------------------------------------------
Assemble the 2D label masks into a 3D label volume for reconstruction.

Steps:
  1. Load the usable core slices (default z = 2..182; ends are empty taper).
  2. Downsample XY (stride, label-preserving) so the volume is ~200 MB.
  3. Fill missing / bad slices (missing indices + fusion slice 43) by
     copying the nearest present slice.
  4. Re-center every slice on the MUSCULARIS (label 4) centroid to remove
     the rigid-registration slice-to-slice wobble.
  5. Emit orthogonal XZ / YZ cross-sections BEFORE and AFTER re-centering
     (so we can see the wobble and whether re-centering fixes it), plus a
     montage of re-centered slices.
  6. Save the re-centered volume (compressed .npz) for meshing.

Nothing here needs the calibrated um/pixel or Z-spacing yet — the volume is
built in voxel units; real spacing is applied later at the meshing step.
"""

import os
import re
import sys
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import shift as ndi_shift
from scipy.ndimage import label as _ndi_label

KEPT_FRAC = []   # per-slice fraction of tissue kept by keep_target_curl()

# 2026-08-28: dispatch on extension. tifffile cannot read PNG, and the male masks
# are PNG -- it raised on every slice and load_slice's bare `except` swallowed it,
# so the run aborted with "No usable slices" while the files were found fine.
try:
    import tifffile as _tifffile
except ImportError:
    _tifffile = None
from skimage.io import imread as _skimread

def _imread(p):
    if _tifffile is not None and p.lower().endswith((".tif", ".tiff")):
        return _tifffile.imread(p)
    return _skimread(p)

# ===== CONFIG ======================================================
# Shared script: the MODEL DIRECTORY is sys.argv[1], else the cwd. It receives
# volume_recentered.npz and may hold params.json ["build_volume"] overriding
# anything below. The literals here are the MALE JEJUNUM baseline and their
# comments are its decision record -- read before overriding.
MASK_DIR   = "D:/Minipig Small Intestine/nnunet_raw/pred_m_si2_jejunum_ordered"
MASK_GLOB  = "*.png"          # male predictions are MSJ_XXXX.png, not *_labels.tif
# MALE 2026-08-28: nnU-Net predictions (Dataset101, folds 1/2/4) after the 3D
# topological ordering fix + speck cleanup. 200 slices, 2226x4520, 4.3808 um/px
# -- the SAME pixel size the female pipeline already used, so STRIDE/XY_UM are
# unchanged. Labels 0-4; serosa (5) is synthesised downstream as before.
MODEL_DIR  = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.getcwd())
OUT_DIR    = MODEL_DIR
# New stack: slices numbered 1..200; every slice carries real tissue (no empty
# taper -- verified in the QC montage). The 9 absent indices (never-cut plus
# excluded 128) are filled from the nearest present slice, which keeps each
# plane at its true z = index x 5 um. Do NOT compress the gaps out.
ZMIN, ZMAX = 1, 200
STRIDE     = 2                 # XY downsample factor (label-preserving)
                              # -> XY voxel = 4.3808 * 2 = 8.7616 um
BAD_SLICES = set()             # slide 43 is now speckle-cleaned (clean_masks.py),
                              # not discarded. 128 is simply absent -> auto-filled.
ANCHOR     = 4                 # muscularis = re-centering anchor
# ---- per-model overrides -----------------------------------------------------
_TUNABLES = ("MASK_DIR", "MASK_GLOB", "ZMIN", "ZMAX", "STRIDE", "BAD_SLICES", "ANCHOR")
DEFAULTS = {k: globals()[k] for k in _TUNABLES}


def _load_params(section="build_volume"):
    """Read params.json[section] from the model directory; unknown keys are fatal."""
    p = dict(DEFAULTS)
    f = os.path.join(MODEL_DIR, "params.json")
    if not os.path.exists(f):
        print("no params.json in %s -- using defaults (male jejunum baseline)" % MODEL_DIR)
        return p
    with open(f) as fh:
        doc = json.load(fh)
    user = doc.get(section, {})
    user = {k: v for k, v in user.items() if not k.startswith("_")}   # _keys are notes
    unknown = sorted(set(user) - set(DEFAULTS))
    if unknown:
        raise SystemExit("params.json[%s]: unknown key(s) %s" % (section, unknown))
    p.update(user)
    p["BAD_SLICES"] = set(p["BAD_SLICES"])       # json has no set type
    print("params.json: %s  [%s]" % (f, section))
    if "_model" in doc:
        print("  model: %s" % doc["_model"])
    diffs = [k for k in _TUNABLES if p[k] != DEFAULTS[k]]
    for k in diffs:
        print("  override  %-12s %s" % (k, p[k]))
    if not diffs:
        print("  no overrides -- identical to the male jejunum baseline")
    return p


globals().update(_load_params())

LABELS     = {0: "bg", 1: "Lumen", 2: "Mucosa", 3: "Submucosa",
              4: "Muscularis", 5: "Serosa"}
# 6-class display colormap (bg transparent-ish black)
CMAP = ListedColormap([(0, 0, 0),
                       (0.55, 0.55, 0.8),   # lumen  periwinkle
                       (0.10, 0.10, 0.45),  # mucosa navy
                       (0.55, 0.27, 0.19),  # submucosa brown
                       (0.25, 0.88, 0.75),  # muscularis cyan
                       (0.7, 0.5, 0.8)])    # serosa
# ===================================================================


def slice_index(fname):
    m = re.search(r"(\d+)", os.path.basename(fname))
    return int(m.group(1)) if m else None


def load_slice(path):
    """Return a downsampled label slice, or None if empty/unreadable."""
    try:
        if os.path.getsize(path) == 0:
            return None
        m = _imread(path)
    except Exception as exc:
        # Do NOT swallow silently -- a reader that cannot open the format fails on
        # every slice and surfaces only as "No usable slices loaded".
        if not getattr(load_slice, "_warned", False):
            print(f"  READ FAILED {os.path.basename(path)}: {type(exc).__name__}: {exc}")
            load_slice._warned = True
        return None
    if m.ndim == 3:
        m = m[..., 0]
    m = m[::STRIDE, ::STRIDE].astype(np.uint8)
    return keep_target_curl(m)


def keep_target_curl(sl):
    """Keep only the largest connected tissue mass.

    MALE 2026-08-28: the rois_1over16 frames carry the target loop PLUS a fragment
    of a neighbouring one -- consistently 80-84% vs 14-18% of tissue on every slice.
    The female pipeline excluded this upstream with the QuPath LeftCurl ROI; the
    nnU-Net segments everything in frame, so it must be dropped here. The second
    mass is a tangential partial cut (muscularis falls to 0.1% by z=180) and could
    not be reconstructed as a tube regardless.

    Largest-component, not the female's leftmost rule -- the size separation here is
    unambiguous. KEPT_FRAC logs it so a slice where the split changes is visible
    rather than silent.
    """
    tis = sl > 0
    if not tis.any():
        return sl
    lab, n = _ndi_label(tis, structure=np.ones((3, 3), bool))
    if n <= 1:
        return sl
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    biggest = int(np.argmax(sizes))
    KEPT_FRAC.append(sizes[biggest] / tis.sum())
    out = sl.copy()
    out[lab != biggest] = 0
    return out


def centroid(sl):
    ys, xs = np.nonzero(sl == ANCHOR)
    if xs.size == 0:
        return None
    return (ys.mean(), xs.mean())


def main():
    files = {slice_index(f): f
             for f in glob.glob(os.path.join(MASK_DIR, MASK_GLOB))}

    zrange = list(range(ZMIN, ZMAX + 1))
    raw = {}                      # z -> downsampled slice (present & good)
    for z in zrange:
        if z in BAD_SLICES or z not in files:
            continue
        sl = load_slice(files[z])
        if sl is None or not np.any(sl == ANCHOR):
            continue
        raw[z] = sl
    present = sorted(raw)
    if not present:
        raise SystemExit("No usable slices loaded — check MASK_DIR.")
    # Slices can differ by a pixel or two -> pad all to a common canvas
    H = max(raw[z].shape[0] for z in present)
    W = max(raw[z].shape[1] for z in present)

    def fit(sl):
        out = np.zeros((H, W), np.uint8)
        h, w = min(sl.shape[0], H), min(sl.shape[1], W)
        out[:h, :w] = sl[:h, :w]
        return out
    raw = {z: fit(sl) for z, sl in raw.items()}
    if KEPT_FRAC:
        _k = np.array(KEPT_FRAC)
        print(f"  curl isolation: kept {100*_k.mean():.1f}% of tissue per slice "
              f"(min {100*_k.min():.1f}%, max {100*_k.max():.1f}%) on {len(_k)} slice(s)")
    print(f"loaded {len(present)} present slices, canvas {H}x{W} "
          f"(stride {STRIDE}); filling {len(zrange) - len(present)} gaps")

    # Fill missing/bad slices with the nearest present slice
    filled = {}
    for z in zrange:
        src = z if z in raw else min(present, key=lambda p: abs(p - z))
        filled[z] = (raw[src], src)

    # Build raw (un-centered) and re-centered volumes
    target = np.array([H / 2.0, W / 2.0])
    vol_raw = np.zeros((len(zrange), H, W), np.uint8)
    vol_ctr = np.zeros((len(zrange), H, W), np.uint8)
    for i, z in enumerate(zrange):
        sl, src = filled[z]
        vol_raw[i] = sl
        c = centroid(sl)
        if c is None:
            vol_ctr[i] = sl
        else:
            dy, dx = target - np.array(c)
            vol_ctr[i] = ndi_shift(sl, (dy, dx), order=0, mode="constant",
                                   cval=0, prefilter=False)

    # ---- Orthogonal cross-sections (mid planes) ----
    def ortho(vol, tag):
        midy, midx = H // 2, W // 2
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        ax[0].imshow(vol[:, midy, :], cmap=CMAP, vmin=0, vmax=5,
                     aspect="auto", interpolation="nearest")
        ax[0].set_title(f"XZ section (y={midy})  [{tag}]")
        ax[0].set_xlabel("x"); ax[0].set_ylabel("slice z")
        ax[1].imshow(vol[:, :, midx], cmap=CMAP, vmin=0, vmax=5,
                     aspect="auto", interpolation="nearest")
        ax[1].set_title(f"YZ section (x={midx})  [{tag}]")
        ax[1].set_xlabel("y"); ax[1].set_ylabel("slice z")
        fig.suptitle(f"Orthogonal section through label volume — {tag} "
                     "(jagged edge = slice-to-slice wobble)")
        fig.tight_layout()
        p = os.path.join(OUT_DIR, f"ortho_{tag}.png")
        fig.savefig(p, dpi=130); plt.close(fig)
        print(f"wrote {p}")

    ortho(vol_raw, "raw")
    ortho(vol_ctr, "recentered")

    # ---- Montage of re-centered slices (every ~15th) ----
    step = max(1, len(zrange) // 12)
    picks = list(range(0, len(zrange), step))[:12]
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    for a, i in zip(axes.ravel(), picks):
        a.imshow(vol_ctr[i], cmap=CMAP, vmin=0, vmax=5, interpolation="nearest")
        a.set_title(f"z={zrange[i]}", fontsize=9); a.axis("off")
    for a in axes.ravel()[len(picks):]:
        a.axis("off")
    fig.suptitle("Re-centered slices (montage)")
    fig.tight_layout()
    pm = os.path.join(OUT_DIR, "montage_recentered.png")
    fig.savefig(pm, dpi=110); plt.close(fig)
    print(f"wrote {pm}")

    # ---- Save the re-centered volume for meshing ----
    outv = os.path.join(OUT_DIR, "volume_recentered.npz")
    np.savez_compressed(outv, volume=vol_ctr, zrange=np.array(zrange),
                        stride=STRIDE)
    print(f"wrote {outv}  (shape {vol_ctr.shape}, "
          f"{os.path.getsize(outv)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
