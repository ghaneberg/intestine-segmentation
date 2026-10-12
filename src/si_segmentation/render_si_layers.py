"""Renders of one crypt-layers SI model (8 regions), for <MODEL>_SI_MODEL_FOR_BANGHO/renders/.

The earlier whole-segment renders (RENDER_*.png, MESH_SLICE_*.png from render_villi.py /
render_mesh_slice.py) show the old 7-label model. These redo them for the layered model:

  RENDER_overview.png   full transverse slice of volume_layers.npz, with zoom boxes
  RENDER_detail.png     six crypt close-ups: crypt base, +4 band and TA in section
  RENDER_ortho.png      transverse + two longitudinal planes
  RENDER_3d.png         3D block: tissue surface from the lumen, and the three crypt
                        layers alone inside the wall
  MESH_SLICE_zNNNN.png  the DELIVERED TET MESH itself, tets within +-HALF of mid-z,
                        rasterised (smallest tet wins a pixel, so thin layers stay visible)

Palette = fjej_epithelium_trial/make_figures.py (the committee-approved trial figures).

Memory: the voxel volume is 3.5-5.2 GB (uint8) and is only ever sliced; the tet mesh is
read in chunks, keeping only tets near the slice. Peak is roughly volume + 2 GB.

    python render_si_layers.py <tag> [--only mesh|voxel]     tag in: fjej mjej file mile fduo
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, Rectangle
from scipy.ndimage import grey_dilation

ROOT = r"D:/Minipig Small Intestine"
MODELS = {
    "fjej": ("FEMALE_JEJUNUM", "Female jejunum", "hpg_fjej_layers_mesh"),
    "mjej": ("MALE_JEJUNUM", "Male jejunum", "hpg_layers_batch/mjej"),
    "file": ("FEMALE_ILEUM", "Female ileum", "hpg_layers_batch/file"),
    "mile": ("MALE_ILEUM", "Male ileum", "hpg_layers_batch/mile"),
    "fduo": ("FEMALE_DUODENUM", "Female duodenum", "hpg_layers_batch/fduo"),
}
NAMES = {0: "background", 1: "Lumen", 2: "Mucosa", 3: "Submucosa", 4: "Muscularis",
         5: "Serosa", 6: "TA", 7: "+4 stem cells", 8: "Crypt base (CBC + Paneth)"}
COLORS = {0: "#ffffff", 1: "#1b2a4a", 2: "#e8743b", 3: "#ffd166", 4: "#2a9d8f",
          5: "#6a4c93", 6: "#0a8fc4", 7: "#f72585", 8: "#7209b7"}
CMAP = ListedColormap([COLORS[i] for i in range(9)])
NORM = BoundaryNorm(np.arange(-0.5, 9.5, 1.0), CMAP.N)
ALL = [1, 2, 3, 4, 5, 6, 7, 8]
HALF = 6.0            # um either side of the mesh slice plane
GP = 2.0              # um per pixel in the mesh slice raster
CHUNK = 8_000_000


def legend(keys):
    return [Patch(facecolor=COLORS[k], edgecolor="#444444", lw=0.4,
                  label=f"{k}  {NAMES[k]}") for k in keys]


def scalebar(ax, um, um_per_px, color="black"):
    n = um / um_per_px
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    x = x0 + 0.04 * abs(x1 - x0)
    y = y0 - 0.05 * abs(y1 - y0) if y0 > y1 else y0 + 0.05 * abs(y1 - y0)
    ax.plot([x, x + n], [y, y], color=color, lw=3, solid_capstyle="butt")
    ax.text(x + n / 2, y, f"{um:g} µm", color=color, ha="center", va="bottom", fontsize=8)


def pick_windows(sl, half, n=6):
    """Windows centred on crypt-base voxels, spread out across the slice."""
    for lab in (8, 7, 6, 2):
        ys, xs = np.nonzero(sl == lab)
        if len(ys):
            break
    else:
        return []
    idx = np.random.default_rng(0).permutation(len(ys))[: n * 400]
    picked = []
    for cy, cx in zip(ys[idx], xs[idx]):
        if all((cy - py) ** 2 + (cx - px) ** 2 > (3 * half) ** 2 for py, px in picked):
            picked.append((cy, cx))
        if len(picked) == n:
            break
    return [(int(np.clip(cy - half, 0, sl.shape[0] - 2 * half)),
             int(np.clip(cx - half, 0, sl.shape[1] - 2 * half)), 2 * half)
            for cy, cx in picked]


def voxel_figs(vol, sp, label, out):
    dz, dy, dx = sp
    z = vol.shape[0] // 2
    sl = vol[z]
    wins = pick_windows(sl, half=int(round(120 / dx)))

    fig, ax = plt.subplots(figsize=(16, 16 * sl.shape[0] / sl.shape[1] + 1))
    ax.imshow(sl, cmap=CMAP, norm=NORM, interpolation="nearest")
    for i, (y0, x0, s) in enumerate(wins):
        ax.add_patch(Rectangle((x0, y0), s, s, fill=False, ec="#00e5ff", lw=1.6))
        ax.text(x0, y0 - 10, str(i + 1), color="#00a5b8", fontsize=12, weight="bold")
    ax.set_title(f"{label}, crypt-layer model — transverse slice at z = {z*dz:.0f} µm "
                 f"(voxel model, {dx:.2f} µm pixels)", fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    scalebar(ax, 1000, dx)
    ax.legend(handles=legend(ALL), loc="lower right", fontsize=9, framealpha=0.92, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{out}/RENDER_overview.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("  RENDER_overview.png")

    fig, axes = plt.subplots(2, 3, figsize=(15, 11.2))
    for i, (ax, (y0, x0, s)) in enumerate(zip(axes.ravel(), wins)):
        ax.imshow(sl[y0:y0 + s, x0:x0 + s], cmap=CMAP, norm=NORM, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"window {i+1}  ({s*dy:.0f} × {s*dx:.0f} µm)", fontsize=10)
        scalebar(ax, 50, dx)
    for ax in axes.ravel()[len(wins):]:
        ax.axis("off")
    fig.suptitle(f"{label} — crypts in section: crypt base (purple), +4 stem band (pink), "
                 f"TA (blue), z = {z*dz:.0f} µm", fontsize=13)
    fig.legend(handles=legend([1, 2, 6, 7, 8, 3]), loc="lower center", ncol=6,
               fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96), h_pad=2.5)
    fig.savefig(f"{out}/RENDER_detail.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("  RENDER_detail.png")

    nz, ny, nx = vol.shape
    cy, cx = ny // 2, nx // 2
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.4, 1])
    a = fig.add_subplot(gs[0, :])
    a.imshow(sl, cmap=CMAP, norm=NORM, interpolation="nearest")
    a.axhline(cy, color="#00e5ff", lw=0.8); a.axvline(cx, color="#00e5ff", lw=0.8)
    a.set_title(f"transverse, z = {z*dz:.0f} µm (cyan lines = the two planes below)",
                fontsize=11)
    a.set_xticks([]); a.set_yticks([]); scalebar(a, 1000, dx)
    b = fig.add_subplot(gs[1, 0])
    b.imshow(vol[:, cy, :], cmap=CMAP, norm=NORM, interpolation="nearest", aspect=dz / dx)
    b.set_title(f"longitudinal, y = {cy*dy:.0f} µm (horizontal cyan line)", fontsize=10)
    b.set_xticks([]); b.set_yticks([])
    c = fig.add_subplot(gs[1, 1])
    c.imshow(vol[:, :, cx], cmap=CMAP, norm=NORM, interpolation="nearest", aspect=dz / dy)
    c.set_title(f"longitudinal, x = {cx*dx:.0f} µm (vertical cyan line)", fontsize=10)
    c.set_xticks([]); c.set_yticks([])
    fig.suptitle(f"{label} — orthogonal planes (the segment is {nz*dz:.0f} µm long in z)",
                 fontsize=13)
    fig.legend(handles=legend(ALL), loc="lower center", ncol=8, fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(f"{out}/RENDER_ortho.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("  RENDER_ortho.png")

    try:
        import pyvista as pv
        from skimage.measure import marching_cubes
    except ImportError as e:
        print(f"  RENDER_3d.png skipped: {e}")
        return
    pv.OFF_SCREEN = True
    from scipy.ndimage import distance_transform_edt
    # Block centred halfway between the first crypt window and the nearest lumen, so it
    # holds both the crypt layers and the mucosal surface; viewed as in the trial FIG3:
    # along the wall normal (submucosa -> lumen), tilted to look down the slab (z).
    s = int(round(600 / dx))
    cy, cx = ((wins[0][0] + wins[0][2] // 2, wins[0][1] + wins[0][2] // 2) if wins
              else (ny // 2, nx // 2))
    _, (iy, ix) = distance_transform_edt(vol[z] != 1, return_indices=True)
    ly, lx = iy[cy, cx], ix[cy, cx]
    y0 = int(np.clip((cy + ly) // 2 - s // 2, 0, ny - s))
    x0 = int(np.clip((cx + lx) // 2 - s // 2, 0, nx - s))
    zs = int(round(300 / dz))
    z0 = int(np.clip(z - zs // 2, 0, nz - zs))
    blk = np.ascontiguousarray(vol[z0:z0 + zs, y0:y0 + s, x0:x0 + s])

    def surf(mask):
        if mask.sum() < 50:
            return None
        v, f, _, _ = marching_cubes(np.pad(mask, 1).astype(np.float32), 0.5, spacing=sp)
        return pv.PolyData(v, np.hstack([np.full((len(f), 1), 3), f]).ravel())

    mid = blk[zs // 2]
    lum, sub = np.nonzero(mid == 1), np.nonzero(mid == 3)
    if len(lum[0]) and len(sub[0]):
        nrm = np.array([0.0, (lum[0].mean() - sub[0].mean()) * sp[1],
                        (lum[1].mean() - sub[1].mean()) * sp[2]])
    else:
        nrm = np.array([0.0, float(ly - cy), float(lx - cx)])
    nrm /= np.linalg.norm(nrm) or 1.0
    view = 0.55 * nrm + 0.85 * np.array([1.0, 0, 0]); view /= np.linalg.norm(view)
    ctr = (np.array(blk.shape) + 2) * np.array(sp) / 2

    def aim():
        p.camera.focal_point = ctr
        p.camera.position = ctr + view * 3000
        p.camera.up = tuple(nrm)
        p.camera.zoom(1.1)

    p = pv.Plotter(shape=(1, 2), window_size=(2000, 1000), off_screen=True)
    p.set_background("white")
    p.subplot(0, 0)
    t = surf(blk >= 2)
    if t is not None:
        p.add_mesh(t, color=COLORS[2], smooth_shading=True)
    p.add_text("tissue surface seen from the lumen (600 × 600 × 300 µm block)",
               font_size=11, color="black")
    aim()
    p.subplot(0, 1)
    for lab, op in ((3, 1.0), (6, 0.55), (7, 1.0), (8, 1.0)):
        m = surf(blk == lab)
        if m is not None:
            p.add_mesh(m, color=COLORS[lab], smooth_shading=True, opacity=op)
    p.add_text("same block, crypt layers only: base (purple), +4 (pink), TA (blue), "
               "on the submucosa (yellow)", font_size=11, color="black")
    aim()
    p.screenshot(f"{out}/RENDER_3d.png"); p.close()
    print("  RENDER_3d.png")


def mesh_slice(mesh_dir, label, out):
    nodes = pd.read_csv(f"{mesh_dir}/intestine.node", sep=r"\s+", skiprows=1, header=None,
                        usecols=[1, 2, 3], dtype=np.float64, engine="c").to_numpy()
    zmid = 0.5 * (nodes[:, 2].min() + nodes[:, 2].max())
    sc, sr, sv = [], [], []
    for ch in pd.read_csv(f"{mesh_dir}/intestine.ele", sep=r"\s+", skiprows=1, header=None,
                          usecols=[1, 2, 3, 4, 5], dtype=np.int64, engine="c",
                          chunksize=CHUNK):
        a = ch.to_numpy()
        p = nodes[a[:, :4]]
        c = p.mean(1)
        k = np.abs(c[:, 2] - zmid) < HALF
        p = p[k]
        sv.append(np.abs(np.einsum("ij,ij->i", p[:, 1] - p[:, 0],
                                   np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))) / 6)
        sc.append(c[k, :2]); sr.append(a[k, 4].astype(np.int8))
    sc, sr, sv = np.concatenate(sc), np.concatenate(sr), np.concatenate(sv)
    lo = nodes[:, :2].min(0)
    gw = int(np.ceil((nodes[:, 0].max() - lo[0]) / GP)) + 2
    gh = int(np.ceil((nodes[:, 1].max() - lo[1]) / GP)) + 2
    del nodes
    gi = np.clip(((sc[:, 0] - lo[0]) / GP).astype(np.int64), 0, gw - 1)
    gj = np.clip(((sc[:, 1] - lo[1]) / GP).astype(np.int64), 0, gh - 1)
    order = np.argsort(-sv)
    img = np.zeros(gw * gh, np.int8)
    img[(gi * gh + gj)[order]] = sr[order]
    img = img.reshape(gw, gh)
    for _ in range(4):
        img = np.where(img == 0, grey_dilation(img, size=(3, 3)), img)
    fig, ax = plt.subplots(figsize=(16, 16 * gh / gw + 1))
    ax.imshow(img.T, origin="lower", cmap=CMAP, norm=NORM, interpolation="nearest",
              extent=[lo[0] / 1000, (lo[0] + gw * GP) / 1000, lo[1] / 1000,
                      (lo[1] + gh * GP) / 1000])
    ax.set_aspect("equal"); ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title(f"{label} — the delivered tet mesh: {len(sr):,} tets with centroid within "
                 f"±{HALF:g} µm of z = {zmid:.0f} µm, by region", fontsize=12)
    ax.legend(handles=legend(ALL), loc="lower right", fontsize=9, framealpha=0.92, ncol=2)
    fig.tight_layout()
    name = f"MESH_SLICE_z{int(round(zmid)):04d}.png"
    fig.savefig(f"{out}/{name}", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  {name}  ({len(sr):,} tets)")


def main():
    tag = sys.argv[1]
    only = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--only" else None
    up, label, vdir = MODELS[tag]
    pkg = f"{ROOT}/{up}_SI_MODEL_FOR_BANGHO"
    out = f"{pkg}/renders"
    os.makedirs(out, exist_ok=True)
    print(f"[{tag}] -> {out}")
    if only != "mesh":
        z = np.load(f"{ROOT}/{vdir}/volume_layers.npz")
        sp = tuple(float(v) for v in z["spacing"])
        vol = z["volume"]
        print(f"  volume {vol.shape}, spacing {sp} µm")
        voxel_figs(vol, sp, label, out)
        del vol, z
    if only != "voxel":
        mesh_slice(f"{pkg}/full_array", label, out)


if __name__ == "__main__":
    main()
