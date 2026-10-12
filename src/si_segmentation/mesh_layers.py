"""
mesh_layers.py
------------------------------------------------------------------
Smooth POLYGON surface meshes (not voxelized), one per tissue layer,
from the cleaned label volume — in real micrometres.

Pipeline per layer:
  binary mask -> marching cubes (spacing in um) -> Taubin smoothing
  (removes voxel stair-stepping) -> quadric decimation -> OBJ.

Layers: 5=Serosa (outer), 4=Muscularis, 3=Submucosa, 2=Mucosa, 1=Lumen.

>>> SCALE (calibration) <<<
  XY_UM        = 14.7   registered 3.68 um/px * stride-4 downsample
                        (registered px size pending confirm of downsample)
  Z_SPACING_UM = 5.0    section thickness = 5 um. THIS ASSUMES SERIAL
                        sections. If sampled every Nth, multiply by N.
Only the tube's length-stretch depends on Z_SPACING_UM; cross-section is set
by XY_UM. Both are single-number edits here once the values are confirmed.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import trimesh
from trimesh.smoothing import filter_taubin
from skimage.measure import marching_cubes

OUT_DIR      = os.path.dirname(os.path.abspath(__file__))
IN_NPZ       = os.path.join(OUT_DIR, "volume_villi.npz")   # 2026-07-30: was
                             # volume_clean.npz -- the PRE-microstructure volume, so
                             # every OBJ this produced had NO villi, crypts or stem
                             # cells. Viewing those was viewing a smooth-walled tube,
                             # not the model.
MESH_DIR     = os.path.join(OUT_DIR, "meshes")
XY_UM        = 8.7616        # 4.3808 um/px (pyramid level 4, exact) * stride 2.
                             # 2026-07-30: was 7.35 (the inferred ~3.68 um/px that
                             # was never a real pyramid level). HANDOFF has flagged
                             # this file as needing the update since the recalibration;
                             # it had not been applied, so all prior OBJs were scaled
                             # wrong by 8.7616/7.35 = 1.19x.
Z_SPACING_UM = 5.0           # <-- see header (serial-section assumption)
TAUBIN_ITERS = 18            # smoothing passes
TARGET_FACES = 200_000       # per-layer face budget after decimation
# Colours match make_presentation_figs.py / the figure scheme, so Blender renders and
# the slide figures agree. 6 and 7 are the microstructure -- 7 is the dose target.
LAYERS = {5: ("serosa",     "#be96d2"),
          4: ("muscularis", "#3ccdb4"),
          3: ("submucosa",  "#96503c"),
          2: ("mucosa",     "#1e2882"),
          1: ("lumen",      "#e8e8f4"),
          6: ("crypt",      "#f0961e"),
          7: ("stem_cells", "#e11414")}


def decimate(mesh, target):
    if len(mesh.faces) <= target:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=target)
    except Exception as e:
        print(f"      (decimation skipped: {type(e).__name__})")
        return mesh


def main():
    os.makedirs(MESH_DIR, exist_ok=True)
    vol = np.load(IN_NPZ)["volume"]
    spacing = (Z_SPACING_UM, XY_UM, XY_UM)
    print(f"volume {vol.shape}  spacing(z,y,x)={spacing} um")

    preview = None
    for lab, (name, _) in LAYERS.items():
        mask = np.pad(vol == lab, 1)
        if mask.sum() < 100:
            print(f"  {name}: too few voxels, skipped")
            continue
        verts, faces, _, _ = marching_cubes(mask.astype(np.uint8), level=0.5,
                                            spacing=spacing, step_size=1)
        mesh = trimesh.Trimesh(vertices=verts[:, ::-1], faces=faces,
                               process=True)     # verts -> (x,y,z) um
        n0 = len(mesh.faces)
        mesh = decimate(mesh, TARGET_FACES)            # shrink FIRST (fast/stable)
        filter_taubin(mesh, iterations=TAUBIN_ITERS)   # de-voxelize on clean mesh
        # drop small disconnected floaters (NO smoothing after this)
        parts = mesh.split(only_watertight=False)
        if len(parts) > 1:
            fmax = max(len(p.faces) for p in parts)
            kept = [p for p in parts if len(p.faces) >= 0.03 * fmax]
            mesh = trimesh.util.concatenate(kept)
            print(f"    ({len(parts)} components -> kept {len(kept)})")
        out = os.path.join(MESH_DIR, f"{name}.obj")
        mesh.export(out)
        print(f"  {name}: {n0:,} -> {len(mesh.faces):,} faces (smoothed) -> {out}")
        if lab == 4:
            preview = mesh

    # ---- quick 3D preview of the smoothed muscularis ----
    if preview is not None:
        v, f = preview.vertices, preview.faces
        fig = plt.figure(figsize=(10, 12))
        for k, (elev, azim) in enumerate([(12, -70), (12, 20)]):
            ax = fig.add_subplot(1, 2, k + 1, projection="3d")
            ax.plot_trisurf(v[:, 0], v[:, 1], f, v[:, 2], color="#40e0c0",
                            edgecolor="none", linewidth=0, antialiased=True)
            ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
            ax.set_zlabel("z (um)"); ax.view_init(elev=elev, azim=azim)
        fig.suptitle("Muscularis — smoothed polygon mesh "
                     f"(Z-spacing {Z_SPACING_UM} um, serial assumption)")
        p = os.path.join(OUT_DIR, "preview_3d.png")
        fig.savefig(p, dpi=120); plt.close(fig)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
