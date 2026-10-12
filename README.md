# intestine-segmentation

Histology → segmentation → 3D layer model of the mini-pig small intestine for
microscale dosimetry. This repository turns registered H&E whole-slide stacks into
per-layer label volumes, polygon meshes, and a PHITS-compatible tetrahedral mesh,
intended for Monte Carlo radiation-transport dosimetry of radiopharmaceutical
therapies.

The package is a clean, reusable implementation of the segmentation/meshing
pipeline used in the author's doctoral research at the University of Florida
(Medical Physics, ALRADS / BOL group). **No research histology data is shipped
with this repository** — it is code + tooling only, with synthetic demo data so
the pipeline can be exercised end-to-end without any UF-acquired images.

## Pipeline at a glance

| Stage | Module | Output |
|---|---|---|
| 1. Stack QC | `stack_qc.py` | slice ordering/continuity report |
| 2. Volume assembly | `build_volume.py` | 3D label volume from 2D masks |
| 3. Finalize | `finalize_volume.py` | lumen hole-fill, Z-smoothing, synthetic serosa |
| 4. Villi | `add_villi.py` | villus layer on the mucosa |
| 5. Layer meshes | `mesh_layers.py` | per-layer surface meshes (OBJ) |
| 6. Tet mesh | `phits_tetmesh_gmsh.py` | conforming PHITS tetrahedral mesh (node/ele) |
| 7. Mesh QC | `mesh_qc.py` | hard validity gate for the tet mesh |
| 8. Render | `render_si_layers.py` / `render_villi.py` | off-screen QC renders |

The QuPath side lives under `src/si_segmentation/qupath/` (Groovy classifiers for
layer annotation and mask export).

## Install

```
pip install -e .
```

Dependencies: `numpy`, `scipy`, `scikit-image`, `meshio`, `tetgen`, `pyvista`,
`gmsh` (all standard on Windows/pip).

## Demo data

`demo-data/` is intentionally minimal and synthetic so the pipeline runs without
any real histology. Drop your own registered label masks into the input dir to
run the real workflow. The PHITS tet mesh is written for the
[Tetrahedral] element block format.

## Citation / publication strategy

This software is the tooling behind the author's dissertation microscale-dosimetry
aims. It may be submitted to the Journal of Open Source Software (JOSS) as a
peer-reviewed software paper; the repo is structured to satisfy JOSS's
requirements (open source license, DOI-ready, clear README). See `docs/` for the
methods outline.

## License
MIT — see `LICENSE`.
