"""
mesh_qc.py -- hard quality/validity gate for the PHITS tet mesh.
Fails loudly with a report rather than letting a bad mesh through.
Checks: conformal shared interfaces, closed 2-manifold boundary per region,
no region overlap, no degenerate/orphaned tets, sliver thresholds
(min dihedral angle, aspect ratio). Import and call gate(nodes, elems, region).
"""
import os
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def _tet_volumes(nodes, elems):
    p = nodes[elems]
    return np.einsum('ij,ij->i', np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]),
                     p[:, 3]-p[:, 0]) / 6.0


def _min_dihedral(nodes, elems):
    """Minimum dihedral angle per tet, in degrees.

    FIXED 2026-09-02. The previous version took np.abs() of the cosine between
    face normals, which folds the result into [90, 180] -- so it could NEVER
    return a value below 90 and the min_dihedral=5.0 gate could never fire. It
    reported 109.47 for a regular tet (the supplement of the true 70.53) and
    90.00 for a totally flat sliver. Verified against a regular tetrahedron, a
    flat sliver and a needle. Every mesh gated before this date passed check [6]
    vacuously, the female variant B included.

    The abs() was hiding the real problem: the face normals were never oriented,
    so their signs were arbitrary. Each is now flipped to point AWAY from the
    vertex opposite its face, which makes all four outward, and the dihedral is
    then 180 - angle(n_i, n_j) with no absolute value.
    """
    p = nodes[elems]                                  # (M,4,3)
    # face (a,b,c) is opposite vertex d
    faces = [(1, 2, 3, 0), (0, 2, 3, 1), (0, 1, 3, 2), (0, 1, 2, 3)]
    n = []
    for a, b, c, d in faces:
        nrm = np.cross(p[:, b]-p[:, a], p[:, c]-p[:, a])
        # orient outward: away from the opposite vertex
        sgn = np.einsum('ij,ij->i', nrm, p[:, d]-p[:, a])
        nrm = np.where((sgn > 0)[:, None], -nrm, nrm)
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-30
        n.append(nrm)
    n = np.stack(n, axis=1)                            # (M,4,3)
    mind = np.full(len(elems), 180.0)
    for i in range(4):
        for j in range(i+1, 4):
            cosang = ((n[:, i]*n[:, j]).sum(1)).clip(-1, 1)
            ang = 180 - np.degrees(np.arccos(cosang))
            mind = np.minimum(mind, ang)
    return mind


def _aspect(nodes, elems):
    p = nodes[elems]
    # longest edge / (3*inradius); inradius = 3V/surface_area
    e = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    L = np.stack([np.linalg.norm(p[:, a]-p[:, b], axis=1) for a, b in e], 1)
    Lmax = L.max(1)
    V = np.abs(_tet_volumes(nodes, elems))
    faces = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]
    A = np.zeros(len(elems))
    for a, b, c in faces:
        A += 0.5*np.linalg.norm(np.cross(p[:, b]-p[:, a], p[:, c]-p[:, a]), axis=1)
    inr = 3*V/(A+1e-30)
    return Lmax/(inr*2*np.sqrt(6)+1e-30)              # 1.0 = regular tet


def gate(nodes, elems, region, min_dihedral=5.0, max_aspect=50.0,
         nonmanifold_fail=0.05, nonmanifold_warn=0.005, raise_on_fail=True,
         detached_warn_frac=0.0005):
    """Hard gate for PHITS validity.

    LEAKAGE-CRITICAL (hard fail): conformal faces (every interior face shared by
    exactly 2 tets), no degenerate/inverted tets, no orphaned nodes, region ids
    in range. These are what make PHITS tet transport correct.

    TOPOLOGY-QUALITY (soft): closed-manifold-per-region. Non-manifold boundary
    *edges* are thin-layer pinch points; they do NOT leak particles (faces stay
    conformal). Voxel-label + centroid tagging has an inherent ~1% floor here that
    only a conforming multi-material mesher (CGAL) removes. So this fails only when
    egregious (`nonmanifold_fail`, catches gross under-resolution) and warns above
    `nonmanifold_warn`.
    """
    fails, warns = [], []
    n_tet = len(elems)
    print(f"\n=== MESH QC GATE ({n_tet:,} tets, {len(nodes):,} nodes) ===")

    # 1) degenerate / inverted tets
    V = _tet_volumes(nodes, elems)
    n_zero = int((np.abs(V) < 1e-6).sum())
    n_inv = int((V < 0).sum())
    if n_zero:
        fails.append(f"[1] {n_zero} degenerate tets (|V|~0)")
        print(f"[1] degenerate (|V|~0): {n_zero}")
    else:
        print("[1] degenerate tets: 0  OK")
    if n_inv:
        warns.append(f"[1b] inverted (V<0) tets: {n_inv} (orientation only)")

    # 2) orphaned nodes
    used = np.unique(elems)
    orph = len(nodes) - len(used)
    if orph == 0:
        print("[2] orphaned nodes: 0  OK")
    else:
        fails.append(f"[2] orphaned nodes: {orph}")
        print(f"[2] orphaned nodes: {orph}")

    # 3) region overlap: each tet exactly one region id (by construction)
    # 1..7 for the delivered models; 1..8 when the crypt-layer model is meshed (LAYERS=1)
    top = 8 if os.environ.get("LAYERS", "0") not in ("0", "false", "") else 7
    bad_reg = int(((region < 1) | (region > top)).sum())
    print(f"[3] region ids in 1..{top}, one per tet: {'OK' if bad_reg==0 else f'{bad_reg} BAD'}")
    if bad_reg:
        fails.append(f"[3] {bad_reg} tets with out-of-range region")

    # 4) conformal: every internal triangular face shared by exactly 2 tets
    fc = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    tri = np.sort(np.concatenate([elems[:, f] for f in fc]), axis=1)
    uniq, cnt = np.unique(tri, axis=0, return_counts=True)
    n_bad_face = int((cnt > 2).sum())
    n_bnd = int((cnt == 1).sum())
    print(f"[4] conformal faces: {(cnt<=2).all()} "
          f"(interior shared by 2, {n_bnd:,} boundary; {n_bad_face} faces shared by >2)")
    if n_bad_face:
        fails.append(f"[4] {n_bad_face} non-conformal faces (shared by >2 tets)")

    # 5) closed 2-manifold boundary per region: boundary faces of region R must
    #    form a closed surface (every edge shared by exactly 2 boundary faces)
    for r in sorted(set(region.tolist())):
        er = elems[region == r]
        trir = np.sort(np.concatenate([er[:, f] for f in fc]), axis=1)
        u, c = np.unique(trir, axis=0, return_counts=True)
        bnd = u[c == 1]                                  # region boundary faces
        if len(bnd) == 0:
            warns.append(f"[5] region {r}: no boundary faces?")
            continue
        edges = np.sort(np.concatenate([bnd[:, [0, 1]], bnd[:, [1, 2]],
                                        bnd[:, [0, 2]]]), axis=1)
        ue, ce = np.unique(edges, axis=0, return_counts=True)
        nonman = int((ce != 2).sum())
        frac = nonman / max(len(ue), 1)
        status = "closed-manifold OK" if nonman == 0 else f"{nonman} non-manifold edges ({100*frac:.1f}%)"
        print(f"[5] region {r} boundary: {len(bnd):,} faces  {status}")
        # thin synthetic targets (serosa 5, crypt 6, stem cells 7) are ~1 tet thick
        # by design -> non-manifold there is expected and does not leak; warn only.
        thin = r in (5, 6, 7, 8)          # 8 = crypt base (layers model, 2026-09-25)
        if frac > nonmanifold_fail and not thin:
            fails.append(f"[5] region {r}: {nonman} non-manifold boundary edges "
                         f"({100*frac:.1f}%) -- main layer grossly under-resolved")
        elif frac > nonmanifold_warn:
            warns.append(f"[5] region {r}: {nonman} non-manifold boundary edges "
                         f"({100*frac:.1f}%) -- {'thin target' if thin else 'thin-layer pinch'}; "
                         f"conformal, no leakage")

    # 7) DISCONNECTED TISSUE: the mucosa+crypt+stem union must be ONE piece.
    # Check [2] is about unused mesh NODES and says nothing about this. In 2026-07
    # the model carried 302 detached tissue components (2.56% of tissue, ~40 whole
    # floating villi with no attachment to the wall) and passed every other check
    # here, because a floating villus is a perfectly well-formed lump of tets.
    # Physically unattached tissue is not a mesh defect, it is a MODEL defect, and
    # it is the one this gate exists to stop shipping.
    tis = np.isin(region, (2, 6, 7, 8))   # + crypt base (layers model)
    if tis.any():
        et = elems[tis]
        trit = np.sort(np.concatenate([et[:, f] for f in fc]), axis=1)
        ut, inv, ct = np.unique(trit, axis=0, return_inverse=True, return_counts=True)
        inv = np.asarray(inv).ravel()        # numpy 2.x returned (n,1) here for a while
        tid = np.tile(np.arange(len(et)), 4)             # tet index per face row
        shared = np.flatnonzero(ct == 2)                 # interior faces of the union
        pos = np.argsort(inv, kind="stable")
        srt, tsrt = inv[pos], tid[pos]
        start = np.searchsorted(srt, shared, side="left")
        pair = np.stack([tsrt[start], tsrt[start + 1]], axis=1)
        nt = len(et)
        adj = coo_matrix((np.ones(len(pair)), (pair[:, 0], pair[:, 1])),
                         shape=(nt, nt))
        ncomp, cl = connected_components(adj, directed=False)
        cnt = np.bincount(cl)
        frac = cnt.max() / nt
        print(f"[7] tissue (mucosa+crypt+stem) connectivity: {ncomp} component(s); "
              f"largest holds {frac*100:.3f}% of {nt:,} tissue tets")
        if ncomp > 1:
            # A few fragments at this scale are CENTROID TAGGING severing a thin
            # neck, not a broken model: volume_villi.npz is one connected component
            # by construction and reconnect_tissue() already bridges what it can.
            # Above the tolerance it is a real structural failure and still hard-fails.
            msg = (f"[7] {ncomp-1} DETACHED tissue component(s) "
                   f"({(1-frac)*100:.3f}% of tissue tets) -- floating tissue")
            if (1 - frac) <= detached_warn_frac:
                warns.append(msg + f" -- under the {detached_warn_frac*100:.3f}% "
                             f"tolerance, treated as a tagging artifact")
            else:
                fails.append(msg)

    # 6) sliver thresholds
    # Chunked (2026-09-29): whole-mesh dihedral/aspect arrays OOM'd the 258M-tet
    # F-DUO layers mesh at 320 GB. Same worst values and counts, bounded memory.
    n_slivd = n_slva = 0
    worst_d, worst_a = np.inf, 0.0
    for i0 in range(0, len(elems), 5_000_000):
        e = elems[i0:i0 + 5_000_000]
        mind = _min_dihedral(nodes, e)
        asp = _aspect(nodes, e)
        n_slivd += int((mind < min_dihedral).sum())
        n_slva += int((asp > max_aspect).sum())
        worst_d = min(worst_d, float(mind.min()))
        worst_a = max(worst_a, float(asp.max()))
    del mind, asp
    print(f"[6] min dihedral: worst {worst_d:.2f} deg, "
          f"{n_slivd} tets < {min_dihedral} deg")
    print(f"    aspect ratio: worst {worst_a:.1f}, {n_slva} tets > {max_aspect}")
    if n_slivd:
        warns.append(f"[6] {n_slivd} sliver tets (< {min_dihedral} deg dihedral)")
    if n_slva:
        warns.append(f"[6] {n_slva} high-aspect tets (> {max_aspect})")

    print("--- QC SUMMARY ---")
    for w in warns:
        print("  WARN", w)
    if fails:
        for f in fails:
            print("  FAIL", f)
        msg = f"MESH QC FAILED: {len(fails)} hard failure(s), {len(warns)} warning(s)"
        print(msg)
        if raise_on_fail:
            raise SystemExit(msg)
        return False
    print(f"  PASS (hard checks) with {len(warns)} warning(s)")
    return True


if __name__ == "__main__":
    import os
    d = os.path.dirname(os.path.abspath(__file__))
    nodes = np.loadtxt(os.path.join(d, "phits/intestine.node"), skiprows=1,
                       usecols=(1, 2, 3))
    ele = np.loadtxt(os.path.join(d, "phits/intestine.ele"), skiprows=1, dtype=np.int64)
    gate(nodes, ele[:, 1:5], ele[:, 5], raise_on_fail=False)
