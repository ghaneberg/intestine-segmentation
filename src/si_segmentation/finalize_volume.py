"""
finalize_volume.py
------------------------------------------------------------------
Clean the re-centered label volume so it is ready to mesh:

  1. LUMEN HOLE-FILL (per slice): the RF Lumen class is unreliable — lumen
     is not a tissue type, it is *white space*, which the classifier cannot
     tell apart from empty slide background. Its extent is therefore set by
     wherever the LeftCurl region happened to be drawn, not by anatomy.
     So the RF lumen label is DISCARDED outright and lumen is derived purely
     geometrically: the large space enclosed by the WALL layers (mucosa,
     submucosa, muscularis). Small intra-mucosa gaps are ignored (size
     threshold), and genuinely open C-shape slices stay open (no enclosed
     space).
     [2026-07-20] This replaced an earlier version that used `tissue = sl > 0`,
     i.e. it trusted the RF lumen label as tissue while also adding geometric
     lumen. After the LeftCurl regions were enlarged to fix the clipped-slide
     bug, RF lumen bled far outside the wall into slide background; `>0` then
     saw one solid blob, `binary_fill_holes` found no holes, and the geometric
     fill silently became a no-op. Deriving from the wall makes the result
     independent of how generous the LeftCurl region is.
  2. Z-SMOOTH: one-hot each class, gaussian-smooth along Z, argmax back to
     labels. Removes residual slice-to-slice wall jaggedness without
     blurring the open<->closed transitions (modest sigma).

In:  volume_recentered.npz
Out: volume_clean.npz  + ortho_clean.png (verify the flicker is gone)
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import (binary_fill_holes, gaussian_filter1d,
                           gaussian_filter, binary_dilation,
                           distance_transform_edt, label as cc_label)

# ---- shared script, per-model configuration ---------------------------------
# ONE copy of this script serves every segment. The MODEL DIRECTORY -- sys.argv[1],
# else the current working directory -- supplies volume_recentered.npz, receives
# volume_clean.npz, and may hold a params.json overriding any tunable below.
#
# Everything from here to CMAP is the MALE JEJUNUM baseline, kept verbatim. Those
# comments are the decision record from the female and male jejunum -- read them
# before overriding anything. The literals ARE the defaults: they are captured
# into DEFAULTS below, so a default and its rationale can never drift apart.
# -----------------------------------------------------------------------------
MODEL_DIR = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.getcwd())
OUT_DIR   = MODEL_DIR
IN_NPZ    = os.path.join(MODEL_DIR, "volume_recentered.npz")
Z_SIGMA   = 1.2                 # Z-smoothing strength (slices)
RELAYER_Z_SIGMA  = 12.0         # STRONG z-smoothing of the radial field (2026-07-23,
RELAYER_XY_SIGMA = 5.0          # user: smooth harder along z / slice-to-slice) ->
                                # smooth layer INTERFACES (mucosa|submucosa|
                                # muscularis). The coverage coat below repairs any
                                # monotonicity holes this creates.
ENV_Z_SIGMA  = 14.0            # STRONG z-smoothing of the TUBE ENVELOPE (outer
ENV_XY_SIGMA = 4.0             # muscularis edge + lumen) -> the tube varies smoothly
                                # slice-to-slice. Idealizes the contour into a clean
                                # smooth tube so the outer layer is one solid object,
                                # not stacked ribbons (user decision 2026-07-21).
# MALE 2026-08-28: the radial relayer rebuilds the wall as concentric shells keyed
# to t = d_lumen/(d_lumen + d_outside). That is a REPAIR STEP for shattered
# segmentation -- the female's QuPath RF submucosa was "~65% shattered per slice,
# 45% outside its largest 3D component" and genuinely needed it.
#
# It assumes the gut is a topological annulus. The male's is COLLAPSED: the mucosa
# is thrown into deep folds whose CORE is submucosa, which sits nowhere near a
# constant radial distance from the outer surface. Running it flattens every fold
# into rings (verified against the segmentation: see RELAYER_before_after.png) and
# is not fixable by tuning -- keeping all lumen components recovered lumen volume
# (2.8% -> 5.4%) but left the architecture wrong.
#
# The male's nnU-Net masks are clean and Z-coherent (CoV 0.036-0.085), so the repair
# is unnecessary here. The outer-coat guarantee is kept -- it lived inside
# relayer_wall() and is extracted below.
USE_RADIAL_RELAYER = False

# Which enclosed spaces the relayer treats as lumen. True = all of them (the male's
# 2026-08-28 rule, required for a COLLAPSED gut whose folds create many channels);
# False = only the largest per slice, the original rule.
# 2026-09-10: the 2026-08-28 note claimed keeping all was "a no-op on the female".
# It is not. Gating this script against her shipped volume showed all 200 of her
# slices carry small secondary enclosed spaces that the old rule turned into wall;
# keeping them moves 608,942 voxels from wall to lumen (0.065% of the volume,
# lumen +0.37%). Keeping all is the better rule -- those spaces are enclosed empty
# space, not tissue -- but it is a real change, so it is selectable and the female
# stays byte-reproducible with it off. Revisit if her model is ever rebuilt.
RELAYER_KEEP_ALL_LUMEN = True

# MALE 2026-08-28: the geometric lumen hole-fill replaces the classifier's lumen with
# "enclosed holes only". It exists because on the FEMALE "the Lumen class flickers" in
# the RF output. The male's nnU-Net lumen is clean and validated, and the geometric
# rule discards every channel that communicates with the section edge in-plane --
# measured 20-33% per slice, 36% of the lumen overall (11.75M -> 7.15M vox). Keep the
# segmented lumen instead.
USE_LUMEN_HOLEFILL = False

# Smooth the OUTER envelope even when the radial relayer is skipped. The relayer was
# doing double duty: concentric re-layering (wrong for a folded gut) AND smoothing the
# outer surface. The synthetic serosa is a dilation of that surface, so leaving it
# rough inflated the shell 3.5x (2.79M -> 9.65M vox, 4.8% of model vs the female's
# 1.08%). This smooths the envelope only; interior labels are untouched.
# MALE 2026-08-28: the nnU-Net inherits the female RF's submucosa leak and, in a
# FOLDED gut, expresses it as spurious submucosa fingers running deep into the mucosa
# along villus/fold cores (those cores are lamina propria = MUCOSA). They connect to
# the real band through the folds, so the "never touches muscularis" test misses them.
#
# Consequence: distance-from-submucosa is small everywhere (male d_sub median 44.5 um,
# p90 123 um), and add_villi.py defines its ENTIRE geometry by d_sub -- villus_zone =
# mucosa & (d_sub > CRYPT_ZONE) came out EMPTY, so zero villi were seeded.
#
# Calibration: distance-from-muscularis inside submucosa.
#     FEMALE  p50  82.6  p90 182.9  p99 289.3  max  469   <- a clean band
#     MALE    p50 126.5  p90 844.5  p99 1362   max 1693   <- band + fingers
# The female p90 (183) matches her measured ~168 um submucosa thickness. 300 um sits
# just above her p99, so it keeps everything a real band contains. Resulting male
# submucosa 9.52 mm^3 vs the female's 8.60 -- slightly larger, as a thicker male wall
# should be. Villus zone becomes 34.5% of mucosa (female 48.2%).
TRIM_SUBMUCOSA_FINGERS = True
SUBMUC_MAX_FROM_MUSC_UM = 300.0

# Only trim far submucosa THINNER than this (local thickness, um). 0 = no guard,
# i.e. trim everything past SUBMUC_MAX_FROM_MUSC_UM -- the male/female behaviour.
#
# ILEUM 2026-09-10: distance from muscularis alone cannot separate a spurious
# villus-core finger from a Peyer's patch, because both sit far from the muscle
# coat. Local thickness can: a villus core is tens of um across, a follicle is
# hundreds. Measured on the ileum, of the 11.98 mm^3 of submucosa lying >300 um
# from muscularis, 71% is thinner than 150 um (fingers) and ~20% is thicker than
# 200 um (follicles) -- cleanly bimodal.
#
# Those fingers are lamina propria, which is MUCOSA, mislabelled because loose
# connective tissue in a villus core looks like submucosa. Each one is a false
# floor for d_sub, which is why hand-measured mucosa of 698 um (n=10) read as
# 196 um in the segmentation.
#
# T=150 um is set by three independent checks: luminal-surface thickness p75
# reaches 690 um against the hand-measured median of 698; submucosa lands at
# 16.7% of wall against the jejunum's 18.5-19%; and every follicle (200-1466 um
# thick) survives.
SUBMUC_MAX_THICK_UM = 0.0

Z_UM_ASSUMED = 5.0            # z voxel size, for the submucosa trim's EDT
SMOOTH_OUTER_ENVELOPE = True
OUTER_ENV_SIGMA = (6.0, 2.5, 2.5)      # (z, y, x) voxels

OUTER_COAT_VOX = 8              # guarantee a THICK muscularis outer band (~70 um,
                                # ~2 tets) so no submucosa tet can poke through it
                                # in the centroid-tagged mesh (2026-07-23 fix).
# --- calibration ---
XY_UM     = 8.7616             # volume XY voxel size (um): 4.3808 um/px * stride 2
                              # (new extraction: 0.2738 um/px * 2^4 * stride 2)
SEROSA_UM = 30.0                # artificial serosa thickness (um), thin outer shell.
                                # NOTE (2026-08-20): this value does NOT reach the
                                # delivered mesh. phits_tetmesh_gmsh.py DISCARDS this
                                # serosa (vol[vol==5]=0) and rebuilds it at
                                # SEROSA_VOX=7 = ~61 um, because a 30 um shell is
                                # thinner than one tet at LC_FINE=35 and cannot mesh
                                # as a closed manifold. So the SHIPPED serosa is 61 um.
                                # The "30 vs 61 um inconsistency" ASSUMPTIONS.md flags
                                # is therefore stale upstream code, not a live conflict.
                                # Serosa is fully synthetic either way -- it is not
                                # segmented, and its thickness is set by mesh
                                # resolution, not by anatomy. Disclose it as such.
LUMEN     = 1
MUCOSA    = 2
SEROSA    = 5
WALL      = (2, 3, 4)           # mucosa, submucosa, muscularis -- the real tissue.
                                # Lumen (1) is white space, NOT tissue; serosa (5) is
                                # synthesized downstream. Anything geometric keys off
                                # WALL so it cannot be skewed by the RF lumen extent.
CMAP = ListedColormap([(0, 0, 0), (0.55, 0.55, 0.8), (0.10, 0.10, 0.45),
                       (0.55, 0.27, 0.19), (0.25, 0.88, 0.75), (0.7, 0.5, 0.8)])

# ---- per-model overrides -----------------------------------------------------
# The literals above are the defaults. params.json in the model directory may
# override any of these; anything it does not mention keeps the value above.
# Overrides are printed at startup so a run's provenance is in its own log.
_TUNABLES = ("Z_SIGMA", "RELAYER_Z_SIGMA", "RELAYER_XY_SIGMA",
             "RELAYER_KEEP_ALL_LUMEN",
             "ENV_Z_SIGMA", "ENV_XY_SIGMA",
             "USE_RADIAL_RELAYER", "USE_LUMEN_HOLEFILL",
             "TRIM_SUBMUCOSA_FINGERS", "SUBMUC_MAX_FROM_MUSC_UM",
             "SUBMUC_MAX_THICK_UM",
             "Z_UM_ASSUMED", "SMOOTH_OUTER_ENVELOPE", "OUTER_ENV_SIGMA",
             "OUTER_COAT_VOX", "XY_UM", "SEROSA_UM")
DEFAULTS = {k: globals()[k] for k in _TUNABLES}


def _load_params(section="finalize_volume"):
    """Read params.json[section] from the model directory.

    One params.json serves the whole pipeline, one section per stage, so a model
    directory declares its complete configuration in a single file. Unknown keys
    inside a section are a hard error -- a silently ignored typo in a parameter
    that changes geometry is exactly the failure this refactor exists to prevent.
    """
    p = dict(DEFAULTS)
    f = os.path.join(MODEL_DIR, "params.json")
    if not os.path.exists(f):
        print("no params.json in %s -- using defaults (male jejunum baseline)"
              % MODEL_DIR)
        return p
    with open(f) as fh:
        doc = json.load(fh)
    user = doc.get(section, {})
    user = {k: v for k, v in user.items() if not k.startswith("_")}   # _keys are notes
    unknown = sorted(set(user) - set(DEFAULTS))
    if unknown:
        raise SystemExit("params.json: unknown key(s) %s -- typo, or a tunable that "
                         "is not parameterised yet" % unknown)
    p.update({k: v for k, v in user.items() if k in DEFAULTS})
    print("params.json: %s  [%s]" % (f, section))
    if "_model" in doc:
        print("  model: %s" % doc["_model"])
    diffs = [k for k in _TUNABLES if p[k] != DEFAULTS[k]]
    if diffs:
        for k in diffs:
            print("  override  %-24s %s   (default %s)" % (k, p[k], DEFAULTS[k]))
    else:
        print("  no overrides -- identical to the male jejunum baseline")
    return p


globals().update(_load_params())
OUTER_ENV_SIGMA = tuple(OUTER_ENV_SIGMA)   # json gives a list; scipy wants a tuple


def fill_lumen(vol):
    """Discard the RF Lumen label, then label the space enclosed by the WALL
    layers as Lumen. Deriving from WALL (not `>0`) is what keeps this immune to
    RF lumen bleeding into slide background -- see module docstring."""
    out = vol.copy()
    n_rf = int((out == LUMEN).sum())
    out[out == LUMEN] = 0                  # drop RF lumen; re-derive below
    slice_area = vol.shape[1] * vol.shape[2]
    min_hole = 0.01 * slice_area           # ignore holes < 1% of the frame
    n_open = 0
    for i in range(out.shape[0]):
        sl = out[i]
        wall = np.isin(sl, WALL)
        if not wall.any():
            continue
        holes = binary_fill_holes(wall) & ~wall
        big = np.zeros_like(holes)
        if holes.any():
            lab, n = cc_label(holes)
            for c in range(1, n + 1):
                comp = lab == c
                if comp.sum() >= min_hole:  # keep only the big central hole(s)
                    big |= comp
        if not big.any():
            n_open += 1                     # open C-shape -> no enclosed lumen
            continue
        sl[big] = LUMEN
        out[i] = sl
    n_new = int((out == LUMEN).sum())
    print(f"  RF lumen discarded: {n_rf:,} vox -> geometric lumen: {n_new:,} vox")
    if n_open:
        print(f"  {n_open} slice(s) had no enclosed interior (open wall) -> no lumen")
    return out


def trim_submucosa_fingers(vol):
    """Reassign submucosa that is too far from the muscularis to be part of the band.

    See SUBMUC_MAX_FROM_MUSC_UM for the calibration. Uses distance from the muscularis,
    not connectivity -- the spurious fingers ARE connected to the real band through the
    folds, which is why the earlier component test did not catch them.
    """
    # scipy's EDT promotes to float64 internally -> 11.2 GB on this volume. The `edt`
    # package is float32 and is what add_villi.py already uses.
    import edt as _edtpkg
    sub = vol == 3
    if not sub.any():
        return vol
    d_mus = _edtpkg.edt((vol != 4).astype(np.uint8),
                        anisotropy=(Z_UM_ASSUMED, XY_UM, XY_UM), black_border=False)
    spur = sub & (d_mus > SUBMUC_MAX_FROM_MUSC_UM)
    n_far = int(spur.sum())
    del d_mus                       # free before the second EDT; both are float32
                                    # over the whole volume and do not fit together
    if SUBMUC_MAX_THICK_UM:
        # local thickness = 2x distance to the edge of the submucosa itself
        thick = 2.0 * _edtpkg.edt(sub.astype(np.uint8),
                                  anisotropy=(Z_UM_ASSUMED, XY_UM, XY_UM),
                                  black_border=False)
        spur &= thick < SUBMUC_MAX_THICK_UM
        del thick
    out = vol.copy()
    out[spur] = MUCOSA
    guard = (f", thinner than {SUBMUC_MAX_THICK_UM:.0f} um"
             if SUBMUC_MAX_THICK_UM else "")
    print(f"  submucosa fingers trimmed (> {SUBMUC_MAX_FROM_MUSC_UM:.0f} um from "
          f"muscularis{guard}): {int(spur.sum()):,} vox -> mucosa "
          f"({100*spur.sum()/sub.sum():.1f}% of submucosa)")
    if SUBMUC_MAX_THICK_UM:
        print(f"    thickness guard kept {n_far - int(spur.sum()):,} vox of far "
              f"submucosa that is thick enough to be follicle, not finger")
    return out


def smooth_outer_envelope(vol):
    """Smooth the OUTER tissue boundary without touching interior layer assignments.

    Fill the tissue to a solid envelope, 3D-smooth that, then (a) drop tissue that
    falls outside the smoothed envelope and (b) fill new envelope interior with the
    nearest existing label. Interior boundaries -- and the folds -- are preserved.
    """
    tis = vol > 0
    env = np.zeros(vol.shape, bool)
    for i in range(vol.shape[0]):
        if tis[i].any():
            env[i] = binary_fill_holes(tis[i])
    env_s = gaussian_filter(env.astype(np.float32), sigma=OUTER_ENV_SIGMA) > 0.5
    out = vol.copy()
    out[~env_s] = 0                                   # trim outside the smooth tube
    gap = env_s & (out == 0)                          # new interior to fill
    # MEMORY: distance_transform_edt(return_indices=True) on the whole volume wants
    # a (3, nz, ny, nx) int32 index array -- 5.6 GB here. The gap is a thin rind at
    # the envelope, so fill it per slice instead; 2D nearest-label is adequate and
    # costs ~30 MB.
    n_filled = 0
    for i in range(out.shape[0]):
        g = gap[i]
        if not g.any():
            continue
        src = out[i]
        if not (src > 0).any():
            continue
        _, idx = distance_transform_edt(src == 0, return_indices=True)
        out[i][g] = src[idx[0][g], idx[1][g]]
        n_filled += int(g.sum())
    gap_n = n_filled
    print(f"  outer envelope smoothed (sigma {OUTER_ENV_SIGMA}): "
          f"trimmed {int((tis & ~env_s).sum()):,} vox, filled {gap_n:,} vox")
    return out


def enforce_outer_coat(vol):
    """Guarantee a solid muscularis at the outer surface, WITHOUT the radial rebuild.

    Extracted from relayer_wall() (2026-08-28) so segmentations that do not need the
    concentric reconstruction can still get the coverage guarantee. Same morphology
    as the original: any wall voxel touching background that is not muscularis
    becomes muscularis, OUTER_COAT_VOX voxels inward.
    """
    out = vol.copy()
    wall = np.isin(out, WALL)
    for _ in range(OUTER_COAT_VOX):
        outside = out == 0
        coat = wall & binary_dilation(outside) & (out != 4)
        if not coat.any():
            break
        out[coat] = 4
    n_exp = int((wall & binary_dilation(out == 0) & (out != 4)).sum())
    print(f"  outer coverage enforced ({OUTER_COAT_VOX} vox); "
          f"non-muscularis exposed after: {n_exp}")
    return out


def relayer_wall(vol):
    """Rebuild mucosa/submucosa/muscularis as SOLID, Z-CONNECTED concentric shells.

    The classifier separates lumen and the muscularis outer edge cleanly, but the
    two internal boundaries (mucosa|submucosa, submucosa|muscularis) are the hard
    distinction -- the submucosa comes out ~65% shattered per slice, and worse,
    the thin shell shifts radially by a few voxels slice-to-slice so it fragments
    ALONG Z into hundreds of disconnected pieces (measured 2026-07-21: submucosa
    45% outside its largest 3D component). Anatomy guarantees the layers are
    concentric tubes, so we assign each wall voxel by its normalized radial
    position t in [0,1] from lumen edge (t=0) to muscularis outer edge (t=1).

    The KEY to a single connected object per layer: t is computed per slice, then
    SMOOTHED ALONG Z (normalized by the wall mask). A z-coherent t field makes the
    shell boundaries move smoothly in z, so each shell is one continuous tube
    instead of a stack of independently-jittering rings.

    Thresholds t1,t2 are VOLUME-PRESERVING: t-quantiles at the classifier's own
    mucosa and mucosa+submucosa volume fractions, so each material keeps its
    measured total volume (what dosimetry needs) while becoming a clean shell.
    """
    nz, ny, nx = vol.shape

    # ---- pass 0: build the tube envelope (filled cross-section) and the enclosed
    # lumen, then 3D-SMOOTH BOTH into a clean tube. The muscularis outer edge IS
    # this envelope, so smoothing it here is what stops the outer layer from
    # rendering as stacked ribbons. This idealizes the outer/inner contours into
    # a smooth tube (user decision 2026-07-21) rather than tracking every 5um
    # section's exact outline.
    core = np.zeros(vol.shape, bool)               # filled tube cross-section
    lumen_v = np.zeros(vol.shape, bool)            # enclosed lumen (inner boundary)
    for i in range(nz):
        w = np.isin(vol[i], WALL)
        if not w.any():
            continue
        env = binary_fill_holes(w)
        core[i] = env
        holes = env & ~w
        # 2026-08-28: keep ALL enclosed spaces, not just the largest.
        #
        # The original kept `lab == sz.argmax()` and let every other enclosed space
        # become wall. That is correct for a topological annulus (the female: one
        # outer boundary, one lumen) but wrong for a COLLAPSED gut, where mucosal
        # folds create many lumen channels. On the male it filled in all but one,
        # turning the wall into a solid blob and degenerating the radial coordinate
        # t into "distance from the one surviving lumen" -- measured: lumen fell
        # 6.73% -> 2.77% of the model, and the apparent 4.4 mm "muscularis" was the
        # relayer's shell around that blob rather than tissue.
        #
        # This is a STRICT GENERALISATION: with a single enclosed space it is
        # identical to the old behaviour, so it is a no-op on the female and both
        # models keep running the same algorithm.
        if holes.any():
            if RELAYER_KEEP_ALL_LUMEN:
                lumen_v[i] = holes
            else:
                lab, n = cc_label(holes)
                if n >= 1:
                    sz = np.bincount(lab.ravel()); sz[0] = 0
                    lumen_v[i] = lab == sz.argmax()
        else:
            lumen_v[i] = vol[i] == LUMEN

    env_sig = (ENV_Z_SIGMA, ENV_XY_SIGMA, ENV_XY_SIGMA)
    core_s = gaussian_filter(core.astype(np.float32), sigma=env_sig) > 0.5
    del core
    lumen_s = gaussian_filter(lumen_v.astype(np.float32), sigma=env_sig) > 0.5
    del lumen_v
    lumen_s &= core_s                              # lumen must sit inside the tube
    wall_vol = core_s & ~lumen_s                   # the smooth-tube wall

    # ---- pass 1: radial coordinate from lumen edge (t=0) to smooth tube outer
    # edge (t=1). NO extra t-smoothing: the tube envelope (core_s / lumen_s) is
    # already 3D-smoothed, so the distance ratio on it is smooth AND z-coherent.
    # Smoothing t here previously broke radial MONOTONICITY near the outer edge,
    # which let submucosa reach the outer surface -> holes in the muscularis
    # (14% of the outer wall was exposed). A clean distance ratio is monotone
    # outward by construction, so the outer voxel is always t~1 = muscularis.
    t_vol = np.zeros(vol.shape, np.float32)
    wf = wall_vol.astype(np.float32)
    for i in range(nz):
        if not wall_vol[i].any():
            continue
        d_in = distance_transform_edt(~lumen_s[i])
        d_out = distance_transform_edt(core_s[i])   # dist to outside the tube
        den = d_in + d_out
        t = np.where(den > 0, d_in / np.maximum(den, 1e-6), 0.0).astype(np.float32)
        # AGGRESSIVE in-plane smoothing of the radial field (normalized by wall)
        num = gaussian_filter(t * wf[i], RELAYER_XY_SIGMA)
        den2 = gaussian_filter(wf[i], RELAYER_XY_SIGMA)
        t_vol[i] = np.where(den2 > 1e-4, num / np.maximum(den2, 1e-6), t)

    # AGGRESSIVE z-smoothing of the radial field (chunked over Y), normalized by
    # the wall mask -> smooth, z-coherent layer interfaces.
    for y0 in range(0, ny, 256):
        y1 = min(y0 + 256, ny)
        tc = t_vol[:, y0:y1, :]; wc = wf[:, y0:y1, :]
        num = gaussian_filter1d(tc * wc, RELAYER_Z_SIGMA, axis=0, mode="nearest")
        den = gaussian_filter1d(wc, RELAYER_Z_SIGMA, axis=0, mode="nearest")
        t_vol[:, y0:y1, :] = np.where(den > 1e-4, num / np.maximum(den, 1e-6), tc)

    # volume-preserving thresholds from the classifier's wall fractions (relative
    # proportions, applied to the new smooth-tube wall).
    tot = {l: int((vol == l).sum()) for l in WALL}
    wsum = max(sum(tot.values()), 1)
    f_muc, f_sub = tot[MUCOSA] / wsum, tot[3] / wsum
    pooled = t_vol[wall_vol]
    t1 = float(np.quantile(pooled, f_muc))
    t2 = float(np.quantile(pooled, f_muc + f_sub))
    print(f"  radial relayer: wall fractions muc {f_muc:.3f} sub {f_sub:.3f} "
          f"mus {tot[4]/wsum:.3f} -> t1={t1:.3f} t2={t2:.3f} "
          f"(env-sigma z={ENV_Z_SIGMA}/xy={ENV_XY_SIGMA})")

    # rebuild the tissue from the smooth tube: clear old lumen+wall, keep bg,
    # no serosa yet (add_serosa runs after).
    out = vol.copy()
    out[np.isin(out, (LUMEN,) + WALL)] = 0
    out[lumen_s] = LUMEN
    out[wall_vol & (t_vol < t1)] = 2
    out[wall_vol & (t_vol >= t1) & (t_vol < t2)] = 3
    out[wall_vol & (t_vol >= t2)] = 4

    # ---- coverage guarantee: no mucosa/submucosa may be exposed at the tube's
    # outer surface (that is what shows as "holes in the muscularis"). Coat the
    # outer boundary: any wall voxel touching the outside that is not muscularis
    # becomes muscularis, for OUTER_COAT_VOX voxels inward. Also close small
    # pinholes in the muscularis shell. Cheap morph, guarantees a solid outer
    # layer; the volume shift is tiny (the muscularis is already ~17 voxels thick).
    for _ in range(OUTER_COAT_VOX):
        outside = out == 0                                  # beyond the tube
        coat = wall_vol & binary_dilation(outside) & (out != 4)
        if not coat.any():
            break
        out[coat] = 4
    n_exp = int((wall_vol & binary_dilation(out == 0) & (out != 4)).sum())
    print(f"  outer coverage enforced ({OUTER_COAT_VOX} vox); "
          f"non-muscularis exposed after: {n_exp}")
    return out


def add_serosa(vol, thickness_vox):
    """Synthesize a thin serosa shell just outside the muscularis. Uses 3D
    dilation so the shell is CONNECTED across slices (per-slice rings don't
    overlap in Z and fragment into thousands of pieces). Serosa is the outer
    mesothelial covering -> a thin ring beyond the existing tissue.

    Grows from WALL only, and is masked off anything already labelled, so it can
    never eat inward into the lumen or overwrite a real layer. [2026-07-20]"""
    out = vol.copy()
    wall = np.isin(vol, WALL)
    occupied = vol > 0                     # wall + lumen: serosa may touch neither
    shell = binary_dilation(wall, iterations=thickness_vox) & ~occupied
    out[shell] = SEROSA
    print(f"  serosa shell: {int(shell.sum()):,} vox "
          f"({shell.sum()/max(int(wall.sum()),1)*100:.1f}% of wall volume)")
    return out


def z_smooth(vol, sigma, y_chunk=128):
    """One-hot -> gaussian along Z -> argmax. Chunked over Y to cap memory
    (smoothing is along Z only, so Y-chunks are exact, not approximate)."""
    nz = int(vol.max()) + 1
    out = np.empty_like(vol)
    for y0 in range(0, vol.shape[1], y_chunk):
        sub = vol[:, y0:y0 + y_chunk, :]
        best_val = np.zeros(sub.shape, np.float32)
        best_idx = np.zeros(sub.shape, np.uint8)
        for c in range(nz):
            chan = (sub == c).astype(np.float32)
            chan = gaussian_filter1d(chan, sigma=sigma, axis=0, mode="nearest")
            upd = chan > best_val
            best_val[upd] = chan[upd]
            best_idx[upd] = c
        out[:, y0:y0 + y_chunk, :] = best_idx
    return out


def ortho(vol, path, tag):
    H, W = vol.shape[1], vol.shape[2]
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].imshow(vol[:, H // 2, :], cmap=CMAP, vmin=0, vmax=5,
                 aspect="auto", interpolation="nearest")
    ax[0].set_title(f"XZ (y={H//2})  [{tag}]"); ax[0].set_xlabel("x")
    ax[0].set_ylabel("slice z")
    ax[1].imshow(vol[:, :, W // 2], cmap=CMAP, vmin=0, vmax=5,
                 aspect="auto", interpolation="nearest")
    ax[1].set_title(f"YZ (x={W//2})  [{tag}]"); ax[1].set_xlabel("y")
    fig.suptitle(f"Cleaned label volume — {tag}")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"wrote {path}")


def main():
    d = np.load(IN_NPZ)
    vol = d["volume"]
    print(f"loaded {IN_NPZ}  shape {vol.shape}")

    if TRIM_SUBMUCOSA_FINGERS:
        print("trim spurious submucosa fingers ...")
        vol = trim_submucosa_fingers(vol)
    if not USE_LUMEN_HOLEFILL:
        print("lumen hole-fill SKIPPED (USE_LUMEN_HOLEFILL=False) -- "
              "keeping the segmented lumen")
    else:
        print("lumen hole-fill ...")
        vol = fill_lumen(vol)
    print(f"z-smooth (sigma={Z_SIGMA}) ...")
    vol = z_smooth(vol, Z_SIGMA)
    # Relayer AFTER z-smooth: z_smooth makes the wall envelope + lumen
    # z-continuous, then the radial reconstruction turns those into clean solid
    # shells. Running it before z_smooth lets the per-label z-blur re-roughen the
    # thin outer muscularis shell (measured: muscularis 23% vs 1% fragmented).
    if USE_RADIAL_RELAYER:
        print("radial relayer (solid concentric mucosa/submucosa/muscularis) ...")
        vol = relayer_wall(vol)
    else:
        print("radial relayer SKIPPED (USE_RADIAL_RELAYER=False) -- outer coat only ...")
        if SMOOTH_OUTER_ENVELOPE:
            vol = smooth_outer_envelope(vol)
        vol = enforce_outer_coat(vol)

    serosa_vox = max(1, round(SEROSA_UM / XY_UM))
    print(f"add artificial serosa ({SEROSA_UM} um -> {serosa_vox} voxel shell) ...")
    vol = add_serosa(vol, serosa_vox)

    outv = os.path.join(OUT_DIR, "volume_clean.npz")
    np.savez_compressed(outv, volume=vol, zrange=d["zrange"],
                        stride=d["stride"])
    print(f"wrote {outv}  ({os.path.getsize(outv)/1e6:.1f} MB)")
    ortho(vol, os.path.join(OUT_DIR, "ortho_clean.png"), "clean")

    # quick per-class voxel tally
    for c, n in zip(*np.unique(vol, return_counts=True)):
        print(f"  label {c}: {n} voxels")

    # ---- self-check: catch the failure modes that produced the bad 2026-07-19
    # volume. Serosa flooding the lumen, and slices that lost their lumen.
    print("\nself-check:")
    t = (vol > 0).sum(axis=(1, 2))
    lum = (vol == LUMEN).sum(axis=(1, 2))
    ser = (vol == SEROSA).sum(axis=(1, 2))
    live = t > 0.05 * np.median(t[t > 0])          # ignore empty taper slices
    ser_frac = np.divide(ser, np.maximum(t, 1)) * 100
    lum_frac = np.divide(lum, np.maximum(t, 1)) * 100
    flood = np.where(live & (ser_frac > 35))[0]
    nolum = np.where(live & (lum_frac < 5))[0]
    thin  = np.where(live & (t < 0.6 * np.median(t[live])))[0]
    zr = d["zrange"]
    def _slides(ix):
        return ", ".join(str(int(zr[i])) for i in ix[:20]) + (" ..." if len(ix) > 20 else "")
    print(f"  serosa >35% of tissue : {len(flood):>3} slice(s)"
          + (f"  -> slides {_slides(flood)}" if len(flood) else "   OK"))
    print(f"  lumen  < 5% of tissue : {len(nolum):>3} slice(s)"
          + (f"  -> slides {_slides(nolum)}" if len(nolum) else "   OK"))
    print(f"  tissue <60% of median : {len(thin):>3} slice(s)"
          + (f"  -> slides {_slides(thin)}" if len(thin) else "   OK"))
    if len(flood) or len(nolum) or len(thin):
        print("  ^^ INVESTIGATE before meshing. A slide range failing all three is "
              "usually a clipped QuPath LeftCurl region (see AIM doc, 2026-07-20).")
    else:
        print("  all checks passed -- volume looks sane to mesh.")


if __name__ == "__main__":
    main()
