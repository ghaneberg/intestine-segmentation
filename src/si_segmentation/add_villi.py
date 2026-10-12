"""
add_villi.py  --  mucosa microstructure (villi + crypts + stem cells)
------------------------------------------------------------------
Builds the radiosensitive microstructure the classifier cannot see:

  1. EXPAND the mucosa inward (into the lumen) to a realistic depth
     (~670 um = villus zone + crypt zone), because the segmented mucosa
     (~345 um) is compressed relative to real jejunum. (user 2026-07-23)
  2. VILLI: carve intervillous lumen grooves into the OUTER mucosa so the
     villus tips sit on the luminal surface (subtraction), oriented along
     the local surface normal. Villi are placed with a 3D COLLISION TEST so
     no two share/touch a voxel (place_villi_no_overlap): a straight finger
     is tried first, then bent into free space if it would collide. Poisson-
     disk seed spacing alone does NOT prevent overlap -- it bounds only the
     tips at the surface, while the 400 um fingers fan along non-parallel
     local normals and cross deeper in (48% overlapped before this).
  3. CRYPTS: tubular glands (label 6) in the DEEP mucosa, extending from
     the villus bases down toward the submucosa. Geometry MEASURED from
     this animal: 55.0 um outer dia (n=25), 61 um spacing (5 fields).
     Crypts per villus is DERIVED from our own two spacings,
     (224/61)^2 = ~13.5 -- not the uncited "6-14" that used to sit here.
  4. STEM CELLS (label 7): the deepest ~40 um of each crypt = the +4 /
     crypt-base columnar stem-cell zone, the dose-limiting radiosensitive
     target (killed by ~1 Gy). This is the point of the whole model.

Literature (pig/human small intestine), kept only for context -- every
number below that could be measured HAS been, and the measurement wins:
crypt depth ~266 um (pig jejunum), dia 50-150 um (uncited range, see
REFERENCES.md section 5); stem cells ~4 cell diameters (~40 um) above the
crypt base (Booth & Potten 2000 -- the one load-bearing citation).
RETRACTED 2026-08-20: the "~35/mm^2" villus density that used to appear
here. MEASURED spacing is 224.4 um (5 fields, 47 villi) = ~20/mm^2, and
the repo's other uncited figure (~20/mm^2) is the one the data supports.
The OLD model realised 17.9/mm^2, within 5% of the measured spacing -- i.e.
the density was already right. VILLUS_SPACING has since been raised 150 ->
165 only because the villi got wider (see VILLUS_R); density is emergent, so
RE-CHECK the realised figure against 19.9/mm^2 after the next build.
See measurements_villus_spacing.csv, analyze_villus_measurements.py.

PROVENANCE: the crypt geometry below is MEASURED from this animal, not taken
from the ranges above -- see ASSUMPTIONS.md for what each number is and
REFERENCES.md for the citations behind the ones that are literature-derived
(the stem-cell depth is the only load-bearing one).

ALL DEPTHS ARE TRUE 3D PERPENDICULAR DISTANCES (2026-07-29). They used to be
per-slice 2D in-plane distances, which is wrong here because the loop is cut
OBLIQUELY: the lumen centroid drifts 0.738 um per um of z, putting the tube axis
36.4 deg off the z-axis. An in-plane distance then overstates true depth by
1/|n_xy| (measured median 1.194x), so the model held 561 um of mucosa where it
claimed 670, 335 um villi where it claimed 400, and a 226 um crypt zone where it
claimed 270 -- ~16% shallow against its own literature basis, straight into the
stem-cell dose.

Every villus is ANCHORED: a candidate is kept only if it touches the deep mucosa
(see place_villi_no_overlap). Villi near the z-faces whose normals point out of
the 1 mm slab get truncated by the array bound and cannot anchor, so they are
rejected -- that surface carries bare mucosa rather than floating fragments.
Nothing is invented to fill the gap. A self-check at the end asserts the tissue
is a single connected component.

Reads volume_clean.npz -> writes volume_villi.npz.
"""

import os
import json
import sys
import numpy as np
import scipy.ndimage as ndi
import edt as _edt          # pip install edt


def edt3(mask, spacing):
    """Exact anisotropic 3D EDT: distance from each True voxel to the nearest False.
    Same convention and result as ndi.distance_transform_edt(mask, sampling=spacing)
    (verified to 5e-6 um), but float32 and without the feature transform -- scipy
    builds a (3, *shape) int64 intermediate and needs 21 GB on this volume."""
    return _edt.edt(np.ascontiguousarray(mask), anisotropy=spacing,
                    black_border=False, parallel=0)

def surface_areas(vol, spacing, ref_sigma_um):
    """Three denominators for villus density. THE CHOICE MATTERS 3x.

    The mucosal surface of a collapsed gut has no scale-free area: isosurfacing
    the lumen after Gaussian pre-smoothing gives 181 mm^2 at sigma 0, 126 at 8 um,
    101 at 12 um and 67 at 25 um, with the lumen VOLUME changing under 5% -- it is
    de-crenulating, not deforming. There is no plateau, so "the" area does not
    exist; you must say at what scale you measured.

    - naive : iface_vox * dy*dx, what this script used to report. Charges every
              interface voxel the area of a Z-FACING face; with dz=5 and dy=dx=8.76
              a y/x-facing face is 43.8 um^2, not 76.8. Inflated ~1.2x over `face`.
    - face  : orientation-correct face counting. The honest voxel-scale area.
    - ref   : isosurface after ref_sigma_um pre-smoothing. This is the scale a
              histologist works at when drawing a perpendicular across a section --
              they ignore the fine crenulation. At sigma 25 um the implied mucosal
              thickness is 725 um against the manually measured 811 um, so this is
              the denominator COMMENSURABLE with the measured 33.4/mm^2 target.
              Report it with the sigma stated; never quote a bare per-mm^2.
    """
    dz, dy, dx = spacing
    A, B = vol == MUCOSA, vol == LUMEN
    naive = float((A & ndi.binary_dilation(B, np.ones((3, 3, 3), bool))).sum()) * dy * dx / 1e6
    fz = int((A[:-1] & B[1:]).sum() + (B[:-1] & A[1:]).sum())
    fy = int((A[:, :-1] & B[:, 1:]).sum() + (B[:, :-1] & A[:, 1:]).sum())
    fx = int((A[:, :, :-1] & B[:, :, 1:]).sum() + (B[:, :, :-1] & A[:, :, 1:]).sum())
    face = (fz * dy * dx + fy * dz * dx + fx * dz * dy) / 1e6
    ref = float("nan")
    try:
        from skimage.measure import marching_cubes
        sig = (ref_sigma_um / dz, ref_sigma_um / dy, ref_sigma_um / dx)
        tot = 0.0
        for z0 in range(0, vol.shape[0] - 1, 40):
            z1 = min(z0 + 41, vol.shape[0])
            f32 = ndi.gaussian_filter(B[z0:z1].astype(np.float32), sig, mode="nearest")
            if f32.max() <= 0.5 or f32.min() >= 0.5:
                continue
            v, fc, _, _ = marching_cubes(f32, level=0.5, spacing=spacing)
            t = v[fc]
            tot += float(np.linalg.norm(np.cross(t[:, 1] - t[:, 0],
                                                 t[:, 2] - t[:, 0]), axis=1).sum() / 2)
        ref = tot / 1e6
    except Exception as e:
        print(f"  (reference surface unavailable: {e})")
    return naive, face, ref


# Shared script: the MODEL DIRECTORY is sys.argv[1], else the cwd. It supplies the
# input volume, receives volume_villi.npz, and may hold params.json
# ["add_villi"] overriding the measured morphometry below.
MODEL_DIR = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.getcwd())
OUT_DIR = MODEL_DIR
# 2026-08-31: default input is now the REPAIRED volume. volume_clean.npz carries
# 972 mucosa islands buried inside the wall and a submucosa shattered into 5,504
# components; every stray submucosa fleck is a FALSE FLOOR for d_sub and deletes
# the mucosa around it from villus_zone (34% of the luminal surface had no villus
# zone at all; after repair 6.4%). See fix_buried_islands.py / fix_submucosa.py.
IN_NPZ = os.path.join(OUT_DIR, os.environ.get("IN_NPZ", "volume_clean_smoothed.npz"))
# scale at which the REFERENCE mucosal surface is measured -- see surface_areas()
REF_SIGMA_UM = float(os.environ.get("REF_SIGMA_UM", "25.0"))
OUT_NPZ = os.path.join(OUT_DIR, "volume_villi.npz")

SPACING = (5.0, 8.7616, 8.7616)     # (z,y,x) um
LUMEN, MUCOSA, SUBMUC, MUSCUL, SEROSA, CRYPT, STEM = 1, 2, 3, 4, 5, 6, 7

# MALE 2026-08-28: EXPANSION DISABLED. `new_muc = inner & (d_sub < TARGET_MUCOSA_UM)`
# converts every voxel within TARGET of submucosa to mucosa, to give a uniform mucosal
# depth. In the female's open tube only 8% of lumen sits that close, so it survives.
# In the male's COLLAPSED tube 82.6% does -- the expansion consumed the ENTIRE lumen
# (12,425,066 vox, exactly the whole of it), leaving no mucosa/lumen interface, ZERO
# villus seeds, and then an abort on 1,164 detached components.
#
# Without it the villus zone is 14.58 mm^3 (30.3% of mucosa; female 10.05 / 48.2%) and
# ~8,000 seed candidates are available, so nothing downstream needs it. The mucosa is
# then NOT uniformly TARGET deep -- it varies, which is correct for a folded gut.
#
# TARGET_MUCOSA_UM is still used for CRYPT_ZONE = TARGET - VILLUS_LEN = 390 um.
EXPAND_MUCOSA = False

TARGET_MUCOSA_UM = 810.9  # MALE 2026-08-28 (F 519.0)            # MEASURED 2026-07-30, not literature. 24 paired
                                    # manual measurements on well-oriented units
                                    # (crypt base->opening, opening->villus tip) in
                                    # near-perpendicular fields: total mucosa mean
                                    # 519.3 um, median 522.5, sd 76.5, CV 14.7%,
                                    # 95% CI 489-550. Was 670 (domestic-pig
                                    # literature) -> that over-modelled mucosal
                                    # tissue 1.29x, and mucosal volume is what
                                    # absorbs dose. This animal is 75% of the
                                    # domestic-pig figure (430+266=696 um); the
                                    # mucosa/submucosa boundary was confirmed
                                    # correct by the user, so that is real biology,
                                    # not a segmentation artifact.
                                    # See measurements_crypt_villus.csv.
VILLUS_LEN = 420.9  # MALE, n=25, sd 86.2 (F 297.0)                  # MEASURED: villus height mean 297.0 um
                                    # (median 294, sd 56.4, CV 19.0%). Was 400.
                                    # CRYPT_ZONE = TARGET - VILLUS_LEN = 222 um,
                                    # which matches the measured crypt depth
                                    # (mean 222.3) by construction. CRYPT_FRAC 0.428,
                                    # villus:crypt 1.34:1 -- the ratio five automated
                                    # attempts failed to extract.
VILLUS_R  = 62.5  # MALE, derived 125 um width (F 82.5)                    # MEASURED-DERIVED 2026-08-20: villus half-width,
                                    # ~165 um dia. Was 65.0 (=130 um dia), a value chosen
                                    # BELOW the uncited literature minimum to buy packing
                                    # room -- that trade-off is now retired.
                                    # NOT the raw mean of the measurements. Direct base
                                    # widths (n=10) mean 235.7 um, but that is inflated:
                                    # the section is 36.4 deg oblique, an oblique cut
                                    # through a convex finger can only look WIDER (never
                                    # narrower), and wide profiles get preferentially
                                    # picked when choosing villi to measure. Both biases
                                    # are one-sided. CV came out 42.6% (range 105-421),
                                    # which is inflation, not biology.
                                    # Two unbiased estimators agree instead:
                                    #   constraint  width = spacing - groove
                                    #               = 224.4 - 49.5 (median) = 174.9
                                    #   lower tail  Q1 of direct widths      = 154.5
                                    #   -> adopted 165 um, VILLUS_R = 82.5
                                    # Context only, NOT the source: the repo's uncited
                                    # literature range 150-185 um brackets this.
                                    # See measurements_villus_width.csv and the
                                    # ROBUST WIDTH ESTIMATE block in
                                    # analyze_villus_measurements.py.
VILLUS_TIP = 0.64                   # MEASURED 2026-08-20: tip radius / base radius
                                    # (finger taper). Mean 0.642 over the 8 of 10 villi
                                    # whose tip was narrower than their own base (2 of 10
                                    # measured a tip WIDER than its base -- impossible for
                                    # a tapering finger, so those cuts are not one villus
                                    # and are excluded). sd 0.203.
                                    # HONEST CAVEAT: the 95% CI is 0.47-0.81, so this is
                                    # NOT a statistically significant move from the old
                                    # 0.55. It is adopted because 0.55 was uncited and
                                    # this is measured, not because the data rejects 0.55.
                                    # A ratio of two in-plane lengths on the SAME villus
                                    # is the one quantity here where obliquity largely
                                    # cancels, which is why taper survived when the
                                    # absolute widths did not.
VILLUS_SPACING = 125.0  # MALE = one villus width (F 165.0)              # Poisson-disk MIN separation of CANDIDATE seeds.
                                    # RAISED 2026-08-20 from 150.0 to exactly one villus
                                    # width (2*VILLUS_R). Below that, neighbouring
                                    # candidates are guaranteed to collide and burn the
                                    # bend-and-retry path for nothing; at one width they
                                    # can just touch and the collision test does the rest.
                                    # This is a candidate-pool knob, NOT the density:
                                    # realised density is emergent. VERIFY IT after the
                                    # run against the MEASURED 19.9/mm^2 (224.4 um
                                    # spacing, 5 fields, 47 villi). The old model realised
                                    # 17.9/mm^2, within 5%. If the wider villi push it
                                    # well below that, this is the knob to retune.
                                    # NOT an overlap guarantee: min-sep only bounds the
                                    # tips at the surface; the 400 um cylinders fan along
                                    # NON-parallel local normals and cross deeper in (at
                                    # 158 um, 48% of villi overlapped -- measured). Overlap
                                    # is prevented by the 3D collision test in
                                    # place_villi_no_overlap(); culling this pool lands at
                                    # ~20 villi/mm^2 (literature).
# --- seeding mode (added 2026-08-24) ------------------------------------------
# VILLUS_SPACING above is a MINIMUM separation handed to dart-throwing Poisson-disk,
# but realised density is set by the MEAN separation, and for Poisson-disk those differ
# by ~1.33x.  At min-sep 165 um the candidate mean lands near 220 um -- the right
# density -- but with a long tail of pairs sitting AT the 165 um floor.  Those are
# precisely the pairs whose 165 um-wide BASES touch, and the 3D collision test then
# discards them: 271 rejects at spacing 160 rising to 704 at 130 (measured 2026-08-24),
# while STRAIGHT placements sat pinned at ~470 across the whole sweep.  Real villi are
# quasi-regular -- nearly every neighbour at the mean, almost none at the floor -- so
# no value of a MINIMUM reproduces them.  "lattice" seeds a centroidal-Voronoi set at
# the measured areal density instead.  Default stays "poisson" so nothing changes
# unless asked; override per run with VILLUS_SEED_MODE=lattice in the environment.
VILLUS_SEED_MODE = os.environ.get("VILLUS_SEED_MODE", "poisson").strip().lower()
# --- anchoring mode (added 2026-08-31) ---------------------------------------
# "surface": legacy. Seed on the mucosa/lumen interface, grow the finger INWARD, and
#   accept only if it reaches `deep` (d_sub <= CRYPT_ZONE). That makes a fixed-length
#   villus span the WHOLE villus zone, so it succeeds only where local mucosal depth
#   is close to VILLUS_LEN + CRYPT_ZONE. Too thin fails (no zone) and TOO THICK fails
#   (cannot bridge). After the submucosa repair deepened the mucosa to its true value
#   the second failure took over: "empty" rejections fell 856 -> 196 but "unanchored"
#   rose 872 -> 1,670, and the placed count did not move (913 -> 898).
# "base":   seed on the crypt line (d_sub ~ CRYPT_ZONE, the same surface the crypts
#   already seed on) and grow OUTWARD to the lumen. Every villus is anchored by
#   construction, so the unanchored class -- 60% of all rejections -- cannot occur.
#   Length becomes the LOCAL villus-zone depth, clamped to the measured range below.
VILLUS_ANCHOR = os.environ.get("VILLUS_ANCHOR", "base").strip().lower()
# Measured male villus heights, n=25: min 255, p50 427, mean 420.9, sd 86.2, max 585.
# Variable length is clamped to the OBSERVED range -- never shorter or longer than a
# villus actually measured in this animal. Below the floor the surface carries no
# villus at all rather than a stub.
# --- PROPORTIONAL zones (2026-08-31) -----------------------------------------
# The zone split used to be ABSOLUTE: crypts always took the first 390 um of mucosa
# and villi got whatever was left. Where the segmented mucosa is 600 um deep that
# leaves 210 um of villus zone -- under the measured villus minimum -- so the surface
# went BARE. Measured: 52% of crypt-line seeds rejected that way, giving correctly
# spaced villi in patches separated by nothing, and an areal density (7.6/mm^2) 4.5x
# below the spacing-implied one (34.5/mm^2). Bare inner wall is not anatomy.
#
# What is conserved biologically is the crypt:villus RATIO, not the absolute depths:
# where the mucosa is thinner, crypts and villi shorten together. So both zones are
# now defined on the normalised radial coordinate
#       t = d_sub / (d_sub + d_lumen)        0 at the submucosa, 1 at the lumen
# with the crypt zone taking t <= CRYPT_FRAC everywhere. That is the same normalised
# radial assignment relayer_wall() uses in finalize_volume.py, so the two agree.
VILLUS_ZONE_MODE = os.environ.get("VILLUS_ZONE_MODE", "proportional").strip().lower()
CRYPT_FRAC = 390.0 / 810.9          # 0.4810 -- MEASURED crypt depth / total mucosa
                                    # (male 390.0 : 420.9). Female CRYPT_FRAC 0.428.
# Villi shorter than the measured mean are scaled ISOTROPICALLY, not just shortened:
# a 200 um villus at the full 125 um width is a squat blob, not a villus. Radius is
# only ever scaled DOWN, so no villus is wider than the measured width -- and thinner
# villi in thin mucosa also collide less, which is where the coverage was being lost.
VILLUS_ASPECT_SCALE = os.environ.get("VILLUS_ASPECT_SCALE", "1") not in ("0", "false")
# With proportional zones the floor exists only to stop degenerate rasters, NOT to
# gate coverage: at 60 um the local mucosa is under ~116 um total, which is oblique
# cut or section edge rather than real wall.
VILLUS_LEN_MIN = float(os.environ.get("VILLUS_LEN_MIN", "60.0"))
# Tolerance for the step-3 connectivity guard. The original volume carried 261,983
# deep stray voxels (inherited segmentation noise, since repaired); after the repairs
# the carve itself leaves ~95 on the full volume. Anything at that scale is the carve
# nicking the wall, not a structural failure, and aborting on it prevented every run
# from ever reaching the crypt/stem stage. Above this, still abort.
GUARD_MAX_DEEP_VOX = int(os.environ.get("GUARD_MAX_DEEP_VOX", "20000"))
# --- crypt placement (2026-09-01) --------------------------------------------
# rasterize_cylinders() OR-ed crypts together with NO collision test -- villi went
# through place_villi_no_overlap(), crypts did not. Worse, crypts were seeded on the
# CRYPT LINE (the villus base) and grown INWARD, but CRYPT_SPACING = 70.6 um was
# measured "along the muscularis mucosae" -- the crypt BASE, the opposite end. So the
# spacing was enforced at the mouth and the tubes converged on the way in, arriving at
# the base far closer than the 70.6 um they were measured at. Result on the 2026-08-31
# volume: 7,953 tubes seeded -> 3,628 connected components, and 69 blobs bigger than
# five crypts held 65.3% of ALL crypt volume. Worst at high curvature, where inward
# normals converge fastest.
#
# CRYPT_ANCHOR="base" seeds at the muscularis mucosae -- the surface the spacing was
# measured on, and where the radiosensitive stem cells sit -- and grows OUTWARD to the
# crypt line, so tubes diverge instead of converging. A no-overlap test is applied
# either way. DOSE-CRITICAL: merged cylinders occupy less volume than separate ones,
# so the merged model UNDER-represents stem volume (stem ~ r^2/s^2).
CRYPT_ANCHOR = os.environ.get("CRYPT_ANCHOR", "base").strip().lower()
# Measured male crypt depths, n=25: min 164, p50 365, mean 390.0, sd 119.0, max 627.
# The FLOOR is deliberately far below the measured minimum, for the same reason
# VILLUS_LEN_MIN is: a floor at the measured 164 um deletes the crypt wherever the
# local mucosa is thin, leaving bare wall in the crypt layer (measured: 2,442 of 5,058
# candidates rejected as "too short"). Crypts shorten with the mucosa instead, matching
# the proportional zone rule. 60 um is a raster floor, not a biological one.
CRYPT_LEN_MIN = float(os.environ.get("CRYPT_LEN_MIN", "60.0"))
CRYPT_LEN_MAX = float(os.environ.get("CRYPT_LEN_MAX", "627.0"))
CRYPT_GAP_ITER = 1                  # >=1 empty voxel between neighbouring crypts
# OFF by default. Villi are tapered fingers that plausibly thin as they shorten, but a
# crypt is a tubular gland whose calibre is set by cell size, not depth -- a shallow
# crypt is still ~41 um wide, just shorter. The measurement agrees: crypt outer diameter
# 41.4 um at CV 13.0%, tightly constrained. Scaling it cost 44% of the stem volume
# (0.175 -> 0.098 mm^3 on the test crop) for no biological warrant.
CRYPT_ASPECT_SCALE = os.environ.get("CRYPT_ASPECT_SCALE", "0") not in ("0", "false")
# --- stem-cell definition (2026-09-03) ---------------------------------------
# "depth" (legacy): stem = crypt voxels with d_sub < STEM_DEPTH, i.e. ABSOLUTE
#   distance from the submucosa. That is a sound proxy only while crypts run
#   PERPENDICULAR to the muscularis mucosae. In this collapsed, folded gut many run
#   tangentially, so a long stretch of such a crypt stays within STEM_DEPTH of the
#   submucosa and the whole tube gets labelled stem instead of just its blind base.
#   Measured on the 2026-09-02 model: stem components averaged 1.65x the volume of a
#   clean basal disc, 27.3% of them exceeded 100 um in extent (against an expected
#   41-55 um) and those held 45.6% of all stem volume. Total stem came out ~1.8x what
#   a per-crypt basal definition gives.
# "axis" (default): stem = the first STEM_LEN_UM of each crypt measured ALONG ITS OWN
#   AXIS from its base, which is where the niche actually is. Only possible because
#   crypts are now base-anchored, so the basal end is known at placement time.
# THIS IS THE PRIMARY RADIATION TARGET -- dose is energy/mass, so the definition sets
# the target mass directly. Keep male and female on the SAME setting or the two models
# stop being comparable.
STEM_MODE = os.environ.get("STEM_MODE", "axis").strip().lower()
STEM_LEN_UM = float(os.environ.get("STEM_LEN_UM", "40.0"))
                                    # = STEM_DEPTH(55) - BASE_MARGIN(15), the same
                                    # 40 um niche the legacy rule intended.
VILLUS_LEN_MAX = float(os.environ.get("VILLUS_LEN_MAX", "585.0"))
VILLUS_DENSITY_PER_MM2 = float(os.environ.get("VILLUS_DENSITY_PER_MM2", "33.4"))
                                    # MALE MEASURED: 33.4/mm^2, derived from the measured
                                    # 173.1 um villus spacing (F 19.9, 5 fields, 47 villi).
                                    # 2026-08-31: this had been left at the FEMALE value, so
                                    # every male run was scored against the wrong animal.
                                    # Used ONLY by lattice mode, where it sets seed COUNT.
VILLUS_LATTICE_RELAX = int(os.environ.get("VILLUS_LATTICE_RELAX", "8"))
                                    # Lloyd iterations. 0 = farthest-point only.
VILLUS_GAP_ITER = 1                 # >=1 empty voxel (26-conn) between any two villi ->
                                    # a real intervillous groove, so fingers stay distinct
# 2026-08-26: made env-tunable so the collision-avoidance search can be widened
# without editing code. Bending is how tightly-packed villi coexist; the default
# 3 amps x 8 dirs = 24 retries per candidate, and 386 of 1,170 failed all 24.
#   VILLUS_CURVE_AMPS="45,90,130,170,210"   VILLUS_CURVE_DIRS=16
VILLUS_CURVE_AMPS = tuple(float(x) for x in os.environ.get(
    "VILLUS_CURVE_AMPS", "45,90,130").split(","))  # if a straight villus collides, bend its base
VILLUS_CURVE_DIRS = int(os.environ.get("VILLUS_CURVE_DIRS", "8"))  # this far (um) into one of this many lateral dirs
VILLUS_CURVE_K = 9                        # axis polyline nodes (curve resolution)
VILLUS_OVERSHOOT = 1.35             # rasterize the axis this far x VILLUS_LEN, then CLIP
                                    # to villus_zone. The clip -- not the length -- sets
                                    # the villus, so the base lands FLUSH on the deep
                                    # mucosa even where the smoothed normal field is a
                                    # few degrees off. Without it a villus that falls
                                    # even one voxel short is left floating, because the
                                    # gap it leaves is itself inside villus_zone and so
                                    # gets carved to lumen (302 detached components).
CARVE_REACH = 225.0                 # only carve intervillous lumen within this far of a
                                    # PLACED villus. Where villi were rejected (the
                                    # z-face band, where the normal points out of the
                                    # slab) this leaves SMOOTH, SOLID, full-thickness
                                    # mucosa instead of gouging the whole 400 um villus
                                    # zone away -- a villus-free surface is honest there,
                                    # a 400 um trench in the wall is not. Mean villus
                                    # spacing is ~219 um at 20.9/mm^2, so this still
                                    # carves every real intervillous groove.
CRYPT_LEN = 375.0  # MALE: depth 390 - BASE_MARGIN 15 (F 205.0)                   # crypt tube length. Measured crypt depth is
                                    # 222.3 um (mean); the tube must fit inside the
                                    # 222 um crypt zone above BASE_MARGIN=15, so
                                    # 222 - 15 = 207 -> 205. Was 260, which no
                                    # longer fits the measured zone.
CRYPT_R   = 20.7  # MALE, dia 41.4 um, n=25 (F 27.5)                    # MEASURED 2026-07-30: crypt outer diameter
                                    # 55.0 um mean (n=25, median 53, sd 11.2,
                                    # CV 20.4%, 95% CI 50.6-59.4, range 38-89).
                                    # Was 35.0 (=70 um dia, literature). Measured
                                    # sits at the LOW end of the literature 50-150 um.
                                    # DOSE-CRITICAL: stem-cell volume scales as r^2,
                                    # so this alone takes the target to 0.618x.
                                    # See measurements_crypt_diameter.csv.
CRYPT_SEED_MODE = os.environ.get("CRYPT_SEED_MODE", "poisson").strip().lower()
                                    # "poisson" (default, male/female behaviour):
                                    # seeds_on(), a hard minimum separation, prone
                                    # to floor-spike collisions -- see seeds_lattice's
                                    # docstring. "lattice": seeds_lattice(), farthest-
                                    # point init + Lloyd relaxation, targeting the
                                    # SAME candidate count seeds_on() would generate
                                    # at the measured CRYPT_SPACING (i.e. the natural
                                    # capacity of this surface at that density), spent
                                    # on a regularised distribution instead of a random
                                    # one. CRYPT_SEED_SPACING is ignored in lattice mode.
CRYPT_SEED_SPACING = 0.0            # Poisson-disk MINIMUM separation used to
                                    # GENERATE crypt candidates. 0 -> use
                                    # CRYPT_SPACING, i.e. no over-seeding (the
                                    # male/female behaviour).
                                    #
                                    # ILEUM 2026-09-10: seeding at the measured
                                    # spacing yielded 3,838 of 8,185 candidates
                                    # (2,363 collide / 1,984 too short) = 1.55
                                    # crypts per villus against a measured ~4.
                                    # The shortfall is NOT anatomy: crypt yield
                                    # per mm of luminal surface is flat across
                                    # mucosal thickness (r=+0.21) and no slice
                                    # approaches the ~117 um depth a crypt needs.
                                    # Over-seeding compensates, the same way
                                    # VILLUS_DENSITY_PER_MM2 already does for villi.
                                    #
                                    # CAUTION: collisions are a real geometric
                                    # ceiling on a curved base surface -- tubes
                                    # converge in concave troughs. Placed count may
                                    # SATURATE rather than scale. Seed moderately,
                                    # read the realised count, and iterate.
CRYPT_SPACING = 70.6  # MALE, 5 fields (F 61.0)                # MEASURED 2026-07-30: 5 fields counted along the
                                    # muscularis mucosae (crypts N over stretch L,
                                    # spacing = L/N -- user confirmed full stretch,
                                    # not centre-to-centre): 54.0, 56.4, 48.1, 79.7,
                                    # 65.7 -> mean 60.8 um. Was 90 (literature
                                    # ~120/mm^2); crypts are ~30% more tightly packed
                                    # than assumed. Measured per-length along the
                                    # muscularis mucosae, which sidesteps the
                                    # crypt-to-villus assignment ambiguity entirely.
                                    # DOSE-CRITICAL: stem volume ~ r^2/s^2, so this
                                    # (1.35x) more than cancels the r^2 reduction from
                                    # CRYPT_R (0.62x). Net vs old model: ~1.35x.
                                    # NOTE this is a Poisson-disk MIN separation, so
                                    # realised density will be below 1/s^2 = 269/mm^2
                                    # -- check the run output and retune if needed.
STEM_DEPTH = 55.0                   # = BASE_MARGIN(15) + 40, so the stem zone is
                                    # exactly the literature ~40 um above the CRYPT
                                    # BASE. Was 60 (=45 above base). This also
                                    # resolves the documented 40-vs-60 contradiction:
                                    # the two numbers used different reference
                                    # frames (submucosa vs crypt base), they were
                                    # never really in conflict. Stem-cell position is
                                    # the one number H&E cannot supply (needs
                                    # Lgr5/Olfm4), so it stays literature-based.
                                    # crypt-base stem-cell zone: crypt within this
                                    # distance of the submucosa (the +4/CBC target).
                                    # SOURCE: ~4 cell diameters above the crypt base,
                                    # Booth & Potten, J Clin Invest 105:1493 (2000),
                                    # doi:10.1172/JCI10229; radiosensitivity of that
                                    # position, Potten & Grant, Br J Cancer 78:993
                                    # (1998), doi:10.1038/bjc.1998.618, PMID 9792141.
                                    # CAVEAT: Lgr5+ CBC cells at +1-3 are reported
                                    # RADIORESISTANT (Hua 2012, PMID 22841781) -- this
                                    # zone spans both populations. See REFERENCES.md
                                    # section 3 before naming the target in a writeup.
BASE_MARGIN = 15.0                  # crypts stop this far short of the submucosa
DS_N = 3                            # downsample for the normal field (finer = less lean)
NORMAL_SMOOTH = 2.5                 # smooth the distance field before taking its
                                    # gradient -> consistent perpendicular villi
SEED = 0


# ---- per-model morphometry ---------------------------------------------------
# The literals above are the MALE JEJUNUM's measured values and their provenance
# comments; they are the defaults. params.json ["add_villi"] in the model directory
# overrides any of them. Only measured morphometry is parameterised here -- every
# env-var knob keeps working exactly as before.
_TUNABLES = ("TARGET_MUCOSA_UM", "VILLUS_LEN", "VILLUS_R", "VILLUS_TIP",
             "VILLUS_SPACING", "CRYPT_FRAC", "CRYPT_LEN", "CRYPT_R",
             "CRYPT_SPACING", "CRYPT_SEED_MODE", "CRYPT_SEED_SPACING",
             "VILLUS_DENSITY_PER_MM2",
             "VILLUS_LEN_MIN", "VILLUS_LEN_MAX", "IN_NAME")
IN_NAME = os.environ.get("IN_NPZ", "volume_clean_smoothed.npz")
DEFAULTS = {k: globals()[k] for k in _TUNABLES}


def _load_params(section="add_villi"):
    """Read params.json[section] from the model directory; unknown keys are fatal."""
    p = dict(DEFAULTS)
    f = os.path.join(MODEL_DIR, "params.json")
    if not os.path.exists(f):
        print("no params.json in %s -- using defaults (male jejunum morphometry)"
              % MODEL_DIR)
        return p
    with open(f) as fh:
        doc = json.load(fh)
    user = doc.get(section, {})
    user = {k: v for k, v in user.items() if not k.startswith("_")}   # _keys are notes
    unknown = sorted(set(user) - set(DEFAULTS))
    if unknown:
        raise SystemExit("params.json[%s]: unknown key(s) %s" % (section, unknown))
    p.update(user)
    print("params.json: %s  [%s]" % (f, section))
    if "_model" in doc:
        print("  model: %s" % doc["_model"])
    if section not in doc:
        print("  no [%s] section -- using the male jejunum's measured morphometry"
              % section)
        return p
    for k in _TUNABLES:
        if p[k] != DEFAULTS[k]:
            print("  override  %-22s %-10s (male default %s)" % (k, p[k], DEFAULTS[k]))
        elif k in user:
            print("  confirmed %-22s %-10s measured for this model, equals the male's"
                  % (k, p[k]))
        elif k != "IN_NAME":
            print("  INHERITED %-22s %-10s <- male jejunum, NOT measured for this model"
                  % (k, p[k]))
    return p


globals().update(_load_params())
for _k in ("VILLUS_LEN_MIN", "VILLUS_LEN_MAX"):   # env still wins, as before
    if _k in os.environ:
        globals()[_k] = float(os.environ[_k])
# env var still wins for the input name, as before
IN_NPZ = os.path.join(MODEL_DIR, os.environ.get("IN_NPZ", IN_NAME))
OUT_NPZ = os.path.join(MODEL_DIR, "volume_villi.npz")


def seeds_on(mask, min_dist_um, dz, dy, dx, rng):
    """Poisson-disk seeds on `mask`: every pair of seeds is >= min_dist_um apart
    (in real um), so villi/crypts placed on them cannot overlap. Grid-hash (the
    old method) allowed neighbours from adjacent cells to sit arbitrarily close,
    which fused the villi into a connected mass."""
    from scipy.spatial import cKDTree
    iz, iy, ix = np.nonzero(mask)
    if len(iz) == 0:
        return np.empty((0, 3), int)
    pts = np.stack([iz*dz, iy*dy, ix*dx], axis=1)
    tree = cKDTree(pts)
    order = rng.permutation(len(pts))
    blocked = np.zeros(len(pts), bool)
    keep = []
    for i in order:
        if blocked[i]:
            continue
        keep.append(i)
        for j in tree.query_ball_point(pts[i], min_dist_um):
            blocked[j] = True
        blocked[i] = False                       # keep self, block only neighbours
        blocked[i] = True                        # (accepted; also block from re-pick)
    keep = np.array(keep)
    return np.stack([iz[keep], iy[keep], ix[keep]], axis=1)


def seeds_lattice(mask, n_target, dz, dy, dx, rng, n_relax=None):
    """Quasi-lattice seeds on `mask`: `n_target` points spread as EVENLY as a curved
    surface allows. Farthest-point initialisation (maximises the minimum separation)
    then Lloyd/centroidal-Voronoi relaxation, each pass snapping the cell centroids
    back onto the surface so the seeds never leave the mucosa.

    The point is the SHAPE of the nearest-neighbour distribution, not its floor. A
    Poisson-disk set with min-sep d has a hard spike at d; a centroidal-Voronoi set of
    the same count has its neighbours bunched at the mean with almost nothing at the
    floor. Since a villus base is 2*VILLUS_R wide, it is the floor-adjacent pairs that
    collide -- so tightening a MINIMUM cannot help, but regularising the DISTRIBUTION
    can. Real mucosa is quasi-regular, which is the arrangement this reproduces.

    No surface parameterisation is needed (the loop is cut obliquely and has no clean
    one), and nothing here assumes the surface is developable."""
    from scipy.spatial import cKDTree
    if n_relax is None:
        n_relax = VILLUS_LATTICE_RELAX
    iz, iy, ix = np.nonzero(mask)
    m = len(iz)
    if m == 0 or n_target <= 0:
        return np.empty((0, 3), int)
    n_target = int(min(int(n_target), m))
    pts = np.stack([iz*dz, iy*dy, ix*dx], axis=1)

    def _fps(seeded, want):
        """Farthest-point: extend `seeded` by `want` maximally-distant surface voxels."""
        d2 = np.full(m, np.inf)
        for j in seeded:
            np.minimum(d2, ((pts - pts[j])**2).sum(1), out=d2)
        out = np.empty(want, np.int64)
        for k in range(want):
            j = int(np.argmax(d2))
            out[k] = j
            np.minimum(d2, ((pts - pts[j])**2).sum(1), out=d2)
        return out

    first = np.array([int(rng.integers(m))], np.int64)
    sel = np.concatenate([first, _fps(first, n_target - 1)]) if n_target > 1 else first

    surf = cKDTree(pts)

    def _floor(idx):
        """Smallest nearest-neighbour distance in a seed set (um)."""
        if len(idx) < 2:
            return np.inf
        q = pts[idx]
        return float(cKDTree(q).query(q, k=2, workers=-1)[0][:, 1].min())

    # Lloyd is only an improvement while the surface has room. Asked for a count near
    # saturation it can CLOSE the floor instead of opening it (measured on a flat test
    # patch: 160 um -> 134 um over 8 passes, i.e. worse than the farthest-point start).
    # So keep the best iterate rather than the last -- the seeder can then never return
    # something tighter than its own initialisation.
    best, best_floor = sel, _floor(sel)
    for _ in range(int(n_relax)):
        lab = cKDTree(pts[sel]).query(pts, workers=-1)[1]
        cnt = np.bincount(lab, minlength=n_target).astype(float)
        cen = np.stack([np.bincount(lab, weights=pts[:, a], minlength=n_target)
                        for a in range(3)], axis=1)
        live = cnt > 0
        cen[live] /= cnt[live, None]
        cen[~live] = pts[sel[~live]]          # a cell nobody claimed: leave it put
        cand = np.unique(surf.query(cen, workers=-1)[1])
        if len(cand) < n_target:              # two centroids snapped to one voxel
            cand = np.concatenate([cand, _fps(cand, n_target - len(cand))])
        sel = cand
        f = _floor(sel)
        if f > best_floor:
            best, best_floor = sel, f
    sel = best
    return np.stack([iz[sel], iy[sel], ix[sel]], axis=1)


def rasterize_cylinders(shape, seeds, normal_at, length, radius, spc,
                        base_frac=0.0, tip=1.0):
    """OR-in cylinders of `radius` from each seed along its inward normal for
    `length` um. If base_frac>0, also return a mask of the deepest base_frac of
    each cylinder (for the stem-cell zone)."""
    dz, dy, dx = spc
    body = np.zeros(shape, bool)
    base = np.zeros(shape, bool) if base_frac > 0 else None
    nz, ny, nx = shape
    for (z, y, x) in seeds:
        n = normal_at(z, y, x)
        p0 = np.array([z*dz, y*dy, x*dx])
        p1 = p0 + length * n
        lo = np.minimum(p0, p1) - radius
        hi = np.maximum(p0, p1) + radius
        z0, z1 = max(int(lo[0]/dz), 0), min(int(hi[0]/dz)+1, nz)
        y0, y1 = max(int(lo[1]/dy), 0), min(int(hi[1]/dy)+1, ny)
        x0, x1 = max(int(lo[2]/dx), 0), min(int(hi[2]/dx)+1, nx)
        if z0 >= z1 or y0 >= y1 or x0 >= x1:
            continue
        zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1]
        P = np.stack([zz*dz, yy*dy, xx*dx], axis=-1)
        AB = p1 - p0
        L2 = float(AB @ AB)
        s = np.clip(((P - p0) @ AB) / max(L2, 1e-6), 0, 1)
        proj = p0 + s[..., None]*AB
        dist = np.linalg.norm(P - proj, axis=-1)
        r_s = radius * (tip + (1.0 - tip) * s)     # taper: narrow tip (s=0)->base
        inside = dist < r_s
        body[z0:z1, y0:y1, x0:x1] |= inside
        if base is not None:
            base[z0:z1, y0:y1, x0:x1] |= inside & (s >= 1 - base_frac)
    return body, base


def _lateral_basis(n):
    """Two orthonormal directions perpendicular to axis n (for lateral bending)."""
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, a); u /= np.linalg.norm(u) + 1e-9
    v = np.cross(n, u); v /= np.linalg.norm(v) + 1e-9
    return u, v


def _villus_nodes(p0, n, amp=0.0, tdir=None):
    """Axis polyline (K nodes) + per-node radius for one villus. Tip at p0 (luminal
    surface, s=0), base VILLUS_LEN deep (s=1), continuing to s=VILLUS_OVERSHOOT so the
    clip to villus_zone -- not the axis length -- decides where the base ends. If amp>0
    the base bends by up to `amp` um along `tdir` (grows as s^1.6 so the tip stays put)
    -> a gently curved finger. Taper and bend are both frozen past s=1 so the overshoot
    only extends the finger, it does not fatten or further deflect it."""
    s = np.linspace(0, VILLUS_OVERSHOOT, VILLUS_CURVE_K)
    sc = np.minimum(s, 1.0)
    centers = p0[None, :] + s[:, None] * (VILLUS_LEN * n)[None, :]
    if amp > 0 and tdir is not None:
        centers = centers + (amp * (sc ** 1.6))[:, None] * tdir[None, :]
    radii = VILLUS_R * (VILLUS_TIP + (1.0 - VILLUS_TIP) * sc)
    return centers, radii


def _villus_nodes_base(p0, n_out, L, amp=0.0, tdir=None, rscale=1.0):
    """Axis polyline for a BASE-ANCHORED villus: base at p0 (on the crypt line, so the
    finger is attached to the deep mucosa by construction) and tip L um OUTWARD along
    n_out toward the lumen. `L` is the local villus-zone depth, already clamped.

    The taper is inverted with respect to _villus_nodes: radius VILLUS_R at the base
    (s=0) narrowing to VILLUS_R*VILLUS_TIP at the tip (s=1) -- the same solid, built
    from the other end. Bend still grows as s^1.6, so here it is the TIP that deflects
    while the anchored base stays put. The axis runs to VILLUS_OVERSHOOT so the clip
    to villus_zone, not the axis length, decides where the tip ends."""
    s = np.linspace(0.0, VILLUS_OVERSHOOT, VILLUS_CURVE_K)
    sc = np.minimum(s, 1.0)
    centers = p0[None, :] + s[:, None] * (L * n_out)[None, :]
    if amp > 0 and tdir is not None:
        centers = centers + (amp * (sc ** 1.6))[:, None] * tdir[None, :]
    radii = (VILLUS_R * rscale) * (1.0 - (1.0 - VILLUS_TIP) * sc)
    return centers, radii


def _raster_tube(shape, spc, nodes, radii):
    """Rasterize a tapered tube along the polyline `nodes` (um) with node radii `radii`
    (um) into a LOCAL bounding box. Returns (mask, (z0,z1,y0,y1,x0,x1)) or None."""
    dz, dy, dx = spc; nz, ny, nx = shape
    lo = nodes.min(0) - radii.max(); hi = nodes.max(0) + radii.max()
    z0, z1 = max(int(lo[0]/dz), 0), min(int(hi[0]/dz)+1, nz)
    y0, y1 = max(int(lo[1]/dy), 0), min(int(hi[1]/dy)+1, ny)
    x0, x1 = max(int(lo[2]/dx), 0), min(int(hi[2]/dx)+1, nx)
    if z0 >= z1 or y0 >= y1 or x0 >= x1:
        return None
    zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1]
    P = np.stack([zz*dz, yy*dy, xx*dx], axis=-1)
    inside = np.zeros(P.shape[:3], bool)
    for k in range(len(nodes) - 1):
        A = nodes[k]; B = nodes[k+1]; AB = B - A; L2 = float(AB @ AB)
        s = np.clip(((P - A) @ AB) / max(L2, 1e-6), 0, 1)
        dist = np.linalg.norm(P - (A + s[..., None]*AB), axis=-1)
        r = radii[k] + (radii[k+1] - radii[k]) * s      # taper along the segment
        inside |= dist < r
    return inside, (z0, z1, y0, y1, x0, x1)


def place_crypts_no_overlap(shape, seeds, normal_at, crypt_zone, spc, rng,
                            len_at, radius, outward=True):
    """Greedy no-overlap crypt placement, mirroring place_villi_no_overlap().

    Crypts are straight constant-radius tubes (measured outer diameter 41.4 um, no
    taper). Each is grown from its seed for the LOCAL crypt-zone depth, clamped to the
    measured range, then clipped to crypt_zone -- so the clip, not the axis length,
    decides where it ends. A candidate is rejected if, dilated by CRYPT_GAP_ITER, it
    touches an already-placed crypt: that guarantees a real interglandular gap and
    stops the merging that produced 656x-oversized blobs.
    """
    placed = np.zeros(shape, bool)
    stem_placed = np.zeros(shape, bool)
    struct = np.ones((3, 3, 3), bool)
    n_ok = n_col = n_short = 0
    lengths = []
    for idx in rng.permutation(len(seeds)):
        z, y, x = seeds[idx]
        n = normal_at(z, y, x)
        d = -n if outward else n
        L = len_at(z, y, x)
        if L < CRYPT_LEN_MIN:
            n_short += 1
            continue
        L = min(L, CRYPT_LEN_MAX)
        p0 = np.array([z*spc[0], y*spc[1], x*spc[2]])
        k = max(int(VILLUS_CURVE_K), 2)
        sv = np.linspace(0.0, 1.0, k)
        nodes = p0[None, :] + sv[:, None] * (L * d)[None, :]
        # Same isotropic scaling the villi use: a 60 um crypt at the full 41.4 um
        # measured diameter is a sphere, not a gland. Scaled DOWN only, so no crypt is
        # ever wider than measured.
        rs = min(1.0, L / CRYPT_LEN) if CRYPT_ASPECT_SCALE else 1.0
        radii = np.full(k, radius * rs, float)
        res = _raster_tube(shape, spc, nodes, radii)
        if res is None:
            n_short += 1
            continue
        body, (z0, z1, y0, y1, x0, x1) = res
        body &= crypt_zone[z0:z1, y0:y1, x0:x1]
        if not body.any():
            n_short += 1
            continue
        dil = ndi.binary_dilation(body, struct, iterations=CRYPT_GAP_ITER)
        if (dil & placed[z0:z1, y0:y1, x0:x1]).any():
            n_col += 1
            continue
        placed[z0:z1, y0:y1, x0:x1] |= body
        n_ok += 1
        lengths.append(L)
        if STEM_MODE == "axis":
            # the niche: first STEM_LEN_UM along THIS crypt's axis, from its base
            Ls = min(STEM_LEN_UM, L)
            ns = p0[None, :] + sv[:, None] * (Ls * d)[None, :]
            rres = _raster_tube(shape, spc, ns, np.full(k, radius * rs, float))
            if rres is not None:
                sb, (a0, a1, b0, b1, c0, c1) = rres
                sb &= crypt_zone[a0:a1, b0:b1, c0:c1]
                stem_placed[a0:a1, b0:b1, c0:c1] |= sb
    return placed, stem_placed, {"placed": n_ok, "collide": n_col,
                                 "short": n_short,
                                 "lengths": np.asarray(lengths, float)}


def place_villi_no_overlap(shape, seeds, normal_at, villus_zone, deep, spc, rng,
                           anchor="surface", len_at=None):
    """Greedily place villi so NO two share or touch a voxel AND every villus is
    ANCHORED to the deep mucosa. For each candidate seed: rasterize a tapered finger
    along the local normal (overshooting, then clipped to villus_zone); accept it only if

      (a) dilated by VILLUS_GAP_ITER it reaches no already-placed villus  -> a real
          intervillous groove, so fingers stay distinct; and
      (b) it TOUCHES `deep` (the solid mucosa below the villus zone)      -> it is part
          of the wall, not a floating island.

    (b) is the fix for the 302 detached components. It subsumes both causes measured:
    a villus seeded near a z-face whose normal points out of the slab is truncated by
    the array bound and never reaches `deep` (265 of 302 components), and a villus that
    falls short over a concave patch likewise fails (37 of 302). Either way the
    candidate is bent and retried, then rejected -- leaving bare mucosa rather than
    orphan tissue. Nothing is invented to fill the gap.

    If the straight finger fails, try bending its base into free space
    (VILLUS_CURVE_AMPS x VILLUS_CURVE_DIRS) before giving up.
    Returns (placed_mask, stats)."""
    placed = np.zeros(shape, bool)
    struct = np.ones((3, 3, 3), bool)
    n_straight = n_curved = 0
    n_rej_collide = n_rej_nofloor = n_rej_empty = 0
    # DIAGNOSTIC (2026-08-24): of the `nofloor` rejections, how many were CLIPPED by
    # the slab's z-faces rather than genuinely floating?  Reporting only -- placement
    # is unchanged.  See the note on the anchor test below.
    n_nofloor_zface = 0
    last_zface = [False]          # set by the most recent attempt() that said nofloor
    acc_idx = []                  # seed indices actually placed (for interior density)
    order = rng.permutation(len(seeds))

    def attempt(nodes, radii):
        """-> (body, bbox) | 'collide' | 'nofloor' | None"""
        res = _raster_tube(shape, spc, nodes, radii)
        if res is None:
            return None
        body, (z0, z1, y0, y1, x0, x1) = res
        body &= villus_zone[z0:z1, y0:y1, x0:x1]      # only tissue that becomes villus
        if not body.any():
            return None
        # the overshoot can clip into a second blob where the wall curves away; keep
        # only the piece continuous with the finger itself (largest component).
        lab, nl = ndi.label(body, structure=struct)
        if nl > 1:
            cnt = np.bincount(lab.ravel())[1:]
            body = lab == (int(np.argmax(cnt)) + 1)
        dil = ndi.binary_dilation(body, struct, iterations=VILLUS_GAP_ITER)
        if (dil & placed[z0:z1, y0:y1, x0:x1]).any():
            return "collide"
        # ANCHOR TEST: the base must reach the solid mucosa under the villus zone.
        if not (ndi.binary_dilation(body, struct) &
                deep[z0:z1, y0:y1, x0:x1]).any():
            # _raster_tube CLAMPS its bbox to the array, so a villus whose base runs
            # past z=0 / z=nz is truncated there and can never reach `deep`.  That is
            # the 1 mm slab running out, not the mucosa -- flag it so the two causes
            # can be told apart in the log.
            last_zface[0] = (z0 == 0 or z1 == shape[0])
            return "nofloor"
        return body, (z0, z1, y0, y1, x0, x1)

    def commit(r):
        body, (z0, z1, y0, y1, x0, x1) = r
        placed[z0:z1, y0:y1, x0:x1] |= body

    use_base = (anchor == "base")
    lengths = []
    for idx in order:
        z, y, x = seeds[idx]
        n = normal_at(z, y, x)
        p0 = np.array([z*spc[0], y*spc[1], x*spc[2]])
        if use_base:
            L = len_at(z, y, x)
            if L < VILLUS_LEN_MIN:      # local zone too shallow for a real villus
                n_rej_empty += 1
                continue
            L = min(L, VILLUS_LEN_MAX)
            _rs = min(1.0, L / VILLUS_LEN) if VILLUS_ASPECT_SCALE else 1.0
            mk = lambda amp=0.0, td=None, _p=p0, _n=n, _L=L, _r=_rs:                 _villus_nodes_base(_p, -_n, _L, amp, td, _r)
        else:
            mk = lambda amp=0.0, td=None, _p=p0, _n=n: _villus_nodes(_p, _n, amp, td)
        last_zface[0] = False
        r = attempt(*mk())
        straight_zface = last_zface[0]      # the curve retries overwrite the flag
        if isinstance(r, tuple):
            commit(r); n_straight += 1; acc_idx.append(idx)
            if use_base: lengths.append(L)
            continue
        if r is None:
            n_rej_empty += 1; continue
        u, v = _lateral_basis(n); done = False
        for amp in VILLUS_CURVE_AMPS:
            for k in range(VILLUS_CURVE_DIRS):
                ang = 2*np.pi*k/VILLUS_CURVE_DIRS
                tdir = np.cos(ang)*u + np.sin(ang)*v
                r2 = attempt(*mk(amp, tdir))
                if isinstance(r2, tuple):
                    commit(r2); n_curved += 1; acc_idx.append(idx)
                    if use_base: lengths.append(L)
                    done = True; break
            if done:
                break
        if not done:
            if r == "nofloor":
                n_rej_nofloor += 1
                if straight_zface:
                    n_nofloor_zface += 1
            else:
                n_rej_collide += 1
    return placed, {"accepted": n_straight + n_curved, "straight": n_straight,
                    "curved": n_curved,
                    "rejected": n_rej_collide + n_rej_nofloor + n_rej_empty,
                    "rej_collide": n_rej_collide, "rej_nofloor": n_rej_nofloor,
                    "rej_empty": n_rej_empty,
                    "rej_nofloor_zface": n_nofloor_zface,
                    "accepted_seeds": np.asarray(acc_idx, dtype=np.int64),
                    "lengths": np.asarray(lengths, dtype=float)}


def main():
    d = np.load(IN_NPZ)
    vol = d["volume"].copy()
    # FAST TEST: crop IN-PLANE only. A z-crop would be useless -- the z-face dead
    # band is 425 um (85 slices) at each face, so a short z stack is all dead band.
    # Cropping x keeps the full 200-slice z extent and its edge behaviour intact.
    _sx = os.environ.get("SUBSET_X")
    if _sx:
        _x0, _x1 = (int(v) for v in _sx.split(","))
        vol = vol[:, :, _x0:_x1].copy()
        print(f"SUBSET_X={_sx}: cropped to {vol.shape} "
              f"(relative comparison only -- the crop adds its own x-face losses)")
    dz, dy, dx = SPACING
    st = ndi.generate_binary_structure(3, 1)
    nzv, nyv, nxv = vol.shape
    print(f"volume {vol.shape}, spacing {SPACING} um")

    # Denominators, measured on the GROSS mucosal surface -- i.e. BEFORE any villus
    # is carved, since the measured 33.4/mm^2 counts villi on the gross surface.
    A_NAIVE, A_FACE, A_REF = surface_areas(vol, SPACING, REF_SIGMA_UM)
    print(f"mucosal surface: naive {A_NAIVE:.2f} | face-count {A_FACE:.2f} | "
          f"reference(sigma {REF_SIGMA_UM:.0f} um) {A_REF:.2f} mm^2")

    # ---- 1) EXPAND mucosa inward to TARGET (band within TARGET of submucosa) ----
    # TRUE 3D distance, not a per-slice 2D one. The loop is sectioned OBLIQUELY --
    # the lumen centroid drifts 0.738 um per um of z, i.e. the tube axis sits 36.4 deg
    # off the z-axis (measured; residual scatter about that drift is only 28.6 um, so
    # it is a systematic tilt, not slice jitter). For a wall tilted like that, an
    # in-plane distance overstates the true perpendicular depth by 1/|n_xy|, measured
    # median 1.194x. Under the old per-slice EDT "670 um" of mucosa was really 561 um,
    # the 400 um villus zone was 335 um, and the 270 um crypt zone was 226 um -- all
    # ~16% shallower than the literature values this model is documented against, which
    # scales straight into the stem-cell dose. Doing it in 3D makes the numbers mean
    # what they say. (float32: the float64 EDT alone is 7.5 GB on this volume.)
    submuc = vol == SUBMUC
    d_sub = edt3(~submuc, SPACING)
    del submuc
    inner = np.isin(vol, (LUMEN, MUCOSA))
    if EXPAND_MUCOSA:
        new_muc = inner & (d_sub < TARGET_MUCOSA_UM)
        grew = int((new_muc & (vol == LUMEN)).sum())
        vol[new_muc] = MUCOSA
        del new_muc
        print(f"expanded mucosa inward: +{grew:,} vox from lumen "
              f"({grew*dz*dy*dx/1e9:.2f} mm^3); mucosa now ~{TARGET_MUCOSA_UM:.0f} um "
              f"deep (TRUE perpendicular)")
    else:
        print(f"mucosa expansion SKIPPED (EXPAND_MUCOSA=False) -- keeping the "
              f"segmented mucosa; {int((vol == LUMEN).sum()):,} lumen vox preserved")
    del inner

    # ---- 2) depth-from-lumen + local inward normal ----
    lumen = vol == LUMEN
    lum_ds = lumen[::DS_N, ::DS_N, ::DS_N]
    d_ds = ndi.distance_transform_edt(~lum_ds, sampling=(dz*DS_N, dy*DS_N, dx*DS_N))
    d_len = d_ds.copy()          # UNSMOOTHED depth field, kept for diagnostics.
                                 # The smoothed copy below is for normal DIRECTIONS,
                                 # where smoothing is wanted; smoothing the magnitude
                                 # would bias every villus length.
    d_ds = ndi.gaussian_filter(d_ds, NORMAL_SMOOTH)   # smooth -> perpendicular normals
    gz, gy, gx = np.gradient(d_ds, dz*DS_N, dy*DS_N, dx*DS_N)
    gm = np.sqrt(gz*gz + gy*gy + gx*gx) + 1e-6
    NZ, NY, NX = gz/gm, gy/gm, gx/gm
    fz, fy, fx = d_ds.shape

    def normal_at(z, y, x):
        jz, jy, jx = min(z//DS_N, fz-1), min(y//DS_N, fy-1), min(x//DS_N, fx-1)
        return np.array([NZ[jz, jy, jx], NY[jz, jy, jx], NX[jz, jy, jx]])

    _LSTEP = 0.9 * min(dz, dy, dx)
    _LLIM = VILLUS_LEN_MAX * 1.15
    _nzS, _nyS, _nxS = vol.shape

    def len_at(z, y, x):
        """Villus-zone depth available ALONG THE GROWTH DIRECTION.

        NOT the distance to the nearest lumen. In a pleated mucosa the nearest
        lumen is usually sideways -- across the fold rather than up it -- so a
        nearest-distance lookup under-reads the depth a villus can actually use
        and rejects seeds that had plenty of room. Measured cost of getting this
        wrong: 662 of 1,076 crypt-line seeds rejected as too shallow against a
        surface that is 64% usable. So march outward along the same normal the
        villus is built on, and stop where the mucosa stops."""
        n_out = -normal_at(z, y, x)
        pz, py, px = z * dz, y * dy, x * dx
        L = 0.0
        while L < _LLIM:
            L += _LSTEP
            iz = int(round((pz + L * n_out[0]) / dz))
            iy = int(round((py + L * n_out[1]) / dy))
            ix = int(round((px + L * n_out[2]) / dx))
            if not (0 <= iz < _nzS and 0 <= iy < _nyS and 0 <= ix < _nxS):
                break
            if vol[iz, iy, ix] != MUCOSA:
                break
        return L - _LSTEP

    rng = np.random.default_rng(SEED)

    # ---- 3) VILLI: collision-free placement, carve the intervillous grooves ----
    # Villi are the KEPT fingers; the rest of the villus zone is carved to lumen.
    # Placement guarantees no two villi share/touch a voxel (a real groove between
    # them), so they stay DISTINCT -- min-sep seeding alone did not (48% overlapped).
    #
    # BOTH zones are now keyed off distance-from-SUBMUCOSA, so they are exactly
    # complementary within the mucosa. Previously villus_zone used distance-from-LUMEN
    # while the crypt zone used distance-from-submucosa; two different metrics for two
    # abutting bands is what let a villus base stop short of the deep mucosa with a
    # sliver of villus_zone left over -- and that sliver, being inside villus_zone and
    # part of no villus, was then carved to lumen, orphaning the finger.
    CRYPT_ZONE = TARGET_MUCOSA_UM - VILLUS_LEN          # ~390 um, absolute-mode only
    _muc = vol == MUCOSA
    if VILLUS_ZONE_MODE == "proportional":
        d_lum_full = edt3(~lumen, SPACING)
        t_rad = d_sub / (d_sub + d_lum_full + 1e-6)     # 0 submucosa -> 1 lumen
        del d_lum_full
        villus_zone = _muc & (t_rad > CRYPT_FRAC)
        deep = _muc & (t_rad <= CRYPT_FRAC)
        _tsurf = np.abs(t_rad - CRYPT_FRAC) < 0.02
        print(f"  zones: PROPORTIONAL, crypt fraction {CRYPT_FRAC:.4f} of local "
              f"mucosal depth (measured 390.0 : 420.9); crypt zone "
              f"{deep.sum()/max(_muc.sum(),1)*100:.1f}% of mucosa by volume")
    else:
        t_rad = None
        villus_zone = _muc & (d_sub > CRYPT_ZONE)
        deep = _muc & (d_sub <= CRYPT_ZONE)
        _tsurf = np.abs(d_sub - CRYPT_ZONE) < dy
        print(f"  zones: ABSOLUTE, crypt zone fixed at {CRYPT_ZONE:.0f} um; "
              f"{deep.sum()/max(_muc.sum(),1)*100:.1f}% of mucosa by volume")
    iface = (vol == MUCOSA) & ndi.binary_dilation(lumen, st)
    # "base" mode seeds on the CRYPT LINE, the same surface the crypts use, and grows
    # outward; "surface" mode seeds on the luminal interface and grows inward.
    if VILLUS_ANCHOR == "base":
        seed_surf = _muc & _tsurf
        print(f"  anchoring: BASE (crypt line, grows outward); length = local zone "
              f"depth clamped to [{VILLUS_LEN_MIN:.0f}, {VILLUS_LEN_MAX:.0f}] um "
              f"(measured range, n=25)")
    else:
        seed_surf = iface
        print(f"  anchoring: SURFACE (legacy, fixed {VILLUS_LEN:.0f} um inward)")
    if VILLUS_SEED_MODE == "lattice":
        # count against the REFERENCE surface -- the scale the target was measured at
        _ssurf = A_REF if np.isfinite(A_REF) and A_REF > 0 else A_FACE
        _n = int(round(VILLUS_DENSITY_PER_MM2 * _ssurf))
        vseeds = seeds_lattice(seed_surf, _n, dz, dy, dx, rng)
        print(f"  seeding: QUASI-LATTICE, {VILLUS_DENSITY_PER_MM2} per mm^2 x "
              f"{_ssurf:.2f} mm^2 = {_n:,} asked -> {len(vseeds):,} candidates "
              f"({VILLUS_LATTICE_RELAX} Lloyd iterations)")
    else:
        vseeds = seeds_on(seed_surf, VILLUS_SPACING, dz, dy, dx, rng)
        print(f"  seeding: POISSON-DISK, min separation {VILLUS_SPACING:.0f} um "
              f"-> {len(vseeds):,} candidates")
    # The nearest-neighbour SPREAD is what separates the two modes: Poisson-disk piles
    # up at its floor, a relaxed lattice bunches at the mean. Pairs near 2*VILLUS_R are
    # the ones whose bases touch, so this predicts the collision count before placement.
    if len(vseeds) > 1:
        from scipy.spatial import cKDTree as _KD
        _sp = np.stack([vseeds[:, 0]*dz, vseeds[:, 1]*dy, vseeds[:, 2]*dx], axis=1)
        _nn = _KD(_sp).query(_sp, k=2, workers=-1)[0][:, 1]
        _tight = int((_nn < 2*VILLUS_R).sum())
        print(f"  seed spacing: min {_nn.min():.0f} / p05 {np.percentile(_nn,5):.0f} / "
              f"median {np.median(_nn):.0f} / max {_nn.max():.0f} um; "
              f"{_tight:,} of {len(_nn):,} closer than one base width "
              f"({2*VILLUS_R:.0f} um)")
    vkeep, vstats = place_villi_no_overlap(vol.shape, vseeds, normal_at,
                                           villus_zone, deep, SPACING, rng,
                                           anchor=VILLUS_ANCHOR, len_at=len_at)
    # Carve the grooves ONLY inside the villous field. Carving all of villus_zone would
    # turn every rejected patch into a 400 um trench down to the crypt zone; restricting
    # it to CARVE_REACH of a placed villus leaves those patches as smooth solid mucosa.
    d_vil = edt3(~vkeep, SPACING)
    # MEMORY (2026-08-26): the original built `villus_zone & ~vkeep` twice and
    # allocated ~4 full-size bool temporaries (895 MiB each) in one expression,
    # which overran the commit limit. `carve` and `bare` partition the SAME set
    # -- every voxel of vz_not_keep is either <= or > CARVE_REACH -- so `bare`
    # is the complement of `carve` within it and needs no second pass.
    vz_not_keep = villus_zone & ~vkeep
    carve = vz_not_keep & (d_vil <= CARVE_REACH)
    bare = int(vz_not_keep.sum()) - int(carve.sum())
    del vz_not_keep, d_vil
    vol[carve] = LUMEN
    # Density is reported on the REFERENCE surface: that is the scale the 33.4/mm^2
    # target was measured at. The naive figure is kept only to relate new runs to the
    # old logs -- it understates density by A_NAIVE/A_REF (~3.7x here).
    surf_naive = iface.sum() * dy * dx / 1e6
    surf_mm2 = A_REF if np.isfinite(A_REF) and A_REF > 0 else A_FACE
    print(f"  villus-free mucosa left SOLID (not carved): {bare:,} vox "
          f"({bare*dz*dy*dx/1e9:.3f} mm^3)")
    print(f"villi: {len(vseeds):,} candidates -> {vstats['accepted']:,} placed "
          f"({vstats['straight']:,} straight + {vstats['curved']:,} curved, "
          f"{vstats['rejected']:,} rejected: {vstats['rej_collide']:,} collide / "
          f"{vstats['rej_nofloor']:,} unanchored / {vstats['rej_empty']:,} empty); "
          f"~{vstats['accepted']/max(surf_mm2,1e-9):.1f}"
          f"/mm^2 on the reference surface; carved {int(carve.sum()):,} vox -> lumen")
    print(f"  density by denominator: reference(sigma {REF_SIGMA_UM:.0f}) "
          f"{vstats['accepted']/max(A_REF,1e-9):.1f}/mm^2 | face-count "
          f"{vstats['accepted']/max(A_FACE,1e-9):.1f} | naive "
          f"{vstats['accepted']/max(surf_naive,1e-9):.1f}   "
          f"[target {VILLUS_DENSITY_PER_MM2:.1f}/mm^2]")
    # SCALE-FREE CHECK. Areal density depends on the surface scale; nearest-neighbour
    # SPACING does not, and the male target was itself derived from a measured spacing
    # of 173.1 um. This is the number to quote when comparing model to histology.
    _len = vstats.get("lengths", np.empty(0))
    if len(_len):
        print(f"  villus length (variable, clamped to the measured range): "
              f"median {np.median(_len):.1f} mean {_len.mean():.1f} sd {_len.std(ddof=1):.1f} um; "
              f"min {_len.min():.0f} max {_len.max():.0f}; "
              f"{100.0*(_len >= VILLUS_LEN_MAX - 1e-6).mean():.0f}% at the {VILLUS_LEN_MAX:.0f} um cap")
        print(f"    configured for this model: VILLUS_LEN {VILLUS_LEN:.1f} um, "
              f"clamped to [{VILLUS_LEN_MIN:.0f}, {VILLUS_LEN_MAX:.0f}]. In PROPORTIONAL "
              f"mode each villus gets {1.0-CRYPT_FRAC:.3f} of LOCAL mucosal depth, so "
              f"the realised median sits below VILLUS_LEN wherever local mucosa is "
              f"thinner than the fields the morphometry was measured on.")
    _acc_i = vstats["accepted_seeds"]
    if len(_acc_i) > 1:
        from scipy.spatial import cKDTree
        _pts = np.asarray(vseeds, float)[_acc_i] * np.array([dz, dy, dx])
        _dd, _ = cKDTree(_pts).query(_pts, k=2)
        _nn = _dd[:, 1]
        print(f"  nearest-neighbour villus spacing: median {np.median(_nn):.1f} um "
              f"(mean {_nn.mean():.1f}, p25 {np.percentile(_nn,25):.1f}, "
              f"p75 {np.percentile(_nn,75):.1f})  "
              f"[measured {1e3/VILLUS_DENSITY_PER_MM2**0.5:.1f} um]")
        print(f"    -> spacing-implied density {1e6/np.median(_nn)**2:.1f}/mm^2 "
              f"[measured-spacing-implied {VILLUS_DENSITY_PER_MM2:.1f}/mm^2]")
    # --- DIAGNOSTIC (2026-08-24): how much of the shortfall is the slab EDGE? -----
    # A villus must traverse the whole villus zone (~VILLUS_LEN um along its normal)
    # to touch `deep`.  Near the z-faces that traverse runs out of ARRAY, not of
    # tissue: _raster_tube clamps its bbox to the bound, the body is clipped, and the
    # candidate is scored `nofloor`.  Those bands therefore carry bare mucosa BY
    # CONSTRUCTION -- yet surf_mm2 counts them in the denominator, so the whole-slab
    # density is biased low against an interior histology field, which is exactly what
    # the measured 19.9 per mm^2 is.  Report both; placement itself is unchanged.
    nz_max = 0.0
    for _s in vseeds:
        nz_max = max(nz_max, abs(float(normal_at(*_s)[0])))
    z_margin = int(np.ceil(VILLUS_LEN * nz_max / dz))
    zlo, zhi = z_margin, vol.shape[0] - z_margin
    _nzf = int(vstats.get("rej_nofloor_zface", 0))
    _nnf = int(vstats["rej_nofloor"])
    _pct = (100.0 * _nzf / _nnf) if _nnf else 0.0
    print(f"  z-face clip: {_nzf:,} of {_nnf:,} unanchored rejections were CLIPPED "
          f"by the slab boundary ({_pct:.0f}%); max |n_z| = {nz_max:.3f} -> dead "
          f"band {z_margin} slices ({z_margin*dz:.0f} um) at each z-face")
    if zhi > zlo:
        _vs = np.asarray(vseeds)
        _acc = vstats["accepted_seeds"]
        _zc = _vs[_acc][:, 0] if len(_acc) else np.empty(0)
        n_core = int(((_zc >= zlo) & (_zc < zhi)).sum())
        # same reference rescaling as the whole-slab figure
        _scale = (surf_mm2 / surf_naive) if surf_naive > 0 else 1.0
        surf_core = float(iface[zlo:zhi].sum()) * dy * dx / 1e6 * _scale
        print(f"  interior-only (z {zlo}-{zhi}): {n_core:,} villi over "
              f"{surf_core:.2f} mm^2 = {n_core/max(surf_core,1e-9):.1f} per mm^2   "
              f"[whole slab {vstats['accepted']:,} over {surf_mm2:.2f} mm^2 = "
              f"{vstats['accepted']/max(surf_mm2,1e-9):.1f} per mm^2; "
              f"measured target {VILLUS_DENSITY_PER_MM2:.1f} per mm^2]")
    else:
        print(f"  interior-only: SKIPPED -- dead band {z_margin} slices exceeds half "
              f"the slab ({vol.shape[0]} slices)")
    # Whatever is left unattached after carving is -- by the carve rule's own
    # definition -- villus-zone mucosa belonging to no villus, so carve it too. In
    # practice this is the free mucosal flap: original segmented mucosa sitting
    # 630-985 um from any submucosa, i.e. with no deep wall under it to anchor to.
    # GUARDED: anything reaching the deep/crypt zone is real wall, and deleting that
    # silently is exactly the kind of thing that should never happen quietly -- so it
    # aborts instead.
    tis = vol == MUCOSA                       # crypts/stem do not exist yet
    lb, nl = ndi.label(tis, structure=np.ones((3, 3, 3), np.uint8))
    if nl > 1:
        # MEMORY (2026-08-26): ndi.sum() -> scipy _stats -> np.bincount(lb.ravel()),
        # which casts all 938M voxels to int64 in one 7 GiB allocation and blows the
        # commit limit. lb labels `tis` itself, so every labelled voxel IS tissue and
        # the boolean sum per label is just the component size. Chunked bincount over
        # z gives the identical vector at ~1/25th the peak.
        _cnt = np.zeros(nl + 1, dtype=np.int64)
        for _z0 in range(0, lb.shape[0], 8):
            _cnt += np.bincount(lb[_z0:_z0 + 8].ravel(), minlength=nl + 1)
        ss = _cnt[1:]
        del _cnt
        stray = tis & (lb != int(np.argmax(ss)) + 1)
        # Use the SAME deep mask the zones use, so this tracks proportional mode too
        # (it used to hard-code d_sub <= CRYPT_ZONE, which is only correct in absolute
        # mode). Deep strays are real wall: never carve them to LUMEN -- that would
        # punch lumen cavities inside the wall, and lumen carries ICRP GI contents.
        deep_stray = stray & deep
        deep_hit = int(deep_stray.sum())
        if deep_hit > GUARD_MAX_DEEP_VOX:
            raise SystemExit(
                f"ABORT: {nl-1} detached component(s) reach the deep zone "
                f"({deep_hit:,} vox > tolerance {GUARD_MAX_DEEP_VOX:,}). That is real "
                f"wall, not a carve remnant -- do not delete it, fix the cause.")
        if deep_hit:
            # Below tolerance this is the carve nicking a few voxels off the wall, not
            # a structural failure. Leave them AS MUCOSA -- attached tissue is the safe
            # error, a lumen cavity in the wall is not -- and say so loudly.
            print(f"  WARNING: {deep_hit:,} vox of detached DEEP wall left as mucosa "
                  f"(under the {GUARD_MAX_DEEP_VOX:,} vox tolerance; "
                  f"{deep_hit*dz*dy*dx/1e9:.6f} mm^3). Not carved.")
        shallow = stray & ~deep
        vol[shallow] = LUMEN
        print(f"  removed {nl-1} unanchored villus-zone remnant(s): "
              f"{int(shallow.sum()):,} vox ({shallow.sum()*dz*dy*dx/1e9:.4f} mm^3)")
        del stray, deep_stray, shallow
    del villus_zone, deep, vkeep, carve, iface, tis, lb

    # ---- 4) CRYPTS in the DEEP zone + STEM CELLS at the base ----
    # zones defined by distance-from-submucosa (clean; unaffected by villi carving).
    # crypt zone = deep mucosa (d_sub < CRYPT_ZONE); villus zone is the rest.
    # crypt openings seed at the top of the crypt zone (= the villus bases)
    if VILLUS_ZONE_MODE == "proportional":
        crypt_zone = (vol == MUCOSA) & (t_rad <= CRYPT_FRAC) & (d_sub > BASE_MARGIN)
    else:
        crypt_zone = (vol == MUCOSA) & (d_sub <= CRYPT_ZONE) & (d_sub > BASE_MARGIN)

    if CRYPT_ANCHOR == "base":
        # Seed on the MUSCULARIS MUCOSAE -- the surface CRYPT_SPACING was measured
        # along, and where the stem cells sit -- and grow outward to the crypt line.
        base_surf = crypt_zone & (d_sub < BASE_MARGIN + 2.0 * dy)
        _outward = True
        print(f"  crypts: BASE-anchored (muscularis mucosae, grows outward); "
              f"spacing {CRYPT_SPACING:.1f} um measured on THIS surface")
    else:
        base_surf = (vol == MUCOSA) & _tsurf
        _outward = False
        print(f"  crypts: crypt-line anchored (legacy, grows inward)")
    if CRYPT_SEED_MODE == "lattice":
        _n = len(seeds_on(base_surf, CRYPT_SPACING, dz, dy, dx, rng))
        cseeds = seeds_lattice(base_surf, _n, dz, dy, dx, rng)
        print(f"    QUASI-LATTICE: {_n:,} asked (this surface's natural capacity at "
              f"{CRYPT_SPACING:.1f} um) -> {len(cseeds):,} candidates "
              f"({VILLUS_LATTICE_RELAX} Lloyd iterations, regularised distribution "
              f"instead of a Poisson floor-spike)")
    else:
        _seed_sp = CRYPT_SEED_SPACING if CRYPT_SEED_SPACING > 0 else CRYPT_SPACING
        if _seed_sp != CRYPT_SPACING:
            print(f"    OVER-SEEDING: candidates at {_seed_sp:.1f} um ({(CRYPT_SPACING/_seed_sp)**2:.2f}x density) to "
                  f"compensate for placement rejections; survivors should approach the "
                  f"measured {CRYPT_SPACING:.1f} um")
        cseeds = seeds_on(base_surf, _seed_sp, dz, dy, dx, rng)

    _cLSTEP = 0.9 * min(dz, dy, dx)
    _cLLIM = CRYPT_LEN_MAX * 1.15

    def crypt_len_at(z, y, x):
        """Crypt-zone depth along the growth direction, by ray-march."""
        n = normal_at(z, y, x)
        d = -n if _outward else n
        pz, py, px = z * dz, y * dy, x * dx
        L = 0.0
        while L < _cLLIM:
            L += _cLSTEP
            iz = int(round((pz + L * d[0]) / dz))
            iy = int(round((py + L * d[1]) / dy))
            ix = int(round((px + L * d[2]) / dx))
            if not (0 <= iz < vol.shape[0] and 0 <= iy < vol.shape[1]
                    and 0 <= ix < vol.shape[2]):
                break
            if not crypt_zone[iz, iy, ix]:
                break
        return L - _cLSTEP

    cbody, cstem, cstats = place_crypts_no_overlap(vol.shape, cseeds, normal_at,
                                                   crypt_zone, SPACING, rng,
                                                   crypt_len_at, CRYPT_R,
                                                   outward=_outward)
    _cl = cstats["lengths"]
    print(f"  crypt placement: {len(cseeds):,} candidates -> {cstats['placed']:,} "
          f"placed ({cstats['collide']:,} collide / {cstats['short']:,} too short)")
    if len(_cl):
        print(f"    crypt length: median {np.median(_cl):.1f} mean {_cl.mean():.1f} "
              f"sd {_cl.std(ddof=1):.1f} um   "
              f"[configured CRYPT_LEN {CRYPT_LEN:.1f} um; realised length is the local\n"
              f"     crypt-zone depth, so it sits below CRYPT_LEN wherever the zone is shallow]")
    crypt = cbody & crypt_zone
    vol[crypt] = CRYPT
    # STEM CELLS = crypt base = crypt voxels nearest the submucosa (radiosensitive)
    if STEM_MODE == "axis":
        # subset of CRYPT by construction, but intersect so it is guaranteed
        stem = cstem & (vol == CRYPT)
        print(f"  stem: AXIS mode -- first {STEM_LEN_UM:.0f} um along each crypt axis "
              f"from its base")
    else:
        stem = (vol == CRYPT) & (d_sub < STEM_DEPTH)
        print(f"  stem: DEPTH mode (legacy) -- crypt voxels with d_sub < "
              f"{STEM_DEPTH:.0f} um")
    vol[stem] = STEM
    print(f"crypts: {len(cseeds):,} tubes, {int(crypt.sum()):,} crypt vox (label 6), "
          f"{int(stem.sum()):,} stem-cell vox (label 7)")
    # Verify the niche is COMPACT, not smeared along tangential crypts. A clean basal
    # disc is ~pi*CRYPT_R^2*STEM_LEN and its longest extent is the crypt width.
    _slb, _snl = ndi.label(stem, structure=np.ones((3, 3, 3), np.uint8))
    if _snl:
        _sc = np.zeros(_snl + 1, dtype=np.int64)
        for _z in range(0, _slb.shape[0], 8):
            _sc += np.bincount(_slb[_z:_z + 8].ravel(), minlength=_snl + 1)
        _ss = _sc[1:]
        _ideal = np.pi * CRYPT_R**2 * STEM_LEN_UM / (dz * dy * dx)
        _ex = []
        for _o in ndi.find_objects(_slb):
            if _o is None:
                continue
            _ex.append(max((_o[0].stop - _o[0].start) * dz,
                           (_o[1].stop - _o[1].start) * dy,
                           (_o[2].stop - _o[2].start) * dx))
        _ex = np.asarray(_ex)
        print(f"  stem niches: {_snl:,} component(s); size p50 {np.percentile(_ss,50):.0f} "
              f"vox vs ideal disc {_ideal:.0f} ({_ss.mean()/_ideal:.2f}x mean)")
        print(f"    longest extent p50 {np.percentile(_ex,50):.0f} p90 "
              f"{np.percentile(_ex,90):.0f} max {_ex.max():.0f} um "
              f"(expect ~{2*CRYPT_R:.0f}-{STEM_LEN_UM:.0f}); "
              f"{(_ex>100).sum():,} over 100 um holding "
              f"{_ss[_ex>100].sum()/_ss.sum()*100:.1f}% of stem volume")

    for lab, nm in [(1, "lumen"), (2, "mucosa"), (3, "submucosa"), (4, "muscularis"),
                    (5, "serosa"), (6, "crypt"), (7, "stem cells")]:
        n = int((vol == lab).sum())
        print(f"  label {lab} {nm:11s}: {n:>10,}  ({n*dz*dy*dx/1e9:.3f} mm^3)")

    # ---- 5) SELF-CHECK: no floating tissue ----
    # The mesh QC gate cannot catch this: its "orphaned" test is about unused mesh
    # NODES, not disconnected tissue, so ~40 unattached villi meshed cleanly and
    # passed. Check it here, on the voxels, where it is cheap and unambiguous.
    def _comp_sizes(lab, n):
        """Per-component voxel counts, bounded memory (see the note below)."""
        cnt = np.zeros(n + 1, dtype=np.int64)
        for _z0 in range(0, lab.shape[0], 8):
            cnt += np.bincount(lab[_z0:_z0 + 8].ravel(), minlength=n + 1)
        return cnt[1:]

    print("\n--- self-check: tissue connectivity ---")
    tissue = np.isin(vol, (MUCOSA, CRYPT, STEM))
    lbl, ncomp = ndi.label(tissue, structure=np.ones((3, 3, 3), np.uint8))
    ok = True
    if ncomp == 0:
        print("  FAIL: no tissue"); ok = False
    else:
        # bincount, not ndi.sum: ndi.sum promotes the mask to float64 weights over
        # the WHOLE volume (4.42 GiB on the male ileum's 592M voxels, which is where
        # it died 2026-09-14 -- after all the carving work and before the savez, so
        # the whole run was lost). The mask is the one these labels were made from,
        # so every labelled voxel weighs exactly 1 and the weighted sum IS the
        # component size. bincount returns that directly, in ints, allocating only
        # ncomp+1 entries.
        #
        # ...but COUNT IN Z-SLABS (2026-09-16). np.bincount upcasts its input to
        # intp, so bincount(lbl.ravel()) still allocates 8 bytes per VOXEL -- 6.99
        # GiB on the female jejunum's 938M, which killed that run at exactly the
        # point this comment warns about. The male jejunum's 503M needed 4.0 GiB
        # and survived, which is why the whole-array form looked fine. Slabbing
        # bounds the copy to the slab and gives identical totals. Same pattern as
        # fix_buried_islands.components().
        sz = _comp_sizes(lbl, ncomp)
        tot = sz.sum()
        main = sz.max()
        print(f"  mucosa+crypt+stem: {ncomp:,} connected component(s); "
              f"largest holds {main/tot*100:.3f}% of tissue")
        if ncomp > 1:
            orph = tot - main
            print(f"  {ncomp-1:,} DETACHED component(s), {orph:,.0f} vox "
                  f"({orph/tot*100:.3f}%)  <-- floating tissue, must be 0")
            ok = False
    lum_l, nlum = ndi.label(vol == LUMEN, structure=np.ones((3, 3, 3), np.uint8))
    if nlum:
        ls = _comp_sizes(lum_l, nlum)                # same reason as above
        print(f"  lumen: {nlum:,} component(s); largest holds "
              f"{ls.max()/ls.sum()*100:.3f}% of lumen")
    print("  all checks passed" if ok else "  *** SELF-CHECK FAILED ***")

    np.savez_compressed(OUT_NPZ, volume=vol, zrange=d["zrange"], stride=d["stride"])
    print(f"wrote {OUT_NPZ}  ({os.path.getsize(OUT_NPZ)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
