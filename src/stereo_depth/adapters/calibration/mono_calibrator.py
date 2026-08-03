"""Mono (single-camera) ChArUco intrinsic calibration.

Fits three distortion models on the same view set and selects the best one
fairly (holdout RPE + sanity checks, never train RPE alone):
  - "pinhole":  cv2.calibrateCamera, flags=0            -> D = (k1,k2,p1,p2,k3)
  - "rational": cv2.calibrateCamera, CALIB_RATIONAL_MODEL -> D = (k1..k3,p1,p2,k4..k6) as (8,)
  - "fisheye":  cv2.fisheye.calibrate                    -> D = (k1..k4) equidistant

Also provides the K_new / undistort-map helpers shared by the `undistort`
command. Remember: K/D pair with RAW images, K_new (D=0) pairs with
UNDISTORTED images — never mix.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from stereo_depth.adapters.calibration.charuco_calibrator import (
    CharucoDetection,
    CollectReport,
    detect_charuco,
)

MODELS = ("pinhole", "rational", "fisheye")

_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
_SUBPIX_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

# Fraction of the corner radius the forward projection must remain injective
# over before a turnover counts as real fold-back.
_CORNER_TOLERANCE = 0.95


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

@dataclass
class MonoViewSet:
    """Detections collected from one camera, size-checked per image."""
    corners: list[np.ndarray]          # each (N,1,2) float32 charuco corners
    ids: list[np.ndarray]              # each (N,1) int32 charuco ids
    names: list[str]                   # source image filename per view
    image_size: tuple[int, int]        # (w, h)
    report: CollectReport
    skipped_size_mismatch: int = 0


def collect_charuco_mono_from_paths(
    img_paths: list[Path],
    board,
    dictionary,
    *,
    min_markers: int = 4,
    min_charuco: int = 10,
    refine: bool = True,
) -> MonoViewSet:
    """Detect ChArUco corners in every image; keep only views that pass.

    Unlike collect_charuco_from_paths, every image's size is checked against
    the first readable one — mismatched images are skipped and counted.
    ``refine=True`` runs cv2.cornerSubPix on the charuco corners.
    """
    all_corners: list[np.ndarray] = []
    all_ids: list[np.ndarray] = []
    names: list[str] = []
    img_size: tuple[int, int] | None = None

    stats = {
        "total": 0, "ok": 0,
        "no_markers": 0, "too_few_markers": 0, "too_few_charuco": 0,
        "min_charuco": 10**9, "max_charuco": 0,
    }
    skipped_size = 0

    for p in img_paths:
        stats["total"] += 1
        img = cv2.imread(str(p))
        if img is None:
            stats["no_markers"] += 1
            continue

        h, w = img.shape[:2]
        if img_size is None:
            img_size = (w, h)
        elif (w, h) != img_size:
            skipped_size += 1
            continue

        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        det: CharucoDetection = detect_charuco(
            gray, board, dictionary, min_markers=min_markers, min_charuco=min_charuco
        )

        stats["min_charuco"] = min(stats["min_charuco"], det.num_charuco)
        stats["max_charuco"] = max(stats["max_charuco"], det.num_charuco)

        if not det.ok:
            stats[det.reason or "no_markers"] += 1
            continue

        corners = det.corners
        if refine:
            corners = cv2.cornerSubPix(
                gray, corners.astype(np.float32), (5, 5), (-1, -1), _SUBPIX_CRIT
            )

        all_corners.append(corners)
        all_ids.append(det.ids)
        names.append(p.name)
        stats["ok"] += 1

    if img_size is None:
        img_size = (0, 0)

    report = CollectReport(
        total=stats["total"],
        ok=stats["ok"],
        fail_no_markers=stats["no_markers"],
        fail_too_few_markers=stats["too_few_markers"],
        fail_too_few_charuco=stats["too_few_charuco"],
        min_charuco_seen=(0 if stats["min_charuco"] == 10**9 else stats["min_charuco"]),
        max_charuco_seen=stats["max_charuco"],
    )
    return MonoViewSet(
        corners=all_corners, ids=all_ids, names=names,
        image_size=img_size, report=report,
        skipped_size_mismatch=skipped_size,
    )


# ---------------------------------------------------------------------------
# Point matching (charuco corners -> obj/img point lists)
# ---------------------------------------------------------------------------

def match_object_points(
    corners: list[np.ndarray],
    ids: list[np.ndarray],
    board,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """Per view: (obj_pts (N,1,3) f32, img_pts (N,1,2) f32, kept view indices)."""
    obj_list: list[np.ndarray] = []
    img_list: list[np.ndarray] = []
    kept: list[int] = []
    for i, (c, i_ids) in enumerate(zip(corners, ids)):
        obj, img = board.matchImagePoints(c, i_ids)
        if obj is not None and len(obj) >= 6:
            obj_list.append(obj.astype(np.float32))
            img_list.append(img.astype(np.float32))
            kept.append(i)
    return obj_list, img_list, kept


# ---------------------------------------------------------------------------
# Model fits
# ---------------------------------------------------------------------------

@dataclass
class MonoModelFit:
    model: str                         # "pinhole" | "rational" | "fisheye"
    ok: bool
    K: np.ndarray | None = None
    D: np.ndarray | None = None        # flattened 1-D
    rms_train: float = float("nan")
    per_view_rpe: list[float] = field(default_factory=list)
    rvecs: list = field(default_factory=list)
    tvecs: list = field(default_factory=list)
    n_views: int = 0
    dropped_views: int = 0             # fisheye ill-conditioned drops
    error: str | None = None


def fit_pinhole(
    corners: list[np.ndarray],
    ids: list[np.ndarray],
    board,
    img_size: tuple[int, int],
    *,
    rational: bool = False,
) -> MonoModelFit:
    """Brown-Conrady fit. rational=True adds k4..k6 (CALIB_RATIONAL_MODEL)."""
    model = "rational" if rational else "pinhole"
    obj_list, img_list, _ = match_object_points(corners, ids, board)
    if not obj_list:
        return MonoModelFit(model, False, error="no usable views after matchImagePoints")

    flags = cv2.CALIB_RATIONAL_MODEL if rational else 0
    try:
        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
            obj_list, img_list, img_size, None, None, flags=flags, criteria=_CRIT
        )
    except cv2.error as e:
        return MonoModelFit(model, False, error=str(e))

    D = np.asarray(D, dtype=np.float64).ravel()
    if rational:
        D = D[:8] if D.size >= 8 else np.pad(D, (0, 8 - D.size))
    else:
        D = D[:5] if D.size >= 5 else np.pad(D, (0, 5 - D.size))

    fit = MonoModelFit(
        model, True, K=np.asarray(K, dtype=np.float64), D=D,
        rms_train=float(rms), rvecs=list(rvecs), tvecs=list(tvecs),
        n_views=len(obj_list),
    )
    fit.per_view_rpe = per_view_rpe(fit, obj_list, img_list)
    return fit


_ILL_COND_RE = re.compile(r"input array (\d+)")


def _focal_guess_ladder(img_size: tuple[int, int], k_guess: np.ndarray | None) -> list[float]:
    """Candidate initial focal lengths for cv2.fisheye.calibrate.

    The fisheye extrinsics initialiser (InitExtrinsics) asserts on a
    degenerate homography when the initial focal length is too far above the
    true one — OpenCV's own default (max(W,H)/pi) is not always safe, and
    without CALIB_USE_INTRINSIC_GUESS it fails outright on wide lenses.
    Trying a ladder from short to long makes the fit robust without knowing
    the lens in advance.
    """
    w, h = img_size
    ladder = [max(w, h) / np.pi, 0.30 * w, 0.40 * w, 0.50 * w, 0.65 * w, 0.85 * w, 1.10 * w]
    if k_guess is not None:
        ladder.insert(0, float(k_guess[0, 0]))
    seen: list[float] = []
    for f in ladder:
        if f > 1.0 and not any(abs(f - s) < 1.0 for s in seen):
            seen.append(float(f))
    return seen


def fit_fisheye(
    corners: list[np.ndarray],
    ids: list[np.ndarray],
    board,
    img_size: tuple[int, int],
    *,
    k_guess: np.ndarray | None = None,
    max_drops: int = 5,
) -> MonoModelFit:
    """Equidistant fisheye fit (cv2.fisheye.calibrate, k1..k4).

    cv2.fisheye.calibrate is brittle in two distinct ways, both handled here:
      1. The extrinsics initialiser asserts (``fabs(norm_u1) > 0``) when the
         initial focal length is too long — worked around with an explicit
         CALIB_USE_INTRINSIC_GUESS ladder (see _focal_guess_ladder).
      2. A near-degenerate view raises "Ill-conditioned matrix for input
         array <i>" — that view is dropped and the fit retried.
    ``k_guess`` (e.g. the pinhole fit's K) is tried first when supplied.
    """
    obj_list, img_list, _ = match_object_points(corners, ids, board)
    if not obj_list:
        return MonoModelFit("fisheye", False, error="no usable views after matchImagePoints")

    w, h = img_size
    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        | cv2.fisheye.CALIB_FIX_SKEW
        | cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
    )

    last_error = "fisheye calibration did not converge for any focal guess"
    for f_guess in _focal_guess_ladder(img_size, k_guess):
        # fisheye wants (1,N,3) float64 obj and (1,N,2) float64 img per view
        objs = [o.reshape(1, -1, 3).astype(np.float64) for o in obj_list]
        imgs = [i.reshape(1, -1, 2).astype(np.float64) for i in img_list]
        dropped = 0
        while True:
            K = np.array([[f_guess, 0.0, w / 2 - 0.5],
                          [0.0, f_guess, h / 2 - 0.5],
                          [0.0, 0.0, 1.0]])
            D = np.zeros((4, 1))
            try:
                rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                    objs, imgs, img_size, K, D, None, None, flags=flags, criteria=_CRIT
                )
                break
            except cv2.error as e:
                last_error = str(e)
                m = _ILL_COND_RE.search(last_error)
                if m is None or dropped >= max_drops or len(objs) <= 4:
                    objs = None
                    break
                bad = int(m.group(1))
                del objs[bad], imgs[bad]
                dropped += 1
        if objs is not None:
            break
    else:
        return MonoModelFit("fisheye", False, error=last_error)

    fit = MonoModelFit(
        "fisheye", True, K=np.asarray(K, dtype=np.float64),
        D=np.asarray(D, dtype=np.float64).ravel()[:4],
        rms_train=float(rms), rvecs=list(rvecs), tvecs=list(tvecs),
        n_views=len(objs), dropped_views=dropped,
    )
    fit.per_view_rpe = per_view_rpe(
        fit,
        [o.reshape(-1, 1, 3).astype(np.float32) for o in objs],
        [i.reshape(-1, 1, 2).astype(np.float32) for i in imgs],
    )
    return fit


def fit_model(
    model: str,
    corners: list[np.ndarray],
    ids: list[np.ndarray],
    board,
    img_size: tuple[int, int],
) -> MonoModelFit:
    if model == "pinhole":
        return fit_pinhole(corners, ids, board, img_size)
    if model == "rational":
        return fit_pinhole(corners, ids, board, img_size, rational=True)
    if model == "fisheye":
        return fit_fisheye(corners, ids, board, img_size)
    raise ValueError(f"unknown model: {model!r}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _project(model: str, obj: np.ndarray, rvec, tvec, K, D) -> np.ndarray:
    """Project (N,1,3) object points -> (N,1,2), model-aware."""
    if model == "fisheye":
        proj, _ = cv2.fisheye.projectPoints(
            obj.reshape(1, -1, 3).astype(np.float64), rvec, tvec, K, D.reshape(4, 1)
        )
        return proj.reshape(-1, 1, 2)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    return proj


def per_view_rpe(
    fit: MonoModelFit,
    obj_list: list[np.ndarray],
    img_list: list[np.ndarray],
) -> list[float]:
    """RMS reprojection error per view using the fit's own extrinsics."""
    out: list[float] = []
    for obj, img, rvec, tvec in zip(obj_list, img_list, fit.rvecs, fit.tvecs):
        proj = _project(fit.model, obj, rvec, tvec, fit.K, fit.D)
        err = proj.reshape(-1, 2) - img.reshape(-1, 2)
        out.append(float(np.sqrt(np.mean(np.sum(err**2, axis=1)))))
    return out


def holdout_rpe(
    fit: MonoModelFit,
    obj_list: list[np.ndarray],
    img_list: list[np.ndarray],
) -> float:
    """RMS reprojection error on views NOT used in the fit.

    Pose per holdout view is solved with the fitted K/D, then reprojected.
    For fisheye, points are first undistorted to the normalized plane so a
    plain solvePnP (K=I, D=0) applies; reprojection uses fisheye.projectPoints.
    """
    if not fit.ok or not obj_list:
        return float("nan")

    sq_errs: list[np.ndarray] = []
    for obj, img in zip(obj_list, img_list):
        if fit.model == "fisheye":
            und = cv2.fisheye.undistortPoints(
                img.reshape(1, -1, 2).astype(np.float64), fit.K, fit.D.reshape(4, 1)
            ).reshape(-1, 1, 2)
            ok, rvec, tvec = cv2.solvePnP(
                obj.astype(np.float64), und, np.eye(3), np.zeros(4),
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        else:
            ok, rvec, tvec = cv2.solvePnP(
                obj.astype(np.float64), img.astype(np.float64), fit.K, fit.D,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        if not ok:
            continue
        proj = _project(fit.model, obj, rvec, tvec, fit.K, fit.D)
        err = proj.reshape(-1, 2) - img.reshape(-1, 2)
        sq_errs.append(np.sum(err**2, axis=1))

    if not sq_errs:
        return float("nan")
    return float(np.sqrt(np.mean(np.concatenate(sq_errs))))


def _radial_curve(
    K: np.ndarray, D: np.ndarray, model: str,
    direction: tuple[float, float], *, n: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-project rays at increasing angle from the optical axis.

    Returns (theta_rad, image_radius_px) measured from the principal point,
    along the unit image-plane ``direction``. This uses only the FORWARD
    (closed-form polynomial) projection — cv2.undistortPoints' iterative
    inverse diverges at large radii and cannot be trusted for these checks.

    The fisheye branch evaluates the equidistant model analytically rather
    than calling cv2.fisheye.projectPoints: that function derives the ray
    angle as atan(|x/z|), so it silently folds rays beyond 90 degrees and
    cannot describe a lens wider than 180 degrees.
    """
    ux, uy = direction
    scale = float(np.hypot(K[0, 0] * ux, K[1, 1] * uy))

    if model == "fisheye":
        th = np.linspace(1e-4, np.deg2rad(140.0), n)
        k = np.asarray(D, dtype=np.float64).ravel()[:4]
        th_d = th * (1 + k[0] * th**2 + k[1] * th**4 + k[2] * th**6 + k[3] * th**8)
        return th, th_d * scale

    th = np.linspace(1e-4, np.deg2rad(88.0), n)
    pts3d = np.stack(
        [np.sin(th) * ux, np.sin(th) * uy, np.cos(th)], axis=1
    ).astype(np.float64)
    zero = np.zeros(3, dtype=np.float64)
    try:
        proj, _ = cv2.projectPoints(pts3d.reshape(-1, 1, 3), zero, zero, K, D)
    except cv2.error:
        return th, np.full(n, np.nan)

    proj = proj.reshape(-1, 2)
    radii = np.hypot(proj[:, 0] - K[0, 2], proj[:, 1] - K[1, 2])
    return th, radii


def _corner_radius(K: np.ndarray, img_size: tuple[int, int]) -> float:
    w, h = img_size
    cx, cy = K[0, 2], K[1, 2]
    return float(max(np.hypot(x - cx, y - cy) for x, y in ((0, 0), (w, 0), (0, h), (w, h))))


def undistort_monotonic(
    K: np.ndarray, D: np.ndarray, img_size: tuple[int, int], model: str,
) -> bool:
    """True when the forward projection is injective across the image.

    Fold-back — two different incoming rays landing on the same pixel — shows
    up as the image radius ceasing to increase with ray angle. Only the
    portion of the curve that actually falls inside the image is judged.
    """
    w, h = img_size
    cx, cy = K[0, 2], K[1, 2]
    corner = max(
        [(0, 0), (w, 0), (0, h), (w, h)],
        key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2,
    )
    d = np.array([corner[0] - cx, corner[1] - cy], dtype=np.float64)
    norm = np.linalg.norm(d)
    if norm < 1e-9:
        return False
    _th, radii = _radial_curve(K, D, model, tuple(d / norm))

    r_max = _corner_radius(K, img_size)
    # longest leading run that is finite and still inside the image
    inside = np.isfinite(radii) & (radii <= r_max)
    if not inside[0]:
        return False
    end = int(np.argmin(inside)) if not inside.all() else len(radii)
    if end < 3:
        return True          # too little of the image covered to judge

    run = radii[:end]
    if np.all(np.diff(run) > 0):
        return True
    # Turnover happened. Tolerate it only in the extreme corners, which are
    # the most extrapolated part of any calibration and are cropped away by
    # the default alpha=0 undistortion anyway.
    turn = int(np.argmin(np.concatenate([[True], np.diff(run) > 0])))
    return bool(run[:turn].max(initial=0.0) >= _CORNER_TOLERANCE * r_max)


def estimate_hfov_deg(
    K: np.ndarray, D: np.ndarray, img_size: tuple[int, int], model: str,
) -> float:
    """Horizontal FOV: the ray angles that project onto the left/right edges."""
    w, _h = img_size
    cx = K[0, 2]
    total = 0.0
    for direction, edge_dist in (((1.0, 0.0), w - cx), ((-1.0, 0.0), cx)):
        th, radii = _radial_curve(K, D, model, direction)
        ok = np.isfinite(radii)
        if ok.sum() < 3:
            return float("nan")
        th, radii = th[ok], radii[ok]
        # keep the strictly-increasing leading run so np.interp is valid
        good = np.concatenate([[True], np.diff(radii) > 0])
        end = int(np.argmin(good)) if not good.all() else len(radii)
        th, radii = th[:end], radii[:end]
        if len(radii) < 2 or radii[-1] < edge_dist:
            return float("nan")   # model never reaches the image edge
        total += float(np.interp(edge_dist, radii, th))
    return float(np.degrees(total))


def coeff_sane(fit: MonoModelFit, img_size: tuple[int, int]) -> tuple[bool, list[str]]:
    """Cheap plausibility checks on the fitted intrinsics."""
    if not fit.ok:
        return False, ["fit failed"]
    w, h = img_size
    K, D = fit.K, fit.D
    problems: list[str] = []

    fx, fy = K[0, 0], K[1, 1]
    if fx <= 0 or fy <= 0:
        problems.append("non-positive focal length")
    else:
        ratio = fx / fy
        if not (0.95 <= ratio <= 1.05):
            problems.append(f"fx/fy ratio {ratio:.3f} outside [0.95, 1.05]")

    cx, cy = K[0, 2], K[1, 2]
    if abs(cx - w / 2) > 0.20 * w or abs(cy - h / 2) > 0.20 * h:
        problems.append("principal point outside central 20% of image")

    if fit.model == "pinhole" and abs(D[4]) > 10.0:
        problems.append(f"|k3|={abs(D[4]):.1f} > 10")

    if fit.model == "rational":
        # denominator 1 + k4 r^2 + k5 r^4 + k6 r^6 must stay positive over the image
        k4, k5, k6 = D[5], D[6], D[7]
        r_max = np.hypot(max(cx, w - cx), max(cy, h - cy)) / min(fx, fy)
        r2 = np.linspace(0, r_max**2, 100)
        denom = 1 + k4 * r2 + k5 * r2**2 + k6 * r2**3
        if np.any(denom <= 1e-6):
            problems.append("rational denominator has a pole inside the image")

    return (len(problems) == 0), problems


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

# Complexity margins: a more complex model must beat the simpler one on
# holdout RPE by BOTH margins to win. Tie-break order = simplest first.
_REL_MARGIN = 0.10
_ABS_MARGIN = 0.05
_HFOV_TOL_DEG = 8.0
_PREFERENCE = ("pinhole", "fisheye", "rational")   # simplest -> most complex


@dataclass
class ModelSelection:
    selected: str
    reason: str
    table: dict[str, dict]     # per-model metrics for the report/YAML


def select_model(
    fits: dict[str, MonoModelFit],
    holdout_scores: dict[str, float],
    hfovs: dict[str, float],
    sanity: dict[str, tuple[bool, list[str]]],
    monotonic: dict[str, bool],
    *,
    has_holdout: bool = True,
) -> ModelSelection:
    """Fair model selection. Ranking metric is holdout RPE (never train RPE);
    a more complex model must beat a simpler one by >10% relative AND
    >0.05 px absolute, else the simpler model wins."""
    table: dict[str, dict] = {}
    qualified: list[str] = []

    finite_hfovs = [v for v in hfovs.values() if np.isfinite(v)]
    hfov_median = float(np.median(finite_hfovs)) if finite_hfovs else float("nan")

    for m in MODELS:
        fit = fits.get(m)
        entry: dict = {"ok": bool(fit and fit.ok)}
        disq: list[str] = []
        if fit is None or not fit.ok:
            disq.append(f"fit failed: {fit.error if fit else 'not run'}")
        else:
            entry["rpe_train_px"] = round(fit.rms_train, 4)
            score = holdout_scores.get(m, float("nan"))
            entry["rpe_holdout_px"] = None if np.isnan(score) else round(score, 4)
            entry["hfov_deg"] = None if np.isnan(hfovs.get(m, float("nan"))) else round(hfovs[m], 1)
            ok_sane, sane_problems = sanity.get(m, (True, []))
            mono_ok = monotonic.get(m, True)
            entry["sanity"] = {
                "coeffs": ok_sane,
                "monotonic": mono_ok,
                "fov_consistent": True,
            }
            if not ok_sane:
                disq.extend(sane_problems)
            if not mono_ok:
                disq.append("undistortion not monotonic (fold-back)")
            hf = hfovs.get(m, float("nan"))
            if np.isfinite(hfov_median) and np.isfinite(hf) and abs(hf - hfov_median) > _HFOV_TOL_DEG:
                entry["sanity"]["fov_consistent"] = False
                disq.append(f"HFOV {hf:.1f} deviates >{_HFOV_TOL_DEG} deg from median {hfov_median:.1f}")
            if has_holdout and np.isnan(score):
                disq.append("holdout RPE could not be computed")
        entry["disqualified"] = disq
        table[m] = entry
        if not disq:
            qualified.append(m)

    if not qualified:
        # last resort: any model that at least fit, by train RPE
        fallback = [m for m in MODELS if fits.get(m) and fits[m].ok]
        if not fallback:
            raise RuntimeError("All distortion models failed to fit")
        best = min(fallback, key=lambda m: fits[m].rms_train)
        return ModelSelection(
            best,
            "all models disqualified by sanity checks; "
            f"falling back to lowest train RPE ({best})",
            table,
        )

    def score_of(m: str) -> float:
        if has_holdout:
            return holdout_scores[m]
        return fits[m].rms_train

    metric = "holdout RPE" if has_holdout else "train RPE (no holdout — collect more views)"

    # walk preference order; a later (more complex) candidate replaces the
    # incumbent only if it clears both margins
    ordered = [m for m in _PREFERENCE if m in qualified]
    best = ordered[0]
    for cand in ordered[1:]:
        s_best, s_cand = score_of(best), score_of(cand)
        if s_cand < s_best * (1 - _REL_MARGIN) and (s_best - s_cand) > _ABS_MARGIN:
            best = cand

    reason = (
        f"{best} selected by {metric}: "
        + ", ".join(f"{m}={score_of(m):.3f}px" for m in ordered)
        + f"; complexity margin >{_REL_MARGIN:.0%} rel and >{_ABS_MARGIN}px abs required to upgrade"
    )
    p_fit, f_fit = fits.get("pinhole"), fits.get("fisheye")
    if p_fit and f_fit and p_fit.ok and f_fit.ok and p_fit.rms_train > 2 * f_fit.rms_train:
        reason += "; NOTE: pinhole train RMS > 2x fisheye — lens is likely fisheye-projection"
    return ModelSelection(best, reason, table)


# ---------------------------------------------------------------------------
# Undistortion helpers (shared with the `undistort` command)
# ---------------------------------------------------------------------------

def new_camera_matrix(
    K: np.ndarray, D: np.ndarray, img_size: tuple[int, int], model: str,
    *, alpha: float = 0.0, balance: float = 0.0,
) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    """K_new for undistorted images.

    pinhole/rational: cv2.getOptimalNewCameraMatrix(alpha) -> (K_new, roi)
    fisheye: cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(balance)
             -> (K_new, None)   (no ROI concept)

    K_new pairs with UNDISTORTED images and zero distortion. K/D pair with
    RAW images. Never mix.
    """
    if model == "fisheye":
        K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D.reshape(4, 1), img_size, np.eye(3), balance=float(balance)
        )
        return K_new, None
    K_new, roi = cv2.getOptimalNewCameraMatrix(K, D, img_size, float(alpha))
    return K_new, tuple(int(v) for v in roi)


def build_undistort_maps(
    K: np.ndarray, D: np.ndarray, K_new: np.ndarray,
    img_size: tuple[int, int], model: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Remap tables (CV_32FC1) for cv2.remap."""
    if model == "fisheye":
        return cv2.fisheye.initUndistortRectifyMap(
            K, D.reshape(4, 1), np.eye(3), K_new, img_size, cv2.CV_32FC1
        )
    return cv2.initUndistortRectifyMap(K, D, None, K_new, img_size, cv2.CV_32FC1)


# ---------------------------------------------------------------------------
# Holdout split
# ---------------------------------------------------------------------------

def split_train_holdout(
    n_views: int, *, holdout_frac: float = 0.25, seed: int = 0, min_train: int = 12,
) -> tuple[list[int], list[int]]:
    """Deterministic shuffled split. Returns (train_idx, holdout_idx).
    Holdout is empty when there are too few views to spare."""
    idx = list(range(n_views))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_hold = int(round(n_views * holdout_frac))
    if n_views - n_hold < min_train:
        n_hold = max(0, n_views - min_train)
    if n_hold == 0:
        return idx, []
    return idx[n_hold:], idx[:n_hold]
