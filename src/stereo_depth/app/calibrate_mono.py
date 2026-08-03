"""Mono intrinsic calibration: live collection and the calibration pipeline.

``run_capture_mono``  — live stream, on-screen guidance, auto-saves good views.
``run_calibrate_mono`` — fits three distortion models, picks one fairly, and
                         writes a YAML carrying Isaac Sim parameters.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from stereo_depth.adapters.calibration.capture_gates import AutoCollectPolicy
from stereo_depth.adapters.calibration.charuco_calibrator import (
    detect_charuco,
    make_charuco_board,
)
from stereo_depth.adapters.calibration.isaac_export import (
    DEFAULT_HORIZONTAL_APERTURE_MM,
    distortion_block,
    usd_camera_params,
)
from stereo_depth.adapters.calibration.mono_calibrator import (
    MODELS,
    coeff_sane,
    collect_charuco_mono_from_paths,
    estimate_hfov_deg,
    fit_fisheye,
    fit_model,
    holdout_rpe,
    match_object_points,
    new_camera_matrix,
    select_model,
    split_train_holdout,
    undistort_monotonic,
)
from stereo_depth.adapters.calibration.mono_yaml_repo import (
    PAIRING_NOTE,
    build_document,
)
from stereo_depth.adapters.camera.uvc_source import open_source
from stereo_depth.entities import MonoIntrinsics
from stereo_depth.infrastructure.config.io import save_yaml
from stereo_depth.infrastructure.io.pairs import list_images
from stereo_depth.infrastructure.viz.overlay import (
    GREEN,
    RED,
    WHITE,
    YELLOW,
    draw_banner,
    draw_coverage_inset,
    draw_progress_bar,
    draw_status_line,
    draw_text,
)

BOARD_DEFAULTS = dict(
    squares_x=7, squares_y=5,
    square_length=0.03, marker_length=0.022,
    dict_name="DICT_5X5_100",
)

_WIN = "stereo-depth capture-mono  |  SPACE force-save  R undo  G auto  Q quit"
_FLASH_FRAMES = 12


# ---------------------------------------------------------------------------
# Live collection
# ---------------------------------------------------------------------------

def run_capture_mono(
    out_dir: Path,
    *,
    path: str = "/dev/video0",
    width: int = 0,
    height: int = 0,
    fps: int = 30,
    squares_x: int = 7,
    squares_y: int = 5,
    square_length: float = 0.03,
    marker_length: float = 0.022,
    dict_name: str = "DICT_5X5_100",
    min_markers: int = 4,
    min_charuco: int = 10,
    target_views: int = 40,
    blur_min: float = 60.0,
    grid: int = 4,
    edge_target: float = 0.85,
    auto: bool = True,
) -> int:
    """Stream one camera and auto-save views that clear every quality gate.

    Returns the number of images written to ``out_dir``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    board, dictionary = make_charuco_board(
        squares_x=squares_x, squares_y=squares_y,
        square_length=square_length, marker_length=marker_length,
        dict_name=dict_name,
    )

    try:
        device = int(path)
        cap = open_source(device=device, width=width, height=height, fps=fps)
    except ValueError:
        cap = open_source(path=path, width=width, height=height, fps=fps)

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError(f"Cannot read frames from {path}")
    h, w = frame.shape[:2]
    image_size = (w, h)

    policy = AutoCollectPolicy(
        image_size, board,
        target_views=target_views, blur_min=blur_min, rows=grid, cols=grid,
        edge_target=edge_target,
    )

    n_saved = 0
    flash_left = 0
    auto_on = auto
    saved_stems: list[str] = []

    print(f"Capturing {w}x{h} from {path} -> {out_dir}")
    print("Move the board around the frame; tilt it left/right/up/down.")

    cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Camera read failed - stopping.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            det = detect_charuco(
                gray, board, dictionary,
                min_markers=min_markers, min_charuco=min_charuco,
            )
            now = time.perf_counter()
            status = policy.evaluate(det, gray, now)

            display = frame.copy()
            if det.ok and det.corners is not None:
                cv2.aruco.drawDetectedCornersCharuco(
                    display, det.corners, det.ids, (0, 255, 255)
                )

            save_now = auto_on and status.all_ok and not policy.done()

            if flash_left > 0:
                overlay = display.copy()
                cv2.rectangle(overlay, (0, 0), (display.shape[1], display.shape[0]),
                              GREEN, -1)
                a = 0.25 * (flash_left / _FLASH_FRAMES)
                display = cv2.addWeighted(overlay, a, display, 1 - a, 0)
                flash_left -= 1

            _draw_capture_hud(display, policy, status, n_saved, auto_on)
            cv2.imshow(_WIN, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("g"), ord("G")):
                auto_on = not auto_on
                print(f"  Auto-capture {'ON' if auto_on else 'OFF'}")
            if key == ord(" ") and det.ok:
                save_now = True          # manual override ignores novelty
            if key in (ord("r"), ord("R")) and saved_stems:
                stem = saved_stems.pop()
                (out_dir / stem).unlink(missing_ok=True)
                policy.undo()
                n_saved -= 1
                print(f"  Removed {stem}")

            if save_now:
                n_saved += 1
                stem = f"{n_saved:04d}.png"
                cv2.imwrite(str(out_dir / stem), frame)
                saved_stems.append(stem)
                policy.accept(det, now)
                flash_left = _FLASH_FRAMES
                p = policy.progress()
                print(
                    f"  Saved {stem}  views {p['views']}/{p['target_views']}"
                    f"  cells {p['cells']}/{p['total_cells']}  tilts {p['tilts']}"
                )

    finally:
        cap.release()
        cv2.destroyAllWindows()

    p = policy.progress()
    print(f"\nCollected {n_saved} images to {out_dir}")
    print(f"  coverage {p['cells']}/{p['total_cells']} cells, "
          f"edge reach {p['edge_frac'] * 100:.0f}%, {p['tilts']} tilt bins")
    if not policy.done():
        missing = policy.coverage.missing()
        if missing:
            print(f"  WARNING: {len(missing)} image regions never covered - "
                  "distortion at the edges will be extrapolated")
        if not p["edge_ok"]:
            print(f"  WARNING: corners only reached {p['edge_frac'] * 100:.0f}% of the "
                  f"way to the frame corners (want {p['edge_target'] * 100:.0f}%) - "
                  "the distortion model has no data out there and will extrapolate")
        if p["tilts"] < p["min_tilt_bins"]:
            print(f"  WARNING: only {p['tilts']} tilt bins seen "
                  f"(want {p['min_tilt_bins']}) - fx/fy may be poorly separated")
    return n_saved


def _draw_capture_hud(img, policy: AutoCollectPolicy, status, n_saved: int, auto_on: bool) -> None:
    h, w = img.shape[:2]
    prog = policy.progress()

    draw_coverage_inset(
        img, policy.coverage.covered, status.current_cells,
        policy.coverage.rows, policy.coverage.cols, origin=(10, 10), cell_px=30,
    )

    draw_status_line(
        img,
        [
            (f"CORNERS {status.num_charuco}", status.detect_ok),
            (f"SHARP {status.blur:.0f}", status.sharp_ok),
            ("STEADY", status.steady_ok),
            ("NEW", status.novel_ok),
            (f"TILT {status.tilt_bin or '-'}", status.tilt_bin is not None),
            (f"EDGE {prog['edge_frac'] * 100:.0f}%", prog["edge_ok"]),
        ],
        (10, int(policy.coverage.rows * 30) + 34),
    )

    done = policy.done()
    if done:
        draw_banner(img, "DONE - press Q to finish", color=GREEN, y=h // 2)
    else:
        color = GREEN if status.all_ok else (YELLOW if status.detect_ok else RED)
        draw_banner(img, status.guidance, color=color, y=h // 2)

    draw_progress_bar(img, prog["views"] / max(prog["target_views"], 1))
    draw_text(
        img,
        f"views {prog['views']}/{prog['target_views']}  "
        f"cells {prog['cells']}/{prog['total_cells']}  "
        f"tilts {prog['tilts']}/{prog['min_tilt_bins']}  "
        f"edge {prog['edge_frac'] * 100:.0f}/{prog['edge_target'] * 100:.0f}%  "
        f"auto {'ON' if auto_on else 'OFF'}  |  SPACE save  R undo  G auto  Q quit",
        (10, h - 14), scale=0.5, color=WHITE,
    )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def run_calibrate_mono(
    data_dir: Path,
    out_yaml: Path,
    *,
    squares_x: int = 7,
    squares_y: int = 5,
    square_length: float = 0.03,
    marker_length: float = 0.022,
    dict_name: str = "DICT_5X5_100",
    min_markers: int = 4,
    min_charuco: int = 10,
    min_views: int = 15,
    model: str = "auto",
    holdout_frac: float = 0.25,
    seed: int = 0,
    refine: bool = True,
    sensor_width_mm: float = DEFAULT_HORIZONTAL_APERTURE_MM,
    report_json: Path | None = None,
    verbose: bool = True,
) -> tuple[Path, Path]:
    """Calibrate one camera from a folder of ChArUco images.

    Returns ``(out_yaml, report_json)``. A report is always written, including
    on failure, so a bad session can be diagnosed without re-running.
    """
    if report_json is None:
        report_json = out_yaml.with_suffix(".report.json")

    board, dictionary = make_charuco_board(
        squares_x=squares_x, squares_y=squares_y,
        square_length=square_length, marker_length=marker_length,
        dict_name=dict_name,
    )
    board_params = {
        "squares_x": squares_x, "squares_y": squares_y,
        "square_length": square_length, "marker_length": marker_length,
        "dict_name": dict_name,
    }

    paths = list_images(data_dir)
    report: dict = {
        "status": "collecting",
        "inputs": {"data_dir": str(data_dir), "num_images": len(paths)},
        "params": {**board_params, "min_views": min_views, "min_markers": min_markers,
                   "min_charuco": min_charuco, "model": model,
                   "holdout_frac": holdout_frac, "seed": seed, "refine": refine},
    }
    if not paths:
        report["status"] = "failed"
        report["reason"] = "no_images_found"
        _write_report(report_json, report)
        raise RuntimeError(f"No images found in {data_dir}. See report: {report_json}")

    views = collect_charuco_mono_from_paths(
        paths, board, dictionary,
        min_markers=min_markers, min_charuco=min_charuco, refine=refine,
    )
    report["status"] = "collected"
    report["collect"] = {
        **views.report.__dict__,
        "skipped_size_mismatch": views.skipped_size_mismatch,
        "image_size": {"width": views.image_size[0], "height": views.image_size[1]},
    }
    _write_report(report_json, report)

    n = len(views.corners)
    if n < min_views:
        report["status"] = "failed"
        report["reason"] = "not_enough_valid_views"
        report["valid_views"] = n
        _write_report(report_json, report)
        raise RuntimeError(
            f"Not enough valid views: got={n}, need>={min_views}. "
            f"See report: {report_json}"
        )

    img_size = views.image_size
    candidates = MODELS if model == "auto" else (model,)
    if model != "auto" and model not in MODELS:
        raise ValueError(f"unknown model {model!r}; choose from auto/{'/'.join(MODELS)}")

    # Holdout keeps model comparison honest: extra coefficients always reduce
    # training error, so ranking on training error would always pick rational.
    train_idx, hold_idx = split_train_holdout(
        n, holdout_frac=holdout_frac, seed=seed, min_train=max(12, min_views // 2)
    )
    has_holdout = bool(hold_idx) and model == "auto"

    tc = [views.corners[i] for i in train_idx]
    ti = [views.ids[i] for i in train_idx]

    fits, scores, hfovs, sanity, mono = {}, {}, {}, {}, {}
    hold_obj, hold_img = [], []
    if hold_idx:
        hold_obj, hold_img, _ = match_object_points(
            [views.corners[i] for i in hold_idx],
            [views.ids[i] for i in hold_idx],
            board,
        )

    for m in candidates:
        if m == "fisheye" and fits.get("pinhole") and fits["pinhole"].ok:
            f = fit_fisheye(tc, ti, board, img_size, k_guess=fits["pinhole"].K)
        else:
            f = fit_model(m, tc, ti, board, img_size)
        fits[m] = f
        scores[m] = holdout_rpe(f, hold_obj, hold_img) if has_holdout else float("nan")
        if f.ok:
            hfovs[m] = estimate_hfov_deg(f.K, f.D, img_size, m)
            sanity[m] = coeff_sane(f, img_size)
            mono[m] = undistort_monotonic(f.K, f.D, img_size, m)
        else:
            hfovs[m] = float("nan")
            sanity[m] = (False, [f"fit failed: {f.error}"])
            mono[m] = False

    if model == "auto":
        selection = select_model(fits, scores, hfovs, sanity, mono, has_holdout=has_holdout)
    else:
        if not fits[model].ok:
            report["status"] = "failed"
            report["reason"] = "requested_model_failed"
            report["error"] = fits[model].error
            _write_report(report_json, report)
            raise RuntimeError(f"Requested model {model!r} failed: {fits[model].error}")
        from stereo_depth.adapters.calibration.mono_calibrator import ModelSelection
        selection = ModelSelection(model, f"{model} forced via --model", {})

    # Refit the winner on every view — the holdout existed only to choose.
    chosen = selection.selected
    if chosen == "fisheye":
        final = fit_fisheye(views.corners, views.ids, board, img_size,
                            k_guess=fits.get("pinhole").K if fits.get("pinhole") and fits["pinhole"].ok else None)
    else:
        final = fit_model(chosen, views.corners, views.ids, board, img_size)
    if not final.ok:
        final = fits[chosen]          # fall back to the training-set fit

    result = MonoIntrinsics(
        model=chosen, image_size=img_size,
        K=final.K, D=final.D,
        rpe_px=float(final.rms_train), views_used=int(final.n_views),
    )

    undistort_section = {}
    for label, alpha, balance in (("alpha_0", 0.0, 0.0), ("alpha_1", 1.0, 1.0)):
        K_new, roi = new_camera_matrix(
            final.K, final.D, img_size, chosen, alpha=alpha, balance=balance
        )
        undistort_section[label] = {
            "alpha" if chosen != "fisheye" else "balance": alpha if chosen != "fisheye" else balance,
            "K_new": np.asarray(K_new, dtype=float).tolist(),
            "roi": list(roi) if roi else None,
        }
    undistort_section["note"] = PAIRING_NOTE

    K_new_tight, _ = new_camera_matrix(final.K, final.D, img_size, chosen,
                                       alpha=0.0, balance=0.0)
    block_name, block = distortion_block(chosen, final.K, final.D, img_size)
    isaac = {
        "usd_camera_raw": {
            **usd_camera_params(final.K, img_size, horizontal_aperture_mm=sensor_width_mm),
            "note": "Pinhole approximation of the RAW K. Only valid together "
                    "with the distortion block below.",
        },
        "usd_camera_undistorted": {
            **usd_camera_params(K_new_tight, img_size, horizontal_aperture_mm=sensor_width_mm),
            "note": "Use with UNDISTORTED frames (stereo-depth undistort, alpha/balance 0) "
                    "and zero distortion in the sim.",
        },
        block_name: block,
    }

    doc = build_document(
        result,
        board=board_params,
        selection={
            "reason": selection.reason,
            "train_views": len(train_idx),
            "holdout_views": len(hold_idx) if has_holdout else 0,
            "seed": seed,
            "forced": model != "auto",
        },
        models=selection.table or None,
        undistort=undistort_section,
        isaac_sim=isaac,
    )
    save_yaml(out_yaml, doc)

    report["status"] = "success"
    report["selected_model"] = chosen
    report["selection_reason"] = selection.reason
    report["models"] = selection.table
    report["rpe_px"] = result.rpe_px
    report["views_used"] = result.views_used
    report["per_view_rpe"] = [
        {"view": name, "rpe_px": round(r, 4)}
        for name, r in zip(views.names, final.per_view_rpe)
    ]
    _write_report(report_json, report)

    if verbose:
        _print_summary(selection, fits, scores, hfovs, result, has_holdout, out_yaml, report_json)

    return out_yaml, report_json


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _print_summary(selection, fits, scores, hfovs, result, has_holdout, out_yaml, report_json) -> None:
    print("\nDistortion model comparison")
    print(f"  {'model':<9} {'coeffs':>6} {'train RPE':>10} {'holdout':>9} {'HFOV':>7}  notes")
    for m in MODELS:
        f = fits.get(m)
        if f is None:
            continue
        if not f.ok:
            print(f"  {m:<9} {'-':>6} {'-':>10} {'-':>9} {'-':>7}  FAILED: {f.error}")
            continue
        hold = scores.get(m, float("nan"))
        hold_s = f"{hold:.3f}" if has_holdout and np.isfinite(hold) else "-"
        hfov = hfovs.get(m, float("nan"))
        hfov_s = f"{hfov:.0f}deg" if np.isfinite(hfov) else "-"
        notes = "; ".join(selection.table.get(m, {}).get("disqualified", [])) or "ok"
        mark = "*" if m == selection.selected else " "
        print(f" {mark}{m:<9} {len(f.D):>6} {f.rms_train:>10.4f} {hold_s:>9} {hfov_s:>7}  {notes}")

    K = result.K
    print(f"\nSelected: {result.model}   ({selection.reason})")
    print(f"  image_size {result.image_size[0]}x{result.image_size[1]}   views {result.views_used}")
    print(f"  fx={K[0, 0]:.2f}  fy={K[1, 1]:.2f}  cx={K[0, 2]:.2f}  cy={K[1, 2]:.2f}")
    print(f"  D = {np.array2string(np.asarray(result.D), precision=5, suppress_small=True)}")
    print(f"  RPE {result.rpe_px:.4f} px" + ("  (target < 0.5)" if result.rpe_px >= 0.5 else ""))
    print(f"\nSaved calibration: {out_yaml}")
    print(f"Saved report:      {report_json}")
    print("\nRemember: K/D go with RAW frames; K_new (in the 'undistort' section)")
    print("goes with UNDISTORTED frames and zero distortion. Never mix them.")
