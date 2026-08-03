"""Auto-collection gate logic — pure, hardware-free."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from stereo_depth.adapters.calibration.capture_gates import (
    TILT_BINS,
    AutoCollectPolicy,
    CoverageTracker,
    SteadyTracker,
    TiltTracker,
    blur_score,
    cell_label,
)
from stereo_depth.adapters.calibration.charuco_calibrator import (
    CharucoDetection,
    make_charuco_board,
)

IMG_SIZE = (1280, 720)


def _corners_at(cx, cy, spread=40, n=9):
    """A small square cluster of n*n corners centred on (cx, cy)."""
    offs = np.linspace(-spread, spread, int(np.sqrt(n)) if n > 1 else 1)
    pts = np.array([[cx + dx, cy + dy] for dy in offs for dx in offs])
    return pts.astype(np.float32).reshape(-1, 1, 2)


def _det(corners, *, ok=True, reason=None, num_charuco=None):
    n = 0 if corners is None else len(corners)
    return CharucoDetection(
        ok=ok, num_markers=n, num_charuco=num_charuco if num_charuco is not None else n,
        corners=corners, ids=None if corners is None else np.arange(n).reshape(-1, 1).astype(np.int32),
        image_size=IMG_SIZE, reason=reason,
    )


# ---------------------------------------------------------------------------
# CoverageTracker
# ---------------------------------------------------------------------------

def test_corners_map_to_the_expected_cell():
    cov = CoverageTracker(IMG_SIZE, 3, 3, min_corners=6)
    w, h = IMG_SIZE

    assert cov.cells_of(_corners_at(w / 6, h / 6)) == {(0, 0)}
    assert cov.cells_of(_corners_at(w / 2, h / 2)) == {(1, 1)}
    assert cov.cells_of(_corners_at(5 * w / 6, 5 * h / 6)) == {(2, 2)}


def test_cell_needs_enough_corners_to_count():
    cov = CoverageTracker(IMG_SIZE, 3, 3, min_corners=6)
    few = _corners_at(640, 360, spread=5, n=4)      # only 4 corners

    assert cov.cells_of(few) == set()


def test_commit_reports_only_newly_covered_cells():
    cov = CoverageTracker(IMG_SIZE, 3, 3, min_corners=6)
    corners = _corners_at(213, 120)

    assert cov.commit(corners) == {(0, 0)}
    assert cov.commit(corners) == set()             # already covered
    assert cov.covered == {(0, 0)}


def test_missing_shrinks_as_cells_are_covered():
    cov = CoverageTracker(IMG_SIZE, 3, 3, min_corners=6)
    assert len(cov.missing()) == 9

    w, h = IMG_SIZE
    for r in range(3):
        for c in range(3):
            cov.commit(_corners_at((c + 0.5) * w / 3, (r + 0.5) * h / 3))
    assert cov.missing() == []


def test_nearest_missing_label_points_at_a_real_gap():
    cov = CoverageTracker(IMG_SIZE, 3, 3, min_corners=6)
    w, h = IMG_SIZE
    for r in range(3):
        for c in range(3):
            if (r, c) != (0, 2):                    # leave TOP-RIGHT open
                cov.commit(_corners_at((c + 0.5) * w / 3, (r + 0.5) * h / 3))

    assert cov.nearest_missing_label(_corners_at(w / 2, h / 2)) == "TOP-RIGHT"


def test_nearest_missing_label_is_none_when_complete():
    cov = CoverageTracker(IMG_SIZE, 3, 3, min_corners=6)
    w, h = IMG_SIZE
    for r in range(3):
        for c in range(3):
            cov.commit(_corners_at((c + 0.5) * w / 3, (r + 0.5) * h / 3))

    assert cov.nearest_missing_label(_corners_at(w / 2, h / 2)) is None


def test_cells_of_handles_no_detection():
    cov = CoverageTracker(IMG_SIZE, 3, 3)
    assert cov.cells_of(None) == set()
    assert cov.cells_of(np.empty((0, 1, 2), dtype=np.float32)) == set()


def test_cell_labels_cover_the_three_by_three_grid():
    labels = {cell_label(r, c) for r in range(3) for c in range(3)}
    assert "TOP-LEFT" in labels and "BOTTOM-RIGHT" in labels and "CENTRE" in labels
    assert len(labels) == 9
    assert cell_label(0, 0, rows=4, cols=4) == "R1C1"


# ---------------------------------------------------------------------------
# blur_score
# ---------------------------------------------------------------------------

def _checkerboard(w=400, h=300, sq=25):
    img = np.zeros((h, w), dtype=np.uint8)
    for y in range(0, h, sq):
        for x in range(0, w, sq):
            if ((x // sq) + (y // sq)) % 2 == 0:
                img[y:y + sq, x:x + sq] = 255
    return img


def test_blur_score_drops_when_the_image_is_blurred():
    sharp = _checkerboard()
    blurred = cv2.GaussianBlur(sharp, (11, 11), 4)

    assert blur_score(sharp) > blur_score(blurred)
    assert blur_score(blurred) < blur_score(sharp) / 2


def test_blur_score_ignores_background_outside_the_board():
    """A sharp, busy background must not rescue a blurred board."""
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (300, 400), dtype=np.uint8)   # noisy background
    board = cv2.GaussianBlur(_checkerboard(120, 100, 20), (11, 11), 4)
    frame[100:200, 100:220] = board
    corners = _corners_at(160, 150, spread=45)

    assert blur_score(frame, corners) < blur_score(frame) / 2


def test_blur_score_falls_back_to_full_frame_without_corners():
    img = _checkerboard()
    assert blur_score(img, None) == pytest.approx(blur_score(img))


# ---------------------------------------------------------------------------
# SteadyTracker
# ---------------------------------------------------------------------------

def test_steady_requires_a_run_of_still_frames():
    tr = SteadyTracker(max_motion_px=2.0, frames=4)
    corners = _corners_at(640, 360)

    assert tr.update(corners) is False               # first frame primes state
    for _ in range(3):
        assert tr.update(corners) is False
    assert tr.update(corners) is True                # 4 still frames elapsed


def test_movement_breaks_the_steady_run():
    tr = SteadyTracker(max_motion_px=2.0, frames=3)
    for i in range(6):
        tr.update(_corners_at(640, 360))
    assert tr.is_steady

    tr.update(_corners_at(700, 360))                 # jumped 60 px
    assert not tr.is_steady


def test_losing_the_board_resets_steadiness():
    tr = SteadyTracker(max_motion_px=2.0, frames=2)
    for _ in range(5):
        tr.update(_corners_at(640, 360))
    assert tr.is_steady

    tr.update(None)
    assert not tr.is_steady


# ---------------------------------------------------------------------------
# TiltTracker
# ---------------------------------------------------------------------------

def test_tilt_bins_classify_by_angle_and_azimuth():
    tr = TiltTracker(IMG_SIZE)

    assert tr.bin_of(3.0, 0.0) == "frontal"
    assert tr.bin_of(25.0, 0.0) == "right"
    assert tr.bin_of(25.0, 90.0) == "down"
    assert tr.bin_of(25.0, 180.0) == "left"
    assert tr.bin_of(25.0, 270.0) == "up"


def test_extreme_tilt_is_rejected():
    """Beyond ~50 deg the board is so foreshortened that corner localisation
    degrades, so such views are not credited to any bin."""
    tr = TiltTracker(IMG_SIZE)
    assert tr.bin_of(75.0, 0.0) is None


def test_commit_tracks_which_bins_are_filled():
    tr = TiltTracker(IMG_SIZE)

    assert tr.commit("frontal") is True
    assert tr.commit("frontal") is False             # already had it
    assert tr.commit(None) is False
    assert tr.filled == {"frontal"}
    assert set(tr.missing()) == set(TILT_BINS) - {"frontal"}


def test_pose_of_recovers_a_frontal_board():
    board, _ = make_charuco_board(7, 5, 0.03, 0.022, "DICT_5X5_100")
    obj = np.asarray(board.getChessboardCorners(), dtype=np.float64)
    K = np.array([[0.8 * 1280, 0, 640.0], [0, 0.8 * 1280, 360.0], [0, 0, 1.0]])
    rvec = np.zeros((3, 1))
    tvec = np.array([[-0.105], [-0.075], [0.6]])
    img, _ = cv2.projectPoints(obj, rvec, tvec, K, np.zeros(5))

    tr = TiltTracker(IMG_SIZE)
    pose = tr.pose_of(img.astype(np.float32),
                      np.arange(len(obj)).reshape(-1, 1).astype(np.int32), board)

    assert pose is not None
    tilt, _az = pose
    assert tilt < 10.0
    assert tr.bin_of(*pose) == "frontal"


def test_pose_of_returns_none_without_enough_corners():
    board, _ = make_charuco_board(7, 5, 0.03, 0.022, "DICT_5X5_100")
    tr = TiltTracker(IMG_SIZE)

    assert tr.pose_of(None, None, board) is None
    assert tr.pose_of(_corners_at(640, 360, n=1), np.zeros((1, 1), np.int32), board) is None


# ---------------------------------------------------------------------------
# AutoCollectPolicy
# ---------------------------------------------------------------------------

def _policy(**kw):
    board, _ = make_charuco_board(7, 5, 0.03, 0.022, "DICT_5X5_100")
    kw.setdefault("target_views", 4)
    kw.setdefault("blur_min", 10.0)
    kw.setdefault("steady_frames", 1)
    kw.setdefault("cooldown_s", 0.5)
    return AutoCollectPolicy(IMG_SIZE, board, **kw)


def _sharp_gray():
    g = np.zeros((720, 1280), dtype=np.uint8)
    g[:] = _checkerboard(1280, 720, 40)
    return g


def test_failed_detection_yields_guidance_and_no_gates():
    pol = _policy()
    st = pol.evaluate(_det(None, ok=False, reason="no_markers"), _sharp_gray(), 0.0)

    assert not st.detect_ok
    assert not st.all_ok
    assert "No board" in st.guidance


@pytest.mark.parametrize("reason,fragment", [
    ("too_few_markers", "closer"),
    ("too_few_charuco", "closer"),
])
def test_detection_reason_drives_the_guidance_text(reason, fragment):
    pol = _policy()
    st = pol.evaluate(_det(None, ok=False, reason=reason), _sharp_gray(), 0.0)

    assert fragment in st.guidance


def test_blurry_frame_is_rejected_with_hold_still_guidance():
    pol = _policy(blur_min=1e9)                      # nothing can be sharp enough
    corners = _corners_at(640, 360)
    st = pol.evaluate(_det(corners), _sharp_gray(), 0.0)

    assert st.detect_ok and not st.sharp_ok
    assert not st.all_ok
    assert "Blurry" in st.guidance


def test_moving_board_is_not_steady():
    pol = _policy(steady_frames=3)
    gray = _sharp_gray()

    pol.evaluate(_det(_corners_at(300, 200)), gray, 0.0)
    st = pol.evaluate(_det(_corners_at(600, 400)), gray, 0.1)

    assert not st.steady_ok
    assert "Hold still" in st.guidance


def test_a_good_frame_passes_every_gate():
    pol = _policy()
    gray = _sharp_gray()
    corners = _corners_at(213, 120)

    st = None
    for i in range(4):
        st = pol.evaluate(_det(corners), gray, i * 0.01)
    assert st.all_ok, st.guidance


def test_repeating_the_same_view_fails_novelty():
    pol = _policy()
    gray = _sharp_gray()
    corners = _corners_at(213, 120)

    for i in range(4):
        st = pol.evaluate(_det(corners), gray, i * 0.01)
    pol.accept(_det(corners), 0.05)

    for i in range(4):
        st = pol.evaluate(_det(corners), gray, 1.0 + i * 0.01)
    assert not st.novel_ok
    assert not st.all_ok


def test_moving_to_a_new_cell_restores_novelty():
    pol = _policy()
    gray = _sharp_gray()
    first = _corners_at(213, 120)

    for i in range(4):
        pol.evaluate(_det(first), gray, i * 0.01)
    pol.accept(_det(first), 0.05)

    second = _corners_at(1066, 600)                  # a different cell
    for i in range(4):
        st = pol.evaluate(_det(second), gray, 1.0 + i * 0.01)
    assert st.novel_ok
    assert st.new_cells


def test_cooldown_blocks_an_immediate_second_capture():
    pol = _policy(cooldown_s=5.0)
    gray = _sharp_gray()
    corners = _corners_at(213, 120)

    for i in range(4):
        pol.evaluate(_det(corners), gray, i * 0.01)
    pol.accept(_det(corners), 0.05)

    st = pol.evaluate(_det(_corners_at(1066, 600)), gray, 0.10)
    assert not st.cooldown_ok
    assert not st.all_ok


def test_accept_advances_progress_and_coverage():
    pol = _policy()
    corners = _corners_at(213, 120)
    pol.accept(_det(corners), 0.0)

    prog = pol.progress()
    assert prog["views"] == 1
    assert prog["cells"] == 1
    assert prog["total_cells"] == 9


def test_undo_steps_the_count_back():
    pol = _policy()
    pol.accept(_det(_corners_at(213, 120)), 0.0)
    pol.undo()

    assert pol.progress()["views"] == 0
    assert pol.undo() is None                        # never goes negative
    assert pol.progress()["views"] == 0


def test_done_requires_views_coverage_and_tilt_diversity():
    pol = _policy(target_views=2, min_tilt_bins=2)
    w, h = IMG_SIZE

    # plenty of views and full coverage, but no tilt variety yet
    for r in range(3):
        for c in range(3):
            pol.accept(_det(_corners_at((c + 0.5) * w / 3, (r + 0.5) * h / 3)), 0.0)
    assert not pol.coverage.missing()
    assert pol.progress()["views"] >= 2
    assert not pol.done()                            # tilt bins still empty

    pol.tilt.commit("frontal")
    pol.tilt.commit("left")
    assert pol.done()


def test_guidance_directs_the_user_to_an_uncovered_cell():
    pol = _policy()
    gray = _sharp_gray()
    w, h = IMG_SIZE
    for r in range(3):
        for c in range(3):
            if (r, c) != (0, 2):
                pol.accept(_det(_corners_at((c + 0.5) * w / 3, (r + 0.5) * h / 3)), 0.0)

    # park the board, unmoving, on a cell that is already covered
    repeat = _corners_at(1.5 * w / 3, 1.5 * h / 3)
    pol.accept(_det(repeat), 9.0)
    for i in range(4):
        st = pol.evaluate(_det(repeat), gray, 10.0 + i * 0.01)

    assert not st.novel_ok
    assert "TOP-RIGHT" in st.guidance


def test_guidance_points_at_the_nearest_gap_not_the_first_one():
    """Two cells open: the user is sent to whichever is closer to the board."""
    pol = _policy()
    gray = _sharp_gray()
    w, h = IMG_SIZE
    open_cells = {(0, 0), (2, 2)}
    for r in range(3):
        for c in range(3):
            if (r, c) not in open_cells:
                pol.accept(_det(_corners_at((c + 0.5) * w / 3, (r + 0.5) * h / 3)), 0.0)

    near_bottom_right = _corners_at(2.5 * w / 3, 1.5 * h / 3)
    pol.accept(_det(near_bottom_right), 9.0)
    for i in range(4):
        st = pol.evaluate(_det(near_bottom_right), gray, 10.0 + i * 0.01)

    assert "BOTTOM-RIGHT" in st.guidance
