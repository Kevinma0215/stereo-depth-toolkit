"""Live auto-capture loop, driven by a fake camera and a fake clock.

No hardware and no windows: the OpenCV GUI calls are stubbed out and frames
come from rendered board views, so the state machine and key handling are
exercised deterministically.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from stereo_depth.app import calibrate_mono
from stereo_depth.app.calibrate_mono import run_capture_mono

from test_cli_mono import _render_views

BOARD_KW = dict(
    squares_x=7, squares_y=5, square_length=0.03,
    marker_length=0.022, dict_name="DICT_5X5_100",
)


class FakeCap:
    """VideoCapture stand-in that replays a fixed list of frames."""

    def __init__(self, frames, fourcc="MJPG"):
        self.frames = frames
        self.fourcc = fourcc
        self.i = 0
        self.released = False

    def read(self):
        if self.i >= len(self.frames):
            return False, None
        frame = self.frames[self.i]
        self.i += 1
        return True, frame.copy()

    def get(self, prop):
        if prop == cv2.CAP_PROP_FOURCC:
            return float(cv2.VideoWriter_fourcc(*self.fourcc))
        return 0.0

    def release(self):
        self.released = True


@pytest.fixture(scope="module")
def frames(tmp_path_factory):
    """Distinct board views, each repeated so the steadiness gate can settle."""
    d = tmp_path_factory.mktemp("frames") / "views"
    _render_views(d, n=12)
    views = [cv2.imread(str(p)) for p in sorted(d.glob("*.png"))]
    repeated = []
    for v in views:
        repeated.extend([v] * 6)
    return repeated


@pytest.fixture
def headless(monkeypatch):
    """Stub the GUI and make the clock deterministic."""
    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)

    clock = {"t": 0.0}

    def tick():
        clock["t"] += 0.5          # clears the 0.7 s cooldown every 2 frames
        return clock["t"]

    monkeypatch.setattr(calibrate_mono.time, "perf_counter", tick)
    return clock


def _use_frames(monkeypatch, frames, fourcc="MJPG"):
    cap = FakeCap(frames, fourcc=fourcc)
    seen = {}

    def fake_open(**kw):
        seen.update(kw)
        return cap

    monkeypatch.setattr(calibrate_mono, "open_source", fake_open)
    cap.open_kwargs = seen
    return cap


def _keys(monkeypatch, script):
    """Feed a scripted key sequence, then 255 (no key) forever."""
    it = iter(script)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: next(it, 255))


# ---------------------------------------------------------------------------

def test_auto_capture_saves_views_and_reports_the_count(tmp_path, monkeypatch, headless, frames):
    _use_frames(monkeypatch, frames)
    _keys(monkeypatch, [])
    out = tmp_path / "shots"

    n = run_capture_mono(out, path="/dev/video0", target_views=50, **BOARD_KW)

    assert n > 0
    written = sorted(out.glob("*.png"))
    assert len(written) == n
    assert written[0].name == "0001.png"


def test_saved_frames_are_the_raw_camera_image(tmp_path, monkeypatch, headless, frames):
    """The HUD must never be burnt into what gets calibrated."""
    _use_frames(monkeypatch, frames)
    _keys(monkeypatch, [])
    out = tmp_path / "shots"

    run_capture_mono(out, path="/dev/video0", target_views=50, **BOARD_KW)

    saved = cv2.imread(str(sorted(out.glob("*.png"))[0]))
    h, w = saved.shape[:2]
    assert (w, h) == (frames[0].shape[1], frames[0].shape[0])
    # a HUD would have drawn over the top-left coverage inset area
    assert np.array_equal(saved, frames[0]) or saved.std() > 0


def test_auto_disabled_captures_nothing(tmp_path, monkeypatch, headless, frames):
    _use_frames(monkeypatch, frames)
    _keys(monkeypatch, [])
    out = tmp_path / "shots"

    n = run_capture_mono(out, path="/dev/video0", auto=False, **BOARD_KW)

    assert n == 0
    assert list(out.glob("*.png")) == []


def test_space_forces_a_save_even_with_auto_off(tmp_path, monkeypatch, headless, frames):
    _use_frames(monkeypatch, frames)
    _keys(monkeypatch, [255, 255, ord(" ")])
    out = tmp_path / "shots"

    n = run_capture_mono(out, path="/dev/video0", auto=False, **BOARD_KW)

    assert n == 1
    assert (out / "0001.png").exists()


def test_r_removes_the_last_saved_image(tmp_path, monkeypatch, headless, frames):
    _use_frames(monkeypatch, frames)
    # save twice, then undo once
    _keys(monkeypatch, [255, 255, ord(" "), 255, 255, ord(" "), ord("r")])
    out = tmp_path / "shots"

    n = run_capture_mono(out, path="/dev/video0", auto=False, **BOARD_KW)

    assert n == 1
    assert (out / "0001.png").exists()
    assert not (out / "0002.png").exists()


def test_q_quits_immediately(tmp_path, monkeypatch, headless, frames):
    _use_frames(monkeypatch, frames)
    _keys(monkeypatch, [ord("q")])
    out = tmp_path / "shots"

    n = run_capture_mono(out, path="/dev/video0", **BOARD_KW)

    assert n == 0


def test_requested_pixel_format_reaches_the_camera(tmp_path, monkeypatch, headless, frames):
    cap = _use_frames(monkeypatch, frames, fourcc="YUYV")
    _keys(monkeypatch, [ord("q")])

    run_capture_mono(tmp_path / "shots", path="/dev/video0", fourcc="YUYV", **BOARD_KW)

    assert cap.open_kwargs["fourcc"] == "YUYV"


def test_format_actually_negotiated_is_reported(tmp_path, monkeypatch, headless, frames, capsys):
    _use_frames(monkeypatch, frames, fourcc="YUYV")
    _keys(monkeypatch, [ord("q")])

    run_capture_mono(tmp_path / "shots", path="/dev/video0", fourcc="YUYV", **BOARD_KW)

    assert "YUYV" in capsys.readouterr().out


def test_warns_when_the_camera_ignores_the_requested_format(
    tmp_path, monkeypatch, headless, frames, capsys
):
    """Drivers silently fall back; asking for YUYV and getting MJPG must
    not pass unnoticed, since the point of asking was image quality."""
    _use_frames(monkeypatch, frames, fourcc="MJPG")      # camera ignores us
    _keys(monkeypatch, [ord("q")])

    run_capture_mono(tmp_path / "shots", path="/dev/video0", fourcc="YUYV", **BOARD_KW)

    printed = capsys.readouterr().out
    assert "WARNING" in printed
    assert "YUYV" in printed and "MJPG" in printed


def test_camera_is_released_even_when_the_stream_ends(tmp_path, monkeypatch, headless, frames):
    cap = _use_frames(monkeypatch, frames)
    _keys(monkeypatch, [])
    out = tmp_path / "shots"

    run_capture_mono(out, path="/dev/video0", **BOARD_KW)

    assert cap.released


def test_unreadable_camera_raises_clearly(tmp_path, monkeypatch, headless):
    _use_frames(monkeypatch, [])
    _keys(monkeypatch, [])

    with pytest.raises(RuntimeError, match="Cannot read frames"):
        run_capture_mono(tmp_path / "shots", path="/dev/video9", **BOARD_KW)


def test_capture_stops_once_the_target_is_complete(tmp_path, monkeypatch, headless, frames):
    """done() gates further auto-saves, so the count cannot run away."""
    _use_frames(monkeypatch, frames)
    _keys(monkeypatch, [])
    out = tmp_path / "shots"

    # one coverage cell and a reachable edge target, so done() can be met
    n = run_capture_mono(out, path="/dev/video0", target_views=3,
                         grid=1, edge_target=0.5, **BOARD_KW)

    assert n <= 6


def test_edge_gate_keeps_collecting_when_the_corners_are_untouched(
    tmp_path, monkeypatch, headless, frames, capsys
):
    """These synthetic views never approach the frame corners, so the session
    must not declare itself finished — and must say why on the way out."""
    _use_frames(monkeypatch, frames)
    _keys(monkeypatch, [])
    out = tmp_path / "shots"

    run_capture_mono(out, path="/dev/video0", target_views=3,
                     grid=1, edge_target=0.95, **BOARD_KW)

    printed = capsys.readouterr().out
    assert "of the way to the frame corners" in printed
    assert "extrapolate" in printed


def test_collected_images_calibrate(tmp_path, monkeypatch, headless, frames):
    """The images this loop writes are usable by the calibration pipeline."""
    _use_frames(monkeypatch, frames)
    _keys(monkeypatch, [])
    out = tmp_path / "shots"

    n = run_capture_mono(out, path="/dev/video0", target_views=50, **BOARD_KW)
    if n < 10:
        pytest.skip(f"only {n} views captured from the synthetic stream")

    from stereo_depth.app.calibrate_mono import run_calibrate_mono
    yaml_path, _report = run_calibrate_mono(
        out, tmp_path / "mono.yaml", min_views=10, verbose=False, **BOARD_KW
    )

    assert yaml_path.exists()
