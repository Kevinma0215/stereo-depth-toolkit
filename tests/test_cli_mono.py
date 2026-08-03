"""Mono CLI commands, end to end on synthetic images (no camera)."""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from stereo_depth.adapters.calibration.charuco_calibrator import make_charuco_board
from stereo_depth.adapters.calibration.mono_yaml_repo import (
    SCHEMA,
    MonoYamlCalibrationRepo,
)
from stereo_depth.cli.app import app

runner = CliRunner()

BOARD = ("--squares-x", "7", "--squares-y", "5",
         "--square-length", "0.03", "--marker-length", "0.022",
         "--dict-name", "DICT_5X5_100")


IMG_W, IMG_H = 960, 720

# Ground-truth camera the synthetic views are rendered through.
K_TRUE = np.array([[700.0, 0.0, IMG_W / 2 - 8.0],
                   [0.0, 700.0, IMG_H / 2 + 6.0],
                   [0.0, 0.0, 1.0]])


def _render_views(out_dir, n=24, size=(IMG_W, IMG_H), seed=0):
    """Render the board into real PNGs through a known camera and pose.

    The board is planar, so a distortion-free view of it is exactly the
    homography ``K [r1 r2 t]``. Building the warp that way keeps every view
    consistent with one intrinsic matrix — arbitrary quadrilaterals are not
    realizable by any single camera and would make the fit meaningless.
    """
    board, _ = make_charuco_board(7, 5, 0.03, 0.022, "DICT_5X5_100")
    tw, th = 560, 400                       # 1.4 aspect == 0.21 m x 0.15 m
    tile = board.generateImage((tw, th))
    if tile.ndim == 2:
        tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)

    board_w, board_h = 7 * 0.03, 5 * 0.03
    # tile pixels -> board-plane metres
    S = np.array([[board_w / tw, 0.0, 0.0],
                  [0.0, board_h / th, 0.0],
                  [0.0, 0.0, 1.0]])
    centre = np.array([board_w / 2, board_h / 2, 0.0])

    tilts = [
        (np.array([0.0, 0.0, 1.0]), 0.0),
        (np.array([0.0, 1.0, 0.0]), +20.0),
        (np.array([0.0, 1.0, 0.0]), -20.0),
        (np.array([1.0, 0.0, 0.0]), +18.0),
        (np.array([1.0, 0.0, 0.0]), -18.0),
    ]
    grid = [(x, y) for y in (-0.20, 0.0, 0.20) for x in (-0.28, 0.0, 0.28)]

    w, h = size
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        axis, deg = tilts[i % len(tilts)]
        rvec = axis * np.deg2rad(deg) + rng.normal(0.0, 0.05, 3)
        R, _ = cv2.Rodrigues(rvec.reshape(3, 1))

        xn, yn = grid[i % len(grid)]
        z = float(rng.uniform(0.45, 0.75))
        t = np.array([xn * z, yn * z, z]) - R @ centre

        H = K_TRUE @ np.column_stack([R[:, 0], R[:, 1], t]) @ S
        canvas = np.full((h, w, 3), 210, dtype=np.uint8)
        cv2.warpPerspective(tile, H, (w, h), dst=canvas,
                            borderMode=cv2.BORDER_TRANSPARENT)
        cv2.imwrite(str(out_dir / f"{i:04d}.png"), canvas)
    return n


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    """One rendered dataset + one calibration, shared by the read-only tests."""
    root = tmp_path_factory.mktemp("mono")
    data = root / "views"
    _render_views(data)
    calib = root / "calib" / "mono.yaml"

    result = runner.invoke(app, [
        "calibrate-mono", "--data", str(data), "--out", str(calib),
        "--min-views", "10", *BOARD,
    ])
    assert result.exit_code == 0, result.output
    return {
        "root": root,
        "data": data,
        "calib": calib,
        "doc": yaml.safe_load(calib.read_text()),
        "output": result.output,
    }


# ---------------------------------------------------------------------------
# Help smoke tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "cmd", ["devices", "capture-mono", "calibrate-mono", "undistort"]
)
def test_command_is_registered_and_has_help(cmd):
    result = runner.invoke(app, [cmd, "--help"])
    assert result.exit_code == 0, result.output
    assert cmd.split("-")[0] in result.output.lower() or "Usage" in result.output


def test_mono_commands_appear_in_the_root_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("devices", "capture-mono", "calibrate-mono", "undistort"):
        assert cmd in result.output


# ---------------------------------------------------------------------------
# devices
# ---------------------------------------------------------------------------

def test_devices_lists_what_the_scanner_returns(monkeypatch):
    from stereo_depth.adapters.camera.v4l2_devices import FormatInfo, VideoDeviceInfo
    import stereo_depth.cli.devices_cmd as devices_cmd

    fake = [VideoDeviceInfo(
        path="/dev/video7", name="Wide Angle Cam", usb_id="1e4e:0100",
        formats=[FormatInfo("MJPG", [(1920, 1080)], {(1920, 1080): [30.0]})],
    )]
    monkeypatch.setattr(devices_cmd, "list_video_devices", lambda **kw: fake)

    result = runner.invoke(app, ["devices"])

    assert result.exit_code == 0
    assert "/dev/video7" in result.output
    assert "Wide Angle Cam" in result.output
    assert "1920x1080" in result.output


def test_devices_passes_the_probe_flag(monkeypatch):
    import stereo_depth.cli.devices_cmd as devices_cmd
    seen = {}

    def fake(**kw):
        seen.update(kw)
        return []

    monkeypatch.setattr(devices_cmd, "list_video_devices", fake)

    assert runner.invoke(app, ["devices", "--probe"]).exit_code == 0
    assert seen["probe"] is True


def test_devices_reports_an_empty_scan_gracefully(monkeypatch):
    import stereo_depth.cli.devices_cmd as devices_cmd
    monkeypatch.setattr(devices_cmd, "list_video_devices", lambda **kw: [])

    result = runner.invoke(app, ["devices"])

    assert result.exit_code == 0
    assert "No /dev/video*" in result.output


# ---------------------------------------------------------------------------
# calibrate-mono
# ---------------------------------------------------------------------------

def test_calibrate_mono_end_to_end(session):
    doc = session["doc"]

    assert session["calib"].exists()
    assert doc["schema"] == SCHEMA
    assert doc["model"] in ("pinhole", "rational", "fisheye")
    assert np.array(doc["K"]).shape == (3, 3)
    assert doc["image_size"] == {"width": IMG_W, "height": IMG_H}
    assert doc["views_used"] >= 10


def test_calibration_recovers_the_camera_it_was_rendered_through(session):
    """The strongest end-to-end check: intrinsics come back out."""
    K = np.array(session["doc"]["K"])

    assert K[0, 0] == pytest.approx(K_TRUE[0, 0], rel=0.05)
    assert K[1, 1] == pytest.approx(K_TRUE[1, 1], rel=0.05)
    assert K[0, 2] == pytest.approx(K_TRUE[0, 2], abs=0.05 * IMG_W)
    assert K[1, 2] == pytest.approx(K_TRUE[1, 2], abs=0.05 * IMG_H)
    assert session["doc"]["rpe_px"] < 1.0


def test_calibrate_mono_writes_isaac_and_undistort_sections(session):
    doc = session["doc"]

    isaac = doc["isaac_sim"]
    assert "usd_camera_raw" in isaac
    assert "usd_camera_undistorted" in isaac
    assert ("opencv_pinhole" in isaac) or ("opencv_fisheye" in isaac)
    assert isaac["usd_camera_raw"]["focal_length_mm"] > 0

    und = doc["undistort"]
    assert np.array(und["alpha_0"]["K_new"]).shape == (3, 3)
    assert np.array(und["alpha_1"]["K_new"]).shape == (3, 3)
    assert "RAW" in und["note"] and "UNDISTORTED" in und["note"]


def test_isaac_focal_length_matches_the_calibrated_fx(session):
    doc = session["doc"]
    usd = doc["isaac_sim"]["usd_camera_raw"]

    fx_back = usd["focal_length_mm"] * IMG_W / usd["horizontal_aperture_mm"]
    assert fx_back == pytest.approx(np.array(doc["K"])[0, 0])


def test_calibrate_mono_reports_every_model_it_tried(session):
    doc = session["doc"]

    assert set(doc["models"]) == {"pinhole", "rational", "fisheye"}
    assert doc["selection"]["reason"]
    assert "Distortion model comparison" in session["output"]


def test_distortion_free_renders_select_the_simplest_model(session):
    """Nothing in these images justifies extra coefficients."""
    assert session["doc"]["model"] == "pinhole"
    assert len(session["doc"]["D"]) == 5


def test_forcing_a_model_skips_selection(tmp_path):
    data = tmp_path / "views"
    _render_views(data)
    out = tmp_path / "mono.yaml"

    result = runner.invoke(app, [
        "calibrate-mono", "--data", str(data), "--out", str(out),
        "--model", "pinhole", "--min-views", "10", *BOARD,
    ])

    assert result.exit_code == 0, result.output
    doc = yaml.safe_load(out.read_text())
    assert doc["model"] == "pinhole"
    assert doc["selection"]["forced"] is True
    assert len(doc["D"]) == 5


def test_report_json_is_written_alongside(session):
    report = session["calib"].with_suffix(".report.json")

    assert report.exists()
    data_json = json.loads(report.read_text())
    assert data_json["status"] == "success"
    assert data_json["selected_model"] in ("pinhole", "rational", "fisheye")
    assert data_json["per_view_rpe"]
    assert all("view" in r and "rpe_px" in r for r in data_json["per_view_rpe"])


def test_missing_images_fails_but_still_writes_a_report(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    out = tmp_path / "mono.yaml"

    result = runner.invoke(app, ["calibrate-mono", "--data", str(empty),
                                 "--out", str(out), *BOARD])

    assert result.exit_code != 0
    report = json.loads(out.with_suffix(".report.json").read_text())
    assert report["status"] == "failed"
    assert report["reason"] == "no_images_found"


def test_too_few_views_fails_with_a_diagnostic_report(tmp_path):
    data = tmp_path / "views"
    _render_views(data, n=4)
    out = tmp_path / "mono.yaml"

    result = runner.invoke(app, ["calibrate-mono", "--data", str(data),
                                 "--out", str(out), "--min-views", "20", *BOARD])

    assert result.exit_code != 0
    report = json.loads(out.with_suffix(".report.json").read_text())
    assert report["status"] == "failed"
    assert report["reason"] == "not_enough_valid_views"
    assert "collect" in report


# ---------------------------------------------------------------------------
# undistort
# ---------------------------------------------------------------------------

def test_undistort_batch_writes_images_and_matching_intrinsics(session, tmp_path):
    out = tmp_path / "undistorted"
    result = runner.invoke(app, ["undistort", "--calib", str(session["calib"]),
                                 "--images", str(session["data"]), "--out", str(out)])

    assert result.exit_code == 0, result.output
    assert len(list(out.glob("*.png"))) >= 10

    sidecar = out / "undistorted_intrinsics.yaml"
    doc = yaml.safe_load(sidecar.read_text())
    assert doc["model"] == "pinhole"
    assert doc["D"] == [0.0, 0.0, 0.0, 0.0, 0.0]

    # the sidecar K must be K_new, not the raw K
    raw_K = np.array(session["doc"]["K"])
    assert not np.allclose(np.array(doc["K"]), raw_K, atol=1.0)
    assert doc["source"]["calibration"] == str(session["calib"])


def test_undistorted_images_keep_their_size_and_name(session, tmp_path):
    out = tmp_path / "undistorted"
    runner.invoke(app, ["undistort", "--calib", str(session["calib"]),
                        "--images", str(session["data"]), "--out", str(out)])

    src = sorted(session["data"].glob("*.png"))[0]
    dst = out / src.name
    assert dst.exists()
    assert cv2.imread(str(dst)).shape == cv2.imread(str(src)).shape


def test_undistort_output_warns_about_the_pairing(session, tmp_path):
    out = tmp_path / "undistorted"
    result = runner.invoke(app, ["undistort", "--calib", str(session["calib"]),
                                 "--images", str(session["data"]), "--out", str(out)])

    assert "pair with K_new" in result.output


def test_undistort_rejects_a_stereo_calibration_file(tmp_path):
    stereo = tmp_path / "stereo.yaml"
    stereo.write_text(yaml.safe_dump({"K1": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}))

    result = runner.invoke(app, ["undistort", "--calib", str(stereo),
                                 "--images", str(tmp_path), "--out", str(tmp_path / "o")])

    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "not a mono calibration file" in str(result.exception)


def test_batch_mode_requires_images_and_out(session):
    result = runner.invoke(app, ["undistort", "--calib", str(session["calib"])])

    assert result.exit_code != 0
    assert "--images" in str(result.exception)


# ---------------------------------------------------------------------------
# Repo round-trip
# ---------------------------------------------------------------------------

def test_repo_reads_back_what_the_cli_wrote(session):
    intr = MonoYamlCalibrationRepo().load(str(session["calib"]))

    assert intr.model in ("pinhole", "rational", "fisheye")
    assert intr.K.shape == (3, 3)
    assert intr.image_size == (IMG_W, IMG_H)
    assert intr.views_used >= 10


def test_repo_rejects_a_stereo_file(tmp_path):
    stereo = tmp_path / "stereo.yaml"
    stereo.write_text(yaml.safe_dump({"K1": [[1, 0, 0]], "image_size": {"width": 1, "height": 1}}))

    with pytest.raises(ValueError, match="not a mono calibration file"):
        MonoYamlCalibrationRepo().load(str(stereo))
