"""V4L2 enumeration — parser tested against a captured transcript."""
from __future__ import annotations

import pytest

from stereo_depth.adapters.camera import v4l2_devices
from stereo_depth.adapters.camera.v4l2_devices import (
    FormatInfo,
    VideoDeviceInfo,
    _parse_v4l2_formats,
    format_device_table,
    list_video_devices,
)

# Real `v4l2-ctl -d /dev/video0 --list-formats-ext` output shape.
TRANSCRIPT = """ioctl: VIDIOC_ENUM_FMT
\tType: Video Capture

\t[0]: 'MJPG' (Motion-JPEG, compressed)
\t\tSize: Discrete 2560x720
\t\t\tInterval: Discrete 0.033s (30.000 fps)
\t\t\tInterval: Discrete 0.067s (15.000 fps)
\t\tSize: Discrete 1280x480
\t\t\tInterval: Discrete 0.017s (60.000 fps)
\t\t\tInterval: Discrete 0.033s (30.000 fps)
\t[1]: 'YUYV' (YUYV 4:2:2)
\t\tSize: Discrete 2560x720
\t\t\tInterval: Discrete 0.200s (5.000 fps)
"""

STEPWISE_TRANSCRIPT = """ioctl: VIDIOC_ENUM_FMT
\tType: Video Capture

\t[0]: 'MJPG' (Motion-JPEG, compressed)
\t\tSize: Stepwise 32x32 - 2592x1944 with step 2/2
"""


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def test_parses_every_pixel_format():
    formats = _parse_v4l2_formats(TRANSCRIPT)
    assert [f.fourcc for f in formats] == ["MJPG", "YUYV"]


def test_parses_discrete_sizes_per_format():
    mjpg, yuyv = _parse_v4l2_formats(TRANSCRIPT)

    assert mjpg.sizes == [(2560, 720), (1280, 480)]
    assert yuyv.sizes == [(2560, 720)]


def test_frame_rates_attach_to_the_right_size():
    mjpg, yuyv = _parse_v4l2_formats(TRANSCRIPT)

    assert mjpg.fps[(2560, 720)] == [30.0, 15.0]
    assert mjpg.fps[(1280, 480)] == [60.0, 30.0]
    assert yuyv.fps[(2560, 720)] == [5.0]


def test_best_fps_returns_the_highest_rate():
    mjpg, _ = _parse_v4l2_formats(TRANSCRIPT)

    assert mjpg.best_fps((2560, 720)) == 30.0
    assert mjpg.best_fps((1280, 480)) == 60.0
    assert mjpg.best_fps((640, 480)) is None


def test_stepwise_ranges_report_their_maximum():
    formats = _parse_v4l2_formats(STEPWISE_TRANSCRIPT)

    assert formats[0].sizes == [(2592, 1944)]


def test_empty_output_yields_no_formats():
    assert _parse_v4l2_formats("") == []
    assert _parse_v4l2_formats("ioctl: VIDIOC_ENUM_FMT\n\tType: Video Capture\n") == []


def test_rates_before_any_format_are_ignored():
    """Stray lines must not crash the parser or invent a format."""
    assert _parse_v4l2_formats("Interval: Discrete 0.033s (30.000 fps)\n") == []


# ---------------------------------------------------------------------------
# VideoDeviceInfo
# ---------------------------------------------------------------------------

def test_index_is_taken_from_the_node_path():
    assert VideoDeviceInfo(path="/dev/video0").index == 0
    assert VideoDeviceInfo(path="/dev/video12").index == 12


def test_max_size_picks_the_largest_area_across_formats():
    dev = VideoDeviceInfo(path="/dev/video0", formats=_parse_v4l2_formats(TRANSCRIPT))
    assert dev.max_size() == (2560, 720)


def test_max_size_is_none_without_formats():
    assert VideoDeviceInfo(path="/dev/video0").max_size() is None


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------

def test_no_devices_gives_actionable_advice():
    out = format_device_table([])
    assert "No /dev/video*" in out
    assert "Plug in the camera" in out


def test_table_shows_name_usb_id_and_modes():
    dev = VideoDeviceInfo(
        path="/dev/video0", name="HBVCAM-W202011HD", usb_id="1e4e:0100",
        formats=_parse_v4l2_formats(TRANSCRIPT),
    )
    out = format_device_table([dev])

    assert "/dev/video0" in out
    assert "HBVCAM-W202011HD" in out
    assert "1e4e:0100" in out
    assert "2560x720" in out
    assert "30 fps" in out
    assert "--path /dev/video0" in out


def test_table_flags_nodes_belonging_to_the_same_camera():
    shared = "/sys/devices/usb1/1-1"
    devs = [
        VideoDeviceInfo(path="/dev/video0", name="cam", group=shared),
        VideoDeviceInfo(path="/dev/video1", name="cam", group=shared),
    ]
    out = format_device_table(devs)

    assert "same camera as /dev/video0" in out


def test_table_reports_probe_results_and_errors():
    devs = [
        VideoDeviceInfo(path="/dev/video0", capture_ok=True),
        VideoDeviceInfo(path="/dev/video1", capture_ok=False, error="device busy"),
    ]
    out = format_device_table(devs)

    assert "[capture OK]" in out
    assert "[no frames]" in out
    assert "device busy" in out


def test_listing_is_empty_when_no_nodes_exist(monkeypatch):
    monkeypatch.setattr(v4l2_devices.glob, "glob", lambda pattern: [])
    assert list_video_devices() == []


def test_listing_reads_names_and_formats_per_node(monkeypatch):
    monkeypatch.setattr(
        v4l2_devices.glob, "glob", lambda pattern: ["/dev/video1", "/dev/video0"]
    )
    monkeypatch.setattr(v4l2_devices, "_sysfs_name", lambda node: f"cam-{node}")
    monkeypatch.setattr(v4l2_devices, "_device_group", lambda node: ("grp", "1e4e:0100"))
    monkeypatch.setattr(
        v4l2_devices, "_query_formats",
        lambda path, timeout=5.0: (_parse_v4l2_formats(TRANSCRIPT), None),
    )

    devs = list_video_devices()

    assert [d.path for d in devs] == ["/dev/video0", "/dev/video1"]   # sorted
    assert devs[0].name == "cam-video0"
    assert devs[0].usb_id == "1e4e:0100"
    assert devs[0].formats[0].fourcc == "MJPG"
    assert all(d.capture_ok is None for d in devs)                    # probe off


def test_probe_failure_on_one_node_does_not_abort_the_scan(monkeypatch):
    monkeypatch.setattr(v4l2_devices.glob, "glob", lambda pattern: ["/dev/video0"])
    monkeypatch.setattr(v4l2_devices, "_sysfs_name", lambda node: "cam")
    monkeypatch.setattr(v4l2_devices, "_device_group", lambda node: (None, None))
    monkeypatch.setattr(v4l2_devices, "_query_formats", lambda path, timeout=5.0: ([], None))

    def boom(path):
        raise RuntimeError("camera on fire")

    monkeypatch.setattr(v4l2_devices, "_probe_capture", boom)

    devs = list_video_devices(probe=True)

    assert len(devs) == 1
    assert devs[0].capture_ok is False
    assert "camera on fire" in devs[0].error


def test_missing_v4l2_ctl_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(v4l2_devices.shutil, "which", lambda name: None)
    formats, error = v4l2_devices._query_formats("/dev/video0")

    assert formats == []
    assert "v4l-utils" in error
