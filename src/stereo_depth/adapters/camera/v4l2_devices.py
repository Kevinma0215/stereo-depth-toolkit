"""Enumerate V4L2 capture devices so the user can find their camera.

A single physical camera usually registers several ``/dev/videoN`` nodes and
only the first actually streams, so this reports the sysfs card name, groups
nodes belonging to the same USB device, and can probe which ones deliver
frames.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

SYSFS_ROOT = Path("/sys/class/video4linux")

_FMT_RE = re.compile(r"^\s*\[\d+\]:\s*'(\w+)'")
_SIZE_RE = re.compile(r"^\s*Size:\s*Discrete\s+(\d+)x(\d+)")
_STEPWISE_RE = re.compile(r"^\s*Size:\s*Stepwise\s+(\d+)x(\d+)\s*-\s*(\d+)x(\d+)")
_FPS_RE = re.compile(r"\(([\d.]+)\s*fps\)")


@dataclass
class FormatInfo:
    fourcc: str
    sizes: list[tuple[int, int]] = field(default_factory=list)
    fps: dict[tuple[int, int], list[float]] = field(default_factory=dict)

    def best_fps(self, size: tuple[int, int]) -> float | None:
        rates = self.fps.get(size)
        return max(rates) if rates else None


@dataclass
class VideoDeviceInfo:
    path: str
    name: str = ""
    usb_id: str | None = None
    group: str | None = None          # shared by nodes of one physical camera
    formats: list[FormatInfo] = field(default_factory=list)
    capture_ok: bool | None = None    # only set when probe=True
    error: str | None = None

    @property
    def index(self) -> int:
        m = re.search(r"(\d+)$", self.path)
        return int(m.group(1)) if m else -1

    def max_size(self) -> tuple[int, int] | None:
        sizes = [s for f in self.formats for s in f.sizes]
        return max(sizes, key=lambda s: s[0] * s[1]) if sizes else None


def _parse_v4l2_formats(text: str) -> list[FormatInfo]:
    """Parse ``v4l2-ctl --list-formats-ext`` output.

    Pure function so it can be tested against a captured transcript without
    any camera attached.
    """
    formats: list[FormatInfo] = []
    current: FormatInfo | None = None
    current_size: tuple[int, int] | None = None

    for line in text.splitlines():
        m = _FMT_RE.match(line)
        if m:
            current = FormatInfo(fourcc=m.group(1))
            formats.append(current)
            current_size = None
            continue
        if current is None:
            continue

        m = _SIZE_RE.match(line)
        if m:
            current_size = (int(m.group(1)), int(m.group(2)))
            if current_size not in current.sizes:
                current.sizes.append(current_size)
            continue

        m = _STEPWISE_RE.match(line)
        if m:
            # report only the maximum of a stepwise range
            current_size = (int(m.group(3)), int(m.group(4)))
            if current_size not in current.sizes:
                current.sizes.append(current_size)
            continue

        m = _FPS_RE.search(line)
        if m and current_size is not None:
            current.fps.setdefault(current_size, []).append(float(m.group(1)))

    return formats


def _sysfs_name(node: str) -> str:
    try:
        return (SYSFS_ROOT / node / "name").read_text().strip()
    except OSError:
        return ""


def _device_group(node: str) -> tuple[str | None, str | None]:
    """(group key, usb vendor:product) for the node's physical device."""
    link = SYSFS_ROOT / node / "device"
    try:
        target = os.path.realpath(link)
    except OSError:
        return None, None
    if not os.path.exists(target):
        return None, None

    usb_id = None
    probe = Path(target)
    for _ in range(4):                      # walk up to the USB device dir
        vid, pid = probe / "idVendor", probe / "idProduct"
        if vid.exists() and pid.exists():
            try:
                usb_id = f"{vid.read_text().strip()}:{pid.read_text().strip()}"
                target = str(probe)
            except OSError:
                pass
            break
        probe = probe.parent

    return target, usb_id


def _query_formats(path: str, timeout: float = 5.0) -> tuple[list[FormatInfo], str | None]:
    if not shutil.which("v4l2-ctl"):
        return [], "v4l2-ctl not found (install v4l-utils for the format list)"
    try:
        proc = subprocess.run(
            ["v4l2-ctl", "-d", path, "--list-formats-ext"],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return [], str(e)
    if proc.returncode != 0:
        return [], (proc.stderr or "").strip() or f"v4l2-ctl exited {proc.returncode}"
    return _parse_v4l2_formats(proc.stdout), None


def _probe_capture(path: str) -> bool:
    """Open the node and try to read one frame."""
    import cv2

    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    try:
        if not cap.isOpened():
            return False
        ok, frame = cap.read()
        return bool(ok and frame is not None)
    finally:
        cap.release()


def list_video_devices(*, probe: bool = False) -> list[VideoDeviceInfo]:
    """All ``/dev/video*`` nodes with their names and supported formats.

    ``probe=True`` additionally opens each node and reads a frame, which
    distinguishes real capture nodes from the metadata-only ones a single
    camera also registers.
    """
    devices: list[VideoDeviceInfo] = []
    for path in sorted(glob.glob("/dev/video*"), key=lambda p: (len(p), p)):
        node = os.path.basename(path)
        group, usb_id = _device_group(node)
        formats, error = _query_formats(path)
        info = VideoDeviceInfo(
            path=path,
            name=_sysfs_name(node),
            usb_id=usb_id,
            group=group,
            formats=formats,
            error=error,
        )
        if probe:
            try:
                info.capture_ok = _probe_capture(path)
            except Exception as e:            # a busy device must not abort the scan
                info.capture_ok = False
                info.error = info.error or str(e)
        devices.append(info)
    return devices


def format_device_table(devices: list[VideoDeviceInfo]) -> str:
    """Human-readable listing for the CLI."""
    if not devices:
        return (
            "No /dev/video* devices found.\n"
            "Plug in the camera and re-run. On WSL or inside a container the "
            "USB device may not be passed through."
        )

    lines: list[str] = []
    seen_groups: dict[str, str] = {}
    for d in devices:
        header = f"{d.path}"
        if d.name:
            header += f"  {d.name}"
        if d.usb_id:
            header += f"  [usb {d.usb_id}]"
        if d.group and d.group in seen_groups:
            header += f"  (same camera as {seen_groups[d.group]})"
        elif d.group:
            seen_groups[d.group] = d.path
        if d.capture_ok is True:
            header += "  [capture OK]"
        elif d.capture_ok is False:
            header += "  [no frames]"
        lines.append(header)

        if d.error:
            lines.append(f"    ! {d.error}")
        for f in d.formats:
            top = sorted(f.sizes, key=lambda s: s[0] * s[1], reverse=True)[:4]
            for w, h in top:
                fps = f.best_fps((w, h))
                rate = f"  @ {fps:g} fps" if fps else ""
                lines.append(f"    {f.fourcc:>5s}  {w}x{h}{rate}")
        lines.append("")

    lines.append(
        "Use the device path with --path, e.g.\n"
        "  stereo-depth capture-mono --out-dir data/mono/run1 --path /dev/video0"
    )
    return "\n".join(lines)
