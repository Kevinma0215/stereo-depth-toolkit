"""stereo_depth/camera_streamer.py — background-threaded camera capture + depth snapshot."""
from __future__ import annotations

import logging
import time
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np

from stereo_depth.adapters.camera.uvc_source import UvcSource
from stereo_depth.rolling_buffer import RollingBuffer

log = logging.getLogger(__name__)

_FPS_WINDOW      = 30   # timestamps kept for fps calculation
_FPS_WARN_BELOW  = 25.0


@dataclass
class SnapshotResult:
    frame_index:    int
    rgb_snapshot:   np.ndarray   # (H, W, 3) uint8 — processor.last_left_rect
    stable_depth:   np.ndarray   # (H, W) float32 metres
    process_time_s: float        # wall time for process_stack() call


class NotReadyError(RuntimeError):
    """Raised by snapshot() when RollingBuffer has fewer than maxlen frames."""


class CameraStreamer:
    """Continuously grabs SBS frames in a daemon thread and exposes depth snapshots.

    Usage::

        source    = UvcSource(device_index=0, width=2560, height=720)
        buffer    = RollingBuffer(maxlen=15)
        pipeline  = StereoPipeline(...)

        with CameraStreamer(source, buffer, pipeline) as streamer:
            while not streamer.is_ready:
                time.sleep(0.05)
            result = streamer.snapshot()
    """

    def __init__(
        self,
        source: UvcSource,
        buffer: RollingBuffer,
        processor,
    ) -> None:
        self._source    = source
        self._buffer    = buffer
        self._processor = processor

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Circular buffer of recent frame-arrival timestamps for fps calculation
        self._ts_lock  = threading.Lock()
        self._ts_deque: deque[float] = deque(maxlen=_FPS_WINDOW)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background capture thread. Returns immediately."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="CameraStreamer-capture",
            daemon=True,
        )
        self._thread.start()
        log.info("CameraStreamer started")

    def stop(self) -> None:
        """Signal the capture thread to stop and release the camera source."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._source.release()
        log.info("CameraStreamer stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def snapshot(self) -> SnapshotResult:
        """Freeze the buffer and run process_stack() on captured frames.

        Raises:
            NotReadyError: if the buffer is not yet full.
        """
        if not self._buffer.is_ready:
            raise NotReadyError(
                "RollingBuffer is not full yet — wait until is_ready is True."
            )

        head_index, frames = self._buffer.snapshot()

        t0             = time.monotonic()
        stable_depth   = self._processor.process_stack(frames)
        process_time_s = time.monotonic() - t0

        rgb_snapshot = self._processor.last_left_rect()

        log.debug(
            "snapshot(): frame_index=%d  process_time_s=%.3f",
            head_index, process_time_s,
        )

        return SnapshotResult(
            frame_index=head_index,
            rgb_snapshot=rgb_snapshot,
            stable_depth=stable_depth,
            process_time_s=process_time_s,
        )

    @property
    def is_ready(self) -> bool:
        """True when the rolling buffer holds a full window of frames."""
        return self._buffer.is_ready

    @property
    def fps(self) -> float:
        """Rolling average capture fps over the last 30 frame timestamps."""
        with self._ts_lock:
            ts = list(self._ts_deque)
        if len(ts) < 2:
            return 0.0
        elapsed = ts[-1] - ts[0]
        if elapsed <= 0.0:
            return 0.0
        return (len(ts) - 1) / elapsed

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "CameraStreamer":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Live preview
    # ------------------------------------------------------------------

    def preview(self, window: str = "CameraStreamer Preview") -> None:
        """Show a live OpenCV window of the raw SBS stream.

        Blocks until the user presses Q or ESC.  The capture thread keeps
        running in the background, so snapshot() still works while previewing.
        """
        import cv2

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1280, 360)   # half-scale: 2560×720 → 1280×360

        while True:
            _, frames = self._buffer.snapshot()
            if frames:
                frame = frames[-1]   # most recent SBS frame
                small = cv2.resize(frame, (1280, 360), interpolation=cv2.INTER_AREA)
                cv2.putText(
                    small,
                    f"FPS: {self.fps:.1f}  frames: {self._buffer.head_index}  Q quit",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 1, cv2.LINE_AA,
                )
                cv2.imshow(window, small)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

        cv2.destroyWindow(window)

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                pair = self._source.grab()
                sbs  = np.concatenate([pair.left, pair.right], axis=1)
                self._buffer.push(sbs)

                now = time.monotonic()
                with self._ts_lock:
                    self._ts_deque.append(now)

                current_fps = self.fps
                if current_fps > 0.0 and current_fps < _FPS_WARN_BELOW:
                    log.warning(
                        "CameraStreamer fps below threshold: %.1f fps (< %.0f)",
                        current_fps, _FPS_WARN_BELOW,
                    )

            except Exception:
                log.exception("CameraStreamer capture error — retrying in 1 s")
                time.sleep(1.0)
