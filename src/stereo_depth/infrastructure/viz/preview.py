from __future__ import annotations
import cv2

from stereo_depth.infrastructure.io.sinks import VideoRecorder
from stereo_depth.infrastructure.io.sbs_capture import SBSSplitter


def preview_sbs(
    cap: cv2.VideoCapture,
    splitter: SBSSplitter,
    recorder: VideoRecorder | None = None,
):
    win = "stereo-depth preview (q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if recorder is not None:
            recorder.maybe_init(frame, fps)
            recorder.write(frame)

        left, right = splitter.split(frame)
        disp = cv2.hconcat([left, right])

        cv2.imshow(win, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    if recorder is not None:
        recorder.close()
        print(f"Saved recording to: {recorder.out_path}")
    cv2.destroyAllWindows()
