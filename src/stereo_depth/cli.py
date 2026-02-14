from __future__ import annotations
import argparse
import cv2
from stereo_depth.io.sbs_capture import SBSSplitter, open_camera

def cmd_preview(args):
    # cap = open_camera(device=args.device, video=args.video)
    cap = open_camera(
        device = args.device,
        path   = args.path,
        width  = args.width,
        height = args.height,
        fps    = args.fps,
        video  = args.video
    )

    if not cap.isOpened():
        raise RuntimeError("Failed to open source")

    splitter = SBSSplitter(swap_lr=args.swap_lr)

    writer = None
    if args.record is not None:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30

        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Cannot read first frame for recording setup")

        h, w = frame.shape[:2]
        writer = cv2.VideoWriter(args.record, fourcc, fps, (w, h))
        writer.write(frame)
    else:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Cannot read frame")
        h, w = frame.shape[:2]

    win = "stereo-depth preview (q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        if writer is None:
            ok, frame = cap.read()
        else:
            ok, frame = cap.read()

        if not ok:
            break

        if writer is not None:
            writer.write(frame)

        left, right = splitter.split(frame)
        disp = cv2.hconcat([left, right])

        cv2.imshow(win, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved recording to: {args.record}")

    cv2.destroyAllWindows()
    return 0


# def cmd_preview(args: argparse.Namespace) -> int:
#     # cap = open_camera(args.device, path=args.path, width=args.width, height=args.height, fps=args.fps)
#     cap = open_camera(
#         device = args.device,
#         path   = args.path,
#         width  = args.width,
#         height = args.height,
#         fps    = args.fps,
#         video  = args.video
#     )

#     splitter = SBSSplitter(
#         swap_lr=args.swap_lr,
#         crop_x=args.crop_x,
#         crop_y=args.crop_y,
#         crop_w=args.crop_w,
#         crop_h=args.crop_h,
#     )

#     win = "stereo-depth preview (press q to quit)"
#     cv2.namedWindow(win, cv2.WINDOW_NORMAL)

#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             if args.video is not None:
#                 print("End of video or decode error.")
#                 break
#             else:
#                 print("Failed to read camera frame, retrying...")
#                 continue
#             # print("Failed to read frame")
#             # break

#         left, right = splitter.split(frame)

#         if args.gray:
#             left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
#             right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

#         # Show side-by-side debug view (left | right)
#         disp = cv2.hconcat([left, right])
#         cv2.imshow(win, disp)

#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q') or key == 27:
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="stereo-depth")
    sub = p.add_subparsers(dest="cmd", required=True)

    prev = sub.add_parser("preview", help="Preview SBS camera and split into left/right.")
    prev.add_argument("--device", type=int, default=0, help="Camera device index (default: 0)")
    prev.add_argument("--width", type=int, default=0, help="Capture width (0: default)")
    prev.add_argument("--height", type=int, default=0, help="Capture height (0: default)")
    prev.add_argument("--fps", type=int, default=0, help="Capture FPS (0: default)")
    prev.add_argument("--swap-lr", action="store_true", help="Swap left/right after split")
    prev.add_argument("--gray", action="store_true", help="Display grayscale")
    prev.add_argument("--path", type=str, default=None, help="V4L2 device path, e.g. /dev/video1")
    prev.add_argument("--video", type=str, default=None, help="Path to SBS video file (mp4/avi)")
    prev.add_argument("--record", type=str, default=None, help="Record preview stream to AVI (MJPG)")


    # Cropping to handle borders/overlays
    prev.add_argument("--crop-x", type=int, default=0)
    prev.add_argument("--crop-y", type=int, default=0)
    prev.add_argument("--crop-w", type=int, default=0)
    prev.add_argument("--crop-h", type=int, default=0)

    prev.set_defaults(func=cmd_preview)
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
