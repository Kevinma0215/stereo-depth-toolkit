from __future__ import annotations
import argparse
import cv2
from stereo_depth.io.sbs_capture import SBSSplitter, open_camera

def cmd_preview(args: argparse.Namespace) -> int:
    cap = open_camera(args.device, args.width, args.height, args.fps)
    splitter = SBSSplitter(
        swap_lr=args.swap_lr,
        crop_x=args.crop_x,
        crop_y=args.crop_y,
        crop_w=args.crop_w,
        crop_h=args.crop_h,
    )

    win = "stereo-depth preview (press q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame")
            break

        left, right = splitter.split(frame)

        if args.gray:
            left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        # Show side-by-side debug view (left | right)
        disp = cv2.hconcat([left, right])
        cv2.imshow(win, disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


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
