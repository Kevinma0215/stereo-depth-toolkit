import argparse
from pathlib import Path
import cv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to SBS video (AVI recommended)")
    ap.add_argument("--out", default="data/calib_frames", help="Output dir")
    ap.add_argument("--every", type=int, default=10, help="Save one pair every N frames")
    ap.add_argument("--swap-lr", action="store_true", help="Swap left/right after split")
    ap.add_argument("--max-pairs", type=int, default=300, help="Stop after saving this many pairs")
    args = ap.parse_args()

    out = Path(args.out)
    left_dir = out / "left"
    right_dir = out / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.every == 0:
            h, w = frame.shape[:2]
            mid = w // 2
            left = frame[:, :mid]
            right = frame[:, mid:]
            if args.swap_lr:
                left, right = right, left

            cv2.imwrite(str(left_dir / f"left_{saved:05d}.png"), left)
            cv2.imwrite(str(right_dir / f"right_{saved:05d}.png"), right)
            saved += 1
            if saved >= args.max_pairs:
                break
        idx += 1

    cap.release()
    print(f"Saved {saved} pairs to: {out}")

if __name__ == "__main__":
    main()
