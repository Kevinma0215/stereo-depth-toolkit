import argparse
import cv2
import numpy as np
import yaml

def load_calib(path):
    with open(path, "r") as f:
        c = yaml.safe_load(f)
    def mat(x): return np.array(x, dtype=np.float64)
    K1 = mat(c["K1"]); D1 = np.array(c["D1"], dtype=np.float64)
    K2 = mat(c["K2"]); D2 = np.array(c["D2"], dtype=np.float64)
    R1 = mat(c["R1"]); R2 = mat(c["R2"])
    P1 = mat(c["P1"]); P2 = mat(c["P2"])
    w = c["image_size"]["width"]; h = c["image_size"]["height"]
    return (K1,D1,K2,D2,R1,R2,P1,P2,(w,h))

def draw_epilines(img, step=40):
    out = img.copy()
    h, w = out.shape[:2]
    for y in range(0, h, step):
        cv2.line(out, (0,y), (w,y), (0,255,0), 1)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--calib", default="data/calib.yaml")
    args = ap.parse_args()

    K1,D1,K2,D2,R1,R2,P1,P2,size = load_calib(args.calib)
    w,h = size

    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w,h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w,h), cv2.CV_32FC1)

    L = cv2.imread(args.left)
    R = cv2.imread(args.right)
    Lr = cv2.remap(L, map1x, map1y, cv2.INTER_LINEAR)
    Rr = cv2.remap(R, map2x, map2y, cv2.INTER_LINEAR)

    disp = cv2.hconcat([draw_epilines(Lr), draw_epilines(Rr)])
    cv2.imshow("rectified (q to quit)", disp)
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == ord('q') or k == 27:
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
