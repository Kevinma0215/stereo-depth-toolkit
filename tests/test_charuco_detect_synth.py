""" 測試（不靠真實相機/真實資料）

你要的是「正確開發方式」：測試要能在任何機器上跑。
我們用 OpenCV 直接合成一張 Charuco board 圖來測 detect。 """

import numpy as np
import cv2
from stereo_depth.adapters.calibration.charuco_calibrator import make_charuco_board, detect_charuco

def test_detect_charuco_on_synth():
    board, dictionary = make_charuco_board(
        squares_x=7, squares_y=5,
        square_length=0.04, marker_length=0.02,
        dict_name="DICT_4X4_50"
    )

    # 合成 board 影像（灰階）
    img = board.generateImage((800, 600))  # (w,h) 生成灰階
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    det = detect_charuco(gray, board, dictionary, min_markers=4, min_charuco=10)
    assert det.ok
    assert det.num_markers >= 4
    assert det.num_charuco >= 10
    assert det.corners is not None
    assert det.ids is not None
