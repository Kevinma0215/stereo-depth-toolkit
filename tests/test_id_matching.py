""" 我們測 _match_ids_one_view 的「會挑出共同 ids」這件事，這是你原本最容易 silently fail 的地方。 """

import numpy as np
from stereo_depth.adapters.calibration.charuco_calibrator import _match_ids_one_view

def test_match_ids_one_view_basic():
    # 假造 corners: Nx1x2
    N = 20
    lc = np.random.rand(N, 1, 2).astype(np.float32)
    rc = np.random.rand(N, 1, 2).astype(np.float32)

    li = np.arange(N).reshape(-1, 1).astype(np.int32)
    ri = np.arange(5, 5+N).reshape(-1, 1).astype(np.int32)  # 共同 ids = 5..19 共 15個

    chess = np.random.rand(100, 3).astype(np.float32)  # 假造 board corners 3D

    out = _match_ids_one_view(lc, li, rc, ri, chess, min_common=10)
    assert out is not None
    obj, ptsL, ptsR = out
    assert obj.shape[0] == ptsL.shape[0] == ptsR.shape[0]
    assert obj.shape[1] == 3
    assert ptsL.shape[1] == 2
