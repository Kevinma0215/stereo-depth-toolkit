"""CudaBmMatcher — IDisparityMatcher backed by cv2.cuda.StereoBM.

Raises ``ImportError`` at module import time if OpenCV was not built with
CUDA support or no CUDA-capable device is present.  This mirrors the
guard pattern used by ``retinify_matcher.py``.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# CUDA availability guard  (equivalent to #ifdef STEREO_CUDA)
# ---------------------------------------------------------------------------

try:
    import cv2 as _cv
    _n_devices: int = _cv.cuda.getCudaEnabledDeviceCount()
    if _n_devices == 0:
        raise RuntimeError("No CUDA-capable GPU found (getCudaEnabledDeviceCount() == 0)")
    if not hasattr(_cv.cuda, "StereoBM_create"):
        raise AttributeError(
            "cv2.cuda.StereoBM_create not available — "
            "rebuild OpenCV with -DWITH_CUDA=ON -DOPENCV_EXTRA_MODULES_PATH=<contrib>"
        )
except Exception as _exc:
    raise ImportError(
        "CudaBmMatcher requires OpenCV built with CUDA support and a CUDA-capable GPU.\n"
        "  Build OpenCV from source with:\n"
        "    -DWITH_CUDA=ON -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib/modules>\n"
        "  CUDA Toolkit ≥ 11.0 required.\n"
        f"  Underlying error: {_exc}"
    ) from _exc

import cv2  # noqa: E402  (import after guard so linters see it)

from stereo_depth.use_cases.ports import IDisparityMatcher  # noqa: E402


class CudaBmMatcher(IDisparityMatcher):
    """IDisparityMatcher backed by cv2.cuda.StereoBM (GPU block matching).

    Uploads rectified grayscale images to the GPU, runs CUDA StereoBM, and
    downloads the result as a float32 disparity map — keeping the same
    contract as ``SgbmMatcher``.

    Args:
        num_disparities: Number of disparity levels. Must be divisible by 16.
        block_size:      Matching block (SAD window) size. Must be odd, 5–31.
        prefilter_cap:   Sobel pre-filter truncation value (default: 31).
    """

    def __init__(
        self,
        num_disparities: int = 64,
        block_size: int = 15,
        prefilter_cap: int = 31,
    ) -> None:
        if num_disparities % 16 != 0:
            raise ValueError(
                f"num_disparities must be divisible by 16, got {num_disparities}"
            )
        if block_size % 2 == 0 or not (5 <= block_size <= 31):
            raise ValueError(
                f"block_size must be an odd number in [5, 31], got {block_size}"
            )

        self._matcher = cv2.cuda.StereoBM_create(
            numDisparities=num_disparities,
            blockSize=block_size,
        )
        self._matcher.setPreFilterCap(prefilter_cap)

    def compute(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Compute disparity using CUDA StereoBM.

        Args:
            left:  uint8 image, shape (H, W) or (H, W, 3).
            right: uint8 image, same shape as left.

        Returns:
            float32 disparity map, shape (H, W).  Invalid pixels have value <= 0.
        """
        left_gray  = _to_gray(left)
        right_gray = _to_gray(right)

        left_gpu  = cv2.cuda_GpuMat()
        right_gpu = cv2.cuda_GpuMat()
        disp_gpu  = cv2.cuda_GpuMat()

        left_gpu.upload(left_gray)
        right_gpu.upload(right_gray)

        self._matcher.compute(left_gpu, right_gpu, disp_gpu)

        # CUDA StereoBM returns fixed-point int16 (raw × 16), same as CPU BM.
        disp: np.ndarray = disp_gpu.download().astype(np.float32) / 16.0
        return disp


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
