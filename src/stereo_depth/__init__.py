__version__ = "0.1.0"

# Entities
from stereo_depth.entities import (
    FramePair,
    RectifiedPair,
    CalibrationResult,
    DepthMap,
    PointCloud,
)

# Use-case ports + pipeline
from stereo_depth.use_cases import (
    ICameraSource,
    ICalibrationRepo,
    IRectifier,
    IDisparityMatcher,
    IDepthEstimator,
    StereoPipeline,
)

# Calibration evaluation
from stereo_depth.adapters.calibration import (
    CalibrationEvaluation,
    evaluate_calibration,
)

__all__ = [
    # entities
    "FramePair",
    "RectifiedPair",
    "CalibrationResult",
    "DepthMap",
    "PointCloud",
    # use cases
    "ICameraSource",
    "ICalibrationRepo",
    "IRectifier",
    "IDisparityMatcher",
    "IDepthEstimator",
    "StereoPipeline",
    # calibration evaluation
    "CalibrationEvaluation",
    "evaluate_calibration",
]
