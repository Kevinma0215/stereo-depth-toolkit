# Pipeline

## Depth Output Spec

- Reference frame: left-rectified image
- dtype: `float32`
- Unit: metres
- Invalid pixels: `NaN`
- Output resolution: matches input resolution (no downscaling by default)

## Data Flow

```
ICameraSource.grab()
  └─> FramePair
        └─> IRectifier.rectify()
              └─> RectifiedPair
                    └─> IDisparityMatcher.compute()
                          └─> disparity float32 (H, W)
                                └─> IDepthEstimator.to_depth()
                                      └─> DepthMap
```

See [architecture.md](architecture.md) for the full module map and port definitions.
