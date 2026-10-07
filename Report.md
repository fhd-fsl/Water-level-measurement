# Real-Time Water Level Measurement Using Feature-Based Tracking and Edge Density Analysis

## Prototype goal

Measuring the water level inside a bucket using a submerged ruler and a fixed camera. The system must be robust to real-world camera shake, without relying on hardcoded coordinates after an initial one-time calibration.

## Pipeline Overview

The system operates in two phases: a one-time **Calibration** phase and a per-frame **Measurement** phase.

### Phase 1: Calibration (One-Time)

A reference frame is captured and the user marks:
- A **bounding box** around the scale.
- **Measurement reference points** corresponding to the markings on the scale.

ORB (Oriented FAST and Rotated BRIEF) keypoints are then extracted from the scale region of this reference frame. These keypoints serve as the visual "fingerprint" of the scale for all subsequent tracking.

### Phase 2: Per-Frame Measurement

Each incoming frame is processed through a 4-stage pipeline:

**Stage 1 — ORB Feature Matching:** ORB keypoints are detected in the new frame and matched against the reference keypoints using a Brute-Force Hamming distance matcher with Lowe's ratio test (threshold: 0.75) to filter reliable matches.

**Stage 2 — Homography Estimation:** A 3×3 homography matrix is computed from the matched keypoints using RANSAC, which is robust to outlier matches. This matrix encodes the full geometric transformation (translation, rotation, scale, and perspective) between the reference frame and the current frame.

**Stage 3 — Coordinate Warping:** The homography is applied to transform the original bounding box and all 13 inch-mark coordinates to their new positions in the current frame. This allows the system to dynamically "follow" the ruler regardless of camera movement.

**Stage 4 — Bottom-Up Edge Density Scan:** The ruler region (now dynamically located) is cropped, converted to grayscale, and processed with Canny edge detection (thresholds: 30, 120). Horizontal edge density is computed per row and smoothed with a 50-pixel convolution window. The algorithm scans from the bottom of the crop upward — the submerged portion of the ruler has very low edge density (water blurs the markings), while the dry portion has very high edge density (sharp markings). The first row where the smoothed density exceeds 25% of the maximum density is identified as the water surface. The pixel coordinate is then interpolated against the warped inch-mark positions to produce the final measurement in inches.

A moving average is applied across consecutive readings to suppress single-frame jitter from surface ripples.

## Testing: Artificial Camera Shake

To validate robustness, **artificial camera shake** was injected into every frame before processing:

| Parameter       | Value       |
|-----------------|-------------|
| Translation (X) | ±15 pixels (random per frame) |
| Translation (Y) | ±15 pixels (random per frame) |
| Rotation        | ±2.0 degrees (random per frame) |

This simulates realistic vibration of a mounted camera. The transformation is applied via an affine warp with replicated border padding to avoid black edges.

## Results

### Camera Shake
The ORB tracker consistently recovered the ruler's position across all shaken frames, typically producing **50–150 inlier matches** per frame — well above the minimum of 8 required for a stable homography. The water level readings remained accurate and stable despite the injected perturbations, with measurements closely matching the ground-truth readings from the unshaken baseline.

### Simulated Edge Cases

All modes were tested across the full 261-frame sequence with shake disabled. Every mode achieved **0 tracking failures**.

| Mode | Description | Min inliers observed |
|---|---|---|
| `normal` | Standard colour camera (baseline) | 95 |
| `nv` | Near-IR / night-vision (grayscale, hotspot, grain) | — |
| `mixed` | First half normal → second half NV | — |
| `brightness` | Exposure compensation + auto-exposure drift + cloud shadows | 42 |
| `glare` | Specular water-surface glare + lens flare ghosts | — |
| `reflections` | Mirror-like water surface reflections with wave distortion | 52 |
| `outdoor_mixed` | All outdoor effects combined | 43 |

Inlier counts drop slightly under heavy outdoor effects (bright glare and cloud shadows wash out some ORB keypoints) but remain well above the 8-match floor, so tracking never fails. The bottom-up edge density scan uses the same normal-camera parameters for all outdoor modes — the Canny thresholds remain effective because the outdoor effects do not add high-frequency noise to the ruler region.

## Brightness Normalization (CLAHE)

Low brightness was identified as a killer edge case — dark frames reduce ORB inlier counts and weaken the Canny edge signal at the water surface.

A CLAHE (Contrast Limited Adaptive Histogram Equalization) step was added as an optional pre-processing pass (`--normalize` flag):

- Applied to the **L channel of LAB color space** — normalizes luminance without shifting hue or saturation
- Uses **local 8×8 tiles** with clip limit 2.0 — handles uneven lighting (IR hotspot, partial shadow) without amplifying noise
- **Adaptive threshold**: only fires when mean frame luminance < 100/255. Skips well-exposed and overexposed frames — applying CLAHE to a bright frame flattens ruler contrast and causes the edge scan to misfire

Applied after all camera simulation filters and shake, before ORB tracking and the edge density scan, so both benefit from the normalized image. The reference frame is normalized with the same function so ORB features are extracted from a consistent representation.

## Performance

The entire pipeline runs at approximately **25–50 FPS on CPU** (Intel i-series), with ORB detection and matching adding only ~20ms of overhead per frame on top of the edge density scan. No GPU is required, making it suitable for deployment on embedded or edge devices.
