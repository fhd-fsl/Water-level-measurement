# Robust Water Level Measurement via Computer Vision

Real-time computer vision pipeline to measure water level inside a container using a submerged ruler and a fixed camera. Robust to camera shake, low light, glare, reflections, and exposure changes.

The system uses **ORB + RANSAC Homography** to compensate for camera shake and **Bottom-Up Edge Density Scanning** to detect the water surface.

![Sample Output](sample_output.jpg)

## Core Features

- **Dynamic ruler tracking** — ORB feature matching + RANSAC homography handles translation, rotation, and perspective shift between frames
- **Water surface detection** — bottom-up Canny edge density scan identifies the dry/submerged transition on the ruler
- **Adaptive brightness normalization** — CLAHE pre-processing recovers detail in dark frames without degrading well-exposed ones
- **7 simulation modes** — test against normal, night-vision, mixed, brightness changes, glare, reflections, and all outdoor effects combined
- **Physical measurement** — pixel detections interpolated against 13 calibrated inch marks via `np.interp`

## File Overview

| File | Purpose |
|---|---|
| `extract_frames.py` | One-time: extract frames from source video at 2 FPS |
| `calibrate.py` | One-time interactive GUI: define ruler bbox + 13 inch marks |
| `config.json` | Calibration output (ruler bbox + pixel↔inch reference points) |
| `core.py` | All frame processing: tracking, edge scan, normalization, shake |
| `run.py` | Main entry point — all modes, flags, video output |
| `simulate_nightvision.py` | IR/NV camera simulation (tunable parameters at top) |
| `simulate_outdoor.py` | Outdoor effects simulation: exposure, glare, reflections (tunable parameters at top) |

## Installation

```bash
git clone https://github.com/yourusername/water-level-measurement.git
cd water-level-measurement

python -m venv venv
.\venv\Scripts\activate          # Windows
# source venv/bin/activate       # Unix

pip install opencv-python numpy
```

## Quick Start

```bash
# 1. Extract frames from your video
python extract_frames.py

# 2. Calibrate once (draws bbox + clicks 13 inch marks)
python calibrate.py

# 3. Run
python run.py --mode normal
```

Output is saved as `output_<mode>.mp4`.

## Run Modes

```bash
python run.py --mode normal           # standard colour camera
python run.py --mode nv               # simulated IR/night-vision camera
python run.py --mode mixed            # first half normal, second half NV
python run.py --mode brightness       # exposure drift + auto-exposure + cloud shadows
python run.py --mode glare            # specular water glare + lens flare
python run.py --mode reflections      # water surface reflections with wave distortion
python run.py --mode outdoor_mixed    # all outdoor effects combined
```

## Flags

| Flag | Default | Effect |
|---|---|---|
| `--mode <mode>` | `normal` | Camera/environment simulation mode (see above) |
| `--normalize` | off | Apply adaptive CLAHE brightness normalization before tracking and edge scan |
| `--no-shake` | off | Disable artificial camera shake injection |
| `--output <file>` | `output_<mode>.mp4` | Override output filename |

### When to use `--normalize`

Use it when your footage is underexposed or you expect low-light conditions. CLAHE equalizes the luminance locally (8×8 tiles) so the ruler markings and water surface edge are recoverable from dark frames. It skips automatically on well-exposed frames (mean luminance ≥ 100/255) to avoid flattening contrast where it isn't needed.

```bash
# Low light + all outdoor effects
python run.py --mode outdoor_mixed --normalize

# NV mode is already dark — normalize helps here too
python run.py --mode nv --normalize

# Isolate optical effects from camera movement
python run.py --mode brightness --normalize --no-shake
```

## Tuning Simulation Parameters

### Night-Vision (`simulate_nightvision.py`)

| Parameter | Default | Effect |
|---|---|---|
| `MODE` | `ir_security` | `ir_security` (grayscale) or `nvg_green` (phosphor tint) |
| `NOISE_SIGMA` | 3.0 | Sensor grain intensity |
| `IR_HOTSPOT_STRENGTH` | 0.35 | Centre bloom from IR illuminator ring |
| `IR_HOTSPOT_RADIUS` | 0.55 | Hotspot width as fraction of image diagonal |
| `GAMMA` | 0.80 | Sensor response curve (< 1 brightens midtones) |
| `BLUR_SIGMA` | 1.2 | NIR focal plane softness |

Run `python simulate_nightvision.py` to preview the effect on a sample frame and save a side-by-side comparison.

### Outdoor Effects (`simulate_outdoor.py`)

| Parameter | Default | Effect |
|---|---|---|
| `EXPOSURE_EV` | 0.0 | Base exposure compensation in stops |
| `AUTO_EXPOSURE_RANGE_EV` | 1.5 | Auto-exposure drift range over the sequence |
| `CLOUD_SHADOW_ENABLED` | True | Random darkening patches |
| `GLARE_ENABLED` | True | Specular highlight on water surface |
| `GLARE_INTENSITY` | 0.6 | Glare brightness (0–1) |
| `LENS_FLARE_ENABLED` | True | Ghost artifacts along light source axis |
| `REFLECTION_ENABLED` | True | Mirror reflection of scene above waterline |
| `REFLECTION_STRENGTH` | 0.4 | Reflection opacity (0–1) |
| `REFLECTION_DISTORTION` | 8.0 | Wave amplitude in pixels |
| `WATER_LEVEL_RATIO` | 0.65 | Where the water surface sits (0 = top, 1 = bottom) |
| `SIMULATION_MODE` | `all` | `all`, `exposure`, `glare`, or `reflection` |

Run `python simulate_outdoor.py` to preview the effect and save a side-by-side comparison.

### Edge Scan Parameters (`run.py` → `EDGE_PARAMS`)

| Parameter | Normal | NV | Effect |
|---|---|---|---|
| `blur_ksize` | 5 | 9 | Gaussian blur before Canny — larger suppresses sensor noise |
| `canny_low/high` | 30/120 | 50/150 | Hysteresis thresholds — raise to ignore weak noise edges |
| `smooth_window` | 75 | 75 | Row density smoothing window in pixels |
| `threshold_frac` | 0.25 | 0.40 | Fraction of peak density used as water surface threshold |

## Pipeline (per frame)

1. **Camera filter** — NV or outdoor simulation applied to the raw frame
2. **Brightness normalization** *(optional)* — adaptive CLAHE on L channel if frame is dark
3. **ORB feature matching** — keypoints matched against reference frame (Lowe ratio 0.75, min 8 matches)
4. **RANSAC homography** — 3×3 matrix encoding full geometric shift (threshold 5px)
5. **Coordinate warping** — ruler bbox + 13 inch marks transformed to current frame position
6. **Bottom-up edge density scan** — Canny → row-wise sum → 75px smoothing → scan upward for water surface
7. **Interpolation** — `np.interp` against warped inch marks → reading in inches
8. **Temporal smoothing** — moving average (window=2) to suppress single-frame ripple jitter

See `Report.md` for a full technical breakdown and test results.

## Test Results

All 7 modes tested with `--normalize --no-shake` across 261 frames — **0 tracking failures** in every mode.

| Mode | Min inliers | Notes |
|---|---|---|
| `normal` | 26 | Baseline |
| `nv` | 43 | NV-tuned Canny params |
| `mixed` | 43 | ORB bridges normal→NV transition |
| `brightness` | 42 | Adaptive CLAHE handles dark phases |
| `glare` | 26 | Glare sits outside ruler bbox |
| `reflections` | 25 | Reflection below waterline, not in ruler region |
| `outdoor_mixed` | 43 | All effects combined |
