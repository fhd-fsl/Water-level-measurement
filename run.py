"""
run.py — unified water level measurement pipeline

Usage:
    python run.py --mode normal              standard camera
    python run.py --mode nv                  night-vision camera (simulated)
    python run.py --mode mixed               first half normal, second half NV
    python run.py --mode brightness          exposure/auto-exposure + cloud shadows
    python run.py --mode glare               glare & specular highlights + lens flare
    python run.py --mode reflections         water surface reflections
    python run.py --mode outdoor_mixed       all outdoor effects combined
    python run.py --mode nv --no-shake       disable artificial shake
    python run.py --mode mixed --output my_run.mp4

The NV filter reads its settings from simulate_nightvision.py.
Outdoor filters read their settings from simulate_outdoor.py.
Edge-scan parameters are tuned per mode in EDGE_PARAMS below.
"""

import argparse
import cv2
import json
import glob
import os
import numpy as np

from core import setup_reference, track_ruler, calculate_water_y, get_water_level, apply_shake, normalize_brightness
from simulate_nightvision import simulate_nv, MODE, NOISE_SIGMA, IR_HOTSPOT_STRENGTH, \
    IR_HOTSPOT_RADIUS, GAMMA, BLUR_SIGMA, CONTRAST_CLIP
from simulate_outdoor import simulate_outdoor, set_seed, SIMULATION_MODE as OUTDOOR_MODE, \
    EXPOSURE_EV, AUTO_EXPOSURE_RANGE_EV, AUTO_EXPOSURE_SPEED, \
    CLOUD_SHADOW_ENABLED, CLOUD_SHADOW_STRENGTH, CLOUD_SHADOW_COUNT, CLOUD_SHADOW_SCALE, \
    GLARE_ENABLED, GLARE_INTENSITY, GLARE_SIZE_X, GLARE_SIZE_Y, GLARE_POSITION_X, GLARE_POSITION_Y, \
    LENS_FLARE_ENABLED, LENS_FLARE_STRENGTH, LENS_FLARE_COUNT, LENS_FLARE_SOURCE_X, LENS_FLARE_SOURCE_Y, \
    REFLECTION_ENABLED, REFLECTION_STRENGTH, REFLECTION_DISTORTION, REFLECTION_WAVE_FREQ, \
    REFLECTION_BLUR_SIGMA, REFLECTION_FADE_START, REFLECTION_FADE_END, \
    WATER_LEVEL_RATIO, RANDOM_SEED

# ---------------------------------------------------------------------------
# Mode routing
# ---------------------------------------------------------------------------

# Outdoor modes and the simulate_outdoor() sub-mode they map to
OUTDOOR_MODES = {
    "brightness":    "exposure",
    "glare":         "glare",
    "reflections":   "reflection",
    "outdoor_mixed": "all",
}

# ---------------------------------------------------------------------------
# Per-mode edge scan parameters
# Outdoor modes keep normal params — the image is still colour, just with
# environmental effects layered on top.
# ---------------------------------------------------------------------------
EDGE_PARAMS = {
    "normal": dict(
        blur_ksize=5,
        canny_low=30, canny_high=120,
        smooth_window=75,
        threshold_frac=0.25,
    ),
    "nv": dict(
        blur_ksize=9,       # stronger blur suppresses IR sensor grain
        canny_low=50, canny_high=150,   # ignore weak noise edges
        smooth_window=50,
        threshold_frac=0.40,  # higher bar — only trigger on strong edge bands
    ),
}
# Outdoor modes share normal edge params
for _m in OUTDOOR_MODES:
    EDGE_PARAMS[_m] = EDGE_PARAMS["normal"]

# Shake settings
SHAKE_MAX_TRANSLATE = 15   # pixels
SHAKE_MAX_ANGLE     = 2.0  # degrees
SMOOTHING_FRAMES    = 2    # moving-average window


# ---------------------------------------------------------------------------
# Frame-level filter helpers
# ---------------------------------------------------------------------------

def apply_nv_filter(frame):
    return simulate_nv(frame, MODE, NOISE_SIGMA, IR_HOTSPOT_STRENGTH,
                       IR_HOTSPOT_RADIUS, GAMMA, BLUR_SIGMA, CONTRAST_CLIP)


def apply_outdoor_filter(frame, outdoor_sub_mode, frame_idx, total_frames):
    """Apply the requested outdoor effect using settings from simulate_outdoor.py."""
    h = frame.shape[0]
    water_y = int(h * WATER_LEVEL_RATIO)
    return simulate_outdoor(frame, frame_idx=frame_idx, total_frames=total_frames,
                            water_y=water_y, mode=outdoor_sub_mode)


def frame_mode_for(i, total, mode):
    """
    Return the per-frame camera type used for edge-param lookup.
    Mixed: first half normal, second half NV.
    Outdoor modes always use 'normal' edge params but their own filter.
    """
    if mode == "nv":
        return "nv"
    if mode == "mixed":
        return "normal" if i < total // 2 else "nv"
    # normal and all outdoor modes
    return "normal"


# ---------------------------------------------------------------------------
# Mode label for on-screen overlay
# ---------------------------------------------------------------------------

MODE_LABELS = {
    "normal":        ("Cam: NORMAL",      (200, 200, 200)),
    "nv":            ("Cam: NV",          (0, 220, 255)),
    "mixed":         ("Cam: MIXED",       (200, 200, 200)),   # updated per-frame below
    "brightness":    ("FX: BRIGHTNESS",   (255, 200, 50)),
    "glare":         ("FX: GLARE",        (50, 200, 255)),
    "reflections":   ("FX: REFLECTIONS",  (50, 255, 200)),
    "outdoor_mixed": ("FX: OUTDOOR ALL",  (100, 255, 100)),
}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process(frame_files, config, mode, enable_shake, output_path, normalize=False):
    n = len(frame_files)

    # Seed the outdoor RNG once for reproducibility
    if mode in OUTDOOR_MODES:
        set_seed(RANDOM_SEED)

    # Reference frame: use NV filter for pure NV mode; raw otherwise.
    # For mixed mode the reference stays normal — ORB bridges the transition well.
    ref_raw   = cv2.imread(frame_files[0])
    ref_frame = apply_nv_filter(ref_raw) if mode == "nv" else ref_raw
    if normalize:
        ref_frame = normalize_brightness(ref_frame)

    print("Setting up ORB reference features...")
    ref_data = setup_reference(config, ref_frame)
    print(f"Extracted {len(ref_data['kp_ref'])} reference keypoints.")
    print(f"Mode: {mode}  |  shake: {'on' if enable_shake else 'off'}  |  normalize: {'on' if normalize else 'off'}")

    if mode == "nv":
        print(f"NV filter: {MODE}, noise={NOISE_SIGMA}, hotspot={IR_HOTSPOT_STRENGTH}, "
              f"gamma={GAMMA}, blur={BLUR_SIGMA}")
    elif mode == "mixed":
        print(f"Mixed split: frames 0–{n//2 - 1} normal, {n//2}–{n-1} NV")
    elif mode in OUTDOOR_MODES:
        sub = OUTDOOR_MODES[mode]
        print(f"Outdoor filter: sub-mode='{sub}' | exposure_ev={EXPOSURE_EV} | "
              f"auto_range={AUTO_EXPOSURE_RANGE_EV}EV")
        print(f"  glare={GLARE_ENABLED}({GLARE_INTENSITY}) | "
              f"flare={LENS_FLARE_ENABLED}({LENS_FLARE_STRENGTH}) | "
              f"reflection={REFLECTION_ENABLED}({REFLECTION_STRENGTH})")
    print()

    h, w = ref_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_path, fourcc, 15.0, (w, h))

    history = []
    failed  = 0

    for i, f_path in enumerate(frame_files):
        raw   = cv2.imread(f_path)
        fmode = frame_mode_for(i, n, mode)   # 'normal' or 'nv', for edge params

        # Apply camera simulation filter
        if fmode == "nv":
            frame = apply_nv_filter(raw)
        elif mode in OUTDOOR_MODES:
            frame = apply_outdoor_filter(raw, OUTDOOR_MODES[mode], i, n)
        else:
            frame = raw

        # Artificial shake (applied after filter so it mimics physical movement)
        shake_info = None
        if enable_shake:
            frame, tx, ty, angle = apply_shake(frame, SHAKE_MAX_TRANSLATE, SHAKE_MAX_ANGLE)
            shake_info = (tx, ty, angle)

        # CLAHE brightness normalization — applied after all camera effects and shake,
        # before ORB tracking and edge scan so both see the same normalized image.
        if normalize:
            frame = normalize_brightness(frame)

        frame_display = frame.copy()

        # 1. Track ruler
        shifted_bbox, shifted_marks, H, match_info = track_ruler(frame, ref_data)
        if match_info["status"] != "ok":
            failed += 1

        # 2. Edge density scan — params chosen by current frame mode
        abs_y, *_ = calculate_water_y(frame, shifted_bbox, **EDGE_PARAMS[fmode])

        # 3. Temporal smoothing
        history.append(abs_y)
        if len(history) > SMOOTHING_FRAMES:
            history.pop(0)
        smoothed_y = int(np.mean(history))

        # 4. Interpolate inches
        water_level_inch = get_water_level(smoothed_y, shifted_marks)

        # --- Annotations ---
        sx1, sy1 = shifted_bbox["x1"], shifted_bbox["y1"]
        sx2, sy2 = shifted_bbox["x2"], shifted_bbox["y2"]

        # Tracking box
        if match_info["status"] == "ok":
            corners = match_info["transformed_corners"].astype(np.int32)
            cv2.polylines(frame_display, [corners], True, (0, 255, 0), 3)
        else:
            cv2.rectangle(frame_display, (sx1, sy1), (sx2, sy2), (0, 0, 255), 3)

        # Inch marks
        for mark in shifted_marks:
            cv2.circle(frame_display, (mark["x"], mark["y"]), 5, (255, 0, 255), -1)

        # Water level line + reading
        cv2.line(frame_display, (sx1, smoothed_y), (sx2, smoothed_y), (0, 255, 255), 4)
        cv2.putText(frame_display, f"Level: {water_level_inch:.2f} in",
                    (sx2 + 20, smoothed_y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)

        # Tracking info
        if match_info["status"] == "ok":
            cv2.putText(frame_display,
                        f"Track: {match_info['num_inliers']}/{match_info['num_matches']}",
                        (30, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        else:
            cv2.putText(frame_display, f"TRACK FAIL: {match_info['status']}",
                        (30, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Shake info
        if shake_info:
            tx, ty, angle = shake_info
            cv2.putText(frame_display, f"Shake: tx={tx} ty={ty} rot={angle:.1f}",
                        (30, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

        # Mode label
        if mode == "mixed":
            half = "1st half" if i < n // 2 else "2nd half"
            label_text  = f"Cam: {fmode.upper()}  [{half}]"
            label_color = (200, 200, 200) if fmode == "normal" else (0, 220, 255)
        else:
            label_text, label_color = MODE_LABELS[mode]
        cv2.putText(frame_display, label_text, (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, label_color, 2)

        # Frame counter
        cv2.putText(frame_display, f"Frame {i+1}/{n}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Live preview
        scale   = 800 / h
        resized = cv2.resize(frame_display, (int(w * scale), 800))
        cv2.imshow(f"Water Measurement [{mode}]", resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        out_video.write(frame_display)

        if (i + 1) % 20 == 0:
            print(f"  Frame {i+1:>3}/{n} | Level: {water_level_inch:.2f} in | "
                  f"Track: {match_info['status']} "
                  f"({match_info.get('num_inliers', 0)} inliers) | cam: {fmode}")

    out_video.release()
    cv2.destroyAllWindows()

    fail_pct = 100 * failed / n
    print(f"\nDone. Saved: {output_path}")
    print(f"Tracking failures: {failed}/{n} ({fail_pct:.1f}%)")


# ---------------------------------------------------------------------------

def main():
    all_modes = ["normal", "nv", "mixed", "brightness", "glare", "reflections", "outdoor_mixed"]

    parser = argparse.ArgumentParser(
        description="Water level measurement — normal / NV / mixed / outdoor modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python run.py --mode normal
  python run.py --mode nv
  python run.py --mode mixed
  python run.py --mode brightness
  python run.py --mode glare
  python run.py --mode reflections
  python run.py --mode outdoor_mixed
  python run.py --mode nv --no-shake
  python run.py --mode glare --output test_glare.mp4
        """,
    )
    parser.add_argument(
        "--mode", choices=all_modes, default="normal",
        help=(
            "normal — standard camera | nv — night-vision | mixed — 1st half normal, 2nd half NV | "
            "brightness — exposure/auto-exposure/cloud shadows | "
            "glare — specular glare + lens flare | "
            "reflections — water surface reflections | "
            "outdoor_mixed — all outdoor effects"
        ),
    )
    parser.add_argument("--no-shake", action="store_true",
                        help="Disable artificial camera shake simulation")
    parser.add_argument("--normalize", action="store_true",
                        help="Apply CLAHE brightness normalization before tracking and edge scan")
    parser.add_argument("--output", default=None,
                        help="Output filename (default: output_<mode>.mp4)")
    args = parser.parse_args()

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")

    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found. Run calibrate.py first.")
        return

    with open(config_path) as f:
        config = json.load(f)

    frames_glob = os.path.join(script_dir, "Data", "Extracted_Frames", "frame_*.jpg")
    frame_files = sorted(glob.glob(frames_glob))
    if not frame_files:
        print("No frames found in Data/Extracted_Frames/")
        return

    output_name = args.output or f"output_{args.mode}.mp4"
    output_path = os.path.join(script_dir, output_name)

    process(frame_files, config, args.mode, not args.no_shake, output_path, normalize=args.normalize)


if __name__ == "__main__":
    main()
