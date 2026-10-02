"""
run.py — unified water level measurement pipeline

Usage:
    python run.py --mode normal          standard camera
    python run.py --mode nv              night-vision camera (simulated)
    python run.py --mode mixed           first half normal, second half NV
    python run.py --mode nv --no-shake   disable artificial shake
    python run.py --mode mixed --output my_run.mp4

The NV filter reads its settings from simulate_nightvision.py.
Edge-scan parameters are tuned per mode in EDGE_PARAMS below.
"""

import argparse
import cv2
import json
import glob
import os
import numpy as np

from core import setup_reference, track_ruler, calculate_water_y, get_water_level, apply_shake
from simulate_nightvision import simulate_nv, MODE, NOISE_SIGMA, IR_HOTSPOT_STRENGTH, \
    IR_HOTSPOT_RADIUS, GAMMA, BLUR_SIGMA, CONTRAST_CLIP

# ---------------------------------------------------------------------------
# Per-mode edge scan parameters
# Tune these if you adjust the NV filter settings in simulate_nightvision.py
# ---------------------------------------------------------------------------
EDGE_PARAMS = {
    "normal": dict(
        blur_ksize=5,
        canny_low=30, canny_high=120,
        smooth_window=50,
        threshold_frac=0.25,
    ),
    "nv": dict(
        blur_ksize=9,       # stronger blur suppresses IR sensor grain
        canny_low=50, canny_high=150,   # ignore weak noise edges
        smooth_window=50,
        threshold_frac=0.40,  # higher bar — only trigger on strong edge bands
    ),
}

# Shake settings (mirrors original measure_water.py)
SHAKE_MAX_TRANSLATE = 15   # pixels
SHAKE_MAX_ANGLE     = 2.0  # degrees
SMOOTHING_FRAMES    = 2    # moving-average window


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_nv_filter(frame):
    return simulate_nv(frame, MODE, NOISE_SIGMA, IR_HOTSPOT_STRENGTH,
                       IR_HOTSPOT_RADIUS, GAMMA, BLUR_SIGMA, CONTRAST_CLIP)


def frame_mode_for(i, total, mode):
    """Return 'normal' or 'nv' for frame index i given the run mode."""
    if mode == "normal":
        return "normal"
    if mode == "nv":
        return "nv"
    # mixed: first half normal, second half nv
    return "normal" if i < total // 2 else "nv"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process(frame_files, config, mode, enable_shake, output_path):
    n = len(frame_files)

    # Reference frame is always normal — ORB is robust enough across the
    # normal→NV transition in mixed mode (confirmed by evaluate_nv.py results)
    ref_raw   = cv2.imread(frame_files[0])
    ref_frame = apply_nv_filter(ref_raw) if mode == "nv" else ref_raw

    print("Setting up ORB reference features...")
    ref_data = setup_reference(config, ref_frame)
    print(f"Extracted {len(ref_data['kp_ref'])} reference keypoints.")
    print(f"Mode: {mode}  |  shake: {'on' if enable_shake else 'off'}")
    if mode == "nv":
        print(f"NV filter: {MODE}, noise={NOISE_SIGMA}, hotspot={IR_HOTSPOT_STRENGTH}, "
              f"gamma={GAMMA}, blur={BLUR_SIGMA}")
    if mode == "mixed":
        print(f"Mixed split: frames 0–{n//2 - 1} normal, {n//2}–{n-1} NV")
    print()

    h, w = ref_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_path, fourcc, 15.0, (w, h))

    history = []
    failed  = 0

    for i, f_path in enumerate(frame_files):
        raw  = cv2.imread(f_path)
        fmode = frame_mode_for(i, n, mode)
        frame = apply_nv_filter(raw) if fmode == "nv" else raw

        # Artificial shake
        shake_info = None
        if enable_shake:
            frame, tx, ty, angle = apply_shake(frame, SHAKE_MAX_TRANSLATE, SHAKE_MAX_ANGLE)
            shake_info = (tx, ty, angle)

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

        # Mode label — white for normal, cyan for NV
        mode_colour = (200, 200, 200) if fmode == "normal" else (0, 220, 255)
        mode_label  = f"Cam: {fmode.upper()}"
        if mode == "mixed":
            half = "1st half" if i < n // 2 else "2nd half"
            mode_label += f"  [{half}]"
        cv2.putText(frame_display, mode_label, (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, mode_colour, 2)

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
    parser = argparse.ArgumentParser(
        description="Water level measurement — normal / NV / mixed modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python run.py --mode normal
  python run.py --mode nv
  python run.py --mode mixed
  python run.py --mode nv --no-shake
  python run.py --mode mixed --output test_mixed.mp4
        """,
    )
    parser.add_argument(
        "--mode", choices=["normal", "nv", "mixed"], default="normal",
        help="normal — standard camera  |  nv — night-vision  |  mixed — 1st half normal, 2nd half NV",
    )
    parser.add_argument("--no-shake", action="store_true",
                        help="Disable artificial camera shake simulation")
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

    process(frame_files, config, args.mode, not args.no_shake, output_path)


if __name__ == "__main__":
    main()
