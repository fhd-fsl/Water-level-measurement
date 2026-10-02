"""
evaluate_nv.py

Runs both the standard and NV pipelines on every frame, collects per-frame
water level readings, and produces two diagnostic outputs:

  evaluation_plot.png      — measurement traces + tracking inliers side by side
  evaluation_edgescan.png  — edge density debug for 6 sample frames (normal vs NV)
"""

import cv2
import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # no display needed — save to file
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from core import setup_reference, track_ruler, calculate_water_y, get_water_level
from simulate_nightvision import simulate_nv, MODE, NOISE_SIGMA, IR_HOTSPOT_STRENGTH, \
    IR_HOTSPOT_RADIUS, GAMMA, BLUR_SIGMA, CONTRAST_CLIP

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SMOOTHING_FRAMES = 2          # keep same as measure_water.py
SAMPLE_FRAME_INDICES = [0, 50, 100, 150, 200, 250]   # frames to show edge debug

# ---------------------------------------------------------------------------

def run_pipeline(frame_files, config, apply_nv=False):
    """
    Runs the full pipeline on all frames.
    Returns:
        levels   — list of float (water level in inches per frame)
        inliers  — list of int (ORB inlier count per frame, 0 on failure)
        statuses — list of str
    """
    ref_raw = cv2.imread(frame_files[0])
    ref_frame = simulate_nv(ref_raw, MODE, NOISE_SIGMA, IR_HOTSPOT_STRENGTH,
                            IR_HOTSPOT_RADIUS, GAMMA, BLUR_SIGMA, CONTRAST_CLIP) \
                if apply_nv else ref_raw

    ref_data = setup_reference(config, ref_frame)

    history = []
    levels, inliers, statuses = [], [], []

    for f_path in frame_files:
        raw = cv2.imread(f_path)
        frame = simulate_nv(raw, MODE, NOISE_SIGMA, IR_HOTSPOT_STRENGTH,
                            IR_HOTSPOT_RADIUS, GAMMA, BLUR_SIGMA, CONTRAST_CLIP) \
                if apply_nv else raw

        shifted_bbox, shifted_marks, H, match_info = track_ruler(frame, ref_data)
        if apply_nv:
            abs_y, *_ = calculate_water_y(
                frame, shifted_bbox,
                blur_ksize=9, canny_low=50, canny_high=150,
                smooth_window=50, threshold_frac=0.40,
            )
        else:
            abs_y, *_ = calculate_water_y(frame, shifted_bbox)

        history.append(abs_y)
        if len(history) > SMOOTHING_FRAMES:
            history.pop(0)
        smoothed_y = int(np.mean(history))

        level = get_water_level(smoothed_y, shifted_marks)
        levels.append(level)
        inliers.append(match_info.get("num_inliers", 0))
        statuses.append(match_info["status"])

    return levels, inliers, statuses


def edge_debug_pair(frame_files, config, sample_indices):
    """
    For each sample index, returns the edge density arrays for normal and NV.
    Returns list of dicts with keys: idx, normal_smoothed, nv_smoothed,
    normal_threshold, nv_threshold, normal_top, nv_top
    """
    ref_raw = cv2.imread(frame_files[0])
    ref_nv  = simulate_nv(ref_raw, MODE, NOISE_SIGMA, IR_HOTSPOT_STRENGTH,
                           IR_HOTSPOT_RADIUS, GAMMA, BLUR_SIGMA, CONTRAST_CLIP)

    ref_data_normal = setup_reference(config, ref_raw)
    ref_data_nv     = setup_reference(config, ref_nv)

    results = []
    for idx in sample_indices:
        if idx >= len(frame_files):
            continue
        raw = cv2.imread(frame_files[idx])
        nv  = simulate_nv(raw, MODE, NOISE_SIGMA, IR_HOTSPOT_STRENGTH,
                          IR_HOTSPOT_RADIUS, GAMMA, BLUR_SIGMA, CONTRAST_CLIP)

        # Normal
        bbox_n, _, _, _ = track_ruler(raw, ref_data_normal)
        _, _, _, sm_n, thr_n, top_n = calculate_water_y(raw, bbox_n)

        # NV
        bbox_v, _, _, _ = track_ruler(nv, ref_data_nv)
        _, _, _, sm_v, thr_v, top_v = calculate_water_y(nv, bbox_v)

        results.append({
            "idx": idx,
            "normal_smoothed":  sm_n,  "normal_threshold":  thr_n,  "normal_top":  top_n,
            "nv_smoothed":      sm_v,  "nv_threshold":      thr_v,  "nv_top":      top_v,
        })
    return results


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")
    frames_glob = os.path.join(script_dir, "Data", "Extracted_Frames", "frame_*.jpg")
    frame_files = sorted(glob.glob(frames_glob))

    with open(config_path) as f:
        config = json.load(f)

    print(f"Frames: {len(frame_files)}")
    print("Running standard pipeline...")
    lev_norm, inl_norm, sta_norm = run_pipeline(frame_files, config, apply_nv=False)
    print("Running NV pipeline...")
    lev_nv,   inl_nv,   sta_nv   = run_pipeline(frame_files, config, apply_nv=True)
    print("Collecting edge-scan debug data...")
    edge_data = edge_debug_pair(frame_files, config, SAMPLE_FRAME_INDICES)

    # -----------------------------------------------------------------------
    # Plot 1 — measurement trace + inliers
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    xs = np.arange(len(frame_files))

    axes[0].plot(xs, lev_norm, color="deepskyblue",  lw=1.5, label="Normal")
    axes[0].plot(xs, lev_nv,   color="orangered",    lw=1.5, label="NV",    alpha=0.85)
    axes[0].set_ylabel("Water level (inches)")
    axes[0].set_title("Water Level — Normal vs NV")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(xs, np.array(lev_norm) - np.array(lev_nv),
                 color="gold", lw=1.2)
    axes[1].axhline(0, color="white", lw=0.8, ls="--")
    axes[1].set_ylabel("Δ level (normal − NV, inches)")
    axes[1].set_title("Measurement Error (NV bias)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(xs, inl_norm, color="deepskyblue", lw=1.2, label="Normal inliers")
    axes[2].plot(xs, inl_nv,   color="orangered",   lw=1.2, label="NV inliers", alpha=0.85)
    axes[2].set_ylabel("ORB inliers")
    axes[2].set_xlabel("Frame index")
    axes[2].set_title("Tracking Inliers")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.patch.set_facecolor("#1e1e2e")
    for ax in axes:
        ax.set_facecolor("#2a2a3e")
        ax.tick_params(colors="white")
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555")

    plt.tight_layout()
    out1 = os.path.join(script_dir, "evaluation_plot.png")
    plt.savefig(out1, dpi=120, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out1}")

    # -----------------------------------------------------------------------
    # Plot 2 — edge density debug panels
    # -----------------------------------------------------------------------
    n_samples = len(edge_data)
    fig2 = plt.figure(figsize=(18, 4 * n_samples))
    fig2.patch.set_facecolor("#1e1e2e")
    gs = gridspec.GridSpec(n_samples, 2, figure=fig2, hspace=0.55, wspace=0.3)

    for row, d in enumerate(edge_data):
        for col, (label, sm, thr, top, colour) in enumerate([
            ("Normal", d["normal_smoothed"], d["normal_threshold"], d["normal_top"], "deepskyblue"),
            ("NV",     d["nv_smoothed"],     d["nv_threshold"],     d["nv_top"],     "orangered"),
        ]):
            ax = fig2.add_subplot(gs[row, col])
            ax.set_facecolor("#2a2a3e")

            ys = np.arange(len(sm))
            ax.plot(sm, ys, color=colour, lw=1.5)
            ax.axvline(thr, color="yellow",  lw=1.2, ls="--", label=f"threshold ({thr:.0f})")
            ax.axhline(top, color="lime",    lw=1.5, ls="-",  label=f"detected y={top}")
            ax.invert_yaxis()

            ax.set_title(f"Frame {d['idx']} — {label}", color="white", fontsize=10)
            ax.set_xlabel("Smoothed edge density", color="white", fontsize=8)
            ax.set_ylabel("Row (0=top)", color="white", fontsize=8)
            ax.tick_params(colors="white", labelsize=7)
            ax.legend(fontsize=7, facecolor="#1e1e2e", labelcolor="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#555")

    out2 = os.path.join(script_dir, "evaluation_edgescan.png")
    plt.savefig(out2, dpi=120, facecolor=fig2.get_facecolor())
    plt.close()
    print(f"Saved: {out2}")

    # -----------------------------------------------------------------------
    # Summary stats
    # -----------------------------------------------------------------------
    diff = np.array(lev_norm) - np.array(lev_nv)
    fail_nv = sum(1 for s in sta_nv if s != "ok")
    print(f"\n--- Summary ---")
    print(f"Normal  mean±std: {np.mean(lev_norm):.3f} ± {np.std(lev_norm):.3f} in")
    print(f"NV      mean±std: {np.mean(lev_nv):.3f} ± {np.std(lev_nv):.3f} in")
    print(f"Bias (normal−NV): mean={np.mean(diff):.3f} in  max={np.max(np.abs(diff)):.3f} in")
    print(f"NV tracking failures: {fail_nv}/{len(frame_files)}")


if __name__ == "__main__":
    main()
