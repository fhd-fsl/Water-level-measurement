import cv2
import json
import glob
import os
import numpy as np

from core import setup_reference, track_ruler, calculate_water_y, get_water_level, apply_shake

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")
    
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found. Run calibrate.py first.")
        return
        
    with open(config_path, "r") as f:
        config = json.load(f)
    
    bbox = config["ruler_bbox"]
    
    # Get all frames
    frames_dir = os.path.join(script_dir, "Data", "Extracted_Frames", "frame_*.jpg")
    frame_files = sorted(glob.glob(frames_dir))
    if not frame_files:
        print("No frames found!")
        return

    # Load reference frame and setup ORB features
    ref_frame = cv2.imread(frame_files[0])
    print("Setting up ORB reference features...")
    ref_data = setup_reference(config, ref_frame)
    print(f"Extracted {len(ref_data['kp_ref'])} reference keypoints.\n")

    # Prepare output video
    output_path = os.path.join(script_dir, "output_measurement.mp4")
    h, w, _ = ref_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_path, fourcc, 15.0, (w, h))

    # Artificial shake settings
    ENABLE_SHAKE = True
    max_translate = 15   # pixels
    max_angle = 2.0      # degrees

    # Temporal Smoothing (Moving Average)
    history = []
    smoothing_frames = 2
    
    print(f"Processing {len(frame_files)} frames with Feature Tracking + Edge Scan...")
    if ENABLE_SHAKE:
        print(f"Artificial shake ENABLED (translate=±{max_translate}px, rotation=±{max_angle}°)")
    print()
    
    for i, f_path in enumerate(frame_files):
        frame = cv2.imread(f_path)
        
        # Apply artificial shake
        shake_info = None
        if ENABLE_SHAKE:
            frame, tx, ty, angle = apply_shake(frame, max_translate, max_angle)
            shake_info = (tx, ty, angle)
        
        frame_display = frame.copy()
        
        # 1. Track the ruler
        shifted_bbox, shifted_marks, H, match_info = track_ruler(frame, ref_data)
        
        # 2. Run bottom-up edge scan on the tracked region
        abs_y, crop, edges, smoothed, threshold, top_y_crop = calculate_water_y(frame, shifted_bbox)
        
        # 3. Temporal smoothing
        history.append(abs_y)
        if len(history) > smoothing_frames:
            history.pop(0)
        smoothed_y = int(np.mean(history))
        
        # 4. Interpolate inches using shifted marks
        water_level_inch = get_water_level(smoothed_y, shifted_marks)
        
        # --- Draw on display frame ---
        sx1, sy1 = shifted_bbox["x1"], shifted_bbox["y1"]
        sx2, sy2 = shifted_bbox["x2"], shifted_bbox["y2"]
        
        # Draw tracked bounding box
        if match_info["status"] == "ok":
            corners = match_info["transformed_corners"].astype(np.int32)
            cv2.polylines(frame_display, [corners], True, (0, 255, 0), 3)
        else:
            cv2.rectangle(frame_display, (sx1, sy1), (sx2, sy2), (0, 0, 255), 3)
        
        # Draw shifted inch marks
        for mark in shifted_marks:
            cv2.circle(frame_display, (mark["x"], mark["y"]), 5, (255, 0, 255), -1)
        
        # Draw water level
        cv2.line(frame_display, (sx1, smoothed_y), (sx2, smoothed_y), (0, 255, 255), 4)
        text = f"Level: {water_level_inch:.2f} in"
        cv2.putText(frame_display, text, (sx2 + 20, smoothed_y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)
        
        # Draw tracking info
        if match_info["status"] == "ok":
            info = f"Track: {match_info['num_inliers']}/{match_info['num_matches']}"
            cv2.putText(frame_display, info, (30, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Draw shake info
        if shake_info:
            tx, ty, angle = shake_info
            shake_text = f"Shake: tx={tx} ty={ty} rot={angle:.1f}"
            cv2.putText(frame_display, shake_text, (30, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
        
        # Frame counter
        cv2.putText(frame_display, f"Frame {i+1}/{len(frame_files)}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Display live
        display_h, display_w = frame_display.shape[:2]
        scale = 800 / display_h
        resized = cv2.resize(frame_display, (int(display_w * scale), int(display_h * scale)))
        
        cv2.imshow("Feature Tracking Water Measurement", resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        out_video.write(frame_display)
        
        # Progress update
        if (i + 1) % 20 == 0:
            status = match_info["status"]
            matches = match_info.get("num_inliers", 0)
            print(f"  Frame {i+1}/{len(frame_files)} | Level: {water_level_inch:.2f} in | "
                  f"Track: {status} ({matches} inliers)")
        
    out_video.release()
    cv2.destroyAllWindows()
    print(f"\nFinished! Output saved to {output_path}")

if __name__ == "__main__":
    main()
