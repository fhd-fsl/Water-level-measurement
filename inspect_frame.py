import cv2
import json
import os
import numpy as np

from core import setup_reference, track_ruler, calculate_water_y, get_water_level, apply_shake

def visualize_frame(frame, ref_data, use_shake=False):
    """
    Full pipeline visualization:
    - Tracks the ruler via ORB + Homography
    - Runs bottom-up edge scan
    - Draws tracked bounding box, feature matches, and water level
    """
    shake_info = None
    if use_shake:
        frame, tx, ty, angle = apply_shake(frame)
        shake_info = (tx, ty, angle)
    
    # 1. Track the ruler
    shifted_bbox, shifted_marks, H, match_info = track_ruler(frame, ref_data)
    
    # 2. Run bottom-up edge scan on the tracked region
    abs_y, crop, edges, smoothed, threshold, top_y_crop = calculate_water_y(frame, shifted_bbox)
    
    # 3. Interpolate inches using shifted marks
    water_level_inch = get_water_level(abs_y, shifted_marks)
    
    # --- Visualization ---
    display = frame.copy()
    x1, y1, x2, y2 = shifted_bbox["x1"], shifted_bbox["y1"], shifted_bbox["x2"], shifted_bbox["y2"]
    
    # Draw the tracked bounding box (green if tracking OK, red if failed)
    if match_info["status"] == "ok":
        # Draw the transformed quadrilateral (not just axis-aligned box)
        corners = match_info["transformed_corners"].astype(np.int32)
        cv2.polylines(display, [corners], True, (0, 255, 0), 3)
        
        # Draw tracking info
        info_text = f"Tracking: {match_info['num_inliers']} inliers / {match_info['num_matches']} matches"
        cv2.putText(display, info_text, (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    else:
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(display, f"Tracking: {match_info['status']}", (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    # Draw shifted inch marks
    for mark in shifted_marks:
        cv2.circle(display, (mark["x"], mark["y"]), 6, (255, 0, 255), -1)
        cv2.putText(display, f"{mark['inch']}\"", (mark["x"] + 10, mark["y"]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    # Draw the detected water level line
    cv2.line(display, (x1, abs_y), (x2, abs_y), (0, 255, 255), 4)
    text = f"Level: {water_level_inch:.2f} in"
    cv2.putText(display, text, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)
    
    # Draw shake info
    if shake_info:
        tx, ty, angle = shake_info
        shake_text = f"Shake: tx={tx} ty={ty} rot={angle:.1f}deg"
        cv2.putText(display, shake_text, (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
    
    # Resize for screen
    h, w = display.shape[:2]
    scale = 800 / h
    resized = cv2.resize(display, (int(w * scale), int(h * scale)))
    
    return resized, water_level_inch, match_info

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")
    
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found. Run calibrate.py first.")
        return
        
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Load reference frame (frame 0) and setup ORB features
    ref_frame_path = os.path.join(script_dir, "Data", "Extracted_Frames", "frame_0000.jpg")
    ref_frame = cv2.imread(ref_frame_path)
    if ref_frame is None:
        print(f"Error: Could not load reference frame: {ref_frame_path}")
        return
    
    print("Setting up ORB reference features...")
    ref_data = setup_reference(config, ref_frame)
    print(f"Extracted {len(ref_data['kp_ref'])} reference keypoints from the ruler region.")
    
    print("\n--- Feature Tracking: Frame Inspector ---")
    print("Type a frame number (e.g., 50, 150) and press Enter.")
    print("Add 's' after the number to apply artificial shake (e.g., '50s').")
    print("Type 'q' to quit.\n")
    
    while True:
        user_input = input("Enter frame number (or 'q' to quit): ").strip()
        if user_input.lower() == 'q':
            break
        
        use_shake = user_input.endswith('s')
        num_str = user_input.rstrip('s')
        
        try:
            frame_num = int(num_str)
        except ValueError:
            print("Please enter a valid number.")
            continue
            
        frame_name = f"frame_{frame_num:04d}.jpg"
        frame_path = os.path.join(script_dir, "Data", "Extracted_Frames", frame_name)
        
        if not os.path.exists(frame_path):
            print(f"Error: Could not find {frame_path}")
            continue
            
        frame = cv2.imread(frame_path)
        
        annotated, level, match_info = visualize_frame(frame, ref_data, use_shake)
        
        status = match_info["status"]
        if status == "ok":
            print(f"  Water level: {level:.2f} in | "
                  f"Tracking: {match_info['num_inliers']} inliers / {match_info['num_matches']} matches")
        else:
            print(f"  Tracking status: {status}")
        
        cv2.imshow("Feature Tracking Pipeline", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
