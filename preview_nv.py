"""
preview_nv.py

Plays all extracted frames through the night-vision filter in real time.

Controls:
    q          — quit
    SPACE      — pause / resume
    +  /  =    — speed up
    -          — slow down
"""

import cv2
import glob
import os
import sys

# Import the simulation function from the existing script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulate_nightvision import simulate_nv, MODE, NOISE_SIGMA, IR_HOTSPOT_STRENGTH, \
    IR_HOTSPOT_RADIUS, GAMMA, BLUR_SIGMA, CONTRAST_CLIP

# ---------------------------------------------------------------------------
# Playback settings
# ---------------------------------------------------------------------------
DISPLAY_HEIGHT = 800      # window height in pixels (width scales automatically)
START_DELAY_MS = 40       # ms between frames (~25 FPS); Space pauses, +/- adjusts

# ---------------------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    frames_dir = os.path.join(script_dir, "Data", "Extracted_Frames", "frame_*.jpg")
    frame_files = sorted(glob.glob(frames_dir))

    if not frame_files:
        print("No frames found in Data/Extracted_Frames/")
        return

    print(f"Found {len(frame_files)} frames.")
    print(f"Mode: {MODE} | noise={NOISE_SIGMA} | hotspot={IR_HOTSPOT_STRENGTH} "
          f"| gamma={GAMMA} | blur={BLUR_SIGMA}")
    print("Controls: SPACE = pause/resume   +/= = faster   - = slower   Q = quit")

    cv2.namedWindow("NV Preview", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    delay_ms = START_DELAY_MS
    paused = False
    i = 0

    while True:
        if not paused:
            frame = cv2.imread(frame_files[i])
            nv = simulate_nv(
                frame,
                mode=MODE,
                noise_sigma=NOISE_SIGMA,
                hotspot_strength=IR_HOTSPOT_STRENGTH,
                hotspot_radius=IR_HOTSPOT_RADIUS,
                gamma=GAMMA,
                blur_sigma=BLUR_SIGMA,
                contrast_clip=CONTRAST_CLIP,
            )

            # Overlay: frame counter + current FPS
            fps_approx = 1000 / max(delay_ms, 1)
            label = f"Frame {i+1}/{len(frame_files)}  |  ~{fps_approx:.0f} FPS  |  SPACE=pause  Q=quit"
            cv2.putText(nv, label, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (180, 180, 180), 3)

            # Resize for display
            h, w = nv.shape[:2]
            scale = DISPLAY_HEIGHT / h
            display = cv2.resize(nv, (int(w * scale), DISPLAY_HEIGHT))

            cv2.imshow("NV Preview", display)
            i = (i + 1) % len(frame_files)   # loop back to start

        key = cv2.waitKey(delay_ms if not paused else 50) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print("Paused." if paused else "Resumed.")
        elif key in (ord('+'), ord('=')):
            delay_ms = max(10, delay_ms - 10)
            print(f"Speed up → ~{1000//delay_ms} FPS")
        elif key == ord('-'):
            delay_ms = min(500, delay_ms + 10)
            print(f"Slow down → ~{1000//delay_ms} FPS")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
