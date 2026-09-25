import cv2
import json
import os

# Global variables for interactive selection
ref_pt = []
ruler_box = []
drawing = False
image_copy = None

def click_and_crop(event, x, y, flags, param):
    global ref_pt, ruler_box, drawing, image_copy

    # 1. Bounding box selection (drag left mouse button)
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(ruler_box) < 2:
            drawing = True
            ruler_box = [(x, y)]
        elif len(ref_pt) < 13:
            # 2. Point selection for markings (single click)
            ref_pt.append((x, y))
            cv2.circle(image_copy, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(image_copy, f"{len(ref_pt)-1} in", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow("Calibration", image_copy)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing and len(ruler_box) == 1:
            temp_img = image_copy.copy()
            cv2.rectangle(temp_img, ruler_box[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("Calibration", temp_img)

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing and len(ruler_box) == 1:
            drawing = False
            ruler_box.append((x, y))
            cv2.rectangle(image_copy, ruler_box[0], ruler_box[1], (0, 255, 0), 2)
            cv2.imshow("Calibration", image_copy)
            print("Bounding box defined. Now click exactly 13 points (from 0 to 12 inches) for the markings.")

def calibrate():
    global image_copy, ref_pt, ruler_box
    
    # Load the first extracted frame
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "Data", "Extracted_Frames", "frame_0000.jpg")
    if not os.path.exists(image_path):
        print(f"Error: Could not find {image_path}. Did you run extract_frames.py?")
        return

    image = cv2.imread(image_path)
    image_copy = image.copy()
    
    h, w = image.shape[:2]
    # Set a reasonable max height for the window (e.g., 800 pixels) and calculate width to maintain aspect ratio
    max_height = 800
    scale = max_height / h
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("Calibration", new_w, new_h)
    cv2.setMouseCallback("Calibration", click_and_crop)

    print("--- INSTRUCTIONS ---")
    print("1. Click, HOLD your mouse button, DRAG, and RELEASE to draw a bounding box around the ruler.")
    print("2. Once the box is drawn, click exactly 13 points inside the box to define each inch mark from 0 to 12.")
    print("3. Press 'c' to confirm and save, or 'r' to reset, or 'q' to quit.")

    while True:
        cv2.imshow("Calibration", image_copy)
        key = cv2.waitKey(1) & 0xFF

        # Reset
        if key == ord("r"):
            image_copy = image.copy()
            ref_pt = []
            ruler_box = []
            print("Resetting... draw bounding box again.")
            
        # Confirm and save
        elif key == ord("c"):
            if len(ruler_box) == 2 and len(ref_pt) == 13:
                # Ensure box coordinates are correct (top-left, bottom-right)
                x1 = min(ruler_box[0][0], ruler_box[1][0])
                y1 = min(ruler_box[0][1], ruler_box[1][1])
                x2 = max(ruler_box[0][0], ruler_box[1][0])
                y2 = max(ruler_box[0][1], ruler_box[1][1])
                
                marks = []
                for i, pt in enumerate(ref_pt):
                    marks.append({"inch": i, "x": pt[0], "y": pt[1]})
                
                config = {
                    "ruler_bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "marks": marks
                }
                
                config_path = os.path.join(script_dir, "config.json")
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=4)
                
                print(f"\nCalibration saved to {config_path}!")
                print(json.dumps(config, indent=2))
                break
            else:
                print("Incomplete! Please draw a box and select exactly 13 points before pressing 'c'.")
                
        # Quit
        elif key == ord("q"):
            print("Exiting without saving.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    calibrate()
