import cv2
import os

def extract_frames(video_path, output_dir, frames_per_second=2):
    """
    Extracts frames from a video at a specified rate.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get the frames per second of the original video
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Original Video FPS: {fps:.2f}")
    
    # Calculate how many frames to skip to get the desired output rate
    frame_interval = max(1, int(fps / frames_per_second))
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Save frame if it matches our interval
        if frame_count % frame_interval == 0:
            output_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1
            
        frame_count += 1

    cap.release()
    print(f"Extraction complete. Saved {saved_count} frames to '{output_dir}'")

if __name__ == "__main__":
    video_path = "Data/VID_20260924_174810.mp4"
    output_dir = "Data/Extracted_Frames"
    
    print(f"Starting frame extraction from {video_path}...")
    # Extracting 2 frames per second to keep dataset size manageable but capture movement
    extract_frames(video_path, output_dir, frames_per_second=2)
