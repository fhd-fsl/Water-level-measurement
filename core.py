import cv2
import numpy as np
import random

# --- ORB Feature Tracking ---

def setup_reference(config, reference_frame):
    """
    Extracts ORB keypoints and descriptors from the ruler region of the reference frame.
    This is called once at the start.
    
    Returns a dict containing everything needed for per-frame tracking.
    """
    bbox = config["ruler_bbox"]
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    
    # Use the full ruler region for feature extraction
    gray_ref = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
    
    # Create a mask that covers only the ruler region
    mask = np.zeros(gray_ref.shape, dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    
    # Detect ORB features in the ruler region
    orb = cv2.ORB_create(nfeatures=1000)
    kp_ref, des_ref = orb.detectAndCompute(gray_ref, mask)
    
    if des_ref is None or len(kp_ref) < 10:
        raise ValueError(f"Only found {len(kp_ref) if kp_ref else 0} features in the ruler region. "
                         "Make sure the bounding box covers the ruler properly.")
    
    return {
        "orb": orb,
        "kp_ref": kp_ref,
        "des_ref": des_ref,
        "bbox": bbox,
        "marks": config["marks"],
        "reference_frame": reference_frame
    }

def track_ruler(frame, ref_data):
    """
    Uses ORB feature matching + homography to find how the ruler has moved/rotated
    relative to the reference frame.
    
    Returns:
        shifted_bbox: dict with shifted x1, y1, x2, y2
        shifted_marks: list of shifted mark dicts
        H: the 3x3 homography matrix (or None if tracking failed)
        match_info: dict with debug info (num_matches, inliers, etc.)
    """
    orb = ref_data["orb"]
    kp_ref = ref_data["kp_ref"]
    des_ref = ref_data["des_ref"]
    bbox = ref_data["bbox"]
    marks = ref_data["marks"]
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect features in the current frame (full frame, no mask)
    kp_cur, des_cur = orb.detectAndCompute(gray, None)
    
    if des_cur is None or len(kp_cur) < 10:
        return bbox, marks, None, {"status": "too_few_features", "num_matches": 0}
    
    # Match features using BFMatcher with Hamming distance (for ORB)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des_ref, des_cur, k=2)
    
    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for m_pair in matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    
    if len(good_matches) < 8:
        return bbox, marks, None, {"status": "too_few_good_matches", "num_matches": len(good_matches)}
    
    # Extract matched point coordinates
    src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_cur[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Compute homography using RANSAC (robust to outliers)
    H, mask_inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        return bbox, marks, None, {"status": "homography_failed", "num_matches": len(good_matches)}
    
    num_inliers = int(mask_inliers.sum()) if mask_inliers is not None else 0
    
    # Transform the bounding box corners through the homography
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    
    # Get axis-aligned bounding box from transformed corners
    h, w = frame.shape[:2]
    tx1 = int(max(0, min(transformed_corners[:, 0])))
    ty1 = int(max(0, min(transformed_corners[:, 1])))
    tx2 = int(min(w, max(transformed_corners[:, 0])))
    ty2 = int(min(h, max(transformed_corners[:, 1])))
    
    shifted_bbox = {"x1": tx1, "y1": ty1, "x2": tx2, "y2": ty2}
    
    # Transform all inch mark coordinates through the homography
    shifted_marks = []
    for mark in marks:
        pt = np.float32([[mark["x"], mark["y"]]]).reshape(-1, 1, 2)
        transformed_pt = cv2.perspectiveTransform(pt, H).reshape(-1, 2)[0]
        shifted_marks.append({
            "inch": mark["inch"],
            "x": int(transformed_pt[0]),
            "y": int(transformed_pt[1])
        })
    
    match_info = {
        "status": "ok",
        "num_matches": len(good_matches),
        "num_inliers": num_inliers,
        "transformed_corners": transformed_corners
    }
    
    return shifted_bbox, shifted_marks, H, match_info


# --- Bottom-Up Edge Density Scan (same proven algorithm) ---

def calculate_water_y(frame, bbox):
    """
    Runs the bottom-up edge density scan on the given bounding box region.
    Returns: abs_y, crop, edges, smoothed, threshold, top_y_crop
    """
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    
    crop = frame[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return y2, crop, np.array([]), np.array([]), 0, 0
    
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge Detection
    edges = cv2.Canny(gray, 30, 120)
    
    # Horizontal Edge Density
    row_sums = np.sum(edges, axis=1)
    
    # Smooth the density vertically (50-pixel window)
    window_size = 50
    smoothed = np.convolve(row_sums, np.ones(window_size)/window_size, mode='same')
    
    # Find water level by scanning from bottom UPWARDS
    threshold = np.max(smoothed) * 0.25
    top_y_crop = len(smoothed) - 1
    
    for y in range(len(smoothed)-1, -1, -1):
        if smoothed[y] > threshold:
            top_y_crop = y
            break
            
    abs_y = top_y_crop + y1
    
    return abs_y, crop, edges, smoothed, threshold, top_y_crop


def get_water_level(y_pixel, marks):
    """Interpolates the physical measurement (inches) based on the y pixel coordinate."""
    sorted_marks = sorted(marks, key=lambda m: m["y"])
    y_vals = [m["y"] for m in sorted_marks]
    inch_vals = [m["inch"] for m in sorted_marks]
    return np.interp(y_pixel, y_vals, inch_vals)


# --- Artificial Shake ---

def apply_shake(frame, max_translate=15, max_angle=2.0):
    """
    Applies random artificial camera shake to a frame.
    Returns: shaken_frame, tx, ty, angle
    """
    h, w = frame.shape[:2]
    
    tx = random.randint(-max_translate, max_translate)
    ty = random.randint(-max_translate, max_translate)
    angle = random.uniform(-max_angle, max_angle)
    
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    
    shaken = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return shaken, tx, ty, angle
