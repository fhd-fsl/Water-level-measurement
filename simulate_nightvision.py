"""
simulate_nightvision.py

Simulates a near-IR / night-vision camera feed on an existing frame.
Tune the parameters at the top, then run:

    python simulate_nightvision.py

Outputs:
    nv_sample.jpg          — the simulated NV frame (full resolution)
    nv_sample_compare.jpg  — side-by-side comparison with the original
"""

import cv2
import numpy as np
import os

# ---------------------------------------------------------------------------
# Parameters — tweak these
# ---------------------------------------------------------------------------

# Which frame to simulate on
FRAME_PATH = os.path.join("Data", "Extracted_Frames", "frame_0050.jpg")

# Output mode:
#   'ir_security' — plain grayscale, typical NIR surveillance cam
#   'nvg_green'   — classic green phosphor tint (night-vision goggles style)
MODE = "ir_security"

# Sensor noise — gaussian std dev added to the image (0 = none, 15 = heavy grain)
# Real short-range IR security cameras are relatively clean; 3 is more realistic
NOISE_SIGMA = 3.0

# IR illuminator hotspot — IR LEDs around the lens create a bright centre bloom.
# Strength 0.0 = no hotspot, 1.0 = strong bloom (+80% brightness at centre)
IR_HOTSPOT_STRENGTH = 0.35

# How wide the hotspot is — fraction of the image diagonal (0.4 = tight, 0.8 = wide)
IR_HOTSPOT_RADIUS = 0.55

# Gamma applied to the grayscale image.
# < 1.0 brightens midtones (typical IR sensor response), 1.0 = linear
GAMMA = 0.80

# Slight blur to simulate NIR focus softness (0 = off, 1.5 = mild, 3.0 = heavy)
BLUR_SIGMA = 1.2

# Auto-contrast clipping percentiles — stretches the final histogram
# (1, 99) is a gentle stretch; (2, 98) is more aggressive
CONTRAST_CLIP = (1, 99)

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def build_hotspot_mask(h, w, strength, radius_fraction):
    """
    Creates a radial gradient mask that is brightest at the centre,
    simulating the fall-off from an IR illuminator ring around the lens.
    Returns a float32 mask in the range [1.0, 1.0 + strength].
    """
    cx, cy = w / 2, h / 2
    diag = np.sqrt(cx**2 + cy**2)
    radius_px = diag * radius_fraction

    ys, xs = np.ogrid[:h, :w]
    dist = np.sqrt((xs - cx)**2 + (ys - cy)**2)

    # Gaussian-shaped hotspot
    hotspot = np.exp(-0.5 * (dist / (radius_px / 2.5))**2)
    return 1.0 + strength * hotspot.astype(np.float32)


def simulate_nv(frame, mode, noise_sigma, hotspot_strength, hotspot_radius,
                gamma, blur_sigma, contrast_clip):
    h, w = frame.shape[:2]

    # 1. Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 2. Gamma correction (brightens midtones, simulates IR sensor response curve)
    gray = 255.0 * np.power(gray / 255.0, gamma)

    # 3. Apply IR illuminator hotspot
    if hotspot_strength > 0:
        mask = build_hotspot_mask(h, w, hotspot_strength, hotspot_radius)
        gray = gray * mask

    # 4. Slight blur (NIR focal plane softness)
    if blur_sigma > 0:
        ksize = int(blur_sigma * 4) | 1   # nearest odd number
        gray = cv2.GaussianBlur(gray, (ksize, ksize), blur_sigma)

    # 5. Sensor noise
    if noise_sigma > 0:
        noise = np.random.normal(0, noise_sigma, gray.shape).astype(np.float32)
        gray = gray + noise

    # 6. Auto-contrast stretch
    lo = np.percentile(gray, contrast_clip[0])
    hi = np.percentile(gray, contrast_clip[1])
    gray = np.clip((gray - lo) / (hi - lo + 1e-6) * 255.0, 0, 255)

    gray_uint8 = gray.astype(np.uint8)

    # 7. Apply colour tint
    if mode == "nvg_green":
        # Classic green phosphor: map grey → greenish BGR
        bgr = cv2.merge([
            (gray_uint8 * 0.2).clip(0, 255).astype(np.uint8),  # B
            (gray_uint8 * 0.9).clip(0, 255).astype(np.uint8),  # G
            (gray_uint8 * 0.2).clip(0, 255).astype(np.uint8),  # R
        ])
    else:
        # Plain grayscale — convert back to 3-channel BGR for consistency
        bgr = cv2.cvtColor(gray_uint8, cv2.COLOR_GRAY2BGR)

    return bgr


def make_side_by_side(original, simulated, label_orig="Original", label_nv="Simulated NV"):
    """Resize both to the same height and place them side by side with labels."""
    target_h = 900
    def resize_to_h(img, th):
        scale = th / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), th))

    orig_r = resize_to_h(original, target_h)
    nv_r   = resize_to_h(simulated, target_h)

    # Add a small divider
    divider = np.full((target_h, 6, 3), 200, dtype=np.uint8)
    combined = np.hstack([orig_r, divider, nv_r])

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, label_orig, (20, 50), font, 1.5, (50, 200, 50), 3)
    cv2.putText(combined, label_nv,   (orig_r.shape[1] + 20, 50), font, 1.5, (50, 200, 50), 3)

    return combined


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    frame_path = os.path.join(script_dir, FRAME_PATH)

    if not os.path.exists(frame_path):
        print(f"Error: frame not found at {frame_path}")
        return

    original = cv2.imread(frame_path)
    print(f"Loaded: {frame_path}  ({original.shape[1]}×{original.shape[0]})")

    print(f"Mode: {MODE} | noise={NOISE_SIGMA} | hotspot={IR_HOTSPOT_STRENGTH} "
          f"| gamma={GAMMA} | blur={BLUR_SIGMA}")

    simulated = simulate_nv(
        original,
        mode=MODE,
        noise_sigma=NOISE_SIGMA,
        hotspot_strength=IR_HOTSPOT_STRENGTH,
        hotspot_radius=IR_HOTSPOT_RADIUS,
        gamma=GAMMA,
        blur_sigma=BLUR_SIGMA,
        contrast_clip=CONTRAST_CLIP,
    )

    # Save outputs
    out_nv      = os.path.join(script_dir, "nv_sample.jpg")
    out_compare = os.path.join(script_dir, "nv_sample_compare.jpg")

    cv2.imwrite(out_nv, simulated)
    print(f"Saved: {out_nv}")

    compare = make_side_by_side(original, simulated)
    cv2.imwrite(out_compare, compare)
    print(f"Saved: {out_compare}")

    # Show comparison (press any key to close)
    h = compare.shape[0]
    scale = min(1.0, 1080 / h)
    display = cv2.resize(compare, (int(compare.shape[1] * scale), int(h * scale)))
    cv2.imshow("Night Vision Simulation — press any key to close", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
