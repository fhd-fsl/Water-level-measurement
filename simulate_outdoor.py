"""
simulate_outdoor.py

Simulates outdoor camera effects on an existing frame:
- Brightness/Exposure changes (auto-exposure, cloud shadows, day/night)
- Glare & Specular Highlights (water surface reflections, lens flare)
- Water Surface Reflections (mirror-like reflection of ruler above water)

Tune the parameters at the top, then run:

    python simulate_outdoor.py

Outputs:
    outdoor_sample.jpg          — the simulated outdoor frame (full resolution)
    outdoor_sample_compare.jpg  — side-by-side comparison with the original
"""

import cv2
import numpy as np
import os
import random

# ---------------------------------------------------------------------------
# Parameters — tweak these
# ---------------------------------------------------------------------------

# Which frame to simulate on
FRAME_PATH = os.path.join("Data", "Extracted_Frames", "frame_0050.jpg")

# Water level ratio (0.0 = top of frame, 1.0 = bottom) — where water surface sits
# Used for glare placement and reflection boundary
WATER_LEVEL_RATIO = 0.65

# ============================================================================
# EXPOSURE / BRIGHTNESS PARAMETERS
# ============================================================================
# Exposure compensation in EV stops (-2 to +2 typical)
# Negative = darker (underexposed), Positive = brighter (overexposed)
EXPOSURE_EV = 0.0

# Simulate auto-exposure adaptation: exposure shifts over time
# Set to 0 to disable; otherwise max EV shift over the sequence
AUTO_EXPOSURE_RANGE_EV = 1.5

# Speed of auto-exposure adaptation (frames to reach target)
AUTO_EXPOSURE_SPEED = 30

# Cloud shadow simulation: random darkening patches
CLOUD_SHADOW_ENABLED = True
CLOUD_SHADOW_STRENGTH = 0.4      # 0-1, how much darker the shadow
CLOUD_SHADOW_COUNT = 3           # number of shadow patches
CLOUD_SHADOW_SCALE = 0.3         # fraction of frame size

# ============================================================================
# GLARE / SPECULAR HIGHLIGHT PARAMETERS
# ============================================================================
# Main glare on water surface (specular reflection of sun/sky)
GLARE_ENABLED = True
GLARE_INTENSITY = 0.6            # 0-1, additive blend strength
GLARE_SIZE_X = 0.15              # fraction of frame width
GLARE_SIZE_Y = 0.08              # fraction of frame height
GLARE_POSITION_X = 0.5           # 0-1, horizontal position (0.5 = center)
GLARE_POSITION_Y = None          # None = use WATER_LEVEL_RATIO, else 0-1

# Lens flare artifacts (ghosting from bright light source)
LENS_FLARE_ENABLED = True
LENS_FLARE_STRENGTH = 0.3        # 0-1, intensity of flare artifacts
LENS_FLARE_COUNT = 5             # number of ghost circles
LENS_FLARE_SOURCE_X = 0.8        # bright light source position (0-1)
LENS_FLARE_SOURCE_Y = 0.2

# ============================================================================
# WATER SURFACE REFLECTION PARAMETERS
# ============================================================================
REFLECTION_ENABLED = True
REFLECTION_STRENGTH = 0.4        # 0-1, alpha blend of reflection
REFLECTION_DISTORTION = 8.0      # wave amplitude in pixels (horizontal offset)
REFLECTION_WAVE_FREQ = 0.02      # waves per pixel (vertical frequency)
REFLECTION_BLUR_SIGMA = 2.0      # Gaussian blur sigma for reflection
REFLECTION_FADE_START = 0.1      # fraction of water height where reflection fades
REFLECTION_FADE_END = 0.4        # fraction of water height where reflection ends

# ============================================================================
# OUTPUT MODE
# ============================================================================
# 'all' = apply all effects, or pick one: 'exposure', 'glare', 'reflection'
SIMULATION_MODE = "all"

# Random seed for reproducibility (None = random each run)
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Simulation Functions
# ---------------------------------------------------------------------------

def set_seed(seed):
    """Set random seeds for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)


def apply_exposure(frame, ev):
    """
    Apply exposure compensation in EV stops.
    EV = log2(multiplier), so multiplier = 2**EV
    """
    if ev == 0:
        return frame

    multiplier = 2.0 ** ev
    frame_float = frame.astype(np.float32) * multiplier
    return np.clip(frame_float, 0, 255).astype(np.uint8)


def apply_auto_exposure(frame, frame_idx, total_frames, max_ev_range, speed):
    """
    Simulate auto-exposure adapting over time.
    Exposure smoothly transitions from -max_ev_range/2 to +max_ev_range/2.
    """
    if max_ev_range == 0 or total_frames <= 1:
        return frame

    # Smooth transition using sigmoid
    progress = frame_idx / (total_frames - 1)
    # S-curve for realistic camera adaptation
    smoothed = 1.0 / (1.0 + np.exp(-10 * (progress - 0.5)))
    ev = (smoothed - 0.5) * max_ev_range

    return apply_exposure(frame, ev)


def apply_cloud_shadows(frame, strength, count, scale):
    """
    Apply random cloud shadow patches (darkened regions).
    """
    if strength == 0 or count == 0:
        return frame

    h, w = frame.shape[:2]
    result = frame.astype(np.float32)

    for _ in range(count):
        # Random shadow center
        cx = random.randint(0, w - 1)
        cy = random.randint(0, h - 1)

        # Shadow size
        radius_x = int(w * scale * random.uniform(0.5, 1.5))
        radius_y = int(h * scale * random.uniform(0.5, 1.5))

        # Create elliptical gradient mask
        ys, xs = np.ogrid[:h, :w]
        dx = (xs - cx) / max(1, radius_x)
        dy = (ys - cy) / max(1, radius_y)
        dist = np.sqrt(dx**2 + dy**2)

        # Soft falloff
        mask = np.exp(-2.0 * dist**2)
        mask = np.clip(mask, 0, 1)

        # Apply darkening
        darken = 1.0 - strength * mask[..., np.newaxis]
        result *= darken

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_glare(frame, intensity, size_x, size_y, pos_x, pos_y, water_ratio):
    """
    Add specular glare highlight on water surface.
    Uses additive blending for bright highlight effect.
    """
    if intensity == 0:
        return frame

    h, w = frame.shape[:2]
    result = frame.astype(np.float32)

    # Glare center
    cx = int(w * pos_x)
    cy = int(h * (pos_y if pos_y is not None else water_ratio))

    # Glare size
    rx = max(1, int(w * size_x))
    ry = max(1, int(h * size_y))

    # Create elliptical glare mask
    ys, xs = np.ogrid[:h, :w]
    dx = (xs - cx) / rx
    dy = (ys - cy) / ry
    dist = np.sqrt(dx**2 + dy**2)

    # Bright core with soft falloff
    glare_mask = np.exp(-4.0 * dist**2)
    glare_mask = np.clip(glare_mask, 0, 1)

    # Additive blend: result = frame + intensity * glare_mask * 255
    glare_value = intensity * 255 * glare_mask[..., np.newaxis]
    result += glare_value

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_lens_flare(frame, strength, count, source_x, source_y):
    """
    Add lens flare ghost artifacts.
    Creates circles/polygons along line from light source to image center.
    """
    if strength == 0 or count == 0:
        return frame

    h, w = frame.shape[:2]
    result = frame.astype(np.float32)

    # Light source position
    sx = int(w * source_x)
    sy = int(h * source_y)

    # Image center
    cx, cy = w // 2, h // 2

    # Vector from source to center
    vx = cx - sx
    vy = cy - sy

    for i in range(count):
        # Position along the line (beyond center, and between source and center)
        t = (i + 1) / (count + 1) * 1.5  # extend past center

        # Ghost position
        gx = int(sx + vx * t)
        gy = int(sy + vy * t)

        # Skip if out of bounds
        if gx < 0 or gx >= w or gy < 0 or gy >= h:
            continue

        # Ghost size (larger further from source)
        radius = max(5, int(min(w, h) * 0.03 * (1 + t * 0.5)))

        # Create circular ghost
        ys, xs = np.ogrid[:h, :w]
        dx = xs - gx
        dy = ys - gy
        dist = np.sqrt(dx**2 + dy**2)

        # Soft circular mask
        ghost_mask = np.exp(-0.5 * (dist / (radius / 1.5))**2)
        ghost_mask = np.clip(ghost_mask, 0, 1)

        # Color tint (warm for lens flare)
        tint = np.array([0.3, 0.6, 1.0], dtype=np.float32)  # BGR - warm orange
        ghost_value = strength * 200 * ghost_mask[..., np.newaxis] * tint
        result += ghost_value

        # Add secondary polygon flare (aperture blades)
        if i % 2 == 0:
            poly_radius = radius * 0.6
            poly_mask = np.exp(-0.5 * (dist / (poly_radius / 1.5))**2)
            poly_mask = np.clip(poly_mask, 0, 1)
            poly_value = strength * 100 * poly_mask[..., np.newaxis] * np.array([0.2, 0.4, 0.8])
            result += poly_value

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_water_reflection(frame, water_y, strength, distortion, wave_freq, blur_sigma, fade_start, fade_end):
    """
    Simulate mirror-like reflection of the scene above water onto the water surface.

    Parameters:
        water_y: pixel y-coordinate of water surface (0 = top)
        strength: reflection opacity (0-1)
        distortion: horizontal wave amplitude in pixels
        wave_freq: vertical wave frequency (waves per pixel)
        blur_sigma: blur applied to reflection
        fade_start: fraction of water height where reflection starts fading
        fade_end: fraction of water height where reflection ends
    """
    if strength == 0 or water_y <= 0 or water_y >= frame.shape[0] - 10:
        return frame

    h, w = frame.shape[:2]
    result = frame.astype(np.float32)

    # Region above water (to be reflected)
    above = frame[0:water_y, :].astype(np.float32)
    above_h = above.shape[0]

    if above_h == 0:
        return frame

    # Flip vertically (mirror)
    reflection = cv2.flip(above, 0)

    # Apply horizontal wave distortion
    ys, xs = np.ogrid[:above_h, :w]
    # Wave offset varies with vertical position
    wave_offset = distortion * np.sin(ys * wave_freq * 2 * np.pi)

    # Remap with distortion — both maps must be full (above_h, w) for cv2.remap
    map_x = (xs + wave_offset).astype(np.float32)                     # already (above_h, w) via broadcast
    map_y = np.broadcast_to(ys, (above_h, w)).astype(np.float32)      # expand (above_h, 1) → (above_h, w)
    reflection = cv2.remap(reflection, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Blur the reflection
    if blur_sigma > 0:
        ksize = int(blur_sigma * 4) | 1
        reflection = cv2.GaussianBlur(reflection, (ksize, ksize), blur_sigma)

    # Water region height
    water_h = h - water_y
    if water_h <= 0:
        return frame

    # Resize reflection to fit water region (may stretch/compress)
    reflection = cv2.resize(reflection, (w, water_h), interpolation=cv2.INTER_LINEAR)

    # Create vertical fade mask (strongest at water surface, fades with depth)
    fade_y = np.linspace(0, 1, water_h)
    fade_mask = np.ones(water_h)

    # Fade zone
    fs = int(water_h * fade_start)
    fe = int(water_h * fade_end)

    if fe > fs:
        fade_mask[:fs] = 1.0
        fade_mask[fs:fe] = 1.0 - (np.arange(fe - fs) / (fe - fs))
        fade_mask[fe:] = 0.0
    else:
        fade_mask[:] = 1.0

    # Apply strength and fade
    alpha = strength * fade_mask[:, np.newaxis, np.newaxis]

    # Alpha blend: result = frame * (1 - alpha) + reflection * alpha
    water_region = result[water_y:h, :]
    blended = water_region * (1 - alpha) + reflection * alpha
    result[water_y:h, :] = blended

    return np.clip(result, 0, 255).astype(np.uint8)


def simulate_outdoor(frame, frame_idx=0, total_frames=1, water_y=None, mode="all"):
    """
    Main simulation pipeline combining all outdoor effects.

    Args:
        frame: Input BGR frame (uint8)
        frame_idx: Current frame index (for temporal effects)
        total_frames: Total frames in sequence (for temporal effects)
        water_y: Water surface y-coordinate in pixels (None = use WATER_LEVEL_RATIO)
        mode: 'all', 'exposure', 'glare', 'reflection'

    Returns:
        Simulated frame (uint8 BGR)
    """
    h, w = frame.shape[:2]

    # Determine water_y if not provided
    if water_y is None:
        water_y = int(h * WATER_LEVEL_RATIO)

    result = frame.copy()

    # 1. Exposure effects
    if mode in ("all", "exposure"):
        # Base exposure
        result = apply_exposure(result, EXPOSURE_EV)

        # Auto-exposure adaptation
        if AUTO_EXPOSURE_RANGE_EV > 0:
            result = apply_auto_exposure(result, frame_idx, total_frames,
                                         AUTO_EXPOSURE_RANGE_EV, AUTO_EXPOSURE_SPEED)

        # Cloud shadows
        if CLOUD_SHADOW_ENABLED:
            result = apply_cloud_shadows(result, CLOUD_SHADOW_STRENGTH,
                                         CLOUD_SHADOW_COUNT, CLOUD_SHADOW_SCALE)

    # 2. Glare effects
    if mode in ("all", "glare"):
        if GLARE_ENABLED:
            result = apply_glare(result, GLARE_INTENSITY, GLARE_SIZE_X, GLARE_SIZE_Y,
                                GLARE_POSITION_X, GLARE_POSITION_Y, WATER_LEVEL_RATIO)

        if LENS_FLARE_ENABLED:
            result = apply_lens_flare(result, LENS_FLARE_STRENGTH, LENS_FLARE_COUNT,
                                     LENS_FLARE_SOURCE_X, LENS_FLARE_SOURCE_Y)

    # 3. Water reflection
    if mode in ("all", "reflection"):
        if REFLECTION_ENABLED:
            result = apply_water_reflection(result, water_y, REFLECTION_STRENGTH,
                                           REFLECTION_DISTORTION, REFLECTION_WAVE_FREQ,
                                           REFLECTION_BLUR_SIGMA, REFLECTION_FADE_START,
                                           REFLECTION_FADE_END)

    return result


def make_side_by_side(original, simulated, label_orig="Original", label_sim="Simulated Outdoor"):
    """Resize both to the same height and place them side by side with labels."""
    target_h = 900
    def resize_to_h(img, th):
        scale = th / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), th))

    orig_r = resize_to_h(original, target_h)
    sim_r  = resize_to_h(simulated, target_h)

    divider = np.full((target_h, 6, 3), 200, dtype=np.uint8)
    combined = np.hstack([orig_r, divider, sim_r])

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, label_orig, (20, 50), font, 1.5, (50, 200, 50), 3)
    cv2.putText(combined, label_sim,   (orig_r.shape[1] + 20, 50), font, 1.5, (50, 200, 50), 3)

    return combined


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    frame_path = os.path.join(script_dir, FRAME_PATH)

    if not os.path.exists(frame_path):
        print(f"Error: frame not found at {frame_path}")
        return

    original = cv2.imread(frame_path)
    print(f"Loaded: {frame_path}  ({original.shape[1]}×{original.shape[0]})")
    print(f"Mode: {SIMULATION_MODE} | exposure_ev={EXPOSURE_EV} | auto_range={AUTO_EXPOSURE_RANGE_EV}EV")
    print(f"  glare={GLARE_ENABLED}({GLARE_INTENSITY}) | flare={LENS_FLARE_ENABLED}({LENS_FLARE_STRENGTH})")
    print(f"  reflection={REFLECTION_ENABLED}({REFLECTION_STRENGTH}) | distort={REFLECTION_DISTORTION}")

    set_seed(RANDOM_SEED)

    # For demo, simulate a mid-sequence frame to show auto-exposure
    demo_frame_idx = 50
    demo_total = 100

    simulated = simulate_outdoor(
        original,
        frame_idx=demo_frame_idx,
        total_frames=demo_total,
        mode=SIMULATION_MODE
    )

    # Save outputs
    out_sim      = os.path.join(script_dir, "outdoor_sample.jpg")
    out_compare  = os.path.join(script_dir, "outdoor_sample_compare.jpg")

    cv2.imwrite(out_sim, simulated)
    print(f"Saved: {out_sim}")

    compare = make_side_by_side(original, simulated)
    cv2.imwrite(out_compare, compare)
    print(f"Saved: {out_compare}")

    # Show comparison
    h = compare.shape[0]
    scale = min(1.0, 1080 / h)
    display = cv2.resize(compare, (int(compare.shape[1] * scale), int(h * scale)))
    cv2.imshow("Outdoor Simulation — press any key to close", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()