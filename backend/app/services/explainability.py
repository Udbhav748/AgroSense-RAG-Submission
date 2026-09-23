"""Leaf Infection Explainability & Visual Saliency Heatmap Service.

Performs computer vision color segmentation in HSV and LAB color spaces
to isolate foliar tissue from background, detect necrotic lesion centers,
identify chlorotic yellowing margins, compute high-frequency edge gradients,
and generate a color-coded saliency heatmap overlay.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

# Try importing cv2 if installed; fallback to pure NumPy/PIL if unavailable
try:
    import cv2  # type: ignore[import-not-found,import-untyped]
    _HAS_CV2 = True
except ImportError:
    cv2 = None
    _HAS_CV2 = False


def _rgb_to_hsv_numpy(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB float image [0, 1] to HSV [H in 0..180, S in 0..255, V in 0..255]."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c

    # Hue calculation
    h = np.zeros_like(max_c)
    mask = delta > 1e-5

    # r is max
    r_max = mask & (max_c == r)
    h[r_max] = (60.0 * ((g[r_max] - b[r_max]) / delta[r_max]) + 360.0) % 360.0

    # g is max
    g_max = mask & (max_c == g) & ~r_max
    h[g_max] = (60.0 * ((b[g_max] - r[g_max]) / delta[g_max]) + 120.0) % 360.0

    # b is max
    b_max = mask & (max_c == b) & ~r_max & ~g_max
    h[b_max] = (60.0 * ((r[b_max] - g[b_max]) / delta[b_max]) + 240.0) % 360.0

    # Scale H to [0, 180] for OpenCV HSV parity
    h = (h / 2.0).astype(np.uint8)
    s = np.zeros_like(max_c)
    s[max_c > 1e-5] = (delta[max_c > 1e-5] / max_c[max_c > 1e-5]) * 255.0
    s = s.astype(np.uint8)
    v = (max_c * 255.0).astype(np.uint8)

    return np.stack([h, s, v], axis=-1)


def _rgb_to_lab_numpy(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB image [0..255] to approximate LAB representation."""
    # Normalized RGB
    r = rgb[..., 0] / 255.0
    g = rgb[..., 1] / 255.0
    b = rgb[..., 2] / 255.0

    # Linearize RGB (sRGB -> linear)
    def linearize(c: np.ndarray) -> np.ndarray:
        return np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)

    r_lin = linearize(r)
    g_lin = linearize(g)
    b_lin = linearize(b)

    # Convert to XYZ (D65 illuminant)
    x = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
    y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
    z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041

    # Normalize by D65 reference white
    x_n = x / 0.95047
    y_n = y / 1.00000
    z_n = z / 1.08883

    def f(t: np.ndarray) -> np.ndarray:
        delta = 6.0 / 29.0
        return np.where(t > delta**3, t ** (1.0 / 3.0), (t / (3.0 * delta**2)) + (4.0 / 29.0))

    fx = f(x_n)
    fy = f(y_n)
    fz = f(z_n)

    # LAB values: L in [0..255], a in [0..255], b in [0..255]
    l_val = np.clip((116.0 * fy - 16.0) * (255.0 / 100.0), 0, 255).astype(np.uint8)
    a_val = np.clip((500.0 * (fx - fy)) + 128.0, 0, 255).astype(np.uint8)
    b_val = np.clip((200.0 * (fy - fz)) + 128.0, 0, 255).astype(np.uint8)

    return np.stack([l_val, a_val, b_val], axis=-1)


def _count_connected_components(binary_mask: np.ndarray, min_pixel_size: int = 10) -> int:
    """Count distinct connected lesion blobs, discarding small noise clusters."""
    if not np.any(binary_mask):
        return 0

    if _HAS_CV2 and cv2 is not None:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_mask.astype(np.uint8), connectivity=8
        )
        count = 0
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_pixel_size:
                count += 1
        return count

    # Pure NumPy 8-connectivity BFS labeling fallback
    h, w = binary_mask.shape
    visited = np.zeros((h, w), dtype=bool)
    count = 0

    for y in range(h):
        for x in range(w):
            if binary_mask[y, x] and not visited[y, x]:
                # BFS to explore component
                queue = [(y, x)]
                visited[y, x] = True
                blob_size = 0

                while queue:
                    cy, cx = queue.pop(0)
                    blob_size += 1

                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = cy + dy, cx + dx
                            if (
                                0 <= ny < h
                                and 0 <= nx < w
                                and binary_mask[ny, nx]
                                and not visited[ny, nx]
                            ):
                                visited[ny, nx] = True
                                queue.append((ny, nx))

                if blob_size >= min_pixel_size:
                    count += 1

    return count


def _compute_edge_gradients(gray_img: np.ndarray) -> np.ndarray:
    """Compute high-frequency edge gradients to emphasize irregular lesion spot margins."""
    if _HAS_CV2 and cv2 is not None:
        sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        norm_grad = np.clip((grad_mag / (grad_mag.max() + 1e-6)) * 255.0, 0, 255).astype(np.uint8)
        return norm_grad

    # PIL-based edge enhancement fallback
    pil_gray = Image.fromarray(gray_img)
    edges = pil_gray.filter(ImageFilter.FIND_EDGES)
    return np.array(edges, dtype=np.uint8)


def generate_leaf_saliency(image_bytes: bytes) -> dict[str, Any]:
    """Compute lesion detection mask and color-coded heatmap overlay from leaf image bytes.

    Segments foliar tissue using HSV and LAB color metrics, classifies healthy foliar
    tissue vs chlorosis (yellowing) margins vs necrotic lesion centers, and produces
    a blended saliency visualization.

    Returns:
        heatmap_base64: PNG base64 string of the color-coded overlay.
        infected_area_percentage: Estimated percentage of leaf surface infected (0.0% to 100.0%).
        lesion_count: Estimated count of distinct lesion spots.
        severity_level: "Mild" (<10%), "Moderate" (10-30%), "Severe" (>30%).
    """
    default_fallback: dict[str, Any] = {
        "heatmap_base64": None,
        "infected_area_percentage": 0.0,
        "lesion_count": 0,
        "severity_level": "Mild",
    }

    if not image_bytes or len(image_bytes) == 0:
        return default_fallback

    try:
        # 1. Load Image
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_w, orig_h = pil_img.size

        if orig_w == 0 or orig_h == 0:
            return default_fallback

        # Resize for fast, consistent processing if large
        max_dim = 640
        if max(orig_w, orig_h) > max_dim:
            scale = max_dim / float(max(orig_w, orig_h))
            proc_w = max(1, int(orig_w * scale))
            proc_h = max(1, int(orig_h * scale))
            pil_img = pil_img.resize((proc_w, proc_h), Image.Resampling.BILINEAR)

        img_rgb = np.array(pil_img, dtype=np.uint8)
        h, w, _ = img_rgb.shape

        # 2. Color Spaces Conversion
        if _HAS_CV2 and cv2 is not None:
            hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        else:
            hsv = _rgb_to_hsv_numpy(img_rgb / 255.0)
            lab = _rgb_to_lab_numpy(img_rgb)
            gray = np.array(pil_img.convert("L"), dtype=np.uint8)

        hue = hsv[..., 0]
        sat = hsv[..., 1]
        val = hsv[..., 2]
        lab_l = lab[..., 0]
        lab_b = lab[..., 2]

        # 3. Leaf Folliage Mask Segmentation (isolate plant tissue from neutral/white/black background)
        # Healthy/diseased leaf hue in OpenCV range [10..95], plus reasonable saturation/luminance
        is_plant_hue = (hue >= 10) & (hue <= 95)
        has_color = (sat >= 25) & (val >= 25) & (val <= 248)
        not_pure_gray = (np.ptp(img_rgb, axis=-1) >= 15)
        leaf_mask = (is_plant_hue | (has_color & not_pure_gray)) & (val >= 20) & (val <= 250)

        total_leaf_pixels = int(np.count_nonzero(leaf_mask))
        if total_leaf_pixels < (h * w * 0.05):
            # If segmented leaf area is too small, treat the entire canvas as leaf region
            leaf_mask = np.ones((h, w), dtype=bool)
            total_leaf_pixels = h * w

        # 4. Disease & Lesion Segmentation
        r_ch = img_rgb[..., 0].astype(int)
        g_ch = img_rgb[..., 1].astype(int)
        b_ch = img_rgb[..., 2].astype(int)
        is_healthy_green = (g_ch > r_ch + 15) & (g_ch > b_ch + 15) & (hue >= 32) & (hue <= 85)

        # High-frequency edge gradient for lesion margins
        edge_grad = _compute_edge_gradients(gray)
        high_grad_mask = (edge_grad >= 55) & leaf_mask & ~is_healthy_green

        # Necrotic brown/black lesions (dark centers, brown/reddish hue or low luminance)
        is_dark_lesion = (val <= 45) & leaf_mask
        is_brown_lesion = (hue <= 22) & (sat >= 30) & (val <= 140) & (r_ch >= g_ch - 10) & leaf_mask
        necrotic_mask = (is_dark_lesion | is_brown_lesion | (high_grad_mask & (val <= 100))) & leaf_mask & ~is_healthy_green

        # Chlorosis yellowing margins (yellow-amber hue [18..35], moderate/high brightness)
        is_yellow_hue = (hue >= 18) & (hue <= 35) & (sat >= 45) & (val >= 70) & (r_ch > b_ch + 30)
        is_lab_yellow = (lab_b >= 165) & (lab_l >= 110)
        chlorosis_mask = (is_yellow_hue | is_lab_yellow) & leaf_mask & ~necrotic_mask & ~is_healthy_green

        # Combined infected foliar mask
        infected_mask = (necrotic_mask | chlorosis_mask) & leaf_mask

        # 5. Calculate Metrics
        infected_pixels = int(np.count_nonzero(infected_mask))
        infected_percentage = round((float(infected_pixels) / float(max(total_leaf_pixels, 1))) * 100.0, 1)
        infected_percentage = min(max(infected_percentage, 0.0), 100.0)

        lesion_count = _count_connected_components(necrotic_mask, min_pixel_size=8)
        if lesion_count == 0 and infected_percentage >= 5.0:
            lesion_count = _count_connected_components(infected_mask, min_pixel_size=12)

        if infected_percentage < 10.0:
            severity_level = "Mild"
        elif infected_percentage <= 30.0:
            severity_level = "Moderate"
        else:
            severity_level = "Severe"

        # 6. Generate Color-Coded Heatmap Overlay
        # Red/Orange for necrotic/active lesion centers ([240, 45, 30])
        # Yellow for chlorotic margins ([245, 190, 25])
        # Emerald Green for healthy foliar tissue ([34, 197, 94])
        overlay_color = np.copy(img_rgb).astype(np.float32)

        # Apply Healthy Green Tint on uninfected leaf tissue
        healthy_mask = leaf_mask & ~infected_mask
        overlay_color[healthy_mask] = (
            overlay_color[healthy_mask] * 0.45 + np.array([34, 197, 94], dtype=np.float32) * 0.55
        )

        # Apply Yellow Tint on Chlorotic Margins
        overlay_color[chlorosis_mask] = (
            overlay_color[chlorosis_mask] * 0.30 + np.array([250, 204, 21], dtype=np.float32) * 0.70
        )

        # Apply Vivid Crimson-Red on Active Lesion Centers
        overlay_color[necrotic_mask] = (
            overlay_color[necrotic_mask] * 0.20 + np.array([239, 68, 68], dtype=np.float32) * 0.80
        )

        # Dim background outside the leaf mask for high-contrast saliency focus
        bg_mask = ~leaf_mask
        overlay_color[bg_mask] = overlay_color[bg_mask] * 0.35

        overlay_uint8 = np.clip(overlay_color, 0, 255).astype(np.uint8)

        # If OpenCV is available, draw subtle glowing boundary contours around lesion spots
        if _HAS_CV2 and cv2 is not None:
            contours, _ = cv2.findContours(
                necrotic_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay_uint8, contours, -1, (255, 255, 255), 1)

        # Encode heatmap to PNG Base64
        overlay_pil = Image.fromarray(overlay_uint8)
        buffer = io.BytesIO()
        overlay_pil.save(buffer, format="PNG", optimize=True)
        heatmap_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "heatmap_base64": heatmap_base64,
            "infected_area_percentage": infected_percentage,
            "lesion_count": lesion_count,
            "severity_level": severity_level,
        }

    except Exception as exc:
        logger.warning(
            "leaf_saliency_generation_failed",
            extra={"extra_fields": {"error": str(exc)}},
        )
        return default_fallback
