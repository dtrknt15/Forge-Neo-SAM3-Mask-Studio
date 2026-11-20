"""Mask set operations, morphological grow/shrink, and preview overlay rendering.

Single source of truth:
All masks are represented internally as 2D `np.ndarray` of dtype `np.uint8`
with binary values `0` (background) and `255` (masked foreground).
"""

from __future__ import annotations

from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

# Distinct RGB palette for per-prompt color-coded debug/comparison overlay (spec §4.2)
PROMPT_PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 75, 75),    # Red
    (50, 180, 255),   # Sky Blue
    (80, 220, 100),   # Green
    (255, 180, 40),   # Orange
    (190, 90, 255),   # Purple
    (255, 105, 180),  # Pink
    (40, 220, 210),   # Cyan
    (230, 230, 50),   # Yellow
)

DEFAULT_FINAL_COLOR: tuple[int, int, int] = (0, 200, 255)


def empty_mask(shape: tuple[int, int]) -> np.ndarray:
    """Return an all-zero uint8 2D mask of shape (H, W)."""
    h, w = int(shape[0]), int(shape[1])
    return np.zeros((h, w), dtype=np.uint8)


def ensure_binary_mask(
    mask: np.ndarray | Image.Image | None,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Convert any 2D/3D array or PIL Image into a strict 2D uint8 (0 or 255) mask.

    If `shape` is given as (H, W) and the mask size differs, resizes with NEAREST.
    """
    if mask is None:
        if shape is None:
            raise ValueError("shape must be provided when mask is None")
        return empty_mask(shape)

    if isinstance(mask, Image.Image):
        arr = np.asarray(mask.convert("L"))
    else:
        arr = np.asarray(mask)

    if arr.ndim == 3:
        arr = arr[..., 0]
    elif arr.ndim > 3:
        arr = arr.reshape(arr.shape[-2], arr.shape[-1])

    if arr.dtype == np.bool_:
        binary = np.where(arr, 255, 0).astype(np.uint8)
    elif np.issubdtype(arr.dtype, np.floating):
        binary = np.where(arr > 0.5, 255, 0).astype(np.uint8)
    else:
        binary = np.where(arr > 127, 255, 0).astype(np.uint8)

    if shape is not None:
        th, tw = int(shape[0]), int(shape[1])
        if binary.shape != (th, tw):
            binary = cv2.resize(binary, (tw, th), interpolation=cv2.INTER_NEAREST)
            binary = np.where(binary > 127, 255, 0).astype(np.uint8)

    return binary


def ensure_grayscale_mask(
    mask: np.ndarray | Image.Image | None,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Convert any 2D/3D array or PIL Image into a 2D uint8 (0..255) soft mask.

    Preserves intermediate grayscale values (e.g. from feathering).
    If `shape` is given as (H, W) and the mask size differs, resizes with LINEAR.
    """
    if mask is None:
        if shape is None:
            raise ValueError("shape must be provided when mask is None")
        return empty_mask(shape)

    if isinstance(mask, Image.Image):
        arr = np.asarray(mask.convert("L"))
    else:
        arr = np.asarray(mask)

    if arr.ndim == 3:
        arr = arr[..., 0]
    elif arr.ndim > 3:
        arr = arr.reshape(arr.shape[-2], arr.shape[-1])

    if arr.dtype == np.bool_:
        gray = np.where(arr, 255, 0).astype(np.uint8)
    elif np.issubdtype(arr.dtype, np.floating):
        max_val = float(arr.max()) if arr.size else 0.0
        scale = 255.0 if max_val <= 1.0 else 1.0
        gray = np.clip(np.round(arr * scale), 0, 255).astype(np.uint8)
    else:
        gray = np.clip(arr, 0, 255).astype(np.uint8)

    if shape is not None:
        th, tw = int(shape[0]), int(shape[1])
        if gray.shape != (th, tw):
            gray = cv2.resize(gray, (tw, th), interpolation=cv2.INTER_LINEAR)
            gray = np.clip(gray, 0, 255).astype(np.uint8)

    return gray


def union_masks(
    masks: Iterable[np.ndarray | Image.Image],
    shape: tuple[int, int],
) -> np.ndarray:
    """Compute bitwise OR of multiple masks as a 2D uint8 (0/255) array of `shape` (H, W)."""
    acc = empty_mask(shape)
    for m in masks:
        if m is None:
            continue
        bm = ensure_binary_mask(m, shape=shape)
        acc = np.bitwise_or(acc, bm)
    return acc


def subtract_mask(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    """Compute set difference `positive - negative` on uint8 (0/255) masks."""
    pos = ensure_binary_mask(positive)
    neg = ensure_binary_mask(negative, shape=pos.shape)
    return np.bitwise_and(pos, np.bitwise_not(neg))


def invert_mask(
    mask: np.ndarray | Image.Image | None,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Invert a binary uint8 (0/255) mask (`0 -> 255`, `255 -> 0`)."""
    bm = ensure_binary_mask(mask, shape=shape)
    return np.bitwise_not(bm)


def apply_grow_shrink(mask: np.ndarray, grow_px: int) -> np.ndarray:
    """Dilate (`grow_px > 0`) or erode (`grow_px < 0`) a binary uint8 (0/255) mask.

    Uses an elliptical structuring element of size `2 * |grow_px| + 1` (spec §5.4).
    """
    bm = ensure_binary_mask(mask)
    px = int(grow_px)
    if px == 0 or not bm.any():
        return bm

    ksize = 2 * abs(px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    if px > 0:
        out = cv2.dilate(bm, kernel, iterations=1)
    else:
        out = cv2.erode(bm, kernel, iterations=1)
    return np.where(out > 127, 255, 0).astype(np.uint8)


def apply_feather(mask: np.ndarray, feather_px: int) -> np.ndarray:
    """Apply Gaussian feathering (`feather_px > 0`) to produce a smooth 0..255 uint8 mask."""
    gm = ensure_grayscale_mask(mask)
    px = int(feather_px)
    if px <= 0 or not gm.any():
        return gm

    sigma = max(0.5, float(px) * 0.5)
    blurred = cv2.GaussianBlur(gm, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(blurred, 0, 255).astype(np.uint8)


def apply_guided_refine(
    image: Image.Image | np.ndarray,
    mask: np.ndarray,
    radius: int = 8,
    eps: float = 1e-2,
) -> np.ndarray:
    """Edge-aware mask refinement using a fast box-filter Guided Filter against `image`."""
    r = int(radius)
    gm = ensure_grayscale_mask(mask)
    if r <= 0 or not gm.any():
        return gm

    if isinstance(image, Image.Image):
        guide_u8 = np.asarray(image.convert("L"), dtype=np.uint8)
    else:
        arr = np.asarray(image)
        if arr.ndim == 3:
            guide_u8 = cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2GRAY)
        else:
            guide_u8 = arr.astype(np.uint8)

    h, w = gm.shape
    if guide_u8.shape != (h, w):
        guide_u8 = cv2.resize(guide_u8, (w, h), interpolation=cv2.INTER_LINEAR)

    I = guide_u8.astype(np.float32) / 255.0
    p = gm.astype(np.float32) / 255.0
    ksize = (2 * r + 1, 2 * r + 1)

    mean_I = cv2.boxFilter(I, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(p, cv2.CV_32F, ksize)
    corr_I = cv2.boxFilter(I * I, cv2.CV_32F, ksize)
    corr_Ip = cv2.boxFilter(I * p, cv2.CV_32F, ksize)

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + float(eps))
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)

    q = np.clip(mean_a * I + mean_b, 0.0, 1.0)
    return np.clip(np.round(q * 255.0), 0, 255).astype(np.uint8)


def apply_edge_smooth(mask: np.ndarray, smooth_px: int = 0) -> np.ndarray:
    """Smooth jagged binary mask contours using Gaussian blur + midpoint re-thresholding."""
    bm = ensure_binary_mask(mask)
    px = int(smooth_px)
    if px <= 0 or not bm.any():
        return bm
    sigma = max(0.5, float(px) * 0.6)
    blurred = cv2.GaussianBlur(bm, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.where(blurred > 127, 255, 0).astype(np.uint8)


def fill_holes(
    mask: np.ndarray | Image.Image | None,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Fill enclosed interior holes in a binary uint8 (0/255) mask using border-padded flood fill."""
    bm = ensure_binary_mask(mask, shape=shape)
    if not bm.any():
        return bm

    # Pad by 1px with 0 so foreground touching the image edge does not block external flood fill.
    padded = cv2.copyMakeBorder(bm, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 255)
    flood_crop = flood[1:-1, 1:-1]
    holes = cv2.bitwise_not(flood_crop)
    return np.bitwise_or(bm, holes)


def remove_small_islands(
    mask: np.ndarray | Image.Image | None,
    min_area: int = 64,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Remove disconnected foreground components smaller than `min_area` pixels."""
    bm = ensure_binary_mask(mask, shape=shape)
    thr = int(min_area)
    if thr <= 0 or not bm.any():
        return bm

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bm, connectivity=8)
    if num_labels <= 1:
        return bm

    keep = np.zeros(num_labels, dtype=bool)
    for label_idx in range(1, num_labels):
        if int(stats[label_idx, cv2.CC_STAT_AREA]) >= thr:
            keep[label_idx] = True

    out = np.where(keep[labels], 255, 0).astype(np.uint8)
    return out


def apply_brush_stroke(
    mask: np.ndarray | Image.Image | None,
    points: Sequence[tuple[int, int]],
    radius: int = 12,
    *,
    erase: bool = False,
    hardness: float = 1.0,
    opacity: float = 1.0,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Draw a freehand brush (`erase=False`) or eraser (`erase=True`) stroke onto a mask.

    When `hardness == 1.0` and `opacity == 1.0`, performs a strict binary 0/255 stroke.
    Lower `hardness` or `opacity` produces soft-edged grayscale alpha strokes.
    """
    hard = float(max(0.05, min(1.0, hardness)))
    opac = float(max(0.05, min(1.0, opacity)))
    is_hard_opaque = hard >= 0.99 and opac >= 0.99

    base = (
        ensure_binary_mask(mask, shape=shape).copy()
        if is_hard_opaque
        else ensure_grayscale_mask(mask, shape=shape).copy()
    )
    if not points:
        return base

    r = max(1, int(radius))
    pts = [(int(x), int(y)) for x, y in points]

    if is_hard_opaque:
        val = 0 if erase else 255
        if len(pts) == 1:
            cv2.circle(base, pts[0], r, val, thickness=-1, lineType=cv2.LINE_8)
        else:
            for i in range(len(pts) - 1):
                cv2.line(base, pts[i], pts[i + 1], val, thickness=r * 2, lineType=cv2.LINE_8)
            cv2.circle(base, pts[-1], r, val, thickness=-1, lineType=cv2.LINE_8)
        return base

    stroke = np.zeros_like(base, dtype=np.uint8)
    if len(pts) == 1:
        cv2.circle(stroke, pts[0], r, 255, thickness=-1, lineType=cv2.LINE_AA)
    else:
        for i in range(len(pts) - 1):
            cv2.line(stroke, pts[i], pts[i + 1], 255, thickness=r * 2, lineType=cv2.LINE_AA)
        cv2.circle(stroke, pts[-1], r, 255, thickness=-1, lineType=cv2.LINE_AA)

    stroke_f = stroke.astype(np.float32) / 255.0
    if hard < 0.99:
        sigma = max(0.5, (1.0 - hard) * float(r) * 0.45)
        stroke_f = cv2.GaussianBlur(stroke_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
        max_s = float(stroke_f.max())
        if max_s > 1e-6:
            stroke_f = stroke_f / max_s
    alpha_f = np.clip(stroke_f * opac, 0.0, 1.0)

    base_f = base.astype(np.float32)
    if erase:
        out_f = base_f * (1.0 - alpha_f)
    else:
        out_f = base_f + (255.0 - base_f) * alpha_f
    return np.clip(np.round(out_f), 0, 255).astype(np.uint8)


def compute_final_mask(
    positive_masks: Iterable[np.ndarray | Image.Image],
    negative_masks: Iterable[np.ndarray | Image.Image],
    shape: tuple[int, int],
    grow_px: int = 0,
) -> np.ndarray:
    """Unified set-operation mask synthesis (spec §3.2 & §5.4):

    positive_set = OR(positive_masks)
    negative_set = OR(negative_masks)
    final_mask   = grow_shrink(positive_set - negative_set, grow_px)
    """
    pos_set = union_masks(positive_masks, shape)
    neg_set = union_masks(negative_masks, shape)
    diff = subtract_mask(pos_set, neg_set)
    return apply_grow_shrink(diff, grow_px)


def scale_click_coords(
    click_x: float,
    click_y: float,
    display_size: tuple[int, int] | None,
    orig_size: tuple[int, int],
) -> tuple[int, int]:
    """Scale click coordinates from preview display size (W, H) to original image size (W, H).

    Clamps coordinates to valid `[0, orig_w - 1]` and `[0, orig_h - 1]`.
    """
    orig_w, orig_h = int(orig_size[0]), int(orig_size[1])
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError(f"Invalid orig_size: {orig_size}")

    if display_size is not None and display_size[0] > 0 and display_size[1] > 0:
        disp_w, disp_h = float(display_size[0]), float(display_size[1])
        sx = orig_w / disp_w
        sy = orig_h / disp_h
        x = int(round(click_x * sx))
        y = int(round(click_y * sy))
    else:
        x = int(round(click_x))
        y = int(round(click_y))

    x = max(0, min(orig_w - 1, x))
    y = max(0, min(orig_h - 1, y))
    return x, y


def _make_checkerboard(h: int, w: int, tile: int = 16) -> np.ndarray:
    """Create a subtle light/dark gray checkerboard RGB float32 array of shape (h, w, 3)."""
    ys = (np.arange(h) // tile)[:, None]
    xs = (np.arange(w) // tile)[None, :]
    dark = ((ys + xs) & 1).astype(np.float32)
    bg = np.where(dark[..., None] > 0, 175.0, 225.0).astype(np.float32)
    return np.repeat(bg, 3, axis=2)


def render_overlay(
    image: Image.Image,
    final_mask: np.ndarray | None = None,
    *,
    prompt_masks: dict[str, np.ndarray] | None = None,
    color_by_prompt: bool = False,
    click_points_pos: Sequence[tuple[int, int]] | None = None,
    click_points_neg: Sequence[tuple[int, int]] | None = None,
    refine_points: Sequence[tuple[int, int]] | None = None,
    refine_labels: Sequence[int] | None = None,
    alpha: float = 0.45,
    view_mode: str = "overlay",
) -> Image.Image:
    """Render preview image according to `view_mode` (`'overlay'`, `'mask_only'`, `'cutout'`, `'outline'`, `'original'`).

    - `'original'`: returns the clean original RGB image for before/after comparison.
    - `'mask_only'`: returns the grayscale mask (`0..255`) as an RGB image.
    - `'cutout'`: alpha-blends `image` over a checkerboard background using `final_mask`.
    - `'outline'`: draws crisp mask contour outlines over `image`.
    - `'overlay'` (default): semi-transparent mask overlay onto `image`, with optional
      per-prompt color coding and positive/negative click markers.
    """
    base_rgb = image.convert("RGB")
    w, h = base_rgb.size
    mode = (view_mode or "overlay").lower()

    if mode == "original":
        return base_rgb

    if mode == "mask_only":
        gm = ensure_grayscale_mask(final_mask, shape=(h, w)) if final_mask is not None else empty_mask((h, w))
        return Image.fromarray(gm, mode="L").convert("RGB")

    base_arr = np.asarray(base_rgb, dtype=np.float32)

    if mode == "cutout":
        gm = ensure_grayscale_mask(final_mask, shape=(h, w)) if final_mask is not None else empty_mask((h, w))
        checker = _make_checkerboard(h, w, tile=max(8, min(24, min(w, h) // 32)))
        alpha_f = gm.astype(np.float32)[..., None] / 255.0
        comp = base_arr * alpha_f + checker * (1.0 - alpha_f)
        result = Image.fromarray(np.clip(comp, 0, 255).astype(np.uint8), mode="RGB")
    elif mode == "outline":
        out_bgr = np.asarray(base_rgb, dtype=np.uint8).copy()
        if final_mask is not None:
            bm = ensure_binary_mask(final_mask, shape=(h, w))
            if bm.any():
                contours, _ = cv2.findContours(bm, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                thickness = max(2, int(round(min(w, h) * 0.004)))
                cv2.drawContours(out_bgr, contours, -1, (0, 0, 0), thickness=thickness + 2, lineType=cv2.LINE_AA)
                cv2.drawContours(out_bgr, contours, -1, DEFAULT_FINAL_COLOR, thickness=thickness, lineType=cv2.LINE_AA)
        result = Image.fromarray(out_bgr, mode="RGB")
    else:
        out_arr = base_arr.copy()
        alpha_clamped = float(max(0.0, min(1.0, alpha)))

        if color_by_prompt and prompt_masks:
            for idx, (_key, pmask) in enumerate(prompt_masks.items()):
                bm = ensure_binary_mask(pmask, shape=(h, w)) > 0
                if not bm.any():
                    continue
                color = np.array(PROMPT_PALETTE[idx % len(PROMPT_PALETTE)], dtype=np.float32)
                out_arr[bm] = out_arr[bm] * (1.0 - alpha_clamped) + color * alpha_clamped
        elif final_mask is not None:
            gm = ensure_grayscale_mask(final_mask, shape=(h, w))
            if gm.any():
                color = np.array(DEFAULT_FINAL_COLOR, dtype=np.float32)
                weight = (gm.astype(np.float32)[..., None] / 255.0) * alpha_clamped
                out_arr = out_arr * (1.0 - weight) + color * weight

        result = Image.fromarray(np.clip(out_arr, 0, 255).astype(np.uint8), mode="RGB")

    # Draw click markers if present
    if click_points_pos or click_points_neg or refine_points:
        draw = ImageDraw.Draw(result)
        radius = max(4, int(round(min(w, h) * 0.008)))
        for x, y in click_points_pos or ():
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=(40, 220, 80),
                outline=(255, 255, 255),
                width=2,
            )
        for x, y in click_points_neg or ():
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=(255, 60, 60),
                outline=(255, 255, 255),
                width=2,
            )
        if refine_points:
            lbls = list(refine_labels) if refine_labels is not None else [1] * len(refine_points)
            r_ref = radius + 2
            for (x, y), lbl in zip(refine_points, lbls):
                fill_col = (0, 255, 160) if int(lbl) == 1 else (255, 80, 120)
                ring_col = (255, 255, 0)
                draw.ellipse(
                    (x - r_ref, y - r_ref, x + r_ref, y + r_ref),
                    fill=fill_col,
                    outline=ring_col,
                    width=2,
                )

    return result
