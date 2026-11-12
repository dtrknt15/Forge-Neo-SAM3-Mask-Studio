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
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Draw a freehand brush (`erase=False`) or eraser (`erase=True`) stroke onto a binary mask."""
    bm = ensure_binary_mask(mask, shape=shape).copy()
    if not points:
        return bm

    r = max(1, int(radius))
    val = 0 if erase else 255
    pts = [(int(x), int(y)) for x, y in points]
    if len(pts) == 1:
        cv2.circle(bm, pts[0], r, val, thickness=-1, lineType=cv2.LINE_8)
    else:
        for i in range(len(pts) - 1):
            cv2.line(bm, pts[i], pts[i + 1], val, thickness=r * 2, lineType=cv2.LINE_8)
        cv2.circle(bm, pts[-1], r, val, thickness=-1, lineType=cv2.LINE_8)
    return bm


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


def render_overlay(
    image: Image.Image,
    final_mask: np.ndarray | None = None,
    *,
    prompt_masks: dict[str, np.ndarray] | None = None,
    color_by_prompt: bool = False,
    click_points_pos: Sequence[tuple[int, int]] | None = None,
    click_points_neg: Sequence[tuple[int, int]] | None = None,
    alpha: float = 0.45,
    view_mode: str = "overlay",
) -> Image.Image:
    """Render preview image according to `view_mode` (`'overlay'`, `'mask_only'`, `'original'`).

    - `'original'`: returns the clean original RGB image for before/after comparison.
    - `'mask_only'`: returns the grayscale mask (`0..255`) as an RGB image.
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
    if click_points_pos or click_points_neg:
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

    return result
