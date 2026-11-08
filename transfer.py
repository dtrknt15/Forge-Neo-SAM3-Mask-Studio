"""Inpaint upload transfer and masked RGBA PNG export (spec §5.5 & §5.6)."""

from __future__ import annotations

import base64
import io
import json
import tempfile
import time
from itertools import count
from pathlib import Path

import numpy as np
from PIL import Image

import mask_ops
from i18n_helper import t

_COUNTER = count()


def _pil_to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _timestamped_filename(prefix: str = "sam3_mask_studio") -> str:
    now = time.time()
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
    ms = int(now * 1000) % 1000
    seq = next(_COUNTER) % 1000
    return f"{prefix}_{stamp}_{ms:03d}_{seq:03d}.png"


def build_inpaint_pair(
    original_image: Image.Image,
    final_mask: np.ndarray,
) -> tuple[Image.Image, Image.Image]:
    """Build `(init_rgb, mask_rgba)` for Forge Neo's `img_inpaint_base` and `img_inpaint_mask`.

    `mask_rgba` carries the binary uint8 (0/255) mask across R, G, B and alpha so
    both Forge Neo's luminance reader and alpha-aware readers interpret white (255)
    as the inpaint region accurately.
    """
    rgb = original_image.convert("RGB")
    w, h = rgb.size
    bm = mask_ops.ensure_binary_mask(final_mask, shape=(h, w))
    mask_l = Image.fromarray(bm, mode="L")
    alpha_full = Image.new("L", (w, h), 255)
    mask_rgba = Image.merge("RGBA", (mask_l, mask_l, mask_l, alpha_full))
    return rgb, mask_rgba


def build_inpaint_transfer_payload(
    original_image: Image.Image,
    final_mask: np.ndarray,
) -> tuple[Image.Image, Image.Image, str, str]:
    """Prepare `(init_rgb, mask_rgba, payload_json, status_message)` for Inpaint upload transfer."""
    init_rgb, mask_rgba = build_inpaint_pair(original_image, final_mask)
    payload = json.dumps(
        {
            "image": _pil_to_b64_png(init_rgb),
            "mask": _pil_to_b64_png(mask_rgba),
        }
    )
    w, h = init_rgb.size
    msg = t("status.sent_to_inpaint", width=w, height=h)
    return init_rgb, mask_rgba, payload, msg


def render_masked_png(
    original_image: Image.Image,
    final_mask: np.ndarray,
    *,
    crop_to_mask: bool = False,
    white_background: bool = False,
) -> Image.Image:
    """Create a masked RGBA PNG from `original_image` and `final_mask` (spec §5.6).

    - `white_background=False` (default): transparent background (alpha = `final_mask`).
    - `white_background=True`: non-masked pixels filled with opaque white `(255, 255, 255, 255)`.
    - `crop_to_mask=True`: crops output to the bounding box of `final_mask > 0`.
    """
    rgba = original_image.convert("RGBA")
    w, h = rgba.size
    bm = mask_ops.ensure_binary_mask(final_mask, shape=(h, w))

    if white_background:
        rgb_arr = np.asarray(rgba.convert("RGB"), dtype=np.uint8).copy()
        outside = bm == 0
        rgb_arr[outside] = 255
        out_img = Image.fromarray(rgb_arr, mode="RGB").convert("RGBA")
    else:
        r, g, b, _ = rgba.split()
        alpha = Image.fromarray(bm, mode="L")
        out_img = Image.merge("RGBA", (r, g, b, alpha))

    if crop_to_mask and bm.any():
        ys, xs = np.nonzero(bm > 0)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        out_img = out_img.crop((x0, y0, x1, y1))

    return out_img


def export_masked_png(
    original_image: Image.Image,
    final_mask: np.ndarray,
    *,
    crop_to_mask: bool = False,
    white_background: bool = False,
    save_dir: str | Path | None = None,
) -> tuple[Path, str, str]:
    """Save masked RGBA PNG to disk and return `(saved_path, download_payload_json, status_msg)`."""
    out_img = render_masked_png(
        original_image,
        final_mask,
        crop_to_mask=crop_to_mask,
        white_background=white_background,
    )
    directory = Path(save_dir) if save_dir else Path(tempfile.gettempdir()) / "sam3_mask_studio"
    directory.mkdir(parents=True, exist_ok=True)
    out_path = directory / _timestamped_filename("sam3_masked")
    out_img.save(out_path, format="PNG")

    payload = json.dumps(
        {
            "image": _pil_to_b64_png(out_img),
            "filename": out_path.name,
        }
    )
    msg = t("status.png_exported", path=str(out_path), width=out_img.width, height=out_img.height)
    return out_path, payload, msg
