"""UI event handlers and state management for SAM3 Mask Studio (M1–M6 + Invert, Undo, Score Threshold & img2img Import)."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import io
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image

from i18n_helper import t
import mask_ops
import sam3_backend
import sam3_downloader
import transfer
from ui import presets as presets_mod

MAX_HISTORY_SIZE = 10
DEFAULT_SCORE_THRESHOLD = 0.35


@dataclass
class HistorySnapshot:
    """Complete mask editing state snapshot for accurate Undo across prompts, clicks, invert, and clear."""

    final_mask: np.ndarray
    base_mask: np.ndarray | None
    positive_prompts: list[str]
    negative_prompts: list[str]
    click_points_pos: list[tuple[int, int]]
    click_points_neg: list[tuple[int, int]]
    pos_prompt_masks: dict[str, np.ndarray]
    neg_prompt_masks: dict[str, np.ndarray]
    pos_click_masks: list[np.ndarray]
    neg_click_masks: list[np.ndarray]
    grow_px: int
    score_threshold: float = DEFAULT_SCORE_THRESHOLD


@dataclass
class MaskStudioState:
    """Per-session state container matching spec §6 Data Flow."""

    original_image: Image.Image | None = None
    positive_prompts: list[str] = field(default_factory=list)
    negative_prompts: list[str] = field(default_factory=list)
    click_points_pos: list[tuple[int, int]] = field(default_factory=list)
    click_points_neg: list[tuple[int, int]] = field(default_factory=list)
    grow_px: int = 0
    score_threshold: float = DEFAULT_SCORE_THRESHOLD
    final_mask: np.ndarray | None = None
    base_mask: np.ndarray | None = None
    history: list[HistorySnapshot | np.ndarray] = field(default_factory=list)
    # Cached raw masks per prompt/click so recomputation is fast
    pos_prompt_masks: dict[str, np.ndarray] = field(default_factory=dict)
    neg_prompt_masks: dict[str, np.ndarray] = field(default_factory=dict)
    pos_click_masks: list[np.ndarray] = field(default_factory=list)
    neg_click_masks: list[np.ndarray] = field(default_factory=list)
    color_by_prompt: bool = False

    def push_history(self) -> None:
        """Save current editing snapshot onto undo history."""
        if self.final_mask is not None:
            snap = HistorySnapshot(
                final_mask=self.final_mask.copy(),
                base_mask=self.base_mask.copy() if self.base_mask is not None else None,
                positive_prompts=list(self.positive_prompts),
                negative_prompts=list(self.negative_prompts),
                click_points_pos=list(self.click_points_pos),
                click_points_neg=list(self.click_points_neg),
                pos_prompt_masks={k: v.copy() for k, v in self.pos_prompt_masks.items()},
                neg_prompt_masks={k: v.copy() for k, v in self.neg_prompt_masks.items()},
                pos_click_masks=[m.copy() for m in self.pos_click_masks],
                neg_click_masks=[m.copy() for m in self.neg_click_masks],
                grow_px=int(self.grow_px),
                score_threshold=float(self.score_threshold),
            )
            self.history.append(snap)
            if len(self.history) > MAX_HISTORY_SIZE:
                self.history = self.history[-MAX_HISTORY_SIZE:]

    def restore_snapshot(self, snap: HistorySnapshot | np.ndarray) -> None:
        """Restore state from a `HistorySnapshot` (or raw `np.ndarray` for backward compatibility)."""
        if isinstance(snap, HistorySnapshot):
            self.final_mask = snap.final_mask.copy()
            self.base_mask = snap.base_mask.copy() if snap.base_mask is not None else None
            self.positive_prompts = list(snap.positive_prompts)
            self.negative_prompts = list(snap.negative_prompts)
            self.click_points_pos = list(snap.click_points_pos)
            self.click_points_neg = list(snap.click_points_neg)
            self.pos_prompt_masks = {k: v.copy() for k, v in snap.pos_prompt_masks.items()}
            self.neg_prompt_masks = {k: v.copy() for k, v in snap.neg_prompt_masks.items()}
            self.pos_click_masks = [m.copy() for m in snap.pos_click_masks]
            self.neg_click_masks = [m.copy() for m in snap.neg_click_masks]
            self.grow_px = int(snap.grow_px)
            self.score_threshold = float(snap.score_threshold)
        else:
            self.final_mask = snap.copy()

    def reset_for_new_image(self, image: Image.Image | None) -> None:
        """Reset all masks, prompts, clicks, and history when a new image is loaded."""
        self.original_image = image.convert("RGB") if image is not None else None
        self.positive_prompts.clear()
        self.negative_prompts.clear()
        self.click_points_pos.clear()
        self.click_points_neg.clear()
        self.pos_prompt_masks.clear()
        self.neg_prompt_masks.clear()
        self.pos_click_masks.clear()
        self.neg_click_masks.clear()
        self.base_mask = None
        self.grow_px = 0
        self.history.clear()
        if self.original_image is not None:
            w, h = self.original_image.size
            self.final_mask = mask_ops.empty_mask((h, w))
        else:
            self.final_mask = None


def ensure_state(state: MaskStudioState | None) -> MaskStudioState:
    return state if isinstance(state, MaskStudioState) else MaskStudioState()


def format_prompt_chips(state: MaskStudioState) -> list[str]:
    """Build chip strings (`+ <prompt>` / `- <prompt>`) for the active prompts list."""
    chips: list[str] = []
    for p in state.positive_prompts:
        token = f"+ {p}"
        if token not in chips:
            chips.append(token)
    for n in state.negative_prompts:
        token = f"- {n}"
        if token not in chips:
            chips.append(token)
    return chips


def _parse_prompt_list(text: str | None) -> list[str]:
    if not text:
        return []
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _decode_data_url_to_pil(value: str | None) -> Image.Image | None:
    if not value or not isinstance(value, str):
        return None
    if not value.startswith("data:image/"):
        return None
    try:
        _, _, b64 = value.partition(",")
        if not b64:
            return None
        raw = base64.b64decode(b64)
        with Image.open(io.BytesIO(raw)) as im:
            return im.convert("RGB")
    except Exception:
        return None


def _value_to_pil(value: Any) -> Image.Image | None:
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, str):
        decoded = _decode_data_url_to_pil(value)
        if decoded is not None:
            return decoded
        try:
            p = Path(value)
            if p.is_file():
                with Image.open(p) as im:
                    return im.convert("RGB")
        except Exception:
            pass
        return None
    if isinstance(value, dict):
        for k in ("image", "value", "composite", "background"):
            im = _value_to_pil(value.get(k))
            if im is not None:
                return im
        return None
    try:
        arr = np.asarray(value)
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            return Image.fromarray(arr[..., :3], mode="RGB")
    except Exception:
        pass
    return None


def recompute_and_render(
    state: MaskStudioState,
    *,
    record_history: bool = False,
) -> Image.Image | None:
    """Recompute `final_mask` via set operations + `grow_px` and return the preview image."""
    if state.original_image is None:
        state.final_mask = None
        return None

    if record_history:
        state.push_history()

    w, h = state.original_image.size
    shape = (h, w)
    pos_list: list[np.ndarray] = []
    if state.base_mask is not None:
        pos_list.append(state.base_mask)
    pos_list.extend(
        state.pos_prompt_masks[p] for p in state.positive_prompts if p in state.pos_prompt_masks
    )
    pos_list.extend(state.pos_click_masks)

    neg_list: list[np.ndarray] = [
        state.neg_prompt_masks[n] for n in state.negative_prompts if n in state.neg_prompt_masks
    ] + list(state.neg_click_masks)

    state.final_mask = mask_ops.compute_final_mask(
        positive_masks=pos_list,
        negative_masks=neg_list,
        shape=shape,
        grow_px=state.grow_px,
    )

    active_pos_masks = {
        p: state.pos_prompt_masks[p] for p in state.positive_prompts if p in state.pos_prompt_masks
    }
    return mask_ops.render_overlay(
        state.original_image,
        state.final_mask,
        prompt_masks=active_pos_masks,
        color_by_prompt=state.color_by_prompt,
        click_points_pos=state.click_points_pos,
        click_points_neg=state.click_points_neg,
    )


# ---------------------------------------------------------------------------
# M1: Image Input, Load from img2img & Color-Coding Mode
# ---------------------------------------------------------------------------
def on_image_Input_change(
    image: Image.Image | None,
    state: MaskStudioState | None,
) -> tuple[Image.Image | None, MaskStudioState, Any, str]:
    """Handle new or cleared input image."""
    st = ensure_state(state)
    st.reset_for_new_image(image)
    if st.original_image is None:
        return None, st, gr.update(choices=[], value=[]), t("status.image_cleared")

    preview = recompute_and_render(st, record_history=False)
    w, h = st.original_image.size
    return preview, st, gr.update(choices=[], value=[]), t("status.image_loaded", width=w, height=h)


def on_load_from_img2img(
    canvas_b64: str | None,
    state: MaskStudioState | None,
    inpaint_base_comp: Any = None,
) -> tuple[Any, Any, MaskStudioState, Any, str]:
    """Import image from `img2img` canvas or `Inpaint upload` into SAM3 Mask Studio."""
    st = ensure_state(state)
    img = _decode_data_url_to_pil(canvas_b64)
    if img is None and inpaint_base_comp is not None:
        img = _value_to_pil(getattr(inpaint_base_comp, "value", None))

    if img is None:
        return (
            gr.update(),
            gr.update(),
            st,
            gr.update(),
            t("status.img2img_not_found"),
        )

    st.reset_for_new_image(img)
    preview = recompute_and_render(st, record_history=False)
    w, h = img.size
    return (
        img,
        preview,
        st,
        gr.update(choices=[], value=[]),
        t("status.img2img_loaded", width=w, height=h),
    )


def on_color_mode_change(
    color_by_prompt: bool,
    state: MaskStudioState | None,
) -> tuple[Image.Image | None, MaskStudioState]:
    """Toggle per-prompt color-coded overlay vs unified final mask overlay."""
    st = ensure_state(state)
    st.color_by_prompt = bool(color_by_prompt)
    if st.original_image is None:
        return None, st
    active_pos_masks = {
        p: st.pos_prompt_masks[p] for p in st.positive_prompts if p in st.pos_prompt_masks
    }
    preview = mask_ops.render_overlay(
        st.original_image,
        st.final_mask,
        prompt_masks=active_pos_masks,
        color_by_prompt=st.color_by_prompt,
        click_points_pos=st.click_points_pos,
        click_points_neg=st.click_points_neg,
    )
    return preview, st


# ---------------------------------------------------------------------------
# M2: Text Mask & Presets
# ---------------------------------------------------------------------------
def _apply_prompts_to_state(
    st: MaskStudioState,
    pos_to_add: list[str],
    neg_to_add: list[str],
    ckpt_name: str | None = None,
    force_recompute: bool = False,
) -> tuple[int, list[str]]:
    """Run SAM3 text inference for positive/negative prompts and store them in `st`."""
    added_count = 0
    warnings: list[str] = []
    thr = float(st.score_threshold)

    for p in pos_to_add:
        if force_recompute or p not in st.pos_prompt_masks:
            mask = sam3_backend.predict_text_mask(
                st.original_image,
                p,
                score_threshold=thr,
                ckpt_name=ckpt_name,
            )
            if not mask.any():
                warnings.append(p)
            st.pos_prompt_masks[p] = mask
        if p not in st.positive_prompts:
            st.positive_prompts.append(p)
            added_count += 1

    for n in neg_to_add:
        if force_recompute or n not in st.neg_prompt_masks:
            mask = sam3_backend.predict_text_mask(
                st.original_image,
                n,
                score_threshold=thr,
                ckpt_name=ckpt_name,
            )
            if not mask.any():
                warnings.append(n)
            st.neg_prompt_masks[n] = mask
        if n not in st.negative_prompts:
            st.negative_prompts.append(n)
            added_count += 1

    return added_count, warnings


def on_add_positive(
    prompt_text: str,
    preset_id: str | None,
    state: MaskStudioState | None,
    ckpt_name: str | None = None,
    presets_path: Path | None = None,
) -> tuple[Image.Image | None, MaskStudioState, Any, str, str]:
    """Add positive prompt(s) and/or apply selected preset (positive + negative sets)."""
    st = ensure_state(state)
    if st.original_image is None:
        return None, st, gr.update(), prompt_text, t("status.no_image")

    pos_to_add = _parse_prompt_list(prompt_text)
    neg_to_add: list[str] = []

    if preset_id:
        preset = presets_mod.find_preset(preset_id, path=presets_path)
        if preset:
            for p in preset.get("positive", []):
                if p not in pos_to_add:
                    pos_to_add.append(p)
            for n in preset.get("negative", []):
                if n not in neg_to_add:
                    neg_to_add.append(n)

    if not pos_to_add and not neg_to_add:
        return (
            recompute_and_render(st, record_history=False),
            st,
            gr.update(),
            prompt_text,
            t("status.no_prompt"),
        )

    try:
        st.push_history()
        _added, warnings = _apply_prompts_to_state(st, pos_to_add, neg_to_add, ckpt_name=ckpt_name)
    except Exception as exc:
        if st.history:
            st.history.pop()
        return (
            recompute_and_render(st, record_history=False),
            st,
            gr.update(),
            prompt_text,
            t("status.inference_error", error=str(exc)),
        )

    preview = recompute_and_render(st, record_history=False)
    chips = format_prompt_chips(st)
    msg = t("status.prompts_applied", count=len(chips))
    if warnings:
        msg += " | " + t("status.empty_mask_warning", prompts=", ".join(warnings))
    return preview, st, gr.update(choices=chips, value=chips), "", msg


def on_add_negative(
    prompt_text: str,
    preset_id: str | None,
    state: MaskStudioState | None,
    ckpt_name: str | None = None,
    presets_path: Path | None = None,
) -> tuple[Image.Image | None, MaskStudioState, Any, str, str]:
    """Add negative prompt(s) or apply selected preset's prompts as negative exclusions."""
    st = ensure_state(state)
    if st.original_image is None:
        return None, st, gr.update(), prompt_text, t("status.no_image")

    neg_to_add = _parse_prompt_list(prompt_text)
    if preset_id:
        preset = presets_mod.find_preset(preset_id, path=presets_path)
        if preset:
            for p in preset.get("positive", []):
                if p not in neg_to_add:
                    neg_to_add.append(p)

    if not neg_to_add:
        return (
            recompute_and_render(st, record_history=False),
            st,
            gr.update(),
            prompt_text,
            t("status.no_prompt"),
        )

    try:
        st.push_history()
        _added, warnings = _apply_prompts_to_state(st, [], neg_to_add, ckpt_name=ckpt_name)
    except Exception as exc:
        if st.history:
            st.history.pop()
        return (
            recompute_and_render(st, record_history=False),
            st,
            gr.update(),
            prompt_text,
            t("status.inference_error", error=str(exc)),
        )

    preview = recompute_and_render(st, record_history=False)
    chips = format_prompt_chips(st)
    msg = t("status.prompts_applied", count=len(chips))
    if warnings:
        msg += " | " + t("status.empty_mask_warning", prompts=", ".join(warnings))
    return preview, st, gr.update(choices=chips, value=chips), "", msg


def on_active_prompts_change(
    selected_chips: list[str] | None,
    state: MaskStudioState | None,
) -> tuple[Image.Image | None, MaskStudioState, Any, str]:
    """Remove any prompt chip unchecked by the user and immediately recompute final_mask."""
    st = ensure_state(state)
    keep = set(selected_chips or [])

    new_pos = [p for p in st.positive_prompts if f"+ {p}" in keep]
    new_neg = [n for n in st.negative_prompts if f"- {n}" in keep]

    if new_pos == st.positive_prompts and new_neg == st.negative_prompts:
        chips = format_prompt_chips(st)
        return (
            recompute_and_render(st, record_history=False),
            st,
            gr.update(choices=chips, value=chips),
            t("status.prompts_applied", count=len(chips)),
        )

    st.push_history()
    st.positive_prompts = new_pos
    st.negative_prompts = new_neg
    st.pos_prompt_masks = {k: v for k, v in st.pos_prompt_masks.items() if k in new_pos}
    st.neg_prompt_masks = {k: v for k, v in st.neg_prompt_masks.items() if k in new_neg}

    preview = recompute_and_render(st, record_history=False)
    chips = format_prompt_chips(st)
    return (
        preview,
        st,
        gr.update(choices=chips, value=chips),
        t("status.prompts_applied", count=len(chips)),
    )


def on_save_custom_preset(
    preset_name: str,
    pos_text: str,
    neg_text: str,
    presets_path: Path | None = None,
) -> tuple[Any, Any, str]:
    """Save or update a preset in `user_presets.json` and refresh preset dropdowns."""
    name = (preset_name or "").strip()
    if not name:
        return gr.update(), gr.update(), t("status.preset_name_required")
    try:
        presets_mod.upsert_preset(name, pos_text, neg_text, path=presets_path)
        choices = presets_mod.get_dropdown_choices(path=presets_path)
        return (
            gr.update(choices=choices, value=name),
            gr.update(choices=choices, value=None),
            t("status.preset_saved", name=name),
        )
    except Exception as exc:
        return gr.update(), gr.update(), str(exc)


def on_delete_custom_preset(
    preset_id: str | None,
    presets_path: Path | None = None,
) -> tuple[Any, Any, str]:
    """Delete a preset from `user_presets.json` and refresh preset dropdowns."""
    target = (preset_id or "").strip()
    if not target:
        return gr.update(), gr.update(), t("status.preset_name_required")
    presets_mod.delete_preset(target, path=presets_path)
    choices = presets_mod.get_dropdown_choices(path=presets_path)
    return (
        gr.update(choices=choices, value=None),
        gr.update(choices=choices, value=None),
        t("status.preset_deleted", name=target),
    )


# ---------------------------------------------------------------------------
# M3: Click Operations (Add / Exclude with Coordinate Scaling)
# ---------------------------------------------------------------------------
def on_preview_click(
    click_mode: str,
    state: MaskStudioState | None,
    evt: gr.SelectData | None = None,
    *,
    coords: tuple[float, float] | None = None,
    display_size: tuple[int, int] | None = None,
    ckpt_name: str | None = None,
) -> tuple[Image.Image | None, MaskStudioState, str]:
    """Handle click on the preview image, scale coordinates, and update positive/negative set."""
    st = ensure_state(state)
    if st.original_image is None:
        return None, st, t("status.no_image")

    if coords is not None:
        raw_x, raw_y = coords
    elif evt is not None and getattr(evt, "index", None) is not None:
        raw_x, raw_y = float(evt.index[0]), float(evt.index[1])
    else:
        return recompute_and_render(st, record_history=False), st, t("status.click_ignored")

    orig_w, orig_h = st.original_image.size
    x, y = mask_ops.scale_click_coords(raw_x, raw_y, display_size=display_size, orig_size=(orig_w, orig_h))

    is_exclude = "Exclude" in (click_mode or "") or "除外" in (click_mode or "")
    try:
        st.push_history()
        click_mask = sam3_backend.predict_point_mask(
            st.original_image,
            point_coords=[(x, y)],
            point_labels=[1],
            ckpt_name=ckpt_name,
        )
    except Exception as exc:
        if st.history:
            st.history.pop()
        return (
            recompute_and_render(st, record_history=False),
            st,
            t("status.inference_error", error=str(exc)),
        )

    if is_exclude:
        st.click_points_neg.append((x, y))
        st.neg_click_masks.append(click_mask)
        mode_str = t("Click Mode: Exclude")
    else:
        st.click_points_pos.append((x, y))
        st.pos_click_masks.append(click_mask)
        mode_str = t("Click Mode: Add")

    preview = recompute_and_render(st, record_history=False)
    return preview, st, t("status.click_applied", mode=mode_str, x=x, y=y)


# ---------------------------------------------------------------------------
# M4: Mask Adjustment (Grow / Shrink, Score Threshold, Invert, Clear, Undo)
# ---------------------------------------------------------------------------
def on_grow_change(
    grow_px: int,
    state: MaskStudioState | None,
    *,
    record_history: bool = True,
) -> tuple[Image.Image | None, MaskStudioState, str]:
    """Apply `grow_px` morphological dilation/erosion to `final_mask`."""
    st = ensure_state(state)
    if st.original_image is None:
        st.grow_px = int(grow_px)
        return None, st, t("status.no_image")

    new_px = int(grow_px)
    if record_history and new_px != st.grow_px and st.final_mask is not None:
        st.push_history()

    st.grow_px = new_px
    preview = recompute_and_render(st, record_history=False)
    return preview, st, t("status.grow_applied", px=st.grow_px)


def on_score_threshold_change(
    score_threshold: float,
    state: MaskStudioState | None,
    ckpt_name: str | None = None,
) -> tuple[Image.Image | None, MaskStudioState, str]:
    """Update SAM3 text detection score threshold and re-evaluate active text prompts."""
    st = ensure_state(state)
    new_thr = round(float(score_threshold), 2)
    if st.original_image is None:
        st.score_threshold = new_thr
        return None, st, t("status.threshold_applied", thr=new_thr)

    has_prompts = bool(st.positive_prompts or st.negative_prompts)
    if has_prompts and abs(new_thr - st.score_threshold) > 1e-4:
        st.push_history()
        st.score_threshold = new_thr
        try:
            _apply_prompts_to_state(
                st,
                list(st.positive_prompts),
                list(st.negative_prompts),
                ckpt_name=ckpt_name,
                force_recompute=True,
            )
        except Exception as exc:
            if st.history:
                st. restore_snapshot(st.history.pop())
            return (
                recompute_and_render(st, record_history=False),
                st,
                t("status.inference_error", error=str(exc)),
            )
    else:
        st.score_threshold = new_thr

    preview = recompute_and_render(st, record_history=False)
    return preview, st, t("status.threshold_applied", thr=st.score_threshold)


def on_invert_mask(
    state: MaskStudioState | None,
) -> tuple[Image.Image | None, MaskStudioState, Any, int, str]:
    """Invert the entire current `final_mask` (`0 <-> 255`) and set it as `base_mask`."""
    st = ensure_state(state)
    if st.original_image is None:
        return None, st, gr.update(), 0, t("status.no_image")

    w, h = st.original_image.size
    st.push_history()

    current = st.final_mask if st.final_mask is not None else mask_ops.empty_mask((h, w))
    inverted = mask_ops.invert_mask(current, shape=(h, w))

    st.base_mask = inverted
    st.positive_prompts.clear()
    st.negative_prompts.clear()
    st.click_points_pos.clear()
    st.click_points_neg.clear()
    st.pos_prompt_masks.clear()
    st.neg_prompt_masks.clear()
    st.pos_click_masks.clear()
    st.neg_click_masks.clear()
    st.grow_px = 0
    st.final_mask = inverted.copy()

    preview = mask_ops.render_overlay(st.original_image, st.final_mask)
    return preview, st, gr.update(choices=[], value=[]), 0, t("status.mask_inverted")


def on_clear_mask(
    state: MaskStudioState | None,
) -> tuple[Image.Image | None, MaskStudioState, Any, int, str]:
    """Clear all prompts, clicks, and mask while saving current state in undo history."""
    st = ensure_state(state)
    if st.original_image is None:
        return None, st, gr.update(choices=[], value=[]), 0, t("status.no_image")

    if st.final_mask is not None and st.final_mask.any():
        st.push_history()

    st.positive_prompts.clear()
    st.negative_prompts.clear()
    st.click_points_pos.clear()
    st.click_points_neg.clear()
    st.pos_prompt_masks.clear()
    st.neg_prompt_masks.clear()
    st.pos_click_masks.clear()
    st.neg_click_masks.clear()
    st.base_mask = None
    st.grow_px = 0
    w, h = st.original_image.size
    st.final_mask = mask_ops.empty_mask((h, w))

    preview = mask_ops.render_overlay(st.original_image, st.final_mask)
    return preview, st, gr.update(choices=[], value=[]), 0, t("status.mask_cleared")


def on_undo(
    state: MaskStudioState | None,
) -> tuple[Image.Image | None, MaskStudioState, str]:
    """Restore the most recent state snapshot from `state.history`."""
    st = ensure_state(state)
    if st.original_image is None:
        return None, st, t("status.no_image")
    if not st.history:
        active_pos_masks = {
            p: st.pos_prompt_masks[p] for p in st.positive_prompts if p in st.pos_prompt_masks
        }
        return (
            mask_ops.render_overlay(
                st.original_image,
                st.final_mask,
                prompt_masks=active_pos_masks,
                color_by_prompt=st.color_by_prompt,
                click_points_pos=st.click_points_pos,
                click_points_neg=st.click_points_neg,
            ),
            st,
            t("status.undo_empty"),
        )

    snap = st.history.pop()
    st.restore_snapshot(snap)
    active_pos_masks = {
        p: st.pos_prompt_masks[p] for p in st.positive_prompts if p in st.pos_prompt_masks
    }
    preview = mask_ops.render_overlay(
        st.original_image,
        st.final_mask,
        prompt_masks=active_pos_masks,
        color_by_prompt=st.color_by_prompt,
        click_points_pos=st.click_points_pos,
        click_points_neg=st.click_points_neg,
    )
    return preview, st, t("status.undo_applied", remaining=len(st.history))


def on_undo_with_ui(
    state: MaskStudioState | None,
) -> tuple[Image.Image | None, MaskStudioState, Any, int, float, str]:
    """Undo handler for UI buttons that syncs prompt chips, `grow_px`, and `score_threshold`."""
    preview, st, msg = on_undo(state)
    chips = format_prompt_chips(st)
    return preview, st, gr.update(choices=chips, value=chips), st.grow_px, st.score_threshold, msg


# ---------------------------------------------------------------------------
# M5: Output (Send to Inpaint upload & Download Masked PNG)
# ---------------------------------------------------------------------------
def on_send_to_inpaint(
    state: MaskStudioState | None,
    has_inpaint_components: bool = False,
) -> tuple[Any, ...]:
    """Prepare image + mask pair for `img2img` Inpaint upload."""
    st = ensure_state(state)
    if st.original_image is None:
        msg = t("status.no_image")
        return (msg, "", gr.update(), gr.update()) if has_inpaint_components else (msg, "")
    if st.final_mask is None or not st.final_mask.any():
        msg = t("status.no_mask")
        return (msg, "", gr.update(), gr.update()) if has_inpaint_components else (msg, "")

    init_rgb, mask_rgba, payload, msg = transfer.build_inpaint_transfer_payload(
        st.original_image, st.final_mask
    )
    if has_inpaint_components:
        return msg, payload, gr.update(value=init_rgb), gr.update(value=mask_rgba)
    return msg, payload


def on_download_masked_png(
    crop_mode: str,
    bg_mode: str,
    state: MaskStudioState | None,
    save_dir: str | Path | None = None,
) -> tuple[str, str]:
    """Export masked RGBA PNG according to region and background options."""
    st = ensure_state(state)
    if st.original_image is None:
        return t("status.no_image"), ""
    if st.final_mask is None or not st.final_mask.any():
        return t("status.no_mask"), ""

    crop_to_mask = "Crop" in (crop_mode or "") or "クロップ" in (crop_mode or "")
    white_bg = "White" in (bg_mode or "") or "白" in (bg_mode or "")
    _path, payload, msg = transfer.export_masked_png(
        st.original_image,
        st.final_mask,
        crop_to_mask=crop_to_mask,
        white_background=white_bg,
        save_dir=save_dir,
    )
    return msg, payload


# ---------------------------------------------------------------------------
# M6: BYOT Model Setup & Model Controls
# ---------------------------------------------------------------------------
def on_check_hf_access(token: str | None) -> str:
    _code, msg = sam3_downloader.check_access(token=token)
    return msg


def on_download_sam3_model(token: str | None) -> tuple[str, Any]:
    ok, msg, _path = sam3_downloader.download_sam3(token=token)
    ckpts = sam3_backend.list_checkpoints()
    default_ckpt = ckpts[0] if ckpts else None
    return msg, gr.update(choices=ckpts, value=default_ckpt)


def on_load_model(ckpt_name: str | None) -> str:
    ok, msg = sam3_backend.load_model(ckpt_name)
    return msg


def on_unload_model() -> str:
    _ok, msg = sam3_backend.unload_model()
    return msg
