"""Forge Neo scripts/ loader shim forwarding to root `script.py` via explicit path."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

EXT_ROOT = Path(__file__).resolve().parents[1]
if str(EXT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXT_ROOT))

_SCRIPT_PATH = EXT_ROOT / "script.py"
_MOD_NAME = "_sam3_mask_studio_root_script"

if _MOD_NAME not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_MOD_NAME, _SCRIPT_PATH)
    if _spec is not None and _spec.loader is not None:
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules[_MOD_NAME] = _mod
        _spec.loader.exec_module(_mod)


def generate_auto_mask(
    image: "Any",
    pos_prompt: str,
    neg_prompt: str = "",
    score_threshold: float = 0.35,
    grow_px: int = 0,
    feather_px: int = 0,
    invert_mask: bool = False,
):
    """Generate a PIL L-mode mask from text prompts for img2img Inpaint automation."""
    from PIL import Image
    import mask_ops
    import sam3_backend

    pos_list = [p.strip() for p in (pos_prompt or "").split(",") if p.strip()]
    neg_list = [n.strip() for n in (neg_prompt or "").split(",") if n.strip()]
    if not pos_list:
        return None

    rgb = image.convert("RGB")
    w, h = rgb.size
    thr = float(score_threshold)
    pos_masks = [sam3_backend.predict_text_mask(rgb, p, score_threshold=thr) for p in pos_list]
    neg_masks = [sam3_backend.predict_text_mask(rgb, n, score_threshold=thr) for n in neg_list]
    final_mask = mask_ops.compute_final_mask(
        positive_masks=pos_masks,
        negative_masks=neg_masks,
        shape=(h, w),
        grow_px=int(grow_px),
    )
    if invert_mask:
        final_mask = mask_ops.invert_mask(final_mask)
    if int(feather_px) > 0:
        final_mask = mask_ops.apply_feather(final_mask, int(feather_px))
    return Image.fromarray(final_mask, mode="L")


try:
    import gradio as gr
    import modules.scripts as scripts

    class SAM3AutoMaskScript(scripts.Script):
        """AlwaysVisible img2img script that auto-generates an inpaint mask via SAM3."""

        def title(self):
            return "SAM3 Auto-Mask"

        def show(self, is_img2img):
            return scripts.AlwaysVisible if is_img2img else False

        def ui(self, is_img2img):
            with gr.Accordion("SAM3 Auto-Mask", open=False, elem_id="sam3_automask_acc"):
                enabled = gr.Checkbox(label="Enable SAM3 Auto-Mask", value=False)
                with gr.Row():
                    pos_prompt = gr.Textbox(
                        label="Positive Prompt",
                        placeholder="e.g. face, hair",
                        scale=2,
                    )
                    neg_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="e.g. background",
                        scale=2,
                    )
                with gr.Row():
                    score_thr = gr.Slider(0.05, 0.95, value=0.35, step=0.05, label="Score Threshold")
                    grow_px = gr.Slider(-64, 64, value=0, step=1, label="Grow / Shrink (px)")
                    feather_px = gr.Slider(0, 64, value=8, step=1, label="Feather (px)")
                with gr.Row():
                    invert_mask = gr.Checkbox(label="Invert Mask", value=False)
                    auto_unload = gr.Checkbox(label="Unload SAM3 before generation", value=True)
            return [enabled, pos_prompt, neg_prompt, score_thr, grow_px, feather_px, invert_mask, auto_unload]

        def process(self, p, enabled, pos_prompt, neg_prompt, score_thr, grow_px, feather_px, invert_mask, auto_unload):
            if not enabled:
                return
            init_images = getattr(p, "init_images", None)
            if not init_images:
                return
            mask_img = generate_auto_mask(
                init_images[0],
                pos_prompt=pos_prompt,
                neg_prompt=neg_prompt,
                score_threshold=score_thr,
                grow_px=grow_px,
                feather_px=feather_px,
                invert_mask=invert_mask,
            )
            if mask_img is not None:
                p.image_mask = mask_img
            if auto_unload:
                import sam3_backend

                if sam3_backend.is_loaded():
                    sam3_backend.unload_model()

except Exception:
    pass
