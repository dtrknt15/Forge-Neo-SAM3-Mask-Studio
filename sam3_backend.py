"""SAM3 backend wrapper for model lifecycle, text prompt inference, and point click inference.

Supports:
- Thread-safe lazy/manual model loading, unloading, and VRAM cleanup
- Image feature caching so sequential text prompts and interactive clicks on the
  same image reuse `backbone_out` without re-running the vision encoder
- Text prompt segmentation (`predict_text_mask`)
- Interactive point segmentation (`predict_point_mask` via `predict_inst` with
  fallback to `geometry_encoder` points)
"""

from __future__ import annotations

from contextlib import nullcontext
import gc
import hashlib
import os
import threading
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

import mask_ops

LOG_PREFIX = "[SAM3 Mask Studio]"
_CKPT_EXTS = {".pt", ".pth", ".safetensors"}
_OFFICIAL_CKPT_NAMES = ("sam3.pt", "sam3.1_multiplex.pt", "sam3.1.pt")

_LOCK = threading.RLock()

_STATE: dict[str, Any] = {
    "model": None,
    "processor": None,
    "ckpt_path": None,
    "device": "cuda",
    "image_hash": None,
    "image_size": None,
    "inference_state": None,
}


def _log(msg: str) -> None:
    print(f"{LOG_PREFIX} {msg}", flush=True)


def _autocast_ctx():
    try:
        import torch

        if torch.cuda.is_available():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    except Exception:
        pass
    return nullcontext()


def get_models_dirs() -> list[Path]:
    """Return candidate SAM3 model directories (`models/sam3` and `models/SAM3`)."""
    try:
        from modules import paths_internal

        base = Path(paths_internal.models_path)
    except Exception:
        base = Path(__file__).resolve().parents[2] / "models"

    dirs: list[Path] = []
    for sub in ("sam3", "SAM3"):
        cand = base / sub
        if cand not in dirs:
            dirs.append(cand)
    return dirs


def primary_models_dir() -> Path:
    """Return the canonical `models/sam3` directory path."""
    return get_models_dirs()[0]


def list_checkpoints() -> list[str]:
    """List available checkpoint filenames in `models/sam3` (and `models/SAM3`)."""
    names: list[str] = []
    env_ckpt = os.environ.get("SAM3_CHECKPOINT", "").strip()
    if env_ckpt and Path(env_ckpt).is_file():
        names.append(Path(env_ckpt).name)

    for folder in get_models_dirs():
        if folder.is_dir():
            for p in sorted(folder.iterdir()):
                if p.is_file() and p.suffix.lower() in _CKPT_EXTS and p.name not in names:
                    names.append(p.name)

    def _sort_key(n: str) -> tuple[int, str]:
        try:
            rank = _OFFICIAL_CKPT_NAMES.index(n)
        except ValueError:
            rank = len(_OFFICIAL_CKPT_NAMES)
        return (rank, n.lower())

    names.sort(key=_sort_key)
    return names


def resolve_checkpoint_path(ckpt_name: str | None = None) -> Path | None:
    """Resolve a checkpoint filename or path to an existing file on disk."""
    env_ckpt = os.environ.get("SAM3_CHECKPOINT", "").strip()
    if ckpt_name:
        cand_path = Path(ckpt_name)
        if cand_path.is_file():
            return cand_path
        for folder in get_models_dirs():
            direct = folder / ckpt_name
            if direct.is_file():
                return direct
        if env_ckpt and Path(env_ckpt).name == ckpt_name and Path(env_ckpt).is_file():
            return Path(env_ckpt)
        return None

    if env_ckpt and Path(env_ckpt).is_file():
        return Path(env_ckpt)

    available = list_checkpoints()
    if not available:
        return None
    for folder in get_models_dirs():
        direct = folder / available[0]
        if direct.is_file():
            return direct
    return None


def is_loaded() -> bool:
    return _STATE["model"] is not None and _STATE["processor"] is not None


def unload_model() -> tuple[bool, str]:
    """Unload SAM3 model and free CUDA cache."""
    with _LOCK:
        _STATE["model"] = None
        _STATE["processor"] = None
        _STATE["ckpt_path"] = None
        _STATE["image_hash"] = None
        _STATE["image_size"] = None
        _STATE["inference_state"] = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        _log("Unloaded SAM3 model and freed VRAM cache.")
        return True, "SAM3 unloaded"


def _remap_safetensors_ckpt(raw: dict[str, Any], has_interactive: bool) -> dict[str, Any]:
    if "model" in raw and isinstance(raw["model"], dict):
        raw = raw["model"]
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if k.startswith("detector_model."):
            out[k[len("detector_model.") :]] = v
        elif k.startswith("detector."):
            out[k[len("detector.") :]] = v
        elif has_interactive and k.startswith("tracker."):
            out[k.replace("tracker.", "inst_interactive_predictor.model.", 1)] = v
    return out


def load_model(ckpt_name: str | None = None) -> tuple[bool, str]:
    """Load SAM3 model (with both text grounding and interactive click predictor)."""
    with _LOCK:
        path = resolve_checkpoint_path(ckpt_name)
        if path is None:
            return False, f"No SAM3 checkpoint found in {primary_models_dir()}"

        path_str = str(path)
        if is_loaded() and _STATE["ckpt_path"] == path_str:
            return True, f"Already loaded: {path.name}"

        if is_loaded():
            unload_model()

        try:
            import torch
        except Exception as exc:
            return False, f"PyTorch is not available: {exc}"

        if not torch.cuda.is_available():
            return False, "CUDA is not available. SAM3 requires a CUDA GPU."

        try:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except Exception as exc:
            return False, f"sam3 package is not installed: {exc}"

        try:
            _log(f"Loading SAM3 checkpoint: {path.name} ...")
            is_st = path.suffix.lower() == ".safetensors"
            model = build_sam3_image_model(
                checkpoint_path=None if is_st else path_str,
                load_from_HF=False,
                device="cuda",
                eval_mode=True,
                enable_segmentation=True,
                enable_inst_interactivity=True,
            )
            if is_st:
                from safetensors.torch import load_file

                raw = load_file(path_str)
                remapped = _remap_safetensors_ckpt(raw, model.inst_interactive_predictor is not None)
                if not remapped:
                    raise RuntimeError("No valid SAM3 detector weights found in safetensors file.")
                model.load_state_dict(remapped, strict=False)
                model = model.cuda()
                model.eval()

            processor = Sam3Processor(model)
            _STATE["model"] = model
            _STATE["processor"] = processor
            _STATE["ckpt_path"] = path_str
            _STATE["device"] = "cuda"
            _STATE["image_hash"] = None
            _STATE["image_size"] = None
            _STATE["inference_state"] = None
            _log(f"Loaded SAM3 checkpoint: {path.name}")
            return True, f"Loaded: {path.name}"
        except Exception as exc:
            unload_model()
            return False, f"Failed to load SAM3: {exc}"


def ensure_loaded(ckpt_name: str | None = None) -> tuple[bool, str]:
    if is_loaded():
        return True, "Loaded"
    return load_model(ckpt_name)


def _compute_image_hash(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    h = hashlib.sha1()
    h.update(f"{rgb.width}x{rgb.height}:".encode("ascii"))
    h.update(rgb.tobytes())
    return h.hexdigest()


def _get_or_set_image_state(image: Image.Image) -> Any:
    """Return cached processor state for `image`, running `set_image` only when image changes."""
    rgb = image.convert("RGB")
    img_hash = _compute_image_hash(rgb)
    if (
        _STATE["inference_state"] is not None
        and _STATE["image_hash"] == img_hash
        and _STATE["image_size"] == rgb.size
    ):
        return _STATE["inference_state"]

    processor = _STATE["processor"]
    if processor is None:
        raise RuntimeError("SAM3 processor is not initialized.")

    with _autocast_ctx():
        state = processor.set_image(rgb)

    _STATE["image_hash"] = img_hash
    _STATE["image_size"] = rgb.size
    _STATE["inference_state"] = state
    return state


def _extract_binary_masks_from_state(
    state: Any,
    target_shape: tuple[int, int],
    score_threshold: float = 0.35,
) -> np.ndarray:
    """Combine all masks passing `score_threshold` into a single uint8 (0/255) mask."""
    if not isinstance(state, dict) or "masks" not in state or state["masks"] is None:
        return mask_ops.empty_mask(target_shape)

    masks_raw = state["masks"]
    scores_raw = state.get("scores")

    try:
        import torch

        if isinstance(masks_raw, torch.Tensor):
            t = masks_raw.detach().float().cpu()
            if t.ndim == 4:
                t = t.squeeze(1) if t.shape[1] == 1 else t.reshape(-1, t.shape[-2], t.shape[-1])
            elif t.ndim == 2:
                t = t.unsqueeze(0)
            masks_list = [((t[i].numpy() > 0.5).astype(np.uint8)) * 255 for i in range(t.shape[0])]
        else:
            arr = np.asarray(masks_raw)
            if arr.ndim == 2:
                arr = arr[None]
            elif arr.ndim == 4:
                arr = arr[:, 0]
            masks_list = [((arr[i] > 0.5).astype(np.uint8)) * 255 for i in range(arr.shape[0])]
    except Exception:
        return mask_ops.empty_mask(target_shape)

    if not masks_list:
        return mask_ops.empty_mask(target_shape)

    if scores_raw is not None:
        try:
            import torch

            if isinstance(scores_raw, torch.Tensor):
                scores_np = scores_raw.detach().float().cpu().numpy().reshape(-1)
            else:
                scores_np = np.asarray(scores_raw, dtype=np.float32).reshape(-1)
            if len(scores_np) == len(masks_list):
                filtered = [m for m, s in zip(masks_list, scores_np) if float(s) >= float(score_threshold)]
                if filtered:
                    masks_list = filtered
        except Exception:
            pass

    return mask_ops.union_masks(masks_list, shape=target_shape)


def predict_text_mask(
    image: Image.Image,
    prompt: str,
    *,
    score_threshold: float = 0.35,
    ckpt_name: str | None = None,
) -> np.ndarray:
    """Run SAM3 text-prompt segmentation and return a 2D uint8 (0/255) mask."""
    clean_prompt = (prompt or "").strip()
    w, h = image.size
    target_shape = (h, w)
    if not clean_prompt:
        return mask_ops.empty_mask(target_shape)

    with _LOCK:
        ok, msg = ensure_loaded(ckpt_name)
        if not ok:
            raise RuntimeError(msg)

        processor = _STATE["processor"]
        model = _STATE["model"]
        state = _get_or_set_image_state(image)

        # Reset geometric prompt before running pure text grounding
        state["geometric_prompt"] = model._get_dummy_prompt()
        old_thr = getattr(processor, "confidence_threshold", 0.5)
        processor.confidence_threshold = float(score_threshold)
        try:
            with _autocast_ctx():
                state = processor.set_text_prompt(prompt=clean_prompt, state=state)
        finally:
            processor.confidence_threshold = old_thr

        return _extract_binary_masks_from_state(state, target_shape=target_shape, score_threshold=score_threshold)


def predict_point_mask(
    image: Image.Image,
    point_coords: Sequence[tuple[int, int]],
    point_labels: Sequence[int] | None = None,
    *,
    multimask_output: bool = True,
    ckpt_name: str | None = None,
) -> np.ndarray:
    """Run SAM3 interactive point segmentation and return a 2D uint8 (0/255) mask.

    `point_coords`: sequence of `(x, y)` in original image pixel coordinates.
    `point_labels`: sequence of `1` (positive) or `0` (negative); defaults to all `1`s.
    """
    w, h = image.size
    target_shape = (h, w)
    if not point_coords:
        return mask_ops.empty_mask(target_shape)

    if point_labels is None:
        labels_list = [1] * len(point_coords)
    else:
        labels_list = [int(lbl) for lbl in point_labels]

    with _LOCK:
        ok, msg = ensure_loaded(ckpt_name)
        if not ok:
            raise RuntimeError(msg)

        model = _STATE["model"]
        processor = _STATE["processor"]
        state = _get_or_set_image_state(image)

        coords_np = np.asarray(point_coords, dtype=np.float32).reshape(-1, 2)
        labels_np = np.asarray(labels_list, dtype=np.int32).reshape(-1)

        if getattr(model, "inst_interactive_predictor", None) is not None and "sam2_backbone_out" in state.get("backbone_out", {}):
            with _autocast_ctx():
                masks_np, iou_np, _low_res = model.predict_inst(
                    state,
                    point_coords=coords_np,
                    point_labels=labels_np,
                    multimask_output=bool(multimask_output),
                    normalize_coords=True,
                )
            if masks_np is None or len(masks_np) == 0:
                return mask_ops.empty_mask(target_shape)
            best_idx = int(np.argmax(iou_np)) if iou_np is not None and len(iou_np) > 0 else 0
            best_mask = masks_np[best_idx]
            return mask_ops.ensure_binary_mask(best_mask, shape=target_shape)

        # Fallback: geometric point prompt on detector if tracker is unavailable
        import torch

        state["geometric_prompt"] = model._get_dummy_prompt()
        if "language_features" not in state["backbone_out"]:
            with _autocast_ctx():
                dummy_txt = model.backbone.forward_text(["visual"], device=_STATE["device"])
            state["backbone_out"].update(dummy_txt)

        norm_pts = coords_np.copy()
        norm_pts[:, 0] /= max(1.0, float(w))
        norm_pts[:, 1] /= max(1.0, float(h))
        pts_tensor = torch.tensor(norm_pts, device=_STATE["device"], dtype=torch.float32).unsqueeze(1)
        lbl_tensor = torch.tensor(labels_np, device=_STATE["device"], dtype=torch.long).unsqueeze(1)
        state["geometric_prompt"].append_points(pts_tensor, lbl_tensor)
        with _autocast_ctx():
            state = processor._forward_grounding(state)
        return _extract_binary_masks_from_state(state, target_shape=target_shape)


def predict_box_mask(
    image: Image.Image,
    box_xyxy: Sequence[float],
    *,
    multimask_output: bool = True,
    ckpt_name: str | None = None,
) -> np.ndarray:
    """Run SAM3 bounding-box segmentation and return a 2D uint8 (0/255) mask.

    `box_xyxy`: `(x0, y0, x1, y1)` in original image pixel coordinates.
    """
    w, h = image.size
    target_shape = (h, w)
    if len(box_xyxy) != 4:
        return mask_ops.empty_mask(target_shape)

    x0 = max(0.0, min(float(w), float(min(box_xyxy[0], box_xyxy[2]))))
    y0 = max(0.0, min(float(h), float(min(box_xyxy[1], box_xyxy[3]))))
    x1 = max(0.0, min(float(w), float(max(box_xyxy[0], box_xyxy[2]))))
    y1 = max(0.0, min(float(h), float(max(box_xyxy[1], box_xyxy[3]))))
    if (x1 - x0) < 2.0 or (y1 - y0) < 2.0:
        return mask_ops.empty_mask(target_shape)

    with _LOCK:
        ok, msg = ensure_loaded(ckpt_name)
        if not ok:
            raise RuntimeError(msg)

        model = _STATE["model"]
        processor = _STATE["processor"]
        state = _get_or_set_image_state(image)

        if getattr(model, "inst_interactive_predictor", None) is not None and "sam2_backbone_out" in state.get("backbone_out", {}):
            box_np = np.asarray([x0, y0, x1, y1], dtype=np.float32)
            with _autocast_ctx():
                masks_np, iou_np, _low_res = model.predict_inst(
                    state,
                    box=box_np,
                    multimask_output=bool(multimask_output),
                    normalize_coords=True,
                )
            if masks_np is None or len(masks_np) == 0:
                return mask_ops.empty_mask(target_shape)
            best_idx = int(np.argmax(iou_np)) if iou_np is not None and len(iou_np) > 0 else 0
            best_mask = masks_np[best_idx]
            return mask_ops.ensure_binary_mask(best_mask, shape=target_shape)

        # Fallback: geometric box prompt on detector
        state["geometric_prompt"] = model._get_dummy_prompt()
        cx = ((x0 + x1) * 0.5) / max(1.0, float(w))
        cy = ((y0 + y1) * 0.5) / max(1.0, float(h))
        bw = (x1 - x0) / max(1.0, float(w))
        bh = (y1 - y0) / max(1.0, float(h))
        with _autocast_ctx():
            state = processor.add_geometric_prompt(box=[cx, cy, bw, bh], label=True, state=state)
        return _extract_binary_masks_from_state(state, target_shape=target_shape)
