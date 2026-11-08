"""Forge extension entry point registering the standalone 'SAM3 Mask' tab
and capturing `img2img` Inpaint upload components (`img_inpaint_base` / `img_inpaint_mask`).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

EXT_ROOT = Path(__file__).resolve().parent
if str(EXT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXT_ROOT))

_INPAINT_BASE: Any = None
_INPAINT_MASK: Any = None
_REGISTERED: bool = False


def _ensure_local_ui_package() -> Any:
    """Ensure `ui.tab` resolves to this extension's `ui/` package even if another
    extension (e.g. `sd-webui-style-organizer`) also has a top-level `ui` package.
    """
    ext_root_str = str(EXT_ROOT)
    if sys.path[:1] != [ext_root_str]:
        if ext_root_str in sys.path:
            sys.path.remove(ext_root_str)
        sys.path.insert(0, ext_root_str)

    existing_ui = sys.modules.get("ui")
    if existing_ui is not None:
        mod_file = getattr(existing_ui, "__file__", None) or ""
        try:
            is_ours = Path(mod_file).resolve().is_relative_to(EXT_ROOT)
        except Exception:
            is_ours = False
        if not is_ours:
            for k in list(sys.modules.keys()):
                if k == "ui" or k.startswith("ui."):
                    sys.modules.pop(k, None)

    from ui import tab as ui_tab

    return ui_tab


def on_after_component(component: Any, **kwargs: Any) -> None:
    global _INPAINT_BASE, _INPAINT_MASK
    elem_id = getattr(component, "elem_id", None) or kwargs.get("elem_id") or ""
    if elem_id == "img_inpaint_base":
        _INPAINT_BASE = component
    elif elem_id == "img_inpaint_mask":
        _INPAINT_MASK = component


def on_ui_tabs():
    ui_tab = _ensure_local_ui_package()
    tab_blocks = ui_tab.build_tab(
        inpaint_base_comp=_INPAINT_BASE,
        inpaint_mask_comp=_INPAINT_MASK,
    )
    return [(tab_blocks, "SAM3 Mask", "sam3_mask_studio_tab")]


def register_callbacks() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    try:
        from modules import script_callbacks

        script_callbacks.on_after_component(on_after_component)
        script_callbacks.on_ui_tabs(on_ui_tabs)
        _REGISTERED = True
    except Exception:
        # Allows importing `script.py` in unit tests outside WebUI runtime
        pass


register_callbacks()
