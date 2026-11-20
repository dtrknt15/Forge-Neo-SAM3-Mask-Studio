"""Lightweight localization helper for dynamic Python status messages and preset names.

Static UI labels remain in English and are translated by WebUI's standard
`localizations/*.json` DOM replacement mechanism. Dynamic messages returned by
Python callbacks use `t()` with safe English fallback.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_CACHE: dict[str, dict[str, str]] = {}
_LOCALIZATIONS_DIR = Path(__file__).resolve().parent / "localizations"


def get_active_language() -> str:
    """Resolve current WebUI localization setting, defaulting to 'en_US'."""
    try:
        from modules.shared import opts

        lang = getattr(opts, "localization", None)
        if isinstance(lang, str) and lang.strip():
            return lang.strip()
    except Exception:
        pass
    env_lang = os.environ.get("SAM3_MASK_STUDIO_LANG", "").strip()
    return env_lang or "en_US"


def clear_cache() -> None:
    """Clear cached localization dictionaries (useful for tests)."""
    _CACHE.clear()


def _load_lang_dict(lang: str) -> dict[str, str]:
    if lang not in _CACHE:
        path = _LOCALIZATIONS_DIR / f"{lang}.json"
        if path.is_file():
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                _CACHE[lang] = {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}
            except Exception:
                _CACHE[lang] = {}
        else:
            _CACHE[lang] = {}
    return _CACHE[lang]


_EN_DEFAULTS: dict[str, str] = {
    "status.instances_updated": "Updated instances ({kept}/{total} active).",
    "status.refine_point_added": "Added refine point ({mode}: {x}, {y}; {count} pts).",
    "status.refine_mode_on": "Multi-point Object Refine enabled (Left-click=Add, Right-click=Exclude).",
    "status.refine_mode_off": "Multi-point Object Refine disabled.",
    "status.refine_empty": "No active refine points or mask.",
    "status.refine_committed": "Committed refined object mask.",
    "status.refine_point_removed": "Removed last refine point ({count} remaining).",
    "status.refine_cleared": "Cleared refine points.",
    "status.edge_refine_applied": "Applied Guided Edge Refine (radius: {radius}).",
    "status.edge_smooth_applied": "Applied Edge Smooth ({px} px).",
    "status.layers_zip_exported": "Exported layers ZIP: {path} ({count} layers).",
    "status.batch_cancelling": "Batch cancellation requested...",
    "status.batch_complete": "Batch complete: {processed} saved (empty: {empty}, errors: {errors}) -> {out_dir}",
    "status.batch_cancelled": "Batch cancelled: {processed} saved (skipped: {skipped}, empty: {empty}, errors: {errors}) -> {out_dir}",
    "status.batch_complete_resumed": "Batch complete: {processed} saved (skipped existing: {skipped}, empty: {empty}, errors: {errors}) -> {out_dir}",
}


def t(key: str, **kwargs: Any) -> str:
    """Translate `key` using the active language JSON, falling back to English defaults or `key`."""
    lang = get_active_language()
    mapping = _load_lang_dict(lang)
    template = mapping.get(key) or _EN_DEFAULTS.get(key, key)
    if kwargs:
        try:
            return template.format(**kwargs)
        except Exception:
            return template
    return template
