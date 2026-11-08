"""Preset definitions, localization mapping, and git-safe persistence (spec §5.2 & §9.5).

Default presets live in `presets.json` (tracked by Git, never modified at runtime).
User customizations are saved to `user_presets.json` (git-ignored) so `git pull`
updates never conflict with user-created presets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from i18n_helper import t

_EXT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRESETS_PATH = _EXT_ROOT / "presets.json"
USER_PRESETS_PATH = _EXT_ROOT / "user_presets.json"

DEFAULT_PRESETS: list[dict[str, Any]] = [
    {"name": "full_body", "positive": ["person"], "negative": []},
    {"name": "face", "positive": ["face"], "negative": ["hair"]},
    {"name": "hair", "positive": ["hair"], "negative": ["face", "skin"]},
    {"name": "background", "positive": ["background"], "negative": ["person"]},
]


def _normalize_preset(entry: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        return None
    name = str(entry.get("name", "")).strip()
    if not name:
        return None
    pos_raw = entry.get("positive", [])
    neg_raw = entry.get("negative", [])
    pos = [str(p).strip() for p in (pos_raw if isinstance(pos_raw, list) else [pos_raw]) if str(p).strip()]
    neg = [str(n).strip() for n in (neg_raw if isinstance(neg_raw, list) else [neg_raw]) if str(n).strip()]
    return {"name": name, "positive": pos, "negative": neg}


def _read_preset_file(file_path: Path) -> list[dict[str, Any]] | None:
    if not file_path.is_file():
        return None
    try:
        with open(file_path, encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            return None
        out = [norm for item in raw if (norm := _normalize_preset(item)) is not None]
        return out if out else None
    except Exception:
        return None


def load_presets(path: Path | None = None) -> list[dict[str, Any]]:
    """Load presets from `user_presets.json` (if present), else `presets.json`, else `DEFAULT_PRESETS`."""
    if path is not None:
        loaded = _read_preset_file(path)
        if loaded is not None:
            return loaded
        save_presets(DEFAULT_PRESETS, path)
        return [dict(p) for p in DEFAULT_PRESETS]

    user_loaded = _read_preset_file(USER_PRESETS_PATH)
    if user_loaded is not None:
        return user_loaded

    default_loaded = _read_preset_file(DEFAULT_PRESETS_PATH)
    if default_loaded is not None:
        return default_loaded

    return [dict(p) for p in DEFAULT_PRESETS]


def save_presets(presets: list[dict[str, Any]], path: Path | None = None) -> None:
    """Persist presets to `user_presets.json` by default so tracked `presets.json` stays clean."""
    target = path if path is not None else USER_PRESETS_PATH
    cleaned = [norm for item in presets if (norm := _normalize_preset(item)) is not None]
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)


def get_preset_display_name(name: str) -> str:
    """Return localized display name for preset ID (`preset.<name>`), or `name` for custom presets."""
    key = f"preset.{name}"
    translated = t(key)
    return name if translated == key else translated


def get_dropdown_choices(path: Path | None = None) -> list[tuple[str, str]]:
    """Return `(display_label, preset_id)` tuples for Gradio Dropdown choices."""
    presets = load_presets(path)
    choices: list[tuple[str, str]] = []
    for p in presets:
        pid = p["name"]
        disp = get_preset_display_name(pid)
        choices.append((disp, pid))
    return choices


def find_preset(name: str, path: Path | None = None) -> dict[str, Any] | None:
    """Look up a preset by its `name` ID."""
    target_name = (name or "").strip()
    if not target_name:
        return None
    for p in load_presets(path):
        if p["name"] == target_name:
            return p
    return None


def upsert_preset(
    name: str,
    positive: list[str] | str,
    negative: list[str] | str,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Add or update a preset by `name` and persist to `user_presets.json` (or `path`)."""
    if isinstance(positive, str):
        pos_list = [s.strip() for s in positive.split(",") if s.strip()]
    else:
        pos_list = [str(s).strip() for s in positive if str(s).strip()]

    if isinstance(negative, str):
        neg_list = [s.strip() for s in negative.split(",") if s.strip()]
    else:
        neg_list = [str(s).strip() for s in negative if str(s).strip()]

    entry = _normalize_preset({"name": name, "positive": pos_list, "negative": neg_list})
    if entry is None:
        raise ValueError("Preset name must not be empty.")

    presets = load_presets(path)
    replaced = False
    for idx, existing in enumerate(presets):
        if existing["name"] == entry["name"]:
            presets[idx] = entry
            replaced = True
            break
    if not replaced:
        presets.append(entry)
    save_presets(presets, path)
    return presets


def delete_preset(name: str, path: Path | None = None) -> list[dict[str, Any]]:
    """Delete a preset by `name` and persist to `user_presets.json` (or `path`)."""
    target_name = (name or "").strip()
    presets = [p for p in load_presets(path) if p["name"] != target_name]
    save_presets(presets, path)
    return presets
