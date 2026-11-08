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
