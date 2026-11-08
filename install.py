"""Forge Neo extension installer for SAM3 Mask Studio.

Checks Python / PyTorch compatibility, ensures `sam3` and `huggingface_hub` are
available without mutating Forge Neo's core PyTorch/CUDA packages, and checks
whether a SAM3 checkpoint exists locally (never auto-downloads gated weights).
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

MIN_PYTHON = (3, 12)
MIN_TORCH = (2, 7)
SAM3_PIP_SPEC = "git+https://github.com/facebookresearch/sam3.git"
LOG_PREFIX = "[SAM3 Mask Studio]"


def _log(msg: str) -> None:
    print(f"{LOG_PREFIX} {msg}", flush=True)


def _parse_version(v: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in v.split("."):
        digits = ""
        for ch in piece:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) if parts else (0,)


def _at_least(version: str, minimum: tuple[int, ...]) -> bool:
    got = _parse_version(version)
    need = minimum + (0,) * (len(got) - len(minimum))
    return got >= need


def check_python() -> str | None:
    if sys.version_info >= MIN_PYTHON:
        return None
    return (
        f"Python {sys.version_info.major}.{sys.version_info.minor} is below the recommended "
        f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ for SAM3."
    )


def check_torch() -> str | None:
    try:
        import torch
    except Exception:
        return "PyTorch is not installed in the current environment."
    version = getattr(torch, "__version__", "") or ""
    if not _at_least(version, MIN_TORCH):
        return (
            f"PyTorch {version or 'unknown'} is below {MIN_TORCH[0]}.{MIN_TORCH[1]}+ "
            "required by SAM3."
        )
    return None


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _ensure_huggingface_hub() -> None:
    if _has_module("huggingface_hub"):
        return
    _log("Installing huggingface_hub...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    except Exception as exc:
        _log(f"Warning: failed to install huggingface_hub ({exc})")


def _ensure_sam3() -> None:
    if _has_module("sam3"):
        try:
            ver = importlib.metadata.version("sam3")
            _log(f"sam3 package already installed (v{ver})")
        except Exception:
            _log("sam3 module already available")
        return

    for err in (check_python(), check_torch()):
        if err:
            _log(f"Skipping automatic sam3 install: {err}")
            _log(f"Manual install command: pip install --no-deps {SAM3_PIP_SPEC}")
            return

    _log("Installing sam3 (--no-deps to preserve Forge Neo PyTorch/CUDA environment)...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--no-deps", SAM3_PIP_SPEC]
        )
        _log("sam3 installation completed.")
    except Exception as exc:
        _log(f"sam3 install failed (UI will still load): {exc}")
        _log(f"Manual install command: pip install --no-deps {SAM3_PIP_SPEC}")


def check_model_exists(base_models_dir: Path | None = None) -> bool:
    """Check if any SAM3 checkpoint exists in models/sam3 (or models/SAM3).

    Per BYOT policy, this only checks existence and NEVER downloads weights.
    """
    if base_models_dir is None:
        # extensions/Forge-Neo-SAM3-Mask-Studio/install.py -> ../../models
        base_models_dir = Path(__file__).resolve().parents[2] / "models"
    for sub in ("sam3", "SAM3"):
        d = base_models_dir / sub
        if d.is_dir():
            for p in d.iterdir():
                if p.is_file() and p.suffix.lower() in {".pt", ".pth", ".safetensors"}:
                    return True
    return False


def main() -> None:
    if os.environ.get("SAM3_MASK_STUDIO_SKIP_INSTALL") or "--skip-install" in sys.argv:
        _log("Install check skipped.")
        return

    _ensure_huggingface_hub()
    _ensure_sam3()

    if not check_model_exists():
        _log(
            "No SAM3 checkpoint found in models/sam3/. "
            "Use the 'Model Setup' section in the SAM3 Mask tab to configure your HF token and download, "
            "or place sam3.pt manually in models/sam3/."
        )


if __name__ == "__main__":
    main()
