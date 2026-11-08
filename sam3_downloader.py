"""BYOT (Bring Your Own Token) access verification and download helper for `facebook/sam3` (spec §7).

Never downloads automatically on startup; only executes `download_sam3` when
explicitly triggered by user action in the UI.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

from i18n_helper import t
import sam3_backend

DEFAULT_REPO_ID = "facebook/sam3"


def _normalize_token(token: str | None) -> str | bool:
    cleaned = (token or "").strip()
    return cleaned if cleaned else True


def check_access(repo_id: str = DEFAULT_REPO_ID, token: str | None = None) -> tuple[str, str]:
    """Verify Hugging Face access to `facebook/sam3`.

    Returns `(status_code, localized_message)` where `status_code` is one of:
    - `"access_granted"`
    - `"gated_not_accepted"`
    - `"not_found"`
    - `"no_token"`
    - `"network_error"`
    """
    tok = _normalize_token(token)
    api = HfApi(token=tok)
    try:
        api.model_info(repo_id, token=tok)
        return "access_granted", t("setup.access_granted")
    except GatedRepoError:
        return "gated_not_accepted", t("setup.gated_not_accepted")
    except RepositoryNotFoundError:
        return "not_found", t("setup.not_found")
    except HfHubHTTPError as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code in (401, 403):
            return "no_token", t("setup.no_token")
        return "network_error", t("setup.network_error", error=str(exc))
    except Exception as exc:
        msg_lower = str(exc).lower()
        if "token" in msg_lower or "401" in msg_lower:
            return "no_token", t("setup.no_token")
        return "network_error", t("setup.network_error", error=str(exc))


def download_sam3(
    target_dir: str | Path | None = None,
    token: str | None = None,
    repo_id: str = DEFAULT_REPO_ID,
) -> tuple[bool, str, Path | None]:
    """Download `facebook/sam3` weights into `target_dir` (`models/sam3` by default).

    Must ONLY be called when the user explicitly clicks the download button.
    """
    dest = Path(target_dir) if target_dir else sam3_backend.primary_models_dir()
    dest.mkdir(parents=True, exist_ok=True)
    tok = _normalize_token(token)

    try:
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(dest),
            token=tok,
        )
        return True, t("setup.download_complete", path=str(downloaded_path)), Path(downloaded_path)
    except GatedRepoError:
        return False, t("setup.gated_not_accepted"), None
    except RepositoryNotFoundError:
        return False, t("setup.not_found"), None
    except Exception as exc:
        return False, t("setup.download_failed", error=str(exc)), None
