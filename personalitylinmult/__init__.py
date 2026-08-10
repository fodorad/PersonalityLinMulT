from importlib.metadata import version
from pathlib import Path

try:
    __version__ = version("personalitylinmult")
except Exception:
    __version__ = "unknown"

# Module level constants
PROJECT_ROOT = Path(__file__).parents[1]
"""Repository root; all config and data paths are relative to it."""

LIBRARY_ROOT = PROJECT_ROOT / "personalitylinmult"
"""The package directory."""

WEIGHTS_DIR = Path().home() / ".cache" / "torch" / "hub" / "checkpoints" / "personality"
"""Cache directory :meth:`PersonalityModel.from_pretrained` downloads
champion checkpoints into (passed as ``huggingface_hub.hf_hub_download``'s
``cache_dir``), so a repeated call for the same model id reuses the local
copy instead of re-downloading."""

CONFIG_DIR = PROJECT_ROOT / "config"
"""Root of the experiment configs (``config/FI/{features,data,model,train}``)."""

from personalitylinmult.model import PersonalityModel  # noqa: E402

__all__ = ["PersonalityModel", "__version__"]
