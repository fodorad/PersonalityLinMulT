from pathlib import Path
from importlib.metadata import version

try:
    __version__ = version("personalitylinmult")
except Exception:
    __version__ = "unknown"

# Module level constants
PROJECT_ROOT = Path(__file__).parents[1]
WEIGHTS_DIR = Path().home() / '.cache' / 'torch' / 'hub' / 'checkpoints' / 'personality_sentiment'
MODEL_DIR = PROJECT_ROOT / 'personalitylinmult' / 'model'
