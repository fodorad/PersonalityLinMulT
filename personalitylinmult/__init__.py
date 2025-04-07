from importlib.metadata import version

try:
    __version__ = version("personalitylinmult")
except Exception:
    __version__ = "unknown"