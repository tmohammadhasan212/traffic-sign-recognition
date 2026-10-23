"""Noninteractive plotting with a writable cache in restricted environments."""

import os
import tempfile
from pathlib import Path

cache = Path(tempfile.gettempdir()) / "traffic-signs-matplotlib"
cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

__all__ = ["plt"]
