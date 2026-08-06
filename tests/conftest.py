"""Agrista test suite ortak yardımcılar."""

from __future__ import annotations

import sys
from pathlib import Path

# examples modülünü içe aktarılabilir yap
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd


def make_sample_df(n: int = 100, seed: int = 7) -> pd.DataFrame:
    """Regresyon/test için deterministik örnek veri üretir."""
    rng = np.random.default_rng(seed)
    sulama = rng.uniform(3000, 12000, n)
    gubre = rng.uniform(50, 400, n)
    verim = 2.0 + 0.001 * sulama + 0.005 * gubre + rng.normal(0, 0.3, n)
    return pd.DataFrame({"sulama": sulama, "gubre": gubre, "verim": verim})
