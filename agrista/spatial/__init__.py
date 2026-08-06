"""
Agrista Spatial Module — Toprak Bilimi Mekânsal Analiz
Yarıvaryogram, mekânsal bağımlılık ve IDW enterpolasyonu.

Literatür dayanağı: docs/01_ALT_BRANS_ISTATISTIK_LITERATUR_LOG.md (Bölüm 6)
Not: Tam kriging için ileride opsiyonel pykrige entegrasyonu planlanıyor.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def semivariogram(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    n_lags: int = 10,
    max_distance: Optional[float] = None,
) -> dict:
    """Deneysel yarıvaryogram hesabı.

    gamma(h) = 1/(2N(h)) * Σ (z(xi) - z(xj))² — mesafe sınıflarına göre.
    """
    x = np.asarray(x, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()
    values = np.asarray(values, dtype=float).flatten()

    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(values))
    x, y, values = x[mask], y[mask], values[mask]
    n = len(x)
    if n < 10:
        raise ValueError("Yarıvaryogram için en az 10 örnekleme noktası önerilir")

    # İkili mesafeler ve yarı fark kareleri
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx ** 2 + dy ** 2)
    sq_diff = (values[:, None] - values[None, :]) ** 2

    iu = np.triu_indices(n, k=1)
    distances = dist[iu]
    sq_diffs = sq_diff[iu]

    if max_distance is None:
        max_distance = float(np.max(distances) / 2)

    lag_edges = np.linspace(0, max_distance, n_lags + 1)
    lags, gammas, pair_counts = [], [], []
    for i in range(n_lags):
        mask_lag = (distances > lag_edges[i]) & (distances <= lag_edges[i + 1])
        n_pairs = int(mask_lag.sum())
        if n_pairs > 0:
            lags.append(float((lag_edges[i] + lag_edges[i + 1]) / 2))
            gammas.append(float(0.5 * np.mean(sq_diffs[mask_lag])))
            pair_counts.append(n_pairs)

    if not lags:
        raise ValueError("Hiçbir mesafe sınıfında ikili bulunamadı")

    gammas_arr = np.array(gammas)
    sill = float(np.var(values, ddof=1))
    nugget = float(gammas_arr[0])
    spatial_dependence = (sill - nugget) / sill if sill > 0 else 0.0

    return {
        "lags": lags,
        "semivariance": gammas,
        "pair_counts": pair_counts,
        "sill": sill,
        "nugget": nugget,
        "spatial_dependence_ratio": float(spatial_dependence),
        "interpretation": (
            "Güçlü mekânsal bağımlılık" if spatial_dependence > 0.75
            else "Orta mekânsal bağımlılık" if spatial_dependence > 0.25
            else "Zayıf mekânsal bağımlılık"
        ),
        "n_points": n,
    }


def idw_interpolation(
    known_x: np.ndarray,
    known_y: np.ndarray,
    known_values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    power: float = 2.0,
) -> np.ndarray:
    """Ters mesafe ağırlıklı (IDW) enterpolasyon.

    grid_x/grid_y: 1D eksen dizileri veya eş boyutlu 2D ızgaralar.
    Döndürülen değer grid_x/grid_y ile aynı şekildedir (2D verilirse).
    """
    kx = np.asarray(known_x, dtype=float).flatten()
    ky = np.asarray(known_y, dtype=float).flatten()
    kv = np.asarray(known_values, dtype=float).flatten()
    mask = ~(np.isnan(kx) | np.isnan(ky) | np.isnan(kv))
    kx, ky, kv = kx[mask], ky[mask], kv[mask]
    if len(kv) == 0:
        raise ValueError("Bilinen örnekleme noktası yok")

    gx = np.asarray(grid_x, dtype=float)
    gy = np.asarray(grid_y, dtype=float)

    if gx.ndim == 1 and gy.ndim == 1:
        gx, gy = np.meshgrid(gx, gy)

    if gx.shape != gy.shape:
        raise ValueError("Izgara boyutları uyumsuz")

    result = np.empty(gx.shape, dtype=float)
    for i in range(gx.shape[0]):
        for j in range(gx.shape[1]):
            dist = np.sqrt((kx - gx[i, j]) ** 2 + (ky - gy[i, j]) ** 2)
            exact = dist < 1e-12
            if np.any(exact):
                result[i, j] = kv[exact][0]
                continue
            weights = 1.0 / dist ** power
            result[i, j] = np.sum(weights * kv) / np.sum(weights)

    return result


def spatial_summary(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    n_lags: int = 10,
) -> dict:
    """Toprak örneklemesi için birleşik mekânsal özet: klasik istatistik + varyogram."""
    values = np.asarray(values, dtype=float).flatten()
    valid = values[~np.isnan(values)]
    if len(valid) < 3:
        raise ValueError("En az 3 geçerli değer gerekli")

    variogram = semivariogram(x, y, values, n_lags=n_lags)

    return {
        "n_points": int(len(valid)),
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid, ddof=1)),
        "cv_pct": float(np.std(valid, ddof=1) / abs(np.mean(valid)) * 100) if np.mean(valid) != 0 else float("inf"),
        "variogram": variogram,
    }
