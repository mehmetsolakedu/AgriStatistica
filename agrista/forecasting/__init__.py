"""
Agrista Forecasting Module — Zaman Serisi ve Kestirim (Premium Program: Forecasting)

Hareketli ortalama, üstel yumuşatma, Holt-Winters (additive), mevsimsel
ayrıştırma ve ARIMA tabanlı kestirim. Tüm gerçeklemeler formül bazlıdır;
ARIMA için statsmodels SARIMAX altyapısı kullanılır.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def moving_average(series: pd.Series | np.ndarray, window: int = 3,
                   centered: bool = True) -> np.ndarray:
    """Hareketli ortalama (Premium Program: Transform → Create Time Series).

    centered=True iken ortalanmış (2x) hareketli ortalama uygulanır;
    bu, çift pencerelerde Premium Program'ın merkezleme davranışıyla uyumludur.
    """
    y = np.asarray(series, dtype=float).flatten()
    if window < 2:
        raise ValueError("Pencere genişliği en az 2 olmalı")
    if len(y) < window:
        raise ValueError("Seri, pencere genişliğinden kısa")

    s = pd.Series(y)
    if centered and window % 2 == 0:
        # Çift pencere: iki aşamalı ortalama (Premium Program 'Centered' seçeneği)
        ma = s.rolling(window, min_periods=window).mean()
        ma = ma.rolling(2, min_periods=2).mean()
    else:
        ma = s.rolling(window, center=centered, min_periods=window).mean()
    return ma.to_numpy()


def seasonal_decomposition(series: pd.Series | np.ndarray,
                           period: int, model: str = "additive") -> dict:
    """Klasik mevsimsel ayrıştırma (Premium Program: Forecasting → Seasonal Decomposition).

    Trend: ortalanmış hareketli ortalama.
    Mevsim bileşeni: trendden arındırılmış serinin dönem ortalamaları
    (additive modelde sıfır ortalamalı olacak şekilde merkezlenir).
    """
    y = np.asarray(series, dtype=float).flatten()
    n = len(y)
    if period < 2:
        raise ValueError("Dönem uzunluğu en az 2 olmalı")
    if n < 2 * period:
        raise ValueError("Seri en az 2 tam dönem içermeli")
    if model != "additive":
        raise ValueError("Şu an yalnızca 'additive' model destekleniyor")

    trend = moving_average(y, window=period, centered=True)

    detrended = y - trend
    # Her mevsim konumunun ortalaması (NaN'lar dışarıda)
    seasonal_means = np.array([
        np.nanmean(detrended[i::period]) for i in range(period)
    ])
    seasonal_means -= seasonal_means.mean()  # sıfır ortalamalı normalize
    seasonal = np.tile(seasonal_means, n // period + 1)[:n]

    resid = y - trend - seasonal
    resid = np.where(np.isnan(trend), np.nan, resid)

    return {
        "model": model,
        "period": int(period),
        "trend": trend,
        "seasonal": seasonal,
        "seasonal_indices": seasonal_means.tolist(),
        "residual": resid,
        "residual_std": float(np.nanstd(resid, ddof=1)),
        "n": int(n),
    }


def exponential_smoothing(series: pd.Series | np.ndarray,
                          alpha: float = 0.3) -> dict:
    """Basit üstel yumuşatma (SES).

    l_t = alpha * y_t + (1 - alpha) * l_{t-1}
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha (0, 1] aralığında olmalı")
    y = np.asarray(series, dtype=float).flatten()
    if len(y) < 2:
        raise ValueError("En az 2 gözlem gerekli")

    fitted = np.empty_like(y)
    fitted[0] = y[0]
    level = y[0]
    for t in range(1, len(y)):
        level = alpha * y[t] + (1 - alpha) * level
        fitted[t] = level

    residual = y - fitted
    return {
        "model": "SES",
        "alpha": float(alpha),
        "fitted": fitted,
        "last_level": float(level),
        "next_forecast": float(level),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "n": int(len(y)),
    }


def holt_winters(series: pd.Series | np.ndarray, period: int,
                 alpha: float = 0.3, beta: float = 0.1, gamma: float = 0.1,
                 horizon: int = 1) -> dict:
    """Holt-Winters additive mevsimsel yumuşatma + h-adım kestirim.

    l_t = α(y_t − s_{t−m}) + (1−α)(l_{t−1} + b_{t−1})
    b_t = β(l_t − l_{t−1}) + (1−β)b_{t−1}
    s_t = γ(y_t − l_t) + (1−γ)s_{t−m}
    """
    if not 0.0 < alpha <= 1.0 or not 0.0 <= beta <= 1.0 or not 0.0 <= gamma <= 1.0:
        raise ValueError("Parametreler geçerli aralıkta olmalı (alpha>0)")
    if period < 2:
        raise ValueError("Dönem uzunluğu en az 2 olmalı")
    if horizon < 1:
        raise ValueError("Kestirim ufku en az 1 olmalı")

    y = np.asarray(series, dtype=float).flatten()
    n = len(y)
    if n < 2 * period:
        raise ValueError("Holt-Winters için en az 2 tam dönem gerekli")

    # Başlangıç değerleri: ilk dönem ortalaması, ilk iki dönem farkından eğim
    level = float(np.mean(y[:period]))
    trend = float((np.mean(y[period:2 * period]) - np.mean(y[:period])) / period)
    seasonals = [float(y[i] - np.mean(y[:period])) for i in range(period)]

    fitted = np.empty(n)
    for t in range(n):
        s_prev = seasonals[t % period]
        if t < period:
            fitted[t] = level + trend + s_prev
            continue
        new_level = alpha * (y[t] - s_prev) + (1 - alpha) * (level + trend)
        new_trend = beta * (new_level - level) + (1 - beta) * trend
        seasonals[t % period] = gamma * (y[t] - new_level) + (1 - gamma) * s_prev
        level, trend = new_level, new_trend
        fitted[t] = level + trend + s_prev

    forecasts = np.array([
        level + h * trend + seasonals[(n + h - 1) % period]
        for h in range(1, horizon + 1)
    ])

    residual = y - fitted
    return {
        "model": "Holt-Winters (additive)",
        "period": int(period),
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "final_level": float(level),
        "final_trend": float(trend),
        "seasonal_components": seasonals,
        "fitted": fitted,
        "forecasts": forecasts,
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "n": int(n),
    }


def arima_forecast(series: pd.Series | np.ndarray, order: tuple = (1, 1, 1),
                   horizon: int = 5) -> dict:
    """ARIMA(p,d,q) modeliyle kestirim (Premium Program: Forecasting → Create ARIMA).

    Parametre tahmini statsmodels ARIMA (tam olabilirlik) ile yapılır;
    güven aralıkları model tabanlıdır.
    """
    from statsmodels.tsa.arima.model import ARIMA

    y = np.asarray(series, dtype=float).flatten()
    if len(y) < 10:
        raise ValueError("ARIMA için en az 10 gözlem önerilir")
    if horizon < 1:
        raise ValueError("Kestirim ufku en az 1 olmalı")

    model = ARIMA(y, order=tuple(order)).fit()
    fc = model.get_forecast(steps=horizon)
    mean_fc = np.asarray(fc.predicted_mean, dtype=float)
    ci = fc.conf_int(alpha=0.05)
    ci = np.asarray(ci, dtype=float)

    fitted = np.asarray(model.fittedvalues, dtype=float)
    residual = y - fitted

    params_arr = np.asarray(model.params, dtype=float)
    param_names = getattr(model, "param_names", None) or [f"p{i}" for i in range(len(params_arr))]

    return {
        "model": f"ARIMA{tuple(order)}",
        "aic": float(model.aic),
        "bic": float(model.bic),
        "params": {str(k): float(v) for k, v in zip(param_names, params_arr)},
        "forecasts": mean_fc.tolist(),
        "ci95_lower": ci[:, 0].tolist(),
        "ci95_upper": ci[:, 1].tolist(),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "n": int(len(y)),
    }
