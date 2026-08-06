"""
Agrista Animal Module — Zootekni
Karışık modeller, laktasyon eğrileri ve hayvan denemeleri analizleri.

Literatür dayanağı: docs/01_ALT_BRANS_ISTATISTIK_LITERATUR_LOG.md (Bölüm 4)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import statsmodels.formula.api as smf
from typing import Optional

# numpy 2.x uyumluluğu
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def mixed_model(
    data: pd.DataFrame,
    response_col: str,
    fixed_effects: list,
    groups_col: str,
    random_slope: Optional[str] = None,
) -> dict:
    """Doğrusal karışık model (LMM) — tekrarlı ölçüm / hayvan denemeleri.

    groups_col: rastgele etki gruplama değişkeni (örn. hayvan_no, ahır).
    random_slope: verilirse bu değişken için rastgele eğim eklenir.
    """
    cols = [response_col] + list(fixed_effects) + [groups_col]
    if random_slope:
        cols.append(random_slope)
    work = data[cols].dropna()
    if len(work) < len(fixed_effects) + 4:
        raise ValueError("Karışık model için yeterli veri yok")

    formula = f"{response_col} ~ {' + '.join(fixed_effects)}"
    if random_slope:
        model = smf.mixedlm(formula, work, groups=work[groups_col],
                            re_formula=f"~{random_slope}")
    else:
        model = smf.mixedlm(formula, work, groups=work[groups_col])

    fitted = model.fit(reml=True)

    random_effects_variance = {}
    cov_re = np.atleast_2d(np.asarray(fitted.cov_re))
    random_effects_variance["random_intercept"] = float(cov_re[0, 0])

    return {
        "converged": bool(fitted.converged),
        "fixed_effects": {
            k_: {"coefficient": float(v), "p_value": float(fitted.pvalues[k_]),
                 "significant_at_005": bool(fitted.pvalues[k_] < 0.05)}
            for k_, v in fitted.params.items()
        },
        "random_effects_variance": random_effects_variance,
        "residual_variance": float(fitted.scale),
        "aic": float(fitted.aic),
        "bic": float(fitted.bic),
        "n_obs": int(fitted.nobs),
        "n_groups": int(work[groups_col].nunique()),
    }


def wood_model(t, a: float, b: float, c: float):
    """Wood (gamma) laktasyon eğrisi: y = a * t^b * exp(-c*t)."""
    t = np.asarray(t, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return a * np.power(np.where(t > 0, t, 1e-9), b) * np.exp(-c * t)


def fit_wood(time_points, milk_yield) -> dict:
    """Wood laktasyon eğrisi uydurma + pik verim parametreleri."""
    t = np.asarray(time_points, dtype=float).flatten()
    y = np.asarray(milk_yield, dtype=float).flatten()

    mask = ~(np.isnan(t) | np.isnan(y)) & (t > 0) & (y > 0)
    t, y = t[mask], y[mask]
    if len(t) < 5:
        raise ValueError("Wood modeli için en az 5 geçerli kayıt gerekli")

    a0 = float(np.median(y))
    b0, c0 = 0.3, 0.05
    popt, _ = curve_fit(wood_model, t, y, p0=[a0, b0, c0], maxfev=20000)
    a, b, c = popt

    fitted = wood_model(t, *popt)
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    peak_time = b / c if c > 0 else float("nan")
    peak_yield = float(wood_model(peak_time, a, b, c)) if c > 0 else float("nan")

    return {
        "a_scale": float(a),
        "b_ascending": float(b),
        "c_descending": float(c),
        "peak_time": float(peak_time),
        "peak_yield": peak_yield,
        "persistency": float(1.0 / c) if c > 0 else float("nan"),
        "r_squared": float(r_squared),
        "n_obs": int(len(t)),
    }


def lactation_summary(milk_records: pd.Series | np.ndarray, interval_days: float = 30.0) -> dict:
    """Kontrollere göre toplam laktasyon verimi (trapez) ve 305-gün özeti."""
    y = np.asarray(milk_records, dtype=float).flatten()
    y = y[~np.isnan(y)]
    if len(y) < 2:
        raise ValueError("Laktasyon özeti için en az 2 kontrol gerekli")

    days = np.arange(len(y)) * interval_days
    total = float(_trapezoid(y, days))

    day_305 = 305.0
    if days[-1] <= day_305:
        projected_305 = total
    else:
        projected_305 = float(_trapezoid(y[days <= day_305], days[days <= day_305]))

    return {
        "total_yield_estimate": total,
        "projected_305_days": projected_305,
        "peak_yield": float(np.max(y)),
        "peak_interval": float(days[int(np.argmax(y))]),
        "n_controls": int(len(y)),
    }
