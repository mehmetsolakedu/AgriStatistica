"""
Agrista Protection Module — Bitki Koruma
Entomoloji, fitopatoloji ve herboloji için biyodeney ve epidemiyoloji analizleri.

Literatür dayanağı: docs/01_ALT_BRANS_ISTATISTIK_LITERATUR_LOG.md (Bölüm 1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import statsmodels.api as sm


def abbott_efficiency(control_response: float, treated_response: float) -> dict:
    """Abbott formülü ile düzeltilmiş etkinlik.

    E = (C - T) / C × 100
    control_response: kontrol grubu yanıtı (örn. mortalite, %)
    treated_response: uygulama grubu yanıtı (%)
    """
    if control_response <= 0:
        raise ValueError("Kontrol yanıtı sıfırdan büyük olmalı")

    efficacy = (control_response - treated_response) / control_response * 100.0
    return {
        "control_response": float(control_response),
        "treated_response": float(treated_response),
        "efficacy_pct": float(efficacy),
        "interpretation": (
            "Yüksek etkinlik" if efficacy > 80
            else "Orta etkinlik" if efficacy > 50
            else "Düşük etkinlik"
        ),
    }


def probit_dose_response(
    dose: pd.Series | np.ndarray,
    dead: pd.Series | np.ndarray,
    total: pd.Series | np.ndarray,
    log_dose: bool = True,
) -> dict:
    """Probit doz-yanıt analizi (GLM, binom aile, probit bağlantı).

    LC50/LD50/ED50 ve LC90 tahminleri döndürür.
    dose: uygulama dozları
    dead: ölen/yanıt veren birey sayıları
    total: toplam birey sayıları
    """
    dose = np.asarray(dose, dtype=float).flatten()
    dead = np.asarray(dead, dtype=float).flatten()
    total = np.asarray(total, dtype=float).flatten()

    mask = ~(np.isnan(dose) | np.isnan(dead) | np.isnan(total)) & (dose > 0) & (total > 0)
    dose, dead, total = dose[mask], dead[mask], total[mask]
    if len(dose) < 3:
        raise ValueError("Probit analizi için en az 3 geçerli doz seviyesi gerekli")

    x = np.log10(dose) if log_dose else dose
    proportion = dead / total

    X = sm.add_constant(x)
    model = sm.GLM(
        proportion, X,
        family=sm.families.Binomial(link=sm.families.links.Probit()),
        freq_weights=total,
    ).fit()

    b0, b1 = float(model.params[0]), float(model.params[1])
    if b1 == 0:
        raise ValueError("Doz etkisi tahmin edilemedi (eğim sıfır)")

    def effective_dose(p: float) -> float:
        # probit(p) = b0 + b1 * x  →  x = (probit(p) - b0) / b1
        x_p = (stats.norm.ppf(p) - b0) / b1
        return float(10 ** x_p) if log_dose else float(x_p)

    return {
        "model": "Probit GLM",
        "lc50": effective_dose(0.50),
        "lc90": effective_dose(0.90),
        "intercept": b0,
        "slope": b1,
        "slope_p_value": float(model.pvalues[1]),
        "dose_significant": bool(model.pvalues[1] < 0.05),
        "n_doses": int(len(dose)),
        "log_dose_used": bool(log_dose),
    }


def loglogistic_dose_response(
    dose: pd.Series | np.ndarray,
    response: pd.Series | np.ndarray,
) -> dict:
    """3-parametreli log-logistik doz-yanıt modeli (yabancı ot / herbisit standardı).

    y = d / (1 + (x/e)^b)
    d: üst asimptot (kontrol yanıtı), e: GR50/ED50, b: eğim
    """
    dose = np.asarray(dose, dtype=float).flatten()
    response = np.asarray(response, dtype=float).flatten()

    mask = ~(np.isnan(dose) | np.isnan(response)) & (dose >= 0)
    dose, response = dose[mask], response[mask]
    if len(dose) < 5:
        raise ValueError("Log-logistik uyum için en az 5 geçerli veri noktası gerekli")

    # Log-uzay parametrizasyonu: y = d / (1 + exp(b·(ln x − u)))
    # Bu biçim (x/e)^b'nin sayısal taşma sorunlarını ortadan kaldırır.
    log_dose = np.where(dose > 0, np.log(np.where(dose > 0, dose, 1.0)), -30.0)

    def ll3_log(ld, b, u, d):
        with np.errstate(over="ignore"):
            return d / (1.0 + np.exp(b * (ld - u)))

    d0 = float(np.max(response))
    positive_doses = dose[dose > 0]
    if len(positive_doses) == 0:
        raise ValueError("Sıfırdan büyük en az bir doz değeri gerekli")
    
    # Başlangıç tahmini: yanıtın yarıya düştüğü doz (enterpolasyonla)
    order = np.argsort(dose)
    do, ro = dose[order], response[order]
    e_cross = None
    for i in range(len(ro) - 1):
        if (ro[i] - d0 / 2) * (ro[i + 1] - d0 / 2) <= 0 and do[i + 1] > do[i]:
            frac = (ro[i] - d0 / 2) / (ro[i] - ro[i + 1]) if ro[i] != ro[i + 1] else 0.5
            e_cross = float(do[i] + frac * (do[i + 1] - do[i]))
            break
    if e_cross is None or e_cross <= 0:
        e_cross = float(np.median(positive_doses))
    u0_base = float(np.log(e_cross))
    
    # Çoklu başlangıç noktası ızgarası (yerel optimum riski)
    u_bounds = (float(np.log(positive_doses.min())) - 2.0, float(np.log(positive_doses.max())) + 2.0)
    best = None
    for u0 in (u0_base - 1.0, u0_base, u0_base + 1.0):
        for b0 in (0.5, 1.0, 2.0, 4.0):
            try:
                popt, _ = curve_fit(ll3_log, log_dose, response, p0=[b0, u0, d0],
                                    bounds=([0.01, u_bounds[0], 0.0],
                                            [20.0, u_bounds[1], np.inf]),
                                    maxfev=20000)
            except (RuntimeError, ValueError):
                continue
            rss = float(np.sum((response - ll3_log(log_dose, *popt)) ** 2))
            if best is None or rss < best[1]:
                best = (popt, rss)
    if best is None:
        raise ValueError("Log-logistik model hiçbir başlangıç noktasıyla uydurulamadı")
    popt = best[0]
    b_fit, e_fit, d_fit = popt[0], float(np.exp(popt[1])), popt[2]

    fitted = ll3_log(log_dose, *popt)
    ss_res = float(np.sum((response - fitted) ** 2))
    ss_tot = float(np.sum((response - np.mean(response)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "model": "Log-logistic (LL3)",
        "gr50": float(e_fit),
        "slope_b": float(b_fit),
        "upper_asymptote_d": float(d_fit),
        "r_squared": float(r_squared),
        "n_obs": int(len(dose)),
    }


def audpc(times: pd.Series | np.ndarray, severity: pd.Series | np.ndarray) -> dict:
    """AUDPC — Hastalık İlerleme Eğrisi Altındaki Alan (trapez yöntemi).

    times: gözlem zamanları (gün vb.)
    severity: hastalık şiddeti (0-100 % veya 0-1 oran)
    """
    t = np.asarray(times, dtype=float).flatten()
    y = np.asarray(severity, dtype=float).flatten()

    mask = ~(np.isnan(t) | np.isnan(y))
    t, y = t[mask], y[mask]
    if len(t) < 2:
        raise ValueError("AUDPC için en az 2 geçerli gözlem gerekli")

    order = np.argsort(t)
    t, y = t[order], y[order]

    # Trapez toplamı
    area = float(np.sum((y[:-1] + y[1:]) / 2.0 * np.diff(t)))
    total_time = float(t[-1] - t[0])
    relative_audpc = area / total_time if total_time > 0 else float("nan")

    return {
        "audpc": area,
        "relative_audpc": float(relative_audpc),
        "total_time": total_time,
        "n_observations": int(len(t)),
        "max_severity": float(np.max(y)),
    }


def disease_progress_fit(
    times: pd.Series | np.ndarray,
    severity: pd.Series | np.ndarray,
    model_type: str = "logistic",
) -> dict:
    """Hastalık ilerleme eğrisine doğrusal olmayan model uydurma.

    model_type: 'logistic', 'gompertz' veya 'monomolecular'
    Enfeksiyon oranı (r) ve uyum kalitesi döndürülür.
    """
    t = np.asarray(times, dtype=float).flatten()
    y = np.asarray(severity, dtype=float).flatten()

    mask = ~(np.isnan(t) | np.isnan(y))
    t, y = t[mask], y[mask]
    if len(t) < 4:
        raise ValueError("Eğri uydurma için en az 4 geçerli gözlem gerekli")

    def logistic(x, K, r, t0):
        return K / (1.0 + np.exp(-r * (x - t0)))

    def gompertz(x, K, r, t0):
        return K * np.exp(-np.exp(-r * (x - t0)) + 1)

    def monomolecular(x, K, r):
        return K * (1.0 - np.exp(-r * x))

    models = {
        "logistic": (logistic, [float(np.max(y)) or 1.0, 0.1, float(np.median(t))]),
        "gompertz": (gompertz, [float(np.max(y)) or 1.0, 0.1, float(np.median(t))]),
        "monomolecular": (monomolecular, [float(np.max(y)) or 1.0, 0.05]),
    }

    if model_type not in models:
        raise ValueError(f"Desteklenmeyen model: {model_type}. Seçenekler: {list(models.keys())}")

    func, p0 = models[model_type]
    popt, _ = curve_fit(func, t, y, p0=p0, maxfev=20000)
    fitted = func(t, *popt)

    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    n_params = len(popt)
    n = len(t)
    aic = n * np.log(ss_res / n) + 2 * n_params if ss_res > 0 else float("inf")

    # Enfeksiyon oranı: logistic/gompertz'te r parametresi; monomolecular'da r
    rate = float(popt[1])

    return {
        "model": model_type,
        "parameters": {f"p{i}": float(v) for i, v in enumerate(popt)},
        "rate_r": rate,
        "carrying_capacity_K": float(popt[0]),
        "r_squared": float(r_squared),
        "aic": float(aic),
        "n_obs": n,
    }
