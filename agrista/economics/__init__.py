"""
Agrista Economics Module — Tarım Ekonomisi
Etkinlik analizi (DEA) ve teknoloji benimseme (logit/probit) modelleri.

Literatür dayanağı: docs/01_ALT_BRANS_ISTATISTIK_LITERATUR_LOG.md (Bölüm 7)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import linprog


def dea_efficiency(
    inputs: pd.DataFrame,
    outputs: pd.DataFrame,
    model: str = "CCR",
    orientation: str = "input",
) -> dict:
    """Veri Zarflama Analizi (DEA) — CCR veya BCC, girdi yönelimli.

    inputs: DMU × girdiler matrisi (index = işletme/DMU adı)
    outputs: DMU × çıktılar matrisi (aynı index)
    Doğrusal programlama (scipy.optimize.linprog) ile çözülür.
    """
    if not inputs.index.equals(outputs.index):
        raise ValueError("Girdi ve çıktı tablolarının indexleri aynı olmalı")
    if orientation != "input":
        raise ValueError("Şu an yalnızca girdi yönelimli (input) DEA destekleniyor")

    X = inputs.to_numpy(dtype=float)
    Y = outputs.to_numpy(dtype=float)
    n, m = X.shape
    s = Y.shape[1]
    if n < 2 or m < 1 or s < 1:
        raise ValueError("DEA için en az 2 DMU, 1 girdi ve 1 çıktı gerekli")

    names = [str(i) for i in inputs.index]
    efficiencies = {}

    for o in range(n):
        # Değişkenler: [theta, lambda_1..lambda_n]
        c = np.zeros(n + 1)
        c[0] = 1.0  # minimize theta

        # Girdi kısıtları: Σ λ_j x_ij ≤ θ x_io  →  Σ λ_j x_ij - θ x_io ≤ 0
        A_in = np.zeros((m, n + 1))
        A_in[:, 1:] = X.T
        A_in[:, 0] = -X[o]
        b_in = np.zeros(m)

        # Çıktı kısıtları: Σ λ_j y_rj ≥ y_ro  →  -Σ λ_j y_rj ≤ -y_ro
        A_out = np.zeros((s, n + 1))
        A_out[:, 1:] = -Y.T
        b_out = -Y[o]

        A_ub = np.vstack([A_in, A_out])
        b_ub = np.concatenate([b_in, b_out])

        bounds = [(None, None)] + [(0, None)] * n

        # BCC: Σ λ = 1
        A_eq, b_eq = None, None
        if model == "BCC":
            A_eq = np.zeros((1, n + 1))
            A_eq[0, 1:] = 1.0
            b_eq = [1.0]
        elif model != "CCR":
            raise ValueError(f"Desteklenmeyen DEA modeli: {model}")

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
        if res.status != 0:
            raise ValueError(f"DMU '{names[o]}' için LP çözülemedi: {res.message}")
        efficiencies[names[o]] = float(min(max(res.x[0], 0.0), 1.0 + 1e-9))

    eff_array = np.array(list(efficiencies.values()))
    efficient_dmus = [name for name, e in efficiencies.items() if e >= 0.999]

    return {
        "model": model,
        "orientation": orientation,
        "efficiencies": efficiencies,
        "mean_efficiency": float(np.mean(eff_array)),
        "efficient_dmus": efficient_dmus,
        "n_dmu": n,
        "n_inputs": m,
        "n_outputs": s,
    }


def adoption_logit(
    data: pd.DataFrame,
    dependent_col: str,
    predictors: list,
    link: str = "logit",
) -> dict:
    """Teknoloji benimseme modeli — ikili yanıt (0/1) için Logit veya Probit.

    Katsayılar, olabilirlik oranı testi ve odds oranları döndürülür.
    """
    cols = [dependent_col] + list(predictors)
    valid = data[cols].dropna()
    if len(valid) < len(predictors) + 5:
        raise ValueError("İkili yanıt modeli için yeterli veri yok")
    if not valid[dependent_col].isin([0, 1]).all():
        raise ValueError("Bağımlı değişken yalnızca 0/1 değerleri almalı")
    if valid[dependent_col].nunique() < 2:
        raise ValueError("Bağımlı değişkende her iki sonuç da gözlenmeli")

    y = valid[dependent_col].astype(float)
    X = sm.add_constant(valid[predictors].astype(float))

    model_cls = sm.Logit if link == "logit" else sm.Probit if link == "probit" else None
    if model_cls is None:
        raise ValueError(f"Desteklenmeyen bağlantı: {link} (logit/probit)")

    model = model_cls(y, X).fit(disp=0)

    odds_ratios = {
        k_: float(np.exp(v)) for k_, v in model.params.items() if k_ != "const"
    } if link == "logit" else {}

    # Sınıflandırma başarısı (0.5 eşik)
    predicted = (model.predict(X) >= 0.5).astype(int)
    accuracy = float((predicted == y.to_numpy()).mean())

    return {
        "model": link,
        "pseudo_r_squared": float(model.prsquared),
        "llf": float(model.llf),
        "aic": float(model.aic),
        "coefficients": {k_: float(v) for k_, v in model.params.items()},
        "p_values": {k_: float(v) for k_, v in model.pvalues.items()},
        "odds_ratios": odds_ratios,
        "significant_predictors": [k_ for k_, v in model.pvalues.items() if v < 0.05 and k_ != "const"],
        "classification_accuracy": accuracy,
        "n_obs": int(len(valid)),
        "adoption_rate": float(y.mean()),
    }


def partial_budget(
    added_revenue: float,
    reduced_costs: float,
    added_costs: float,
    reduced_revenue: float = 0.0,
) -> dict:
    """Kısmi bütçe analizi — teknoloji değişikliğinin net kârlılığı."""
    total_benefits = added_revenue + reduced_costs
    total_costs = added_costs + reduced_revenue
    net_change = total_benefits - total_costs

    return {
        "total_benefits": float(total_benefits),
        "total_costs": float(total_costs),
        "net_change": float(net_change),
        "profitable": bool(net_change > 0),
        "benefit_cost_ratio": float(total_benefits / total_costs) if total_costs > 0 else float("inf"),
    }
