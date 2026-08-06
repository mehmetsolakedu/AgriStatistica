"""
Agrista Engineering Module — Tarım Makineleri
Yanıt yüzey yöntemi (RSM), Taguchi ve deney tasarımı optimizasyonları.

Literatür dayanağı: docs/01_ALT_BRANS_ISTATISTIK_LITERATUR_LOG.md (Bölüm 5)
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Yanıt Yüzey Yöntemi (RSM)
# ---------------------------------------------------------------------------


def rsm_ccd(factors: dict, alpha: float = "rotatable", n_center: int = 3) -> pd.DataFrame:
    """Central Composite Design (CCD) üretici.

    factors: {"faktör": (alt, üst)} — gerçek alt/üst sınırlar
    2^k faktöriyel + aksiyel (±alpha) + merkez noktaları döndürülür.
    Kodlu (-1..+1) ve gerçek değerler birlikte verilir.
    """
    names = list(factors.keys())
    k = len(names)
    if k < 2 or k > 5:
        raise ValueError("CCD için 2-5 arası faktör desteklenir")

    if alpha == "rotatable":
        alpha_val = float(2 ** (k / 4))
    else:
        alpha_val = float(alpha)

    rows = []

    def add(coded, point_type):
        row = {"tip": point_type}
        for i, name in enumerate(names):
            low, high = factors[name]
            center, half = (high + low) / 2, (high - low) / 2
            row[f"{name}_kodlu"] = float(coded[i])
            row[name] = float(center + coded[i] * half)
        rows.append(row)

    for combo in itertools.product([-1, 1], repeat=k):
        add(combo, "faktöriyel")
    for axis in range(k):
        for sign in (-1, 1):
            point = [0] * k
            point[axis] = sign * alpha_val
            add(point, "aksiyel")
    for _ in range(n_center):
        add([0] * k, "merkez")

    df = pd.DataFrame(rows)
    df.insert(0, "deney_no", range(1, len(df) + 1))
    df.attrs["alpha"] = alpha_val
    return df


def rsm_bbd(factors: dict, n_center: int = 3) -> pd.DataFrame:
    """Box-Behnken Design (BBD) üretici (genel yapı: tüm ikili ±1 kombinasyonları)."""
    names = list(factors.keys())
    k = len(names)
    if k < 3 or k > 6:
        raise ValueError("BBD için 3-6 arası faktör desteklenir")

    rows = []

    def add(coded):
        row = {}
        for i, name in enumerate(names):
            low, high = factors[name]
            center, half = (high + low) / 2, (high - low) / 2
            row[f"{name}_kodlu"] = float(coded[i])
            row[name] = float(center + coded[i] * half)
        rows.append(row)

    for pair in itertools.combinations(range(k), 2):
        for signs in itertools.product([-1, 1], repeat=2):
            point = [0] * k
            point[pair[0]] = signs[0]
            point[pair[1]] = signs[1]
            add(point)
    for _ in range(n_center):
        add([0] * k)

    df = pd.DataFrame(rows)
    df.insert(0, "deney_no", range(1, len(df) + 1))
    return df


def rsm_fit(data: pd.DataFrame, response_col: str, factors: list) -> dict:
    """Kuadratik yanıt yüzey modeli uydurma (kodlu veya gerçek değerlerle).

    Doğrusal + ikili etkileşim + kare terimleri OLS ile tahmin edilir.
    """
    work = data[[response_col] + list(factors)].dropna()
    if len(work) < len(factors) * 2 + 3:
        raise ValueError("Kuadratik model için yeterli deneme noktası yok")

    X = pd.DataFrame(index=work.index)
    for f in factors:
        X[f] = work[f].astype(float)
    for fa, fb in itertools.combinations(factors, 2):
        X[f"{fa}*{fb}"] = work[fa] * work[fb]
    for f in factors:
        X[f"{f}^2"] = work[f] ** 2

    model = sm.OLS(work[response_col].astype(float), sm.add_constant(X)).fit()

    return {
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "model_p_value": float(model.f_pvalue),
        "coefficients": {k_: float(v) for k_, v in model.params.items()},
        "p_values": {k_: float(v) for k_, v in model.pvalues.items()},
        "significant_terms": [k_ for k_, v in model.pvalues.items() if v < 0.05],
        "n_obs": int(model.nobs),
        "_model": model,
    }


def find_optimum(result: dict, factors: dict, maximize: bool = True) -> dict:
    """RSM modelinden optimum faktör ayarlarını bulur (kodlu sınırlar içinde).

    factors: {"faktör": (alt, üst)} — arama sınırı olarak kullanılır.
    """
    model = result["_model"]
    names = list(factors.keys())

    def predict(x_arr):
        row = {"const": 1.0}
        for i, f in enumerate(names):
            row[f] = x_arr[i]
        for (fa, fb) in itertools.combinations(names, 2):
            row[f"{fa}*{fb}"] = x_arr[names.index(fa)] * x_arr[names.index(fb)]
        for f in names:
            row[f"{f}^2"] = x_arr[names.index(f)] ** 2
        aligned = [row.get(term, 0.0) for term in model.model.exog_names]
        return float(np.dot(model.params, aligned))

    bounds = [(low, high) for low, high in factors.values()]
    x0 = np.array([(lo + hi) / 2 for lo, hi in bounds])
    objective = (lambda x: -predict(x)) if maximize else predict

    opt = minimize(objective, x0, bounds=bounds, method="L-BFGS-B")
    if not opt.success:
        raise ValueError(f"Optimizasyon yakınsamadı: {opt.message}")

    return {
        "optimal_values": {name: float(v) for name, v in zip(names, opt.x)},
        "predicted_response": float(-opt.fun if maximize else opt.fun),
        "maximize": bool(maximize),
    }


# ---------------------------------------------------------------------------
# Taguchi yöntemi
# ---------------------------------------------------------------------------

_L9 = np.array([
    [1, 1, 1, 1],
    [1, 2, 2, 2],
    [1, 3, 3, 3],
    [2, 1, 2, 3],
    [2, 2, 3, 1],
    [2, 3, 1, 2],
    [3, 1, 3, 2],
    [3, 2, 1, 3],
    [3, 3, 2, 1],
])


def taguchi_design(factors: dict) -> pd.DataFrame:
    """Taguchi L9 (3 seviye) ortogonal dizisi ile deney tasarımı.

    factors: {"faktör": [seviye1, seviye2, seviye3]} — en fazla 4 faktör.
    """
    names = list(factors.keys())
    if len(names) < 2 or len(names) > 4:
        raise ValueError("L9 tasarımı 2-4 faktör destekler")
    for name, levels in factors.items():
        if len(levels) != 3:
            raise ValueError(f"'{name}' için tam 3 seviye gerekli (L9)")

    df = pd.DataFrame({"deney_no": range(1, 10)})
    for idx, name in enumerate(names):
        levels = list(factors[name])
        df[name] = [levels[v - 1] for v in _L9[:, idx]]
        df[f"{name}_seviye"] = _L9[:, idx]

    return df


def sn_ratio(values: pd.Series | np.ndarray, goal: str = "nominal") -> float:
    """Taguchi Sinyal/Gürültü oranı.

    goal: 'smaller' (küçüğü hedefle), 'larger' (büyüğü hedefle), 'nominal'
    """
    y = np.asarray(values, dtype=float).flatten()
    y = y[~np.isnan(y)]
    if len(y) == 0:
        raise ValueError("S/N hesabı için geçerli veri yok")

    if goal == "smaller":
        return float(-10 * np.log10(np.mean(y ** 2)))
    elif goal == "larger":
        if np.any(y <= 0):
            raise ValueError("'larger' hedefi için tüm değerler pozitif olmalı")
        return float(-10 * np.log10(np.mean(1.0 / y ** 2)))
    elif goal == "nominal":
        mean, std = np.mean(y), np.std(y, ddof=1)
        if std == 0:
            return float("inf")
        return float(10 * np.log10(mean ** 2 / std ** 2))
    else:
        raise ValueError(f"Desteklenmeyen hedef: {goal}")


def taguchi_analyze(data: pd.DataFrame, response_col: str, factors: list, goal: str = "nominal", n_reps: int = 1) -> dict:
    """Taguchi S/N analizi — faktör seviyesi başına ortalama S/N ve optimum seçim.

    n_reps > 1 ise her deney satırının tekrarlandığı (satır başına n_reps ölçüm
    sütunu) kabul edilir: yanıt sütunları 'yanit_1..yanit_n' adlarında aranır.
    Tek ölçümde (n_reps=1) S/N yerine doğrudan yanıt ortalaması sıralaması yapılır.
    """
    if n_reps == 1:
        work = data[[response_col] + list(factors)].dropna()
        if len(work) == 0:
            raise ValueError("Analiz için geçerli veri yok")
        sn_per_run = work[response_col].to_numpy(dtype=float)
    else:
        rep_cols = [f"yanit_{i+1}" for i in range(n_reps)]
        missing = [c for c in rep_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Tekrar sütunları eksik: {missing}")
        work = data[list(factors) + rep_cols].dropna()
        if len(work) == 0:
            raise ValueError("Analiz için geçerli veri yok")
        sn_per_run = np.array([
            sn_ratio(row[rep_cols].to_numpy(dtype=float), goal=goal)
            for _, row in work.iterrows()
        ])

    work = work.copy()
    work["_sn"] = sn_per_run

    factor_summary = {}
    optimal_levels = {}
    for f in factors:
        level_means = work.groupby(f)["_sn"].mean()
        best_level = level_means.idxmax()
        optimal_levels[str(f)] = best_level
        factor_summary[str(f)] = {
            "level_sn": {str(k_): float(v) for k_, v in level_means.items()},
            "range": float(level_means.max() - level_means.min()),
            "optimal_level": best_level,
        }

    # Etki sıralaması (aralığa göre)
    ranking = sorted(factor_summary.items(), key=lambda kv: kv[1]["range"], reverse=True)

    return {
        "goal": goal,
        "n_runs": int(len(work)),
        "factor_summary": factor_summary,
        "optimal_levels": optimal_levels,
        "effect_ranking": [f for f, _ in ranking],
    }
