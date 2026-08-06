"""
Agrista Quality Module — Kalite Kontrol (Premium Program: Quality Control)

X̄-R kontrol grafikleri, p-grafiği (kusurlu oranı) ve Pareto analizi.
Kontrol limiti katsayıları (A2, D3, D4) standart SPC tablolarındandır.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# SPC katsayıları: alt grup büyüklüğü n → (A2, D3, D4)
_SPC_CONSTANTS = {
    2: (1.880, 0.0, 3.267),
    3: (1.023, 0.0, 2.574),
    4: (0.729, 0.0, 2.282),
    5: (0.577, 0.0, 2.114),
    6: (0.483, 0.0, 2.004),
    7: (0.419, 0.076, 1.924),
    8: (0.373, 0.136, 1.864),
    9: (0.337, 0.184, 1.816),
    10: (0.308, 0.223, 1.777),
}


def xbar_r_chart(data: pd.Series | np.ndarray, subgroup_size: int = 5) -> dict:
    """X̄-R kontrol grafiği (Premium Program: Quality Control → Control Charts).

    data sıralı üretim/ölçüm değerleri; ardışık subgroup_size'lık
    alt gruplara bölünür. X̄ grafiği limiti: x̿ ± A2·R̄,
    R grafiği limitleri: D3·R̄ ve D4·R̄.
    """
    y = np.asarray(data, dtype=float).flatten()
    y = y[~np.isnan(y)]
    n = len(y)
    if subgroup_size < 2 or subgroup_size > 10:
        raise ValueError("Alt grup büyüklüğü 2-10 aralığında olmalı")
    if n < subgroup_size * 2:
        raise ValueError("En az 2 tam alt grup gerekli")

    n_subgroups = n // subgroup_size
    trimmed = y[: n_subgroups * subgroup_size].reshape(n_subgroups, subgroup_size)

    sub_means = trimmed.mean(axis=1)
    sub_ranges = trimmed.max(axis=1) - trimmed.min(axis=1)

    x_double_bar = float(sub_means.mean())
    r_bar = float(sub_ranges.mean())

    a2, d3, d4 = _SPC_CONSTANTS[subgroup_size]

    xbar_limits = {
        "center": x_double_bar,
        "ucl": float(x_double_bar + a2 * r_bar),
        "lcl": float(x_double_bar - a2 * r_bar),
    }
    r_limits = {
        "center": r_bar,
        "ucl": float(d4 * r_bar),
        "lcl": float(d3 * r_bar),
    }

    xbar_out = int(np.sum((sub_means > xbar_limits["ucl"]) | (sub_means < xbar_limits["lcl"])))
    r_out = int(np.sum((sub_ranges > r_limits["ucl"]) | (sub_ranges < r_limits["lcl"])))

    return {
        "chart": "X-bar / R",
        "subgroup_size": int(subgroup_size),
        "n_subgroups": int(n_subgroups),
        "subgroup_means": sub_means.tolist(),
        "subgroup_ranges": sub_ranges.tolist(),
        "xbar_limits": xbar_limits,
        "r_limits": r_limits,
        "constants": {"A2": a2, "D3": d3, "D4": d4},
        "xbar_out_of_control": xbar_out,
        "r_out_of_control": r_out,
        "in_control": bool(xbar_out == 0 and r_out == 0),
    }


def p_chart(defectives: pd.Series | np.ndarray,
            inspected: pd.Series | np.ndarray) -> dict:
    """p-grafiği — kusurlu oranı kontrol grafiği.

    Her örneklem için kusurlu sayısı ve incelenen adet girilir;
    limitler örneklem büyüklüğüne göre değişir:
    p̄ ± 3·sqrt(p̄(1−p̄)/n_i).
    """
    d = np.asarray(defectives, dtype=float).flatten()
    n = np.asarray(inspected, dtype=float).flatten()
    if len(d) != len(n):
        raise ValueError("defectives ve inspected uzunlukları eşit olmalı")
    mask = ~(np.isnan(d) | np.isnan(n)) & (n > 0)
    d, n = d[mask], n[mask]
    if len(d) < 3:
        raise ValueError("En az 3 örnekleme grubu gerekli")
    if np.any(d > n) or np.any(d < 0):
        raise ValueError("Kusurlu sayısı 0 ile incelenen adet arasında olmalı")

    p_bar = float(d.sum() / n.sum())
    proportions = d / n

    ucl = p_bar + 3 * np.sqrt(p_bar * (1 - p_bar) / n)
    lcl = np.maximum(0.0, p_bar - 3 * np.sqrt(p_bar * (1 - p_bar) / n))

    out = int(np.sum((proportions > ucl) | (proportions < lcl)))

    return {
        "chart": "p",
        "p_bar": p_bar,
        "proportions": proportions.tolist(),
        "ucl": ucl.tolist(),
        "lcl": lcl.tolist(),
        "out_of_control": out,
        "in_control": bool(out == 0),
        "n_samples": int(len(d)),
    }


def pareto_analysis(categories: pd.Series | np.ndarray) -> dict:
    """Pareto analizi (Premium Program: Quality Control → Pareto).

    Kategoriler frekansa göre sıralanır; kümülatif yüzde ve
    %80 eşiğine göre 'kritik azınlık' (vital few) işaretlenir.
    """
    values = pd.Series(categories).dropna()
    if len(values) == 0:
        raise ValueError("Pareto analizi için veri yok")

    counts = values.value_counts().sort_values(ascending=False)
    total = int(counts.sum())
    pct = counts / total * 100
    cum = pct.cumsum()

    vital_few = [str(cat) for cat, c in cum.items() if c <= 80.0]
    # Eşiği ilk aşan kategori de kritik azınlığa dahildir
    for cat, c in cum.items():
        if str(cat) not in vital_few:
            vital_few.append(str(cat))
            break

    return {
        "categories": [str(c) for c in counts.index],
        "counts": counts.tolist(),
        "percent": pct.tolist(),
        "cumulative_percent": cum.tolist(),
        "vital_few": vital_few,
        "total": total,
        "n_categories": int(counts.size),
    }


def process_capability(data: pd.Series | np.ndarray,
                       lsl: float, usl: float) -> dict:
    """Süreç yeterlilik analizi — Cp ve Cpk
    (Premium Program: Quality Control → Process Capability Analysis).
    
    Cp  = (USL − LSL) / (6σ)
    Cpk = min(USL − μ, μ − LSL) / (3σ)
    """
    y = np.asarray(data, dtype=float).flatten()
    y = y[~np.isnan(y)]
    if len(y) < 10:
        raise ValueError("Süreç yeterliliği için en az 10 gözlem önerilir")
    if usl <= lsl:
        raise ValueError("USL, LSL'den büyük olmalı")
    
    mu = float(np.mean(y))
    sd = float(np.std(y, ddof=1))
    if sd <= 0:
        raise ValueError("Standart sapma sıfırdan büyük olmalı")
    
    cp = (usl - lsl) / (6 * sd)
    cpk = min(usl - mu, mu - lsl) / (3 * sd)
    
    # Spesifikasyon dışı oranlar
    out_below = float(np.mean(y < lsl))
    out_above = float(np.mean(y > usl))
    
    return {
        "cp": float(cp),
        "cpk": float(cpk),
        "mean": mu,
        "std": sd,
        "lsl": float(lsl),
        "usl": float(usl),
        "proportion_below_lsl": out_below,
        "proportion_above_usl": out_above,
        "total_nonconforming": float(out_below + out_above),
        "capable_at_133": bool(cpk >= 1.33),
        "n": int(len(y)),
    }
