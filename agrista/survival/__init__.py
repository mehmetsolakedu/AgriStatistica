"""
Agrista Survival Module — Yaşam Analizi (Premium Program: Survival)

Kaplan-Meier sağkalım tahmini (Greenwood standart hatalı) ve iki grup
karşılaştırması için log-rank testi. Tarımda tohum çimlenme süreleri,
makinelerde arızaya kadar geçen süre ve hayvan yaşam süreleri için kullanılır.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def kaplan_meier(time: pd.Series | np.ndarray,
                 event: pd.Series | np.ndarray) -> dict:
    """Kaplan-Meier sağkalım tahmini (Premium Program: Survival → Kaplan-Meier).

    time: olay/sansür zamanları (≥ 0)
    event: 1 = olay gözlendi, 0 = sansürlendi
    S(t) çarpım limiti tahmini ve Greenwood varyansı döndürülür.
    """
    t = np.asarray(time, dtype=float).flatten()
    e = np.asarray(event, dtype=float).flatten()
    if len(t) != len(e):
        raise ValueError("time ve event dizilerinin uzunluğu eşit olmalı")
    mask = ~(np.isnan(t) | np.isnan(e)) & (t >= 0)
    t, e = t[mask], e[mask].astype(int)
    if len(t) < 2:
        raise ValueError("En az 2 geçerli gözlem gerekli")
    if not np.isin(e, [0, 1]).all():
        raise ValueError("event yalnızca 0 (sansür) veya 1 (olay) olabilir")

    order = np.argsort(t, kind="stable")
    t, e = t[order], e[order]

    distinct_times = np.unique(t[e == 1])
    survival, se, n_risk_list, n_events_list = [], [], [], []

    s = 1.0
    greenwood_sum = 0.0
    n_total = len(t)
    for d_time in distinct_times:
        n_at_risk = int(np.sum(t >= d_time))
        n_events = int(np.sum((t == d_time) & (e == 1)))
        if n_at_risk > 0 and n_events > 0:
            s *= (1.0 - n_events / n_at_risk)
            if n_at_risk > n_events:
                greenwood_sum += n_events / (n_at_risk * (n_at_risk - n_events))
        survival.append(s)
        se.append(s * np.sqrt(greenwood_sum))
        n_risk_list.append(n_at_risk)
        n_events_list.append(n_events)

    # Medyan sağkalım: S(t) ≤ 0.5 olan ilk zaman
    median_survival = None
    for d_time, s_val in zip(distinct_times, survival):
        if s_val <= 0.5:
            median_survival = float(d_time)
            break

    return {
        "time": distinct_times.tolist(),
        "n_risk": n_risk_list,
        "n_events": n_events_list,
        "survival": survival,
        "std_error": se,
        "median_survival": median_survival,
        "n_obs": n_total,
        "n_events_total": int(e.sum()),
        "n_censored": int((1 - e).sum()),
    }


def log_rank_test(time1, event1, time2, event2) -> dict:
    """Log-rank testi — iki sağkalım eğrisinin karşılaştırılması
    (Premium Program: Survival → Kaplan-Meier, factor karşılaştırması).

    Mantel-Cox biçimi: X² = (O1 − E1)² / V1.
    """
    t1 = np.asarray(time1, dtype=float).flatten()
    e1 = np.asarray(event1, dtype=float).flatten().astype(int)
    t2 = np.asarray(time2, dtype=float).flatten()
    e2 = np.asarray(event2, dtype=float).flatten().astype(int)

    if len(t1) < 2 or len(t2) < 2:
        raise ValueError("Her grupta en az 2 gözlem gerekli")
    if e1.sum() == 0 or e2.sum() == 0:
        raise ValueError("Her iki grupta da en az 1 olay gerekli")

    all_times = np.unique(np.concatenate([t1[e1 == 1], t2[e2 == 1]]))

    o1_total, e1_expected, variance = 0.0, 0.0, 0.0
    for d_time in all_times:
        n1 = int(np.sum(t1 >= d_time))
        n2 = int(np.sum(t2 >= d_time))
        d1 = int(np.sum((t1 == d_time) & (e1 == 1)))
        d2 = int(np.sum((t2 == d_time) & (e2 == 1)))
        n = n1 + n2
        d = d1 + d2
        if n == 0 or d == 0:
            continue
        o1_total += d1
        expected = d * n1 / n
        e1_expected += expected
        if n > 1:
            variance += d * n1 * n2 * (n - d) / (n ** 2 * (n - 1))

    if variance <= 0:
        raise ValueError("Log-rank testi için yeterli olay/zaman dağılımı yok")

    chi2 = (o1_total - e1_expected) ** 2 / variance
    p_value = float(stats.chi2.sf(chi2, df=1))

    return {
        "test": "Log-rank (Mantel-Cox)",
        "chi_square": float(chi2),
        "p_value": p_value,
        "degrees_of_freedom": 1,
        "observed_group1": int(o1_total),
        "expected_group1": float(e1_expected),
        "significant_at_005": bool(p_value < 0.05),
    }


def life_tables(
    time: pd.Series | np.ndarray,
    event: pd.Series | np.ndarray,
    interval_width: float = 1.0,
) -> dict:
    """Yaşam tabloları — aktüeryal yöntem (Premium Program: Survival → Life Tables).

    Sabit genişlikli aralıklarda riske giren/sansürlenen/olay sayıları,
    düzeltilmiş risk kümesi, ölüm oranı, kümülatif sağkalım ve tehlike
    oranı raporlanır.
    """
    t = np.asarray(time, dtype=float).flatten()
    e = np.asarray(event, dtype=float).flatten()
    mask = ~(np.isnan(t) | np.isnan(e))
    t, e = t[mask], e[mask]
    if len(t) < 4:
        raise ValueError("Yaşam tablosu için en az 4 geçerli gözlem gerekli")
    if not np.isin(e, [0, 1]).all():
        raise ValueError("event yalnızca 0/1 değerleri alabilir")
    if interval_width <= 0:
        raise ValueError("interval_width pozitif olmalı")

    n_intervals = int(np.floor(t.max() / interval_width)) + 1
    rows = []
    survival_cum = 1.0
    median = None
    prev_survival = 1.0

    for i in range(n_intervals):
        lo, hi = i * interval_width, (i + 1) * interval_width
        in_interval = (t >= lo) & (t < hi)
        d = int((in_interval & (e == 1)).sum())     # olay
        w = int((in_interval & (e == 0)).sum())     # sansür
        n_enter = int((t >= lo).sum())              # aralık başında riskte
        n_adj = n_enter - w / 2
        q = d / n_adj if n_adj > 0 else 0.0
        p = 1.0 - q
        survival_cum *= p
        hazard = q / (interval_width * (1 - q / 2)) if q < 1 else float("inf")
        rows.append({
            "interval_start": float(lo),
            "interval_end": float(hi),
            "n_entering": n_enter,
            "n_withdrawn": w,
            "n_terminated": d,
            "n_adjusted": float(n_adj),
            "proportion_terminating": float(q),
            "survival": float(survival_cum),
            "hazard_rate": float(hazard),
        })
        # Medyan interpolasyonu: S ilk kez 0.5 altına düşerken
        if median is None and survival_cum < 0.5 and prev_survival >= 0.5:
            if prev_survival > survival_cum:
                frac = (prev_survival - 0.5) / (prev_survival - survival_cum)
                median = float(lo - interval_width + frac * interval_width)
            else:
                median = float(lo)
        prev_survival = survival_cum

    return {
        "model": "Life Tables (actuarial)",
        "table": pd.DataFrame(rows),
        "interval_width": float(interval_width),
        "median_survival": median,
        "final_survival": float(survival_cum),
        "n_events": int((e == 1).sum()),
        "n_censored": int((e == 0).sum()),
        "n": int(len(t)),
    }


def cox_regression(
    time: pd.Series | np.ndarray,
    event: pd.Series | np.ndarray,
    covariates: pd.DataFrame | np.ndarray,
    max_iter: int = 100,
) -> dict:
    """Cox orantılı tehlike regresyonu (Premium Program: Survival → Cox Regression).

    Kısmi olabilirlik (Breslow bağ yöntemi) Newton-Raphson ile ençoklanır.
    Katsayılar, exp(β) tehlike oranları, Wald testleri ve Harrell C
    uyum indeksi döndürülür.
    """
    t = np.asarray(time, dtype=float).flatten()
    e = np.asarray(event, dtype=float).flatten()
    X = np.asarray(covariates, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    mask = ~(np.isnan(t) | np.isnan(e) | np.isnan(X).any(axis=1))
    t, e, X = t[mask], e[mask], X[mask]
    n, p = X.shape
    if n < 10:
        raise ValueError("Cox regresyonu için en az 10 geçerli gözlem gerekli")
    if not np.isin(e, [0, 1]).all():
        raise ValueError("event yalnızca 0/1 değerleri alabilir")
    if int(e.sum()) < 2:
        raise ValueError("En az 2 olay gerekli")

    # Azalan zamana göre sırala; risk kümeleri kümülatif toplamla kurulur
    order = np.argsort(-t, kind="stable")
    t_s, e_s, X_s = t[order], e[order], X[order]

    def _neg_partial(beta: np.ndarray):
        """Negatif log-kısmi-olabilirlik, gradyan ve Hessian (Breslow)."""
        eta = X_s @ beta
        eta = np.clip(eta, -50, 50)
        risk = np.exp(eta)
        S0 = np.cumsum(risk)
        S1 = np.cumsum(risk[:, None] * X_s, axis=0)
        S2 = np.cumsum(risk[:, None, None] * X_s[:, :, None] * X_s[:, None, :], axis=0)
        events = e_s == 1
        # Aynı zamandaki bağlar: azalan sıralamada her zaman bloğunun
        # son indeksi, o andaki tam risk kümesi toplamını verir
        boundaries = np.where(np.diff(t_s) != 0)[0]
        boundaries = np.concatenate([boundaries, [n - 1]])
        block_end = np.repeat(boundaries, np.diff(np.concatenate([[-1], boundaries])))
        S0b, S1b, S2b = S0[block_end], S1[block_end], S2[block_end]
        mean = S1b / S0b[:, None]
        var = S2b / S0b[:, None, None] - mean[:, :, None] * mean[:, None, :]
        ll = float(np.sum(eta[events] - np.log(S0b[events])))
        grad = (X_s[events] - mean[events]).sum(axis=0)
        hess = -var[events].sum(axis=0)
        return -ll, -grad, -hess

    beta = np.zeros(p)
    for _ in range(max_iter):
        _, grad, hess = _neg_partial(beta)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        beta_new = beta - step
        if np.all(np.abs(step) < 1e-8):
            beta = beta_new
            break
        beta = beta_new

    neg_ll, _, hess = _neg_partial(beta)
    try:
        cov = np.linalg.inv(hess)
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)

    z = beta / se
    p_values = 2 * stats.norm.sf(np.abs(z))

    # Null model olabilirliği (β = 0)
    neg_ll_null, _, _ = _neg_partial(np.zeros(p))
    lr_stat = float(2 * (neg_ll_null - neg_ll))
    lr_p = float(stats.chi2.sf(lr_stat, df=p))

    # Harrell C indeksi: t_i < t_j ve i olay yaşadıysa, karşılaştırılabilir
    # çift eta_i > eta_j durumunda uyumludur (yüksek tehlike = kısa süre)
    concordant = comparable = 0
    for i in range(n):
        if e[i] != 1:
            continue
        eta_i = float(X[i] @ beta)
        for j in range(n):
            if i == j or t[j] <= t[i]:
                continue
            comparable += 1
            eta_j = float(X[j] @ beta)
            if eta_i > eta_j:
                concordant += 1
            elif eta_i == eta_j:
                concordant += 0.5
    c_index = concordant / comparable if comparable > 0 else float("nan")

    return {
        "model": "Cox Proportional Hazards (Breslow)",
        "coefficients": [float(b) for b in beta],
        "exp_coef": [float(np.exp(b)) for b in beta],
        "std_errors": [float(s) for s in se],
        "wald_z": [float(v) for v in z],
        "p_values": [float(v) for v in p_values],
        "log_partial_likelihood": float(-neg_ll),
        "lr_chi_square": lr_stat,
        "lr_p_value": lr_p,
        "concordance_index": float(c_index),
        "n_events": int(e.sum()),
        "n": int(n),
        "significant_at_005": bool(lr_p < 0.05),
    }
