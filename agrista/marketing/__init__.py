"""
Agrista Marketing Module — Doğrudan Pazarlama (Premium Program: Direct Marketing)

RFM analizi, posta/teşvik testleri (control vs package) ve aday müşteri
profilleri. Tarımda kooperatif üyelik kampanyaları, tohum/gübre bayi
bilgilendirme postaları ve sadakat programları için kullanılır.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def rfm_analysis(
    data: pd.DataFrame,
    customer_col: str,
    date_col: str,
    amount_col: str,
    reference_date: str | pd.Timestamp | None = None,
    quantiles: int = 5,
) -> dict:
    """RFM analizi (Premium Program: Analyze → Direct Marketing → RFM Analysis).

    Recency (son işlem yakınlığı), Frequency (işlem sıklığı) ve Monetary
    (harcama) skorları kantil bazlı hesaplanır; müşteriler segmentlere
    ayrılır. reference_date verilmezse veri içindeki en son tarih alınır.
    """
    for col in (customer_col, date_col, amount_col):
        if col not in data.columns:
            raise ValueError(f"Sütun bulunamadı: {col}")
    if quantiles < 2:
        raise ValueError("quantiles en az 2 olmalı")

    valid = data[[customer_col, date_col, amount_col]].dropna()
    if len(valid) < quantiles * 2:
        raise ValueError("RFM analizi için yeterli veri yok")

    dates = pd.to_datetime(valid[date_col])
    ref = pd.to_datetime(reference_date) if reference_date is not None \
        else dates.max()
    if dates.max() > ref:
        raise ValueError("reference_date verideki en son tarihten eski olamaz")

    summary = pd.DataFrame({
        "recency": (ref - dates).dt.days.groupby(valid[customer_col]).min(),
        "frequency": valid.groupby(customer_col).size(),
        "monetary": valid.groupby(customer_col)[amount_col].sum(),
    })

    def _score(series: pd.Series, reverse: bool = False) -> pd.Series:
        ranks = series.rank(method="first", pct=True)
        scores = np.ceil(ranks * quantiles).astype(int)
        return quantiles + 1 - scores if reverse else scores

    # R: düşük gün = iyi (yüksek skor); F ve M: yüksek = iyi
    summary["r_score"] = _score(summary["recency"], reverse=True)
    summary["f_score"] = _score(summary["frequency"])
    summary["m_score"] = _score(summary["monetary"])
    summary["rfm_score"] = (summary["r_score"] * 100
                            + summary["f_score"] * 10
                            + summary["m_score"])

    def _segment(row) -> str:
        r, f = row["r_score"], row["f_score"]
        hi = quantiles - quantiles // 2  # üst dilim eşiği
        if r >= hi and f >= hi:
            return "Şampiyon"
        if r >= hi:
            return "Yeni/Yakın"
        if f >= hi:
            return "Sadık ama uzaklaşan"
        return "Riskte"

    summary["segment"] = summary.apply(_segment, axis=1)

    segment_counts = summary["segment"].value_counts()
    return {
        "model": "RFM",
        "customers": summary,
        "segment_summary": {
            str(seg): {"count": int(cnt),
                       "mean_monetary": float(summary.loc[
                           summary["segment"] == seg, "monetary"].mean())}
            for seg, cnt in segment_counts.items()
        },
        "reference_date": str(ref.date()),
        "quantiles": int(quantiles),
        "n_customers": int(len(summary)),
        "n_transactions": int(len(valid)),
    }


def mailing_test(
    control_responses: int,
    control_size: int,
    treatment_responses: int,
    treatment_size: int,
) -> dict:
    """Posta/kampanya testi (Premium Program: Direct Marketing → Control vs Package).

    Kontrol ve uygulama gruplarının yanıt oranları iki oranın z-testiyle
    karşılaştırılır; kaldırma (lift) ve fark için güven aralığı raporlanır.
    """
    for val in (control_responses, control_size, treatment_responses, treatment_size):
        if val < 0:
            raise ValueError("Negatif değer kabul edilmez")
    if control_size == 0 or treatment_size == 0:
        raise ValueError("Grup büyüklükleri sıfır olamaz")
    if control_responses > control_size or treatment_responses > treatment_size:
        raise ValueError("Yanıt sayısı grup büyüklüğünü aşamaz")

    p_c = control_responses / control_size
    p_t = treatment_responses / treatment_size
    diff = p_t - p_c

    # Havuzlanmış oranın z-testi
    p_pool = (control_responses + treatment_responses) / (control_size + treatment_size)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / control_size + 1 / treatment_size))
    z_stat = diff / se_pool if se_pool > 0 else 0.0
    p_value = float(2 * stats.norm.sf(abs(z_stat)))

    se_diff = np.sqrt(p_c * (1 - p_c) / control_size
                      + p_t * (1 - p_t) / treatment_size)
    lift = float(diff / p_c) if p_c > 0 else float("inf")

    return {
        "test": "Two-proportion z-test (Direct Marketing mailing test)",
        "control_rate": float(p_c),
        "treatment_rate": float(p_t),
        "difference": float(diff),
        "ci95_lower": float(diff - 1.96 * se_diff),
        "ci95_upper": float(diff + 1.96 * se_diff),
        "lift": lift,
        "z_statistic": float(z_stat),
        "p_value": p_value,
        "significant_at_005": bool(p_value < 0.05),
        "recommendation": (
            "✅ Uygulama paketi anlamlı şekilde daha iyi — kullanıma alın"
            if p_value < 0.05 and diff > 0 else
            "❌ Anlamlı fark yok — mevcut kontrolle devam edin"
        ),
    }


def prospect_profiles(
    data: pd.DataFrame,
    response_col: str,
    predictors: list[str],
    positive_value=None,
) -> dict:
    """Aday profilleri (Premium Program: Direct Marketing → Prospect Profiles).

    Her kategorik değişken için yanıt oranı tablosu ve genel yanıt
    oranına göre kaldırma (lift) hesaplanır; en güçlü segmentler sıralanır.
    positive_value verilmezse en sık kategori hedef alınır.
    """
    predictors = list(predictors)
    for col in [response_col] + predictors:
        if col not in data.columns:
            raise ValueError(f"Sütun bulunamadı: {col}")

    valid = data[[response_col] + predictors].dropna(subset=[response_col])
    if valid[response_col].nunique() != 2:
        raise ValueError("Yanıt değişkeni tam 2 kategori içermeli")
    if len(valid) < 10:
        raise ValueError("Profil analizi için yeterli veri yok")

    positive = positive_value if positive_value is not None \
        else valid[response_col].value_counts().index[0]
    if positive not in set(valid[response_col]):
        raise ValueError(f"positive_value veride bulunamadı: {positive}")
    overall_rate = float((valid[response_col] == positive).mean())

    profiles = {}
    for col in predictors:
        rows = []
        for name, grp in valid.groupby(col, observed=True):
            rate = float((grp[response_col] == positive).mean())
            rows.append({
                "category": str(name),
                "n": int(len(grp)),
                "response_rate": rate,
                "lift": float(rate / overall_rate) if overall_rate > 0 else 0.0,
            })
        rows.sort(key=lambda r_: -r_["lift"])
        profiles[col] = rows

    all_segments = sorted(
        ((col, row) for col, rows in profiles.items() for row in rows),
        key=lambda cr: -cr[1]["lift"],
    )

    return {
        "model": "Prospect Profiles",
        "positive_category": str(positive),
        "overall_response_rate": overall_rate,
        "profiles": profiles,
        "top_segments": [
            {"variable": col, **row} for col, row in all_segments[:5]
        ],
        "n": int(len(valid)),
    }
