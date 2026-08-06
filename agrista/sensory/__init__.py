"""
Agrista Sensory Module — Bahçe Bitkileri Duyusal Analiz
Hedonik değerlendirme, panelist uyumu ve panel varyans analizi.

Literatür dayanağı: docs/01_ALT_BRANS_ISTATISTIK_LITERATUR_LOG.md (Bölüm 3)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def kendall_w(matrix: pd.DataFrame | np.ndarray) -> dict:
    """Kendall W uyum katsayısı — panelist/jüri değerlendirmeleri arası uyum.

    matrix: satırlar = örnekler, sütunlar = panelistler (puan veya sıra).
    """
    df = pd.DataFrame(matrix).dropna()
    k_objects, n_judges = df.shape
    if k_objects < 2 or n_judges < 2:
        raise ValueError("En az 2 örnek ve 2 panelist gerekli")

    # Her panelist içinde sıraya dönüştür
    ranks = df.rank(axis=0)
    rank_sums = ranks.sum(axis=1)
    mean_sum = rank_sums.mean()
    S = float(np.sum((rank_sums - mean_sum) ** 2))

    W = 12.0 * S / (n_judges ** 2 * (k_objects ** 3 - k_objects))
    chi2 = n_judges * (k_objects - 1) * W
    p_value = float(stats.chi2.sf(chi2, df=k_objects - 1))

    return {
        "kendall_w": float(W),
        "chi_square": float(chi2),
        "p_value": p_value,
        "degrees_of_freedom": int(k_objects - 1),
        "agreement_significant": bool(p_value < 0.05),
        "n_objects": int(k_objects),
        "n_judges": int(n_judges),
        "interpretation": (
            "Güçlü uyum" if W > 0.7 else "Orta uyum" if W > 0.4 else "Zayıf uyum"
        ),
    }


def hedonic_summary(
    data: pd.DataFrame,
    sample_col: str,
    panelist_col: str,
    score_col: str,
) -> dict:
    """9'lu hedonik ölçek özeti + Friedman testi.

    Örnek başına ortalama/standart sapma ve örnekler arası fark testi.
    """
    valid = data[[sample_col, panelist_col, score_col]].dropna()
    if valid[sample_col].nunique() < 2:
        raise ValueError("En az 2 örnek gerekli")

    per_sample = valid.groupby(sample_col)[score_col].agg(["mean", "std", "count"])

    # Friedman için panelist × örnek matrisi
    pivot = valid.pivot_table(index=panelist_col, columns=sample_col, values=score_col)
    pivot = pivot.dropna()
    friedman_result = None
    if pivot.shape[0] >= 2 and pivot.shape[1] >= 2:
        chi2, p_val = stats.friedmanchisquare(*[pivot[c].values for c in pivot.columns])
        friedman_result = {
            "chi_square": float(chi2),
            "p_value": float(p_val),
            "significant_at_005": bool(p_val < 0.05),
        }

    ranking = per_sample["mean"].sort_values(ascending=False)
    return {
        "sample_statistics": {
            str(idx): {"mean": float(row["mean"]), "std": float(row["std"]) if not np.isnan(row["std"]) else 0.0,
                       "n": int(row["count"])}
            for idx, row in per_sample.iterrows()
        },
        "preference_ranking": [str(s) for s in ranking.index],
        "friedman_test": friedman_result,
        "n_panelists": int(valid[panelist_col].nunique()),
        "n_samples": int(valid[sample_col].nunique()),
    }


def panel_anova(
    data: pd.DataFrame,
    score_col: str,
    sample_col: str,
    panelist_col: str,
) -> dict:
    """Panel ANOVA'sı — örnek etkisi, panelistler rastgele blok kabul edilerek
    kısmi F-testi ile sınanır (etkileşimsiz iki yönlü model)."""
    valid = data[[score_col, sample_col, panelist_col]].dropna()
    if valid[sample_col].nunique() < 2 or valid[panelist_col].nunique() < 2:
        raise ValueError("En az 2 örnek ve 2 panelist gerekli")

    y = valid[score_col].astype(float)
    sample_terms = pd.get_dummies(valid[sample_col], prefix="ornek", drop_first=True).astype(float)
    panelist_terms = pd.get_dummies(valid[panelist_col], prefix="panelist", drop_first=True).astype(float)

    X_full = sm.add_constant(pd.concat([sample_terms, panelist_terms], axis=1))
    X_reduced = sm.add_constant(panelist_terms)

    model_full = sm.OLS(y, X_full).fit()
    model_reduced = sm.OLS(y, X_reduced).fit()

    df_diff = int(sample_terms.shape[1])
    df_resid = int(model_full.df_resid)
    if df_resid <= 0 or model_full.ssr <= 0:
        raise ValueError("Panel ANOVA için yeterli serbestlik derecesi yok")

    ss_diff = max(model_reduced.ssr - model_full.ssr, 0.0)
    f_stat = (ss_diff / df_diff) / (model_full.ssr / df_resid)
    p_value = float(stats.f.sf(f_stat, df_diff, df_resid))

    return {
        "sample_effect": {
            "f_statistic": float(f_stat),
            "p_value": p_value,
            "degrees_of_freedom": df_diff,
            "significant_at_005": bool(p_value < 0.05),
        },
        "sample_means": {
            str(name): float(grp[score_col].mean())
            for name, grp in valid.groupby(sample_col)
        },
        "n_panelists": int(valid[panelist_col].nunique()),
        "n_samples": int(valid[sample_col].nunique()),
    }


def cronbach_alpha(matrix: pd.DataFrame | np.ndarray) -> dict:
    """Cronbach alfa güvenirlik katsayısı (Premium Program: Scale → Reliability Analysis).
    
    matrix: satırlar = denekler, sütunlar = maddeler/panelistler.
    Duyusal panellerde ve anket çalışmalarında iç tutarlılık ölçütü.
    """
    df = pd.DataFrame(matrix).dropna()
    n_subjects, k = df.shape
    if k < 2:
        raise ValueError("Cronbach alfa için en az 2 madde gerekli")
    if n_subjects < 2:
        raise ValueError("En az 2 denek gerekli")
    
    item_vars = df.var(ddof=1)
    total_var = df.sum(axis=1).var(ddof=1)
    if total_var == 0:
        raise ValueError("Toplam puan varyansı sıfır; güvenirlik hesaplanamaz")
    
    alpha = float(k / (k - 1) * (1 - item_vars.sum() / total_var))
    
    # Madde silindiğinde alfa (Premium Program 'Alpha if item deleted')
    alpha_if_deleted = {}
    for col in df.columns:
        rest = df.drop(columns=[col])
        rest_vars = rest.var(ddof=1)
        rest_total_var = rest.sum(axis=1).var(ddof=1)
        k_rest = k - 1
        if rest_total_var > 0 and k_rest >= 2:
            alpha_if_deleted[str(col)] = float(
                k_rest / (k_rest - 1) * (1 - rest_vars.sum() / rest_total_var)
            )
    
    return {
        "cronbach_alpha": alpha,
        "n_items": int(k),
        "n_subjects": int(n_subjects),
        "alpha_if_item_deleted": alpha_if_deleted,
        "interpretation": (
            "Mükemmel" if alpha >= 0.9 else
            "İyi" if alpha >= 0.8 else
            "Kabul edilebilir" if alpha >= 0.7 else
            "Sorgulanabilir" if alpha >= 0.6 else "Zayıf"
        ),
    }
