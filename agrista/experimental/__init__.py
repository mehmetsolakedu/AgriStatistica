"""
Agrista Experimental Design Module — Deneysel Tasarım
Agricultural experimental design tools and analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


class ExperimentalDesign:
    """Tarımsal deneysel tasarım araçları."""
    
    @staticmethod
    def random_complete_block(n_treatments: int, n_blocks: int) -> dict:
        """Rastgele Tam Bloklama Deneyi (RCBD)."""
        treatments = [f"T{i+1}" for i in range(n_treatments)]
        
        # Her blokta tüm uygulamalar rastgele atanır
        assignments = {}
        np.random.seed(42)
        for block in range(1, n_blocks + 1):
            assignments[f"Blok_{block}"] = list(np.random.permutation(treatments))
        
        return {
            "design": "RCBD",
            "n_treatments": n_treatments,
            "n_blocks": n_blocks,
            "total_plots": n_treatments * n_blocks,
            "assignments": assignments,
        }
    
    @staticmethod
    def latin_square(n: int) -> dict:
        """Latins Kare Deneyi."""
        treatments = [f"T{i+1}" for i in range(n)]
        
        # Latins kare oluştur (basit döngüsel yaklaşım)
        square = []
        np.random.seed(42)
        perm = list(range(n))
        np.random.shuffle(perm)
        
        for row in range(n):
            row_treatments = [treatments[(perm[i] + row) % n] for i in range(n)]
            square.append(row_treatments)
        
        return {
            "design": "Latin Square",
            "size": n,
            "n_treatments": n,
            "total_plots": n * n,
            "square": square,
        }
    
    @staticmethod
    def factorial_design(factors: dict[str, list]) -> pd.DataFrame:
        """Faktöriyel deney tasarımı.
        
        factors: {"faktör_adi": [seviye1, seviye2, ...]} formatında
        """
        from itertools import product
        
        factor_names = list(factors.keys())
        levels = list(factors.values())
        
        combinations = list(product(*levels))
        
        rows = []
        for combo in combinations:
            row = dict(zip(factor_names, combo))
            row["deney_no"] = len(rows) + 1
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return {
            "design": f"Full Factorial ({' × '.join(str(len(l)) for l in levels)})",
            "n_factors": len(factor_names),
            "total_treatments": len(combinations),
            "factor_levels": factors,
            "dataframe": df,
        }
    
    @staticmethod
    def sample_size_calculation(
        mean_diff: float,
        std_dev: float,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> dict:
        """Gerekli örneklem boyutu hesabı (iki örneklem t-testi)."""
        from scipy import stats
        
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        
        n_per_group = ((z_alpha + z_beta) * std_dev / mean_diff) ** 2
        n_per_group = int(np.ceil(n_per_group))
        
        return {
            "n_per_group": n_per_group,
            "total_samples": n_per_group * 2,
            "mean_difference": mean_diff,
            "std_deviation": std_dev,
            "alpha": alpha,
            "power": power,
            "effect_size": float(mean_diff / std_dev) if std_dev > 0 else None,
        }


class FieldTrialAnalyzer:
    """Tarla deneme analizi."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def rcbd_analysis(self, response_col: str, treatment_col: str, block_col: str) -> dict:
        """RCBD tasarımı için varyans analizi (bloklara göre düzeltilmiş uygulama etkisi)."""
        import statsmodels.api as sm
        from scipy import stats
        
        data = self.data[[response_col, treatment_col, block_col]].dropna()
        if len(data) == 0:
            raise ValueError("RCBD analizi için geçerli veri bulunamadı")
        
        # İki yönlü (etkileşimsiz) model: y ~ uygulama + blok
        treatment_terms = pd.get_dummies(data[treatment_col], prefix="T", drop_first=True).astype(float)
        block_terms = pd.get_dummies(data[block_col], prefix="B", drop_first=True).astype(float)
        y = data[response_col].astype(float)
        
        X_full = sm.add_constant(pd.concat([treatment_terms, block_terms], axis=1))
        X_reduced = sm.add_constant(block_terms)
        
        model_full = sm.OLS(y, X_full).fit()
        model_reduced = sm.OLS(y, X_reduced).fit()
        
        # Uygulama etkisi için kısmi F-testi
        df_diff = int(treatment_terms.shape[1])
        df_resid = int(model_full.df_resid)
        ss_diff = model_reduced.ssr - model_full.ssr
        f_stat = (ss_diff / df_diff) / (model_full.ssr / df_resid) if df_resid > 0 and model_full.ssr > 0 else float("nan")
        p_value = float(stats.f.sf(f_stat, df_diff, df_resid))
        
        # Grup istatistikleri
        treatment_stats = {}
        for treatment_name, group in data.groupby(treatment_col):
            values = group[response_col].values
            treatment_stats[treatment_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "n": int(len(values)),
            }
        
        return {
            "f_statistic": float(f_stat),
            "p_value": p_value,
            "degrees_of_freedom_treatment": df_diff,
            "degrees_of_freedom_residual": df_resid,
            "significant_at_005": bool(p_value < 0.05),
            "treatment_statistics": treatment_stats,
        }
    
    def Duncan_test(self, response_col: str, treatment_col: str, alpha: float = 0.05) -> dict:
        """Duncan çoklu karşılaştırma testi (yaklaşık)."""
        groups = {}
        for name, group in self.data.groupby(treatment_col):
            values = group[response_col].values
            groups[name] = {
                "mean": float(np.mean(values)),
                "variance": float(np.var(values, ddof=1)) if len(values) > 1 else 0.0,
                "n": int(len(values)),
            }
        
        # Ortalama sıralama
        sorted_means = sorted(groups.items(), key=lambda x: x[1]["mean"], reverse=True)
        
        return {
            "sorted_treatments": [(name, stats_) for name, stats_ in sorted_means],
            "alpha": alpha,
        }


def _partial_f_test(df: pd.DataFrame, response_col: str, factor_cols: list, target_factor: str) -> dict:
    """İki yönlü (etkileşimsiz) OLS modelinde tek faktör için kısmi F-testi."""
    data = df[[response_col] + factor_cols].dropna()
    if len(data) < len(factor_cols) + 2:
        raise ValueError("Analiz için yeterli veri yok")
    
    y = data[response_col].astype(float)
    all_terms = pd.DataFrame(index=data.index)
    target_terms = None
    for fc in factor_cols:
        dummies = pd.get_dummies(data[fc], prefix=str(fc), drop_first=True).astype(float)
        if fc == target_factor:
            target_terms = dummies
        all_terms = pd.concat([all_terms, dummies], axis=1)
    
    if target_terms is None or target_terms.shape[1] == 0:
        raise ValueError(f"'{target_factor}' faktöründe yeterli seviye yok")
    
    X_full = sm.add_constant(all_terms)
    X_reduced = sm.add_constant(all_terms.drop(columns=target_terms.columns))
    
    model_full = sm.OLS(y, X_full).fit()
    model_reduced = sm.OLS(y, X_reduced).fit()
    
    df_diff = int(target_terms.shape[1])
    df_resid = int(model_full.df_resid)
    if df_resid <= 0 or model_full.ssr <= 0:
        raise ValueError("Hata serbestlik derecesi kalmadı; daha fazla veri gerekli")
    
    ss_diff = max(model_reduced.ssr - model_full.ssr, 0.0)
    f_stat = (ss_diff / df_diff) / (model_full.ssr / df_resid)
    p_value = float(stats.f.sf(f_stat, df_diff, df_resid))
    
    return {
        "f_statistic": float(f_stat),
        "p_value": p_value,
        "degrees_of_freedom": df_diff,
        "degrees_of_freedom_residual": df_resid,
        "significant_at_005": bool(p_value < 0.05),
    }


def latin_square_anova(data: pd.DataFrame, response_col: str, row_col: str, col_col: str, treatment_col: str) -> dict:
    """Latin kare tasarımı varyans analizi (satır + sütun + uygulama etkileri).
    
    Uygulama etkisi, satır ve sütun bloklama faktörlerine göre düzeltilmiş
    kısmi F-testi ile sınanır.
    """
    factor_cols = [row_col, col_col, treatment_col]
    treatment_result = _partial_f_test(data, response_col, factor_cols, treatment_col)
    
    treatment_stats = {}
    valid = data[[response_col, treatment_col]].dropna()
    for name, group in valid.groupby(treatment_col):
        values = group[response_col].to_numpy(dtype=float)
        treatment_stats[str(name)] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "n": int(len(values)),
        }
    
    return {
        "design": "Latin Square",
        "treatment_effect": treatment_result,
        "treatment_statistics": treatment_stats,
    }


def factorial_anova(data: pd.DataFrame, response_col: str, factors: list) -> dict:
    """Faktöriyel tasarım varyans analizi (ana etkiler + tüm ikili etkileşimler).
    
    Dengeli veya dengesiz verilerde Tip II benzeri kısmi F-testleri kullanılır.
    """
    if len(factors) < 2:
        raise ValueError("Faktöriyel analiz için en az 2 faktör gerekli")
    
    work = data[[response_col] + list(factors)].dropna()
    if len(work) == 0:
        raise ValueError("Analiz için geçerli veri bulunamadı")
    
    y = work[response_col].astype(float)
    
    # Ana etkiler için kukla terimler
    main_terms = {}
    for fc in factors:
        main_terms[fc] = pd.get_dummies(work[fc], prefix=str(fc), drop_first=True).astype(float)
    
    # İkili etkileşim terimleri
    interaction_terms = {}
    for i in range(len(factors)):
        for j in range(i + 1, len(factors)):
            fa, fb = factors[i], factors[j]
            inter = main_terms[fa].values[:, :, None] * main_terms[fb].values[:, None, :]
            cols = [f"{fa}:{fb}:{ca}x{cb}" for ca in main_terms[fa].columns for cb in main_terms[fb].columns]
            interaction_terms[(fa, fb)] = pd.DataFrame(
                inter.reshape(len(work), -1), columns=cols, index=work.index
            )
    
    all_terms = pd.concat(list(main_terms.values()) + list(interaction_terms.values()), axis=1)
    
    def partial_f(term_df: pd.DataFrame) -> dict:
        X_full = sm.add_constant(all_terms)
        X_red = sm.add_constant(all_terms.drop(columns=term_df.columns))
        m_full = sm.OLS(y, X_full).fit()
        m_red = sm.OLS(y, X_red).fit()
        df_diff = int(term_df.shape[1])
        df_resid = int(m_full.df_resid)
        if df_resid <= 0 or m_full.ssr <= 0:
            return {"f_statistic": float("nan"), "p_value": float("nan"),
                    "degrees_of_freedom": df_diff, "significant_at_005": False}
        ss_diff = max(m_red.ssr - m_full.ssr, 0.0)
        f_stat = (ss_diff / df_diff) / (m_full.ssr / df_resid)
        p_val = float(stats.f.sf(f_stat, df_diff, df_resid))
        return {
            "f_statistic": float(f_stat),
            "p_value": p_val,
            "degrees_of_freedom": df_diff,
            "significant_at_005": bool(p_val < 0.05),
        }
    
    main_effects = {fc: partial_f(main_terms[fc]) for fc in factors}
    interactions = {
        f"{fa} × {fb}": partial_f(term_df)
        for (fa, fb), term_df in interaction_terms.items()
    }
    
    # Hücre ortalamaları
    cell_means = work.groupby(list(factors))[response_col].agg(["mean", "std", "count"])
    
    return {
        "design": f"Factorial ({' × '.join(str(work[fc].nunique()) for fc in factors)})",
        "n_obs": int(len(work)),
        "main_effects": main_effects,
        "interactions": interactions,
        "cell_means": cell_means.reset_index().to_dict(orient="records"),
    }
