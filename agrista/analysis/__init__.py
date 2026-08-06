"""
Agrista Analysis Module — İstatistiksel Analiz
Statistical analysis functions for agricultural data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from typing import Optional, Callable


def descriptive_stats(data: pd.DataFrame | list) -> dict:
    """Betimsel istatistikler hesapla."""
    if isinstance(data, list):
        data = pd.Series(data)
    
    numeric_data = data.select_dtypes(include=[np.number]) if isinstance(data, pd.DataFrame) else data
    
    if numeric_data.empty:
        raise ValueError("Sayısal veri bulunamadı")
    
    results = {}
    for col in numeric_data.columns if isinstance(numeric_data, pd.DataFrame) else [numeric_data.name or "data"]:
        series = numeric_data[col] if isinstance(numeric_data, pd.DataFrame) else numeric_data
        
        valid = series.dropna()
        if len(valid) == 0:
            continue
            
        results[col] = {
            "count": int(len(valid)),
            "mean": float(np.mean(valid)),
            "median": float(np.median(valid)),
            "std": float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
            "variance": float(np.var(valid, ddof=1)) if len(valid) > 1 else 0.0,
            "min": float(np.min(valid)),
            "max": float(np.max(valid)),
            "range": float(np.max(valid) - np.min(valid)),
            "skewness": float(stats.skew(valid)) if len(valid) >= 3 else None,
            "kurtosis": float(stats.kurtosis(valid)) if len(valid) >= 4 else None,
            "q1": float(np.percentile(valid, 25)),
            "q3": float(np.percentile(valid, 75)),
            "iqr": float(np.percentile(valid, 75) - np.percentile(valid, 25)),
        }
    
    return results


def correlation_analysis(data: pd.DataFrame, method: str = "pearson") -> dict:
    """Korelasyon analizi yapar."""
    numeric_data = data.select_dtypes(include=[np.number])
    if numeric_data.empty:
        raise ValueError("Sayısal sütun bulunamadı")
    
    corr_matrix = numeric_data.corr(method=method)
    
    # Anlamlılık testi
    p_values = pd.DataFrame(np.nan, index=numeric_data.columns, columns=numeric_data.columns)
    
    for i in range(len(numeric_data.columns)):
        for j in range(i + 1, len(numeric_data.columns)):
            col_i = numeric_data.columns[i]
            col_j = numeric_data.columns[j]
            # Çift bazında eksik değerleri birlikte ele al (eşleşmiş gözlemler)
            mask = numeric_data[col_i].notna() & numeric_data[col_j].notna()
            xi, xj = numeric_data.loc[mask, col_i], numeric_data.loc[mask, col_j]
            if len(xi) < 3:
                continue
            if method == "pearson":
                corr_val, p_val = stats.pearsonr(xi, xj)
            elif method == "spearman":
                corr_val, p_val = stats.spearmanr(xi, xj)
            else:
                raise ValueError(f"Desteklenmeyen yöntem: {method}")
            
            p_values.loc[col_i, col_j] = p_val
            p_values.loc[col_j, col_i] = p_val
    
    return {
        "correlation_matrix": corr_matrix,
        "p_values": p_values,
        "significant_pairs": _find_significant_pairs(corr_matrix, p_values),
    }


def _find_significant_pairs(corr_matrix: pd.DataFrame, p_values: pd.DataFrame, alpha: float = 0.05) -> list:
    """Anlamlı korelasyon çiftlerini bul."""
    pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            p_val = p_values.iloc[i, j]
            if not np.isnan(p_val) and p_val < alpha:
                pairs.append({
                    "variable_1": cols[i],
                    "variable_2": cols[j],
                    "correlation": float(corr_matrix.iloc[i, j]),
                    "p_value": float(p_val),
                    "significant": True,
                })
    return sorted(pairs, key=lambda x: abs(x["correlation"]), reverse=True)


def one_sample_t_test(data: pd.Series | list | np.ndarray, test_value: float = 0.0) -> dict:
    """Tek örneklem t-testi (Premium Program: Compare Means → One-Sample T Test).
    
    Örneklem ortalaması, belirtilen test değerinden anlamlı şekilde
    farklı mı diye sınar.
    """
    arr = np.asarray(data, dtype=float).flatten()
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        raise ValueError("En az 2 geçerli veri gerekli")
    
    t_stat, p_value = stats.ttest_1samp(arr, test_value)
    n = len(arr)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    se = sd / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    
    return {
        "test": "One-Sample T Test",
        "test_value": float(test_value),
        "mean": mean,
        "std": sd,
        "std_error": float(se),
        "ci95_lower": float(mean - t_crit * se),
        "ci95_upper": float(mean + t_crit * se),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "degrees_of_freedom": int(n - 1),
        "cohens_d": float((mean - test_value) / sd) if sd > 0 else 0.0,
        "n": int(n),
        "significant_at_005": bool(p_value < 0.05),
        "significant_at_001": bool(p_value < 0.01),
    }


def paired_t_test(data1: pd.Series | list, data2: pd.Series | list) -> dict:
    """Eşleştirilmiş (bağımlı) iki örneklem t-testi
    (Premium Program: Compare Means → Paired-Samples T Test).
    
    Aynı deneklerin önce/sonra ölçümleri gibi eşleşmiş veriler için.
    """
    d1 = np.asarray(data1, dtype=float).flatten()
    d2 = np.asarray(data2, dtype=float).flatten()
    if len(d1) != len(d2):
        raise ValueError("Eşleştirilmiş test için iki dizinin uzunluğu eşit olmalı")
    
    mask = ~(np.isnan(d1) | np.isnan(d2))
    d1, d2 = d1[mask], d2[mask]
    if len(d1) < 2:
        raise ValueError("En az 2 geçerli eşleşme gerekli")
    
    t_stat, p_value = stats.ttest_rel(d1, d2)
    diffs = d1 - d2
    n = len(diffs)
    mean_diff = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1))
    se = sd_diff / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    
    return {
        "test": "Paired-Samples T Test",
        "mean_difference": mean_diff,
        "std_difference": sd_diff,
        "ci95_lower": float(mean_diff - t_crit * se),
        "ci95_upper": float(mean_diff + t_crit * se),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "degrees_of_freedom": int(n - 1),
        "cohens_d": float(mean_diff / sd_diff) if sd_diff > 0 else 0.0,
        "n_pairs": int(n),
        "significant_at_005": bool(p_value < 0.05),
        "significant_at_001": bool(p_value < 0.01),
    }


def t_test(data1: pd.Series | list, data2: pd.Series | list, alternative: str = "two-sided") -> dict:
    """Bağımsız iki örneklem t-testi."""
    d1 = np.asarray(data1, dtype=float) if not isinstance(data1, np.ndarray) else data1
    d2 = np.asarray(data2, dtype=float) if not isinstance(data2, np.ndarray) else data2
    
    d1 = d1[~np.isnan(d1)]
    d2 = d2[~np.isnan(d2)]
    
    if len(d1) < 2 or len(d2) < 2:
        raise ValueError("Her iki grupta en az 2 geçerli veri olmalı")
    
    # Levene testi ile varyans eşitliği kontrolü
    levene_stat, levene_p = stats.levene(d1, d2)
    equal_var = levene_p > 0.05
    
    # alternative parametresi: scipy uyumluluğu için her zaman two-sided kullan
    alt_param = "two-sided" if alternative == "two-sided" else alternative
    t_stat, p_value = stats.ttest_ind(d1, d2, equal_var=equal_var, alternative=alt_param)
    
    # Cohen's d etki boyutu
    pooled_std = np.sqrt(((len(d1) - 1) * np.std(d1, ddof=1)**2 + (len(d2) - 1) * np.std(d2, ddof=1)**2) / (len(d1) + len(d2) - 2))
    cohens_d = (np.mean(d1) - np.mean(d2)) / pooled_std if pooled_std > 0 else 0
    
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "degrees_of_freedom": int(len(d1) + len(d2) - 2),
        "equal_variances_assumed": bool(equal_var),
        "levene_test_p": float(levene_p),
        "cohens_d": float(cohens_d),
        "group1_mean": float(np.mean(d1)),
        "group2_mean": float(np.mean(d2)),
        "group1_std": float(np.std(d1, ddof=1)),
        "group2_std": float(np.std(d2, ddof=1)),
        "significant_at_005": bool(p_value < 0.05),
        "significant_at_001": bool(p_value < 0.01),
    }


def anova_one_way(*groups: pd.Series | list) -> dict:
    """Tek yönlü varyans analizi (One-Way ANOVA)."""
    if len(groups) < 2:
        raise ValueError("En az 2 grup gerekli")
    
    data_arrays = []
    for g in groups:
        arr = np.asarray(g, dtype=float) if not isinstance(g, np.ndarray) else g
        arr = arr[~np.isnan(arr)]
        if len(arr) < 1:
            raise ValueError("Her grupta en az 1 geçerli veri olmalı")
        data_arrays.append(arr)
    
    f_stat, p_value = stats.f_oneway(*data_arrays)
    
    # Eta-kare (etki boyutu)
    all_data = np.concatenate(data_arrays)
    grand_mean = np.mean(all_data)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in data_arrays)
    ss_total = sum((x - grand_mean)**2 for x in all_data)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    
    return {
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "degrees_of_freedom_between": int(len(groups) - 1),
        "degrees_of_freedom_within": int(sum(len(g) for g in data_arrays) - len(groups)),
        "eta_squared": float(eta_squared),
        "group_means": [float(np.mean(g)) for g in data_arrays],
        "group_stds": [float(np.std(g, ddof=1)) if len(g) > 1 else 0.0 for g in data_arrays],
        "significant_at_005": bool(p_value < 0.05),
        "significant_at_001": bool(p_value < 0.01),
    }


def linear_regression(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> dict:
    """Basit doğrusal regresyon."""
    x = np.asarray(x, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()
    
    # Null değerleri çıkar
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    
    if len(x) < 3:
        raise ValueError("En az 3 geçerli veri noktası gerekli")
    
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    
    return {
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "p_value": float(model.f_pvalue),
        "coefficients": {
            "intercept": float(model.params[0]),
            "slope": float(model.params[1]),
        },
        "std_errors": {
            "intercept": float(model.bse[0]),
            "slope": float(model.bse[1]),
        },
        "p_values": {
            "intercept": float(model.pvalues[0]),
            "slope": float(model.pvalues[1]),
        },
        "f_statistic": float(model.fvalue),
        "residual_std_error": float(np.sqrt(model.mse_resid)),
        "n_obs": int(model.nobs),
    }


def multiple_regression(data: pd.DataFrame, target: str, predictors: list[str]) -> dict:
    """Çoklu doğrusal regresyon."""
    valid_data = data.dropna(subset=[target] + predictors)
    
    if len(valid_data) < len(predictors) + 2:
        raise ValueError("Veri sayısı yetersiz")
    
    X = sm.add_constant(valid_data[predictors])
    y = valid_data[target]
    
    model = sm.OLS(y, X).fit()
    
    params_arr = np.asarray(model.params)
    bse_arr = np.asarray(model.bse)
    pvalues_arr = np.asarray(model.pvalues)
    
    results = {
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "p_value": float(model.f_pvalue),
        "coefficients": {"const": float(params_arr[0])},
        "std_errors": {"const": float(bse_arr[0])},
        "p_values": {"const": float(pvalues_arr[0])},
        "f_statistic": float(model.fvalue),
        "residual_std_error": float(np.sqrt(model.mse_resid)),
        "n_obs": int(model.nobs),
    }
    
    # Model parametrelerinin sütun isimlerini al (const hariç)
    model_cols = [c for c in model.model.exog_names if c != 'const']
    
    for i, col in enumerate(predictors):
        if col in model_cols:
            # +1: params dizisinin 0. elemanı sabit terim (const)
            idx = model_cols.index(col) + 1
            results["coefficients"][col] = float(params_arr[idx])
            results["std_errors"][col] = float(bse_arr[idx])
            results["p_values"][col] = float(pvalues_arr[idx])
        else:
            # Model tarafından atlanan değişken (sabit veya çoklu bağlantı)
            results["coefficients"][col] = None
            results["std_errors"][col] = None
            results["p_values"][col] = None
    
    return results


def chi_square_test(observed: pd.Series | list, expected: Optional[pd.Series | list] = None) -> dict:
    """Ki-kare uyumluluk testi."""
    obs = np.asarray(observed, dtype=float).flatten()
    
    if expected is not None:
        exp = np.asarray(expected, dtype=float).flatten()
    else:
        exp = np.mean(obs) * np.ones_like(obs)
    
    # Null değerleri çıkar
    mask = ~(np.isnan(obs) | np.isnan(exp)) & (exp > 0)
    obs = obs[mask]
    exp = exp[mask]
    
    if len(obs) < 2:
        raise ValueError("En az 2 geçerli kategori gerekli")
    
    chi2_stat, p_value = stats.chisquare(obs, f_exp=exp)
    
    # Cramer's V (etki boyutu)
    n = np.sum(obs)
    k = len(obs)
    cramers_v = np.sqrt(chi2_stat / (n * (k - 1))) if k > 1 else 0
    
    return {
        "chi_square_statistic": float(chi2_stat),
        "p_value": float(p_value),
        "degrees_of_freedom": int(len(obs) - 1),
        "cramers_v": float(cramers_v),
        "observed": obs.tolist(),
        "expected": exp.tolist(),
        "significant_at_005": bool(p_value < 0.05),
    }


# ---------------------------------------------------------------------------
# P0: ANOVA sonrası çoklu karşılaştırma, varsayım testleri ve
# parametrik olmayan testler (literatür logu: docs/01_ALT_BRANS_...)
# ---------------------------------------------------------------------------


def _group_data_from_df(data: pd.DataFrame, response_col: str, group_col: str) -> tuple:
    """DataFrame'den geçerli (ad, değerler) grup listesi çıkar."""
    groups = []
    for name, grp in data.groupby(group_col):
        values = grp[response_col].dropna().to_numpy(dtype=float)
        if len(values) > 0:
            groups.append((name, values))
    if len(groups) < 2:
        raise ValueError("En az 2 geçerli grup gerekli")
    return groups


def posthoc_tukey(data: pd.DataFrame, response_col: str, group_col: str, alpha: float = 0.05) -> dict:
    """Tukey HSD çoklu karşılaştırma testi."""
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from itertools import combinations
    
    valid = data[[response_col, group_col]].dropna()
    res = pairwise_tukeyhsd(endog=valid[response_col], groups=valid[group_col], alpha=alpha)
    
    group_names = list(res.groupsunique)
    means = {str(g): float(valid.loc[valid[group_col] == g, response_col].mean()) for g in group_names}
    
    comparisons = []
    for (g1, g2), meandiff, p_adj, (lower, upper), reject in zip(
        combinations(group_names, 2), res.meandiffs, res.pvalues, res.confint, res.reject
    ):
        comparisons.append({
            "group_1": str(g1),
            "group_2": str(g2),
            "mean_diff": float(meandiff),
            "p_value": float(p_adj),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "significant": bool(reject),
        })
    
    return {
        "test": "Tukey HSD",
        "alpha": alpha,
        "group_means": means,
        "comparisons": comparisons,
        "significant_pairs": [c for c in comparisons if c["significant"]],
    }


def posthoc_duncan(data: pd.DataFrame, response_col: str, group_col: str, alpha: float = 0.05) -> dict:
    """Duncan çoklu aralık testi (studentized range tabanlı gerçek gerçekleme).
    
    Homojen alt gruplar harf kodlarıyla raporlanır (yaklaşık bağlama algoritması).
    """
    groups = _group_data_from_df(data, response_col, group_col)
    values_all = [v for _, v in groups]
    
    # Tek yönlü ANOVA'dan MSE ve hata serbestlik derecesi
    f_stat, anova_p = stats.f_oneway(*values_all)
    n_total = sum(len(v) for v in values_all)
    k = len(values_all)
    ss_within = sum(((v - v.mean()) ** 2).sum() for v in values_all)
    df_within = n_total - k
    if df_within <= 0:
        raise ValueError("Hata serbestlik derecesi için yeterli veri yok")
    mse = ss_within / df_within
    
    # Harmonik ortalama örneklem büyüklüğü
    n_harmonic = k / sum(1.0 / len(v) for v in values_all)
    se = float(np.sqrt(mse / n_harmonic))
    
    # Ortalamaları küçükten büyüğe sırala
    ordered = sorted(groups, key=lambda g: g[1].mean())
    means = [float(v.mean()) for _, v in ordered]
    names = [str(name) for name, _ in ordered]
    
    # Duncan kritik aralıkları: SSR_p = q(1 - (1-alpha)^(p-1); p, df) * SE
    critical_ranges = {}
    for p in range(2, k + 1):
        alpha_p = 1.0 - (1.0 - alpha) ** (p - 1)
        q_crit = stats.studentized_range.ppf(1 - alpha_p, p, df_within)
        critical_ranges[p] = float(q_crit * se)
    
    # İkili anlamlılık: sıra farkı p olan çiftler SSR_p ile karşılaştırılır
    significant_matrix = {}
    for i in range(k):
        for j in range(i + 1, k):
            p_span = j - i + 1
            sig = abs(means[j] - means[i]) > critical_ranges[p_span]
            significant_matrix[(names[i], names[j])] = bool(sig)
    
    # Harf atama: uyumlu grup (klik) örtmesi + onarım turu
    if k > 26:
        raise ValueError("Harf kodlaması en fazla 26 grup destekler")
    
    def _pair_key(n1, n2):
        return (n1, n2) if (n1, n2) in significant_matrix else (n2, n1)
    
    def _compatible(name, group):
        return all(not significant_matrix[_pair_key(name, m)] for m in group)
    
    letter_groups: list = []          # her öğe bir isim kümesi
    member_letters: dict = {n: set() for n in names}
    
    # Açgözlü geçiş (büyükten küçüğe): ilk uyumlu harf grubuna katıl
    for idx in range(k - 1, -1, -1):
        name = names[idx]
        joined = False
        for gi, group in enumerate(letter_groups):
            if _compatible(name, group):
                group.add(name)
                member_letters[name].add(gi)
                joined = True
                break
        if not joined:
            letter_groups.append({name})
            member_letters[name].add(len(letter_groups) - 1)
    
    # Onarım: anlamlı fark olmayan ama harf paylaşmayan çiftleri birleştir
    for i in range(k):
        for j in range(i + 1, k):
            ni, nj = names[i], names[j]
            if significant_matrix[(ni, nj)] or (member_letters[ni] & member_letters[nj]):
                continue
            placed = False
            for gi in sorted(member_letters[nj]):
                if _compatible(ni, letter_groups[gi]):
                    letter_groups[gi].add(ni)
                    member_letters[ni].add(gi)
                    placed = True
                    break
            if not placed:
                for gi in sorted(member_letters[ni]):
                    if _compatible(nj, letter_groups[gi]):
                        letter_groups[gi].add(nj)
                        member_letters[nj].add(gi)
                        placed = True
                        break
            if not placed:
                letter_groups.append({ni, nj})
                gi = len(letter_groups) - 1
                member_letters[ni].add(gi)
                member_letters[nj].add(gi)
    
    # Grup indekslerini harflere çevir (ilk grup = 'a', en büyük ortalama)
    letter_map = {gi: chr(ord("a") + gi) for gi in range(len(letter_groups))}
    group_labels = {
        name: "".join(sorted(letter_map[gi] for gi in member_letters[name])) or "-"
        for name in names
    }
    
    return {
        "test": "Duncan",
        "alpha": alpha,
        "anova_f_statistic": float(f_stat),
        "anova_p_value": float(anova_p),
        "mse": float(mse),
        "degrees_of_freedom_error": int(df_within),
        "standard_error": se,
        "critical_ranges": critical_ranges,
        "sorted_means": {names[i]: means[i] for i in range(k)},
        "group_labels": {name: group_labels[name] for name in reversed(names)},
        "significant_matrix": significant_matrix,
    }


def normality_test(data: pd.Series | list | np.ndarray, alpha: float = 0.05) -> dict:
    """Shapiro-Wilk normallik testi."""
    arr = np.asarray(data, dtype=float).flatten()
    arr = arr[~np.isnan(arr)]
    if len(arr) < 3:
        raise ValueError("Shapiro-Wilk için en az 3 geçerli veri gerekli")
    
    w_stat, p_value = stats.shapiro(arr)
    return {
        "test": "Shapiro-Wilk",
        "statistic": float(w_stat),
        "p_value": float(p_value),
        "normal_at_alpha": bool(p_value > alpha),
        "alpha": alpha,
        "n": int(len(arr)),
    }


def homogeneity_test(*groups: pd.Series | list | np.ndarray, alpha: float = 0.05) -> dict:
    """Levene varyans homojenliği testi."""
    arrays = []
    for g in groups:
        arr = np.asarray(g, dtype=float).flatten()
        arr = arr[~np.isnan(arr)]
        if len(arr) < 2:
            raise ValueError("Her grupta en az 2 geçerli veri olmalı")
        arrays.append(arr)
    if len(arrays) < 2:
        raise ValueError("En az 2 grup gerekli")
    
    w_stat, p_value = stats.levene(*arrays)
    return {
        "test": "Levene",
        "statistic": float(w_stat),
        "p_value": float(p_value),
        "homogeneous_at_alpha": bool(p_value > alpha),
        "alpha": alpha,
        "n_groups": int(len(arrays)),
    }


def mann_whitney_u(data1, data2, alternative: str = "two-sided") -> dict:
    """Mann-Whitney U testi (bağımsız iki örneklem, parametrik olmayan)."""
    d1 = np.asarray(data1, dtype=float).flatten()
    d2 = np.asarray(data2, dtype=float).flatten()
    d1, d2 = d1[~np.isnan(d1)], d2[~np.isnan(d2)]
    if len(d1) < 2 or len(d2) < 2:
        raise ValueError("Her iki grupta en az 2 geçerli veri olmalı")
    
    u_stat, p_value = stats.mannwhitneyu(d1, d2, alternative=alternative)
    # Etki boyutu: r = Z / sqrt(N) — U'dan Z'ye yaklaşık dönüşüm
    n1, n2 = len(d1), len(d2)
    mu_u = n1 * n2 / 2
    sigma_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z_approx = (u_stat - mu_u) / sigma_u if sigma_u > 0 else 0.0
    
    return {
        "test": "Mann-Whitney U",
        "u_statistic": float(u_stat),
        "p_value": float(p_value),
        "z_approx": float(z_approx),
        "effect_size_r": float(abs(z_approx) / np.sqrt(n1 + n2)) if (n1 + n2) > 0 else 0.0,
        "group1_median": float(np.median(d1)),
        "group2_median": float(np.median(d2)),
        "significant_at_005": bool(p_value < 0.05),
    }


def kruskal_wallis(*groups) -> dict:
    """Kruskal-Wallis testi (tek yönlü ANOVA'nın parametrik olmayan karşılığı)."""
    arrays = []
    for g in groups:
        arr = np.asarray(g, dtype=float).flatten()
        arr = arr[~np.isnan(arr)]
        if len(arr) < 1:
            raise ValueError("Her grupta en az 1 geçerli veri olmalı")
        arrays.append(arr)
    if len(arrays) < 2:
        raise ValueError("En az 2 grup gerekli")
    
    h_stat, p_value = stats.kruskal(*arrays)
    n_total = sum(len(a) for a in arrays)
    k = len(arrays)
    # Etki boyutu: eta-kare benzeri H / (N - 1)
    eta_sq_h = float(h_stat / (n_total - 1)) if n_total > 1 else 0.0
    
    return {
        "test": "Kruskal-Wallis",
        "h_statistic": float(h_stat),
        "p_value": float(p_value),
        "degrees_of_freedom": int(k - 1),
        "eta_squared_h": eta_sq_h,
        "group_medians": [float(np.median(a)) for a in arrays],
        "significant_at_005": bool(p_value < 0.05),
    }


def wilcoxon_test(data1, data2) -> dict:
    """Wilcoxon işaretli sıralar testi (eşleştirilmiş iki örneklem)."""
    d1 = np.asarray(data1, dtype=float).flatten()
    d2 = np.asarray(data2, dtype=float).flatten()
    if len(d1) != len(d2):
        raise ValueError("Eşleştirilmiş test için grup uzunlukları eşit olmalı")
    mask = ~(np.isnan(d1) | np.isnan(d2))
    d1, d2 = d1[mask], d2[mask]
    diffs = d1 - d2
    diffs = diffs[diffs != 0]
    if len(diffs) < 6:
        raise ValueError("Wilcoxon testi için en az 6 sıfırdan farklı fark önerilir")
    
    w_stat, p_value = stats.wilcoxon(d1, d2)
    return {
        "test": "Wilcoxon",
        "w_statistic": float(w_stat),
        "p_value": float(p_value),
        "median_difference": float(np.median(d1 - d2)),
        "n_pairs": int(len(diffs)),
        "significant_at_005": bool(p_value < 0.05),
    }


def friedman_test(matrix: pd.DataFrame | np.ndarray) -> dict:
    """Friedman testi (tekrarlı ölçüm / blok tasarımı, parametrik olmayan).
    
    matrix: satırlar = denekler/bloklar, sütunlar = uygulamalar.
    """
    df = pd.DataFrame(matrix)
    df = df.dropna()
    if len(df) < 2 or df.shape[1] < 2:
        raise ValueError("En az 2 denek ve 2 uygulama gerekli")
    
    chi2_stat, p_value = stats.friedmanchisquare(*[df[col].values for col in df.columns])
    n, k = df.shape
    # Kendall W: W = chi2 / (n * (k - 1))
    kendall_w = float(chi2_stat / (n * (k - 1))) if n * (k - 1) > 0 else 0.0
    
    return {
        "test": "Friedman",
        "chi_square_statistic": float(chi2_stat),
        "p_value": float(p_value),
        "degrees_of_freedom": int(k - 1),
        "kendall_w": kendall_w,
        "n_subjects": int(n),
        "n_treatments": int(k),
        "treatment_medians": {str(col): float(df[col].median()) for col in df.columns},
        "significant_at_005": bool(p_value < 0.05),
    }


# ---------------------------------------------------------------------------
# Premium Program parity: Frekans tabloları ve çapraz tablolar (Descriptive Statistics)
# ---------------------------------------------------------------------------


def frequencies(data: pd.DataFrame, columns: Optional[list] = None) -> dict:
    """Frekans tablosu (Premium Program: Descriptive Statistics → Frequencies).
    
    Her değer için sayı, yüzde, geçerli yüzde ve kümülatif yüzde üretir.
    columns verilmezse kategorik (object/category) sütunlar kullanılır.
    """
    if columns is None:
        columns = list(data.select_dtypes(include=["object", "category", "bool"]).columns)
    if not columns:
        raise ValueError("Frekans tablosu için kategorik sütun bulunamadı")
    missing = [c for c in columns if c not in data.columns]
    if missing:
        raise ValueError(f"Sütunlar bulunamadı: {missing}")
    
    results = {}
    for col in columns:
        series = data[col]
        n_total = len(series)
        n_missing = int(series.isna().sum())
        valid = series.dropna()
        counts = valid.value_counts()
        
        rows = []
        cumulative = 0.0
        for value, count in counts.items():
            pct = float(count) / n_total * 100 if n_total else 0.0
            valid_pct = float(count) / len(valid) * 100 if len(valid) else 0.0
            cumulative += valid_pct
            rows.append({
                "value": value if not isinstance(value, float) else float(value),
                "count": int(count),
                "percent": pct,
                "valid_percent": valid_pct,
                "cumulative_percent": float(cumulative),
            })
        
        results[str(col)] = {
            "table": rows,
            "n_valid": int(len(valid)),
            "n_missing": n_missing,
            "n_total": n_total,
            "n_unique": int(counts.size),
            "mode": counts.index[0] if counts.size else None,
        }
    
    return results


def crosstabs(
    data: pd.DataFrame,
    row_col: str,
    col_col: str,
    expected_min_warning: float = 5.0,
) -> dict:
    """Çapraz tablo + ki-kare bağımsızlık testi
    (Premium Program: Descriptive Statistics → Crosstabs).
    
    Gözlenen/beklenen frekanslar, ki-kare istatistiği, serbestlik derecesi,
    p-değeri, Cramer's V ve phi katsayısı döndürülür.
    """
    if row_col not in data.columns or col_col not in data.columns:
        raise ValueError(f"Sütun bulunamadı: {row_col}/{col_col}")
    
    valid = data[[row_col, col_col]].dropna()
    if len(valid) < 4:
        raise ValueError("Çapraz tablo için yeterli veri yok")
    
    observed = pd.crosstab(valid[row_col], valid[col_col])
    if observed.shape[0] < 2 or observed.shape[1] < 2:
        raise ValueError("Her iki değişken de en az 2 kategori içermeli")
    
    # Premium Program'ın birincil "Pearson Chi-Square" satırıyla uyum için düzeltmesiz
    chi2, p_value, dof, expected = stats.chi2_contingency(observed, correction=False)
    expected_df = pd.DataFrame(expected, index=observed.index, columns=observed.columns)
    
    n = int(observed.values.sum())
    k_min = min(observed.shape) - 1
    cramers_v = float(np.sqrt(chi2 / (n * k_min))) if n * k_min > 0 else 0.0
    phi = float(np.sqrt(chi2 / n)) if n > 0 else 0.0
    low_expected_cells = int((expected_df < expected_min_warning).values.sum())
    
    return {
        "observed": observed,
        "expected": expected_df,
        "chi_square": float(chi2),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "cramers_v": cramers_v,
        "phi": phi,
        "n": n,
        "low_expected_cells": low_expected_cells,
        "expected_cell_warning": bool(low_expected_cells > 0),
        "significant_at_005": bool(p_value < 0.05),
    }


# ---------------------------------------------------------------------------
# Premium Program parity: ROC eğrisi, Bootstrapping, Loglinear modeller
# ---------------------------------------------------------------------------


def roc_curve(actual: pd.Series | np.ndarray,
              predicted: pd.Series | np.ndarray) -> dict:
    """ROC eğrisi ve AUC (Premium Program: Analyze → ROC Curve).
    
    actual: 0/1 gerçek sınıf; predicted: sürekli skor/olasılık.
    AUC trapez yöntemiyle, Youden J ile optimum eşik bulunur.
    """
    a = np.asarray(actual, dtype=float).flatten()
    p = np.asarray(predicted, dtype=float).flatten()
    mask = ~(np.isnan(a) | np.isnan(p))
    a, p = a[mask], p[mask]
    if len(a) < 4:
        raise ValueError("ROC için en az 4 geçerli gözlem gerekli")
    if not np.isin(a, [0, 1]).all():
        raise ValueError("actual yalnızca 0/1 değerleri alabilir")
    
    n_pos = int((a == 1).sum())
    n_neg = int((a == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Her iki sınıf da örnek içermeli")
    
    order = np.argsort(-p, kind="stable")
    a_sorted = a[order]
    p_sorted = p[order]
    
    tp = np.cumsum(a_sorted == 1)
    fp = np.cumsum(a_sorted == 0)
    tpr = tp / n_pos
    fpr = fp / n_neg
    
    # (0,0) başlangıç noktası
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    thresholds = np.concatenate([[p_sorted[0] + 1.0], p_sorted])
    
    auc = float(np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid")
                else np.trapz(tpr, fpr))
    
    # Youden J = TPR - FPR; maksimum noktadaki eşik optimumdur
    j = tpr - fpr
    idx = int(np.argmax(j))
    
    return {
        "auc": auc,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "optimal_threshold": float(thresholds[idx]),
        "youden_j": float(j[idx]),
        "sensitivity_at_optimal": float(tpr[idx]),
        "specificity_at_optimal": float(1.0 - fpr[idx]),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "interpretation": (
            "Mükemmel" if auc >= 0.9 else
            "İyi" if auc >= 0.8 else
            "Orta" if auc >= 0.7 else
            "Zayıf" if auc >= 0.6 else "Başarısız"
        ),
    }


def bootstrap_statistic(
    data: pd.Series | np.ndarray,
    statistic: Optional[Callable] = None,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: Optional[int] = None,
) -> dict:
    """Bootstrap güven aralığı (Premium Program: Analyze → Bootstrapping).
    
    statistic: tek diziden skaler üreten fonksiyon (varsayılan: ortalama).
    Yüzdelik (percentile) yöntemiyle GA hesaplanır.
    """
    if statistic is None:
        statistic = np.mean
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap en az 100 olmalı")
    if not 0.0 < ci < 1.0:
        raise ValueError("ci (0, 1) aralığında olmalı")
    
    arr = np.asarray(data, dtype=float).flatten()
    arr = arr[~np.isnan(arr)]
    if len(arr) < 3:
        raise ValueError("Bootstrap için en az 3 geçerli gözlem gerekli")
    
    rng = np.random.default_rng(seed)
    n = len(arr)
    estimates = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = arr[rng.integers(0, n, size=n)]
        estimates[i] = statistic(sample)
    
    lower = float(np.percentile(estimates, (1 - ci) / 2 * 100))
    upper = float(np.percentile(estimates, (1 + ci) / 2 * 100))
    
    return {
        "point_estimate": float(statistic(arr)),
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_level": float(ci),
        "bootstrap_std": float(np.std(estimates, ddof=1)),
        "n_bootstrap": int(n_bootstrap),
        "n_obs": int(n),
    }


def loglinear_analysis(data: pd.DataFrame, row_col: str, col_col: str) -> dict:
    """Log-lineer model (Premium Program: Analyze → Loglinear).
    
    Çapraz tablo hücre sayılarına Poisson GLM uydurulur:
    bağımsızlık modeli (satır + sütun) doymuş modelle (satır × sütun)
    olabilirlik oranı testiyle karşılaştırılır.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    
    if row_col not in data.columns or col_col not in data.columns:
        raise ValueError(f"Sütun bulunamadı: {row_col}/{col_col}")
    
    table = pd.crosstab(data[row_col].dropna(), data[col_col].dropna())
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError("Her iki değişken de en az 2 kategori içermeli")
    
    long_df = table.stack().reset_index()
    long_df.columns = ["satir", "sutun", "sayi"]
    
    fam = sm.families.Poisson()
    model_ind = smf.glm("sayi ~ C(satir) + C(sutun)", data=long_df, family=fam).fit()
    model_sat = smf.glm("sayi ~ C(satir) * C(sutun)", data=long_df, family=fam).fit()
    
    df_diff = (table.shape[0] - 1) * (table.shape[1] - 1)
    lrt_stat = float(2 * (model_sat.llf - model_ind.llf))
    p_value = float(stats.chi2.sf(lrt_stat, df=df_diff))
    
    # Beklenen frekanslar (bağımsızlık modeli üzerinden)
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.values.sum()
    
    return {
        "model": "Loglinear (Poisson GLM)",
        "likelihood_ratio_chi2": lrt_stat,
        "p_value": p_value,
        "degrees_of_freedom": int(df_diff),
        "independence_rejected": bool(p_value < 0.05),
        "observed": table,
        "expected": pd.DataFrame(expected, index=table.index, columns=table.columns),
        "aic_independence": float(model_ind.aic),
        "aic_saturated": float(model_sat.aic),
        "n": int(table.values.sum()),
    }


# ---------------------------------------------------------------------------
# Premium Program parity: Multinomial/Ordinal Lojistik, Discriminant, Correspondence
# ---------------------------------------------------------------------------


def _check_columns(data: pd.DataFrame, columns: list) -> None:
    missing = [c for c in columns if c not in data.columns]
    if missing:
        raise ValueError(f"Sütunlar bulunamadı: {missing}")


def _validate_numeric_predictors(valid: pd.DataFrame, predictors: list) -> None:
    non_numeric = [c for c in predictors
                   if not pd.api.types.is_numeric_dtype(valid[c])]
    if non_numeric:
        raise ValueError(f"Sayısal olmayan açıklayıcı değişkenler: {non_numeric}")
    constant = [c for c in predictors if valid[c].std(ddof=0) == 0]
    if constant:
        raise ValueError(f"Sabit (varyansı sıfır) değişkenler: {constant}")


def multinomial_logistic_regression(
    data: pd.DataFrame,
    dependent_col: str,
    predictors: list[str],
) -> dict:
    """Multinomial lojistik regresyon (Premium Program: Analyze → Regression →
    Multinomial Logistic Regression).

    İkiden fazla kategorili (sırasız) yanıt değişkeni için. Referans
    kategori sıralı ilk kategoridir; katsayılar, olabilirlik oranı testi,
    odds oranları ve sınıflandırma başarısı döndürülür.
    """
    predictors = list(predictors)
    _check_columns(data, [dependent_col] + predictors)
    if not predictors:
        raise ValueError("En az bir açıklayıcı değişken gerekli")

    valid = data[[dependent_col] + predictors].dropna()
    if len(valid) < len(predictors) * 3 + 10:
        raise ValueError("Multinomial lojistik için yeterli veri yok")
    _validate_numeric_predictors(valid, predictors)

    categories = sorted(valid[dependent_col].unique(), key=str)
    if len(categories) < 2:
        raise ValueError("Bağımlı değişken en az 2 kategori içermeli")

    code_map = {cat: i for i, cat in enumerate(categories)}
    y = valid[dependent_col].map(code_map).to_numpy()
    X = sm.add_constant(valid[predictors].to_numpy(dtype=float))

    model = sm.MNLogit(y, X).fit(disp=0)

    # Olabilirlik oranı testi: null modelin log-olabilirliği analitik olarak
    # (yalnızca sabit terim → kategori oranları) hesaplanır
    counts = np.bincount(y, minlength=len(categories))
    props = counts / len(y)
    llnull = float(np.sum(counts * np.log(props)))
    lr_stat = float(2 * (model.llf - llnull))
    lr_df = int(len(predictors) * (len(categories) - 1))
    p_value = float(stats.chi2.sf(lr_stat, df=lr_df))

    exog_names = ["const"] + predictors
    reference = str(categories[0])
    # MNLogit params biçimi (K, J-1): satırlar = değişkenler, sütunlar = kategoriler
    params = np.asarray(model.params).T
    coefficients: dict = {}
    odds_ratios: dict = {}
    for row_idx, cat in enumerate(categories[1:]):
        cat_params = {exog_names[j]: float(params[row_idx, j])
                      for j in range(len(exog_names))}
        coefficients[str(cat)] = cat_params
        odds_ratios[str(cat)] = {k: float(np.exp(v)) for k, v in cat_params.items()}

    pred_codes = np.asarray(model.predict(X)).argmax(axis=1)
    accuracy = float((pred_codes == y).mean())

    return {
        "model": "Multinomial Logistic",
        "reference_category": reference,
        "categories": [str(c) for c in categories],
        "coefficients": coefficients,
        "odds_ratios": odds_ratios,
        "pseudo_r_squared": float(1 - model.llf / llnull),
        "lr_chi_square": lr_stat,
        "degrees_of_freedom": lr_df,
        "p_value": p_value,
        "llf": float(model.llf),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "classification_accuracy": accuracy,
        "n": int(len(valid)),
        "significant_at_005": bool(p_value < 0.05),
    }


def ordinal_logistic_regression(
    data: pd.DataFrame,
    dependent_col: str,
    predictors: list[str],
) -> dict:
    """Ordinal (sıralı) lojistik regresyon — kümülatif logit
    (Premium Program: Analyze → Regression → Ordinal / PLUM).

    Model: logit(P(Y ≤ j)) = eşik_j − β'x. Premium Program PLUM işaret uzlaşımıyla
    uyumlu olarak katsayılar, eşikler, odds oranları ve olabilirlik
    oranı testi döndürülür.
    """
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    predictors = list(predictors)
    _check_columns(data, [dependent_col] + predictors)
    if not predictors:
        raise ValueError("En az bir açıklayıcı değişken gerekli")

    valid = data[[dependent_col] + predictors].dropna()
    if len(valid) < len(predictors) * 3 + 10:
        raise ValueError("Ordinal lojistik için yeterli veri yok")
    _validate_numeric_predictors(valid, predictors)

    categories = sorted(valid[dependent_col].unique(), key=str)
    if len(categories) < 3:
        raise ValueError("Ordinal model için en az 3 kategori gerekli")

    code_map = {cat: i for i, cat in enumerate(categories)}
    y = valid[dependent_col].map(code_map).to_numpy()
    X = valid[predictors].to_numpy(dtype=float)

    model = OrderedModel(y, X, distr="logit").fit(method="bfgs", disp=0)
    null_model = OrderedModel(y, None, distr="logit").fit(method="bfgs", disp=0)

    lr_stat = float(2 * (model.llf - null_model.llf))
    lr_df = int(len(predictors))
    p_value = float(stats.chi2.sf(lr_stat, df=lr_df))

    params = np.asarray(model.params)
    coef_params = params[: len(predictors)]
    threshold_params = params[len(predictors):]

    coefficients = {predictors[i]: float(coef_params[i])
                    for i in range(len(predictors))}
    thresholds = {
        f"{str(categories[i])} | {str(categories[i + 1])}": float(threshold_params[i])
        for i in range(len(categories) - 1)
    }
    odds_ratios = {k: float(np.exp(v)) for k, v in coefficients.items()}

    return {
        "model": "Ordinal Logistic (cumulative logit)",
        "categories": [str(c) for c in categories],
        "coefficients": coefficients,
        "thresholds": thresholds,
        "odds_ratios": odds_ratios,
        "pseudo_r_squared": float(1 - model.llf / null_model.llf),
        "lr_chi_square": lr_stat,
        "degrees_of_freedom": lr_df,
        "p_value": p_value,
        "llf": float(model.llf),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "n": int(len(valid)),
        "significant_at_005": bool(p_value < 0.05),
    }


def discriminant_analysis(
    data: pd.DataFrame,
    group_col: str,
    predictors: list[str],
) -> dict:
    """Doğrusal ayrımsama analizi — Fisher LDA
    (Premium Program: Analyze → Classify → Discriminant Analysis).

    Wilks lambda + Bartlett ki-kare yaklaşıklığı, kanonik korelasyonlar,
    grup sentroidleri, sınıflandırma matrisi ve genel doğruluk döndürülür.
    Önsel olasılıklar grup oranlarından alınır (Premium Program varsayılanı).
    """
    predictors = list(predictors)
    _check_columns(data, [group_col] + predictors)
    if not predictors:
        raise ValueError("En az bir açıklayıcı değişken gerekli")

    valid = data[[group_col] + predictors].dropna()
    _validate_numeric_predictors(valid, predictors)

    groups = sorted(valid[group_col].unique(), key=str)
    if len(groups) < 2:
        raise ValueError("En az 2 grup gerekli")

    X = valid[predictors].to_numpy(dtype=float)
    labels = valid[group_col].to_numpy()
    n, p = X.shape
    g = len(groups)
    if n <= p + g:
        raise ValueError("Discriminant analizi için yeterli veri yok "
                         "(n > değişken sayısı + grup sayısı olmalı)")
    for grp in groups:
        if int((labels == grp).sum()) < 2:
            raise ValueError(f"'{grp}' grubunda en az 2 gözlem gerekli")

    grand_mean = X.mean(axis=0)

    # Toplam, grup-içi ve gruplar-arası kareler/carpımlar matrisleri
    t_ss = (X - grand_mean).T @ (X - grand_mean)
    w_ss = np.zeros((p, p))
    b_ss = np.zeros((p, p))
    group_means = {}
    for grp in groups:
        x_g = X[labels == grp]
        mean_g = x_g.mean(axis=0)
        group_means[grp] = mean_g
        centered = x_g - mean_g
        w_ss += centered.T @ centered
        diff = (mean_g - grand_mean).reshape(-1, 1)
        b_ss += len(x_g) * (diff @ diff.T)

    # Kanonik kökler: genelleştirilmiş özdeğer problemi B v = λ T v
    from scipy.linalg import eigh
    
    n_funcs = min(g - 1, p)
    eigvals, eigvecs = eigh(b_ss, t_ss)
    eig_order = np.argsort(eigvals)[::-1][:n_funcs]
    eig_funcs = np.clip(eigvals[eig_order], 0.0, 1.0 - 1e-12)
    canonical_correlations = [float(np.sqrt(ev)) for ev in eig_funcs]

    wilks_lambda = float(np.prod(1.0 - eig_funcs))
    # Bartlett yaklaşımı: ki-kare = -ln(λ) * (n - 1 - (p + g) / 2)
    chi_square = float(-np.log(wilks_lambda) * (n - 1 - (p + g) / 2))
    df = int(p * (g - 1))
    p_value = float(stats.chi2.sf(chi_square, df=df))

    total_eig = float(np.sum(eig_funcs))
    percent_variance = [float(ev / total_eig * 100) for ev in eig_funcs] \
        if total_eig > 0 else [0.0] * n_funcs

    # Kanonik sentroidler (grup ortalamaları, kanonik uzayda)
    V = eigvecs[:, eig_order]
    canonical_centroids = {
        str(grp): [float(group_means[grp] @ v) for v in V.T]
        for grp in groups
    }

    # Doğrusal sınıflandırma (havuzlanmış kovaryans + oransal önseller)
    w_pooled = w_ss / (n - g)
    w_inv = np.linalg.inv(w_pooled)
    log_priors = np.log(np.array([(labels == grp).sum() for grp in groups]) / n)

    def _scores(x: np.ndarray) -> np.ndarray:
        return np.array([
            float(x @ w_inv @ group_means[grp]
                  - 0.5 * group_means[grp] @ w_inv @ group_means[grp]
                  + log_priors[k])
            for k, grp in enumerate(groups)
        ])

    predicted = [groups[int(np.argmax(_scores(x)))] for x in X]
    correct = np.array([pr == ob for pr, ob in zip(predicted, labels)])

    group_stats = []
    for grp in groups:
        mask = labels == grp
        n_grp = int(mask.sum())
        n_correct = int(correct[mask].sum())
        group_stats.append({
            "group": str(grp),
            "n": n_grp,
            "n_correct": n_correct,
            "percent_correct": float(n_correct / n_grp * 100),
        })

    return {
        "model": "Fisher Linear Discriminant",
        "groups": [str(grp) for grp in groups],
        "group_centroids": {
            str(grp): {predictors[j]: float(group_means[grp][j])
                       for j in range(p)}
            for grp in groups
        },
        "canonical_correlations": canonical_correlations,
        "eigenvalues": [float(ev) for ev in eig_funcs],
        "percent_of_variance": percent_variance,
        "canonical_centroids": canonical_centroids,
        "wilks_lambda": wilks_lambda,
        "chi_square": chi_square,
        "degrees_of_freedom": df,
        "p_value": p_value,
        "significant_at_005": bool(p_value < 0.05),
        "classification": group_stats,
        "overall_accuracy": float(correct.mean()),
        "n": int(n),
    }


def correspondence_analysis(
    data: pd.DataFrame,
    row_col: str,
    col_col: str,
    n_dims: int = 2,
) -> dict:
    """Uyuşum (Correspondence) analizi
    (Premium Program: Analyze → Dimension Reduction → Correspondence Analysis).

    Çapraz tablonun standartlaştırılmış artıklarının SVD ayrışımıyla
    satır/sütun ana koordinatları, asal eylemsizlikler (özdeğerler),
    kalite (cos²) ve katkılar hesaplanır.
    """
    if n_dims < 1:
        raise ValueError("n_dims en az 1 olmalı")
    _check_columns(data, [row_col, col_col])

    valid = data[[row_col, col_col]].dropna()
    if len(valid) < 4:
        raise ValueError("Uyuşum analizi için yeterli veri yok")

    table = pd.crosstab(valid[row_col], valid[col_col])
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError("Her iki değişken de en az 2 kategori içermeli")

    n = int(table.values.sum())
    P = table.values / n
    row_mass = P.sum(axis=1)
    col_mass = P.sum(axis=0)

    expected = np.outer(row_mass, col_mass)
    resid = (P - expected) / np.sqrt(expected)

    U, s, Vt = np.linalg.svd(resid, full_matrices=False)
    k_max = min(table.shape) - 1
    k = min(n_dims, k_max)

    principal_inertias = s[:k_max] ** 2
    total_inertia = float(principal_inertias.sum())
    chi_square = float(n * total_inertia)
    dof = (table.shape[0] - 1) * (table.shape[1] - 1)
    p_value = float(stats.chi2.sf(chi_square, df=dof))
    explained_percent = [float(pi / total_inertia * 100) for pi in principal_inertias]

    # Ana koordinatlar: satır F = D_r^-1 U Σ, sütun G = D_c^-1 V Σ
    dim_names = [f"Boyut_{i + 1}" for i in range(k)]
    row_coords = pd.DataFrame(
        (U[:, :k] * s[:k]) / row_mass[:, None],
        index=table.index.astype(str),
        columns=dim_names,
    )
    col_coords = pd.DataFrame(
        (Vt[:k, :].T * s[:k]) / col_mass[:, None],
        index=table.columns.astype(str),
        columns=dim_names,
    )

    # Kalite (cos²): ana koordinatların kareleri toplamı / ki-kare mesafesi²
    # d²_satır = Σ_j resid²_ij / r_i ; d²_sütun = Σ_i resid²_ij / c_j
    row_chi2_dist = ((resid ** 2) / col_mass[None, :]).sum(axis=1) / row_mass
    col_chi2_dist = ((resid ** 2) / row_mass[:, None]).sum(axis=0) / col_mass
    row_quality = {
        str(cat): float((row_coords.loc[cat] ** 2).sum() / d2) if d2 > 0 else 0.0
        for cat, d2 in zip(table.index, row_chi2_dist)
    }
    col_quality = {
        str(cat): float((col_coords.loc[cat] ** 2).sum() / d2) if d2 > 0 else 0.0
        for cat, d2 in zip(table.columns, col_chi2_dist)
    }

    # Boyut katkısı: satır/sütun kitlesi * koordinat² / asal eylemsizlik
    row_contributions = pd.DataFrame(
        row_mass[:, None] * row_coords.values ** 2 / principal_inertias[:k],
        index=row_coords.index, columns=row_coords.columns,
    )
    col_contributions = pd.DataFrame(
        col_mass[:, None] * col_coords.values ** 2 / principal_inertias[:k],
        index=col_coords.index, columns=col_coords.columns,
    )

    return {
        "model": "Correspondence Analysis",
        "observed": table,
        "eigenvalues": [float(pi) for pi in principal_inertias],
        "inertia_per_dimension": [float(pi) for pi in principal_inertias],
        "explained_percent": explained_percent,
        "explained_inertia_pct": {
            f"Boyut_{i + 1}": pct for i, pct in enumerate(explained_percent)
        },
        "total_inertia": total_inertia,
        "chi_square": chi_square,
        "degrees_of_freedom": int(dof),
        "p_value": p_value,
        "independence_rejected": bool(p_value < 0.05),
        "n_dimensions": int(k),
        "n_dims": int(k),
        "row_coordinates": row_coords,
        "column_coordinates": col_coords,
        "row_masses": {str(cat): float(m) for cat, m in zip(table.index, row_mass)},
        "column_masses": {str(cat): float(m) for cat, m in zip(table.columns, col_mass)},
        "row_quality": row_quality,
        "column_quality": col_quality,
        "row_contributions": row_contributions,
        "column_contributions": col_contributions,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Premium Program parity: kalan Nonparametric Tests + Partial Correlation
# ---------------------------------------------------------------------------


def binomial_test(data: pd.Series | np.ndarray, test_proportion: float = 0.5,
                  success_value=None) -> dict:
    """Binom testi (Premium Program: Nonparametric Tests → Binomial).

    Dikotom bir değişkenin gözlenen oranını test oranıyla karşılaştırır.
    success_value verilmezse en sık görülen değer 'başarı' kabul edilir.
    """
    s = pd.Series(data).dropna()
    if len(s) < 2:
        raise ValueError("Binom testi için en az 2 gözlem gerekli")
    if not 0.0 < test_proportion < 1.0:
        raise ValueError("test_proportion (0,1) aralığında olmalı")
    
    if s.nunique() > 2:
        raise ValueError("Binom testi için en fazla 2 kategori gerekli")
    
    if success_value is None:
        success_value = s.value_counts().index[0]
    
    k = int((s == success_value).sum())
    n = int(len(s))
    p_obs = k / n
    p_value = float(stats.binomtest(k, n, p=test_proportion).pvalue)
    
    return {
        "test": "Binomial",
        "category": success_value,
        "n_success": k,
        "n_total": n,
        "observed_proportion": float(p_obs),
        "test_proportion": float(test_proportion),
        "p_value": p_value,
        "significant_at_005": bool(p_value < 0.05),
    }


def runs_test(data: pd.Series | np.ndarray, cut_point: Optional[float] = None) -> dict:
    """Koşu (runs) testi (Premium Program: Nonparametric Tests → Runs).

    Dizinin kesim noktasına göre üstünde/altında olma sırasının
    rastgeleliğini sınar.
    """
    arr = np.asarray(data, dtype=float).flatten()
    arr = arr[~np.isnan(arr)]
    if len(arr) < 4:
        raise ValueError("Koşu testi için en az 4 gözlem gerekli")
    
    cut = float(np.median(arr)) if cut_point is None else float(cut_point)
    binary = arr >= cut
    
    n1 = int(binary.sum())
    n2 = len(binary) - n1
    if n1 == 0 or n2 == 0:
        raise ValueError("Kesim noktası her iki tarafta da gözlem bırakmalı")
    
    runs = 1 + int(np.sum(binary[1:] != binary[:-1]))
    expected = 1 + 2 * n1 * n2 / (n1 + n2)
    var = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / \
          ((n1 + n2) ** 2 * (n1 + n2 - 1))
    if var <= 0:
        raise ValueError("Koşu testi varyansı hesaplanamadı")
    
    z = (runs - expected) / np.sqrt(var)
    p_value = float(2 * stats.norm.sf(abs(z)))
    
    return {
        "test": "Runs",
        "cut_point": cut,
        "n_runs": runs,
        "expected_runs": float(expected),
        "z_statistic": float(z),
        "p_value": p_value,
        "n_above": n1,
        "n_below": n2,
        "random_at_005": bool(p_value >= 0.05),
    }


def kolmogorov_smirnov_one_sample(data: pd.Series | np.ndarray,
                                  mean: Optional[float] = None,
                                  std: Optional[float] = None) -> dict:
    """Tek örneklem Kolmogorov-Smirnov testi (Premium Program Legacy: 1-Sample K-S).

    Verinin normal dağılıma uygunluğunu sınar; parametreler verilmezse
    örneklem ortalaması/ss kullanılır (Premium Program davranışı).
    """
    arr = np.asarray(data, dtype=float).flatten()
    arr = arr[~np.isnan(arr)]
    if len(arr) < 5:
        raise ValueError("K-S testi için en az 5 gözlem gerekli")
    
    mu = float(np.mean(arr)) if mean is None else float(mean)
    sd = float(np.std(arr, ddof=1)) if std is None else float(std)
    if sd <= 0:
        raise ValueError("Standart sapma sıfırdan büyük olmalı")
    
    stat_result = stats.kstest(arr, "norm", args=(mu, sd))
    
    return {
        "test": "One-Sample Kolmogorov-Smirnov",
        "statistic": float(stat_result.statistic),
        "p_value": float(stat_result.pvalue),
        "test_mean": mu,
        "test_std": sd,
        "n": int(len(arr)),
        "normal_at_005": bool(stat_result.pvalue >= 0.05),
    }


def cochran_q_test(matrix: pd.DataFrame | np.ndarray) -> dict:
    """Cochran Q testi (Premium Program: Nonparametric → K Related Samples, dikotom veri).

    İlişkili k örneklemin dikotom (0/1) yanıtlarında farkı sınar.
    Satırlar = denekler, sütunlar = koşullar.
    """
    df = pd.DataFrame(matrix)
    if not df.apply(lambda c: c.dropna().isin([0, 1]).all()).all():
        raise ValueError("Cochran Q yalnızca 0/1 verisiyle çalışır")
    df = df.dropna()
    n, k = df.shape
    if k < 2 or n < 3:
        raise ValueError("En az 2 koşul ve 3 denek gerekli")
    
    row_sums = df.sum(axis=1).to_numpy()
    col_sums = df.sum(axis=0).to_numpy()
    
    denom = k * np.sum(row_sums) - np.sum(row_sums ** 2)
    if denom <= 0:
        raise ValueError("Cochran Q paydası sıfır; veri çeşitliliği yok")
    
    q_stat = float((k - 1) * (k * np.sum(col_sums ** 2) - np.sum(col_sums) ** 2) / denom)
    dof = k - 1
    p_value = float(stats.chi2.sf(q_stat, df=dof))
    
    return {
        "test": "Cochran Q",
        "q_statistic": q_stat,
        "p_value": p_value,
        "degrees_of_freedom": dof,
        "n_subjects": int(n),
        "n_conditions": int(k),
        "condition_proportions": (col_sums / n).tolist(),
        "significant_at_005": bool(p_value < 0.05),
    }


def partial_correlation(data: pd.DataFrame, x_col: str, y_col: str,
                        control_cols: list, method: str = "pearson") -> dict:
    """Kısmi korelasyon (Premium Program: Correlate → Partial).

    Kontrol değişkenlerinin etkisi arındırıldıktan sonra x-y ilişkisi.
    Artık regresyonu yöntemiyle hesaplanır.
    """
    _check_columns(data, [x_col, y_col] + list(control_cols))
    valid = data[[x_col, y_col] + list(control_cols)].dropna()
    if len(valid) < len(control_cols) + 5:
        raise ValueError("Kısmi korelasyon için yeterli veri yok")
    
    controls = sm.add_constant(valid[control_cols].astype(float))
    
    def _residual(col):
        model = sm.OLS(valid[col].astype(float), controls).fit()
        return model.resid
    
    res_x = _residual(x_col)
    res_y = _residual(y_col)
    
    if method == "pearson":
        r, p = stats.pearsonr(res_x, res_y)
    elif method == "spearman":
        r, p = stats.spearmanr(res_x, res_y)
    else:
        raise ValueError(f"Desteklenmeyen yöntem: {method}")
    
    dof = len(valid) - len(control_cols) - 2
    return {
        "partial_correlation": float(r),
        "p_value": float(p),
        "degrees_of_freedom": int(dof),
        "method": method,
        "control_variables": list(control_cols),
        "n": int(len(valid)),
        "significant_at_005": bool(p < 0.05),
    }


# ---------------------------------------------------------------------------
# Premium Program parity wave 4: Tablolar, Sınıflandırma, Boyut İndirgeme,
# Çoklu Yanıt, Mesafeler, Ortalama Raporları
# ---------------------------------------------------------------------------


def custom_tables(
    data: pd.DataFrame,
    rows: list[str],
    columns: Optional[list[str]] = None,
    values: Optional[str] = None,
    statistics: Optional[list[str]] = None,
) -> dict:
    """Özel tablolar (Premium Program: Analyze → Tables → Custom Tables).

    Satır/sütun kategorik değişkenlerine göre sayısal özet değişkeninin
    seçilen istatistiklerini içeren çok indeksli tablo üretir.
    """
    rows = list(rows)
    columns = list(columns) if columns else []
    allowed = {"count", "mean", "median", "std", "sum", "percent"}
    statistics = list(statistics) if statistics else ["count", "mean"]
    unknown = [s for s in statistics if s not in allowed]
    if unknown:
        raise ValueError(f"Desteklenmeyen istatistikler: {unknown}")
    _check_columns(data, rows + columns + ([values] if values else []))
    if values is None and any(s != "count" for s in statistics):
        raise ValueError("count dışı istatistikler için values sütunu gerekli")

    valid = data[rows + columns + ([values] if values else [])].dropna()
    if len(valid) == 0:
        raise ValueError("Tablo için geçerli gözlem yok")

    grouped = valid.groupby(rows + columns, observed=True)
    frames = {}
    for stat in statistics:
        if stat == "count":
            frames[stat] = grouped.size() if values is None else grouped[values].count()
        elif stat == "percent":
            sizes = grouped.size()
            frames[stat] = (sizes / len(valid) * 100)
        else:
            frames[stat] = getattr(grouped[values], stat)()
    table = pd.concat(frames.values(), axis=1, keys=statistics)

    return {
        "table": table,
        "rows": rows,
        "columns": columns,
        "values": values,
        "statistics": statistics,
        "n_cases": int(len(valid)),
    }


def means_report(
    data: pd.DataFrame,
    response_col: str,
    group_cols: list[str],
) -> dict:
    """Ortalama raporu (Premium Program: Analyze → Compare Means → Means).

    Katmanlı grup değişkenlerine göre ortalama/standart sapma/vaka sayısı
    tablosu; genel toplam satırı dahildir.
    """
    group_cols = list(group_cols)
    _check_columns(data, [response_col] + group_cols)
    if not group_cols:
        raise ValueError("En az bir grup değişkeni gerekli")

    valid = data[[response_col] + group_cols].dropna()
    if len(valid) < 2:
        raise ValueError("Ortalama raporu için yeterli veri yok")

    def _summary(df: pd.DataFrame) -> dict:
        return {
            "n": int(len(df)),
            "mean": float(df[response_col].mean()),
            "std": float(df[response_col].std(ddof=1)) if len(df) > 1 else 0.0,
        }

    grand = {"Genel": _summary(valid)}
    layers = {}
    for col in group_cols:
        layers[col] = {
            str(name): _summary(grp) for name, grp in valid.groupby(col, observed=True)
        }

    return {
        "response": response_col,
        "grand_total": grand["Genel"],
        "layers": layers,
        "n": int(len(valid)),
    }


def case_summaries(
    data: pd.DataFrame,
    columns: Optional[list[str]] = None,
    n_cases: int = 100,
) -> dict:
    """Vaka özetleri (Premium Program: Analyze → Reports → Case Summaries)."""
    if n_cases < 1:
        raise ValueError("n_cases en az 1 olmalı")
    cols = list(columns) if columns else list(data.columns)
    _check_columns(data, cols)
    valid = data[cols].dropna(how="all")
    shown = valid.head(n_cases)
    return {
        "cases": shown.reset_index(drop=True),
        "n_shown": int(len(shown)),
        "n_total": int(len(valid)),
        "columns": cols,
    }


def ratio_statistics(
    data: pd.DataFrame,
    numerator_col: str,
    denominator_col: str,
    group_col: Optional[str] = None,
) -> dict:
    """Oran istatistikleri (Premium Program: Analyze → Descriptive Statistics → Ratios).

    Oranların ortalaması, medyanı, değişim katsayısı (COV), ortalama
    mutlak sapma (AAD) ve fiyat ilişkili diferansiyel (PRD) raporlanır.
    """
    cols = [numerator_col, denominator_col] + ([group_col] if group_col else [])
    _check_columns(data, cols)
    valid = data[cols].dropna()
    if len(valid) < 2:
        raise ValueError("Oran istatistikleri için yeterli veri yok")
    if (valid[denominator_col] == 0).any():
        raise ValueError("Payda sıfır içeremez")

    ratios = valid[numerator_col] / valid[denominator_col]
    mean_r = float(ratios.mean())
    std_r = float(ratios.std(ddof=1))
    aad = float((ratios - mean_r).abs().mean())

    result = {
        "mean_ratio": mean_r,
        "median_ratio": float(ratios.median()),
        "cov": float(std_r / mean_r) if mean_r != 0 else float("nan"),
        "aad": aad,
        "n": int(len(valid)),
    }

    if group_col is not None:
        per_group = {}
        for name, grp in valid.groupby(group_col, observed=True):
            r_g = grp[numerator_col] / grp[denominator_col]
            per_group[str(name)] = {
                "mean_ratio": float(r_g.mean()),
                "median_ratio": float(r_g.median()),
                "n": int(len(grp)),
            }
        sizes = valid.groupby(group_col, observed=True).size()
        weighted_mean = float(np.average(
            [per_group[str(k)]["mean_ratio"] for k in sizes.index],
            weights=sizes.values))
        result["groups"] = per_group
        # PRD: ağırlıklı ortalama oran / ağırlıksız ortalama oran
        result["prd"] = float(weighted_mean / mean_r) if mean_r != 0 else float("nan")

    return result


def distance_matrix(
    data: pd.DataFrame,
    variables: list[str],
    measure: str = "euclidean",
    between: str = "cases",
) -> dict:
    """Mesafe matrisi (Premium Program: Analyze → Correlate → Distances).

    Ölçütler: euclidean, manhattan, cosine, correlation.
    between='cases' vakalar arası, 'variables' değişkenler arası mesafe.
    """
    from scipy.spatial.distance import pdist, squareform

    variables = list(variables)
    _check_columns(data, variables)
    if len(variables) < 2:
        raise ValueError("En az 2 değişken gerekli")

    valid = data[variables].dropna()
    X = valid.to_numpy(dtype=float)

    if measure == "correlation":
        if between == "variables":
            mat = 1.0 - np.corrcoef(X.T)
        else:
            mat = 1.0 - np.corrcoef(X)
        labels = list(variables) if between == "variables" \
            else [str(i) for i in range(len(valid))]
    else:
        metric_map = {"euclidean": "euclidean", "manhattan": "cityblock",
                      "cosine": "cosine"}
        if measure not in metric_map:
            raise ValueError(
                f"Desteklenmeyen ölçüt: {measure} (euclidean/manhattan/cosine/correlation)")
        mat_in = X.T if between == "variables" else X
        mat = squareform(pdist(mat_in, metric=metric_map[measure]))
        labels = list(variables) if between == "variables" \
            else [str(i) for i in range(len(valid))]

    if not np.isfinite(mat).all():
        raise ValueError(
            f"'{measure}' ölçütü sonlu olmayan mesafeler üretti "
            "(sabit/boş vektör içeren verilerle tanımsızdır)")

    return {
        "measure": measure,
        "between": between,
        "distances": pd.DataFrame(mat, index=labels, columns=labels),
        "n": int(mat.shape[0]),
    }


def nearest_neighbor_analysis(
    data: pd.DataFrame,
    group_col: str,
    predictors: list[str],
    k: int = 3,
) -> dict:
    """En yakın komşu sınıflandırması (Premium Program: Analyze → Classify →
    Nearest Neighbor).

    Standartlaştırılmış değişkenlerle k-NN; vakalar kendisi hariç
    tutularak (leave-one-out) sınıflandırılır.
    """
    predictors = list(predictors)
    _check_columns(data, [group_col] + predictors)
    if k < 1:
        raise ValueError("k en az 1 olmalı")
    _validate_numeric_predictors(data[[group_col] + predictors].dropna(), predictors)

    valid = data[[group_col] + predictors].dropna()
    groups = sorted(valid[group_col].unique(), key=str)
    if len(groups) < 2:
        raise ValueError("En az 2 grup gerekli")
    X = valid[predictors].to_numpy(dtype=float)
    sd = X.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    X_std = (X - X.mean(axis=0)) / sd
    labels = valid[group_col].to_numpy()
    n = len(valid)
    if k >= n:
        raise ValueError("k, vaka sayısından küçük olmalı")

    # İkili mesafe matrisi üzerinden LOO sınıflandırma
    diffs = X_std[:, None, :] - X_std[None, :, :]
    dist = np.sqrt((diffs ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)

    predicted = []
    for i in range(n):
        neighbor_idx = np.argsort(dist[i])[:k]
        neighbor_labels = labels[neighbor_idx]
        counts = {grp: int((neighbor_labels == grp).sum()) for grp in groups}
        predicted.append(max(counts, key=lambda g_: (counts[g_], -groups.index(g_))))

    correct = np.array([pr == ob for pr, ob in zip(predicted, labels)])
    classification = []
    for grp in groups:
        mask = labels == grp
        classification.append({
            "group": str(grp),
            "n": int(mask.sum()),
            "n_correct": int(correct[mask].sum()),
            "percent_correct": float(correct[mask].mean() * 100),
        })

    return {
        "model": f"k-Nearest Neighbors (k={k}, LOO)",
        "k": int(k),
        "groups": [str(grp) for grp in groups],
        "classification": classification,
        "overall_accuracy": float(correct.mean()),
        "n": int(n),
    }


def multidimensional_scaling(
    distances: pd.DataFrame | np.ndarray,
    n_dims: int = 2,
) -> dict:
    """Klasik (metrik) çok boyutlu ölçekleme (Premium Program: Analyze →
    Dimension Reduction → Multidimensional Scaling / ALSCAL başlangıcı).

    Çift merkezleme + özayrışım ile koordinatlar; uyum kalitesi
    mesafelerin yeniden üretim oranıyla (R²) raporlanır.
    """
    D = np.asarray(distances, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("Kare mesafe matrisi gerekli")
    if np.isnan(D).any():
        raise ValueError("Mesafe matrisi NaN içeremez")
    if not np.isfinite(D).all():
        raise ValueError("Mesafe matrisi sonlu değerler içermeli (inf tespit edildi)")
    if (D < 0).any():
        raise ValueError("Mesafe matrisi negatif değer içeremez")
    if n_dims < 1:
        raise ValueError("n_dims en az 1 olmalı")
    n = D.shape[0]
    if n < 3:
        raise ValueError("MDS için en az 3 nokta gerekli")

    n_dims = min(n_dims, n - 1)
    D2 = D ** 2
    # Çift merkezleme: B = -0.5 * H D² H, satır/genel ortalamalarla eşdeğer.
    # Not: matmul (@) yerine bu cebirsel form kullanılır; Accelerate BLAS
    # ile numpy matmul geçerli girişlerde bile sahte FP uyarıları üretir.
    row_means = D2.mean(axis=1)
    B = -0.5 * (D2 - row_means[:, None] - row_means[None, :] + D2.mean())

    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1][:n_dims]
    lam = np.clip(eigvals[order], 0.0, None)
    coords = eigvecs[:, order] * np.sqrt(lam)

    positive_total = float(eigvals[eigvals > 0].sum())
    r_squared = float(lam.sum() / positive_total) if positive_total > 0 else 0.0

    labels = list(distances.index.astype(str)) if isinstance(distances, pd.DataFrame) \
        else [str(i) for i in range(n)]

    return {
        "model": "Classical (metric) MDS",
        "coordinates": pd.DataFrame(
            coords, index=labels,
            columns=[f"Boyut_{i + 1}" for i in range(n_dims)]),
        "eigenvalues": [float(v) for v in lam],
        "r_squared": r_squared,
        "stress_proxy": float(np.sqrt(max(1.0 - r_squared, 0.0))),
        "n_dimensions": int(n_dims),
        "n": int(n),
    }


def multiple_response_frequencies(
    data: pd.DataFrame,
    dichotomy_columns: list[str],
    counted_value,
    labels: Optional[dict] = None,
) -> dict:
    """Çoklu yanıt frekansları (Premium Program: Analyze → Multiple Response →
    Frequencies). Dikotomi grubundaki her kategori için vaka sayısı,
    vakaların yüzdesi ve yanıtların yüzdesi raporlanır.
    """
    dichotomy_columns = list(dichotomy_columns)
    _check_columns(data, dichotomy_columns)
    if len(dichotomy_columns) < 2:
        raise ValueError("Çoklu yanıt için en az 2 dikotomi sütunu gerekli")

    valid = data[dichotomy_columns].dropna(how="all")
    n_cases = len(valid)
    counts = {}
    for col in dichotomy_columns:
        counts[col] = int((valid[col] == counted_value).sum())
    total_responses = sum(counts.values())

    table = []
    for col in dichotomy_columns:
        label = str(labels.get(col, col)) if labels else str(col)
        table.append({
            "category": label,
            "count": counts[col],
            "percent_of_responses": float(counts[col] / total_responses * 100)
            if total_responses else 0.0,
            "percent_of_cases": float(counts[col] / n_cases * 100) if n_cases else 0.0,
        })

    return {
        "table": table,
        "counted_value": counted_value,
        "total_responses": int(total_responses),
        "n_cases": int(n_cases),
        "mean_responses_per_case": float(total_responses / n_cases) if n_cases else 0.0,
    }


def weight_estimation(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    powers: Optional[list[float]] = None,
) -> dict:
    """Ağırlık kestirimi (Premium Program: Analyze → Regression → Weight Estimation).

    Varyans sabitsizliğinde w = 1/x^p ağırlık adaylarıyla WLS uydurulur;
    en yüksek R² veren kuvvet seçilir.
    """
    x_arr = np.asarray(x, dtype=float).flatten()
    y_arr = np.asarray(y, dtype=float).flatten()
    mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
    x_arr, y_arr = x_arr[mask], y_arr[mask]
    if len(x_arr) < 5:
        raise ValueError("Ağırlık kestirimi için en az 5 geçerli gözlem gerekli")
    if (x_arr <= 0).any():
        raise ValueError("Ağırlık kestiriminde x değerleri pozitif olmalı")
    if powers is None:
        powers = [1.0, 2.0]

    X = sm.add_constant(x_arr)
    candidates = {}
    for p in powers:
        w = 1.0 / np.power(x_arr, p)
        model = sm.WLS(y_arr, X, weights=w).fit()
        candidates[float(p)] = {
            "r_squared": float(model.rsquared),
            "intercept": float(model.params[0]),
            "slope": float(model.params[1]),
        }
    best_power = max(candidates, key=lambda p_: candidates[p_]["r_squared"])

    return {
        "model": "Weight Estimation (WLS, w = 1/x^p)",
        "candidates": candidates,
        "best_power": float(best_power),
        "best_r_squared": float(candidates[best_power]["r_squared"]),
        "best_coefficients": {
            "intercept": candidates[best_power]["intercept"],
            "slope": candidates[best_power]["slope"],
        },
        "n": int(len(x_arr)),
    }


def two_stage_least_squares(
    data: pd.DataFrame,
    dependent_col: str,
    endogenous_cols: list[str],
    instruments: list[str],
    exogenous_cols: Optional[list[str]] = None,
) -> dict:
    """İki aşamalı en küçük kareler — 2SLS (Premium Program: Analyze → Regression →
    2SLS). İçsel değişkenler araç değişkenlerle tahmin edilir."""
    from statsmodels.sandbox.regression.gmm import IV2SLS

    endogenous_cols = list(endogenous_cols)
    instruments = list(instruments)
    exogenous_cols = list(exogenous_cols) if exogenous_cols else []
    all_cols = [dependent_col] + endogenous_cols + instruments + exogenous_cols
    _check_columns(data, all_cols)
    if not instruments:
        raise ValueError("En az bir araç değişken gerekli")

    valid = data[all_cols].dropna()
    if len(valid) < len(instruments) + len(endogenous_cols) + 3:
        raise ValueError("2SLS için yeterli veri yok")

    y = valid[dependent_col].to_numpy(dtype=float)
    # İkinci aşama tasarım matrisi: sabit + içsel + dışsal
    X = sm.add_constant(valid[endogenous_cols + exogenous_cols].to_numpy(dtype=float))
    # Araç matrisi: sabit + araçlar + dışsal
    Z = sm.add_constant(valid[instruments + exogenous_cols].to_numpy(dtype=float))

    model = IV2SLS(y, X, instrument=Z).fit()
    names = ["const"] + endogenous_cols + exogenous_cols
    params = np.asarray(model.params)
    bse = np.asarray(model.bse)
    tvalues = np.asarray(model.tvalues)
    pvalues = np.asarray(model.pvalues)

    return {
        "model": "2SLS",
        "coefficients": {names[i]: float(params[i]) for i in range(len(names))},
        "std_errors": {names[i]: float(bse[i]) for i in range(len(names))},
        "t_values": {names[i]: float(tvalues[i]) for i in range(len(names))},
        "p_values": {names[i]: float(pvalues[i]) for i in range(len(names))},
        "r_squared": float(model.rsquared),
        "endogenous": endogenous_cols,
        "instruments": instruments,
        "n": int(len(valid)),
    }


def twostep_cluster(
    data: pd.DataFrame,
    columns: Optional[list[str]] = None,
    max_clusters: int = 8,
    seed: Optional[int] = None,
) -> dict:
    """İki aşamalı kümeleme (Premium Program: Analyze → Classify → TwoStep Cluster).

    Standartlaştırılmış değişkenler üzerinde k=1..max_clusters için
    k-ortalamalar çalıştırılır; BIC ölçütünü küçülten küme sayısı
    otomatik seçilir (Premium Program'ın otomatik seçim davranışının karşılığı).
    """
    from scipy.cluster.vq import kmeans2

    if max_clusters < 2:
        raise ValueError("max_clusters en az 2 olmalı")
    if columns is None:
        columns = list(data.select_dtypes(include=[np.number]).columns)
    else:
        columns = list(columns)
        _check_columns(data, columns)
    if len(columns) < 1:
        raise ValueError("En az bir sayısal değişken gerekli")

    X = data[columns].dropna().to_numpy(dtype=float)
    n = len(X)
    if n < max_clusters + 2:
        raise ValueError("İki aşamalı kümeleme için yeterli veri yok")
    sd = X.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    X_std = (X - X.mean(axis=0)) / sd

    p = len(columns)
    # kmeans2'nin '++' başlatması numpy küresel rastgele durumunu kullanır;
    # tohumlanabilirlik için durum geçici olarak sabitlenir
    old_state = np.random.get_state()
    if seed is not None:
        np.random.seed(seed)
    bic_values = {}
    fitted = {}
    n_restarts = 5
    try:
        for k in range(1, max_clusters + 1):
            best_sse, best_labels, best_centroids = np.inf, None, None
            if k == 1:
                best_labels = np.zeros(n, dtype=int)
                best_centroids = X_std.mean(axis=0, keepdims=True)
                best_sse = float(((X_std - best_centroids[best_labels]) ** 2).sum())
            else:
                for _ in range(n_restarts):
                    centroids, labels = kmeans2(X_std, k, minit="++")
                    sse = float(((X_std - centroids[labels]) ** 2).sum())
                    if sse < best_sse:
                        best_sse, best_labels, best_centroids = sse, labels, centroids
            variance = max(best_sse / (max(n - k, 1) * p), 1e-12)
            # K-ortalamalar BIC'si: Gauss karışımı log-olabilirliği
            # (karışım oranları dahil) + parametre cezası
            log_likelihood = 0.0
            for k_ in range(k):
                n_j = float((best_labels == k_).sum())
                if n_j == 0:
                    continue
                log_likelihood += (n_j * np.log(n_j / n)
                                   - n_j * p / 2 * np.log(2 * np.pi * variance)
                                   - (n_j - 1) * p / 2)
            n_params = k * (p + 1) - 1
            bic = -2 * log_likelihood + n_params * np.log(n)
            bic_values[int(k)] = float(bic)
            fitted[int(k)] = (best_labels, best_centroids)
    finally:
        np.random.set_state(old_state)

    best_k = min(bic_values, key=lambda k_: bic_values[k_])
    labels, centroids = fitted[best_k]
    sizes = {int(k_): int((labels == k_).sum()) for k_ in range(best_k)}
    centroids_original = centroids * sd + X.mean(axis=0)

    return {
        "model": "TwoStep Cluster (BIC seçimi)",
        "n_clusters": int(best_k),
        "labels": labels.tolist(),
        "cluster_sizes": sizes,
        "cluster_centroids": pd.DataFrame(
            centroids_original,
            index=[f"Küme {k_}" for k_ in range(best_k)],
            columns=columns),
        "bic": bic_values,
        "n": int(n),
    }


def glm_univariate(data: pd.DataFrame, response: str,
                   between_factors: list, covariates: Optional[list] = None,
                   ss_type: int = 3, posthoc: Optional[str] = "tukey",
                   alpha: float = 0.05) -> dict:
    """Genel Doğrusal Model — tek değişkenli (faktöriyel + kovaryeteli).

    Premium Program GLM Univariate denkliği: Tip I/II/III kareler toplamı,
    kısmi eta-kare efekt büyüklükleri ve tek faktörde post-hoc.
    """
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    cols = [response] + list(between_factors) + list(covariates or [])
    _check_columns(data, cols)
    work = data[cols].dropna()
    if work.empty:
        raise ValueError("GLM için geçerli veri yok")
    for f in between_factors:
        if work[f].nunique() < 2:
            raise ValueError(f"'{f}' faktörü en az 2 düzey içermeli")

    terms = [f"C({f}, Sum)" for f in between_factors]
    terms += list(covariates or [])
    if not terms:
        raise ValueError("En az bir faktör veya kovaryet gerekli")
    model = smf.ols(f"{response} ~ {' + '.join(terms)}", data=work).fit()
    anova = anova_lm(model, typ=ss_type)
    ss_resid = float(anova["sum_sq"].iloc[-1])
    if ss_resid <= 0:
        raise ValueError("Artık kareler toplamı hesaplanamadı (yetersiz df)")

    table = []
    for name, row in anova.iterrows():
        if name in ("Intercept", "Residual"):
            continue
        table.append({
            "source": str(name),
            "ss": float(row["sum_sq"]),
            "df": int(row["df"]),
            "ms": float(row["sum_sq"] / row["df"]),
            "f_value": float(row["F"]) if not np.isnan(row["F"]) else None,
            "p_value": float(row["PR(>F)"]) if not np.isnan(row["PR(>F)"]) else None,
        })
    effect_sizes = {t["source"]: float(t["ss"] / (t["ss"] + ss_resid))
                    for t in table if t["f_value"] is not None}

    posthoc_result = None
    if posthoc in ("tukey", "duncan") and len(between_factors) == 1:
        fn = posthoc_tukey if posthoc == "tukey" else posthoc_duncan
        posthoc_result = fn(work, response, between_factors[0], alpha)

    return {
        "model": "GLM Univariate",
        "ss_type": int(ss_type),
        "anova_table": table,
        "effect_sizes": effect_sizes,
        "r_squared": float(model.rsquared),
        "posthoc": posthoc_result,
        "n_obs": int(len(work)),
    }


def _helmert_orthogonal(k: int) -> np.ndarray:
    """k koşul için (k-1)×k ortonormal kontrast matrisi (Mauchly için)."""
    C = np.zeros((k - 1, k))
    for i in range(1, k):
        C[i - 1, :i] = 1.0 / i
        C[i - 1, i] = -1.0
    Q, _ = np.linalg.qr(C.T)
    return Q.T


def glm_repeated_measures(data: pd.DataFrame, response_cols: list,
                          subject_col: str,
                          between_factor: Optional[str] = None,
                          alpha: float = 0.05) -> dict:
    """Genel Doğrusal Model — tekrarlı ölçümler (univariate yaklaşım).

    Wide formatta her sütun bir ölçüm zamanıdır. Mauchly küresellik testi,
    Greenhouse-Geisser ve Huynh-Feldt düzeltmeleri raporlanır.
    """
    from statsmodels.stats.anova import AnovaRM

    k = len(response_cols)
    if k < 3:
        raise ValueError("Tekrarlı ölçüm için en az 3 koşul gerekli")
    ekstra = [between_factor] if between_factor else []
    _check_columns(data, list(response_cols) + [subject_col] + ekstra)
    work = data[list(response_cols) + [subject_col] + ekstra].dropna()
    n_subj = work[subject_col].nunique()
    if n_subj < 2:
        raise ValueError("Tekrarlı ölçüm için en az 2 denek gerekli")

    # Within-subject F testi (AnovaRM)
    long = work.melt(id_vars=[subject_col] + ekstra,
                     value_vars=list(response_cols),
                     var_name="kosul", value_name="yanit")
    aov = AnovaRM(long, depvar="yanit", subject=subject_col,
                  within=["kosul"], aggregate_func="mean").fit()
    satir = aov.anova_table.iloc[0]
    f_deger = float(satir["F Value"])
    df1 = float(satir["Num DF"])
    df2 = float(satir["Den DF"])
    p_deger = float(satir["Pr > F"])

    # Mauchly küresellik testi (ortonormal kontrast özdeğerleri)
    S = work[list(response_cols)].cov(ddof=1).to_numpy()
    M = _helmert_orthogonal(k) @ S @ _helmert_orthogonal(k).T
    eig = np.maximum(np.linalg.eigvalsh(M), 1e-12)
    p = k - 1
    W = float(np.prod(eig) / (np.sum(eig) / p) ** p)
    df_mauchly = p * (p + 1) / 2 - 1
    chi2 = -((n_subj - 1) - (2 * p * p + p + 2) / (6 * p)) * np.log(W)
    p_mauchly = float(1 - stats.chi2.cdf(chi2, df_mauchly))

    # Epsilon düzeltmeleri
    eps_gg = float((np.sum(eig) ** 2) / (p * np.sum(eig ** 2)))
    eps_hf = float(min(1.0, (n_subj * p * eps_gg - 2)
                       / (p * (n_subj - 1 - p * eps_gg))))
    corrected = {}
    for ad, eps in (("greenhouse_geisser", eps_gg), ("huynh_feldt", eps_hf)):
        corrected[ad] = {
            "df1": float(df1 * eps), "df2": float(df2 * eps),
            "p_value": float(1 - stats.f.cdf(f_deger, df1 * eps, df2 * eps)),
        }

    between_effect = None
    if between_factor:
        denek_ortalama = work.groupby([subject_col, between_factor])[
            list(response_cols)].mean().mean(axis=1).reset_index()
        denek_ortalama.columns = [subject_col, between_factor, "ortalama"]
        gruplar = [denek_ortalama.loc[denek_ortalama[between_factor] == g,
                                      "ortalama"]
                   for g in sorted(denek_ortalama[between_factor].unique())]
        between_effect = anova_one_way(*gruplar)

    return {
        "model": "GLM Repeated Measures",
        "within_effect": {"f_value": f_deger, "df1": df1, "df2": df2,
                          "p_value": p_deger},
        "mauchly": {"w": W, "chi2": float(chi2), "p_value": p_mauchly},
        "epsilon": {"greenhouse_geisser": eps_gg, "huynh_feldt": eps_hf},
        "corrected": corrected,
        "between_effect": between_effect,
        "n_subjects": int(n_subj),
    }


def gee_model(data: pd.DataFrame, response: str, covariates: list,
              group_col: str, family: str = "gaussian",
              cov_struct: str = "independent",
              time_col: Optional[str] = None) -> dict:
    """Genelleştirilmiş Tahmin Denklemleri (GEE) — marjinal model.

    Premium Program GEE denkliği: kümelenmiş/korelasyonlu veri için
    population-averaged katsayılar, robust (sandwich) standart hatalar,
    çalışma korelasyon yapısı ve QIC.
    """
    import statsmodels.formula.api as smf

    cols = [response, group_col] + list(covariates)
    _check_columns(data, cols)
    work = data[cols + ([time_col] if time_col else [])].dropna()
    if work[group_col].nunique() < 2:
        raise ValueError("GEE için en az 2 grup gerekli")

    families = {"gaussian": sm.families.Gaussian,
                "binomial": sm.families.Binomial,
                "poisson": sm.families.Poisson,
                "gamma": sm.families.Gamma}
    if family not in families:
        raise ValueError(f"Bilinmeyen aile: {family}")
    if family == "binomial" and not work[response].isin([0, 1]).all():
        raise ValueError("Binomial aile için yanıt 0/1 olmalı")

    # Not: kurulu statsmodels sürümü kovaryans yapısını örnek (instance)
    # olarak bekler; dize takma adları desteklenmez. Autoregressive için
    # grid=True sayısal kök bulma kararlılığı sağlar.
    structs = {"independent": sm.cov_struct.Independence,
               "exchangeable": sm.cov_struct.Exchangeable,
               "autoregressive": lambda: sm.cov_struct.Autoregressive(
                   grid=True)}
    if cov_struct not in structs:
        raise ValueError(f"Bilinmeyen korelasyon yapısı: {cov_struct}")
    if cov_struct == "autoregressive" and time_col is None:
        raise ValueError("Autoregressive yapı için time_col gerekli")

    formula = f"{response} ~ {' + '.join(covariates)}"
    model = smf.gee(formula, groups=group_col, data=work,
                    family=families[family](),
                    cov_struct=structs[cov_struct](),
                    time=time_col)
    fitted = model.fit()

    coefficients = {}
    for ad in fitted.params.index:
        coefficients[ad] = {
            "coefficient": float(fitted.params[ad]),
            "std_err": float(fitted.bse[ad]),
            "z_value": float(fitted.tvalues[ad]),
            "p_value": float(fitted.pvalues[ad]),
        }
    try:
        qic = float(fitted.qic())
    except (AttributeError, TypeError):
        qic = None

    return {
        "model": "GEE",
        "family": family,
        "cov_struct": cov_struct,
        "coefficients": coefficients,
        "qic": qic,
        "n_groups": int(work[group_col].nunique()),
        "n_obs": int(len(work)),
        "converged": bool(fitted.converged),
    }
