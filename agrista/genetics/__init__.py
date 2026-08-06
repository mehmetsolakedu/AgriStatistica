"""
Agrista Genetics Module — Tarla Bitkileri Islahı ve Biyometri
Çok değişkenli ıslah analizleri ve G×E (genotip × çevre) stabilite yöntemleri.

Literatür dayanağı: docs/01_ALT_BRANS_ISTATISTIK_LITERATUR_LOG.md (Bölüm 2)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, mahalanobis
from typing import Optional


def path_analysis(data: pd.DataFrame, target: str, predictors: list) -> dict:
    """Yol (path) katsayısı analizi.

    Standartlaştırılmış doğrudan etkiler, korelasyon matrisinin tersi ile
    hedef korelasyonlarının çarpımından elde edilir; dolaylı etkiler
    fark üzerinden ayrıştırılır.
    """
    cols = list(predictors) + [target]
    valid = data[cols].dropna()
    if len(valid) < len(predictors) + 3:
        raise ValueError("Yol analizi için yeterli veri yok")

    corr = valid.corr()
    r_xx = corr.loc[predictors, predictors].to_numpy()
    r_xy = corr.loc[predictors, target].to_numpy()

    try:
        direct = np.linalg.solve(r_xx, r_xy)
    except np.linalg.LinAlgError:
        raise ValueError("Korelasyon matrisi tekil; çoklu bağlantı çok yüksek")

    total_corr = {p: float(corr.loc[p, target]) for p in predictors}
    direct_effects = {p: float(d) for p, d in zip(predictors, direct)}

    # Dolaylı etkiler: toplam korelasyon - doğrudan etki (basit ayrıştırma)
    indirect_effects = {p: total_corr[p] - direct_effects[p] for p in predictors}

    # Belirlilik: hedef varyansının açıklanan kısmı
    r_squared = float(np.dot(direct, r_xy))

    return {
        "direct_effects": direct_effects,
        "indirect_effects": indirect_effects,
        "total_correlations": total_corr,
        "r_squared": r_squared,
        "target": target,
        "n_obs": int(len(valid)),
    }


def pca_analysis(data: pd.DataFrame, columns: Optional[list] = None, n_components: Optional[int] = None) -> dict:
    """Temel Bileşenler Analizi (PCA) — standartlaştırılmış veri üzerinden SVD."""
    cols = columns or list(data.select_dtypes(include=[np.number]).columns)
    valid = data[cols].dropna()
    if len(valid) < 3 or len(cols) < 2:
        raise ValueError("PCA için en az 3 gözlem ve 2 sayısal sütun gerekli")

    X = valid.to_numpy(dtype=float)
    X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    X_std = np.nan_to_num(X_std, nan=0.0)

    U, S, Vt = np.linalg.svd(X_std, full_matrices=False)
    var_ratios = (S ** 2) / np.sum(S ** 2)

    if n_components is None:
        n_components = int(np.searchsorted(np.cumsum(var_ratios), 0.80) + 1)
        n_components = min(max(n_components, 2), len(cols))

    scores = U[:, :n_components] * S[:n_components]
    loadings = Vt[:n_components].T

    return {
        "explained_variance_ratio": {f"PC{i+1}": float(v) for i, v in enumerate(var_ratios)},
        "cumulative_variance": float(np.sum(var_ratios[:n_components])),
        "loadings": pd.DataFrame(loadings, index=cols, columns=[f"PC{i+1}" for i in range(n_components)]),
        "scores": pd.DataFrame(scores, index=valid.index, columns=[f"PC{i+1}" for i in range(n_components)]),
        "n_components": n_components,
        "columns_used": cols,
    }


def cluster_genotypes(
    data: pd.DataFrame,
    columns: list,
    n_clusters: Optional[int] = None,
    method: str = "ward",
    metric: str = "euclidean",
) -> dict:
    """Hiyerarşik kümeleme (genetik çeşitlilik gruplandırması)."""
    valid = data[columns].dropna()
    if len(valid) < 3:
        raise ValueError("Kümeleme için en az 3 gözlem gerekli")

    X = valid.to_numpy(dtype=float)
    X = (X - X.mean(axis=0)) / np.where(X.std(axis=0, ddof=1) > 0, X.std(axis=0, ddof=1), 1.0)

    distances = pdist(X, metric=metric)
    Z = linkage(distances, method=method)

    if n_clusters is None:
        # Mesafe sıçramasıyla otomatik küme sayısı (basit sezgi)
        merge_distances = Z[:, 2]
        gaps = np.diff(merge_distances)
        n_clusters = int(np.argmax(gaps) + 2) if len(gaps) > 0 else 2
        n_clusters = min(max(n_clusters, 2), len(valid))

    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    return {
        "n_clusters": int(n_clusters),
        "labels": pd.Series(labels, index=valid.index, name="kume"),
        "cluster_sizes": {int(k): int((labels == k).sum()) for k in np.unique(labels)},
        "linkage_matrix": Z.tolist(),
        "method": method,
        "metric": metric,
    }


def mahalanobis_d2(data: pd.DataFrame, columns: list, group_col: Optional[str] = None) -> dict:
    """Mahalanobis D² genetik uzaklık matrisi.

    group_col verilmezse her satır bir genotip kabul edilir.
    """
    if group_col:
        means = data.groupby(group_col)[columns].mean().dropna()
    else:
        means = data[columns].dropna()

    if len(means) < 2:
        raise ValueError("D² hesabı için en az 2 genotip/grup gerekli")

    X = data[columns].dropna().to_numpy(dtype=float)
    if X.shape[0] <= X.shape[1]:
        raise ValueError("Kovaryans tahmini için gözlem sayısı özellik sayısından fazla olmalı")

    cov = np.cov(X, rowvar=False)
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    names = [str(i) for i in means.index]
    values = means.to_numpy(dtype=float)
    k = len(values)
    d2 = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            d2[i, j] = d2[j, i] = mahalanobis(values[i], values[j], cov_inv) ** 2

    return {
        "d2_matrix": pd.DataFrame(d2, index=names, columns=names),
        "mean_d2": float(d2[np.triu_indices(k, 1)].mean()),
        "max_pair": _max_pair(d2, names),
        "n_genotypes": k,
    }


def _max_pair(d2: np.ndarray, names: list) -> dict:
    k = len(names)
    best, best_val = None, -1.0
    for i in range(k):
        for j in range(i + 1, k):
            if d2[i, j] > best_val:
                best_val, best = d2[i, j], (names[i], names[j])
    return {"genotype_1": best[0], "genotype_2": best[1], "d2": float(best_val)}


def heritability(variance_genotypic: float, variance_error: float, n_reps: int = 1) -> dict:
    """Geniş anlamda kalıtım derecesi.

    h² = Vg / (Vg + Ve/n) — giriş ortalaması temelinde.
    """
    if variance_genotypic < 0 or variance_error < 0:
        raise ValueError("Varyans bileşenleri negatif olamaz")
    if n_reps < 1:
        raise ValueError("Tekerrür sayısı en az 1 olmalı")

    denominator = variance_genotypic + variance_error / n_reps
    h2 = variance_genotypic / denominator if denominator > 0 else 0.0

    return {
        "h2_broad_sense": float(h2),
        "variance_genotypic": float(variance_genotypic),
        "variance_error": float(variance_error),
        "n_reps": int(n_reps),
        "interpretation": (
            "Yüksek kalıtım" if h2 > 0.6 else "Orta kalıtım" if h2 > 0.3 else "Düşük kalıtım"
        ),
    }


# ---------------------------------------------------------------------------
# P2: G×E etkileşimi ve stabilite analizleri
# ---------------------------------------------------------------------------


def _two_way_means(data: pd.DataFrame, genotype_col: str, environment_col: str, response_col: str) -> pd.DataFrame:
    pivot = data.pivot_table(index=genotype_col, columns=environment_col, values=response_col, aggfunc="mean")
    if pivot.isna().any().any():
        raise ValueError("AMMI için dengeli veri gerekli (eksik genotip×çevre hücreleri var)")
    return pivot


def ammi_analysis(
    data: pd.DataFrame,
    genotype_col: str,
    environment_col: str,
    response_col: str,
    n_ipca: int = 2,
) -> dict:
    """AMMI analizi — ANOVA ana etkileri + etkileşimin PCA ayrıştırması.

    IPCA skorları ve açıklanan etkileşim varyansları döndürülür;
    biplot için (IPCA1, IPCA2) koordinatları verilir.
    """
    pivot = _two_way_means(data, genotype_col, environment_col, response_col)
    Y = pivot.to_numpy(dtype=float)
    g, e = Y.shape

    grand_mean = Y.mean()
    geno_means = Y.mean(axis=1)
    env_means = Y.mean(axis=0)

    # ANOVA kareler toplamı
    ss_g = e * np.sum((geno_means - grand_mean) ** 2)
    ss_e = g * np.sum((env_means - grand_mean) ** 2)
    residuals = Y - grand_mean - (geno_means - grand_mean)[:, None] - (env_means - grand_mean)[None, :]
    ss_ge = np.sum(residuals ** 2)
    ss_total = np.sum((Y - grand_mean) ** 2)

    df_g, df_e = g - 1, e - 1
    df_ge = df_g * df_e

    # Etkileşim matrisinin SVD ayrıştırması
    U, S, Vt = np.linalg.svd(residuals, full_matrices=False)
    n_ipca = min(n_ipca, len(S))
    ipca_var = (S ** 2) / ss_ge if ss_ge > 0 else np.zeros_like(S)

    genotypes = [str(i) for i in pivot.index]
    environments = [str(c) for c in pivot.columns]

    geno_scores = {}
    for i, gname in enumerate(genotypes):
        geno_scores[gname] = {
            f"IPCA{k+1}": float(U[i, k] * S[k]) for k in range(n_ipca)
        }
        geno_scores[gname]["mean"] = float(geno_means[i])

    env_scores = {}
    for j, ename in enumerate(environments):
        env_scores[ename] = {
            f"IPCA{k+1}": float(Vt[k, j] * S[k]) for k in range(n_ipca)
        }
        env_scores[ename]["mean"] = float(env_means[j])

    return {
        "anova": {
            "ss_genotype": float(ss_g),
            "ss_environment": float(ss_e),
            "ss_interaction": float(ss_ge),
            "ss_total": float(ss_total),
            "df_genotype": df_g,
            "df_environment": df_e,
            "df_interaction": df_ge,
        },
        "ipca_explained_variance": {f"IPCA{k+1}": float(ipca_var[k]) for k in range(len(S))},
        "genotype_scores": geno_scores,
        "environment_scores": env_scores,
        "n_genotypes": g,
        "n_environments": e,
    }


def gge_biplot(
    data: pd.DataFrame,
    genotype_col: str,
    environment_col: str,
    response_col: str,
) -> dict:
    """GGE biplot — çevre-merkezli veri üzerinden SVD (Genotype + G×E)."""
    pivot = _two_way_means(data, genotype_col, environment_col, response_col)
    Y = pivot.to_numpy(dtype=float)
    env_centered = Y - Y.mean(axis=0, keepdims=True)

    U, S, Vt = np.linalg.svd(env_centered, full_matrices=False)
    total_var = np.sum(S ** 2)

    genotypes = [str(i) for i in pivot.index]
    environments = [str(c) for c in pivot.columns]

    geno_coords = {}
    for i, gname in enumerate(genotypes):
        geno_coords[gname] = {
            "PC1": float(U[i, 0] * S[0]),
            "PC2": float(U[i, 1] * S[1]) if len(S) > 1 else 0.0,
            "mean": float(Y[i].mean()),
        }

    env_coords = {}
    for j, ename in enumerate(environments):
        env_coords[ename] = {
            "PC1": float(Vt[0, j] * S[0]),
            "PC2": float(Vt[1, j] * S[1]) if len(S) > 1 else 0.0,
        }

    return {
        "pc_explained_variance": {
            "PC1": float(S[0] ** 2 / total_var),
            "PC2": float(S[1] ** 2 / total_var) if len(S) > 1 else 0.0,
        },
        "genotype_coordinates": geno_coords,
        "environment_coordinates": env_coords,
        "n_genotypes": len(genotypes),
        "n_environments": len(environments),
    }


def stability_indices(
    data: pd.DataFrame,
    genotype_col: str,
    environment_col: str,
    response_col: str,
) -> dict:
    """Finlay-Wilkinson stabilite regresyonu (bi ve S²d).

    Her genotipin ortalaması çevresel indekse karşı doğrusal regresyona
    sokulur: bi=1 ve S²d≈0 ideal stabilite ölçütüdür.
    """
    pivot = _two_way_means(data, genotype_col, environment_col, response_col)
    Y = pivot.to_numpy(dtype=float)
    env_index = Y.mean(axis=0)

    results = {}
    for i, gname in enumerate(pivot.index):
        y = Y[i]
        mask = np.ones(len(y), dtype=bool)
        slope, intercept, r_value, p_value, std_err = stats.linregress(env_index[mask], y[mask])
        fitted = intercept + slope * env_index
        deviations = y - fitted
        ss_dev = float(np.sum(deviations ** 2))
        df_dev = len(y) - 2
        s2d = ss_dev / df_dev if df_dev > 0 else 0.0

        results[str(gname)] = {
            "mean": float(np.mean(y)),
            "bi": float(slope),
            "s2d": float(s2d),
            "r_squared": float(r_value ** 2),
            "stable": bool(abs(slope - 1.0) < 0.25 and s2d < np.var(y)),
        }

    return {
        "method": "Finlay-Wilkinson",
        "environment_index_mean": float(np.mean(env_index)),
        "genotypes": results,
        "most_stable": min(results.items(), key=lambda kv: abs(kv[1]["bi"] - 1.0) + kv[1]["s2d"])[0],
    }


# ---------------------------------------------------------------------------
# Premium Program parity: Faktör analizi (Dimension Reduction → Factor)
# ---------------------------------------------------------------------------


def _varimax_rotation(loadings: np.ndarray, max_iter: int = 500,
                      tol: float = 1e-8) -> np.ndarray:
    """Kaiser-normalizasyonlu varimax döndürmesi (SVD tabanlı)."""
    p, k = loadings.shape
    if k < 2:
        return loadings.copy()
    R = np.eye(k)
    d_old = 0.0
    for _ in range(max_iter):
        Lambda = loadings @ R
        # Varimax ölçütünün gradyanı: Λ³ − Λ·(satır normları²/p)
        row_norms_sq = np.sum(Lambda ** 2, axis=1)
        M = loadings.T @ (Lambda ** 3 - Lambda * (row_norms_sq / p)[:, None])
        U, s, Vt = np.linalg.svd(M)
        R_new = U @ Vt
        d_new = float(s.sum())
        if abs(d_new - d_old) < tol:
            R = R_new
            break
        R, d_old = R_new, d_new
    return loadings @ R


def factor_analysis(
    data: pd.DataFrame,
    columns: Optional[list] = None,
    n_factors: int = 2,
    rotation: str = "varimax",
) -> dict:
    """Faktör analizi (Premium Program: Dimension Reduction → Factor).
    
    Temel bileşen çıkarması + isteğe bağlı varimax döndürmesi;
    komunallıklar ve faktör başına açıklanan varyans raporlanır.
    """
    cols = columns or list(data.select_dtypes(include=[np.number]).columns)
    valid = data[cols].dropna()
    if len(valid) < 10:
        raise ValueError("Faktör analizi için en az 10 gözlem gerekli")
    if len(cols) < 3:
        raise ValueError("En az 3 değişken gerekli")
    if n_factors < 1 or n_factors > len(cols):
        raise ValueError("n_factors 1 ile değişken sayısı arasında olmalı")
    if rotation not in ("varimax", "none"):
        raise ValueError("rotation 'varimax' veya 'none' olabilir")
    
    X = valid.to_numpy(dtype=float)
    X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    corr = np.corrcoef(X_std, rowvar=False)
    
    eigvals, eigvecs = np.linalg.eigh(corr)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    
    sqrt_vals = np.sqrt(np.maximum(eigvals[:n_factors], 0.0))
    loadings = eigvecs[:, :n_factors] * sqrt_vals
    
    if rotation == "varimax" and n_factors > 1:
        loadings = _varimax_rotation(loadings)
    
    communalities = np.sum(loadings ** 2, axis=1)
    var_per_factor = np.sum(loadings ** 2, axis=0)
    
    factor_names = [f"Faktor_{i+1}" for i in range(n_factors)]
    return {
        "loadings": pd.DataFrame(loadings, index=cols, columns=factor_names),
        "eigenvalues": eigvals.tolist(),
        "variance_per_factor": {f: float(v) for f, v in zip(factor_names, var_per_factor)},
        "explained_variance_pct": {f: float(v / len(cols) * 100) for f, v in zip(factor_names, var_per_factor)},
        "total_explained_pct": float(var_per_factor.sum() / len(cols) * 100),
        "communalities": {c: float(h) for c, h in zip(cols, communalities)},
        "n_factors": int(n_factors),
        "rotation": rotation,
        "n_obs": int(len(valid)),
    }


def kmeans_cluster(
    data: pd.DataFrame,
    columns: list,
    n_clusters: int,
    seed: Optional[int] = 42,
    max_iter: int = 100,
) -> dict:
    """K-Ortalamalar kümeleme (Premium Program: Classify → K-Means Cluster).
    
    Değişkenler standartlaştırıldıktan sonra Lloyd algoritması çalıştırılır;
    küme etiketleri, merkezler ve boyut bazında ANOVA-F benzeri ayrışma
    raporu döndürülür.
    """
    from scipy.cluster.vq import kmeans2
    
    missing = [c for c in columns if c not in data.columns]
    if missing:
        raise ValueError(f"Sütunlar bulunamadı: {missing}")
    if n_clusters < 2:
        raise ValueError("En az 2 küme gerekli")
    
    valid = data[columns].dropna()
    if len(valid) < n_clusters * 2:
        raise ValueError("K-Ortalamalar için yeterli gözlem yok")
    
    X = valid.to_numpy(dtype=float)
    std = X.std(axis=0, ddof=1)
    std = np.where(std > 0, std, 1.0)
    X_std = (X - X.mean(axis=0)) / std
    
    rng = np.random.default_rng(seed)
    init_idx = rng.choice(len(X_std), size=n_clusters, replace=False)
    centroids, labels = kmeans2(X_std, X_std[init_idx], minit="matrix",
                                iter=max_iter, seed=seed)
    
    label_series = pd.Series(labels, index=valid.index, name="kume")
    sizes = {int(k): int((labels == k).sum()) for k in np.unique(labels)}
    
    # Küme merkezleri (orijinal ölçekte)
    centers_original = {}
    for k in np.unique(labels):
        centers_original[int(k)] = {
            c: float(X[labels == k, j].mean()) for j, c in enumerate(columns)
        }
    
    # Boyut bazında ayrışma gücü: eta-kare (kümeler arası / toplam)
    dimension_eta = {}
    for j, c in enumerate(columns):
        grand = X_std[:, j].mean()
        ss_between = sum(
            (labels == k).sum() * (X_std[labels == k, j].mean() - grand) ** 2
            for k in np.unique(labels)
        )
        ss_total = float(np.sum((X_std[:, j] - grand) ** 2))
        dimension_eta[c] = float(ss_between / ss_total) if ss_total > 0 else 0.0
    
    return {
        "n_clusters": int(n_clusters),
        "labels": label_series,
        "cluster_sizes": sizes,
        "centroids_standardized": pd.DataFrame(
            centroids, columns=columns,
            index=[f"Kume_{int(k)}" for k in np.unique(labels)]),
        "centers_original": centers_original,
        "dimension_eta_squared": dimension_eta,
        "n_obs": int(len(valid)),
    }
