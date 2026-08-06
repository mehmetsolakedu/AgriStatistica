"""
Agrista Transform Module — Veri Dönüşümü (Premium Program: Transform menüsü)

Compute Variable, Recode, Binning, Rank Cases, Count Values ve
Replace Missing Values işlemlerinin tamamı yeni DataFrame döndürür;
girdi verisi hiçbir zaman yerinde değiştirilmez.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def compute(data: pd.DataFrame, new_column: str, expression: str) -> pd.DataFrame:
    """Yeni değişken hesapla (Premium Program: Transform → Compute Variable).

    expression: sütun adları ve aritmetik işleçlerle pandas.eval ifadesi.
    Örnek: "sulama * 0.001 + gubre * 0.005"
    """
    if not new_column or not new_column.strip():
        raise ValueError("Yeni sütun adı boş olamaz")
    new_column = new_column.strip()

    try:
        values = data.eval(expression)
    except Exception as e:
        raise ValueError(f"İfade değerlendirilemedi: {e}")

    result = data.copy()
    result[new_column] = values
    return result


def recode(
    data: pd.DataFrame,
    column: str,
    mapping: dict,
    new_column: Optional[str] = None,
    default=np.nan,
) -> pd.DataFrame:
    """Değişkeni yeniden kodla (Premium Program: Transform → Recode into Different Variables).

    mapping: {eski_değer: yeni_değer} sözlüğü.
    default: eşleşmeyen değerlere atanır (Premium Program 'ELSE' karşılığı);
    varsayılan NaN'dır.
    """
    if column not in data.columns:
        raise ValueError(f"'{column}' sütunu bulunamadı")

    target = new_column or f"{column}_recoded"
    result = data.copy()
    mapped = result[column].map(mapping)
    if default is not None and not (isinstance(default, float) and np.isnan(default)):
        mapped = mapped.fillna(default)
    result[target] = mapped
    return result


def bin_variable(
    data: pd.DataFrame,
    column: str,
    bins: int | list = 4,
    method: str = "equal_width",
    new_column: Optional[str] = None,
    labels: Optional[list] = None,
) -> pd.DataFrame:
    """Değişkeni kategorilere böl (Premium Program: Visual/Optimal Binning).

    method: 'equal_width' (eşit aralık, pd.cut) veya
            'equal_freq' (eşit frekans, pd.qcut)
    bins: tam sayı (kategori sayısı) veya sınır listesi.
    """
    if column not in data.columns:
        raise ValueError(f"'{column}' sütunu bulunamadı")
    if method not in ("equal_width", "equal_freq"):
        raise ValueError(f"Desteklenmeyen yöntem: {method}")

    target = new_column or f"{column}_binned"
    result = data.copy()
    series = result[column].astype(float)

    try:
        if method == "equal_width":
            result[target] = pd.cut(series, bins=bins, labels=labels)
        else:
            result[target] = pd.qcut(series, q=bins, labels=labels, duplicates="drop")
    except ValueError as e:
        raise ValueError(f"Kategorileme başarısız: {e}")

    return result


def rank_cases(
    data: pd.DataFrame,
    column: str,
    new_column: Optional[str] = None,
    ascending: bool = True,
    method: str = "average",
) -> pd.DataFrame:
    """Vakaları sırala (Premium Program: Transform → Rank Cases).

    method: 'average' (Premium Program varsayılanı), 'min', 'max', 'first', 'dense'.
    """
    if column not in data.columns:
        raise ValueError(f"'{column}' sütunu bulunamadı")

    target = new_column or f"{column}_rank"
    result = data.copy()
    result[target] = result[column].rank(ascending=ascending, method=method)
    return result


def count_values(data: pd.DataFrame, columns: list, value, new_column: str) -> pd.DataFrame:
    """Belirli bir değerin kaç değişkende görüldüğünü say
    (Premium Program: Transform → Count Values)."""
    missing = [c for c in columns if c not in data.columns]
    if missing:
        raise ValueError(f"Sütunlar bulunamadı: {missing}")
    if not new_column or not new_column.strip():
        raise ValueError("Yeni sütun adı boş olamaz")

    result = data.copy()
    result[new_column] = (result[columns] == value).sum(axis=1)
    return result


def replace_missing(
    data: pd.DataFrame,
    columns: Optional[list] = None,
    method: str = "mean",
) -> tuple:
    """Eksik değerleri tamamla (Premium Program: Transform → Replace Missing Values).

    method: 'mean', 'median', 'ffill' (bir önceki değer),
            'interpolate' (doğrusal enterpolasyon)
    Döndürür: (yeni DataFrame, sütun başına tamamlanan eksik sayısı)
    """
    methods = ("mean", "median", "ffill", "interpolate")
    if method not in methods:
        raise ValueError(f"Desteklenmeyen yöntem: {method}. Seçenekler: {methods}")

    if columns is None:
        columns = list(data.select_dtypes(include=[np.number]).columns)
    missing_cols = [c for c in columns if c not in data.columns]
    if missing_cols:
        raise ValueError(f"Sütunlar bulunamadı: {missing_cols}")
    if not columns:
        raise ValueError("Eksik tamamlama için sayısal sütun bulunamadı")

    result = data.copy()
    report = {}
    for col in columns:
        n_missing = int(result[col].isna().sum())
        if n_missing == 0:
            report[col] = 0
            continue
        if method == "mean":
            if not np.issubdtype(result[col].dtype, np.number):
                raise ValueError(f"'mean' yöntemi '{col}' için yalnızca sayısal sütunlarda çalışır")
            result[col] = result[col].fillna(result[col].mean())
        elif method == "median":
            if not np.issubdtype(result[col].dtype, np.number):
                raise ValueError(f"'median' yöntemi '{col}' için yalnızca sayısal sütunlarda çalışır")
            result[col] = result[col].fillna(result[col].median())
        elif method == "ffill":
            result[col] = result[col].ffill()
        elif method == "interpolate":
            result[col] = result[col].interpolate(method="linear").bfill()
        report[col] = n_missing

    return result, report


def create_time_series(
    data: pd.DataFrame,
    column: str,
    function: str = "lag",
    periods: int = 1,
    window: int = 3,
    new_column: Optional[str] = None,
) -> pd.DataFrame:
    """Zaman serisi değişkeni oluştur (Premium Program: Transform → Create Time Series).

    function: 'lag' (gecikme), 'difference' (fark), 'moving_average'
    (hareketli ortalama), 'seasonal_difference' (mevsimsel fark).
    """
    if column not in data.columns:
        raise ValueError(f"Sütun bulunamadı: {column}")
    if periods < 1:
        raise ValueError("periods en az 1 olmalı")

    series = data[column]
    if new_column is None:
        new_column = f"{column}_{function}{periods if function != 'moving_average' else f'_w{window}'}"

    result = data.copy()
    if function == "lag":
        result[new_column] = series.shift(periods)
    elif function == "difference":
        result[new_column] = series.diff(periods)
    elif function == "seasonal_difference":
        result[new_column] = series.diff(periods)
    elif function == "moving_average":
        if window < 2:
            raise ValueError("Hareketli ortalama için window en az 2 olmalı")
        result[new_column] = series.rolling(window).mean()
    else:
        raise ValueError(
            f"Desteklenmeyen işlev: {function} "
            "(lag/difference/seasonal_difference/moving_average)")

    return result


def random_numbers(
    n: int,
    distribution: str = "normal",
    new_column: str = "rasgele",
    data: Optional[pd.DataFrame] = None,
    seed: Optional[int] = None,
    **params,
) -> pd.DataFrame:
    """Rastgele sayı üret (Premium Program: Transform → Random Number Generators).

    distribution: 'normal' (mean, std), 'uniform' (low, high),
    'binomial' (n_trials, p), 'poisson' (lam), 'exponential' (scale).
    data verilirse yeni sütun o tabloya eklenir.
    """
    if n < 1:
        raise ValueError("n en az 1 olmalı")
    rng = np.random.default_rng(seed)
    if distribution == "normal":
        values = rng.normal(params.get("mean", 0.0), params.get("std", 1.0), n)
    elif distribution == "uniform":
        values = rng.uniform(params.get("low", 0.0), params.get("high", 1.0), n)
    elif distribution == "binomial":
        values = rng.binomial(params.get("n_trials", 1), params.get("p", 0.5), n)
    elif distribution == "poisson":
        values = rng.poisson(params.get("lam", 1.0), n)
    elif distribution == "exponential":
        values = rng.exponential(params.get("scale", 1.0), n)
    else:
        raise ValueError(
            f"Desteklenmeyen dağılım: {distribution} "
            "(normal/uniform/binomial/poisson/exponential)")

    result = data.copy() if data is not None else pd.DataFrame(index=range(n))
    if len(result) != n:
        raise ValueError("data satır sayısı n ile eşleşmeli")
    result[new_column] = values
    return result


def automatic_recode(
    data: pd.DataFrame,
    column: str,
    new_column: Optional[str] = None,
) -> pd.DataFrame:
    """Otomatik yeniden kodlama (Premium Program: Transform → Automatic Recode).

    Kategori değerlerini sıralı 1..k tamsayılarına kodlar; eşleme
    tablonun attrs'ında saklanır.
    """
    if column not in data.columns:
        raise ValueError(f"Sütun bulunamadı: {column}")
    if new_column is None:
        new_column = f"{column}_recoded"
    categories = sorted(data[column].dropna().unique(), key=str)
    mapping = {cat: i + 1 for i, cat in enumerate(categories)}
    result = data.copy()
    result[new_column] = result[column].map(mapping)
    labels = dict(result.attrs.get("automatic_recode_labels", {}))
    labels[new_column] = {str(k): int(v) for k, v in mapping.items()}
    result.attrs["automatic_recode_labels"] = labels
    return result
