"""
Agrista Data Module — Veri Yönetimi
Handles loading, validation, and management of agricultural datasets.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


class AgristaData:
    """Tarımsal veri setlerini yöneten sınıf."""

    def __init__(self, df: pd.DataFrame = None):
        self._df = df
        self.metadata: dict = {}
        if df is not None:
            self._extract_metadata()

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._df

    @dataframe.setter
    def dataframe(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("DataFrame bekleniyor")
        self._df = df
        self._extract_metadata()

    def _extract_metadata(self):
        """Veri setinden otomatik metadata çıkar."""
        if self._df is None:
            return
        self.metadata = {
            "rows": len(self._df),
            "columns": list(self._df.columns),
            "column_count": len(self._df.columns),
            "row_count": len(self._df),
            "dtypes": {col: str(dtype) for col, dtype in self._df.dtypes.items()},
            "null_counts": {col: int(count) for col, count in self._df.isnull().sum().items()},
            "numeric_columns": list(self._df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": list(self._df.select_dtypes(include=["object", "category"]).columns),
        }

    def head(self, n: int = 5) -> pd.DataFrame:
        """İlk n satırı göster."""
        if self._df is None:
            raise ValueError("Veri yüklenmemiş")
        return self._df.head(n)

    def info(self) -> dict:
        """Veri seti hakkında bilgi döndür."""
        if self._df is None:
            raise ValueError("Veri yüklenmemiş")
        info = {
            "shape": self._df.shape,
            "columns": list(self._df.columns),
            "dtypes": {str(k): str(v) for k, v in self._df.dtypes.items()},
            "null_counts": {col: int(count) for col, count in self._df.isnull().sum().items()},
            "memory_usage_mb": round(self._df.memory_usage(deep=True).sum() / 1024**2, 3),
        }
        return info

    def describe_numeric(self) -> pd.DataFrame:
        """Sayısal sütunlar için betimsel istatistik."""
        if self._df is None:
            raise ValueError("Veri yüklenmemiş")
        numeric_cols = self._df.select_dtypes(include=[np.number])
        if numeric_cols.empty:
            raise ValueError("Sayısal sütun bulunamadı")
        return numeric_cols.describe()

    def get_column(self, name: str) -> pd.Series:
        """Belirli bir sütunu döndür."""
        if self._df is None:
            raise ValueError("Veri yüklenmemiş")
        if name not in self._df.columns:
            raise KeyError(f"'{name}' sütunu bulunamadı. Mevcut sütunlar: {list(self._df.columns)}")
        return self._df[name]

    def filter(self, **kwargs) -> "AgristaData":
        """Koşula göre veri filtrele."""
        if self._df is None:
            raise ValueError("Veri yüklenmemiş")
        mask = pd.Series(True, index=self._df.index)
        for col, value in kwargs.items():
            if col not in self._df.columns:
                raise KeyError(f"'{col}' sütunu bulunamadı")
            mask &= (self._df[col] == value)
        return AgristaData(self._df[mask].copy())

    def select_columns(self, columns: list[str]) -> "AgristaData":
        """Belirli sütunları seç."""
        if self._df is None:
            raise ValueError("Veri yüklenmemiş")
        missing = set(columns) - set(self._df.columns)
        if missing:
            raise KeyError(f"Şu sütunlar bulunamadı: {missing}")
        return AgristaData(self._df[columns].copy())

    def drop_nulls(self, columns: Optional[list[str]] = None) -> "AgristaData":
        """Null değerleri içeren satırları sil."""
        if self._df is None:
            raise ValueError("Veri yüklenmemiş")
        df = self._df.dropna(subset=columns) if columns else self._df.dropna()
        return AgristaData(df.copy())

    def rename_columns(self, mapping: dict[str, str]) -> "AgristaData":
        """Sütun isimlerini yeniden adlandır."""
        if self._df is None:
            raise ValueError("Veri yüklenmemiş")
        return AgristaData(self._df.rename(columns=mapping).copy())

    def export_csv(self, path: str):
        """Veriyi CSV olarak dışa aktar."""
        if self._df is None:
            raise ValueError("Veri yüklenmemiş")
        self._df.to_csv(path, index=False)

    def __repr__(self):
        if self._df is None:
            return "AgristaData(veri yok)"
        return f"AgristaData(rows={len(self._df)}, columns={list(self._df.columns)})"


def load_csv(filepath: str | Path, **kwargs) -> AgristaData:
    """CSV dosyasından veri yükle."""
    df = pd.read_csv(filepath, **kwargs)
    return AgristaData(df)


def load_excel(filepath: str | Path, sheet_name: int | str = 0, **kwargs) -> AgristaData:
    """Excel dosyasından veri yükle."""
    df = pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)
    return AgristaData(df)


def load_json(filepath: str | Path, **kwargs) -> AgristaData:
    """JSON dosyasından veri yükle (tablo formatı beklenir)."""
    df = pd.read_json(filepath, **kwargs)
    return AgristaData(df)


# ---------------------------------------------------------------------------
# Premium Program Data menüsü denkliği: Sort Cases, Aggregate, Weight Cases,
# Merge Files, Split File, Select Cases
# ---------------------------------------------------------------------------


def sort_cases(data: pd.DataFrame, by: list[str],
               ascending: bool | list[bool] = True) -> pd.DataFrame:
    """Vakaları sırala (Premium Program: Data → Sort Cases)."""
    by = list(by)
    missing = [c for c in by if c not in data.columns]
    if missing:
        raise ValueError(f"Sütunlar bulunamadı: {missing}")
    return data.sort_values(by=by, ascending=ascending).reset_index(drop=True)


def aggregate_data(
    data: pd.DataFrame,
    break_cols: list[str],
    agg_map: dict[str, str | list[str]],
) -> pd.DataFrame:
    """Veriyi grup bazında özetle (Premium Program: Data → Aggregate).

    agg_map: {sütun: 'mean'|'median'|'sum'|'count'|'min'|'max'} veya
    birden çok istatistik listesi.
    """
    break_cols = list(break_cols)
    missing = [c for c in break_cols + list(agg_map) if c not in data.columns]
    if missing:
        raise ValueError(f"Sütunlar bulunamadı: {missing}")
    allowed = {"mean", "median", "sum", "count", "min", "max"}
    normalized = {}
    for col, funcs in agg_map.items():
        func_list = [funcs] if isinstance(funcs, str) else list(funcs)
        unknown = [f for f in func_list if f not in allowed]
        if unknown:
            raise ValueError(f"Desteklenmeyen toplama işlevleri: {unknown}")
        normalized[col] = func_list
    result = data.groupby(break_cols, observed=True).agg(normalized)
    result.columns = ["_".join(c) for c in result.columns]
    return result.reset_index()


def weight_cases(data: pd.DataFrame, weight_col: str) -> dict:
    """Vaka ağırlıklandırma raporu (Premium Program: Data → Weight Cases).

    Ağırlıklar doğrulanır; toplam ağırlık, eşdeğer vaka sayısı ve
    sayısal sütunların ağırlıklı ortalamaları döndürülür.
    """
    if weight_col not in data.columns:
        raise ValueError(f"Sütun bulunamadı: {weight_col}")
    valid = data.dropna(subset=[weight_col])
    w = valid[weight_col].to_numpy(dtype=float)
    if (w < 0).any():
        raise ValueError("Ağırlıklar negatif olamaz")
    if w.sum() == 0:
        raise ValueError("Ağırlık toplamı sıfır olamaz")

    weighted_means = {}
    for col in valid.select_dtypes(include=[np.number]).columns:
        if col == weight_col:
            continue
        x = valid[col].to_numpy(dtype=float)
        mask = ~np.isnan(x)
        if mask.sum() == 0 or w[mask].sum() == 0:
            continue
        weighted_means[col] = float(np.average(x[mask], weights=w[mask]))

    return {
        "weight_col": weight_col,
        "n_cases": int(len(valid)),
        "sum_of_weights": float(w.sum()),
        "equivalent_n": float(w.sum() ** 2 / (w ** 2).sum()),
        "weighted_means": weighted_means,
    }


def merge_files(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: Optional[str | list[str]] = None,
    how: str = "inner",
    add_cases: bool = False,
) -> pd.DataFrame:
    """Dosya birleştirme (Premium Program: Data → Merge Files).

    add_cases=False: değişken ekleme (join); add_cases=True: vaka ekleme
    (concat). how: inner/left/right/outer.
    """
    if add_cases:
        return pd.concat([left, right], ignore_index=True, sort=False)
    if on is None:
        raise ValueError("Değişken birleştirmede 'on' anahtar sütunu gerekli")
    if how not in ("inner", "left", "right", "outer"):
        raise ValueError(f"Desteklenmeyen birleştirme türü: {how}")
    return left.merge(right, on=on, how=how)


def split_file(data: pd.DataFrame, group_col: str) -> dict:
    """Dosyayı böl (Premium Program: Data → Split File) — grup → alt veri sözlüğü."""
    if group_col not in data.columns:
        raise ValueError(f"Sütun bulunamadı: {group_col}")
    return {
        str(name): grp.reset_index(drop=True)
        for name, grp in data.groupby(group_col, observed=True)
    }


def identify_duplicates(data: pd.DataFrame,
                        columns: Optional[list[str]] = None) -> dict:
    """Yinelenen vakaları tespit et (Premium Program: Data → Identify Duplicate Cases)."""
    subset = list(columns) if columns else None
    if subset:
        missing = [c for c in subset if c not in data.columns]
        if missing:
            raise ValueError(f"Sütunlar bulunamadı: {missing}")
    mask = data.duplicated(subset=subset, keep=False)
    return {
        "duplicate_mask": mask,
        "n_duplicate_cases": int(mask.sum()),
        "n_unique_cases": int(len(data) - mask.sum() + data[mask].drop_duplicates(subset=subset).shape[0]),
        "duplicates": data[mask],
    }


def transpose_data(data: pd.DataFrame,
                   index_col: Optional[str] = None) -> pd.DataFrame:
    """Veriyi devir (Premium Program: Data → Transpose)."""
    if index_col is not None:
        if index_col not in data.columns:
            raise ValueError(f"Sütun bulunamadı: {index_col}")
        data = data.set_index(index_col)
    numeric = data.select_dtypes(include=[np.number])
    if numeric.shape != data.shape:
        raise ValueError("Transpose yalnızca sayısal sütunlarla yapılabilir")
    return numeric.T.reset_index().rename(columns={"index": "case"})


def restructure_data(
    data: pd.DataFrame,
    direction: str = "long",
    id_cols: Optional[list[str]] = None,
    value_cols: Optional[list[str]] = None,
    var_name: str = "degisken",
    value_name: str = "deger",
    index_col: Optional[str] = None,
    columns_col: Optional[str] = None,
) -> pd.DataFrame:
    """Veriyi yeniden yapılandır (Premium Program: Data → Restructure).

    direction='long': geniş → uzun (melt); direction='wide': uzun → geniş
    (pivot).
    """
    if direction == "long":
        return data.melt(id_vars=id_cols, value_vars=value_cols,
                         var_name=var_name, value_name=value_name)
    if direction == "wide":
        if not (index_col and columns_col):
            raise ValueError("Wide dönüşüm için index_col ve columns_col gerekli")
        return data.pivot(index=index_col, columns=columns_col,
                          values=value_name)
    raise ValueError(f"Desteklenmeyen yön: {direction} (long/wide)")


def compare_datasets(left: pd.DataFrame, right: pd.DataFrame,
                     key_col: str) -> dict:
    """Veri setlerini karşılaştır (Premium Program: Data → Compare Datasets)."""
    if key_col not in left.columns or key_col not in right.columns:
        raise ValueError(f"Anahtar sütun bulunamadı: {key_col}")
    left_keys = set(left[key_col])
    right_keys = set(right[key_col])
    common_cols = [c for c in left.columns if c in right.columns and c != key_col]
    merged = left.merge(right, on=key_col, suffixes=("_sol", "_sag"))
    mismatches = {}
    for col in common_cols:
        l, r = merged[f"{col}_sol"], merged[f"{col}_sag"]
        diff = (l != r) & ~(l.isna() & r.isna())
        mismatches[col] = int(diff.sum())
    return {
        "only_in_left": int(len(left_keys - right_keys)),
        "only_in_right": int(len(right_keys - left_keys)),
        "matched": int(len(left_keys & right_keys)),
        "value_mismatches": mismatches,
        "total_mismatched_cells": int(sum(mismatches.values())),
    }


def define_measurement_level(data: pd.DataFrame, column: str,
                             level: str) -> pd.DataFrame:
    """Ölçüm düzeyi tanımla (Premium Program: Data → Define Measurement Level).

    level: 'nominal', 'ordinal', 'scale'. Bilgi sütun attrs'ında saklanır.
    """
    if column not in data.columns:
        raise ValueError(f"Sütun bulunamadı: {column}")
    if level not in ("nominal", "ordinal", "scale"):
        raise ValueError(f"Geçersiz düzey: {level} (nominal/ordinal/scale)")
    result = data.copy()
    levels = dict(result.attrs.get("measurement_levels", {}))
    levels[column] = level
    result.attrs["measurement_levels"] = levels
    return result
