"""
Agrista Visualization Module — Görselleştirme
Plotting and visualization tools for agricultural data analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Optional


class AgristaPlotter:
    """Tarımsal veri görselleştirme aracı."""
    
    def __init__(self, style: str = "whitegrid", theme: str = None):
        from agrista.viz.themes import apply_theme
        self.theme = theme if theme is not None else "agrista"
        t = apply_theme(self.theme)
        self._theme = t
        self._palette = list(t["palette"])
        self.style = t["style"] if theme is not None else style
        sns.set_style(self.style)
        plt.rcParams.update(t["rc"])
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rcParams["font.size"] = 12
    
    def _ensure_fig(self):
        """Gerekirse yeni figür oluştur."""
        if not plt.get_fignums():
            plt.figure()
    
    def histogram(
        self,
        data: pd.Series | np.ndarray,
        title: str = "Dağılım Grafiği",
        bins: int = 30,
        color: str = "#2E86AB",
        show_stats: bool = True,
    ) -> plt.Figure:
        """Veri dağılımı histogramı."""
        arr = np.asarray(data).flatten()
        arr = arr[~np.isnan(arr)]
        
        self._ensure_fig()
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(arr, bins=bins, color=color, alpha=0.7, edgecolor="white")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Değer")
        ax.set_ylabel("Frekans")
        
        if show_stats:
            mean = np.mean(arr)
            std = np.std(arr, ddof=1)
            ax.axvline(mean, color="red", linestyle="--", linewidth=2, label=f"Ortalama: {mean:.2f}")
            ax.axvline(mean + std, color="orange", linestyle=":", linewidth=1.5, label=f"+1σ: {mean+std:.2f}")
            ax.axvline(mean - std, color="orange", linestyle=":", linewidth=1.5, label=f"-1σ: {mean-std:.2f}")
            ax.legend()
        
        plt.tight_layout()
        return fig
    
    def boxplot(
        self,
        data: pd.DataFrame,
        x_col: str = None,
        y_col: str = None,
        title: str = "Kutu Grafiği",
    ) -> plt.Figure:
        """Boks plot — dağılım ve aykırı değer analizi."""
        self._ensure_fig()
        
        if x_col and y_col:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=data, x=x_col, y=y_col, ax=ax)
            ax.set_title(title, fontsize=14, fontweight="bold")
        else:
            numeric_data = data.select_dtypes(include=[np.number])
            if numeric_data.empty:
                raise ValueError("Sayısal sütun bulunamadı")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=numeric_data, ax=ax)
            ax.set_title(title, fontsize=14, fontweight="bold")
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    def scatter(
        self,
        x: pd.Series | np.ndarray,
        y: pd.Series | np.ndarray,
        title: str = "Dağılım Grafiği",
        xlabel: str = None,
        ylabel: str = None,
        color: str = "#2E86AB",
        regression_line: bool = False,
    ) -> plt.Figure:
        """Saçılım grafiği — iki değişken ilişkisi."""
        x_arr = np.asarray(x).flatten()
        y_arr = np.asarray(y).flatten()
        
        mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
        x_clean = x_arr[mask]
        y_clean = y_arr[mask]
        
        if len(x_clean) < 2:
            raise ValueError("En az 2 geçerli veri noktası gerekli")
        
        self._ensure_fig()
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(x_clean, y_clean, color=color, alpha=0.6, edgecolors="white", s=50)
        
        if regression_line:
            coeffs = np.polyfit(x_clean, y_clean, 1)
            poly = np.poly1d(coeffs)
            x_line = np.linspace(np.min(x_clean), np.max(x_clean), 100)
            ax.plot(x_line, poly(x_line), "r--", linewidth=2, label=f"Regresyon (R²={np.corrcoef(x_clean, y_clean)[0,1]**2:.3f})")
            ax.legend()
        
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(xlabel or "X")
        ax.set_ylabel(ylabel or "Y")
        
        plt.tight_layout()
        return fig
    
    def time_series(
        self,
        data: pd.Series | np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        title: str = "Zaman Serisi",
        xlabel: str = "Tarih",
        ylabel: str = "Değer",
    ) -> plt.Figure:
        """Zaman serisi grafiği."""
        arr = np.asarray(data).flatten()
        
        self._ensure_fig()
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if dates is not None and len(dates) == len(arr):
            ax.plot(dates, arr, marker="o", linewidth=2, markersize=4)
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m"))
            plt.xticks(rotation=45)
        else:
            ax.plot(range(len(arr)), arr, marker="o", linewidth=2, markersize=4)
        
        # Hareketli ortalama (3 periyot)
        if len(arr) >= 3:
            ma = pd.Series(arr).rolling(window=min(3, len(arr)//2), min_periods=1).mean()
            ax.plot(range(len(ma)), ma, "r--", linewidth=1.5, label="Hareketli Ortalama")
            ax.legend()
        
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        plt.tight_layout()
        return fig
    
    def correlation_heatmap(
        self,
        data: pd.DataFrame,
        title: str = "Korelasyon Isı Haritası",
    ) -> plt.Figure:
        """Korelasyon ısı haritası."""
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            raise ValueError("Sayısal sütun bulunamadı")
        
        self._ensure_fig()
        fig, ax = plt.subplots(figsize=(10, 8))
        
        corr_matrix = numeric_data.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            square=True,
            ax=ax,
            vmin=-1,
            vmax=1,
        )
        
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig
    
    def bar_chart(
        self,
        categories: list[str],
        values: list[float] | np.ndarray,
        title: str = "Çubuk Grafiği",
        xlabel: str = "Kategori",
        ylabel: str = "Değer",
        color: str = "#2E86AB",
    ) -> plt.Figure:
        """Çubuk grafiği."""
        self._ensure_fig()
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(categories, values, color=color, alpha=0.8, edgecolor="white")
        
        # Değerleri çubukların üzerine yaz
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    def qq_plot(
        self,
        data: pd.Series | np.ndarray,
        title: str = "Q-Q Grafiği",
    ) -> plt.Figure:
        """Normal Q-Q grafiği (Premium Program: Descriptive Statistics → Explore)."""
        arr = np.asarray(data, dtype=float).flatten()
        arr = arr[~np.isnan(arr)]
        if len(arr) < 3:
            raise ValueError("Q-Q grafiği için en az 3 geçerli veri gerekli")
        
        self._ensure_fig()
        fig, ax = plt.subplots(figsize=(8, 8))
        
        (osm, osr), (slope, intercept, r) = stats.probplot(arr, dist="norm")
        ax.scatter(osm, osr, color="#2E86AB", alpha=0.6, edgecolors="white", s=50)
        x_line = np.linspace(osm.min(), osm.max(), 100)
        ax.plot(x_line, intercept + slope * x_line, "r--", linewidth=2,
                label=f"Referans doğrusu (r={r:.3f})")
        
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Teorik Kantiller")
        ax.set_ylabel("Gözlenen Kantiller")
        ax.legend()
        plt.tight_layout()
        return fig

    def pp_plot(
        self,
        data: pd.Series | np.ndarray,
        title: str = "P-P Grafiği",
    ) -> plt.Figure:
        """Normal P-P grafiği (Premium Program: Descriptive Statistics → Explore)."""
        arr = np.asarray(data, dtype=float).flatten()
        arr = arr[~np.isnan(arr)]
        if len(arr) < 3:
            raise ValueError("P-P grafiği için en az 3 geçerli veri gerekli")

        self._ensure_fig()
        fig, ax = plt.subplots(figsize=(8, 8))

        observed = np.sort(arr)
        observed_prob = (np.arange(1, len(observed) + 1) - 0.5) / len(observed)
        expected_prob = stats.norm.cdf(observed, loc=observed.mean(),
                                       scale=observed.std(ddof=1))
        ax.scatter(expected_prob, observed_prob, color="#2E86AB", alpha=0.6,
                   edgecolors="white", s=50)
        ax.plot([0, 1], [0, 1], "r--", linewidth=2, label="Referans doğrusu")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Beklenen Kümülatif Olasılık")
        ax.set_ylabel("Gözlenen Kümülatif Olasılık")
        ax.legend()
        plt.tight_layout()
        return fig

    def pie_chart(
        self,
        labels: list,
        values: list,
        title: str = "Pasta Grafiği",
    ) -> plt.Figure:
        """Pasta grafiği (Premium Program: Graphs → Legacy Dialogs → Pie)."""
        values = np.asarray(values, dtype=float)
        if len(labels) != len(values) or len(values) < 2:
            raise ValueError("En az 2 etiket/değer çifti gerekli")
        if (values < 0).any():
            raise ValueError("Değerler negatif olamaz")

        self._ensure_fig()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90,
               wedgeprops={"edgecolor": "white"})
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig

    def line_chart(
        self,
        x: list | np.ndarray,
        y: list | np.ndarray,
        title: str = "Çizgi Grafiği",
        xlabel: str = "X",
        ylabel: str = "Y",
    ) -> plt.Figure:
        """Çizgi grafiği (Premium Program: Graphs → Legacy Dialogs → Line)."""
        x_arr = np.asarray(x)
        y_arr = np.asarray(y, dtype=float)
        if len(x_arr) != len(y_arr) or len(y_arr) < 2:
            raise ValueError("En az 2 eşleşmiş nokta gerekli")

        self._ensure_fig()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_arr, y_arr, marker="o", color="#2E86AB", linewidth=2)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.tight_layout()
        return fig

    def error_bar(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        ci: float = 0.95,
        title: str = "Hata Çubukları",
    ) -> plt.Figure:
        """Hata çubuğu grafiği — grup ortalaması ± GA
        (Premium Program: Graphs → Legacy Dialogs → Error Bar)."""
        for col in (x_col, y_col):
            if col not in data.columns:
                raise ValueError(f"Sütun bulunamadı: {col}")

        valid = data[[x_col, y_col]].dropna()
        grouped = valid.groupby(x_col, observed=True)[y_col]
        means = grouped.mean()
        n = grouped.count()
        se = grouped.std() / np.sqrt(n)
        z = stats.norm.ppf(0.5 + ci / 2)
        errors = z * se

        self._ensure_fig()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar([str(v) for v in means.index], means.values,
                    yerr=errors.values, fmt="o", color="#2E86AB",
                    capsize=6, linewidth=2)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        plt.tight_layout()
        return fig
    
    def save(self, filename: str, fig: Optional[plt.Figure] = None):
        """Grafiği dosyaya kaydet.
        
        fig belirtilmezse aktif figür kaydedilir.
        """
        target = fig if fig is not None else plt.gcf()
        target.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Grafik kaydedildi: {filename}")
    
    @staticmethod
    def close():
        """Mevcut figürü kapat."""
        plt.close("all")
