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

    def _dogrula_sutunlar(self, data: pd.DataFrame, cols: list):
        """Verilen sütun adlarının DataFrame'de varlığını doğrular."""
        eksik = [c for c in cols if c and c not in data.columns]
        if eksik:
            raise ValueError(f"Sütun bulunamadı: {eksik}")

    def _palet(self, hue_kolon=None, data=None):
        """Hue düzey sayısına kırpılmış palet (seaborn uyarılarını önler)."""
        if hue_kolon is None or data is None:
            return None
        n = int(data[hue_kolon].nunique())
        if len(self._palette) > n:
            return self._palette[:n]
        return self._palette

    def violin_plot(self, data: pd.DataFrame, x_col: str, y_col: str,
                    hue: str = None,
                    title: str = "Violin Grafiği") -> plt.Figure:
        """Violin grafiği — grup bazında dağılım yoğunluğu."""
        self._dogrula_sutunlar(data, [x_col, y_col, hue])
        self._ensure_fig()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(data=data, x=x_col, y=y_col, hue=hue, inner="box",
                       palette=self._palet(hue, data), ax=ax)
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig

    def raincloud_plot(self, data: pd.DataFrame, y_col: str, group_col: str,
                       title: str = "Raincloud Grafiği") -> plt.Figure:
        """Raincloud grafiği — yarım yoğunluk + kutu + jitter noktaları."""
        self._dogrula_sutunlar(data, [y_col, group_col])
        work = data[[y_col, group_col]].dropna()
        groups = sorted(work[group_col].unique(), key=str)
        if len(groups) < 2:
            raise ValueError("Raincloud için en az 2 grup gerekli")
        self._ensure_fig()
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, g in enumerate(groups):
            vals = work.loc[work[group_col] == g, y_col].to_numpy(
                dtype=float)
            renk = self._palette[i % len(self._palette)]
            kde = stats.gaussian_kde(vals)
            xs = np.linspace(vals.min(), vals.max(), 200)
            dens = kde(xs)
            dens = dens / dens.max() * 0.45
            ax.fill_betweenx(xs, i - dens, i, color=renk, alpha=0.55)
            ax.boxplot([vals], positions=[i + 0.18], widths=0.1,
                       patch_artist=True, showfliers=False,
                       boxprops={"facecolor": renk, "alpha": 0.85})
            rng = np.random.default_rng(i)
            jit = i + 0.34 + rng.uniform(-0.035, 0.035, len(vals))
            ax.scatter(jit, vals, s=14, alpha=0.45, color=renk)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([str(g) for g in groups])
        ax.set_ylabel(y_col)
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig

    def ridge_plot(self, data: pd.DataFrame, value_col: str, group_col: str,
                   title: str = "Ridge (Joyplot) Grafiği") -> plt.Figure:
        """Ridge grafiği — üst üste kaymış grup yoğunlukları."""
        self._dogrula_sutunlar(data, [value_col, group_col])
        work = data[[value_col, group_col]].dropna()
        groups = sorted(work[group_col].unique(), key=str)
        if len(groups) < 2:
            raise ValueError("Ridge için en az 2 grup gerekli")
        self._ensure_fig()
        fig, ax = plt.subplots(figsize=(10, 7))
        step = 1.0
        for i, g in enumerate(groups):
            vals = work.loc[work[group_col] == g, value_col].to_numpy(
                dtype=float)
            kde = stats.gaussian_kde(vals)
            xs = np.linspace(vals.min(), vals.max(), 200)
            dens = kde(xs)
            dens = dens / dens.max() * step * 0.9
            renk = self._palette[i % len(self._palette)]
            ax.fill_between(xs, i * step, i * step + dens, color=renk,
                            alpha=0.65)
            ax.plot(xs, i * step + dens, color=renk, linewidth=1.5)
        ax.set_yticks([i * step for i in range(len(groups))])
        ax.set_yticklabels([str(g) for g in groups])
        ax.set_xlabel(value_col)
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig

    def pair_grid(self, data: pd.DataFrame, cols: list, hue: str = None,
                  title: str = "Pair Grid"):
        """Çok değişkenli scatter matrisi (en çok 6 sütun).

        seaborn PairGrid döndürür; ``.axes`` (n, n) ndarray'dır,
        ``.figure`` ile Figure'e erişilir.
        """
        kolonlar = list(cols) + ([hue] if hue else [])
        self._dogrula_sutunlar(data, kolonlar)
        if len(cols) < 2:
            raise ValueError("Pair grid için en az 2 sütun gerekli")
        if len(cols) > 6:
            raise ValueError("Pair grid en çok 6 sütun destekler")
        sayisal = data[list(cols)].select_dtypes(include=[np.number])
        if sayisal.shape[1] != len(cols):
            raise ValueError("Pair grid sütunları sayısal olmalı")
        g = sns.pairplot(data, vars=list(cols), hue=hue,
                         palette=self._palet(hue, data))
        g.figure.suptitle(title, y=1.02, fontsize=14, fontweight="bold")
        return g

    def grouped_boxplot(self, data: pd.DataFrame, y_col: str, x_col: str,
                        hue_col: str,
                        title: str = "Gruplu Kutu Grafiği") -> plt.Figure:
        """İki faktörlü kutu grafiği (x × hue)."""
        self._dogrula_sutunlar(data, [y_col, x_col, hue_col])
        self._ensure_fig()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=data, x=x_col, y=y_col, hue=hue_col,
                    palette=self._palet(hue_col, data), ax=ax)
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig

    def strip_plot(self, data: pd.DataFrame, x_col: str, y_col: str,
                   jitter: bool = True,
                   title: str = "Strip Grafiği") -> plt.Figure:
        """Strip (nokta) grafiği — bireysel gözlemler."""
        self._dogrula_sutunlar(data, [x_col, y_col])
        self._ensure_fig()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.stripplot(data=data, x=x_col, y=y_col, jitter=jitter,
                      color=self._palette[0], ax=ax, alpha=0.7)
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig

    def forest_plot(self, effects: list, ci_lower: list, ci_upper: list,
                    labels: list,
                    title: str = "Orman (Forest) Grafiği") -> plt.Figure:
        """Orman grafiği — etki tahminleri ± güven aralıkları."""
        diziler = (effects, ci_lower, ci_upper, labels)
        if len(set(len(d) for d in diziler)) != 1 or len(effects) == 0:
            raise ValueError("Tüm diziler aynı uzunlukta olmalı")
        effects = np.asarray(effects, dtype=float)
        lo = np.asarray(ci_lower, dtype=float)
        hi = np.asarray(ci_upper, dtype=float)
        poz = np.arange(len(effects))
        self._ensure_fig()
        fig, ax = plt.subplots(figsize=(10, max(4, 0.6 * len(effects) + 2)))
        ax.errorbar(effects, poz, xerr=[effects - lo, hi - effects],
                    fmt="o", color=self._palette[0], capsize=5,
                    linewidth=2, markersize=7)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_yticks(poz)
        ax.set_yticklabels([str(x) for x in labels])
        ax.set_xlabel("Etki büyüklüğü")
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig

    def bland_altman_plot(self, x, y,
                          title: str = "Bland-Altman Grafiği") -> plt.Figure:
        """Bland-Altman uyum grafiği — fark vs ortalama, ±1.96 SD."""
        x_arr = np.asarray(x, dtype=float).flatten()
        y_arr = np.asarray(y, dtype=float).flatten()
        if len(x_arr) != len(y_arr) or len(x_arr) < 5:
            raise ValueError("En az 5 eşleşmiş ölçüm gerekli")
        fark = x_arr - y_arr
        ort = (x_arr + y_arr) / 2
        md = float(np.mean(fark))
        sd = float(np.std(fark, ddof=1))
        self._ensure_fig()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(ort, fark, color=self._palette[0], alpha=0.6,
                   edgecolors="white", s=50)
        for cizgi, etiket in ((md, f"Orta fark: {md:.2f}"),
                              (md + 1.96 * sd, "+1.96 SD"),
                              (md - 1.96 * sd, "-1.96 SD")):
            ax.axhline(cizgi, color="red", linestyle="--",
                       linewidth=1.5, label=etiket)
        ax.set_xlabel("İki ölçümün ortalaması")
        ax.set_ylabel("Fark (x − y)")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend()
        plt.tight_layout()
        return fig

    def roc_plot(self, actual, predicted,
                 title: str = "ROC Eğrisi") -> plt.Figure:
        """ROC eğrisi + AUC (analiz.roc_curve üzerine çizim)."""
        from agrista.analysis import roc_curve
        res = roc_curve(actual, predicted)
        self._ensure_fig()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(res["fpr"], res["tpr"], color=self._palette[0],
                linewidth=2, label=f"AUC = {res['auc']:.3f}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Şans doğrusu")
        ax.set_xlabel("1 − Özgüllük (FPR)")
        ax.set_ylabel("Duyarlılık (TPR)")
        ax.set_title(f"{title} (AUC={res['auc']:.3f})",
                     fontsize=14, fontweight="bold")
        ax.legend()
        plt.tight_layout()
        return fig

    def survival_plot(self, time, event, group=None,
                      title: str = "Kaplan-Meier Sağkalım Eğrisi"
                      ) -> plt.Figure:
        """Kaplan-Meier basamak eğrisi (grupluysa bir eğri/grup)."""
        from agrista.survival import kaplan_meier
        self._ensure_fig()
        fig, ax = plt.subplots(figsize=(10, 6))
        if group is None:
            km = kaplan_meier(time, event)
            t = np.concatenate([[0.0], np.asarray(km["time"], dtype=float)])
            s = np.concatenate([[1.0], np.asarray(km["survival"],
                                                  dtype=float)])
            ax.step(t, s, where="post", color=self._palette[0],
                    linewidth=2, label="Tümü")
        else:
            t_arr = np.asarray(time)
            e_arr = np.asarray(event)
            g_arr = np.asarray(group)
            for i, g in enumerate(sorted(np.unique(g_arr), key=str)):
                maske = g_arr == g
                km = kaplan_meier(t_arr[maske], e_arr[maske])
                t = np.concatenate([[0.0],
                                    np.asarray(km["time"], dtype=float)])
                s = np.concatenate([[1.0], np.asarray(km["survival"],
                                                      dtype=float)])
                ax.step(t, s, where="post",
                        color=self._palette[i % len(self._palette)],
                        linewidth=2, label=str(g))
        ax.set_xlabel("Zaman")
        ax.set_ylabel("Sağkalım olasılığı S(t)")
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend()
        plt.tight_layout()
        return fig

    def control_chart(self, values, subgroup_size: int = 5,
                      title: str = "Kontrol Grafiği (X̄)") -> plt.Figure:
        """X̄ kontrol grafiği — alt grup ortalamaları ± 3σ limitleri."""
        arr = np.asarray(values, dtype=float).flatten()
        arr = arr[~np.isnan(arr)]
        if subgroup_size < 2 or len(arr) < 2 * subgroup_size:
            raise ValueError("En az 2 alt grup dolusu veri gerekli")
        n_alt = len(arr) // subgroup_size
        alt = arr[:n_alt * subgroup_size].reshape(n_alt, subgroup_size)
        ortalamalar = alt.mean(axis=1)
        merkez = float(ortalamalar.mean())
        sigma = float(ortalamalar.std(ddof=1))
        self._ensure_fig()
        fig, ax = plt.subplots(figsize=(11, 6))
        x = np.arange(n_alt)
        ihlal = (np.abs(ortalamalar - merkez) > 3 * sigma)
        ax.plot(x, ortalamalar, "o-", color=self._palette[0], linewidth=2)
        ax.scatter(x[ihlal], ortalamalar[ihlal], color="red", s=80,
                   zorder=5, label="Limit ihlali")
        for cizgi, etiket in ((merkez, "Merkez"),
                              (merkez + 3 * sigma, "+3σ"),
                              (merkez - 3 * sigma, "-3σ")):
            ax.axhline(cizgi, color="red" if etiket != "Merkez" else "gray",
                       linestyle="--", linewidth=1.2, label=etiket)
        ax.set_xlabel("Alt grup")
        ax.set_ylabel("Ortalama")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend()
        plt.tight_layout()
        return fig

    def residual_plot(self, fitted, residuals,
                      title: str = "Artık Grafiği") -> plt.Figure:
        """Artık diyagnostiği — uydurulan değerlere karşı artıkler."""
        f = np.asarray(fitted, dtype=float).flatten()
        r = np.asarray(residuals, dtype=float).flatten()
        if len(f) != len(r) or len(f) < 5:
            raise ValueError("En az 5 eşleşmiş artık gerekli")
        self._ensure_fig()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(f, r, color=self._palette[0], alpha=0.6,
                   edgecolors="white", s=50)
        ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Uydurulan değerler")
        ax.set_ylabel("Artıklar")
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig

    def hexbin_plot(self, x, y, gridsize: int = 30,
                    title: str = "Hexbin (2B Yoğunluk)") -> plt.Figure:
        """Hexbin grafiği — büyük veri için 2B yoğunluk."""
        x_arr = np.asarray(x, dtype=float).flatten()
        y_arr = np.asarray(y, dtype=float).flatten()
        maske = ~(np.isnan(x_arr) | np.isnan(y_arr))
        x_arr, y_arr = x_arr[maske], y_arr[maske]
        if len(x_arr) < 10:
            raise ValueError("Hexbin için en az 10 nokta gerekli")
        self._ensure_fig()
        fig, ax = plt.subplots(figsize=(10, 6))
        hb = ax.hexbin(x_arr, y_arr, gridsize=gridsize, cmap="viridis",
                       mincnt=1)
        fig.colorbar(hb, ax=ax, label="Yoğunluk")
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig

    def stacked_area(self, labels: list, series_dict: dict,
                     title: str = "Yığılmış Alan Grafiği") -> plt.Figure:
        """Yığılmış alan grafiği — bileşenlerin zamana göre payı."""
        if len(labels) == 0 or len(series_dict) == 0:
            raise ValueError("Etiket ve en az bir seri gerekli")
        for ad, seri in series_dict.items():
            if len(seri) != len(labels):
                raise ValueError(f"'{ad}' serisi etiket uzunluğunda değil")
        self._ensure_fig()
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.stackplot(range(len(labels)), *series_dict.values(),
                     labels=list(series_dict.keys()),
                     colors=self._palette, alpha=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([str(x) for x in labels])
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="upper left")
        plt.tight_layout()
        return fig

    def growth_curve_plot(self, time, observed, model: str = "logistic",
                          title: str = None) -> plt.Figure:
        """Büyüme eğrisi — gözlemler + uydurulan model.

        Desteklenen modeller: logistic, monomolecular.
        """
        from agrista.models import GrowthModel
        t = np.asarray(time, dtype=float).flatten()
        y = np.asarray(observed, dtype=float).flatten()
        maske = ~(np.isnan(t) | np.isnan(y))
        t, y = t[maske], y[maske]
        if len(t) < 4:
            raise ValueError("Büyüme eğrisi için en az 4 nokta gerekli")
        gm = GrowthModel(model_type=model)
        if model == "logistic":
            gm.fit_logistic(t, y)
            fonk = gm.logistic
            args = (gm.params["K"], gm.params["r"], gm.params["t0"])
            ad = "Lojistik"
        elif model == "monomolecular":
            gm.fit_monomolecular(t, y)
            fonk = gm.monomolecular
            p = gm.params
            args = (p["A"], p["k"], p.get("t0", 0.0))
            ad = "Monomoleküler"
        else:
            raise ValueError(f"Desteklenmeyen model: {model}")
        self._ensure_fig()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(t, y, color=self._palette[0], alpha=0.7,
                   label="Gözlem")
        t_cek = np.linspace(t.min(), t.max(), 200)
        ax.plot(t_cek, fonk(t_cek, *args), color="red", linewidth=2,
                label=f"{ad} uydurma")
        ax.set_xlabel("Zaman")
        ax.set_ylabel("Ölçüm")
        ax.set_title(title or f"Büyüme Eğrisi ({ad})",
                     fontsize=14, fontweight="bold")
        ax.legend()
        plt.tight_layout()
        return fig

    def slope_plot(self, before, after, labels: list = None,
                   title: str = "Eğim (Slope) Grafiği") -> plt.Figure:
        """Eğim grafiği — eşleşmiş önce/sonra değişimleri."""
        b = np.asarray(before, dtype=float).flatten()
        a = np.asarray(after, dtype=float).flatten()
        if len(b) != len(a) or len(b) < 2:
            raise ValueError("before/after eşit ve ≥2 uzunlukta olmalı")
        etiketler = labels or [str(i + 1) for i in range(len(b))]
        self._ensure_fig()
        fig, ax = plt.subplots(figsize=(9, 6))
        for i in range(len(b)):
            ax.plot([0, 1], [b[i], a[i]], "o-",
                    color=self._palette[i % len(self._palette)],
                    linewidth=1.8, label=str(etiketler[i]))
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Önce", "Sonra"])
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        return fig

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
    
    def save(self, filename: str, fig: Optional[plt.Figure] = None,
             dpi: Optional[int] = None):
        """Grafiği dosyaya kaydet (varsayılan dpi temadan gelir)."""
        target = fig if fig is not None else plt.gcf()
        hedef_dpi = dpi or self._theme["dpi"]
        target.savefig(filename, dpi=hedef_dpi, bbox_inches="tight")
        print(f"Grafik kaydedildi: {filename}")

    def save_multi(self, filename_base: str,
                   figs: list, fmts: tuple = ("png", "svg")) -> list:
        """Birden çok grafiği birden çok formatta kaydeder; yolları döndürür."""
        yollar = []
        for i, fig in enumerate(figs):
            for fmt in fmts:
                yol = f"{filename_base}_{i + 1}.{fmt}"
                fig.savefig(yol, dpi=self._theme["dpi"],
                            bbox_inches="tight")
                yollar.append(yol)
        print(f"{len(yollar)} grafik dosyası kaydedildi.")
        return yollar

    def export_html(self, fig: plt.Figure, path: str,
                    title: str = "Agrista Grafik"):
        """Grafiği tek dosyalık HTML raporu olarak dışa aktarır."""
        import base64
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=self._theme["dpi"],
                    bbox_inches="tight")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        html = (
            "<!DOCTYPE html><html lang=\"tr\"><head><meta charset=\"utf-8\">"
            f"<title>{title}</title><style>"
            "body{margin:0;padding:24px;font-family:sans-serif;"
            "background:#f4f6f8}h1{color:#2E86AB;font-size:20px}"
            ".kart{background:#fff;border:1px solid #dfe3e8;border-radius:"
            "8px;padding:16px;max-width:960px;margin:0 auto}"
            "img{max-width:100%;height:auto}</style></head><body>"
            f"<div class=\"kart\"><h1>{title}</h1>"
            f"<img src=\"data:image/png;base64,{b64}\" alt=\"{title}\"/>"
            "<p style=\"color:#888;font-size:12px\">Agrista ile "
            "üretilmiştir.</p></div></body></html>")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)
        print(f"HTML raporu kaydedildi: {path}")
    
    @staticmethod
    def close():
        """Mevcut figürü kapat."""
        plt.close("all")
