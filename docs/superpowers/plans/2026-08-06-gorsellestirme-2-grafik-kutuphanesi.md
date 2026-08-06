# Plan: Görselleştirme — Premium Grafik Kütüphanesi (16 yeni metot)

**Spec:** `docs/superpowers/specs/2026-08-06-gorsellestirme-design.md` (§5)

**Goal:** `AgristaPlotter`'a 16 yeni istatistiksel grafik metodu eklemek
(dağılım, tanı, model grafikleri); her metot `Figure` döndürür.

**Architecture:** Tüm metotlar `agrista/viz/plotter.py` sınıfına eklenir;
analiz bağımlılıkları (`roc_curve`, `kaplan_meier`, `GrowthModel`)
döngüsel import'u önlemek için metot İÇİNDE import edilir.

**Tech Stack:** matplotlib, seaborn, scipy.stats, numpy, pandas.
Ön koşul: "Temel Katman" planı tamamlanmış (`plotter.py`, `themes.py`,
`self._palette`, `self._theme` mevcut).

**Global Constraints (spec'ten aynen):**
1. Plotly çekirdek bağımlılık (bu planda kullanılmaz).
2. Yalnızca "Premium Program" adı; eski ad yasak.
3. Statik grafikler `AgristaPlotter` metodu, `Figure` döner, Türkçe
   docstring; mevcut 12 metodun imzası/davranışı DEĞİŞMEZ.
4. CLI kuralı sonraki planda.
5. Menü kuralı sonraki planda.
6. TDD zorunlu; grafik testleri `matplotlib.use("Agg")` altında; dosya
   testleri `tmp_path` kullanır.
7. Sürüm bu planda değişmez.
8. `pytest` tam paket yeşil + `flake8` temiz olmadan görev bitmez.

## Dosya Yapısı

| Dosya | Sorumluluk |
|---|---|
| `agrista/viz/plotter.py` | 16 yeni metot |
| `tests/test_viz_premium.py` (yeni) | tüm yeni metot testleri |

---

## Task 1: Dağılım grafikleri (6 metot, TDD)

**Files:** Test: `tests/test_viz_premium.py` (Create) · Modify: `agrista/viz/plotter.py`
**Interfaces:** `violin_plot`, `raincloud_plot`, `ridge_plot`, `pair_grid`, `grouped_boxplot`, `strip_plot`.

- [ ] **RED** — `tests/test_viz_premium.py` oluştur:

```python
"""Agrista premium grafik kütüphanesi testleri (16 yeni metot)."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from agrista.viz import AgristaPlotter


@pytest.fixture(autouse=True)
def _kapla():
    yield
    AgristaPlotter.close()


@pytest.fixture
def plotter():
    return AgristaPlotter()


def _grup_veri(seed=3, n=60):
    rng = np.random.default_rng(seed)
    grup = rng.choice(["A", "B", "C"], n)
    etki = {"A": 0.0, "B": 1.5, "C": 3.0}
    return pd.DataFrame({"grup": grup,
                         "verim": np.array([etki[g] for g in grup])
                                  + rng.normal(0, 1, n)})


class TestDagilimGrafikleri:
    def test_violin(self, plotter):
        fig = plotter.violin_plot(_grup_veri(), x_col="grup", y_col="verim")
        assert fig.axes[0].get_title() == "Violin Grafiği"

    def test_violin_hue(self, plotter):
        df = _grup_veri()
        df["blok"] = ["x", "y"] * (len(df) // 2)
        fig = plotter.violin_plot(df, x_col="grup", y_col="verim",
                                  hue="blok")
        assert len(fig.axes) >= 1

    def test_raincloud(self, plotter):
        fig = plotter.raincloud_plot(_grup_veri(), y_col="verim",
                                     group_col="grup")
        assert "Raincloud" in fig.axes[0].get_title()

    def test_ridge(self, plotter):
        fig = plotter.ridge_plot(_grup_veri(), value_col="verim",
                                 group_col="grup")
        assert len(fig.axes[0].collections) >= 3

    def test_pair_grid(self, plotter):
        rng = np.random.default_rng(5)
        df = pd.DataFrame({"a": rng.normal(size=40),
                           "b": rng.normal(size=40),
                           "c": rng.normal(size=40)})
        fig = plotter.pair_grid(df, cols=["a", "b", "c"])
        assert fig.axes.shape == (3, 3)

    def test_grouped_boxplot(self, plotter):
        df = _grup_veri()
        df["blok"] = ["x", "y"] * (len(df) // 2)
        fig = plotter.grouped_boxplot(df, y_col="verim", x_col="grup",
                                      hue_col="blok")
        assert fig.axes[0].get_title() == "Gruplu Kutu Grafiği"

    def test_strip(self, plotter):
        fig = plotter.strip_plot(_grup_veri(), x_col="grup", y_col="verim")
        assert len(fig.axes[0].collections) >= 1

    def test_violin_eksik_sutun_hatasi(self, plotter):
        with pytest.raises(ValueError):
            plotter.violin_plot(_grup_veri(), x_col="yok", y_col="verim")

    def test_pair_grid_cok_sutun_hatasi(self, plotter):
        df = pd.DataFrame(np.random.default_rng(1).normal(size=(10, 8)),
                          columns=[f"s{i}" for i in range(8)])
        with pytest.raises(ValueError):
            plotter.pair_grid(df, cols=list(df.columns))

    def test_raincloud_tek_grup_hatasi(self, plotter):
        df = _grup_veri()
        df["grup"] = "A"
        with pytest.raises(ValueError):
            plotter.raincloud_plot(df, y_col="verim", group_col="grup")
```

- [ ] Çalıştır: `.venv/bin/python -m pytest tests/test_viz_premium.py -x -q`
      → AttributeError (metot yok).
- [ ] **GREEN** — `agrista/viz/plotter.py` sınıfına ekle:

```python
    def _sütunlari_dogrula(self, data: pd.DataFrame, cols: list):
        eksik = [c for c in cols if c and c not in data.columns]
        if eksik:
            raise ValueError(f"Sütun bulunamadı: {eksik}")

    def violin_plot(self, data: pd.DataFrame, x_col: str, y_col: str,
                    hue: str = None,
                    title: str = "Violin Grafiği") -> plt.Figure:
        """Violin grafiği — grup bazında dağılım yoğunluğu."""
        self._sütunlari_dogrula(data, [x_col, y_col, hue])
        self._ensure_fig()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(data=data, x=x_col, y=y_col, hue=hue, inner="box",
                       palette=self._palette, ax=ax)
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig

    def raincloud_plot(self, data: pd.DataFrame, y_col: str, group_col: str,
                       title: str = "Raincloud Grafiği") -> plt.Figure:
        """Raincloud grafiği — yarım yoğunluk + kutu + jitter noktaları."""
        self._sütunlari_dogrula(data, [y_col, group_col])
        work = data[[y_col, group_col]].dropna()
        groups = sorted(work[group_col].unique(), key=str)
        if len(groups) < 2:
            raise ValueError("Raincloud için en az 2 grup gerekli")
        self._ensure_fig()
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, g in enumerate(groups):
            vals = work.loc[work[group_col] == g, y_col].to_numpy(dtype=float)
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
        self._sütunlari_dogrula(data, [value_col, group_col])
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
                  title: str = "Pair Grid") -> plt.Figure:
        """Çok değişkenli scatter matrisi (en çok 6 sütun)."""
        kolonlar = list(cols) + ([hue] if hue else [])
        self._sütunlari_dogrula(data, kolonlar)
        if len(cols) < 2:
            raise ValueError("Pair grid için en az 2 sütun gerekli")
        if len(cols) > 6:
            raise ValueError("Pair grid en çok 6 sütun destekler")
        sayisal = data[list(cols)].select_dtypes(include=[np.number])
        if sayisal.shape[1] != len(cols):
            raise ValueError("Pair grid sütunları sayısal olmalı")
        g = sns.pairplot(data, vars=list(cols), hue=hue,
                         palette=self._palette)
        g.figure.suptitle(title, y=1.02, fontsize=14, fontweight="bold")
        return g.figure

    def grouped_boxplot(self, data: pd.DataFrame, y_col: str, x_col: str,
                        hue_col: str,
                        title: str = "Gruplu Kutu Grafiği") -> plt.Figure:
        """İki faktörlü kutu grafiği (x × hue)."""
        self._sütunlari_dogrula(data, [y_col, x_col, hue_col])
        self._ensure_fig()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=data, x=x_col, y=y_col, hue=hue_col,
                    palette=self._palette, ax=ax)
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig

    def strip_plot(self, data: pd.DataFrame, x_col: str, y_col: str,
                   jitter: bool = True,
                   title: str = "Strip Grafiği") -> plt.Figure:
        """Strip (nokta) grafiği — bireysel gözlemler."""
        self._sütunlari_dogrula(data, [x_col, y_col])
        self._ensure_fig()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.stripplot(data=data, x=x_col, y=y_col, jitter=jitter,
                      palette=self._palette, ax=ax, alpha=0.7)
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig
```

  Not: `_sütunlari_dogrula` Türkçe karakterli tanımlayıcı Python'da
  geçerlidir ancak kod tabanı ASCII adlandırma kullanır; yardımcı metodun
  adı `_dogrula_sutunlar` OLACAK (yukarıdaki iki kullanım da
  `self._dogrula_sutunlar(...)` olarak yazılır). Bu düzeltme GREEN
  adımının parçasıdır.
- [ ] Çalıştır: `.venv/bin/python -m pytest tests/test_viz_premium.py -q` → yeşil.
- [ ] Commit: `feat(viz): dağılım grafikleri — violin, raincloud, ridge, pair, gruplu box, strip`

## Task 2: Tanı ve model grafikleri (6 metot, TDD)

**Files:** Modify: `tests/test_viz_premium.py`, `agrista/viz/plotter.py`
**Interfaces:** `forest_plot`, `bland_altman_plot`, `roc_plot`, `survival_plot`, `control_chart`, `residual_plot`.

- [ ] **RED** — `tests/test_viz_premium.py` sonuna ekle:

```python
class TestTaniVeModelGrafikleri:
    def test_forest(self, plotter):
        fig = plotter.forest_plot(effects=[0.5, -0.2, 1.1],
                                  ci_lower=[0.1, -0.6, 0.4],
                                  ci_upper=[0.9, 0.2, 1.8],
                                  labels=["A", "B", "C"])
        assert len(fig.axes[0].lines) >= 1  # sıfır çizgisi

    def test_forest_uzunluk_hatasi(self, plotter):
        with pytest.raises(ValueError):
            plotter.forest_plot(effects=[0.5], ci_lower=[0.1, 0.2],
                                ci_upper=[0.9], labels=["A", "B"])

    def test_bland_altman(self, plotter):
        rng = np.random.default_rng(4)
        x = rng.normal(50, 5, 40)
        fig = plotter.bland_altman_plot(x, x + rng.normal(0, 2, 40))
        # orta çizgi + 2 limit = 3 yatay çizgi
        assert len(fig.axes[0].lines) >= 3

    def test_roc(self, plotter):
        rng = np.random.default_rng(6)
        gercek = rng.integers(0, 2, 100)
        skor = gercek * 0.6 + rng.normal(0, 0.4, 100)
        fig = plotter.roc_plot(gercek, skor)
        assert "AUC" in fig.axes[0].get_title()

    def test_survival(self, plotter):
        rng = np.random.default_rng(7)
        zaman = rng.exponential(10, 50)
        olay = rng.integers(0, 2, 50)
        fig = plotter.survival_plot(zaman, olay)
        assert "Sağkalım" in fig.axes[0].get_title()

    def test_survival_gruplu(self, plotter):
        rng = np.random.default_rng(8)
        zaman = rng.exponential(10, 60)
        olay = rng.integers(0, 2, 60)
        grup = rng.choice(["kontrol", "ilaç"], 60)
        fig = plotter.survival_plot(zaman, olay, group=grup)
        assert len(fig.axes[0].lines) >= 2

    def test_control_chart(self, plotter):
        rng = np.random.default_rng(9)
        vals = rng.normal(100, 2, 50)
        vals[22] = 115.0  # ihlal
        fig = plotter.control_chart(vals, subgroup_size=5)
        assert len(fig.axes[0].lines) >= 3  # merkez + 2 limit

    def test_residual(self, plotter):
        rng = np.random.default_rng(10)
        fitted = rng.normal(0, 1, 60)
        fig = plotter.residual_plot(fitted, rng.normal(0, 0.3, 60))
        assert len(fig.axes[0].lines) >= 1

    def test_bland_altman_az_veri_hatasi(self, plotter):
        with pytest.raises(ValueError):
            plotter.bland_altman_plot([1.0, 2.0], [1.1, 2.1])
```

- [ ] Çalıştır: `-k "Tani"` → AttributeError.
- [ ] **GREEN** — sınıfa ekle:

```python
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
```

- [ ] Çalıştır: `.venv/bin/python -m pytest tests/test_viz_premium.py -q` → yeşil.
- [ ] Commit: `feat(viz): tanı/model grafikleri — forest, Bland-Altman, ROC, sağkalım, kontrol, artık`

## Task 3: Kalan 4 metot + plan kapanışı (TDD)

**Files:** Modify: `tests/test_viz_premium.py`, `agrista/viz/plotter.py`
**Interfaces:** `hexbin_plot`, `stacked_area`, `growth_curve_plot`, `slope_plot`.

- [ ] **RED** — test dosyası sonuna ekle:

```python
class TestDigerGrafikler:
    def test_hexbin(self, plotter):
        rng = np.random.default_rng(11)
        fig = plotter.hexbin_plot(rng.normal(size=300),
                                  rng.normal(size=300))
        assert fig.axes[0].get_title() == "Hexbin (2B Yoğunluk)"

    def test_stacked_area(self, plotter):
        fig = plotter.stacked_area(
            labels=[2020, 2021, 2022],
            series_dict={"buğday": [10.0, 12.0, 11.0],
                         "arpa": [5.0, 6.0, 7.0]})
        assert len(fig.axes[0].collections) >= 2

    def test_stacked_area_bos_hatasi(self, plotter):
        with pytest.raises(ValueError):
            plotter.stacked_area(labels=[], series_dict={})

    def test_growth_curve(self, plotter):
        t = np.linspace(1, 30, 20)
        y = 100 / (1 + np.exp(-0.25 * (t - 15))) + np.random.default_rng(
            12).normal(0, 1.5, 20)
        fig = plotter.growth_curve_plot(t, y, model="logistic")
        assert "Lojistik" in fig.axes[0].get_title() or \
               "büyüme" in fig.axes[0].get_title().lower()

    def test_slope(self, plotter):
        fig = plotter.slope_plot(before=[1.0, 2.0, 3.0],
                                 after=[2.0, 1.5, 4.0],
                                 labels=["a", "b", "c"])
        assert len(fig.axes[0].lines) >= 3

    def test_slope_uzunluk_hatasi(self, plotter):
        with pytest.raises(ValueError):
            plotter.slope_plot(before=[1.0, 2.0], after=[2.0])
```

- [ ] **GREEN** — sınıfa ekle:

```python
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
        """Büyüme eğrisi — gözlemler + uydurulan model (logistic/monomolecular)."""
        from agrista.models import GrowthModel
        from scipy.optimize import curve_fit
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
```

  Not: `GrowthModel.fit_monomolecular` parametre adları (`A`, `k`, `t0`)
  uygulamadan ÖNCE `agrista/models/__init__.py` içinden doğrulanır;
  farklıysa kod o adlara uydurulur.
- [ ] Çalıştır: `.venv/bin/python -m pytest tests/test_viz_premium.py -q` → yeşil.
- [ ] Plan kapanış doğrulaması: `.venv/bin/python -m pytest tests/ -q`
      ve `.venv/bin/python -m flake8 agrista tests` → temiz.
- [ ] Commit: `feat(viz): hexbin, yığılmış alan, büyüme eğrisi, slope — 16 metot tamam`
