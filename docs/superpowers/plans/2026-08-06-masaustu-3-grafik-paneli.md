# Plan: Masaüstü Wave 1 — Grafik Paneli (Gömülü Canvas)

**Spec:** `docs/superpowers/specs/2026-08-06-masaustu-dagitim-design.md` (§7)

**Goal:** "Grafik" sekmesini 12 grafik tipini üreten gömülü matplotlib
canvas paneline dönüştürmek; `AgristaPlotter` birebir yeniden kullanılır.

**Architecture:** `chart_view.py` içinde `CHART_TYPES` bildirimsel
grafik kaydı + `ChartPanel(QWidget)`; `MainWindow.grafik_sekmesi`
placeholder'ı bu panelle değiştirilir, veri yüklendiğinde panele bildirilir.

**Tech Stack:** PySide6, `matplotlib.backends.backend_qtagg`, mevcut
`AgristaPlotter`. Ön koşul: Plan 1-2 tamamlanmış.

**Global Constraints (spec'ten aynen):**
1. PySide6 opsiyonel ekstra; başka yeni bağımlılık yok; ağ kodu stdlib.
2. Yalnızca "Premium Program" adı; eski ad yasak.
3. GUI Türkçe; mevcut analiz/grafik fonksiyonları değişmez.
4. Menü kuralı bu planda geçerli değil.
5. İmza konusu bu planda yok.
6. TDD zorunlu; widget testleri pytest-qt + offscreen.
7. Sürüm bu planda değişmez.
8. `pytest` tam paket yeşil + `flake8` temiz olmadan görev bitmez.

## Dosya Yapısı

| Dosya | Sorumluluk |
|---|---|
| `agrista/gui/chart_view.py` (yeni) | `CHART_TYPES`, `ChartPanel` |
| `agrista/gui/main_window.py` | placeholder → ChartPanel |
| `tests/test_gui_chart.py` (yeni) | grafik paneli testleri |

## Grafik Tipi Kaydı (12 tip)

| Tip | Girdi alanları | Plotter çağrısı |
|---|---|---|
| Histogram | degisken | `histogram(df[k])` |
| Kutu Grafiği | grup, yanit | `boxplot(df, x_col, y_col)` |
| Saçılım | x, y | `scatter(df[x], df[y])` |
| Violin | grup, yanit | `violin_plot(df, x_col=grup, y_col=yanit)` |
| Raincloud | grup, yanit | `raincloud_plot(df, y_col=yanit, group_col=grup)` |
| Ridge | grup, yanit | `ridge_plot(df, value_col=yanit, group_col=grup)` |
| Çubuk (ortalama) | kategori, deger | `bar_chart` (grup ortalaması) |
| Çizgi | x, y | `line_chart(list(df[x]), list(df[y]))` |
| Pasta | kategori, deger | `pie_chart` (grup toplamları) |
| Q-Q | degisken | `qq_plot(df[k])` |
| Korelasyon Isı Haritası | — | `correlation_heatmap(df)` |
| Hata Çubuğu | grup, yanit | `error_bar(df, x_col, y_col)` |

---

## Task 1: ChartPanel (TDD)

**Files:** Test: `tests/test_gui_chart.py` (Create) · Create: `agrista/gui/chart_view.py`
**Interfaces:** `ChartPanel(df=None)`; `uret(tip, degerler) -> Figure`; `kaydet(path)`; `set_dataframe(df)`.

- [ ] **RED** — `tests/test_gui_chart.py` oluştur:

```python
"""Agrista GUI grafik paneli testleri."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
import pytest


@pytest.fixture
def df():
    return pd.DataFrame({
        "grup": ["A", "A", "B", "B", "A", "B"],
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "y": [2.0, 4.0, 5.0, 4.0, 6.0, 7.0],
    })


@pytest.fixture
def panel(qtbot, df):
    from agrista.gui.chart_view import ChartPanel
    p = ChartPanel(df)
    qtbot.addWidget(p)
    return p


class TestChartPanel:
    def test_12_tip_kayitli(self):
        from agrista.gui.chart_view import CHART_TYPES
        assert len(CHART_TYPES) == 12

    def test_histogram_uret(self, panel):
        fig = panel.uret("Histogram", {"degisken": "x"})
        assert len(fig.axes) >= 1

    def test_scatter_uret(self, panel):
        fig = panel.uret("Saçılım", {"x": "x", "y": "y"})
        assert len(fig.axes) >= 1

    def test_violin_uret(self, panel):
        fig = panel.uret("Violin", {"grup": "grup", "yanit": "y"})
        assert len(fig.axes) >= 1

    def test_isi_haritasi_alansiz(self, panel):
        fig = panel.uret("Korelasyon Isı Haritası", {})
        assert len(fig.axes) >= 1

    def test_bilinmeyen_tip_hatasi(self, panel):
        with pytest.raises(ValueError):
            panel.uret("Olmayan", {})

    def test_kaydet(self, panel, tmp_path):
        panel.uret("Histogram", {"degisken": "x"})
        hedef = tmp_path / "g.png"
        panel.kaydet(str(hedef))
        assert hedef.exists() and hedef.stat().st_size > 0

    def test_eksik_deger_hatasi(self, panel):
        with pytest.raises(ValueError):
            panel.uret("Saçılım", {"x": "x"})
```

- [ ] Çalıştır: `.venv/bin/python -m pytest tests/test_gui_chart.py -x -q`
      → ModuleNotFoundError.
- [ ] **GREEN** — `agrista/gui/chart_view.py` oluştur:

```python
"""Agrista GUI grafik paneli — gömülü matplotlib canvas."""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QPushButton,
                               QVBoxLayout, QWidget)

from agrista.viz import AgristaPlotter


def _bar_ortalama(plotter, df, d):
    say = df.groupby(d["kategori"], observed=True)[d["deger"]].mean()
    return plotter.bar_chart([str(k) for k in say.index], list(say.values),
                             xlabel=d["kategori"],
                             ylabel=f"Ortalama {d['deger']}")


def _pasta_toplam(plotter, df, d):
    say = df.groupby(d["kategori"], observed=True)[d["deger"]].sum()
    return plotter.pie_chart([str(k) for k in say.index], list(say.values))


CHART_TYPES = {
    "Histogram": {"alanlar": ["degisken"], "ciz":
                  lambda p, df, d: p.histogram(df[d["degisken"]].dropna())},
    "Kutu Grafiği": {"alanlar": ["grup", "yanit"], "ciz":
                     lambda p, df, d: p.boxplot(df, d["grup"], d["yanit"])},
    "Saçılım": {"alanlar": ["x", "y"], "ciz":
                lambda p, df, d: p.scatter(df[d["x"]], df[d["y"]])},
    "Violin": {"alanlar": ["grup", "yanit"], "ciz":
               lambda p, df, d: p.violin_plot(df, x_col=d["grup"],
                                              y_col=d["yanit"])},
    "Raincloud": {"alanlar": ["grup", "yanit"], "ciz":
                  lambda p, df, d: p.raincloud_plot(df, y_col=d["yanit"],
                                                   group_col=d["grup"])},
    "Ridge": {"alanlar": ["grup", "yanit"], "ciz":
              lambda p, df, d: p.ridge_plot(df, value_col=d["yanit"],
                                            group_col=d["grup"])},
    "Çubuk (ortalama)": {"alanlar": ["kategori", "deger"],
                         "ciz": _bar_ortalama},
    "Çizgi": {"alanlar": ["x", "y"], "ciz":
              lambda p, df, d: p.line_chart(list(df[d["x"]]),
                                            list(df[d["y"]]))},
    "Pasta": {"alanlar": ["kategori", "deger"], "ciz": _pasta_toplam},
    "Q-Q": {"alanlar": ["degisken"], "ciz":
            lambda p, df, d: p.qq_plot(df[d["degisken"]].dropna())},
    "Korelasyon Isı Haritası": {"alanlar": [], "ciz":
                                lambda p, df, d: p.correlation_heatmap(df)},
    "Hata Çubuğu": {"alanlar": ["grup", "yanit"], "ciz":
                    lambda p, df, d: p.error_bar(df, d["grup"], d["yanit"])},
}


class ChartPanel(QWidget):
    """Grafik tipi seçimi + gömülü canvas."""

    def __init__(self, df=None, parent=None):
        super().__init__(parent)
        self.df = df
        self.plotter = AgristaPlotter()
        self._son_figur = None

        duzen = QVBoxLayout(self)
        self.bilgi = QLabel("Bir grafik tipi seçin ve alanları doldurun.")
        duzen.addWidget(self.bilgi)
        self.canvas = FigureCanvasQTAgg(Figure(figsize=(8, 5)))
        duzen.addWidget(self.canvas, stretch=1)

    def set_dataframe(self, df):
        self.df = df

    def uret(self, tip: str, degerler: dict) -> Figure:
        """Grafik üretir, canvas'a çizer ve figürü döndürür."""
        if tip not in CHART_TYPES:
            raise ValueError(f"Bilinmeyen grafik tipi: {tip}")
        if self.df is None or self.df.empty:
            raise ValueError("Grafik için önce veri yükleyin")
        spec = CHART_TYPES[tip]
        eksik = [a for a in spec["alanlar"] if not degerler.get(a)]
        if eksik:
            raise ValueError(f"Eksik alanlar: {eksik}")
        fig = spec["ciz"](self.plotter, self.df, degerler)
        self._son_figur = fig
        self.canvas.figure = fig
        self.canvas.draw()
        self.bilgi.setText(f"{tip} grafiği üretildi.")
        return fig

    def kaydet(self, path: str):
        """Son üretilen grafiği dosyaya kaydeder."""
        if self._son_figur is None:
            raise ValueError("Kaydedilecek grafik yok")
        self.plotter.save(path, fig=self._son_figur)
```

  Not: `FigureCanvasQTAgg` import'u Qt backend'ini otomatik kurar;
  ayrıca `matplotlib.use` çağrısı yapılmaz (testlerdeki Agg ayarı
  conftest düzeyinde kalır, canvas import'u ile çelişmez).
- [ ] Çalıştır: `.venv/bin/python -m pytest tests/test_gui_chart.py -q` → yeşil.
- [ ] Commit: `feat(gui): grafik paneli — 12 tip, gömülü canvas, kaydetme`

## Task 2: Ana pencere entegrasyonu + kapanış (TDD)

**Files:** Modify: `agrista/gui/main_window.py`, `tests/test_gui_window.py`
**Interfaces:** `MainWindow.grafik_paneli`; veri açılınca `set_dataframe`.

- [ ] **RED** — `tests/test_gui_window.py` sonuna ekle:

```python
class TestGrafikEntegrasyon:
    def test_grafik_sekmesi_panel(self, pencere):
        from agrista.gui.chart_view import ChartPanel
        assert isinstance(pencere.grafik_paneli, ChartPanel)

    def test_veri_acilinca_panel_guncellenir(self, pencere, tmp_path):
        pencere.open_file(_csv(tmp_path))
        assert pencere.grafik_paneli.df is pencere.df
```

- [ ] Çalıştır: `-k GrafikEntegrasyon` → AttributeError.
- [ ] **GREEN** — `main_window.py` değişiklikleri: import
      `from agrista.gui.chart_view import ChartPanel`; `__init__` içinde
      placeholder `QTextEdit` yerine `self.grafik_paneli = ChartPanel()`
      ve `sekmeler.addTab(self.grafik_paneli, "Grafik")`; `open_file`
      sonuna `self.grafik_paneli.set_dataframe(self.df)` ekle.
- [ ] Çalıştır: `.venv/bin/python -m pytest tests/ -q` → tam paket yeşil;
      `.venv/bin/python -m flake8 agrista tests` temiz.
- [ ] Commit: `feat(gui): grafik paneli ana pencere entegrasyonu`
