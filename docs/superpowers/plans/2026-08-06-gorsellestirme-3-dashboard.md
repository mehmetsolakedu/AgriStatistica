# Plan: Görselleştirme — Etkileşimli Dashboard (Plotly)

**Spec:** `docs/superpowers/specs/2026-08-06-gorsellestirme-design.md` (§6)

**Goal:** Plotly tabanlı 6 etkileşimli grafik fonksiyonu ve tek HTML
üreten `build_dashboard` ile keşif paneli sunmak.

**Architecture:** `agrista/viz/interactive.py` saf fonksiyonlar modülü;
`plotly.express`/`plotly.graph_objects` kullanır; dashboard alt satplot
(`make_subplots`) ile KPI + histogramlar + korelasyon haritasını tek
figürde toplar, `write_html` ile kaydeder.

**Tech Stack:** plotly (Temel Katman planında çekirdek bağımlılık yapıldı),
pandas, numpy.

**Global Constraints (spec'ten aynen):**
1. Plotly çekirdek bağımlılıktır; başka yeni bağımlılık yok.
2. Yalnızca "Premium Program" adı; eski ad yasak.
3. (Statik metot kuralı bu planda geçerli değil — Plotly figürleri
   `plotly.graph_objects.Figure` döndürür.)
4. CLI kuralı sonraki planda.
5. Menü kuralı sonraki planda.
6. TDD zorunlu; dosya testleri `tmp_path` kullanır.
7. Sürüm bu planda değişmez.
8. `pytest` tam paket yeşil + `flake8` temiz olmadan görev bitmez.

## Dosya Yapısı

| Dosya | Sorumluluk |
|---|---|
| `agrista/viz/interactive.py` (yeni) | 6 `interactive_*` fonksiyonu + `build_dashboard` |
| `agrista/viz/__init__.py` | yeni export'lar |
| `tests/test_viz_interactive.py` (yeni) | plotly/dashboard testleri |

---

## Task 1: Etkileşimli grafik fonksiyonları (TDD)

**Files:** Test: `tests/test_viz_interactive.py` (Create) · Create: `agrista/viz/interactive.py` · Modify: `agrista/viz/__init__.py`
**Interfaces:** `interactive_scatter/line/bar/heatmap/box/histogram(df, ...) -> plotly Figure`.

- [ ] **RED** — `tests/test_viz_interactive.py` oluştur:

```python
"""Agrista etkileşimli (Plotly) grafik ve dashboard testleri."""
import numpy as np
import pandas as pd
import pytest
from plotly import graph_objects as go

from agrista.viz.interactive import (
    interactive_scatter, interactive_line, interactive_bar,
    interactive_heatmap, interactive_box, interactive_histogram,
    build_dashboard)


def _veri(seed=2, n=80):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "sulama": rng.uniform(3000, 12000, n),
        "gubre": rng.uniform(50, 400, n),
        "verim": rng.normal(500, 60, n),
        "bolge": rng.choice(["ic", "ege"], n),
    })


class TestEtkilesimliGrafikler:
    def test_scatter(self):
        fig = interactive_scatter(_veri(), x="sulama", y="verim",
                                  color="bolge")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_line(self):
        df = pd.DataFrame({"gun": range(10),
                           "deger": np.arange(10.0)})
        fig = interactive_line(df, x="gun", y="deger")
        assert isinstance(fig.data[0], go.Scatter)

    def test_bar(self):
        df = _veri().groupby("bolge", as_index=False)["verim"].mean()
        fig = interactive_bar(df, x="bolge", y="verim")
        assert isinstance(fig.data[0], go.Bar)

    def test_heatmap(self):
        fig = interactive_heatmap(_veri())
        assert isinstance(fig.data[0], go.Heatmap)

    def test_box(self):
        fig = interactive_box(_veri(), x="bolge", y="verim")
        assert isinstance(fig.data[0], go.Box)

    def test_histogram(self):
        fig = interactive_histogram(_veri(), column="verim")
        assert isinstance(fig.data[0], go.Histogram)

    def test_eksik_sutun_hatasi(self):
        with pytest.raises(ValueError):
            interactive_scatter(_veri(), x="yok", y="verim")
```

- [ ] Çalıştır: `.venv/bin/python -m pytest tests/test_viz_interactive.py -x -q`
      → ModuleNotFoundError.
- [ ] **GREEN** — `agrista/viz/interactive.py` oluştur:

```python
"""
Agrista Etkileşimli Görselleştirme — Plotly tabanlı grafikler ve dashboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RENKLER = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#44BBA4",
           "#3B1F2B"]


def _dogrula(df: pd.DataFrame, cols: list):
    eksik = [c for c in cols if c is not None and c not in df.columns]
    if eksik:
        raise ValueError(f"Sütun bulunamadı: {eksik}")


def interactive_scatter(df: pd.DataFrame, x: str, y: str,
                        color: str = None,
                        trendline: bool = False) -> go.Figure:
    """Etkileşimli saçılım grafiği."""
    _dogrula(df, [x, y, color])
    fig = px.scatter(df, x=x, y=y, color=color, trendline="ols"
                     if trendline else None, color_discrete_sequence=RENKLER,
                     template="plotly_white")
    fig.update_layout(title="Etkileşimli Saçılım Grafiği")
    return fig


def interactive_line(df: pd.DataFrame, x: str, y: str) -> go.Figure:
    """Etkileşimli çizgi grafiği."""
    _dogrula(df, [x, y])
    fig = px.line(df, x=x, y=y, color_discrete_sequence=RENKLER,
                  template="plotly_white")
    fig.update_layout(title="Etkileşimli Çizgi Grafiği")
    return fig


def interactive_bar(df: pd.DataFrame, x: str, y: str) -> go.Figure:
    """Etkileşimli çubuk grafiği."""
    _dogrula(df, [x, y])
    fig = px.bar(df, x=x, y=y, color_discrete_sequence=RENKLER,
                 template="plotly_white")
    fig.update_layout(title="Etkileşimli Çubuk Grafiği")
    return fig


def interactive_heatmap(df: pd.DataFrame,
                        columns: list = None) -> go.Figure:
    """Etkileşimli korelasyon ısı haritası."""
    sayisal = df[columns] if columns else df.select_dtypes(
        include=[np.number])
    if sayisal.shape[1] < 2:
        raise ValueError("Isı haritası için en az 2 sayısal sütun gerekli")
    corr = sayisal.corr()
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=list(corr.columns),
                                    y=list(corr.columns), colorscale="RdBu",
                                    zmin=-1, zmax=1))
    fig.update_layout(title="Korelasyon Isı Haritası",
                      template="plotly_white")
    return fig


def interactive_box(df: pd.DataFrame, x: str, y: str) -> go.Figure:
    """Etkileşimli kutu grafiği."""
    _dogrula(df, [x, y])
    fig = px.box(df, x=x, y=y, color_discrete_sequence=RENKLER,
                 template="plotly_white")
    fig.update_layout(title="Etkileşimli Kutu Grafiği")
    return fig


def interactive_histogram(df: pd.DataFrame, column: str) -> go.Figure:
    """Etkileşimli histogram."""
    _dogrula(df, [column])
    fig = px.histogram(df, x=column, color_discrete_sequence=RENKLER,
                       template="plotly_white")
    fig.update_layout(title=f"{column} Dağılımı")
    return fig
```

- [ ] `agrista/viz/__init__.py`'ye export ekle:

```python
from agrista.viz.interactive import (
    interactive_scatter, interactive_line, interactive_bar,
    interactive_heatmap, interactive_box, interactive_histogram)
```

  ve `__all__` listesine bu adları ekle.
- [ ] Çalıştır: `.venv/bin/python -m pytest tests/test_viz_interactive.py -q`
      → 7 passed (build_dashboard sonraki task'ta).
- [ ] Commit: `feat(viz): plotly etkileşimli grafikler (6 fonksiyon)`

## Task 2: `build_dashboard` (TDD)

**Files:** Modify: `tests/test_viz_interactive.py`, `agrista/viz/interactive.py`, `agrista/viz/__init__.py`
**Interfaces:** `build_dashboard(df, output_path, target=None, title=...) -> dict`.

- [ ] **RED** — test dosyası sonuna ekle:

```python
class TestDashboard:
    def test_dashboard_uretir(self, tmp_path):
        hedef = tmp_path / "panel.html"
        res = build_dashboard(_veri(), str(hedef))
        assert hedef.exists() and hedef.stat().st_size > 1000
        assert res["path"] == str(hedef)
        assert res["n_rows"] == 80
        assert res["n_figures"] >= 3

    def test_dashboard_hedefli(self, tmp_path):
        hedef = tmp_path / "panel2.html"
        res = build_dashboard(_veri(), str(hedef), target="verim")
        assert hedef.exists()
        assert res["n_figures"] >= 4

    def test_dashboard_bos_df_hatasi(self, tmp_path):
        with pytest.raises(ValueError):
            build_dashboard(pd.DataFrame(), str(tmp_path / "x.html"))
```

- [ ] Çalıştır: `-k Dashboard` → ImportError (build_dashboard yok).
- [ ] **GREEN** — `interactive.py` sonuna ekle:

```python
def build_dashboard(df: pd.DataFrame, output_path: str,
                    target: str = None,
                    title: str = "Agrista Keşif Paneli") -> dict:
    """Tek HTML keşif paneli: KPI kartları + histogramlar + korelasyon."""
    if df.empty:
        raise ValueError("Dashboard için boş olmayan veri gerekli")
    sayisal = df.select_dtypes(include=[np.number])
    sayisal_kolonlar = list(sayisal.columns)[:6]
    if not sayisal_kolonlar:
        raise ValueError("Dashboard için en az 1 sayısal sütun gerekli")

    n_satir, n_sutun = df.shape
    eksik_oran = float(df.isna().sum().sum() / max(n_satir * n_sutun, 1))

    n_hist = len(sayisal_kolonlar)
    fig = make_subplots(rows=2, cols=max(n_hist, 2),
                        specs=[[{"type": "xy"}] * max(n_hist, 2),
                               [{"type": "xy"}, {"type": "xy"}]
                               + [None] * (max(n_hist, 2) - 2)],
                        subplot_titles=tuple(sayisal_kolonlar)
                        + ("Korelasyon", "KPI Özet"),
                        vertical_spacing=0.18, horizontal_spacing=0.08)
    n_figures = 0
    for i, kol in enumerate(sayisal_kolonlar):
        fig.add_trace(go.Histogram(x=df[kol].dropna(), name=kol,
                                   marker_color=RENKLER[i % len(RENKLER)],
                                   showlegend=False),
                      row=1, col=i + 1)
        n_figures += 1

    if len(sayisal_kolonlar) >= 2:
        corr = sayisal[sayisal_kolonlar].corr()
        fig.add_trace(go.Heatmap(z=corr.values, x=sayisal_kolonlar,
                                 y=sayisal_kolonlar, colorscale="RdBu",
                                 zmin=-1, zmax=1, showscale=True),
                      row=2, col=1)
        n_figures += 1

    kpi_metin = (f"Satır: {n_satir}<br>Sütun: {n_sutun}<br>"
                 f"Eksik oranı: {eksik_oran:.2%}")
    fig.add_trace(go.Table(header=dict(values=["Özet"]),
                           cells=dict(values=[[kpi_metin]])),
                  row=2, col=2)
    n_figures += 1

    if target is not None:
        if target not in df.columns:
            raise ValueError(f"Hedef sütun bulunamadı: {target}")
        kategori = df.select_dtypes(exclude=[np.number]).columns
        if len(kategori) > 0:
            fig = interactive_box(df, x=str(kategori[0]), y=target)
            n_figures += 1

    fig.update_layout(title_text=title, height=760,
                      template="plotly_white")
    fig.write_html(output_path, include_plotlyjs="cdn")
    return {"path": output_path, "n_figures": int(n_figures),
            "n_rows": int(n_satir)}
```

- [ ] Çalıştır: `.venv/bin/python -m pytest tests/test_viz_interactive.py -q`
      → yeşil.
- [ ] `agrista/viz/__init__.py` export'larına `build_dashboard` eklenir.
- [ ] Tam doğrulama: `.venv/bin/python -m pytest tests/ -q` ve
      `.venv/bin/python -m flake8 agrista tests` → temiz.
- [ ] Commit: `feat(viz): build_dashboard — tek HTML keşif paneli`
