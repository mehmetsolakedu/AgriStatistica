# Plan: Görselleştirme — Temel Katman (yeniden yapılandırma + temalar + dışa aktarım)

**Spec:** `docs/superpowers/specs/2026-08-06-gorsellestirme-design.md` (§3, §4)

**Goal:** `agrista/viz` paketini alt dosyalara bölüp tema sistemi ve çok
formatlı dışa aktarım ile temel katmanı kurmak; mevcut 12 metot ve tüm
testler aynen korunur.

**Architecture:** `AgristaPlotter` gövdesi `viz/plotter.py`'ye taşınır,
`__init__.py` re-export eder; temalar `viz/themes.py`'de saf veri +
`apply_theme`; dışa aktarım plotter üzerinde genişletilir.

**Tech Stack:** matplotlib, seaborn, base64 (stdlib). Plotly bu planda
kullanılmaz ama `pyproject.toml`'a çekirdek bağımlılık olarak bu planda
eklenir (ileriki planlar için).

**Global Constraints (spec'ten aynen):**
1. Plotly çekirdek bağımlılık olur; başka yeni bağımlılık yok.
2. Yalnızca "Premium Program" adı; eski ad yasak.
3. Statik grafikler `AgristaPlotter` metodu, `Figure` döner, Türkçe
   docstring; mevcut 12 metodun imzası/davranışı DEĞİŞMEZ.
4. CLI kuralı bu planda geçerli değil (CLI sonraki plan).
5. Menü kuralı bu planda geçerli değil.
6. TDD zorunlu; grafik testleri `matplotlib.use("Agg")` altında; dosya
   testleri `tmp_path` kullanır.
7. Sürüm bu planda değişmez (0.3.0 son planda).
8. `pytest` tam paket yeşil + `flake8` temiz olmadan görev bitmez.

## Dosya Yapısı

| Dosya | Sorumluluk |
|---|---|
| `agrista/viz/plotter.py` (yeni) | `AgristaPlotter` sınıfı (taşınmış + genişletilmiş) |
| `agrista/viz/themes.py` (yeni) | `THEMES`, `apply_theme` |
| `agrista/viz/__init__.py` | re-export |
| `tests/test_viz_themes.py` (yeni) | tema + dışa aktarım testleri |
| `pyproject.toml` | plotly bağımlılığı |

---

## Task 1: Yeniden yapılandırma + plotly bağımlılığı

**Files:** Create: `agrista/viz/plotter.py` · Modify: `agrista/viz/__init__.py`, `pyproject.toml`
**Interfaces:** `from agrista.viz import AgristaPlotter` kesilmemeli.

- [ ] `agrista/viz/__init__.py` içeriği TAM olarak `agrista/viz/plotter.py`'ye
      kopyalanır; docstring ilk satırı korunur.
- [ ] `agrista/viz/__init__.py` şu içerikle değiştirilir:

```python
"""
Agrista Visualization Module — Görselleştirme
Plotting and visualization tools for agricultural data analysis.
"""

from agrista.viz.plotter import AgristaPlotter

__all__ = ["AgristaPlotter"]
```

- [ ] `pyproject.toml` `dependencies` listesine `"plotly>=5.18"` satırı
      eklenir (mevcut madde sırasının sonuna).
- [ ] Kurulum yenilenir: `pip install -e . -q` (venv: `.venv/bin/pip`).
- [ ] Doğrulama: `.venv/bin/python -m pytest tests/test_viz_cli.py tests/test_integration.py -q`
      → yeşil (davranış değişmedi).
- [ ] `flake8 agrista/viz/` temiz.
- [ ] Commit: `refactor(viz): plotter alt modülü + plotly çekirdek bağımlılık`

## Task 2: Temalar (TDD)

**Files:** Test: `tests/test_viz_themes.py` (Create) · Create: `agrista/viz/themes.py` · Modify: `agrista/viz/plotter.py`
**Interfaces:** `apply_theme(name: str) -> dict` ({style, rc, palette, dpi}) · `AgristaPlotter(theme=...)`.

- [ ] **RED** — `tests/test_viz_themes.py` oluştur:

```python
"""Agrista tema sistemi ve gelişmiş dışa aktarım testleri."""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from agrista.viz import AgristaPlotter
from agrista.viz.themes import THEMES, apply_theme


@pytest.fixture(autouse=True)
def _kapla():
    yield
    AgristaPlotter.close()


class TestTemalar:
    def test_dort_tema_tanimli(self):
        assert set(THEMES) == {"agrista", "yayin", "minimal", "karanlik"}

    def test_apply_theme_icerik(self):
        t = apply_theme("yayin")
        assert t["dpi"] == 300
        assert len(t["palette"]) >= 4
        assert isinstance(t["rc"], dict)

    def test_bilinmeyen_tema_hatasi(self):
        with pytest.raises(ValueError):
            apply_theme("olmayan")

    def test_plotter_tema_uygular(self):
        p = AgristaPlotter(theme="yayin")
        assert p.theme == "yayin"
        assert plt.rcParams["savefig.dpi"] == 300

    def test_plotter_varsayilan_tema(self):
        p = AgristaPlotter()
        assert p.theme == "agrista"

    def test_eski_style_parametresi_calismaya_devam_eder(self):
        p = AgristaPlotter(style="darkgrid")
        assert p.style == "darkgrid"
```

- [ ] Çalıştır: `.venv/bin/python -m pytest tests/test_viz_themes.py -x -q`
      → `ModuleNotFoundError: No module named 'agrista.viz.themes'`
- [ ] **GREEN** — `agrista/viz/themes.py` oluştur:

```python
"""
Agrista Tema Sistemi — bilimsel yayın kalitesinde grafik stilleri.
"""

THEMES = {
    "agrista": {
        "style": "whitegrid",
        "rc": {"axes.edgecolor": "#333333", "figure.dpi": 100,
               "savefig.dpi": 150},
        "palette": ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D",
                    "#3B1F2B", "#44BBA4"],
        "dpi": 150,
    },
    "yayin": {
        "style": "white",
        "rc": {"font.family": "serif", "axes.grid": False,
               "axes.spines.top": False, "axes.spines.right": False,
               "figure.dpi": 150, "savefig.dpi": 300},
        "palette": ["#1B1B1B", "#5A5A5A", "#8C8C8C", "#B0B0B0",
                    "#2E86AB", "#C73E1D"],
        "dpi": 300,
    },
    "minimal": {
        "style": "ticks",
        "rc": {"axes.grid": False, "figure.facecolor": "white",
               "savefig.dpi": 150},
        "palette": ["#2E86AB", "#44BBA4", "#F18F01", "#A23B72"],
        "dpi": 150,
    },
    "karanlik": {
        "style": "darkgrid",
        "rc": {"figure.facecolor": "#121212", "axes.facecolor": "#1E1E1E",
               "text.color": "#E0E0E0", "axes.labelcolor": "#E0E0E0",
               "xtick.color": "#B0B0B0", "ytick.color": "#B0B0B0",
               "savefig.dpi": 150},
        "palette": ["#4FC3F7", "#81C784", "#FFB74D", "#F06292"],
        "dpi": 150,
    },
}


def apply_theme(name: str) -> dict:
    """Tema adı doğrula ve içerik sözlüğünü döndür."""
    if name not in THEMES:
        raise ValueError(f"Bilinmeyen tema: {name}. "
                         f"Seçenekler: {sorted(THEMES)}")
    return THEMES[name]
```

- [ ] `agrista/viz/plotter.py` içinde `AgristaPlotter.__init__` şu şekilde
      güncellenir (eski davranış korunur):

```python
    def __init__(self, style: str = "whitegrid", theme: str = None):
        from agrista.viz.themes import apply_theme
        self.theme = theme if theme is not None else "agrista"
        t = apply_theme(self.theme)
        self._theme = t
        self._palette = list(t["palette"])
        self.style = t["style"] if theme is not None else style
        import seaborn as sns
        sns.set_style(self.style)
        plt.rcParams.update(t["rc"])
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rcParams["font.size"] = 12
```

  (Dosyanın üstündeki `import seaborn as sns` zaten vardır; metot içi
  import satırı çıkarılıp doğrudan üst düzey `sns` kullanılabilir —
  uygulayıcı ikisinden birini seçer, flake8 temiz kalmalı.)
- [ ] Çalıştır: `.venv/bin/python -m pytest tests/test_viz_themes.py tests/test_viz_cli.py -q` → yeşil.
- [ ] Commit: `feat(viz): tema sistemi — agrista, yayın, minimal, karanlık`

## Task 3: Geliştirilmiş dışa aktarım (TDD)

**Files:** Modify: `tests/test_viz_themes.py`, `agrista/viz/plotter.py`
**Interfaces:** `save(filename, fig=None, dpi=None)`, `save_multi(...)`, `export_html(fig, path, title)`.

- [ ] **RED** — `tests/test_viz_themes.py` sonuna ekle:

```python
class TestDisaAktarim:
    def test_save_png_ve_svg(self, tmp_path):
        p = AgristaPlotter()
        fig = p.histogram(np.random.default_rng(1).normal(0, 1, 50))
        for uzanti in ("png", "svg", "pdf"):
            hedef = tmp_path / f"g.{uzanti}"
            p.save(str(hedef), fig=fig)
            assert hedef.exists() and hedef.stat().st_size > 0

    def test_save_tema_dpi(self, tmp_path):
        p = AgristaPlotter(theme="yayin")
        fig = p.histogram(np.random.default_rng(2).normal(0, 1, 50))
        hedef = tmp_path / "y.png"
        p.save(str(hedef), fig=fig)
        assert hedef.exists()

    def test_save_multi(self, tmp_path):
        p = AgristaPlotter()
        f1 = p.histogram(np.arange(10, dtype=float))
        f2 = p.bar_chart(["a", "b"], [1.0, 2.0])
        yollar = p.save_multi(str(tmp_path / "cok"), [f1, f2],
                              fmts=("png", "svg"))
        assert len(yollar) == 4
        for yol in yollar:
            import pathlib
            assert pathlib.Path(yol).exists()

    def test_export_html(self, tmp_path):
        p = AgristaPlotter()
        fig = p.histogram(np.arange(10, dtype=float))
        hedef = tmp_path / "g.html"
        p.export_html(fig, str(hedef), title="Deneme Grafik")
        icerik = hedef.read_text(encoding="utf-8")
        assert "Deneme Grafik" in icerik
        assert "data:image/png;base64" in icerik
```

- [ ] Çalıştır: `-k DisaAktarim` → AttributeError (save_multi yok).
- [ ] **GREEN** — `agrista/viz/plotter.py` içinde `save` metodunu güncelle,
      `save_multi` ve `export_html` ekle:

```python
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
```

- [ ] Çalıştır: `.venv/bin/python -m pytest tests/test_viz_themes.py -q` → yeşil.
- [ ] Tam doğrulama: `.venv/bin/python -m pytest tests/ -q` ve
      `.venv/bin/python -m flake8 agrista tests` → temiz.
- [ ] Commit: `feat(viz): save_multi + export_html dışa aktarım`
