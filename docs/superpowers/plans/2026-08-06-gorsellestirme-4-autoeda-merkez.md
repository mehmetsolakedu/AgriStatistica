# Plan: Görselleştirme — Auto-EDA + Grafik Merkezi (CLI/menü) + Teslim Kapanışı

**Spec:** `docs/superpowers/specs/2026-08-06-gorsellestirme-design.md` (§7, §8, §10)

**Goal:** Değişken türüne göre grafik öneren auto-EDA motoru, 6 CLI
komutu + menü [21] Grafikler kategorisi ve sürüm 0.3.0 teslim kapanışı.

**Architecture:** `agrista/viz/auto_eda.py` öneri motoru + rapor üretimi;
CLI komutları `agrista/cli/__init__.py`; menüye 21. kategori eklenir.

**Tech Stack:** pandas, numpy, matplotlib (Agg), mevcut viz/interactive,
click. Ön koşul: Plan 1-3 tamamlanmış.

**Global Constraints (spec'ten aynen):**
1. Plotly çekirdek bağımlılıktır; başka yeni bağımlılık yok.
2. Yalnızca "Premium Program" adı; eski ad yasak.
3. Statik grafikler `AgristaPlotter` metotları üzerinden çağrılır.
4. CLI: click komutları; Türkçe seçenekler; `_load_file(path).dataframe`;
   Choice default'ları STR.
5. Menü handler'ları `_prompt_or_eof`/`_ask_file`/`_ask_column` kullanır;
   FLOWS formatı `(kategori başlığı, işlem başlığı, inputs_str, expect_str)`.
6. TDD zorunlu; dosya testleri `tmp_path` kullanır.
7. Sürüm: 0.2.0 → 0.3.0 (`pyproject.toml`, `__version__`, banner,
   `version_option`, `TestCli::test_version`).
8. `pytest` tam paket yeşil + `flake8` temiz olmadan görev bitmez.

## Dosya Yapısı

| Dosya | Sorumluluk |
|---|---|
| `agrista/viz/auto_eda.py` (yeni) | `infer_column_types`, `chart_suggestion`, `auto_eda` |
| `agrista/viz/__init__.py` | auto_eda export'ları |
| `agrista/cli/__init__.py` | 6 komut + menü [21] + handler'lar |
| `tests/test_viz_autoeda.py` (yeni) | öneri motoru testleri |
| `tests/test_viz_cli.py` | CLI testleri (ek) |
| `menu_smoke_test.py` | FLOWS akışları |
| `README.md`, `pyproject.toml`, `docs/02` | teslim güncellemeleri |

---

## Task 1: Auto-EDA motoru (TDD)

**Files:** Test: `tests/test_viz_autoeda.py` (Create) · Create: `agrista/viz/auto_eda.py` · Modify: `agrista/viz/__init__.py`
**Interfaces:** `infer_column_types(df) -> dict`, `chart_suggestion(df) -> list[dict]`, `auto_eda(df, output_dir) -> dict`.

- [ ] **RED** — `tests/test_viz_autoeda.py` oluştur:

```python
"""Agrista auto-EDA (akıllı grafik önerisi) testleri."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from agrista.viz import AgristaPlotter
from agrista.viz.auto_eda import (auto_eda, chart_suggestion,
                                  infer_column_types)


@pytest.fixture(autouse=True)
def _kapla():
    yield
    AgristaPlotter.close()


def _veri(seed=3, n=60):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "verim": rng.normal(500, 50, n),
        "sulama": rng.uniform(3000, 10000, n),
        "gubre": rng.uniform(50, 300, n),
        "bolge": rng.choice(["ic", "ege", "marmara"], n),
    })


class TestSutunTurleri:
    def test_tur_cikarimi(self):
        turler = infer_column_types(_veri())
        assert turler["verim"] == "sayisal"
        assert turler["bolge"] == "kategorik"

    def test_tarih_turu(self):
        df = pd.DataFrame({"tarih": pd.date_range("2025-01-01", periods=5),
                           "deger": [1.0, 2, 3, 4, 5]})
        assert infer_column_types(df)["tarih"] == "tarih"


class TestOneriler:
    def test_sayisal_icin_histogram_ve_qq(self):
        turler = {s["type"] for s in
                  chart_suggestion(_veri()[["verim"]])}
        assert "histogram" in turler and "qq_plot" in turler

    def test_iki_sayisal_scatter(self):
        turler = {s["type"] for s in
                  chart_suggestion(_veri()[["verim", "sulama"]])}
        assert "scatter" in turler and "hexbin" in turler

    def test_sayisal_kategorik_violin(self):
        turler = {s["type"] for s in
                  chart_suggestion(_veri()[["verim", "bolge"]])}
        assert "violin_plot" in turler and "grouped_boxplot" not in turler

    def test_uc_sayisal_korelasyon_ve_pair(self):
        turler = {s["type"] for s in chart_suggestion(_veri()[
            ["verim", "sulama", "gubre"]])}
        assert "correlation_heatmap" in turler and "pair_grid" in turler

    def test_tarih_cizgi(self):
        df = pd.DataFrame({"tarih": pd.date_range("2025-01-01", periods=8),
                           "deger": np.arange(8.0)})
        turler = {s["type"] for s in chart_suggestion(df)}
        assert "line_chart" in turler

    def test_kategorik_bar(self):
        turler = {s["type"] for s in
                  chart_suggestion(_veri()[["bolge"]])}
        assert "bar_chart" in turler


class TestAutoEdaRapor:
    def test_rapor_uretir(self, tmp_path):
        res = auto_eda(_veri(), str(tmp_path / "eda"))
        import pathlib
        assert pathlib.Path(res["html_path"]).exists()
        assert len(res["figures"]) >= 4
        assert len(res["suggestions"]) >= 4

    def test_bos_df_hatasi(self, tmp_path):
        with pytest.raises(ValueError):
            auto_eda(pd.DataFrame(), str(tmp_path / "x"))
```

- [ ] Çalıştır: `.venv/bin/python -m pytest tests/test_viz_autoeda.py -x -q`
      → ModuleNotFoundError.
- [ ] **GREEN** — `agrista/viz/auto_eda.py` oluştur:

```python
"""
Agrista Auto-EDA — değişken türüne göre akıllı grafik önerisi ve rapor.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from agrista.viz.plotter import AgristaPlotter


def infer_column_types(df: pd.DataFrame) -> dict:
    """Her sütun için tür çıkarımı: sayisal/kategorik/tarih/metin."""
    turler = {}
    for kol in df.columns:
        s = df[kol]
        if pd.api.types.is_datetime64_any_dtype(s):
            turler[kol] = "tarih"
        elif pd.api.types.is_numeric_dtype(s):
            turler[kol] = "sayisal"
        elif s.nunique(dropna=True) <= max(2, min(20, len(s) // 10)):
            turler[kol] = "kategorik"
        else:
            turler[kol] = "metin"
    return turler


def chart_suggestion(df: pd.DataFrame) -> list:
    """Deterministik grafik öneri listesi (tür kurallarına göre)."""
    if df.empty:
        raise ValueError("Öneri için boş olmayan veri gerekli")
    turler = infer_column_types(df)
    sayisal = [c for c, t in turler.items() if t == "sayisal"]
    kategorik = [c for c, t in turler.items() if t == "kategorik"]
    tarih = [c for c, t in turler.items() if t == "tarih"]

    oneriler = []

    def ekle(tip, kolonlar, gerekce):
        oneriler.append({"type": tip, "columns": list(kolonlar),
                         "reason": gerekce})

    for c in sayisal[:4]:
        ekle("histogram", [c], "Tek sayısal değişken dağılımı")
        ekle("qq_plot", [c], "Normallik değerlendirmesi")
    for i in range(min(len(sayisal), 3)):
        for j in range(i + 1, min(len(sayisal), 3)):
            ekle("scatter", [sayisal[i], sayisal[j]], "İki sayısal ilişki")
            ekle("hexbin", [sayisal[i], sayisal[j]], "Yoğunluk görünümü")
    for k in kategorik[:2]:
        for c in sayisal[:3]:
            ekle("violin_plot", [k, c], "Sayısal × kategorik dağılım")
    if tarih and sayisal:
        ekle("line_chart", [tarih[0], sayisal[0]], "Zaman serisi")
    if len(sayisal) >= 3:
        ekle("correlation_heatmap", sayisal[:6], "Korelasyon matrisi")
        ekle("pair_grid", sayisal[:4], "Çok değişkenli matris")
    for k in kategorik[:2]:
        ekle("bar_chart", [k], "Kategori frekansları")
    return oneriler


def _ciz(p: AgristaPlotter, df: pd.DataFrame, oneri: dict, yol: str):
    """Bir öneriyi çizip kaydeder."""
    tip, kol = oneri["type"], oneri["columns"]
    if tip == "histogram":
        fig = p.histogram(df[kol[0]].dropna(), title=f"{kol[0]} Dağılımı")
    elif tip == "qq_plot":
        fig = p.qq_plot(df[kol[0]].dropna(), title=f"{kol[0]} Q-Q")
    elif tip == "scatter":
        fig = p.scatter(df[kol[0]], df[kol[1]], title=f"{kol[0]} × {kol[1]}",
                        xlabel=kol[0], ylabel=kol[1])
    elif tip == "hexbin":
        fig = p.hexbin_plot(df[kol[0]], df[kol[1]],
                            title=f"{kol[0]} × {kol[1]} Yoğunluk")
    elif tip == "violin_plot":
        fig = p.violin_plot(df, x_col=kol[0], y_col=kol[1],
                            title=f"{kol[1]} ~ {kol[0]}")
    elif tip == "line_chart":
        fig = p.line_chart(list(df[kol[0]]), list(df[kol[1]]),
                           title="Zaman Serisi", xlabel=kol[0], ylabel=kol[1])
    elif tip == "correlation_heatmap":
        fig = p.correlation_heatmap(df[kol])
    elif tip == "pair_grid":
        fig = p.pair_grid(df, cols=kol)
    elif tip == "bar_chart":
        say = df[kol[0]].value_counts()
        fig = p.bar_chart([str(x) for x in say.index], list(say.values),
                          title=f"{kol[0]} Frekansları", xlabel=kol[0])
    else:
        return None
    p.save(yol, fig=fig)
    AgristaPlotter.close()
    return yol


def auto_eda(df: pd.DataFrame, output_dir: str) -> dict:
    """Tam keşif raporu: öneriler + PNG'ler + report.html."""
    if df.empty:
        raise ValueError("Auto-EDA için boş olmayan veri gerekli")
    os.makedirs(output_dir, exist_ok=True)
    oneriler = chart_suggestion(df)
    p = AgristaPlotter()
    yollar = []
    for i, oneri in enumerate(oneriler):
        yol = os.path.join(output_dir, f"{i + 1:02d}_{oneri['type']}.png")
        if _ciz(p, df, oneri, yol):
            yollar.append(yol)
    html = ["<!DOCTYPE html><html lang=\"tr\"><head>"
            "<meta charset=\"utf-8\"><title>Agrista Auto-EDA</title>"
            "<style>body{font-family:sans-serif;background:#f4f6f8;"
            "margin:24px}img{max-width:900px;background:#fff;border:1px "
            "solid #dfe3e8;border-radius:8px;margin:8px 0;padding:8px}"
            "h1{color:#2E86AB}</style></head><body>",
            "<h1>Agrista Otomatik Keşif Raporu</h1>",
            f"<p>{len(df)} satır, {len(df.columns)} sütun, "
            f"{len(oneriler)} öneri.</p>"]
    for yol in yollar:
        html.append(f"<img src=\"{os.path.basename(yol)}\"/>")
    html.append("</body></html>")
    rapor_yolu = os.path.join(output_dir, "report.html")
    with open(rapor_yolu, "w", encoding="utf-8") as fh:
        fh.write("".join(html))
    return {"html_path": rapor_yolu, "figures": yollar,
            "suggestions": oneriler}
```

- [ ] `agrista/viz/__init__.py`'ye ekle:

```python
from agrista.viz.auto_eda import (auto_eda, chart_suggestion,
                                  infer_column_types)
```

  ve `__all__` güncellenir.
- [ ] Çalıştır: `.venv/bin/python -m pytest tests/test_viz_autoeda.py -q` → yeşil.
- [ ] Commit: `feat(viz): auto-EDA öneri motoru + keşif raporu`

## Task 2: CLI komutları (TDD)

**Files:** Test: `tests/test_viz_cli.py` (Modify) · Modify: `agrista/cli/__init__.py`
**Interfaces:** `plot`, `plot-forest`, `plot-roc`, `plot-survival`, `dashboard`, `autoeda`.

- [ ] **RED** — `tests/test_viz_cli.py` sonuna ekle (sınıf idiyomu:
      `class TestGrafikMerkeziCli` + `runner` fixture + csv fixture):

```python
class TestGrafikMerkeziCli:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def grafik_csv(self, tmp_path):
        rng = np.random.default_rng(21)
        df = pd.DataFrame({
            "grup": rng.choice(["A", "B"], 40),
            "x": rng.normal(0, 1, 40),
            "y": rng.normal(5, 1, 40),
            "skor": rng.uniform(0, 1, 40),
        })
        df["gercek"] = (df["skor"] > 0.5).astype(int)
        csv = tmp_path / "grafik.csv"
        df.to_csv(csv, index=False)
        return str(csv)

    def test_plot_histogram(self, runner, grafik_csv, tmp_path):
        cikti = str(tmp_path / "h.png")
        result = runner.invoke(cli_main, ["plot", grafik_csv,
                                          "--tip", "histogram",
                                          "--x", "y", "--cikti", cikti])
        assert result.exit_code == 0, result.output
        assert (tmp_path / "h.png").exists()

    def test_plot_scatter(self, runner, grafik_csv, tmp_path):
        cikti = str(tmp_path / "s.png")
        result = runner.invoke(cli_main, ["plot", grafik_csv,
                                          "--tip", "scatter",
                                          "--x", "x", "--y", "y",
                                          "--cikti", cikti])
        assert result.exit_code == 0, result.output

    def test_plot_hatali_tip(self, runner, grafik_csv, tmp_path):
        result = runner.invoke(cli_main, ["plot", grafik_csv,
                                          "--tip", "olmayan",
                                          "--x", "y",
                                          "--cikti", str(tmp_path / "x.png")])
        assert result.exit_code != 0

    def test_plot_roc(self, runner, grafik_csv, tmp_path):
        cikti = str(tmp_path / "roc.png")
        result = runner.invoke(cli_main, ["plot-roc", grafik_csv,
                                          "--gercek", "gercek",
                                          "--skor", "skor",
                                          "--cikti", cikti])
        assert result.exit_code == 0, result.output

    def test_plot_forest(self, runner, tmp_path):
        cikti = str(tmp_path / "f.png")
        df = pd.DataFrame({"etki": [0.5, -0.2], "alt": [0.1, -0.6],
                           "ust": [0.9, 0.2], "ad": ["a", "b"]})
        csv = tmp_path / "forest.csv"
        df.to_csv(csv, index=False)
        result = runner.invoke(cli_main, ["plot-forest", str(csv),
                                          "--etki", "etki", "--alt", "alt",
                                          "--ust", "ust", "--etiket", "ad",
                                          "--cikti", cikti])
        assert result.exit_code == 0, result.output

    def test_plot_survival(self, runner, tmp_path):
        rng = np.random.default_rng(22)
        df = pd.DataFrame({"zaman": rng.exponential(10, 40),
                           "olay": rng.integers(0, 2, 40)})
        csv = tmp_path / "surv.csv"
        df.to_csv(csv, index=False)
        result = runner.invoke(cli_main, ["plot-survival", str(csv),
                                          "--zaman", "zaman",
                                          "--olay", "olay",
                                          "--cikti", str(tmp_path / "k.png")])
        assert result.exit_code == 0, result.output

    def test_dashboard_komutu(self, runner, grafik_csv, tmp_path):
        cikti = str(tmp_path / "panel.html")
        result = runner.invoke(cli_main, ["dashboard", grafik_csv,
                                          "--cikti", cikti])
        assert result.exit_code == 0, result.output
        assert (tmp_path / "panel.html").exists()

    def test_autoeda_komutu(self, runner, grafik_csv, tmp_path):
        dizin = str(tmp_path / "eda")
        result = runner.invoke(cli_main, ["autoeda", grafik_csv,
                                          "--cikti-dir", dizin])
        assert result.exit_code == 0, result.output
        assert (tmp_path / "eda" / "report.html").exists()
```

- [ ] Çalıştır: `-k GrafikMerkezi` → `No such command 'plot'`.
- [ ] **GREEN** — `agrista/cli/__init__.py` içine ekle (komutlar mevcut
      blok düzenine):

```python
@main.command("plot")
@click.argument("filepath")
@click.option("--tip", default="histogram",
              type=click.Choice(["histogram", "scatter", "box", "violin",
                                 "bar", "heat", "qq", "cizgi", "errorbar"]))
@click.option("--x", "x_col", default=None, help="X/yatay sütun")
@click.option("--y", "y_col", default=None, help="Y/dikey sütun")
@click.option("--grup", default=None, help="Grup sütunu")
@click.option("--tema", default="agrista",
              type=click.Choice(["agrista", "yayin", "minimal", "karanlik"]))
@click.option("--cikti", required=True, help="Kaydedilecek dosya yolu")
def plot(filepath: str, tip: str, x_col: str, y_col: str, grup: str,
         tema: str, cikti: str):
    """Hızlı grafik üretimi (premium grafik kütüphanesi)."""
    from agrista.viz import AgristaPlotter
    df = _load_file(filepath).dataframe
    p = AgristaPlotter(theme=tema)
    try:
        if tip == "histogram" and x_col:
            fig = p.histogram(df[x_col].dropna(), title=f"{x_col} Dağılımı")
        elif tip == "scatter" and x_col and y_col:
            fig = p.scatter(df[x_col], df[y_col], xlabel=x_col, ylabel=y_col)
        elif tip == "box" and y_col:
            fig = p.boxplot(df, x_col=grup, y_col=y_col)
        elif tip == "violin" and grup and y_col:
            fig = p.violin_plot(df, x_col=grup, y_col=y_col)
        elif tip == "bar" and x_col and y_col:
            say = df.groupby(x_col, observed=True)[y_col].mean()
            fig = p.bar_chart([str(v) for v in say.index], list(say.values),
                              xlabel=x_col, ylabel=f"Ortalama {y_col}")
        elif tip == "heat":
            fig = p.correlation_heatmap(df)
        elif tip == "qq" and x_col:
            fig = p.qq_plot(df[x_col].dropna())
        elif tip == "cizgi" and x_col and y_col:
            fig = p.line_chart(list(df[x_col]), list(df[y_col]),
                               xlabel=x_col, ylabel=y_col)
        elif tip == "errorbar" and grup and y_col:
            fig = p.error_bar(df, x_col=grup, y_col=y_col)
        else:
            raise click.UsageError(
                "Bu grafik tipi için gerekli sütunlar verilmedi")
    except ValueError as e:
        raise click.ClickException(str(e))
    p.save(cikti, fig=fig)
    AgristaPlotter.close()


@main.command("plot-forest")
@click.argument("filepath")
@click.option("--etki", required=True)
@click.option("--alt", required=True)
@click.option("--ust", required=True)
@click.option("--etiket", required=True)
@click.option("--tema", default="agrista",
              type=click.Choice(["agrista", "yayin", "minimal", "karanlik"]))
@click.option("--cikti", required=True)
def plot_forest(filepath: str, etki: str, alt: str, ust: str, etiket: str,
                tema: str, cikti: str):
    """Orman (forest) grafiği."""
    from agrista.viz import AgristaPlotter
    df = _load_file(filepath).dataframe
    p = AgristaPlotter(theme=tema)
    try:
        fig = p.forest_plot(list(df[etki]), list(df[alt]), list(df[ust]),
                            labels=[str(x) for x in df[etiket]])
    except ValueError as e:
        raise click.ClickException(str(e))
    p.save(cikti, fig=fig)
    AgristaPlotter.close()


@main.command("plot-roc")
@click.argument("filepath")
@click.option("--gercek", required=True)
@click.option("--skor", required=True)
@click.option("--tema", default="agrista",
              type=click.Choice(["agrista", "yayin", "minimal", "karanlik"]))
@click.option("--cikti", required=True)
def plot_roc(filepath: str, gercek: str, skor: str, tema: str, cikti: str):
    """ROC eğrisi grafiği."""
    from agrista.viz import AgristaPlotter
    df = _load_file(filepath).dataframe
    p = AgristaPlotter(theme=tema)
    try:
        fig = p.roc_plot(df[gercek], df[skor])
    except ValueError as e:
        raise click.ClickException(str(e))
    p.save(cikti, fig=fig)
    AgristaPlotter.close()


@main.command("plot-survival")
@click.argument("filepath")
@click.option("--zaman", required=True)
@click.option("--olay", required=True)
@click.option("--grup", default=None)
@click.option("--tema", default="agrista",
              type=click.Choice(["agrista", "yayin", "minimal", "karanlik"]))
@click.option("--cikti", required=True)
def plot_survival(filepath: str, zaman: str, olay: str, grup: str,
                  tema: str, cikti: str):
    """Kaplan-Meier sağkalım grafiği."""
    from agrista.viz import AgristaPlotter
    df = _load_file(filepath).dataframe
    p = AgristaPlotter(theme=tema)
    try:
        fig = p.survival_plot(df[zaman], df[olay],
                              group=df[grup] if grup else None)
    except ValueError as e:
        raise click.ClickException(str(e))
    p.save(cikti, fig=fig)
    AgristaPlotter.close()


@main.command("dashboard")
@click.argument("filepath")
@click.option("--hedef", default=None, help="Hedef değişken sütunu")
@click.option("--baslik", default="Agrista Keşif Paneli")
@click.option("--cikti", default="dashboard.html")
def dashboard(filepath: str, hedef: str, baslik: str, cikti: str):
    """Etkileşimli keşif paneli (tek HTML)."""
    from agrista.viz.interactive import build_dashboard
    df = _load_file(filepath).dataframe
    try:
        res = build_dashboard(df, cikti, target=hedef, title=baslik)
    except ValueError as e:
        raise click.ClickException(str(e))
    click.echo(f"Panel oluşturuldu: {res['path']} "
               f"({res['n_figures']} grafik)")


@main.command("autoeda")
@click.argument("filepath")
@click.option("--cikti-dir", default="agrista_eda")
def autoeda(filepath: str, cikti_dir: str):
    """Otomatik keşif raporu (öneri + grafikler + HTML)."""
    from agrista.viz.auto_eda import auto_eda
    df = _load_file(filepath).dataframe
    try:
        res = auto_eda(df, cikti_dir)
    except ValueError as e:
        raise click.ClickException(str(e))
    click.echo(f"Rapor: {res['html_path']} "
               f"({len(res['figures'])} grafik üretildi)")
```

- [ ] Çalıştır: `.venv/bin/python -m pytest tests/test_viz_cli.py -k GrafikMerkezi -q` → yeşil.
- [ ] Commit: `feat(cli): grafik merkezi komutları — plot, plot-forest/roc/survival, dashboard, autoeda`

## Task 3: Menü [21] Grafikler + smoke + docs/02 (TDD)

**Files:** Modify: `menu_smoke_test.py`, `agrista/cli/__init__.py`,
`tests/test_menu_flows.py` (docstring sayısı), `docs/02_PREMIUM_PROGRAM_MENU_YAPISI_LOG.md`

- [ ] **RED** — `menu_smoke_test.py::FLOWS`'a 4 akış ekle (kategori
      başlığı `"🎨 Grafikler"`): "Hızlı grafik (plot)" (csv + histogram
      girdileri, expect `"kaydedildi"`), "Etkileşimli dashboard"
      (expect `"Panel"`), "Otomatik keşif (Auto-EDA)" (expect `"Rapor"`),
      "Model grafikleri (ROC, sağkalım, orman, büyüme eğrisi)" (ROC
      girdileri, expect `"kaydedildi"`). Mevcut akışların veri dosyası
      hazırlama desenini birebir izle.
- [ ] Çalıştır: `.venv/bin/python -m pytest tests/test_menu_flows.py -k Grafikler -q`
      → başarısız (kategori yok).
- [ ] **GREEN** — `_build_menu_structure()` SONUNA ekle:

```python
("🎨 Grafikler", [
    ("Hızlı grafik (plot)", _menu_plot),
    ("Dağılım grafikleri (violin/ridge/raincloud)", _menu_dagilim),
    ("Tanı grafikleri (Q-Q, artık, Bland-Altman)", _menu_tani),
    ("Model grafikleri (ROC, sağkalım, orman, büyüme eğrisi)", _menu_model),
    ("Etkileşimli dashboard", _menu_dashboard),
    ("Otomatik keşif (Auto-EDA)", _menu_autoeda),
]),
```

  Handler'lar (her biri `_ask_file` → `_load_file(path).dataframe` →
  `_ask_column`/`_prompt_or_eof` → çizim → `p.save` + `AgristaPlotter.close()`;
  tema `_prompt_or_eof("Tema", default="agrista")` ile sorulur):

```python
def _menu_plot():
    path = _ask_file("Veri dosyası yolu")
    df = _load_file(path).dataframe
    x_col = _ask_column(df, "Grafiğin X/değişken sütunu")
    tema = _prompt_or_eof("Tema (agrista/yayin/minimal/karanlik)",
                          default="agrista")
    cikti = _prompt_or_eof("Kayıt yolu", default="agrista_grafik.png")
    p = AgristaPlotter(theme=tema)
    fig = p.histogram(df[x_col].dropna(), title=f"{x_col} Dağılımı")
    p.save(cikti, fig=fig)
    AgristaPlotter.close()


def _menu_dagilim():
    path = _ask_file("Veri dosyası yolu")
    df = _load_file(path).dataframe
    y_col = _ask_column(df, "Sayısal değişken")
    grup = _ask_column(df, "Grup sütunu")
    cikti = _prompt_or_eof("Kayıt yolu", default="violin.png")
    p = AgristaPlotter()
    fig = p.violin_plot(df, x_col=grup, y_col=y_col)
    p.save(cikti, fig=fig)
    AgristaPlotter.close()


def _menu_tani():
    path = _ask_file("Veri dosyası yolu")
    df = _load_file(path).dataframe
    x_col = _ask_column(df, "Değişken")
    cikti = _prompt_or_eof("Kayıt yolu", default="qq.png")
    p = AgristaPlotter()
    fig = p.qq_plot(df[x_col].dropna())
    p.save(cikti, fig=fig)
    AgristaPlotter.close()


def _menu_model():
    path = _ask_file("Veri dosyası yolu")
    df = _load_file(path).dataframe
    gercek = _ask_column(df, "Gerçek sınıf (0/1)")
    skor = _ask_column(df, "Skor sütunu")
    cikti = _prompt_or_eof("Kayıt yolu", default="roc.png")
    p = AgristaPlotter()
    fig = p.roc_plot(df[gercek], df[skor])
    p.save(cikti, fig=fig)
    AgristaPlotter.close()


def _menu_dashboard():
    path = _ask_file("Veri dosyası yolu")
    df = _load_file(path).dataframe
    cikti = _prompt_or_eof("Kayıt yolu", default="dashboard.html")
    res = build_dashboard(df, cikti)
    click.echo(f"Panel oluşturuldu: {res['path']}")


def _menu_autoeda():
    path = _ask_file("Veri dosyası yolu")
    df = _load_file(path).dataframe
    dizin = _prompt_or_eof("Çıktı dizini", default="agrista_eda")
    res = auto_eda(df, dizin)
    click.echo(f"Rapor: {res['html_path']}")
```

  Dosya başına gerekli import'lar eklenir: `from agrista.viz import
  AgristaPlotter`, `from agrista.viz.interactive import build_dashboard`,
  `from agrista.viz.auto_eda import auto_eda`.
- [ ] `tests/test_menu_flows.py` docstring'indeki akış sayısı güncellenir
      (61 + eklenen kadar).
- [ ] Çalıştır: `.venv/bin/python -m pytest tests/test_menu_flows.py -q` → yeşil.
- [ ] `docs/02` log'una "## Güncelleme 6 — Görselleştirme Hamlesi" bölümü
      eklenir: 16 yeni grafik, temalar, dışa aktarım, plotly dashboard,
      auto-EDA, 6 CLI komutu, menü [21], test dosyaları.
- [ ] Commit: `feat(menu): [21] Grafikler menüsü + docs/02 Güncelleme 6`

## Task 4: Teslim kapanışı — README, sürüm 0.3.0, son doğrulama

**Files:** Modify: `README.md`, `pyproject.toml`, `agrista/__init__.py`
(`__version__`), `agrista/cli/__init__.py` (banner, `version_option`),
`tests/test_viz_cli.py` (`TestCli::test_version` 0.3.0 bekler)

- [ ] README güncellemeleri: özellik listesine görselleştirme maddeleri
      (16 grafik tipi, temalar, dashboard, auto-EDA); CLI örnek bloğuna
      `agrista plot veri.csv --tip scatter --x sulama --y verim --cikti s.png`,
      `agrista dashboard veri.csv`, `agrista autoeda veri.csv` satırları;
      menü paragrafı "20 kategori" → "21 kategori" + **[21] Grafikler**.
- [ ] Sürüm: `pyproject.toml` 0.3.0; `agrista/__init__.py` `__version__`
      0.3.0; banner "v0.3.0"; `version_option(version="0.3.0")`;
      `TestCli::test_version` beklentisi 0.3.0.
- [ ] Son doğrulama (taze, tam): `.venv/bin/python -m pytest tests/ -q`
      ve `.venv/bin/python -m flake8 agrista tests menu_smoke_test.py`.
- [ ] Commit: `chore: v0.3.0 — görselleştirme hamlesi teslim güncellemeleri`
