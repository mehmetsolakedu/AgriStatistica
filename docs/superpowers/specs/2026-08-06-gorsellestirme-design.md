# Tasarım: Agrista Görselleştirme Hamlesi (Premium Grafik Kütüphanesi + Grafik Merkezi + Etkileşimli Dashboard + Auto-EDA)

**Tarih:** 2026-08-06
**Durum:** Onaylandı (Aşama 1 — Brainstorming; Yaklaşım A, 4 özellik ailesi, Plotly çekirdek bağımlılık)
**Temel dal:** `feat/premium-kalan-bosluklar` (v0.2.0) üzerine yeni özellik dalı

## 1. Amaç

Agrista'yı tarımsal istatistik alanında en kapsamlı grafik üretim aracına
dönüştürmek: bilimsel yayın kalitesinde 16 yeni istatistiksel grafik tipi,
tema sistemi, çok formatlı dışa aktarım, Plotly tabanlı etkileşimli
dashboard, değişken türüne göre akıllı grafik önerisi (auto-EDA) ve tümünü
toplayan menü/CLI grafik merkezi.

## 2. Global Kısıtlar

1. **Bağımlılıklar:** Plotly bu hamle ile ÇEKİRDEK bağımlılık olur
   (`pyproject.toml` dependencies). Başka yeni bağımlılık eklenmez
   (matplotlib, seaborn, pandas, numpy, scipy zaten var).
2. **Adlandırma:** Her yerde yalnızca "Premium Program" adı; referans
   paketinin eski adı hiçbir dosyada yer alamaz.
3. **Arayüz tutarlılığı:** Statik grafikler `AgristaPlotter` sınıfının
   metotları olarak eklenir; her metot `matplotlib.figure.Figure` döndürür;
   Türkçe docstring; mevcut 12 metodun imzası ve davranışı DEĞİŞMEZ.
4. **CLI:** click komutları `agrista/cli/__init__.py` içinde; Türkçe
   seçenekler; veri yükleme `_load_file(path).dataframe`; Choice
   default'ları STR.
5. **Menü:** handler'lar `_prompt_or_eof` / `_ask_file` / `_ask_column`
   kullanır; smoke testler `menu_smoke_test.py::FLOWS` formatında
   `(kategori başlığı, işlem başlığı, inputs_str, expect_str)`.
6. **Test:** TDD zorunlu; grafik testleri `matplotlib.use("Agg")` altında,
   `Figure` döndüğünü + temel içerik doğrular (axes sayısı, başlık,
   etiketler); dosya kaydeden testler `tmp_path` kullanır.
7. **Sürüm:** 0.2.0 → 0.3.0 (`pyproject.toml`, `__version__`, banner,
   `version_option`).
8. **Kalite kapısı:** `pytest` tam paket yeşil + `flake8` temiz olmadan
   teslim yok.

## 3. Mimari ve Dosya Yerleşimi

`agrista/viz` paketi alt dosyalara bölünür; `__init__.py` yeniden
export eder (mevcut `from agrista.viz import AgristaPlotter` kesilmez):

| Dosya | Sorumluluk |
|---|---|
| `agrista/viz/themes.py` (yeni) | `THEMES` sözlüğü (4 tema), `apply_theme(ad) -> dict` |
| `agrista/viz/plotter.py` (yeni) | `AgristaPlotter` sınıfı: mevcut 12 metot + 16 yeni metot + geliştirilmiş `save` |
| `agrista/viz/interactive.py` (yeni) | Plotly etkileşimli grafikler + `build_dashboard` |
| `agrista/viz/auto_eda.py` (yeni) | `infer_column_types`, `chart_suggestion`, `auto_eda` |
| `agrista/viz/__init__.py` | Mevcut sınıf gövdesi `plotter.py`'ye taşınır; re-export |
| `agrista/cli/__init__.py` | 6 yeni komut + menü [21] |

Test dosyaları: `tests/test_viz_premium.py` (statik),
`tests/test_viz_interactive.py` (plotly/dashboard),
`tests/test_viz_autoeda.py` (öneri motoru), `tests/test_viz_cli.py`
(CLI, mevcut dosyaya ek), `tests/test_menu_flows.py` (otomatik).

## 4. Alt Proje 1 — Temel Katman (yeniden yapılandırma + temalar + dışa aktarım)

### 4.1 Yeniden yapılandırma
`AgristaPlotter` gövdesi birebir `agrista/viz/plotter.py`'ye taşınır;
`__init__.py` şu hale gelir: docstring + `from agrista.viz.plotter import
AgristaPlotter`. Tüm mevcut testler değişmeden geçmelidir.

### 4.2 Temalar (`themes.py`)

```python
THEMES = {
    "agrista":  {...},   # varsayılan mavi-yeşil palet, whitegrid
    "yayin":    {...},   # dergi kalitesi: serif font, 300 dpi, gri ızgara yok
    "minimal":  {...},   # ızgarasız, beyaz arka plan
    "karanlik": {...},   # koyu arka plan, açık metin
}
def apply_theme(name: str) -> dict: ...   # bilinmeyen ad → ValueError
```
Her tema: `rc` (rcParams güncellemeleri), `palette` (renk listesi),
`style` (seaborn stili). `AgristaPlotter(theme="agrista")` yeni ctor
parametresi; eski `style` parametresi geriye uyumlu kalır
(`theme=None, style="whitegrid"` — theme verilirse style'ı ezer).

### 4.3 Geliştirilmiş dışa aktarım
`save(filename, fig=None, dpi=None)`: tema dpi'ı varsayılan;
`save_multi(filename_base, figs, fmts=("png","svg"))`.
HTML dışa aktarım `export_html(fig, path, title)`: PNG'yi base64 gömerek
Plotly-stil şablonla tek HTML üretir (plotly şablonu ile aynı görünüm).

## 5. Alt Proje 2 — Premium Grafik Kütüphanesi (16 yeni metot)

Hepsi `AgristaPlotter` metodu; her biri `Figure` döndürür, girdi
doğrulaması `ValueError` fırlatır:

| # | Metot | Girdi (özet) | İçerik |
|---|---|---|---|
| 1 | `violin_plot(data, x_col, y_col, hue=None)` | DataFrame | sns.violinplot + iç kutu |
| 2 | `raincloud_plot(data, y_col, group_col)` | DataFrame | yarım violin + kutu + jitter (elle çizim) |
| 3 | `ridge_plot(data, value_col, group_col)` | DataFrame | üst üste kaymış yoğunluklar (JOYPLOT) |
| 4 | `pair_grid(data, cols, hue=None)` | DataFrame | sns.pairplot alt kümesi (en çok 6 sütun) |
| 5 | `grouped_boxplot(data, y_col, x_col, hue_col)` | DataFrame | iki faktörlü kutu |
| 6 | `strip_plot(data, x_col, y_col, jitter=True)` | DataFrame | sns.stripplot |
| 7 | `forest_plot(effects, ci_lower, ci_upper, labels)` | diziler | etki ± GA, dikey sıfır çizgisi |
| 8 | `bland_altman_plot(x, y)` | iki dizi | fark-ortalama, ±1.96 SD limitleri |
| 9 | `roc_plot(actual, predicted)` | 0/1 + skor | ROC eğrisi + AUC (analiz.roc_curve ile) |
| 10 | `survival_plot(time, event, group=None)` | diziler | Kaplan-Meier basamak eğrisi (survival.kaplan_meier ile) |
| 11 | `control_chart(values, subgroup_size=5)` | dizi | X̄-R tarzı: merkez çizgi ± 3σ, kural ihlali işaretleri |
| 12 | `hexbin_plot(x, y, gridsize=30)` | iki dizi | 2B yoğunluk |
| 13 | `stacked_area(labels, series_dict)` | sözlük | yığılmış alan grafiği |
| 14 | `growth_curve_plot(time, observed, model="logistic")` | diziler | gözlem + models.GrowthModel uydurma eğrisi |
| 15 | `slope_plot(before, after, labels)` | diziler | eşleşmiş değişim çizgileri |
| 16 | `residual_plot(fitted, residuals)` | diziler | artık diyagnostiği + yatay sıfır çizgisi |

Ortak davranış: başlık `fontsize=14, fontweight="bold"`, `tight_layout`,
tema paleti kullanımı; eksik/boş veride `ValueError`.

## 6. Alt Proje 3 — Etkileşimli Dashboard (`interactive.py`)

```python
def interactive_scatter(df, x, y, color=None, trendline=False) -> go.Figure
def interactive_line(df, x, y) -> go.Figure
def interactive_bar(df, x, y) -> go.Figure
def interactive_heatmap(df, columns=None) -> go.Figure
def interactive_box(df, x, y) -> go.Figure
def interactive_histogram(df, column) -> go.Figure

def build_dashboard(df: pd.DataFrame, output_path: str,
                    target: str | None = None,
                    title: str = "Agrista Keşif Paneli") -> dict
```

`build_dashboard` tek HTML üretir: KPI kartları (n, sütun sayısı,
eksik oranı), sayısal sütunlar için histogram matrisi, korelasyon ısı
haritası, `target` verilirse hedefe karşı scatter/box; `fig.write_html`
tam sayfa layout. Dönüş: `{"path": str, "n_figures": int, "n_rows": int}`.

## 7. Alt Proje 4 — Auto-EDA (`auto_eda.py`)

```python
def infer_column_types(df) -> dict            # her sütun: sayısal/kategorik/tarih/metin
def chart_suggestion(df) -> list[dict]        # [{"type", "reason", "columns"}, ...]
def auto_eda(df, output_dir: str) -> dict     # tam keşif raporu
```

Öneri kuralları (deterministik, öncelik sıralı): tek sayısal → histogram+
Q-Q; iki sayısal → scatter + hexbin; sayısal × kategorik → violin + gruplu
box; zaman sütunu + sayısal → çizgi; ≥3 sayısal → korelasyon ısı haritası +
pair_grid; tek kategorik → bar. `auto_eda`: betimsel özet tablosu + tüm
önerilen grafikleri `output_dir`'e PNG kaydeder + `report.html` üretir;
dönüş `{"html_path", "figures": [...], "suggestions": [...]}`.

## 8. CLI ve Menü (Grafik Merkezi)

| Komut | Seçenekler (özet) |
|---|---|
| `agrista plot FILE` | `--tip` (histogram/bar/scatter/box/violin/heat/qq/cizgi/...), `--x`, `--y`, `--grup`, `--tema`, `--cikti` |
| `agrista plot-forest FILE` | `--etki --alt --ust --etiket --cikti` |
| `agrista plot-roc FILE` | `--gercek --skor --cikti` |
| `agrista plot-survival FILE` | `--zaman --olay --grup --cikti` |
| `agrista dashboard FILE` | `--hedef --baslik --cikti` (varsayılan `dashboard.html`) |
| `agrista autoeda FILE` | `--cikti-dir` (varsayılan `agrista_eda`) |

Menü: **[21] 🎨 Grafikler** kategorisi — alt öğeler: "Hızlı grafik
(plot)", "Dağılım grafikleri (violin/ridge/raincloud)", "Tanı grafikleri
(Q-Q, artık, Bland-Altman)", "Model grafikleri (ROC, sağkalım, orman,
büyüme eğrisi)", "Etkileşimli dashboard", "Otomatik keşif (Auto-EDA)".
Kategori sayısı 20 → 21.

## 9. Test Stratejisi

- Her yeni metot için en az: (a) Figure döndürme + eksen/başlık kontrolü,
  (b) bir hata durumu testi.
- Tema testleri: rcParams değişimi + bilinmeyen tema hatası.
- Dışa aktarım: tmp_path'e png/svg/pdf/html dosyaları oluşur ve boyut > 0.
- Plotly: fig type kontrolü (`type(fig.data[0])`), dashboard HTML varlığı.
- Auto-EDA: deterministik veriyle öneri listesi birebir beklenir.
- CLI: CliRunner ile her komut için başarı + hata testi.
- Hedef: mevcut 459 test kırılmaz; yeni ≥ 70 test; kapsam düşmez.

## 10. Teslim

1. README: özellik listesi + menü 20 → 21 + yeni CLI örnekleri.
2. `docs/02` log'una "Güncelleme 6 — Görselleştirme Hamlesi" bölümü.
3. Sürüm 0.3.0; tam paket yeşil; teslim seçenekleri (birleştir / PR / dal).

## 11. Başarı Kriterleri

- [ ] `agrista/viz` alt dosya yapısı kurulu; mevcut 12 metot davranışı aynı.
- [ ] 16 yeni grafik metodu + 4 tema + save_multi/export_html çalışıyor.
- [ ] 6 etkileşimli grafik + `build_dashboard` HTML üretiyor.
- [ ] Auto-EDA öneri motoru + `auto_eda` raporu çalışıyor.
- [ ] 6 CLI komutu + menü [21] (21 kategori, smoke akışları yeşil).
- [ ] `pytest tests/` 0 hata; `flake8` temiz; sürüm 0.3.0; README/docs güncel.
