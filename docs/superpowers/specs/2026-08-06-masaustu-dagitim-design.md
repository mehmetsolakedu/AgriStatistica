# Tasarım: Masaüstü Dağıtım Hamlesi — Wave 1 (PySide6 GUI + Paketleme + Güncelleme Hattı)

**Tarih:** 2026-08-06
**Durum:** Onaylandı (Aşama 1 — Brainstorming; Yaklaşım A, imzasız ama imzaya hazır dağıtım)
**Temel dal:** `feat/gorsellestirme-hamlesi` (v0.3.0) üzerine yeni özellik dalı

## 1. Amaç

Agrista'yı macOS/Windows'ta kurulabilen, uygulama içinden güncellenebilen
bir PySide6 masaüstü uygulaması olarak dağıtıma çıkarmak. Wave 1: GUI
iskeleti + veri görünümü + bildirimsel analiz koşucu + gömülü grafik
paneli + PyInstaller paketleme hattı (DMG/NSIS) + GitHub Releases tabanlı
güncelleme mekanizması.

## 2. Global Kısıtlar

1. **Bağımlılıklar:** PySide6 OPSIYONEL ekstra olur (`pip install
   agrista[gui]`); çekirdek CLI kurulumu hafif kalır. pytest-qt yalnız
   dev ekstra. Başka yeni bağımlılık yok; güncelleyici ağ erişimi stdlib
   (`urllib`) ile yapılır.
2. **Adlandırma:** Her yerde yalnızca "Premium Program" adı; referans
   paketinin eski adı hiçbir dosyada yer alamaz.
3. **Arayüz dili:** GUI metinleri Türkçe; mevcut analiz fonksiyonları
   değiştirilmez — GUI, `agrista.analysis`/branş modüllerini birebir
   çağırır.
4. **Menü denkliği:** GUI menü çubuğu mevcut 21 kategoriyle birebir
   kurulur; bildirimsel kayda bağlı olmayan öğeler menüde devre dışı ve
   "(planlanıyor)" son ekli görünür.
5. **İmza:** Wave 1 build'leri imzasızdır (macOS ad-hoc codesign);
   workflow'lar sertifika secret'ları eklendiğinde imza/notarizasyona
   hazır yapıda yazılır.
6. **Test:** TDD zorunlu; widget testleri `pytest-qt` + `offscreen`
   platformunda; ağ içeren kod saf fonksiyonlara ayrılır ve mock'lanır.
7. **Sürüm:** 0.3.0 → 0.4.0 (`pyproject.toml`, `__version__`, banner,
   `version_option`).
8. **Kalite kapısı:** `pytest` tam paket yeşil + `flake8` temiz olmadan
   teslim yok.

## 3. Mimari ve Dosya Yerleşimi

| Dosya | Sorumluluk |
|---|---|
| `agrista/gui/__init__.py` | paket; `run()` |
| `agrista/gui/main.py` (yeni) | `main()` giriş noktası (`agrista-gui` script) |
| `agrista/gui/theme.py` (yeni) | koyu/açık QSS temaları, palet |
| `agrista/gui/data_model.py` (yeni) | `DataFrameModel(QAbstractTableModel)` — pandas → tablo |
| `agrista/gui/registry.py` (yeni) | `Param`, `AnalysisSpec`, `REGISTRY` bildirimsel analiz kaydı + adaptörler |
| `agrista/gui/analysis_dialog.py` (yeni) | otomatik form diyaloğu (parametre şemasından üretilir) |
| `agrista/gui/chart_view.py` (yeni) | `FigureCanvas` sarmalayıcı + grafik tipi seçici |
| `agrista/gui/main_window.py` (yeni) | `MainWindow(QMainWindow)`: menü, veri tablosu, sonuç paneli, grafik paneli, durum çubuğu |
| `agrista/gui/updater.py` (yeni) | `parse_version`, `check_update(url)` — saf + ağ katmanı ayrık |
| `packaging/agrista-gui.spec` (yeni) | PyInstaller spesifikasyonu |
| `packaging/build_macos.sh` (yeni) | .app + ad-hoc codesign + DMG |
| `packaging/build_windows.ps1` (yeni) | .exe + NSIS kurulumcu |
| `packaging/agrista.nsi` (yeni) | NSIS betiği |
| `.github/workflows/release.yml` (yeni) | etiket tetiklemeli matrix build + Release varlıkları + `latest.json` |
| `.github/workflows/ci.yml` | GUI test job'ı ekleme (offscreen) |
| `tests/test_gui_*.py` (yeni) | widget + kayıt + güncelleyici testleri |

`pyproject.toml`: `[project.optional-dependencies] gui = ["PySide6>=6.6"]`;
dev ekstraya `pytest-qt`; `[project.scripts]` içine
`agrista-gui = "agrista.gui.main:main"`; `description` alanına masaüstü
uygulama notu.

## 4. Alt Proje 1 — GUI İskeleti

`MainWindow`: sol tarafta `QTableView` (veri), sağda sekmeli panel
("Sonuçlar" metin/ağaç görünümü, "Grafik" canvas), üstte menü çubuğu,
altta durum çubuğu. Menü, 21 kategorinin `QMenu` karşılığıdır; her öğe
kayda bağlıysa tıklanınca `AnalysisDialog` açar, değilse devre dışı.
Dosya menüsü: "Veri Aç…" (CSV/Excel), "Grafiği Kaydet…", "Güncellemeleri
Denetle…", "Çıkış". Tema: `theme.py` içinde iki QSS sözlüğü
(`LIGHT_QSS`, `DARK_QSS`); Görünüm menüsünden geçiş; varsayılan açık tema.

## 5. Alt Proje 2 — Veri Görünümü

`DataFrameModel`: `pd.DataFrame`'i satır/sütun limitiyle (ilk 10.000
satır) gösterir; başlıkta dtype bilgisi. `MainWindow.open_file(path)`:
`agrista.data.load_csv/load_excel` (mevcut `AgristaData`) ile yükler,
tabloya bağlar, durum çubuğuna `n satır × n sütun` yazar. Boş/eksik
dosyada hata iletişim kutusu.

## 6. Alt Proje 3 — Analiz Koşucu

```python
@dataclass
class Param:
    name: str; label: str
    kind: str            # "column" | "columns" | "numeric" | "choice"
    required: bool = True
    default: object = None
    choices: tuple = ()

@dataclass
class AnalysisSpec:
    key: str; menu_category: str; label: str
    run: Callable[[pd.DataFrame, dict], dict]   # adaptör
    params: list

REGISTRY: list[AnalysisSpec]
```

Adaptörler `registry.py` içinde tanımlanır; her adaptör `(df, p) -> dict`
imzalıdır ve mevcut analiz fonksiyonlarını çağırır. Wave 1 kaydı (16
analiz; imzalar planlama aşamasında koddan doğrulanır): betimsel özet,
frekanslar, çapraz tablolar, tek örneklem t, bağımsız iki örneklem t
(grup sütununa göre bölme adaptörü), eşleştirilmiş t, ANOVA + Tukey,
korelasyon, çoklu regresyon, normallik, GLM univariate, GEE, ROC,
Kaplan-Meier, AUDPC, oran istatistikleri. `AnalysisDialog`: şemadan
otomatik form üretir (`QComboBox` sütun seçici — veri sütunlarından;
`QDoubleSpinBox` sayısal; `QComboBox` choice). Sonuç dict'i "Sonuçlar"
panelinde hiyerarşik metin olarak çizilir (dict/list yinelemeli
serileştirici `format_result`).

## 7. Alt Proje 4 — Grafik Paneli

"Grafik" sekmesi: grafik tipi seçici (12 tip: histogram, boxplot,
scatter, violin, raincloud, ridge, bar, çizgi, pasta, Q-Q, ısı haritası,
hata çubuğu) + gereken sütunları toplayan mini form + `FigureCanvas`
(matplotlib Qt backend). `AgristaPlotter(theme=...)` yeniden kullanılır;
"Kaydet…" PNG/SVG/PDF kaydeder.

## 8. Alt Proje 5 — Paketleme Hattı

- `agrista-gui.spec`: tek klasör (`--onedir`) build; giriş
  `agrista/gui/main.py`; hiddenimports pandas/matplotlib backends;
  matplotlib data dosyaları toplanır.
- macOS betiği: pyinstaller → `codesign --force --deep --sign -`
  (ad-hoc) → `hdiutil create` ile `Agrista-<sürüm>-macOS.dmg`. İmza
  secret'ı (`MACOS_SIGNING_IDENTITY`) tanımlıysa gerçek kimlikle imza
  ve `xcrun notarytool` adımı etkinleşir (betikte koşul).
- Windows betiği: pyinstaller → `makensis packaging/agrista.nsi` →
  `Agrista-<sürüm>-Setup.exe`.
- `release.yml`: `on: push: tags: ['v*']`; matrix `macos-latest` /
  `windows-latest`; Python 3.11; `pip install .[gui] pyinstaller`;
  build betiği; varlıkları `softprops/action-gh-release` ile yükle.
  Ayrı `latest` job'ı: `latest.json` üretir
  (`{"version", "notes", "assets": {"macos": url, "windows": url}}`)
  ve Release'e ekler.

## 9. Alt Proje 6 — Uygulama İçi Güncelleyici

```python
def parse_version(v: str) -> tuple           # "0.4.0" -> (0,4,0)
def compare_versions(a: str, b: str) -> int  # -1/0/1
def build_update_info(payload: dict, current: str) -> dict | None
def fetch_latest(url: str, timeout: float = 5.0) -> dict  # urllib, JSON
def check_update(current: str, url: str = DEFAULT_URL) -> dict | None
```

`DEFAULT_URL`, GitHub Releases'taki `latest.json` adresidir (depo
public varsayımı; private kalırsa adres yapılandırma ile değişir).
Başlangıçta arka plan denetimi yok — "Güncellemeleri Denetle…" menü
öğesi elle tetikler; yeni sürüm varsa sürüm notları + "İndir" düğmesi
(platform varlık URL'sini tarayıcıda açar). Ağ hatası sessizce "güncel
denetim yapılamadı" mesajı üretir.

## 10. Test Stratejisi

- Widget testleri (`pytest-qt`, `QT_QPA_PLATFORM=offscreen`): pencere
  açılışı; menüde 21 kategori; CSV fixture yükleme → tablo boyutu;
  betimsel analiz akışı (diyalog parametreleri → sonuç paneli metni);
  grafik üretimi (canvas axes sayısı); tema geçişi.
- Saf birim testleri: `parse_version`/`compare_versions`/
  `build_update_info` (ağ yok); `format_result` dict serileştirme;
  `DataFrameModel` satır/sütun sayısı ve limit davranışı.
- CI'ya `gui` job'ı: ubuntu-latest, `QT_QPA_PLATFORM=offscreen` ile
  GUI testleri.
- Hedef: mevcut 526 test kırılmaz; yeni ≥ 30 test.

## 11. Teslim

1. README: "Masaüstü Uygulaması" bölümü (kurulum `pip install
   agrista[gui]`, `agrista-gui`, indirme bağlantıları yeri).
2. `docs/02` log'una "Güncelleme 7 — Masaüstü Dağıtım (Wave 1)".
3. Sürüm 0.4.0; tam paket yeşil; teslim seçenekleri.
4. Dal birleşip etiket (`v0.4.0`) basıldığında `release.yml` varlık
   üretir — bu adım kullanıcıyla birlikte doğrulanır.

## 12. Başarı Kriterleri

- [ ] `agrista-gui` macOS'ta pencere açar; menü 21 kategori; veri açma
      ve tablo görünümü çalışır.
- [ ] 16 analiz kayıt üzerinden uçtan uca çalışır; bağlı olmayan öğeler
      "(planlanıyor)" ile devre dışı.
- [ ] 12 grafik tipi gömülü canvas'ta üretilir ve kaydedilir.
- [ ] PyInstaller build betikleri + release workflow tam; `latest.json`
      şeması testli.
- [ ] Güncelleyici saf fonksiyonları testli; menü denetimi çalışır.
- [ ] `pytest tests/` 0 hata; `flake8` temiz; sürüm 0.4.0.

## 13. Açık Varsayım

Depo dağıtım için public yapılacaktır (kod MIT lisanslıdır); anonim
indirme ve güncelleme denetimi buna bağlıdır. Private kalması istenirse
`DEFAULT_URL` farklı bir dağıtım noktasına çevrilir.
