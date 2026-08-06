# Plan: Masaüstü Wave 1 — GUI İskeleti + Veri Görünümü + Tema

**Spec:** `docs/superpowers/specs/2026-08-06-masaustu-dagitim-design.md` (§3-5)

**Goal:** PySide6 uygulama kabuğunu (menü çubuğu 21 kategori, veri tablosu,
sonuç paneli), pandas tablo modelini, tema sistemini ve `agrista-gui`
giriş noktasını kurmak; analiz kaydı bu planda boş şema olarak oluşturulur
(doldurulması Plan 2).

**Architecture:** `agrista/gui` paketi: `theme.py` (QSS sabitleri),
`data_model.py` (QAbstractTableModel), `registry.py` (veri sınıfları + boş
REGISTRY + format_result), `main_window.py` (QMainWindow), `main.py`
(giriş). Menü başlık/etiketleri `agrista.cli._build_menu_structure()`
çıktısından okunur (CLI ile birebir denklik).

**Tech Stack:** PySide6 (opsiyonel ekstra), pandas, pytest-qt (dev).

**Global Constraints (spec'ten aynen):**
1. PySide6 opsiyonel ekstra (`agrista[gui]`); pytest-qt yalnız dev;
   başka yeni bağımlılık yok; ağ kodu stdlib.
2. Yalnızca "Premium Program" adı; eski ad yasak.
3. GUI Türkçe; mevcut analiz fonksiyonları değişmez.
4. Menü 21 kategoriyle birebir; kayıtsız öğeler devre dışı "(planlanıyor)".
5. İmza konusu bu planda yok.
6. TDD zorunlu; widget testleri pytest-qt + offscreen; ağ kodu ayrık.
7. Sürüm bu planda değişmez (0.4.0 son planda).
8. `pytest` tam paket yeşil + `flake8` temiz olmadan görev bitmez.

## Dosya Yapısı

| Dosya | Sorumluluk |
|---|---|
| `pyproject.toml` | gui/dev ekstraları + `agrista-gui` script |
| `tests/conftest.py` | offscreen ortam değişkeni |
| `agrista/gui/__init__.py` · `theme.py` · `data_model.py` · `registry.py` · `main_window.py` · `main.py` | GUI paket dosyaları |
| `tests/test_gui_core.py` · `tests/test_gui_window.py` | saf + widget testleri |
| `.github/workflows/ci.yml` | gui test job'ı |

---

## Task 1: pyproject + offscreen conftest + tema/veri modeli (TDD)

**Files:** Modify: `pyproject.toml`, `tests/conftest.py` · Create: `agrista/gui/__init__.py`, `agrista/gui/theme.py`, `agrista/gui/data_model.py` · Test: `tests/test_gui_core.py`
**Interfaces:** `DataFrameModel(df=None)`, `LIGHT_QSS`, `DARK_QSS`.

- [ ] **RED** — `tests/test_gui_core.py` oluştur (önce `pyproject.toml`
      güncellenir ve `.venv/bin/pip install -e ".[gui,dev]"` çalıştırılır):

```python
"""Agrista GUI çekirdek testleri (model, tema, format_result)."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest


def _df():
    return pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})


class TestDataFrameModel:
    def test_satir_sutun(self):
        from agrista.gui.data_model import DataFrameModel
        m = DataFrameModel(_df())
        assert m.rowCount() == 3
        assert m.columnCount() == 2

    def test_baslik_dtype(self):
        from agrista.gui.data_model import DataFrameModel
        from PySide6.QtCore import Qt
        m = DataFrameModel(_df())
        assert "float" in m.headerData(0, Qt.Horizontal)
        assert m.headerData(0, Qt.Vertical) == "1"

    def test_hucre_degeri(self):
        from agrista.gui.data_model import DataFrameModel
        from PySide6.QtCore import Qt
        m = DataFrameModel(_df())
        assert m.data(m.index(2, 1)) == "z"

    def test_satir_limiti(self):
        from agrista.gui.data_model import DataFrameModel, MAX_ROWS
        buyuk = pd.DataFrame({"x": np.arange(MAX_ROWS + 50, dtype=float)})
        assert DataFrameModel(buyuk).rowCount() == MAX_ROWS

    def test_bos_model(self):
        from agrista.gui.data_model import DataFrameModel
        m = DataFrameModel()
        assert m.rowCount() == 0 and m.columnCount() == 0


class TestTema:
    def test_temalar_tanimli(self):
        from agrista.gui.theme import LIGHT_QSS, DARK_QSS
        assert "QMainWindow" in LIGHT_QSS
        assert "QMainWindow" in DARK_QSS

    def test_tema_secici(self):
        from agrista.gui.theme import tema_qss
        assert tema_qss("açık") != tema_qss("koyu")
        with pytest.raises(ValueError):
            tema_qss("olmayan")
```

- [ ] Çalıştır: `.venv/bin/python -m pytest tests/test_gui_core.py -x -q`
      → ModuleNotFoundError (`agrista.gui`).
- [ ] **GREEN** — `pyproject.toml` değişiklikleri: `[project.optional-dependencies]`
      altına `gui = ["PySide6>=6.6"]`; mevcut `dev` listesine `"pytest-qt>=4.2"`
      maddesi; `[project.scripts]` içine `agrista-gui = "agrista.gui.main:main"`.
      Ardından `.venv/bin/pip install -e ".[gui,dev]" -q`.
- [ ] `tests/conftest.py` başına (import'lardan önce) ekle:

```python
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
```

- [ ] `agrista/gui/__init__.py`:

```python
"""Agrista GUI — PySide6 masaüstü uygulaması."""
```

- [ ] `agrista/gui/theme.py`:

```python
"""Agrista GUI tema sistemi — açık ve koyu QSS temaları."""

ACCENT = "#2E86AB"

LIGHT_QSS = f"""
QMainWindow {{ background: #f7f9fa; }}
QMenuBar {{ background: #ffffff; color: #1b1b1b; }}
QMenuBar::item:selected {{ background: {ACCENT}; color: white; }}
QMenu {{ background: #ffffff; }}
QMenu::item:selected {{ background: {ACCENT}; color: white; }}
QTableView {{ background: #ffffff; gridline-color: #dfe3e8; }}
QTabWidget::pane {{ border: 1px solid #dfe3e8; }}
QPushButton {{ background: {ACCENT}; color: white; border: none;
              border-radius: 4px; padding: 6px 14px; }}
QStatusBar {{ background: #eef2f4; }}
"""

DARK_QSS = f"""
QMainWindow {{ background: #1e1e1e; }}
QMenuBar {{ background: #252526; color: #e0e0e0; }}
QMenuBar::item:selected {{ background: {ACCENT}; color: white; }}
QMenu {{ background: #2d2d30; color: #e0e0e0; }}
QMenu::item:selected {{ background: {ACCENT}; color: white; }}
QMenu::item:disabled {{ color: #6f6f6f; }}
QTableView {{ background: #1e1e1e; color: #e0e0e0;
             gridline-color: #333333; }}
QTextEdit, QPlainTextEdit {{ background: #252526; color: #e0e0e0; }}
QTabWidget::pane {{ border: 1px solid #333333; }}
QPushButton {{ background: {ACCENT}; color: white; border: none;
              border-radius: 4px; padding: 6px 14px; }}
QStatusBar {{ background: #252526; color: #e0e0e0; }}
"""


def tema_qss(ad: str) -> str:
    """Tema adı → QSS metni."""
    temalar = {"açık": LIGHT_QSS, "koyu": DARK_QSS}
    if ad not in temalar:
        raise ValueError(f"Bilinmeyen tema: {ad}")
    return temalar[ad]
```

- [ ] `agrista/gui/data_model.py`:

```python
"""Agrista GUI veri modeli — pandas DataFrame → Qt tablo."""

from __future__ import annotations

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, Qt

MAX_ROWS = 10000


class DataFrameModel(QAbstractTableModel):
    """Salt-okunur pandas tablo modeli (ilk MAX_ROWS satır)."""

    def __init__(self, df: pd.DataFrame = None, parent=None):
        super().__init__(parent)
        self._df = df.head(MAX_ROWS).reset_index(drop=True) \
            if df is not None else pd.DataFrame()

    def rowCount(self, parent=None) -> int:
        return len(self._df)

    def columnCount(self, parent=None) -> int:
        return self._df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and index.isValid():
            return str(self._df.iat[index.row(), index.column()])
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            kolon = self._df.columns[section]
            return f"{kolon} ({self._df[kolon].dtype})"
        return str(section + 1)

    def set_dataframe(self, df: pd.DataFrame):
        self.beginResetModel()
        self._df = df.head(MAX_ROWS).reset_index(drop=True)
        self.endResetModel()
```

- [ ] Çalıştır: `.venv/bin/python -m pytest tests/test_gui_core.py -q` → yeşil.
- [ ] Commit: `feat(gui): paket iskeleti, tema sistemi, DataFrameModel`

## Task 2: Kayıt şeması + format_result (TDD)

**Files:** Test: `tests/test_gui_core.py` (ek) · Create: `agrista/gui/registry.py`
**Interfaces:** `Param`, `AnalysisSpec`, `REGISTRY`, `format_result(obj) -> str`.

- [ ] **RED** — test dosyası sonuna ekle:

```python
class TestKayitVeFormat:
    def test_kayit_baslangicta_bos(self):
        from agrista.gui.registry import REGISTRY
        assert isinstance(REGISTRY, list)

    def test_param_ve_spec(self):
        from agrista.gui.registry import AnalysisSpec, Param
        p = Param(name="kolon", label="Kolon", kind="column")
        s = AnalysisSpec(key="k", menu_category="m", label="l",
                         run=lambda df, p: {}, params=[p])
        assert s.params[0].kind == "column"

    def test_format_result_dict(self):
        from agrista.gui.registry import format_result
        metin = format_result({"a": 1.5, "b": {"c": "x"}})
        assert "a: 1.5" in metin and "c: x" in metin

    def test_format_result_liste_ve_skaler(self):
        from agrista.gui.registry import format_result
        assert "1" in format_result([1, 2])
        assert format_result(3.25) == "3.25"
```

- [ ] Çalıştır → ModuleNotFoundError.
- [ ] **GREEN** — `agrista/gui/registry.py` oluştur:

```python
"""Agrista GUI analiz kaydı — bildirimsel analiz tanımları.

Her AnalysisSpec bir menü öğesini bir analiz fonksiyonuna bağlar;
formlar parametre şemasından otomatik üretilir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd


@dataclass
class Param:
    """Analiz parametresi şeması.

    kind: "column" | "columns" | "numeric" | "choice"
    """
    name: str
    label: str
    kind: str
    required: bool = True
    default: object = None
    choices: tuple = ()


@dataclass
class AnalysisSpec:
    """Bir analiz öğesinin menü + çalışma tanımı."""
    key: str
    menu_category: str
    label: str
    run: Callable[[pd.DataFrame, dict], dict]
    params: list = field(default_factory=list)


REGISTRY: list = []


def format_result(obj, indent: int = 0) -> str:
    """dict/list sonuçlarını hiyerarşik metne çevirir."""
    girinti = "  " * indent
    if isinstance(obj, dict):
        satirlar = []
        for anahtar, deger in obj.items():
            if isinstance(deger, (dict, list)):
                satirlar.append(f"{girinti}{anahtar}:")
                satirlar.append(format_result(deger, indent + 1))
            else:
                satirlar.append(f"{girinti}{anahtar}: {_skaler(deger)}")
        return "\n".join(satirlar)
    if isinstance(obj, (list, tuple)):
        return "\n".join(f"{girinti}- {format_result(o, 0).strip()}"
                         if not isinstance(o, (dict, list))
                         else format_result(o, indent + 1) for o in obj)
    return f"{girinti}{_skaler(obj)}"


def _skaler(deger) -> str:
    if isinstance(deger, float):
        return f"{deger:.6g}"
    return str(deger)
```

- [ ] Çalıştır: `.venv/bin/python -m pytest tests/test_gui_core.py -q` → yeşil.
- [ ] Commit: `feat(gui): analiz kayıt şeması + format_result`

## Task 3: MainWindow + menü + veri açma (TDD)

**Files:** Test: `tests/test_gui_window.py` (Create) · Create: `agrista/gui/main_window.py`, `agrista/gui/main.py`
**Interfaces:** `MainWindow()`, `MainWindow.open_file(path)`, `main()`.

- [ ] **RED** — `tests/test_gui_window.py` oluştur:

```python
"""Agrista GUI pencere widget testleri (pytest-qt, offscreen)."""
import pandas as pd
import pytest


@pytest.fixture
def pencere(qtbot):
    from agrista.gui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _csv(tmp_path):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "g": ["a", "b", "a"]})
    yol = tmp_path / "veri.csv"
    df.to_csv(yol, index=False)
    return str(yol)


class TestMainWindow:
    def test_menu_21_kategori(self, pencere):
        bar = pencere.menuBar()
        basliklar = [a.text() for a in bar.actions()]
        assert len(basliklar) >= 23  # Dosya + Görünüm + 21 kategori
        assert "📁 Dosya" in basliklar
        assert "🎨 Grafikler" in basliklar

    def test_kayitsiz_ogeler_devre_disi(self, pencere):
        bar = pencere.menuBar()
        kategori = [a.menu() for a in bar.actions()
                    if a.text() == "📊 Betimsel İstatistikler"][0]
        ogeler = kategori.actions()
        assert len(ogeler) >= 1
        assert all(not o.isEnabled() or "(planlanıyor)" in o.text()
                   for o in ogeler)

    def test_veri_acma(self, pencere, qtbot, tmp_path):
        pencere.open_file(_csv(tmp_path))
        assert pencere.model.rowCount() == 3
        assert pencere.model.columnCount() == 2
        assert "3 satır" in pencere.statusBar().currentMessage()

    def test_veri_acma_hatali_dosya(self, pencere, tmp_path):
        with pytest.raises(ValueError):
            pencere.open_file(str(tmp_path / "yok.csv"), sessiz=True)

    def test_tema_gecisi(self, pencere):
        pencere.tema_uygula("koyu")
        assert "QMainWindow" in pencere.styleSheet()
        assert pencere._tema == "koyu"
```

- [ ] Çalıştır: `.venv/bin/python -m pytest tests/test_gui_window.py -x -q`
      → ModuleNotFoundError (`main_window`).
- [ ] **GREEN** — `agrista/gui/main_window.py` oluştur:

```python
"""Agrista GUI ana pencere — menü, veri tablosu, sonuç paneli."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFileDialog, QMainWindow, QMessageBox,
                               QSplitter, QTabWidget, QTableView,
                               QTextEdit)

from agrista.data import load_csv, load_excel
from agrista.gui.data_model import DataFrameModel
from agrista.gui.registry import REGISTRY, format_result
from agrista.gui.theme import tema_qss


def _menu_yapisi():
    """CLI menü yapısının başlık/etiketleri (denklik tek kaynaktan)."""
    from agrista.cli import _build_menu_structure
    return [(baslik, [etiket for etiket, _ in islemler])
            for baslik, islemler in _build_menu_structure()]


class MainWindow(QMainWindow):
    """Agrista masaüstü ana penceresi."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Agrista — Tarımsal İstatistik Yazılımı")
        self.resize(1200, 760)
        self._tema = "açık"
        self.df = pd.DataFrame()

        self.model = DataFrameModel()
        self.tablo = QTableView()
        self.tablo.setModel(self.model)

        self.sonuc_paneli = QTextEdit()
        self.sonuc_paneli.setReadOnly(True)
        self.grafik_sekmesi = QTextEdit()  # Plan 3'te canvas ile değişir
        self.grafik_sekmesi.setReadOnly(True)
        sekmeler = QTabWidget()
        sekmeler.addTab(self.sonuc_paneli, "Sonuçlar")
        sekmeler.addTab(self.grafik_sekmesi, "Grafik")

        bolucu = QSplitter(Qt.Horizontal)
        bolucu.addWidget(self.tablo)
        bolucu.addWidget(sekmeler)
        bolucu.setStretchFactor(0, 3)
        bolucu.setStretchFactor(1, 2)
        self.setCentralWidget(bolucu)

        self._menu_kur()
        self.statusBar().showMessage("Hazır — veri açmak için Dosya → Veri Aç")
        self.tema_uygula(self._tema)

    # -- menü ---------------------------------------------------------
    def _menu_kur(self):
        bar = self.menuBar()
        dosya = bar.addMenu("📁 Dosya")
        dosya.addAction("Veri Aç…", self._veri_ac_dialog)
        dosya.addSeparator()
        dosya.addAction("Güncellemeleri Denetle…", self._guncelleme_denetle)
        dosya.addSeparator()
        dosya.addAction("Çıkış", self.close)

        gorunum = bar.addMenu("👁 Görünüm")
        gorunum.addAction("Açık Tema", lambda: self.tema_uygula("açık"))
        gorunum.addAction("Koyu Tema", lambda: self.tema_uygula("koyu"))

        bagli = {(s.menu_category, s.label): s for s in REGISTRY}
        for baslik, etiketler in _menu_yapisi():
            if baslik.startswith("📁"):
                continue  # Dosya menüsü zaten kuruldu
            menu = bar.addMenu(baslik)
            for etiket in etiketler:
                spec = bagli.get((baslik, etiket))
                if spec is not None:
                    menu.addAction(etiket,
                                   lambda s=spec: self.analiz_calistir(s))
                else:
                    eylem = menu.addAction(f"{etiket} (planlanıyor)")
                    eylem.setEnabled(False)

    # -- veri ---------------------------------------------------------
    def _veri_ac_dialog(self):
        yol, _ = QFileDialog.getOpenFileName(
            self, "Veri dosyası seç", "",
            "Veri dosyaları (*.csv *.xlsx *.xls)")
        if yol:
            self.open_file(yol)

    def open_file(self, path: str, sessiz: bool = False):
        """CSV/Excel dosyasını yükleyip tabloya bağlar."""
        p = Path(path)
        if not p.exists():
            raise ValueError(f"Dosya bulunamadı: {path}")
        if p.suffix.lower() == ".csv":
            veri = load_csv(str(p))
        elif p.suffix.lower() in (".xlsx", ".xls"):
            veri = load_excel(str(p))
        else:
            raise ValueError(f"Desteklenmeyen uzantı: {p.suffix}")
        self.df = veri.dataframe
        self.model.set_dataframe(self.df)
        mesaj = f"{len(self.df)} satır × {self.df.shape[1]} sütun — {p.name}"
        self.statusBar().showMessage(mesaj)
        if not sessiz:
            self.sonuc_paneli.setPlainText(f"Veri yüklendi: {mesaj}")

    # -- analiz / tema / güncelleme ------------------------------------
    def analiz_calistir(self, spec):
        """Kayıtlı analizi çalıştırır (diyalog Plan 2'de bağlanır)."""
        self.sonuc_paneli.setPlainText(
            f"{spec.label}: diyalog entegrasyonu bekleniyor")

    def sonuc_goster(self, baslik: str, sonuc: dict):
        self.sonuc_paneli.setPlainText(
            f"{baslik}\n{'=' * len(baslik)}\n\n{format_result(sonuc)}")

    def tema_uygula(self, ad: str):
        self._tema = ad
        self.setStyleSheet(tema_qss(ad))

    def _guncelleme_denetle(self):
        QMessageBox.information(self, "Güncellemeler",
                                "Güncelleme denetimi Plan 5'te bağlanacak.")
```

- [ ] `agrista/gui/main.py` oluştur:

```python
"""Agrista GUI giriş noktası (`agrista-gui`)."""

from __future__ import annotations

import sys


def main() -> None:
    from PySide6.QtWidgets import QApplication

    from agrista.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Agrista")
    pencere = MainWindow()
    pencere.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

- [ ] Çalıştır: `.venv/bin/python -m pytest tests/test_gui_window.py -q` → yeşil.
- [ ] Commit: `feat(gui): ana pencere — 21 kategorili menü, veri tablosu, tema`

## Task 4: CI gui job'ı + plan kapanışı

**Files:** Modify: `.github/workflows/ci.yml`

- [ ] `ci.yml` içine yeni job ekle (`test` job'ından sonra):

```yaml
  gui:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install package + gui extras
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[gui,dev]"
      - name: GUI tests (offscreen)
        env:
          QT_QPA_PLATFORM: offscreen
        run: python -m pytest tests/test_gui_core.py tests/test_gui_window.py -q
```

- [ ] Tam doğrulama: `.venv/bin/python -m pytest tests/ -q` ve
      `.venv/bin/python -m flake8 agrista tests` → temiz.
- [ ] Commit: `ci: GUI test job'ı (offscreen)`
