"""Agrista GUI ana pencere — menü, veri tablosu, sonuç paneli."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFileDialog, QMainWindow, QMenu, QMessageBox,
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
        self._menuler = []

        def _ekle(baslik):
            # Açık ebeveyn + referans: PySide6 GC'sine karşı koruma
            menu = QMenu(baslik, bar)
            self._menuler.append(menu)
            bar.addMenu(menu)
            return menu

        dosya = _ekle("📁 Dosya")
        dosya.addAction("Veri Aç…", self._veri_ac_dialog)
        dosya.addSeparator()
        dosya.addAction("Güncellemeleri Denetle…", self._guncelleme_denetle)
        dosya.addSeparator()
        dosya.addAction("Çıkış", self.close)

        gorunum = _ekle("👁 Görünüm")
        gorunum.addAction("Açık Tema", lambda: self.tema_uygula("açık"))
        gorunum.addAction("Koyu Tema", lambda: self.tema_uygula("koyu"))

        bagli = {(s.menu_category, s.label): s for s in REGISTRY}
        for baslik, etiketler in _menu_yapisi():
            menu = _ekle(baslik)
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
