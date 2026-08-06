"""Agrista GUI pencere widget testleri (pytest-qt, offscreen)."""
import pandas as pd
import pytest

pytest.importorskip("PySide6")


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
        kayitsiz = [o for o in ogeler if "(planlanıyor)" in o.text()]
        assert len(kayitsiz) >= 1  # örn. Q-Q grafiği kayıtsız
        assert all(not o.isEnabled() for o in kayitsiz)

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


class TestAnalizAkisi:
    def test_bagli_ogeler_etkin(self, pencere):
        bar = pencere.menuBar()
        kategori = [a.menu() for a in bar.actions()
                    if a.text() == "📊 Betimsel İstatistikler"][0]
        adlar = {o.text() for o in kategori.actions()}
        assert "Betimsel özet tablosu" in adlar
        bagli = [o for o in kategori.actions()
                 if o.text() == "Betimsel özet tablosu"][0]
        assert bagli.isEnabled()

    def test_analiz_uc_tan_uca(self, pencere, qtbot, tmp_path, monkeypatch):
        from agrista.gui.analysis_dialog import AnalysisDialog
        from agrista.gui.registry import REGISTRY
        pencere.open_file(_csv(tmp_path))
        spec = next(s for s in REGISTRY if s.key == "betimsel")
        monkeypatch.setattr(AnalysisDialog, "exec", lambda self: 1)
        monkeypatch.setattr(AnalysisDialog, "degerler",
                            lambda self: {"kolonlar": "x"})
        pencere.analiz_calistir(spec)
        assert "count" in pencere.sonuc_paneli.toPlainText()

    def test_verisiz_analiz_uyarisi(self, pencere, qtbot, monkeypatch):
        from PySide6.QtWidgets import QMessageBox
        from agrista.gui.registry import REGISTRY
        monkeypatch.setattr(QMessageBox, "warning",
                            staticmethod(lambda *a, **k: None))
        spec = REGISTRY[0]
        pencere.analiz_calistir(spec)
        assert pencere.sonuc_paneli.toPlainText() == ""


class TestGrafikEntegrasyon:
    def test_grafik_sekmesi_panel(self, pencere):
        from agrista.gui.chart_view import ChartPanel
        assert isinstance(pencere.grafik_paneli, ChartPanel)

    def test_veri_acilinca_panel_guncellenir(self, pencere, tmp_path):
        pencere.open_file(_csv(tmp_path))
        assert pencere.grafik_paneli.df is pencere.df


class TestGuncellemeMenusu:
    def test_denetim_ag_hatasi_bilgisi(self, pencere, qtbot, monkeypatch):
        from PySide6.QtWidgets import QMessageBox
        import agrista.gui.updater as up
        mesajlar = []
        monkeypatch.setattr(up, "check_update", lambda *a, **k: None)
        monkeypatch.setattr(QMessageBox, "information",
                            staticmethod(lambda ebeveyn, baslik, metin:
                                         mesajlar.append(metin)))
        pencere._guncelleme_denetle()
        assert any("denetim" in m.lower() or "denetle" in m.lower()
                   for m in mesajlar)

    def test_denetim_yeni_surum(self, pencere, qtbot, monkeypatch):
        from PySide6.QtWidgets import QMessageBox
        import agrista.gui.updater as up
        mesajlar = []
        monkeypatch.setattr(up, "check_update", lambda *a, **k:
                            {"en_yeni": "9.9.9", "notes": "n",
                             "url": {}, "platform_url": None,
                             "guncelleme_var": True})
        monkeypatch.setattr(QMessageBox, "information",
                            staticmethod(lambda ebeveyn, baslik, metin:
                                         mesajlar.append(metin)))
        pencere._guncelleme_denetle()
        assert any("9.9.9" in m for m in mesajlar)
