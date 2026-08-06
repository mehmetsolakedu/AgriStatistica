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
