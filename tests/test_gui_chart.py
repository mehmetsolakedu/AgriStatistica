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
    from agrista.viz import AgristaPlotter
    p = ChartPanel(df)
    qtbot.addWidget(p)
    yield p
    AgristaPlotter.close()


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
