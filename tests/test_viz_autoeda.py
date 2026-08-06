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
