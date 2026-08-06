"""Agrista premium grafik kütüphanesi testleri (16 yeni metot)."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from agrista.viz import AgristaPlotter


@pytest.fixture(autouse=True)
def _kapla():
    yield
    AgristaPlotter.close()


@pytest.fixture
def plotter():
    return AgristaPlotter()


def _grup_veri(seed=3, n=60):
    rng = np.random.default_rng(seed)
    grup = rng.choice(["A", "B", "C"], n)
    etki = {"A": 0.0, "B": 1.5, "C": 3.0}
    return pd.DataFrame({"grup": grup,
                         "verim": np.array([etki[g] for g in grup])
                                  + rng.normal(0, 1, n)})


class TestDagilimGrafikleri:
    def test_violin(self, plotter):
        fig = plotter.violin_plot(_grup_veri(), x_col="grup", y_col="verim")
        assert fig.axes[0].get_title() == "Violin Grafiği"

    def test_violin_hue(self, plotter):
        df = _grup_veri()
        df["blok"] = ["x", "y"] * (len(df) // 2)
        fig = plotter.violin_plot(df, x_col="grup", y_col="verim",
                                  hue="blok")
        assert len(fig.axes) >= 1

    def test_raincloud(self, plotter):
        fig = plotter.raincloud_plot(_grup_veri(), y_col="verim",
                                     group_col="grup")
        assert "Raincloud" in fig.axes[0].get_title()

    def test_ridge(self, plotter):
        fig = plotter.ridge_plot(_grup_veri(), value_col="verim",
                                 group_col="grup")
        assert len(fig.axes[0].collections) >= 3

    def test_pair_grid(self, plotter):
        rng = np.random.default_rng(5)
        df = pd.DataFrame({"a": rng.normal(size=40),
                           "b": rng.normal(size=40),
                           "c": rng.normal(size=40)})
        fig = plotter.pair_grid(df, cols=["a", "b", "c"])
        assert fig.axes.shape == (3, 3)

    def test_grouped_boxplot(self, plotter):
        df = _grup_veri()
        df["blok"] = ["x", "y"] * (len(df) // 2)
        fig = plotter.grouped_boxplot(df, y_col="verim", x_col="grup",
                                      hue_col="blok")
        assert fig.axes[0].get_title() == "Gruplu Kutu Grafiği"

    def test_strip(self, plotter):
        fig = plotter.strip_plot(_grup_veri(), x_col="grup", y_col="verim")
        assert len(fig.axes[0].collections) >= 1

    def test_violin_eksik_sutun_hatasi(self, plotter):
        with pytest.raises(ValueError):
            plotter.violin_plot(_grup_veri(), x_col="yok", y_col="verim")

    def test_pair_grid_cok_sutun_hatasi(self, plotter):
        df = pd.DataFrame(np.random.default_rng(1).normal(size=(10, 8)),
                          columns=[f"s{i}" for i in range(8)])
        with pytest.raises(ValueError):
            plotter.pair_grid(df, cols=list(df.columns))

    def test_raincloud_tek_grup_hatasi(self, plotter):
        df = _grup_veri()
        df["grup"] = "A"
        with pytest.raises(ValueError):
            plotter.raincloud_plot(df, y_col="verim", group_col="grup")
