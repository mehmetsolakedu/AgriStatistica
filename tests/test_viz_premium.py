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


class TestTaniVeModelGrafikleri:
    def test_forest(self, plotter):
        fig = plotter.forest_plot(effects=[0.5, -0.2, 1.1],
                                  ci_lower=[0.1, -0.6, 0.4],
                                  ci_upper=[0.9, 0.2, 1.8],
                                  labels=["A", "B", "C"])
        assert len(fig.axes[0].lines) >= 1  # sıfır çizgisi

    def test_forest_uzunluk_hatasi(self, plotter):
        with pytest.raises(ValueError):
            plotter.forest_plot(effects=[0.5], ci_lower=[0.1, 0.2],
                                ci_upper=[0.9], labels=["A", "B"])

    def test_bland_altman(self, plotter):
        rng = np.random.default_rng(4)
        x = rng.normal(50, 5, 40)
        fig = plotter.bland_altman_plot(x, x + rng.normal(0, 2, 40))
        # orta çizgi + 2 limit = 3 yatay çizgi
        assert len(fig.axes[0].lines) >= 3

    def test_roc(self, plotter):
        rng = np.random.default_rng(6)
        gercek = rng.integers(0, 2, 100)
        skor = gercek * 0.6 + rng.normal(0, 0.4, 100)
        fig = plotter.roc_plot(gercek, skor)
        assert "AUC" in fig.axes[0].get_title()

    def test_survival(self, plotter):
        rng = np.random.default_rng(7)
        zaman = rng.exponential(10, 50)
        olay = rng.integers(0, 2, 50)
        fig = plotter.survival_plot(zaman, olay)
        assert "Sağkalım" in fig.axes[0].get_title()

    def test_survival_gruplu(self, plotter):
        rng = np.random.default_rng(8)
        zaman = rng.exponential(10, 60)
        olay = rng.integers(0, 2, 60)
        grup = rng.choice(["kontrol", "ilaç"], 60)
        fig = plotter.survival_plot(zaman, olay, group=grup)
        assert len(fig.axes[0].lines) >= 2

    def test_control_chart(self, plotter):
        rng = np.random.default_rng(9)
        vals = rng.normal(100, 2, 50)
        vals[22] = 115.0  # ihlal
        fig = plotter.control_chart(vals, subgroup_size=5)
        assert len(fig.axes[0].lines) >= 3  # merkez + 2 limit

    def test_residual(self, plotter):
        rng = np.random.default_rng(10)
        fitted = rng.normal(0, 1, 60)
        fig = plotter.residual_plot(fitted, rng.normal(0, 0.3, 60))
        assert len(fig.axes[0].lines) >= 1

    def test_bland_altman_az_veri_hatasi(self, plotter):
        with pytest.raises(ValueError):
            plotter.bland_altman_plot([1.0, 2.0], [1.1, 2.1])


class TestDigerGrafikler:
    def test_hexbin(self, plotter):
        rng = np.random.default_rng(11)
        fig = plotter.hexbin_plot(rng.normal(size=300),
                                  rng.normal(size=300))
        assert fig.axes[0].get_title() == "Hexbin (2B Yoğunluk)"

    def test_stacked_area(self, plotter):
        fig = plotter.stacked_area(
            labels=[2020, 2021, 2022],
            series_dict={"buğday": [10.0, 12.0, 11.0],
                         "arpa": [5.0, 6.0, 7.0]})
        assert len(fig.axes[0].collections) >= 2

    def test_stacked_area_bos_hatasi(self, plotter):
        with pytest.raises(ValueError):
            plotter.stacked_area(labels=[], series_dict={})

    def test_growth_curve(self, plotter):
        t = np.linspace(1, 30, 20)
        y = 100 / (1 + np.exp(-0.25 * (t - 15))) + np.random.default_rng(
            12).normal(0, 1.5, 20)
        fig = plotter.growth_curve_plot(t, y, model="logistic")
        assert "Lojistik" in fig.axes[0].get_title() or \
               "büyüme" in fig.axes[0].get_title().lower()

    def test_slope(self, plotter):
        fig = plotter.slope_plot(before=[1.0, 2.0, 3.0],
                                 after=[2.0, 1.5, 4.0],
                                 labels=["a", "b", "c"])
        assert len(fig.axes[0].lines) >= 3

    def test_slope_uzunluk_hatasi(self, plotter):
        with pytest.raises(ValueError):
            plotter.slope_plot(before=[1.0, 2.0], after=[2.0])
