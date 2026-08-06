"""Agrista etkileşimli (Plotly) grafik ve dashboard testleri."""
import numpy as np
import pandas as pd
import pytest
from plotly import graph_objects as go

from agrista.viz.interactive import (
    interactive_scatter, interactive_line, interactive_bar,
    interactive_heatmap, interactive_box, interactive_histogram)


def _veri(seed=2, n=80):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "sulama": rng.uniform(3000, 12000, n),
        "gubre": rng.uniform(50, 400, n),
        "verim": rng.normal(500, 60, n),
        "bolge": rng.choice(["ic", "ege"], n),
    })


class TestEtkilesimliGrafikler:
    def test_scatter(self):
        fig = interactive_scatter(_veri(), x="sulama", y="verim",
                                  color="bolge")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_line(self):
        df = pd.DataFrame({"gun": range(10),
                           "deger": np.arange(10.0)})
        fig = interactive_line(df, x="gun", y="deger")
        assert isinstance(fig.data[0], go.Scatter)

    def test_bar(self):
        df = _veri().groupby("bolge", as_index=False)["verim"].mean()
        fig = interactive_bar(df, x="bolge", y="verim")
        assert isinstance(fig.data[0], go.Bar)

    def test_heatmap(self):
        fig = interactive_heatmap(_veri())
        assert isinstance(fig.data[0], go.Heatmap)

    def test_box(self):
        fig = interactive_box(_veri(), x="bolge", y="verim")
        assert isinstance(fig.data[0], go.Box)

    def test_histogram(self):
        fig = interactive_histogram(_veri(), column="verim")
        assert isinstance(fig.data[0], go.Histogram)

    def test_eksik_sutun_hatasi(self):
        with pytest.raises(ValueError):
            interactive_scatter(_veri(), x="yok", y="verim")
