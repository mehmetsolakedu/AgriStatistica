"""Agrista tema sistemi ve gelişmiş dışa aktarım testleri."""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np  # noqa: F401 (Task 3 dışa aktarım testlerinde kullanılır)
import pytest

from agrista.viz import AgristaPlotter
from agrista.viz.themes import THEMES, apply_theme


@pytest.fixture(autouse=True)
def _kapla():
    yield
    AgristaPlotter.close()


class TestTemalar:
    def test_dort_tema_tanimli(self):
        assert set(THEMES) == {"agrista", "yayin", "minimal", "karanlik"}

    def test_apply_theme_icerik(self):
        t = apply_theme("yayin")
        assert t["dpi"] == 300
        assert len(t["palette"]) >= 4
        assert isinstance(t["rc"], dict)

    def test_bilinmeyen_tema_hatasi(self):
        with pytest.raises(ValueError):
            apply_theme("olmayan")

    def test_plotter_tema_uygular(self):
        p = AgristaPlotter(theme="yayin")
        assert p.theme == "yayin"
        assert plt.rcParams["savefig.dpi"] == 300

    def test_plotter_varsayilan_tema(self):
        p = AgristaPlotter()
        assert p.theme == "agrista"

    def test_eski_style_parametresi_calismaya_devam_eder(self):
        p = AgristaPlotter(style="darkgrid")
        assert p.style == "darkgrid"
