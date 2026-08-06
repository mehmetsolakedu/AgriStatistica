"""Agrista tema sistemi ve gelişmiş dışa aktarım testleri."""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
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


class TestDisaAktarim:
    def test_save_png_ve_svg(self, tmp_path):
        p = AgristaPlotter()
        fig = p.histogram(np.random.default_rng(1).normal(0, 1, 50))
        for uzanti in ("png", "svg", "pdf"):
            hedef = tmp_path / f"g.{uzanti}"
            p.save(str(hedef), fig=fig)
            assert hedef.exists() and hedef.stat().st_size > 0

    def test_save_tema_dpi(self, tmp_path):
        p = AgristaPlotter(theme="yayin")
        fig = p.histogram(np.random.default_rng(2).normal(0, 1, 50))
        hedef = tmp_path / "y.png"
        p.save(str(hedef), fig=fig)
        assert hedef.exists()

    def test_save_multi(self, tmp_path):
        p = AgristaPlotter()
        f1 = p.histogram(np.arange(10, dtype=float))
        f2 = p.bar_chart(["a", "b"], [1.0, 2.0])
        yollar = p.save_multi(str(tmp_path / "cok"), [f1, f2],
                              fmts=("png", "svg"))
        assert len(yollar) == 4
        for yol in yollar:
            import pathlib
            assert pathlib.Path(yol).exists()

    def test_export_html(self, tmp_path):
        p = AgristaPlotter()
        fig = p.histogram(np.arange(10, dtype=float))
        hedef = tmp_path / "g.html"
        p.export_html(fig, str(hedef), title="Deneme Grafik")
        icerik = hedef.read_text(encoding="utf-8")
        assert "Deneme Grafik" in icerik
        assert "data:image/png;base64" in icerik
