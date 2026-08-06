"""Agrista GUI analiz kaydı ve adaptör testleri."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def df():
    rng = np.random.default_rng(3)
    n = 60
    grup = rng.choice(["A", "B"], n)
    etki = np.where(grup == "A", 0.0, 2.0)
    return pd.DataFrame({
        "grup": grup,
        "verim": 10 + etki + rng.normal(0, 1, n),
        "nem": rng.uniform(20, 80, n),
        "once": rng.normal(5, 1, n),
        "sonra": rng.normal(6, 1, n),
        "zaman": np.tile([1, 2, 3, 4], 15).astype(float),
        "siddet": rng.uniform(0, 100, n),
        "gercek": rng.integers(0, 2, n),
        "skor": rng.uniform(0, 1, n),
        "olay": rng.integers(0, 2, n),
        "sure": rng.exponential(10, n),
    })


class TestKayitButunlugu:
    def test_onalti_analiz(self):
        from agrista.gui.registry import REGISTRY
        assert len(REGISTRY) == 16

    def test_menu_denkligi(self):
        """Kayıttaki her (kategori, etiket) CLI menüsünde var olmalı."""
        from agrista.cli import _build_menu_structure
        from agrista.gui.registry import REGISTRY
        cli = {baslik: {etiket for etiket, _ in islemler}
               for baslik, islemler in _build_menu_structure()}
        for spec in REGISTRY:
            assert spec.label in cli.get(spec.menu_category, set()), \
                f"Menüde yok: {spec.menu_category} / {spec.label}"

    def test_key_benzersiz(self):
        from agrista.gui.registry import REGISTRY
        keyler = [s.key for s in REGISTRY]
        assert len(keyler) == len(set(keyler))


class TestAdaptorler:
    def _calistir(self, key, df, p):
        from agrista.gui.registry import REGISTRY
        spec = next(s for s in REGISTRY if s.key == key)
        return spec.run(df, p)

    def test_betimsel(self, df):
        res = self._calistir("betimsel", df, {"kolonlar": "verim,nem"})
        assert "verim" in res and "count" in res["verim"]

    def test_frekans(self, df):
        res = self._calistir("frekans", df, {"kolonlar": "grup"})
        assert res  # dolu dict

    def test_capraz(self, df):
        res = self._calistir("capraz", df,
                             {"satir": "grup", "sutun": "gercek"})
        assert res

    def test_tek_orneklem(self, df):
        res = self._calistir("tek_orneklem_t", df,
                             {"kolon": "verim", "deger": 10.0})
        assert "p_value" in res

    def test_bagimsiz_t(self, df):
        res = self._calistir("bagimsiz_t", df,
                             {"yanit": "verim", "grup": "grup"})
        assert res["p_value"] < 0.05  # etki 2.0 tasarlandı

    def test_eslestirilmis_t(self, df):
        res = self._calistir("eslestirilmis_t", df,
                             {"once": "once", "sonra": "sonra"})
        assert "p_value" in res

    def test_anova_tukey(self, df):
        res = self._calistir("anova_tukey", df,
                             {"yanit": "verim", "grup": "grup"})
        assert "anova" in res and "tukey" in res

    def test_glm(self, df):
        res = self._calistir("glm_univariate", df,
                             {"yanit": "verim", "faktorler": "grup"})
        assert res["model"] == "GLM Univariate"

    def test_korelasyon(self, df):
        res = self._calistir("korelasyon", df, {})
        assert res

    def test_roc(self, df):
        res = self._calistir("roc", df,
                             {"gercek": "gercek", "skor": "skor"})
        assert "auc" in res

    def test_kaplan_meier(self, df):
        res = self._calistir("kaplan_meier", df,
                             {"zaman": "sure", "olay": "olay"})
        assert "survival" in res

    def test_audpc(self, df):
        res = self._calistir("audpc", df,
                             {"zaman": "zaman", "siddet": "siddet"})
        assert res

    def test_eksik_sutun_hatasi(self, df):
        with pytest.raises(ValueError):
            self._calistir("betimsel", df, {"kolonlar": "yok"})
