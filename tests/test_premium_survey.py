"""Premium Program Karmaşık Örneklem (Taylor linearizasyonu) testleri."""
import numpy as np
import pandas as pd
import pytest

from agrista.survey import (survey_design, svy_mean, svy_total, svy_ratio,
                            survey_logistic)


def _hand_design():
    """Elle hesaplanabilir küçük tasarım: 2 tabaka × 2 PSU, ağırlık 1."""
    return pd.DataFrame({
        "tabaka": ["A", "A", "B", "B"],
        "psu": ["a1", "a2", "b1", "b2"],
        "y": [2.0, 4.0, 6.0, 8.0],
        "x": [1.0, 1.0, 2.0, 2.0],
    })


class TestSurveyDesign:
    def test_tasarim_ozeti(self):
        d = survey_design(_hand_design(), id_col="psu", strata_col="tabaka")
        assert d["n_psu"] == 4
        assert d["n_strata"] == 2

    def test_tek_psulu_tabaka_hatasi(self):
        df = _hand_design().iloc[[0, 2]]  # her tabakada tek PSU
        d = survey_design(df, id_col="psu", strata_col="tabaka")
        with pytest.raises(ValueError, match="Tek PSU"):
            svy_mean(d, "y")

    def test_eksik_sutun_hatasi(self):
        with pytest.raises(ValueError):
            survey_design(_hand_design(), id_col="yok")


class TestSvyMean:
    def test_elle_hesaplanan_deger(self):
        d = survey_design(_hand_design(), id_col="psu", strata_col="tabaka")
        res = svy_mean(d, "y")
        assert res["estimate"] == pytest.approx(5.0)
        # lin = w*(y-5)/4 -> [-0.75,-0.25,0.25,0.75]; var = 0.25+0.25 = 0.5
        assert res["std_err"] == pytest.approx(np.sqrt(0.5), rel=1e-9)
        assert res["ci_lower"] < 5.0 < res["ci_upper"]

    def test_deff_basit_sansimana_bir(self):
        rng = np.random.default_rng(12)
        df = pd.DataFrame({"y": rng.normal(10, 2, 40)})  # PSU yok -> SRS
        d = survey_design(df)
        res = svy_mean(d, "y")
        assert res["design_effect"] == pytest.approx(1.0, rel=1e-9)


class TestSvyTotalAndRatio:
    def test_toplam(self):
        d = survey_design(_hand_design(), id_col="psu", strata_col="tabaka")
        res = svy_total(d, "y")
        assert res["estimate"] == pytest.approx(20.0)
        assert res["std_err"] > 0
        assert res["design_effect"] is None

    def test_oran(self):
        d = survey_design(_hand_design(), id_col="psu", strata_col="tabaka")
        res = svy_ratio(d, numerator="y", denominator="x")
        assert res["estimate"] == pytest.approx(20.0 / 6.0)
        assert res["std_err"] >= 0

    def test_agirlikli_ortalama(self):
        df = _hand_design().assign(w=[2.0, 1.0, 1.0, 1.0])
        d = survey_design(df, weight_col="w", id_col="psu",
                          strata_col="tabaka")
        res = svy_mean(d, "y")
        beklenen = (2 * 2 + 1 * 4 + 1 * 6 + 1 * 8) / 5.0
        assert res["estimate"] == pytest.approx(beklenen)


class TestSurveyLogistic:
    def test_katsayilar_ve_cluster_se(self):
        rng = np.random.default_rng(8)
        psu = np.repeat(np.arange(30), 10)
        x = rng.normal(0, 1, 300)
        p = 1 / (1 + np.exp(-1.2 * x))
        y = (rng.uniform(size=300) < p).astype(int)
        df = pd.DataFrame({"psu": psu, "x": x, "y": y})
        d = survey_design(df, id_col="psu")
        res = survey_logistic(d, response="y", predictors=["x"])
        assert res["coefficients"]["x"]["coefficient"] == pytest.approx(1.2,
                                                                        abs=0.4)
        assert res["coefficients"]["x"]["std_err"] > 0
        assert res["n_psu"] == 30

    def test_psu_zorunlu(self):
        df = _hand_design().assign(y=[0, 1, 1, 0])
        d = survey_design(df)
        with pytest.raises(ValueError):
            survey_logistic(d, response="y", predictors=["x"])

    def test_ikili_olmayan_yanit(self):
        d = survey_design(_hand_design(), id_col="psu")
        with pytest.raises(ValueError):
            survey_logistic(d, response="y", predictors=["x"])
