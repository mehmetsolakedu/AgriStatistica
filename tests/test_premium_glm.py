"""Premium Program GLM (tek değişkenli + tekrarlı ölçüm) testleri."""
import numpy as np
import pandas as pd
import pytest

from agrista.analysis import anova_one_way, glm_univariate


def _two_factor_df(seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    effects = {"A": 2.0, "B": 0.0}
    for grup in ["A", "B"]:
        for blok in ["x", "y", "z"]:
            for _ in range(5):
                rows.append({
                    "grup": grup,
                    "blok": blok,
                    "verim": 10 + effects[grup]
                             + {"x": 0, "y": 1, "z": 2}[blok]
                             + rng.normal(0, 0.4),
                })
    return pd.DataFrame(rows)


class TestGlmUnivariate:
    def test_tek_faktor_anova_ile_ayni_sonuc(self):
        df = _two_factor_df()
        glm = glm_univariate(df, response="verim", between_factors=["grup"],
                             posthoc=None)
        klasik = anova_one_way(df.loc[df["grup"] == "A", "verim"],
                               df.loc[df["grup"] == "B", "verim"])
        assert glm["anova_table"][0]["f_value"] == pytest.approx(
            klasik["f_statistic"], rel=1e-6)
        assert glm["anova_table"][0]["p_value"] == pytest.approx(
            klasik["p_value"], rel=1e-6)

    def test_tip3_faktoryel_ve_efekt_buyuklugu(self):
        df = _two_factor_df()
        res = glm_univariate(df, response="verim",
                             between_factors=["grup", "blok"], posthoc=None)
        assert res["ss_type"] == 3
        sources = {r["source"] for r in res["anova_table"]}
        assert any("grup" in s for s in sources)
        assert any("blok" in s for s in sources)
        eta = res["effect_sizes"]["C(grup, Sum)"]
        assert 0.5 < eta <= 1.0  # büyük tasarlanmış etki

    def test_kovaryeteli_model(self):
        df = _two_factor_df()
        df["nem"] = np.linspace(0, 1, len(df)) + 0.01
        res = glm_univariate(df, response="verim", between_factors=["grup"],
                             covariates=["nem"], posthoc=None)
        assert res["n_obs"] == len(df)
        assert 0 <= res["r_squared"] <= 1

    def test_posthoc_tukey_eklenir(self):
        df = _two_factor_df()
        res = glm_univariate(df, response="verim", between_factors=["grup"],
                             posthoc="tukey")
        assert res["posthoc"]["test"] == "Tukey HSD"
        assert len(res["posthoc"]["comparisons"]) == 1

    def test_iki_faktorde_posthoc_none(self):
        df = _two_factor_df()
        res = glm_univariate(df, response="verim",
                             between_factors=["grup", "blok"],
                             posthoc="tukey")
        assert res["posthoc"] is None

    def test_eksik_sutun_hatasi(self):
        df = _two_factor_df()
        with pytest.raises(ValueError):
            glm_univariate(df, response="yok", between_factors=["grup"])

    def test_tek_duzeyli_faktor_hatasi(self):
        df = _two_factor_df()
        df["sabit"] = "a"
        with pytest.raises(ValueError):
            glm_univariate(df, response="verim", between_factors=["sabit"])
