"""Premium Program GLM (tek değişkenli + tekrarlı ölçüm) testleri."""
import numpy as np
import pandas as pd
import pytest

from agrista.analysis import (anova_one_way, glm_repeated_measures,
                              glm_univariate)


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


def _repeated_df(seed=5):
    rng = np.random.default_rng(seed)
    rows = []
    for denek in range(12):
        taban = rng.normal(0, 1.0)
        for t, ad in enumerate(["t1", "t2", "t3", "t4"]):
            rows.append({"denek": f"d{denek}", "zaman": ad,
                         "skor": 5 + 0.8 * t + taban + rng.normal(0, 0.3)})
    return pd.DataFrame(rows).pivot(index="denek", columns="zaman",
                                    values="skor").reset_index()


class TestGlmRepeatedMeasures:
    def test_zaman_efekti_anlamli(self):
        df = _repeated_df()
        res = glm_repeated_measures(df, response_cols=["t1", "t2", "t3", "t4"],
                                    subject_col="denek")
        assert res["within_effect"]["p_value"] < 0.05
        assert res["n_subjects"] == 12

    def test_mauchly_ve_epsilon_araliklari(self):
        df = _repeated_df()
        res = glm_repeated_measures(df, response_cols=["t1", "t2", "t3", "t4"],
                                    subject_col="denek")
        assert 0 < res["mauchly"]["w"] <= 1
        assert 0 <= res["mauchly"]["p_value"] <= 1
        eps = res["epsilon"]
        assert 0 < eps["greenhouse_geisser"] <= 1
        assert 0 < eps["huynh_feldt"] <= 1

    def test_duzeltilmis_p_degerleri(self):
        df = _repeated_df()
        res = glm_repeated_measures(df, response_cols=["t1", "t2", "t3", "t4"],
                                    subject_col="denek")
        for anahtar in ("greenhouse_geisser", "huynh_feldt"):
            corr = res["corrected"][anahtar]
            assert corr["df1"] <= res["within_effect"]["df1"]
            assert 0 <= corr["p_value"] <= 1

    def test_az_kosul_hatasi(self):
        df = _repeated_df()
        with pytest.raises(ValueError):
            glm_repeated_measures(df, response_cols=["t1", "t2"],
                                  subject_col="denek")

    def test_tek_denek_hatasi(self):
        df = _repeated_df().head(1)
        with pytest.raises(ValueError):
            glm_repeated_measures(df, response_cols=["t1", "t2", "t3", "t4"],
                                  subject_col="denek")
