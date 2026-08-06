"""Premium Program GLMM (PQL) testleri."""
import numpy as np
import pandas as pd
import pytest

from agrista.animal import mixed_model
from agrista.models import glmm


def _continuous_df(seed=4, n_groups=20, per=10):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        b = rng.normal(0, 1.2)
        for _ in range(per):
            x = rng.normal(0, 1)
            rows.append({"grup": g, "x": x,
                         "y": 3.0 + 1.5 * x + b + rng.normal(0, 0.5)})
    return pd.DataFrame(rows)


def _binary_glmm_df(seed=9, n_groups=25, per=12):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        b = rng.normal(0, 0.7)
        for _ in range(per):
            x = rng.normal(0, 1)
            p = 1 / (1 + np.exp(-(0.9 * x + b)))
            rows.append({"grup": g, "x": x,
                         "y": int(rng.uniform() < p)})
    return pd.DataFrame(rows)


def _poisson_glmm_df(seed=6, n_groups=25, per=12):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        b = rng.normal(0, 0.4)
        for _ in range(per):
            x = rng.normal(0, 1)
            lam = np.exp(0.8 + 0.5 * x + b)
            rows.append({"grup": g, "x": x,
                         "y": int(rng.poisson(lam))})
    return pd.DataFrame(rows)


class TestGlmmGaussian:
    def test_animal_mixed_model_ile_ayni(self):
        df = _continuous_df()
        res = glmm(df, response="y", fixed_effects=["x"], groups_col="grup",
                   family="gaussian")
        ref = mixed_model(df, response_col="y", fixed_effects=["x"],
                          groups_col="grup")
        assert res["method"] == "REML"
        for ad in res["fixed_effects"]:
            assert res["fixed_effects"][ad]["coefficient"] == pytest.approx(
                ref["fixed_effects"][ad]["coefficient"], rel=1e-6)
        assert res["random_effects_variance"]["random_intercept"] == \
            pytest.approx(ref["random_effects_variance"]["random_intercept"],
                          rel=1e-6)
        assert res["aic"] == pytest.approx(ref["aic"], rel=1e-6)


class TestGlmmPql:
    def test_binomial_katsayi_yonu_ve_buyuklugu(self):
        df = _binary_glmm_df()
        res = glmm(df, response="y", fixed_effects=["x"], groups_col="grup",
                   family="binomial")
        assert res["method"] == "PQL"
        assert res["family"] == "binomial"
        assert res["fixed_effects"]["x"]["coefficient"] == pytest.approx(0.9,
                                                                         abs=0.45)
        assert res["converged"] is True
        assert res["aic"] is None

    def test_binomial_rastgele_etki_pozitif(self):
        df = _binary_glmm_df()
        res = glmm(df, response="y", fixed_effects=["x"], groups_col="grup",
                   family="binomial")
        assert res["random_effects_variance"]["random_intercept"] > 0

    def test_poisson_katsayi(self):
        df = _poisson_glmm_df()
        res = glmm(df, response="y", fixed_effects=["x"], groups_col="grup",
                   family="poisson")
        assert res["fixed_effects"]["x"]["coefficient"] == pytest.approx(0.5,
                                                                         abs=0.3)

    def test_wald_se_ve_p(self):
        df = _binary_glmm_df()
        res = glmm(df, response="y", fixed_effects=["x"], groups_col="grup",
                   family="binomial")
        c = res["fixed_effects"]["x"]
        assert c["std_err"] > 0
        assert 0 <= c["p_value"] <= 1
        assert abs(c["z_value"]) > 1.5


class TestGlmmErrors:
    def test_bilinmeyen_aile(self):
        df = _continuous_df()
        with pytest.raises(ValueError):
            glmm(df, response="y", fixed_effects=["x"], groups_col="grup",
                 family="negbin")

    def test_tek_grup_hatasi(self):
        df = _continuous_df().query("grup == 0")
        with pytest.raises(ValueError):
            glmm(df, response="y", fixed_effects=["x"], groups_col="grup")

    def test_grup_basina_tek_gozlem_hatasi(self):
        df = _continuous_df().groupby("grup").head(1)
        with pytest.raises(ValueError):
            glmm(df, response="y", fixed_effects=["x"], groups_col="grup")
