"""Premium Program GEE testleri."""
import numpy as np
import pandas as pd
import pytest

from agrista.analysis import gee_model


def _clustered_df(seed=3, n_groups=20, per=8):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        ortak = rng.normal(0, 1.5)
        for t in range(per):
            x = rng.normal(0, 1)
            rows.append({"grup": g, "zaman": t, "x": x,
                         "y": 2.0 + 1.5 * x + ortak + rng.normal(0, 0.5)})
    return pd.DataFrame(rows)


class TestGeeModel:
    def test_gaussian_exchangeable_katsayi(self):
        df = _clustered_df()
        res = gee_model(df, response="y", covariates=["x"],
                        group_col="grup", cov_struct="exchangeable")
        assert res["model"] == "GEE"
        assert res["coefficients"]["x"]["coefficient"] == pytest.approx(1.5,
                                                                         abs=0.2)
        assert res["n_groups"] == 20
        assert res["n_obs"] == 160
        assert res["converged"] is True

    def test_robust_se_pozitif(self):
        df = _clustered_df()
        res = gee_model(df, response="y", covariates=["x"],
                        group_col="grup", cov_struct="independent")
        for ad in ("x", "Intercept"):
            assert res["coefficients"][ad]["std_err"] > 0

    def test_binomial_aile(self):
        rng = np.random.default_rng(7)
        df = pd.DataFrame({
            "grup": np.repeat(np.arange(30), 6),
            "x": rng.normal(0, 1, 180),
        })
        logit = 0.8 * df["x"]
        df["y"] = (rng.uniform(size=180) < 1 / (1 + np.exp(-logit))).astype(int)
        res = gee_model(df, response="y", covariates=["x"],
                        group_col="grup", family="binomial")
        assert res["family"] == "binomial"
        assert res["coefficients"]["x"]["coefficient"] > 0

    def test_autoregressive_zaman_sutunu_zorunlu(self):
        df = _clustered_df()
        with pytest.raises(ValueError):
            gee_model(df, response="y", covariates=["x"],
                      group_col="grup", cov_struct="autoregressive")

    def test_autoregressive_zaman_ile_calisir(self):
        df = _clustered_df()
        res = gee_model(df, response="y", covariates=["x"],
                        group_col="grup", cov_struct="autoregressive",
                        time_col="zaman")
        assert res["cov_struct"] == "autoregressive"

    def test_tek_grup_hatasi(self):
        df = _clustered_df().query("grup == 0")
        with pytest.raises(ValueError):
            gee_model(df, response="y", covariates=["x"], group_col="grup")

    def test_binomial_ikili_olmayan_yanit_hatasi(self):
        df = _clustered_df()
        with pytest.raises(ValueError):
            gee_model(df, response="y", covariates=["x"],
                      group_col="grup", family="binomial")

    def test_bilinmeyen_aile_hatasi(self):
        df = _clustered_df()
        with pytest.raises(ValueError):
            gee_model(df, response="y", covariates=["x"],
                      group_col="grup", family="negbin")
