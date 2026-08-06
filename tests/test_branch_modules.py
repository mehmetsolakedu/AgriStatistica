"""P1/P2 testleri — engineering, animal, sensory, spatial, economics modülleri."""

import numpy as np
import pandas as pd
import pytest

from agrista.engineering import (
    rsm_ccd,
    rsm_bbd,
    rsm_fit,
    find_optimum,
    taguchi_design,
    sn_ratio,
    taguchi_analyze,
)
from agrista.animal import mixed_model, fit_wood, lactation_summary
from agrista.models import GrowthModel
from agrista.sensory import kendall_w, hedonic_summary, panel_anova
from agrista.spatial import semivariogram, idw_interpolation, spatial_summary
from agrista.economics import dea_efficiency, adoption_logit, partial_budget


class TestEngineering:
    def test_ccd_structure(self):
        df = rsm_ccd({"a": (0, 10), "b": (0, 10)})
        # 2^2 faktöriyel + 4 aksiyel + 3 merkez = 11
        assert len(df) == 11
        assert (df["tip"] == "merkez").sum() == 3
        # Merkez noktaları gerçek aralığın ortasında
        center_rows = df[df["tip"] == "merkez"]
        assert np.allclose(center_rows["a"], 5.0)

    def test_bbd_structure(self):
        df = rsm_bbd({"a": (0, 10), "b": (0, 10), "c": (0, 10)})
        assert len(df) == 12 + 3  # C(3,2)*4 + merkez

    def test_rsm_fit_and_optimum_recovery(self):
        rng = np.random.default_rng(50)
        ccd = rsm_ccd({"hiz": (100, 300), "derinlik": (2, 8)})
        # Gerçek optimum (200, 5) merkezde
        ccd["y"] = (-((ccd["hiz"] - 200) ** 2) / 2000
                    - ((ccd["derinlik"] - 5) ** 2) / 2
                    + 50 + rng.normal(0, 0.2, len(ccd)))
        fit = rsm_fit(ccd, "y", ["hiz", "derinlik"])
        assert fit["r_squared"] > 0.99
        assert "hiz^2" in fit["significant_terms"]

        opt = find_optimum(fit, {"hiz": (100, 300), "derinlik": (2, 8)})
        assert opt["optimal_values"]["hiz"] == pytest.approx(200, abs=10)
        assert opt["optimal_values"]["derinlik"] == pytest.approx(5, abs=1)

    def test_taguchi_design_valid(self):
        df = taguchi_design({"a": [1, 2, 3], "b": [10, 20, 30]})
        assert len(df) == 9
        # Her seviye her faktörde 3 kez görünmeli
        assert df["a"].value_counts().tolist() == [3, 3, 3]

    def test_taguchi_wrong_levels_raises(self):
        with pytest.raises(ValueError):
            taguchi_design({"a": [1, 2], "b": [1, 2, 3]})

    def test_sn_ratio_goals(self):
        # smaller-the-better: küçük değerler daha yüksek S/N verir
        assert sn_ratio([2, 2, 2], "smaller") > sn_ratio([5, 5, 5], "smaller")
        assert sn_ratio([5, 5, 5], "larger") > sn_ratio([2, 2, 2], "larger")

    def test_taguchi_analyze_selects_factor(self):
        rng = np.random.default_rng(51)
        df = taguchi_design({"isi": [100, 150, 200], "basinc": [1, 2, 3]})
        df["y"] = np.where(df["isi"] == 200, 10.0, 5.0) + rng.normal(0, 0.1, 9)
        result = taguchi_analyze(df, "y", ["isi", "basinc"])
        assert result["optimal_levels"]["isi"] == 200
        assert result["effect_ranking"][0] == "isi"

    def test_taguchi_analyze_with_reps(self):
        rng = np.random.default_rng(60)
        df = taguchi_design({"isi": [100, 150, 200], "basinc": [1, 2, 3]})
        base = np.where(df["isi"] == 200, 10.0, 5.0)
        for rep in (1, 2, 3):
            df[f"yanit_{rep}"] = base + rng.normal(0, 0.1, 9)
        result = taguchi_analyze(df, "y", ["isi", "basinc"], goal="larger", n_reps=3)
        assert result["optimal_levels"]["isi"] == 200

    def test_taguchi_analyze_missing_reps_raises(self):
        df = taguchi_design({"isi": [100, 150, 200], "basinc": [1, 2, 3]})
        df["y"] = 1.0
        with pytest.raises(ValueError):
            taguchi_analyze(df, "y", ["isi"], n_reps=3)


class TestAnimal:
    def test_mixed_model_detects_fixed_effect(self):
        rng = np.random.default_rng(52)
        rows = []
        for hay in range(1, 13):
            base = rng.normal(20, 2)
            for week in range(1, 9):
                yem = "A" if week % 2 else "B"
                rows.append({"hayvan": f"H{hay}", "yem": yem,
                             "sut": base + (1.5 if yem == "A" else 0)
                                    + rng.normal(0, 0.4)})
        result = mixed_model(pd.DataFrame(rows), "sut", ["yem"], groups_col="hayvan")
        assert result["converged"] is True
        assert result["n_groups"] == 12
        assert result["fixed_effects"]["yem[T.B]"]["significant_at_005"] is True

    def test_wood_model_recovery(self):
        rng = np.random.default_rng(53)
        t = np.arange(1, 41, 3)
        true_a, true_b, true_c = 25.0, 0.25, 0.05
        y = true_a * t ** true_b * np.exp(-true_c * t) + rng.normal(0, 0.3, len(t))
        result = fit_wood(t, y)
        assert result["r_squared"] > 0.99
        assert result["peak_time"] == pytest.approx(true_b / true_c, rel=0.2)
        assert result["persistency"] == pytest.approx(1 / true_c, rel=0.2)

    def test_lactation_summary(self):
        records = [20, 25, 28, 27, 24, 21, 18, 15, 12, 10]
        result = lactation_summary(records, interval_days=30)
        assert result["peak_yield"] == 28
        assert result["n_controls"] == 10
        assert result["total_yield_estimate"] > 0

    def test_growth_model_monomolecular(self):
        rng = np.random.default_rng(54)
        t = np.linspace(0, 50, 40)
        y = 80 * (1 - np.exp(-0.08 * t)) + rng.normal(0, 1, len(t))
        model = GrowthModel()
        params = model.fit_monomolecular(t, y)
        assert params["A"] == pytest.approx(80, rel=0.15)
        assert params["k"] == pytest.approx(0.08, rel=0.3)
        pred = model.predict(np.array([50.0]))
        assert pred[0] > 70


class TestSensory:
    def test_kendall_w_perfect_agreement(self):
        mat = pd.DataFrame({"p1": [5, 3, 1], "p2": [5, 4, 1], "p3": [4, 3, 1]})
        result = kendall_w(mat)
        assert result["kendall_w"] == pytest.approx(1.0)
        assert result["agreement_significant"] is True

    def test_kendall_w_random_low(self):
        rng = np.random.default_rng(55)
        mat = pd.DataFrame(rng.permutation(60).reshape(12, 5))
        result = kendall_w(mat)
        assert result["kendall_w"] < 0.5

    @pytest.fixture
    def panel_df(self):
        rng = np.random.default_rng(56)
        rows = []
        for p in range(10):
            bias = rng.normal(0, 0.3)
            for s, base in [("X", 7.0), ("Y", 5.0), ("Z", 3.0)]:
                rows.append({"ornek": s, "panelist": f"P{p}",
                             "puan": base + bias + rng.normal(0, 0.3)})
        return pd.DataFrame(rows)

    def test_hedonic_summary_ranking(self, panel_df):
        result = hedonic_summary(panel_df, "ornek", "panelist", "puan")
        assert result["preference_ranking"] == ["X", "Y", "Z"]
        assert result["friedman_test"]["significant_at_005"] is True

    def test_panel_anova_significant(self, panel_df):
        result = panel_anova(panel_df, "puan", "ornek", "panelist")
        assert result["sample_effect"]["significant_at_005"] is True


class TestSpatial:
    @pytest.fixture
    def spatial_data(self):
        rng = np.random.default_rng(57)
        x = rng.uniform(0, 100, 60)
        y = rng.uniform(0, 100, 60)
        z = 5 + 0.05 * x + rng.normal(0, 0.4, 60)
        return x, y, z

    def test_semivariogram_increases_with_distance(self, spatial_data):
        x, y, z = spatial_data
        result = semivariogram(x, y, z, n_lags=8)
        assert len(result["lags"]) == len(result["semivariance"])
        gamma = result["semivariance"]
        # Mekânsal bağımlılık: kısa mesafe varyansı < genel varyans
        assert gamma[0] < result["sill"]
        assert result["spatial_dependence_ratio"] > 0.25

    def test_semivariogram_too_few_points_raises(self):
        with pytest.raises(ValueError):
            semivariogram([0, 1, 2], [0, 1, 2], [1, 2, 3])

    def test_idw_exact_at_known_points(self):
        grid = idw_interpolation([0, 10], [0, 10], [1.0, 5.0],
                                 np.array([0.0]), np.array([0.0]))
        assert grid[0, 0] == pytest.approx(1.0)

    def test_idw_midpoint_average(self):
        grid = idw_interpolation([0, 10], [0, 0], [0.0, 10.0],
                                 np.array([5.0]), np.array([0.0]))
        assert grid[0, 0] == pytest.approx(5.0)

    def test_spatial_summary(self, spatial_data):
        x, y, z = spatial_data
        result = spatial_summary(x, y, z)
        assert result["n_points"] == 60
        assert "variogram" in result
        assert result["cv_pct"] > 0


class TestEconomics:
    def test_dea_identifies_efficient(self):
        inputs = pd.DataFrame({"arazi": [10, 20, 15, 12, 30],
                               "emek": [5, 8, 6, 4, 10]},
                              index=["C1", "C2", "C3", "C4", "C5"])
        outputs = pd.DataFrame({"urun": [100, 150, 140, 130, 200]},
                               index=["C1", "C2", "C3", "C4", "C5"])
        result = dea_efficiency(inputs, outputs)
        assert 0 < result["mean_efficiency"] <= 1
        for eff in result["efficiencies"].values():
            assert 0 <= eff <= 1.0 + 1e-9
        assert len(result["efficient_dmus"]) >= 1

    def test_dea_bcc_not_worse_than_ccr(self):
        rng = np.random.default_rng(58)
        n = 8
        idx = [f"F{i}" for i in range(n)]
        inputs = pd.DataFrame({"girdi": rng.uniform(5, 30, n)}, index=idx)
        outputs = pd.DataFrame({"cikti": rng.uniform(50, 200, n)}, index=idx)
        ccr = dea_efficiency(inputs, outputs, model="CCR")
        bcc = dea_efficiency(inputs, outputs, model="BCC")
        for name in idx:
            assert bcc["efficiencies"][name] >= ccr["efficiencies"][name] - 1e-6

    def test_dea_index_mismatch_raises(self):
        inputs = pd.DataFrame({"a": [1, 2]}, index=["x", "y"])
        outputs = pd.DataFrame({"b": [3, 4]}, index=["p", "q"])
        with pytest.raises(ValueError):
            dea_efficiency(inputs, outputs)

    def test_adoption_logit_recovers_predictors(self):
        rng = np.random.default_rng(59)
        n = 300
        egitim = rng.normal(8, 2, n)
        arazi = rng.uniform(5, 50, n)
        p = 1 / (1 + np.exp(-(0.3 * egitim + 0.05 * arazi - 4)))
        df = pd.DataFrame({"benimseme": rng.binomial(1, p),
                           "egitim": egitim, "arazi": arazi})
        result = adoption_logit(df, "benimseme", ["egitim", "arazi"])
        assert "egitim" in result["significant_predictors"]
        assert result["odds_ratios"]["egitim"] > 1
        assert result["classification_accuracy"] > 0.65

    def test_adoption_logit_non_binary_raises(self):
        df = pd.DataFrame({"y": [0, 1, 2, 0, 1], "x": [1.0, 2, 3, 4, 5]})
        with pytest.raises(ValueError):
            adoption_logit(df, "y", ["x"])

    def test_partial_budget(self):
        result = partial_budget(5000, 1000, 3000)
        assert result["net_change"] == 3000
        assert result["profitable"] is True
        assert result["benefit_cost_ratio"] == pytest.approx(2.0)
