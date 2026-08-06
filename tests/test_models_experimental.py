"""agrista.models ve agrista.experimental modülleri testleri."""

import numpy as np
import pandas as pd
import pytest

from agrista.models import GrowthModel, YieldPredictionModel, RiskAnalysisModel
from agrista.experimental import ExperimentalDesign, FieldTrialAnalyzer


class TestGrowthModel:
    def test_logistic_fit_recovers_parameters(self):
        model = GrowthModel(model_type="logistic")
        rng = np.random.default_rng(5)
        t = np.linspace(0, 30, 60)
        y = model.logistic(t, 100, 0.3, 15) + rng.normal(0, 1.0, len(t))
        params = model.fit_logistic(t, y)
        assert params["K"] == pytest.approx(100, abs=5)
        assert params["r"] == pytest.approx(0.3, abs=0.05)
        assert params["t0"] == pytest.approx(15, abs=1)

    def test_predict_requires_fit(self):
        model = GrowthModel()
        with pytest.raises(ValueError):
            model.predict(5.0)

    def test_unknown_model_type_raises(self):
        model = GrowthModel(model_type="üstel")
        model.params = {"x": 1}
        with pytest.raises(ValueError):
            model.predict(1.0)


class TestYieldPredictionModel:
    def test_linear_prediction(self):
        model = YieldPredictionModel()
        model.intercept = 1.0
        model.add_factor("sulama", 0.001)
        model.add_factor("gubre", 0.005)
        result = model.predict({"sulama": 1000, "gubre": 100})
        assert result == pytest.approx(1.0 + 1.0 + 0.5)

    def test_missing_factor_raises(self):
        model = YieldPredictionModel()
        model.add_factor("sulama", 0.001)
        with pytest.raises(KeyError):
            model.predict({})

    def test_summary(self):
        model = YieldPredictionModel()
        model.intercept = 2.0
        model.add_factor("yagis", 0.002)
        summary = model.summary()
        assert summary["intercept"] == 2.0
        assert summary["coefficients"]["yagis"] == 0.002


class TestRiskAnalysis:
    def test_value_at_risk_below_mean(self):
        var95 = RiskAnalysisModel.value_at_risk(5.0, 1.0, 0.95)
        assert var95 < 5.0
        assert var95 == pytest.approx(5.0 - 1.6449, abs=0.01)

    def test_coefficient_of_variation(self):
        cv = RiskAnalysisModel.coefficient_of_variation([10, 12, 11, 9, 10])
        assert 0 < cv < 30

    def test_monte_carlo_structure(self):
        result = RiskAnalysisModel.monte_carlo_yield(5.0, 1.0, n_simulations=1000)
        assert result["mean"] == pytest.approx(5.0, abs=0.1)
        assert set(result["percentiles"].keys()) == {"90%", "95%", "99%"}
        assert result["percentiles"]["95%"]["var"] < result["percentiles"]["90%"]["var"]


class TestExperimentalDesign:
    def test_rcbd_structure(self):
        design = ExperimentalDesign.random_complete_block(4, 5)
        assert design["total_plots"] == 20
        assert len(design["assignments"]) == 5
        for block_assignments in design["assignments"].values():
            assert sorted(block_assignments) == ["T1", "T2", "T3", "T4"]

    def test_latin_square_valid(self):
        result = ExperimentalDesign.latin_square(4)
        square = result["square"]
        # Her satır ve sütunda her uygulama tam bir kez bulunmalı
        for row in square:
            assert sorted(row) == ["T1", "T2", "T3", "T4"]
        for col_idx in range(4):
            col = [square[r][col_idx] for r in range(4)]
            assert sorted(col) == ["T1", "T2", "T3", "T4"]

    def test_factorial_design(self):
        result = ExperimentalDesign.factorial_design({"a": [1, 2, 3], "b": ["x", "y"]})
        assert result["total_treatments"] == 6
        assert result["design"] == "Full Factorial (3 × 2)"
        assert len(result["dataframe"]) == 6

    def test_sample_size(self):
        result = ExperimentalDesign.sample_size_calculation(mean_diff=1.0, std_dev=2.0)
        assert result["n_per_group"] >= 2
        assert result["total_samples"] == 2 * result["n_per_group"]
        assert result["effect_size"] == pytest.approx(0.5)


class TestFieldTrialAnalyzer:
    @pytest.fixture
    def rcbd_data(self):
        # 3 uygulama, 4 blok; T3 belirgin şekilde yüksek verimli
        rng = np.random.default_rng(9)
        rows = []
        treatment_effects = {"T1": 5.0, "T2": 5.5, "T3": 8.0}
        block_effects = {"B1": 0.0, "B2": 0.5, "B3": -0.3, "B4": 0.2}
        for block, be in block_effects.items():
            for treatment, te in treatment_effects.items():
                rows.append({
                    "blok": block,
                    "uygulama": treatment,
                    "verim": te + be + rng.normal(0, 0.1),
                })
        return pd.DataFrame(rows)

    def test_rcbd_detects_treatment_effect(self, rcbd_data):
        analyzer = FieldTrialAnalyzer(rcbd_data)
        result = analyzer.rcbd_analysis("verim", "uygulama", "blok")
        assert result["p_value"] < 0.001
        assert result["significant_at_005"] is True
        assert result["degrees_of_freedom_treatment"] == 2
        means = {k: v["mean"] for k, v in result["treatment_statistics"].items()}
        assert means["T3"] > means["T1"]

    def test_rcbd_no_false_positive(self):
        rng = np.random.default_rng(10)
        rows = [
            {"blok": f"B{b}", "uygulama": f"T{t}", "verim": rng.normal(5.0, 0.1)}
            for b in range(1, 5)
            for t in range(1, 4)
        ]
        analyzer = FieldTrialAnalyzer(pd.DataFrame(rows))
        result = analyzer.rcbd_analysis("verim", "uygulama", "blok")
        assert result["p_value"] > 0.05

    def test_duncan_sorts_means(self, rcbd_data):
        analyzer = FieldTrialAnalyzer(rcbd_data)
        result = analyzer.Duncan_test("verim", "uygulama")
        sorted_names = [name for name, _ in result["sorted_treatments"]]
        assert sorted_names[0] == "T3"
