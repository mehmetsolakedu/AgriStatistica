"""agrista.analysis modülü testleri."""

import numpy as np
import pandas as pd
import pytest

from agrista.analysis import (
    descriptive_stats,
    correlation_analysis,
    t_test,
    anova_one_way,
    linear_regression,
    multiple_regression,
    chi_square_test,
)
from tests.conftest import make_sample_df


class TestDescriptiveStats:
    def test_series_basic_values(self):
        result = descriptive_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = result["data"]
        assert stats["count"] == 5
        assert stats["mean"] == pytest.approx(3.0)
        assert stats["median"] == pytest.approx(3.0)
        assert stats["std"] == pytest.approx(np.std([1, 2, 3, 4, 5], ddof=1))
        assert stats["range"] == pytest.approx(4.0)

    def test_dataframe_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = descriptive_stats(df)
        assert set(result.keys()) == {"a"}

    def test_empty_numeric_raises(self):
        with pytest.raises(ValueError):
            descriptive_stats(pd.DataFrame({"a": ["x", "y"]}))


class TestCorrelation:
    def test_perfect_positive_correlation(self):
        df = pd.DataFrame({"x": np.arange(20, dtype=float), "y": np.arange(20, dtype=float) * 2})
        result = correlation_analysis(df)
        assert result["correlation_matrix"].loc["x", "y"] == pytest.approx(1.0)
        assert len(result["significant_pairs"]) == 1

    def test_pairwise_nan_handling(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=30)
        df = pd.DataFrame({"x": x, "y": x + rng.normal(0, 0.1, 30)})
        df.loc[0, "x"] = np.nan
        df.loc[1:3, "y"] = np.nan
        result = correlation_analysis(df)
        assert result["correlation_matrix"].loc["x", "y"] > 0.9

    def test_unsupported_method_raises(self):
        df = pd.DataFrame({"x": [1.0, 2, 3], "y": [4.0, 5, 6]})
        with pytest.raises(ValueError):
            correlation_analysis(df, method="kendall")


class TestTTest:
    def test_significant_difference(self):
        rng = np.random.default_rng(1)
        g1 = rng.normal(10, 1, 50)
        g2 = rng.normal(12, 1, 50)
        result = t_test(g1, g2)
        assert result["p_value"] < 0.05
        assert result["significant_at_005"] is True
        assert abs(result["cohens_d"]) > 0.5

    def test_no_difference(self):
        rng = np.random.default_rng(2)
        g = rng.normal(10, 1, 50)
        result = t_test(g, rng.normal(10, 1, 50))
        assert result["p_value"] > 0.05

    def test_too_small_groups_raise(self):
        with pytest.raises(ValueError):
            t_test([1.0], [2.0, 3.0])


class TestAnova:
    def test_significant_groups(self):
        rng = np.random.default_rng(3)
        result = anova_one_way(rng.normal(5, 1, 30), rng.normal(8, 1, 30), rng.normal(11, 1, 30))
        assert result["p_value"] < 0.001
        assert 0 < result["eta_squared"] <= 1
        assert result["degrees_of_freedom_between"] == 2

    def test_single_group_raises(self):
        with pytest.raises(ValueError):
            anova_one_way([1, 2, 3])


class TestRegression:
    def test_linear_recovery(self):
        rng = np.random.default_rng(4)
        x = rng.uniform(0, 10, 100)
        y = 2.0 + 3.0 * x + rng.normal(0, 0.1, 100)
        result = linear_regression(x, y)
        assert result["coefficients"]["slope"] == pytest.approx(3.0, abs=0.05)
        assert result["coefficients"]["intercept"] == pytest.approx(2.0, abs=0.2)
        assert result["r_squared"] > 0.99

    def test_multiple_regression_with_intercept(self):
        df = make_sample_df()
        result = multiple_regression(df, "verim", ["sulama", "gubre"])
        assert result["r_squared"] > 0.95
        assert "const" in result["coefficients"]
        assert result["coefficients"]["const"] == pytest.approx(2.0, abs=0.5)
        assert result["coefficients"]["sulama"] == pytest.approx(0.001, abs=0.0002)
        assert result["coefficients"]["gubre"] == pytest.approx(0.005, abs=0.001)

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError):
            linear_regression([1, 2], [3, 4])


class TestChiSquare:
    def test_uniform_not_significant(self):
        result = chi_square_test([25, 25, 25, 25])
        assert result["chi_square_statistic"] == pytest.approx(0.0)
        assert result["significant_at_005"] is False

    def test_skewed_significant(self):
        result = chi_square_test([90, 10, 10, 10])
        assert result["significant_at_005"] is True
        assert result["degrees_of_freedom"] == 3
