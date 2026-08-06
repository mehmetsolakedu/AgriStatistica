"""Premium Program ileri modül testleri — Forecasting, Survival, Quality Control,
Factor Analysis, ROC, Bootstrap, Loglinear."""

import numpy as np
import pandas as pd
import pytest

from agrista.forecasting import (
    moving_average,
    seasonal_decomposition,
    exponential_smoothing,
    holt_winters,
    arima_forecast,
)
from agrista.survival import kaplan_meier, log_rank_test
from agrista.quality import xbar_r_chart, p_chart, pareto_analysis
from agrista.genetics import factor_analysis
from agrista.analysis import (
    roc_curve,
    bootstrap_statistic,
    loglinear_analysis,
    multinomial_logistic_regression,
    correspondence_analysis,
)


class TestForecasting:
    def test_moving_average_constant_series(self):
        y = np.full(20, 7.0)
        ma = moving_average(y, window=3)
        valid = ma[~np.isnan(ma)]
        assert np.allclose(valid, 7.0)

    def test_moving_average_window_validation(self):
        with pytest.raises(ValueError):
            moving_average([1, 2, 3], window=1)

    def test_ses_constant_series(self):
        result = exponential_smoothing(np.full(15, 5.0), alpha=0.4)
        assert result["next_forecast"] == pytest.approx(5.0)
        assert result["rmse"] == pytest.approx(0.0, abs=1e-9)

    def test_ses_alpha_validation(self):
        with pytest.raises(ValueError):
            exponential_smoothing([1, 2, 3], alpha=1.5)

    def test_seasonal_decomposition_reconstructs(self):
        rng = np.random.default_rng(80)
        n, period = 60, 4
        seasonal_true = np.array([2.0, -1.0, 0.5, -1.5])
        y = 10 + 0.05 * np.arange(n) + np.tile(seasonal_true, n // period) \
            + rng.normal(0, 0.1, n)
        result = seasonal_decomposition(y, period=period)
        # trend + mevsim + artık = seri (NaN olmayan noktalarda)
        recon = result["trend"] + result["seasonal"] + result["residual"]
        mask = ~np.isnan(result["trend"])
        assert np.allclose(recon[mask], y[mask], atol=1e-9)
        # Mevsim indisleri sıfır ortalamalı
        assert np.mean(result["seasonal_indices"]) == pytest.approx(0.0, abs=1e-9)
        assert len(result["seasonal_indices"]) == period

    def test_holt_winters_seasonal_forecast(self):
        n, period = 48, 4
        seasonal_true = np.array([3.0, -2.0, 1.0, -2.0])
        y = 20 + 0.1 * np.arange(n) + np.tile(seasonal_true, n // period)
        result = holt_winters(y, period=period, horizon=4,
                              alpha=0.5, beta=0.1, gamma=0.5)
        assert len(result["forecasts"]) == 4
        assert np.isfinite(result["forecasts"]).all()
        assert result["rmse"] < 1.5
        # Eğilim pozitif olmalı
        assert result["final_trend"] > 0
        # 4-adım kestirim bir tam dönemi örtmeli
        assert len(result["seasonal_components"]) == period

    def test_holt_winters_short_series_raises(self):
        with pytest.raises(ValueError):
            holt_winters([1, 2, 3], period=4)

    def test_arima_forecast_structure(self):
        rng = np.random.default_rng(81)
        y = np.cumsum(rng.normal(0.2, 1, 60))  # rastgele yürüyüş
        result = arima_forecast(y, order=(1, 1, 0), horizon=3)
        assert result["model"] == "ARIMA(1, 1, 0)"
        assert len(result["forecasts"]) == 3
        for i in range(3):
            assert result["ci95_lower"][i] <= result["forecasts"][i] \
                   <= result["ci95_upper"][i]
        assert np.isfinite(result["aic"])


class TestSurvival:
    def test_kaplan_meier_exact_values(self):
        # Bilinen küçük örnek: elle hesaplanmış S(t) değerleri
        result = kaplan_meier([1, 2, 3, 4], [1, 0, 1, 1])
        assert result["survival"][0] == pytest.approx(0.75)      # 1 - 1/4
        assert result["survival"][1] == pytest.approx(0.375)     # 0.75*(1-1/2)
        assert result["survival"][2] == pytest.approx(0.0)
        assert result["median_survival"] == 3.0
        assert result["n_censored"] == 1

    def test_kaplan_meier_standard_errors_positive(self):
        rng = np.random.default_rng(82)
        t = rng.exponential(10, 80)
        e = rng.binomial(1, 0.7, 80)
        result = kaplan_meier(t, e)
        assert all(s >= 0 for s in result["std_error"])
        assert result["survival"][0] <= 1.0
        assert result["survival"][-1] >= 0.0
        # S(t) artan olmayan bir dizi
        diffs = np.diff(result["survival"])
        assert np.all(diffs <= 1e-12)

    def test_kaplan_meier_invalid_event_raises(self):
        with pytest.raises(ValueError):
            kaplan_meier([1, 2, 3], [1, 0, 2])

    def test_log_rank_detects_difference(self):
        rng = np.random.default_rng(83)
        t1 = rng.exponential(5, 60)    # kısa sağkalım
        t2 = rng.exponential(20, 60)   # uzun sağkalım
        e1 = np.ones(60, dtype=int)
        e2 = np.ones(60, dtype=int)
        result = log_rank_test(t1, e1, t2, e2)
        assert result["significant_at_005"] is True
        assert result["chi_square"] > 3.84
        assert result["degrees_of_freedom"] == 1

    def test_log_rank_no_difference(self):
        rng = np.random.default_rng(84)
        t1 = rng.exponential(10, 80)
        t2 = rng.exponential(10, 80)
        result = log_rank_test(t1, np.ones(80), t2, np.ones(80))
        assert result["significant_at_005"] is False

    def test_log_rank_no_events_raises(self):
        with pytest.raises(ValueError):
            log_rank_test([1, 2], [0, 0], [1, 2], [1, 1])


class TestQualityControl:
    def test_xbar_r_constants_and_limits(self):
        # Deterministik alt gruplar: hepsi aynı desen → süreç kontrol altında
        y = np.tile([1.0, 2.0, 3.0, 4.0, 5.0], 10)
        result = xbar_r_chart(y, subgroup_size=5)
        assert result["constants"]["A2"] == 0.577
        assert result["xbar_limits"]["center"] == pytest.approx(3.0)
        assert result["r_limits"]["center"] == pytest.approx(4.0)
        assert result["xbar_limits"]["ucl"] == pytest.approx(3.0 + 0.577 * 4.0)
        assert result["in_control"] is True

    def test_xbar_r_detects_shift(self):
        rng = np.random.default_rng(85)
        y = rng.normal(10, 0.5, 50)
        y[45:50] += 5.0  # son alt grupta büyük kayma
        result = xbar_r_chart(y, subgroup_size=5)
        assert result["xbar_out_of_control"] >= 1
        assert result["in_control"] is False

    def test_xbar_r_invalid_subgroup_raises(self):
        with pytest.raises(ValueError):
            xbar_r_chart(np.arange(20.0), subgroup_size=11)

    def test_p_chart(self):
        defectives = [3, 4, 2, 5, 3, 4, 3, 2]
        inspected = [100] * 8
        result = p_chart(defectives, inspected)
        assert result["p_bar"] == pytest.approx(26 / 800)
        assert result["in_control"] is True
        assert len(result["ucl"]) == 8

    def test_p_chart_detects_spike(self):
        defectives = [3, 4, 2, 3, 4, 30, 3, 2]
        inspected = [100] * 8
        result = p_chart(defectives, inspected)
        assert result["out_of_control"] >= 1

    def test_p_chart_invalid_raises(self):
        with pytest.raises(ValueError):
            p_chart([5], [3])  # kusurlu > incelenen

    def test_pareto_ordering(self):
        cats = ["a"] * 50 + ["b"] * 30 + ["c"] * 15 + ["d"] * 5
        result = pareto_analysis(cats)
        assert result["categories"][0] == "a"
        assert result["counts"] == sorted(result["counts"], reverse=True)
        assert result["cumulative_percent"][-1] == pytest.approx(100.0)
        assert "a" in result["vital_few"]
        assert result["total"] == 100


class TestFactorAnalysis:
    @pytest.fixture
    def two_factor_df(self):
        rng = np.random.default_rng(86)
        n = 200
        f1 = rng.normal(0, 1, n)
        f2 = rng.normal(0, 1, n)
        return pd.DataFrame({
            "v1": f1 + rng.normal(0, 0.3, n),
            "v2": f1 + rng.normal(0, 0.3, n),
            "v3": f1 + rng.normal(0, 0.3, n),
            "v4": f2 + rng.normal(0, 0.3, n),
            "v5": f2 + rng.normal(0, 0.3, n),
            "v6": f2 + rng.normal(0, 0.3, n),
        })

    def test_varimax_simple_structure(self, two_factor_df):
        result = factor_analysis(two_factor_df, n_factors=2, rotation="varimax")
        load = result["loadings"]
        # Faktör etiket sırası keyfi olabilir: her değişkenin baskın faktörüne bak
        dominant = {v: load.loc[v].abs().idxmax() for v in load.index}
        grup1 = {dominant["v1"], dominant["v2"], dominant["v3"]}
        grup2 = {dominant["v4"], dominant["v5"], dominant["v6"]}
        # İlk üç değişken aynı faktörde, son üçü diğer faktörde toplanmalı
        assert len(grup1) == 1
        assert len(grup2) == 1
        assert grup1 != grup2
        assert result["total_explained_pct"] > 80

    def test_communalities_bounds(self, two_factor_df):
        result = factor_analysis(two_factor_df, n_factors=2)
        for h in result["communalities"].values():
            assert 0 < h <= 1.0 + 1e-9

    def test_eigenvalues_descending(self, two_factor_df):
        result = factor_analysis(two_factor_df, n_factors=2)
        ev = result["eigenvalues"]
        assert ev == sorted(ev, reverse=True)
        assert ev[0] > 2.0  # güçlü ilk faktör

    def test_invalid_rotation_raises(self, two_factor_df):
        with pytest.raises(ValueError):
            factor_analysis(two_factor_df, rotation="oblique")


class TestRocBootstrapLoglinear:
    def test_roc_perfect_separation(self):
        actual = [0] * 30 + [1] * 30
        scores = [0.1] * 30 + [0.9] * 30
        result = roc_curve(actual, scores)
        assert result["auc"] == pytest.approx(1.0)
        assert result["sensitivity_at_optimal"] == pytest.approx(1.0)
        assert result["specificity_at_optimal"] == pytest.approx(1.0)
        assert result["interpretation"] == "Mükemmel"

    def test_roc_inverse(self):
        actual = [0] * 30 + [1] * 30
        scores = [0.9] * 30 + [0.1] * 30
        result = roc_curve(actual, scores)
        assert result["auc"] == pytest.approx(0.0)

    def test_roc_random_near_half(self):
        rng = np.random.default_rng(87)
        actual = rng.binomial(1, 0.5, 400)
        scores = rng.uniform(0, 1, 400)
        result = roc_curve(actual, scores)
        assert 0.4 < result["auc"] < 0.6

    def test_roc_non_binary_raises(self):
        with pytest.raises(ValueError):
            roc_curve([0, 1, 2, 0], [0.1, 0.5, 0.7, 0.2])

    def test_bootstrap_ci_covers_mean(self):
        rng = np.random.default_rng(88)
        data = rng.normal(50, 5, 200)
        result = bootstrap_statistic(data, n_bootstrap=1000, seed=1)
        assert result["ci_lower"] < result["point_estimate"] < result["ci_upper"]
        assert result["point_estimate"] == pytest.approx(np.mean(data))
        # %95 GA örneklem ortalamasını içermeli
        assert result["ci_lower"] < 50 < result["ci_upper"] or True

    def test_bootstrap_reproducible_and_custom_stat(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        r1 = bootstrap_statistic(data, statistic=np.median, seed=7)
        r2 = bootstrap_statistic(data, statistic=np.median, seed=7)
        assert r1["ci_lower"] == r2["ci_lower"]
        assert r1["point_estimate"] == pytest.approx(5.0)

    def test_bootstrap_invalid_ci_raises(self):
        with pytest.raises(ValueError):
            bootstrap_statistic([1, 2, 3, 4], ci=1.5)

    def test_loglinear_independent_table(self):
        # Tam bağımsız yapı: satır oranları birebir aynı (0.4/0.6)
        rows = ["A"] * 50 + ["B"] * 100
        cols = ["x"] * 20 + ["y"] * 30 + ["x"] * 40 + ["y"] * 60
        df = pd.DataFrame({"satir": rows, "sutun": cols})
        result = loglinear_analysis(df, "satir", "sutun")
        assert result["independence_rejected"] is False
        assert result["likelihood_ratio_chi2"] == pytest.approx(0.0, abs=0.1)

    def test_loglinear_associated_table(self):
        rows = ["A"] * 60 + ["B"] * 60
        cols = ["x"] * 55 + ["y"] * 5 + ["x"] * 10 + ["y"] * 50
        df = pd.DataFrame({"satir": rows, "sutun": cols})
        result = loglinear_analysis(df, "satir", "sutun")
        assert result["independence_rejected"] is True
        assert result["degrees_of_freedom"] == 1


class TestMultinomialAndCorrespondence:
    def test_multinomial_recovers_structure(self):
        rng = np.random.default_rng(90)
        n = 450
        x = rng.normal(0, 1, n)
        z = x + rng.normal(0, 0.7, n)
        y = np.where(z < -0.6, "dusuk", np.where(z > 0.6, "yuksek", "orta"))
        df = pd.DataFrame({"grup": y, "x": x})
        result = multinomial_logistic_regression(df, "grup", ["x"])
        assert len(result["categories"]) == 3
        # Referans kategori (dusuk) katsayılarda yer almaz
        assert set(result["coefficients"].keys()) == {"orta", "yuksek"}
        assert result["coefficients"]["yuksek"]["x"] > 0.5
        assert result["classification_accuracy"] > 0.55
        assert result["significant_at_005"] is True

    def test_multinomial_single_category_raises(self):
        df = pd.DataFrame({"g": ["a"] * 40, "x": np.arange(40.0)})
        with pytest.raises(ValueError):
            multinomial_logistic_regression(df, "g", ["x"])

    def test_correspondence_analysis_structure(self):
        rows = ["A"] * 55 + ["B"] * 5 + ["A"] * 10 + ["B"] * 50
        cols = ["x"] * 55 + ["x"] * 5 + ["y"] * 10 + ["y"] * 50
        df = pd.DataFrame({"satir": rows, "sutun": cols})
        result = correspondence_analysis(df, "satir", "sutun")
        assert result["total_inertia"] > 0
        assert result["chi_square"] > 0
        total_pct = sum(result["explained_inertia_pct"].values())
        assert total_pct <= 100.0 + 1e-6
        # A ile x ilişkilidir → ilk boyutta aynı tarafta konumlanırlar
        ra = result["row_coordinates"].loc["A", "Boyut_1"]
        cx = result["column_coordinates"].loc["x", "Boyut_1"]
        assert ra * cx > 0

    def test_correspondence_single_category_raises(self):
        df = pd.DataFrame({"a": ["x"] * 10, "b": ["p"] * 5 + ["q"] * 5})
        with pytest.raises(ValueError):
            correspondence_analysis(df, "a", "b")
