"""Premium Program parity testleri — frequencies, crosstabs, t-test ailesi,
transform işlemleri, Cronbach alfa ve Q-Q plot."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from agrista.analysis import (
    one_sample_t_test,
    paired_t_test,
    frequencies,
    crosstabs,
)
from agrista.transform import (
    compute,
    recode,
    bin_variable,
    rank_cases,
    count_values,
    replace_missing,
)
from agrista.sensory import cronbach_alpha
from agrista.viz import AgristaPlotter


class TestOneSampleTTest:
    def test_detects_deviation(self):
        rng = np.random.default_rng(70)
        result = one_sample_t_test(rng.normal(10, 1, 40), test_value=9.0)
        assert result["significant_at_005"] is True
        assert result["cohens_d"] > 0.5
        assert result["ci95_lower"] <= result["mean"] <= result["ci95_upper"]

    def test_no_deviation(self):
        rng = np.random.default_rng(71)
        result = one_sample_t_test(rng.normal(5, 1, 40), test_value=5.0)
        assert result["significant_at_005"] is False

    def test_too_small_raises(self):
        with pytest.raises(ValueError):
            one_sample_t_test([3.0])


class TestPairedTTest:
    def test_detects_change(self):
        rng = np.random.default_rng(72)
        once = rng.normal(10, 1, 30)
        sonra = once + rng.normal(1.2, 0.5, 30)
        result = paired_t_test(once, sonra)
        assert result["significant_at_005"] is True
        assert result["mean_difference"] < 0  # önce - sonra
        assert result["n_pairs"] == 30

    def test_handles_nan_pairs(self):
        rng = np.random.default_rng(73)
        once = rng.normal(10, 1, 20)
        sonra = once + 1.0
        once[0] = np.nan
        result = paired_t_test(once, sonra)
        assert result["n_pairs"] == 19

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            paired_t_test([1, 2, 3], [1, 2])


class TestFrequencies:
    @pytest.fixture
    def df(self):
        return pd.DataFrame({
            "urun": ["buğday", "buğday", "arpa", "mısır", None],
            "bolge": ["Ege", "Ege", "İç", "İç", "Ege"],
            "verim": [4.0, 5.0, 3.0, 7.0, 6.0],
        })

    def test_counts_and_percents(self, df):
        result = frequencies(df, ["urun"])
        table = result["urun"]["table"]
        bugday = next(r for r in table if r["value"] == "buğday")
        assert bugday["count"] == 2
        assert bugday["percent"] == pytest.approx(40.0)
        assert bugday["valid_percent"] == pytest.approx(50.0)
        assert table[-1]["cumulative_percent"] == pytest.approx(100.0)
        assert result["urun"]["n_missing"] == 1
        assert result["urun"]["mode"] == "buğday"

    def test_default_categorical_columns(self, df):
        result = frequencies(df)
        assert set(result.keys()) == {"urun", "bolge"}

    def test_no_categorical_raises(self, df):
        numeric_only = df[["verim"]]
        with pytest.raises(ValueError):
            frequencies(numeric_only)

    def test_missing_column_raises(self, df):
        with pytest.raises(ValueError):
            frequencies(df, ["olmayan"])


class TestCrosstabs:
    def test_detects_association(self):
        # Tam bağımlı yapı: grup A her zaman x, grup B her zaman y
        df = pd.DataFrame({
            "grup": ["A"] * 30 + ["B"] * 30,
            "yanit": ["x"] * 30 + ["y"] * 30,
        })
        result = crosstabs(df, "grup", "yanit")
        assert result["significant_at_005"] is True
        assert result["chi_square"] > 30
        assert result["cramers_v"] == pytest.approx(1.0, abs=0.01)
        # Beklenen frekanslar marj toplamlarıyla tutarlı
        assert result["expected"].values.sum() == pytest.approx(60)

    def test_independence(self):
        rng = np.random.default_rng(74)
        df = pd.DataFrame({
            "a": rng.choice(["x", "y"], 200),
            "b": rng.choice(["p", "q"], 200),
        })
        result = crosstabs(df, "a", "b")
        assert result["significant_at_005"] is False
        assert result["degrees_of_freedom"] == 1

    def test_single_category_raises(self):
        df = pd.DataFrame({"a": ["x"] * 10, "b": ["p"] * 5 + ["q"] * 5})
        with pytest.raises(ValueError):
            crosstabs(df, "a", "b")


class TestTransform:
    @pytest.fixture
    def df(self):
        return pd.DataFrame({
            "sulama": [1000.0, 2000.0, np.nan, 4000.0],
            "gubre": [100.0, 200.0, 300.0, 400.0],
            "grup": [1, 2, 3, 4],
        })

    def test_compute_expression(self, df):
        result = compute(df, "toplam", "sulama * 0.001 + gubre * 0.005")
        assert result["toplam"].iloc[0] == pytest.approx(1.5)
        assert result["toplam"].iloc[1] == pytest.approx(3.0)
        # Orijinal değişmez
        assert "toplam" not in df.columns

    def test_compute_invalid_expression_raises(self, df):
        with pytest.raises(ValueError):
            compute(df, "x", "olmayan_sutun * 2 + (")

    def test_recode_with_else(self, df):
        result = recode(df, "grup", {1: 10, 2: 20}, default=99)
        assert result["grup_recoded"].tolist() == [10, 20, 99, 99]

    def test_recode_nan_default(self, df):
        result = recode(df, "grup", {1: 10})
        assert result["grup_recoded"].iloc[0] == 10
        assert np.isnan(result["grup_recoded"].iloc[1])

    def test_recode_missing_column_raises(self, df):
        with pytest.raises(ValueError):
            recode(df, "yok", {1: 2})

    def test_bin_equal_width(self, df):
        result = bin_variable(df, "gubre", bins=2, method="equal_width")
        binned = result["gubre_binned"].dropna()
        assert binned.nunique() == 2
        assert len(binned) == 4

    def test_bin_equal_freq(self, df):
        full = pd.DataFrame({"x": np.arange(100, dtype=float)})
        result = bin_variable(full, "x", bins=4, method="equal_freq")
        counts = result["x_binned"].value_counts()
        assert counts.tolist() == [25, 25, 25, 25]

    def test_rank_cases_average(self):
        df = pd.DataFrame({"y": [10.0, 30.0, 20.0, 20.0]})
        result = rank_cases(df, "y")
        assert result["y_rank"].tolist() == [1.0, 4.0, 2.5, 2.5]

    def test_rank_descending(self):
        df = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
        result = rank_cases(df, "y", ascending=False)
        assert result["y_rank"].tolist() == [3.0, 2.0, 1.0]

    def test_count_values(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 2, 2], "c": [1, 1, 1]})
        result = count_values(df, ["a", "b", "c"], 1, "n_adet")
        assert result["n_adet"].tolist() == [3, 2, 1]

    def test_replace_missing_mean(self, df):
        result, report = replace_missing(df, ["sulama"], method="mean")
        assert result["sulama"].isna().sum() == 0
        # Ortalama: (1000+2000+4000)/3 ≈ 2333.33
        assert result["sulama"].iloc[2] == pytest.approx(7000 / 3)
        assert report["sulama"] == 1

    def test_replace_missing_interpolate(self, df):
        result, _ = replace_missing(df, ["sulama"], method="interpolate")
        assert result["sulama"].iloc[2] == pytest.approx(3000.0)

    def test_replace_missing_invalid_method_raises(self, df):
        with pytest.raises(ValueError):
            replace_missing(df, ["sulama"], method="sihirli")

    def test_replace_missing_default_numeric(self, df):
        result, report = replace_missing(df, method="median")
        assert result["sulama"].isna().sum() == 0
        assert report["sulama"] == 1


class TestCronbachAlpha:
    def test_perfect_consistency(self):
        # Tüm maddeler aynı kalıbı izliyor → alfa 1'e yakın
        rng = np.random.default_rng(75)
        base = rng.normal(5, 2, 50)
        mat = pd.DataFrame({
            "m1": base, "m2": base + 0.5, "m3": base - 0.5, "m4": base + 1.0,
        })
        result = cronbach_alpha(mat)
        assert result["cronbach_alpha"] > 0.99
        assert result["n_items"] == 4

    def test_random_low_alpha(self):
        rng = np.random.default_rng(76)
        mat = pd.DataFrame(rng.normal(0, 1, (100, 4)),
                           columns=["m1", "m2", "m3", "m4"])
        result = cronbach_alpha(mat)
        assert result["cronbach_alpha"] < 0.3

    def test_alpha_if_item_deleted(self):
        rng = np.random.default_rng(77)
        base = rng.normal(5, 1, 60)
        mat = pd.DataFrame({
            "iyi1": base + rng.normal(0, 0.2, 60),
            "iyi2": base + rng.normal(0, 0.2, 60),
            "kotu": rng.normal(0, 1, 60),
        })
        result = cronbach_alpha(mat)
        # Gürültülü madde silinince alfa artmalı
        assert result["alpha_if_item_deleted"]["kotu"] > result["cronbach_alpha"]

    def test_single_item_raises(self):
        with pytest.raises(ValueError):
            cronbach_alpha(pd.DataFrame({"m1": [1.0, 2.0, 3.0]}))


class TestQQPlot:
    def test_qq_plot_returns_figure(self):
        rng = np.random.default_rng(78)
        plotter = AgristaPlotter()
        fig = plotter.qq_plot(rng.normal(0, 1, 50), title="Test Q-Q")
        assert fig is not None
        assert len(fig.axes[0].lines) >= 1  # referans doğrusu
        AgristaPlotter.close()

    def test_qq_plot_too_small_raises(self):
        plotter = AgristaPlotter()
        with pytest.raises(ValueError):
            plotter.qq_plot([1.0, 2.0])
