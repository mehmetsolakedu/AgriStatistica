"""Dayanıklılık ve sınır durum testleri — geçersiz girdilerde zarif
hata yönetimi (çökme/traceback yok, anlamlı mesaj var)."""

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from agrista.cli import main as cli_main
from agrista.analysis import (
    crosstabs,
    ratio_statistics,
    nearest_neighbor_analysis,
    multidimensional_scaling,
    custom_tables,
    means_report,
)
from agrista.data import weight_cases, merge_files, split_file
from agrista.survival import cox_regression, life_tables
from agrista.marketing import rfm_analysis, mailing_test


@pytest.fixture
def runner():
    return CliRunner()


class TestCliGracefulFailures:
    """CLI komutları hatalı girdide traceback değil, anlaşılır mesaj üretmeli."""

    def test_missing_file_every_command(self, runner):
        commands = [
            ["describe", "/yok.csv"],
            ["corr", "/yok.csv"],
            ["onesample", "/yok.csv", "--kolon", "x", "--deger", "0"],
            ["ctable", "/yok.csv", "--satirlar", "a"],
            ["cox", "/yok.csv", "--zaman", "t", "--olay", "e",
             "--degiskenler", "x"],
        ]
        for cmd in commands:
            result = runner.invoke(cli_main, cmd)
            assert result.exit_code == 0, cmd
            assert ("bulunamadı" in result.output
                    or "Hata" in result.output), cmd
            assert result.exception is None or isinstance(
                result.exception, SystemExit), cmd

    def test_missing_column_message(self, runner, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3.0]})
        path = tmp_path / "d.csv"
        df.to_csv(path, index=False)
        result = runner.invoke(cli_main, ["onesample", str(path),
                                          "--kolon", "yok", "--deger", "0"])
        assert "bulunamadı" in result.output


class TestInvalidDataHandling:
    def test_ratio_zero_denominator(self):
        df = pd.DataFrame({"p": [1.0, 2.0], "d": [0.0, 1.0]})
        with pytest.raises(ValueError, match="sıfır"):
            ratio_statistics(df, "p", "d")

    def test_crosstabs_single_category(self):
        df = pd.DataFrame({"r": ["a"] * 10, "c": ["x", "y"] * 5})
        with pytest.raises(ValueError):
            crosstabs(df, "r", "c")

    def test_crosstabs_too_few_rows(self):
        df = pd.DataFrame({"r": ["a", "b"], "c": ["x", "y"]})
        with pytest.raises(ValueError):
            crosstabs(df, "r", "c")

    def test_knn_constant_predictor(self):
        df = pd.DataFrame({"g": ["A", "B"] * 10, "x": np.ones(20)})
        # Sabit değişken doğrulanmalı (sıfır varyans)
        with pytest.raises(ValueError):
            nearest_neighbor_analysis(df, "g", ["x"], k=3)

    def test_mds_nan_raises(self):
        D = np.array([[0, np.nan], [1, 0]])
        with pytest.raises(ValueError):
            multidimensional_scaling(D)

    def test_mds_asymmetric_small(self):
        with pytest.raises(ValueError):
            multidimensional_scaling(np.zeros((2, 2)))

    def test_custom_tables_missing_column(self):
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError):
            custom_tables(df, rows=["yok"])

    def test_means_report_empty_groups(self):
        df = pd.DataFrame({"y": [np.nan, np.nan], "g": [None, None]})
        with pytest.raises(ValueError):
            means_report(df, "y", ["g"])

    def test_weight_cases_all_zero(self):
        df = pd.DataFrame({"w": [0.0, 0.0], "x": [1.0, 2.0]})
        with pytest.raises(ValueError):
            weight_cases(df, "w")

    def test_merge_files_bad_how(self):
        left_df = pd.DataFrame({"id": [1], "x": [1]})
        r = pd.DataFrame({"id": [1], "y": [2]})
        with pytest.raises(ValueError):
            merge_files(left_df, r, on="id", how="cross-join")

    def test_split_file_missing_column(self):
        with pytest.raises(ValueError):
            split_file(pd.DataFrame({"a": [1]}), "yok")

    def test_cox_no_events(self):
        with pytest.raises(ValueError):
            cox_regression(np.arange(1, 15.0), np.zeros(14),
                           np.random.default_rng(1).normal(size=(14, 1)))

    def test_life_tables_bad_interval(self):
        with pytest.raises(ValueError):
            life_tables([1, 2, 3, 4.0], [1, 1, 0, 1], interval_width=0)

    def test_rfm_past_reference_date(self):
        df = pd.DataFrame({
            "m": ["a", "b", "c"],
            "t": ["2025-06-01", "2025-06-02", "2025-06-03"],
            "x": [10.0, 20.0, 30.0],
        })
        with pytest.raises(ValueError):
            rfm_analysis(df, "m", "t", "x", reference_date="2020-01-01")

    def test_mailing_negative_inputs(self):
        with pytest.raises(ValueError):
            mailing_test(-1, 100, 10, 100)


class TestNanTolerance:
    """Analiz fonksiyonları NaN içeren veride satır atlayarak çalışmalı."""

    def test_one_sample_with_nan(self):
        from agrista.analysis import one_sample_t_test
        x = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
        result = one_sample_t_test(x, 2.5)
        assert result["n"] == 4

    def test_ttest_with_nan(self):
        from agrista.analysis import t_test
        a = np.array([1.0, np.nan, 3.0, 4.0])
        b = np.array([2.0, 3.0, np.nan, 5.0])
        result = t_test(a, b)
        assert np.isfinite(result["t_statistic"])

    def test_correlation_pairwise_nan(self):
        from agrista.analysis import correlation_analysis
        df = pd.DataFrame({
            "x": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "y": [2.0, 4.0, 6.0, np.nan, 10.0, 12.0],
        })
        result = correlation_analysis(df)
        assert np.isfinite(result["correlation_matrix"].loc["x", "y"])
