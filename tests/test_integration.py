"""Entegrasyon testleri — katmanlar arası tutarlılık ve bağımsız
referanslara (scipy) karşı çapraz doğrulama."""

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from scipy import stats

from agrista.cli import main as cli_main
from agrista.data import load_csv
from agrista.analysis import (
    crosstabs,
    one_sample_t_test,
    paired_t_test,
    correlation_analysis,
    frequencies,
    ratio_statistics,
    distance_matrix,
)


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def farm_csv(tmp_path):
    rng = np.random.default_rng(101)
    df = pd.DataFrame({
        "verim": rng.normal(5, 1, 40),
        "alan": rng.uniform(1, 10, 40),
        "bolge": rng.choice(["Ege", "İç"], 40),
    })
    path = tmp_path / "farm.csv"
    df.to_csv(path, index=False)
    return str(path), df


class TestGroundTruth:
    """Analiz fonksiyonlarının bağımsız scipy referanslarıyla çapraz
    doğrulanması."""

    def test_crosstabs_matches_scipy(self):
        rows = ["A"] * 40 + ["B"] * 10 + ["A"] * 10 + ["B"] * 40
        cols = ["x"] * 40 + ["y"] * 10 + ["x"] * 10 + ["y"] * 40
        df = pd.DataFrame({"r": rows, "c": cols})
        result = crosstabs(df, "r", "c")
        table = pd.crosstab(df["r"], df["c"])
        chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
        assert result["chi_square"] == pytest.approx(chi2, rel=1e-9)
        assert result["p_value"] == pytest.approx(p, rel=1e-6)
        assert result["degrees_of_freedom"] == dof

    def test_one_sample_matches_scipy(self):
        rng = np.random.default_rng(102)
        x = rng.normal(5.5, 1.2, 35)
        result = one_sample_t_test(x, test_value=5.0)
        t_ref, p_ref = stats.ttest_1samp(x, 5.0)
        assert result["t_statistic"] == pytest.approx(t_ref, rel=1e-9)
        assert result["p_value"] == pytest.approx(p_ref, rel=1e-9)

    def test_paired_matches_scipy(self):
        rng = np.random.default_rng(103)
        a = rng.normal(10, 1, 30)
        b = a + rng.normal(0.8, 0.5, 30)
        result = paired_t_test(a, b)
        t_ref, p_ref = stats.ttest_rel(a, b)
        assert result["t_statistic"] == pytest.approx(t_ref, rel=1e-9)
        assert result["p_value"] == pytest.approx(p_ref, rel=1e-9)

    def test_correlation_matches_scipy(self):
        rng = np.random.default_rng(104)
        x = rng.normal(0, 1, 50)
        df = pd.DataFrame({"x": x, "y": 0.7 * x + rng.normal(0, 0.5, 50)})
        result = correlation_analysis(df, method="pearson")
        r_ref, p_ref = stats.pearsonr(df["x"], df["y"])
        assert result["correlation_matrix"].loc["x", "y"] == pytest.approx(
            r_ref, rel=1e-9)
        assert result["p_values"].loc["x", "y"] == pytest.approx(
            p_ref, rel=1e-6)

    def test_ratio_statistics_manual(self):
        df = pd.DataFrame({"pay": [10.0, 20.0, 30.0],
                           "payda": [2.0, 4.0, 5.0]})
        result = ratio_statistics(df, "pay", "payda")
        oranlar = np.array([5.0, 5.0, 6.0])
        assert result["mean_ratio"] == pytest.approx(oranlar.mean())
        assert result["median_ratio"] == pytest.approx(np.median(oranlar))

    def test_distance_matches_scipy(self):
        df = pd.DataFrame({"a": [0.0, 3.0], "b": [4.0, 0.0]})
        result = distance_matrix(df, ["a", "b"], measure="euclidean",
                                 between="variables")
        # değişken vektörleri: a=[0,3], b=[4,0] → Öklid mesafesi 5
        assert result["distances"].loc["a", "b"] == pytest.approx(5.0)


class TestLayerConsistency:
    """Veri katmanı → analiz katmanı → CLI katmanı tutarlılığı."""

    def test_cli_onesample_matches_api(self, runner, farm_csv):
        path, df = farm_csv
        api = one_sample_t_test(df["verim"], 5.0)
        result = runner.invoke(cli_main, ["onesample", path,
                                          "--kolon", "verim", "--deger", "5"])
        assert result.exit_code == 0
        assert f"{api['t_statistic']:.4f}" in result.output

    def test_cli_crosstabs_matches_api(self, runner, tmp_path):
        rows = ["A"] * 30 + ["B"] * 10 + ["A"] * 5 + ["B"] * 25
        cols = ["x"] * 30 + ["y"] * 10 + ["x"] * 5 + ["y"] * 25
        df = pd.DataFrame({"r": rows, "c": cols})
        path = tmp_path / "capraz.csv"
        df.to_csv(path, index=False)
        api = crosstabs(df, "r", "c")
        result = runner.invoke(cli_main, ["crosstabs", str(path),
                                          "--satir", "r", "--sutun", "c"])
        assert result.exit_code == 0
        assert f"{api['chi_square']:.4f}" in result.output

    def test_cli_frequencies_matches_api(self, runner, tmp_path):
        df = pd.DataFrame({"grup": ["A"] * 12 + ["B"] * 8})
        path = tmp_path / "freq.csv"
        df.to_csv(path, index=False)
        api = frequencies(df, ["grup"])
        result = runner.invoke(cli_main, ["frequencies", str(path),
                                          "--kolonlar", "grup"])
        assert result.exit_code == 0
        # API'deki A sayısı CLI çıktısında görünmeli
        a_count = api["grup"]["table"][0]["count"]
        assert str(a_count) in result.output

    def test_cli_describe_matches_data_layer(self, runner, farm_csv):
        path, df = farm_csv
        data = load_csv(path)
        assert data.dataframe.shape == df.shape
        result = runner.invoke(cli_main, ["describe", path])
        assert result.exit_code == 0
        assert "verim" in result.output


class TestSaveLoadRoundTrip:
    def test_export_load_roundtrip(self, tmp_path):
        rng = np.random.default_rng(105)
        df = pd.DataFrame({"x": rng.normal(0, 1, 20),
                           "g": rng.choice(["a", "b"], 20)})
        from agrista.data import AgristaData
        ad = AgristaData(df)
        out = tmp_path / "rt.csv"
        ad.export_csv(str(out))
        loaded = load_csv(str(out))
        assert loaded.dataframe.shape == df.shape
        assert list(loaded.dataframe.columns) == ["x", "g"]
        assert np.allclose(loaded.dataframe["x"], df["x"])
