"""Premium Program dördüncü dalga testleri — Custom Tables, Reports, Ratios,
Distances, k-NN, MDS, Multiple Response, Life Tables, Cox, Direct
Marketing (RFM/mailing/profil) ve Data menüsü denkliği."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from agrista.viz import AgristaPlotter
from agrista.cli import main as cli_main
from agrista.analysis import (
    custom_tables,
    means_report,
    case_summaries,
    ratio_statistics,
    distance_matrix,
    nearest_neighbor_analysis,
    multidimensional_scaling,
    multiple_response_frequencies,
    weight_estimation,
    two_stage_least_squares,
    twostep_cluster,
)
from agrista.survival import life_tables, cox_regression
from agrista.marketing import rfm_analysis, mailing_test, prospect_profiles
from agrista.transform import create_time_series, random_numbers, automatic_recode
from agrista.data import (
    sort_cases,
    aggregate_data,
    weight_cases,
    merge_files,
    split_file,
    identify_duplicates,
    transpose_data,
    restructure_data,
    compare_datasets,
    define_measurement_level,
)


@pytest.fixture
def farm_df():
    rng = np.random.default_rng(60)
    n = 120
    bolge = rng.choice(["Ege", "İç"], n)
    verim = np.where(bolge == "Ege", rng.normal(6, 1, n), rng.normal(4.5, 1, n))
    return pd.DataFrame({
        "bolge": bolge,
        "cesit": rng.choice(["c1", "c2"], n),
        "verim": verim,
        "alan": rng.uniform(1, 10, n),
    })


class TestTablesReports:
    def test_custom_tables_structure(self, farm_df):
        result = custom_tables(farm_df, rows=["bolge"], columns=["cesit"],
                               values="verim", statistics=["count", "mean"])
        table = result["table"]
        assert set(table.columns.get_level_values(0)) == {"count", "mean"}
        assert table.loc[("Ege", "c1"), "count"] == len(
            farm_df[(farm_df["bolge"] == "Ege") & (farm_df["cesit"] == "c1")])
        assert result["n_cases"] == 120

    def test_custom_tables_count_only(self, farm_df):
        result = custom_tables(farm_df, rows=["bolge"], statistics=["count"])
        assert result["table"]["count"].sum() == 120

    def test_custom_tables_unknown_stat_raises(self, farm_df):
        with pytest.raises(ValueError):
            custom_tables(farm_df, rows=["bolge"], values="verim",
                          statistics=["geometric_mean"])

    def test_means_report_layers(self, farm_df):
        result = means_report(farm_df, "verim", ["bolge"])
        assert result["grand_total"]["n"] == 120
        ege = result["layers"]["bolge"]["Ege"]
        beklenen = farm_df.loc[farm_df["bolge"] == "Ege", "verim"]
        assert ege["mean"] == pytest.approx(beklenen.mean())
        # Ege ortalaması İç'ten yüksek olmalı (veri üretim deseni)
        assert ege["mean"] > result["layers"]["bolge"]["İç"]["mean"]

    def test_case_summaries_limits(self, farm_df):
        result = case_summaries(farm_df, n_cases=5)
        assert result["n_shown"] == 5
        assert result["n_total"] == 120
        assert len(result["cases"]) == 5

    def test_ratio_statistics_values(self, farm_df):
        result = ratio_statistics(farm_df, "verim", "alan")
        oranlar = farm_df["verim"] / farm_df["alan"]
        assert result["mean_ratio"] == pytest.approx(oranlar.mean())
        assert result["median_ratio"] == pytest.approx(oranlar.median())
        assert result["cov"] == pytest.approx(oranlar.std(ddof=1) / oranlar.mean())

    def test_ratio_statistics_zero_denominator_raises(self, farm_df):
        df = farm_df.copy()
        df.loc[df.index[0], "alan"] = 0
        with pytest.raises(ValueError):
            ratio_statistics(df, "verim", "alan")


class TestDistancesKnnMds:
    def test_distance_matrix_symmetric(self, farm_df):
        result = distance_matrix(farm_df, ["verim", "alan"], measure="euclidean")
        D = result["distances"]
        assert D.shape == (120, 120)
        assert np.allclose(D.values, D.values.T)
        assert np.allclose(np.diag(D.values), 0.0)

    def test_distance_correlation_between_variables(self, farm_df):
        result = distance_matrix(farm_df, ["verim", "alan"], measure="correlation",
                                 between="variables")
        D = result["distances"]
        beklenen = 1.0 - np.corrcoef(farm_df["verim"], farm_df["alan"])[0, 1]
        assert D.loc["verim", "alan"] == pytest.approx(beklenen)

    def test_knn_separated_groups(self):
        rng = np.random.default_rng(61)
        n = 80
        grup = np.repeat(["A", "B"], n // 2)
        x = np.where(grup == "A", rng.normal(0, 0.5, n), rng.normal(5, 0.5, n))
        df = pd.DataFrame({"grup": grup, "x": x})
        result = nearest_neighbor_analysis(df, "grup", ["x"], k=3)
        assert result["overall_accuracy"] > 0.95
        assert result["model"].startswith("k-Nearest")

    def test_knn_invalid_k_raises(self, farm_df):
        df = farm_df.assign(grup=np.where(farm_df["verim"] > 5, "A", "B"))
        with pytest.raises(ValueError):
            nearest_neighbor_analysis(df, "grup", ["verim"], k=0)

    def test_mds_reconstructs_geometry(self):
        # 3-4-5 üçgeni: MDS koordinatları mesafeleri yeniden üretmeli
        D = np.array([[0, 3, 5], [3, 0, 4], [5, 4, 0]], dtype=float)
        result = multidimensional_scaling(D, n_dims=2)
        coords = result["coordinates"].values
        d01 = np.linalg.norm(coords[0] - coords[1])
        d02 = np.linalg.norm(coords[0] - coords[2])
        assert d01 == pytest.approx(3.0, rel=1e-6)
        assert d02 == pytest.approx(5.0, rel=1e-6)
        assert result["r_squared"] == pytest.approx(1.0, abs=1e-6)

    def test_mds_non_square_raises(self):
        with pytest.raises(ValueError):
            multidimensional_scaling(np.zeros((3, 4)))


class TestMultipleResponseSurvival:
    def test_multiple_response_counts(self):
        df = pd.DataFrame({
            "tohum": [1, 1, 0, 1, 0],
            "gubre": [1, 0, 0, 1, 1],
            "ilac": [0, 0, 0, 0, 1],
        })
        result = multiple_response_frequencies(df, ["tohum", "gubre", "ilac"], 1)
        assert result["total_responses"] == 7
        assert result["n_cases"] == 5
        assert result["mean_responses_per_case"] == pytest.approx(1.4)
        tohum_row = result["table"][0]
        assert tohum_row["count"] == 3
        assert tohum_row["percent_of_cases"] == pytest.approx(60.0)

    def test_multiple_response_single_column_raises(self):
        df = pd.DataFrame({"a": [1, 0], "b": [0, 1]})
        with pytest.raises(ValueError):
            multiple_response_frequencies(df, ["a"], 1)

    def test_life_tables_actuarial(self):
        # Deterministik: 10 vaka, hepsi t=2'de olay, aralık 1
        t = np.full(10, 2.0)
        e = np.ones(10)
        result = life_tables(t, e, interval_width=1.0)
        tbl = result["table"]
        # [0,1) ve [1,2) aralıklarında olay yok; [2,3)'te 10 olay
        assert tbl.loc[0, "n_terminated"] == 0
        assert tbl.loc[2, "n_terminated"] == 10
        assert tbl.loc[2, "proportion_terminating"] == pytest.approx(1.0)
        assert result["final_survival"] == pytest.approx(0.0)
        assert result["median_survival"] is not None

    def test_life_tables_censored(self):
        rng = np.random.default_rng(62)
        t = rng.exponential(8, 100)
        e = rng.binomial(1, 0.6, 100)
        result = life_tables(t, e, interval_width=2.0)
        surv = result["table"]["survival"].values
        # Sağkalım artmayan ve [0,1] aralığında
        assert np.all(np.diff(surv) <= 1e-12)
        assert np.all((surv >= 0) & (surv <= 1))

    def test_cox_recovers_effect(self):
        rng = np.random.default_rng(63)
        n = 200
        x = rng.normal(0, 1, n)
        e = rng.binomial(1, 0.8, n)
        t = rng.exponential(np.exp(-0.7 * x) * 8, n)  # gerçek beta ≈ 0.7
        result = cox_regression(t, e, np.column_stack([x]))
        assert result["coefficients"][0] == pytest.approx(0.7, abs=0.3)
        assert result["significant_at_005"] is True
        assert result["concordance_index"] > 0.55
        assert result["exp_coef"][0] == pytest.approx(
            np.exp(result["coefficients"][0]))

    def test_cox_no_effect_not_significant(self):
        rng = np.random.default_rng(64)
        n = 150
        x = rng.normal(0, 1, n)
        t = rng.exponential(10, n)  # x ile ilişkisiz
        e = rng.binomial(1, 0.7, n)
        result = cox_regression(t, e, np.column_stack([x]))
        assert result["significant_at_005"] is False

    def test_cox_too_few_events_raises(self):
        with pytest.raises(ValueError):
            cox_regression([5, 6, 7, 1, 2, 3, 4, 8, 9, 10],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           np.arange(10.0))


class TestMarketing:
    @pytest.fixture
    def rfm_df(self):
        rng = np.random.default_rng(65)
        rows = []
        for i in range(30):
            n_txn = rng.integers(1, 6)
            for _ in range(n_txn):
                rows.append({
                    "musteri": f"M{i}",
                    "tarih": pd.Timestamp("2025-12-31")
                             - pd.Timedelta(days=int(rng.integers(1, 365))),
                    "tutar": float(rng.uniform(20, 800)),
                })
        return pd.DataFrame(rows)

    def test_rfm_scores_bounded(self, rfm_df):
        result = rfm_analysis(rfm_df, "musteri", "tarih", "tutar",
                              reference_date="2026-01-01", quantiles=5)
        cust = result["customers"]
        for col in ("r_score", "f_score", "m_score"):
            assert cust[col].between(1, 5).all()
        assert result["n_customers"] == 30
        assert set(result["segment_summary"]) <= {
            "Şampiyon", "Yeni/Yakın", "Sadık ama uzaklaşan", "Riskte"}

    def test_rfm_frequency_ordering(self, rfm_df):
        result = rfm_analysis(rfm_df, "musteri", "tarih", "tutar",
                              reference_date="2026-01-01")
        cust = result["customers"]
        # En çok işlem yapan müşterinin F skoru maksimum olmalı
        assert cust.loc[cust["frequency"].idxmax(), "f_score"] == 5

    def test_rfm_past_reference_raises(self, rfm_df):
        with pytest.raises(ValueError):
            rfm_analysis(rfm_df, "musteri", "tarih", "tutar",
                         reference_date="2020-01-01")

    def test_mailing_test_significant_lift(self):
        result = mailing_test(20, 1000, 40, 1000)
        assert result["significant_at_005"] is True
        assert result["lift"] == pytest.approx(1.0)  # %100 kaldırma
        assert result["ci95_lower"] > 0

    def test_mailing_test_no_difference(self):
        result = mailing_test(30, 1000, 32, 1000)
        assert result["significant_at_005"] is False

    def test_mailing_test_invalid_raises(self):
        with pytest.raises(ValueError):
            mailing_test(50, 40, 10, 100)  # yanıt > grup

    def test_prospect_profiles_lift(self):
        rng = np.random.default_rng(66)
        n = 400
        kanal = rng.choice(["posta", "sms"], n)
        # sms kanalında yanıt belirgin şekilde yüksek
        p = np.where(kanal == "sms", 0.6, 0.2)
        yanit = rng.binomial(1, p)
        df = pd.DataFrame({"yanit": yanit, "kanal": kanal})
        result = prospect_profiles(df, "yanit", ["kanal"], positive_value=1)
        sms = next(r for r in result["profiles"]["kanal"]
                   if r["category"] == "sms")
        posta = next(r for r in result["profiles"]["kanal"]
                     if r["category"] == "posta")
        assert sms["response_rate"] > posta["response_rate"]
        assert sms["lift"] > 1.0 > posta["lift"]
        assert result["top_segments"][0]["category"] == "sms"

    def test_prospect_profiles_non_binary_raises(self):
        df = pd.DataFrame({"y": [1, 2, 3] * 10, "k": ["a"] * 30})
        with pytest.raises(ValueError):
            prospect_profiles(df, "y", ["k"])


class TestDataManagement:
    def test_sort_cases(self, farm_df):
        result = sort_cases(farm_df, ["verim"])
        assert result["verim"].is_monotonic_increasing
        result_desc = sort_cases(farm_df, ["verim"], ascending=False)
        assert result_desc["verim"].is_monotonic_decreasing

    def test_aggregate_data(self, farm_df):
        result = aggregate_data(farm_df, ["bolge"],
                                {"verim": ["mean", "count"], "alan": "sum"})
        assert set(result.columns) == {"bolge", "verim_mean", "verim_count",
                                       "alan_sum"}
        ege = result[result["bolge"] == "Ege"].iloc[0]
        assert ege["verim_mean"] == pytest.approx(
            farm_df.loc[farm_df["bolge"] == "Ege", "verim"].mean())

    def test_aggregate_unknown_function_raises(self, farm_df):
        with pytest.raises(ValueError):
            aggregate_data(farm_df, ["bolge"], {"verim": "mode"})

    def test_weight_cases(self, farm_df):
        df = farm_df.assign(agirlik=np.where(farm_df["bolge"] == "Ege", 2.0, 1.0))
        result = weight_cases(df, "agirlik")
        assert result["sum_of_weights"] == pytest.approx(df["agirlik"].sum())
        # Ağırlıklı ortalama Ege lehine düz ortalamadan yüksek olmalı
        assert result["weighted_means"]["verim"] > farm_df["verim"].mean()
        assert result["equivalent_n"] <= result["n_cases"]

    def test_weight_cases_negative_raises(self, farm_df):
        df = farm_df.assign(agirlik=-1.0)
        with pytest.raises(ValueError):
            weight_cases(df, "agirlik")

    def test_merge_files_variables(self):
        left = pd.DataFrame({"id": [1, 2, 3], "x": [10, 20, 30]})
        right = pd.DataFrame({"id": [2, 3, 4], "y": [200, 300, 400]})
        result = merge_files(left, right, on="id", how="inner")
        assert list(result["id"]) == [2, 3]
        assert "y" in result.columns

    def test_merge_files_cases(self):
        left = pd.DataFrame({"x": [1, 2]})
        right = pd.DataFrame({"x": [3, 4]})
        result = merge_files(left, right, add_cases=True)
        assert list(result["x"]) == [1, 2, 3, 4]

    def test_split_file(self, farm_df):
        parts = split_file(farm_df, "bolge")
        assert set(parts) == {"Ege", "İç"}
        assert sum(len(p) for p in parts.values()) == 120

    def test_identify_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2, 3], "b": ["x", "x", "y", "z"]})
        result = identify_duplicates(df)
        assert result["n_duplicate_cases"] == 2

    def test_transpose_data(self, farm_df):
        result = transpose_data(farm_df[["verim", "alan"]].head(5))
        assert result.shape == (2, 6)
        assert list(result.columns)[0] == "case"

    def test_restructure_long_wide(self):
        wide = pd.DataFrame({"id": [1, 2], "t1": [10, 20], "t2": [30, 40]})
        long = restructure_data(wide, direction="long", id_cols=["id"],
                                value_cols=["t1", "t2"])
        assert len(long) == 4
        back = restructure_data(long, direction="wide", index_col="id",
                                columns_col="degisken", value_name="deger")
        assert back.shape == (2, 2)
        assert back.loc[1, "t2"] == 30

    def test_compare_datasets(self):
        left = pd.DataFrame({"id": [1, 2, 3], "x": [1, 2, 3]})
        right = pd.DataFrame({"id": [2, 3, 4], "x": [2, 99, 4]})
        result = compare_datasets(left, right, "id")
        assert result["only_in_left"] == 1
        assert result["only_in_right"] == 1
        assert result["matched"] == 2
        assert result["value_mismatches"]["x"] == 1

    def test_define_measurement_level(self, farm_df):
        result = define_measurement_level(farm_df, "bolge", "nominal")
        assert result.attrs["measurement_levels"]["bolge"] == "nominal"
        with pytest.raises(ValueError):
            define_measurement_level(farm_df, "bolge", "ratio")

    def test_random_numbers(self):
        df = random_numbers(100, distribution="normal", seed=5, mean=10, std=2)
        assert len(df) == 100
        assert 9 < df["rasgele"].mean() < 11
        again = random_numbers(100, distribution="normal", seed=5, mean=10, std=2)
        assert np.allclose(df["rasgele"], again["rasgele"])

    def test_random_numbers_invalid_distribution(self):
        with pytest.raises(ValueError):
            random_numbers(10, distribution="cauchy")

    def test_automatic_recode(self, farm_df):
        result = automatic_recode(farm_df, "bolge")
        assert set(result["bolge_recoded"].dropna().unique()) == {1, 2}
        labels = result.attrs["automatic_recode_labels"]["bolge_recoded"]
        assert labels["Ege"] == 1  # alfabetik sıralama


class TestRegressionExtras:
    def test_weight_estimation_picks_best_power(self):
        rng = np.random.default_rng(72)
        x = rng.uniform(1, 10, 100)
        # sd, x ile büyür → düşük kuvvetli ağırlık iyi uyum verir
        y = 2 + 0.5 * x + rng.normal(0, 1, 100) * x / 3
        result = weight_estimation(x, y)
        assert result["best_power"] in (1.0, 2.0)
        assert result["best_r_squared"] > 0.3
        assert set(result["candidates"]) == {1.0, 2.0}

    def test_weight_estimation_nonpositive_x_raises(self):
        with pytest.raises(ValueError):
            weight_estimation([0, 1, 2, 3, 4], [1, 2, 3, 4, 5])

    def test_two_stage_least_squares_recovers_effect(self):
        rng = np.random.default_rng(73)
        n = 400
        z = rng.normal(0, 1, n)                     # araç
        u = rng.normal(0, 1, n)                     # içsellik kaynağı
        x_endo = 0.7 * z + 0.6 * u + rng.normal(0, 0.3, n)
        y = 1.0 + 2.0 * x_endo + u                  # gerçek beta = 2
        df = pd.DataFrame({"y": y, "x": x_endo, "z": z})
        result = two_stage_least_squares(df, "y", ["x"], ["z"])
        assert result["coefficients"]["x"] == pytest.approx(2.0, abs=0.4)
        assert "const" in result["coefficients"]

    def test_two_stage_least_squares_no_instrument_raises(self):
        df = pd.DataFrame({"y": [1, 2, 3, 4, 5, 6.0], "x": [1, 2, 3, 4, 5, 6.0]})
        with pytest.raises(ValueError):
            two_stage_least_squares(df, "y", ["x"], [])

    def test_twostep_cluster_finds_true_k(self):
        rng = np.random.default_rng(74)
        # 3 iyi ayrılmış küme
        blocks = [rng.normal([0, 0], 0.4, (60, 2)),
                  rng.normal([5, 5], 0.4, (60, 2)),
                  rng.normal([10, 0], 0.4, (60, 2))]
        X = np.vstack(blocks)
        df = pd.DataFrame(X, columns=["v1", "v2"])
        result = twostep_cluster(df, ["v1", "v2"], max_clusters=6, seed=7)
        assert result["n_clusters"] == 3
        assert sum(result["cluster_sizes"].values()) == 180
        assert result["cluster_centroids"].shape == (3, 2)

    def test_twostep_cluster_reproducible(self):
        rng = np.random.default_rng(75)
        df = pd.DataFrame({"a": rng.normal(0, 1, 80), "b": rng.normal(0, 1, 80)})
        r1 = twostep_cluster(df, seed=3)
        r2 = twostep_cluster(df, seed=3)
        assert r1["n_clusters"] == r2["n_clusters"]

    def test_create_time_series_functions(self):
        df = pd.DataFrame({"y": [1.0, 2.0, 4.0, 7.0, 11.0]})
        lag = create_time_series(df, "y", function="lag", periods=1)
        assert np.isnan(lag["y_lag1"].iloc[0])
        assert lag["y_lag1"].iloc[1] == 1.0
        diff = create_time_series(df, "y", function="difference")
        assert diff["y_difference1"].iloc[2] == 2.0
        ma = create_time_series(df, "y", function="moving_average", window=3)
        assert ma["y_moving_average_w3"].iloc[2] == pytest.approx((1 + 2 + 4) / 3)

    def test_create_time_series_invalid_raises(self):
        df = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError):
            create_time_series(df, "y", function="cumprod")

    def test_twostep_command(self):
        runner = CliRunner()
        rng = np.random.default_rng(76)
        X = np.vstack([rng.normal(0, 0.5, (40, 2)), rng.normal(6, 0.5, (40, 2))])
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                         delete=False) as fh:
            pd.DataFrame(X, columns=["v1", "v2"]).to_csv(fh, index=False)
            path = fh.name
        try:
            result = runner.invoke(cli_main, ["twostep", path,
                                              "--kolonlar", "v1,v2"])
            assert result.exit_code == 0
            assert "TwoStep" in result.output
            assert "Seçilen küme sayısı" in result.output
        finally:
            os.unlink(path)


class TestWave4Viz:
    @pytest.fixture
    def plotter(self):
        p = AgristaPlotter()
        yield p
        AgristaPlotter.close()

    def test_pie_chart(self, plotter):
        fig = plotter.pie_chart(["A", "B", "C"], [30, 20, 50])
        assert fig is not None

    def test_pie_chart_negative_raises(self, plotter):
        with pytest.raises(ValueError):
            plotter.pie_chart(["A", "B"], [1, -1])

    def test_line_chart(self, plotter):
        fig = plotter.line_chart([1, 2, 3], [4, 6, 5])
        assert len(fig.axes[0].lines) >= 1

    def test_error_bar(self, plotter, farm_df):
        fig = plotter.error_bar(farm_df, "bolge", "verim")
        assert fig is not None

    def test_pp_plot(self, plotter):
        rng = np.random.default_rng(67)
        fig = plotter.pp_plot(rng.normal(0, 1, 50))
        assert fig is not None

    def test_pp_plot_insufficient_raises(self, plotter):
        with pytest.raises(ValueError):
            plotter.pp_plot([1.0, 2.0])


class TestWave4Cli:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def farm_csv(self, tmp_path, farm_df):
        path = tmp_path / "ciftlik.csv"
        farm_df.to_csv(path, index=False)
        return str(path)

    def test_ctable_command(self, runner, farm_csv):
        result = runner.invoke(cli_main, ["ctable", farm_csv,
                                          "--satirlar", "bolge",
                                          "--deger", "verim",
                                          "--istatistikler", "count,mean"])
        assert result.exit_code == 0
        assert "Özel Tablo" in result.output

    def test_means_command(self, runner, farm_csv):
        result = runner.invoke(cli_main, ["means", farm_csv,
                                          "--yanit", "verim",
                                          "--gruplar", "bolge"])
        assert result.exit_code == 0
        assert "Ortalama Raporu" in result.output

    def test_ratios_command(self, runner, farm_csv):
        result = runner.invoke(cli_main, ["ratios", farm_csv,
                                          "--pay", "verim", "--payda", "alan"])
        assert result.exit_code == 0
        assert "Oran İstatistikleri" in result.output

    def test_knn_command(self, runner, tmp_path):
        rng = np.random.default_rng(68)
        grup = np.repeat(["A", "B"], 40)
        x = np.where(grup == "A", rng.normal(0, 0.5, 80), rng.normal(5, 0.5, 80))
        path = tmp_path / "knn.csv"
        pd.DataFrame({"grup": grup, "x": x}).to_csv(path, index=False)
        result = runner.invoke(cli_main, ["knn", str(path),
                                          "--grup", "grup", "--degiskenler", "x"])
        assert result.exit_code == 0
        assert "Genel doğruluk" in result.output

    def test_mds_command(self, runner, farm_csv):
        result = runner.invoke(cli_main, ["mds", farm_csv,
                                          "--kolonlar", "verim,alan"])
        assert result.exit_code == 0
        assert "Çok Boyutlu Ölçekleme" in result.output

    def test_lifetable_command(self, runner, tmp_path):
        rng = np.random.default_rng(69)
        path = tmp_path / "yasam.csv"
        pd.DataFrame({"zaman": rng.exponential(5, 60),
                      "olay": rng.binomial(1, 0.7, 60)}).to_csv(path, index=False)
        result = runner.invoke(cli_main, ["lifetable", str(path),
                                          "--zaman", "zaman", "--olay", "olay",
                                          "--aralik", "2"])
        assert result.exit_code == 0
        assert "Yaşam Tabloları" in result.output

    def test_cox_command(self, runner, tmp_path):
        rng = np.random.default_rng(70)
        n = 120
        x = rng.normal(0, 1, n)
        path = tmp_path / "cox.csv"
        pd.DataFrame({"zaman": rng.exponential(np.exp(-0.5 * x) * 6, n),
                      "olay": rng.binomial(1, 0.8, n),
                      "x": x}).to_csv(path, index=False)
        result = runner.invoke(cli_main, ["cox", str(path),
                                          "--zaman", "zaman", "--olay", "olay",
                                          "--degiskenler", "x"])
        assert result.exit_code == 0
        assert "Cox Regresyonu" in result.output
        assert "Harrell C" in result.output

    def test_mailing_command(self, runner):
        result = runner.invoke(cli_main, ["mailing",
                                          "--kontrol-yanit", "20",
                                          "--kontrol-boyut", "1000",
                                          "--uygulama-yanit", "40",
                                          "--uygulama-boyut", "1000"])
        assert result.exit_code == 0
        assert "Kampanya/Posta Testi" in result.output

    def test_rfm_command(self, runner, tmp_path):
        rng = np.random.default_rng(71)
        rows = []
        for i in range(20):
            for _ in range(3):
                rows.append({"musteri": f"M{i}",
                             "tarih": (pd.Timestamp("2025-12-31")
                                       - pd.Timedelta(days=int(rng.integers(1, 300)))).strftime("%Y-%m-%d"),
                             "tutar": float(rng.uniform(10, 500))})
        path = tmp_path / "rfm.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        result = runner.invoke(cli_main, ["rfm", str(path),
                                          "--musteri", "musteri",
                                          "--tarih", "tarih",
                                          "--tutar", "tutar"])
        assert result.exit_code == 0
        assert "RFM Analizi" in result.output
