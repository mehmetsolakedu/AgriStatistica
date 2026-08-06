"""Premium Program üçüncü dalga testleri — Multinomial/Ordinal Lojistik Regresyon,
Discriminant Analizi, Uyuşum (Correspondence) Analizi."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from agrista.analysis import (
    multinomial_logistic_regression,
    ordinal_logistic_regression,
    discriminant_analysis,
    correspondence_analysis,
)


def _make_multinomial_df(n=300, seed=42):
    """x'e göre ayrışan üç kategorili yapay veri (kategori 'A' referans)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    z = rng.normal(0, 1, n)
    # kategoriler: 0=A (orta), 1=B (x arttıkça), 2=C (x azaldıkça)
    latent = x
    p_b = 1 / (1 + np.exp(-latent))
    p_c = 1 / (1 + np.exp(latent))
    p_b, p_c = 0.45 * p_b, 0.45 * p_c
    p_a = 1 - p_b - p_c
    y = np.array([rng.choice(["A", "B", "C"], p=[pa, pb, pc])
                  for pa, pb, pc in zip(p_a, p_b, p_c)])
    return pd.DataFrame({"yanit": y, "x": x, "z": z})


class TestMultinomialLogistic:
    @pytest.fixture
    def mlogit_df(self):
        return _make_multinomial_df()

    def test_model_significant_and_accurate(self, mlogit_df):
        result = multinomial_logistic_regression(mlogit_df, "yanit", ["x", "z"])
        assert result["significant_at_005"] is True
        assert result["reference_category"] == "A"
        assert result["categories"] == ["A", "B", "C"]
        assert result["classification_accuracy"] > 0.5
        assert result["degrees_of_freedom"] == 4  # 2 değişken × (3-1) kategori

    def test_coefficient_signs(self, mlogit_df):
        result = multinomial_logistic_regression(mlogit_df, "yanit", ["x"])
        # B kategorisi x ile artar, C kategorisi x ile azalır (A'ya göre)
        assert result["coefficients"]["B"]["x"] > 0
        assert result["coefficients"]["C"]["x"] < 0
        # Odds oranı katsayının üsteli
        for cat in ("B", "C"):
            assert result["odds_ratios"][cat]["x"] == pytest.approx(
                np.exp(result["coefficients"][cat]["x"]))

    def test_missing_column_raises(self, mlogit_df):
        with pytest.raises(ValueError):
            multinomial_logistic_regression(mlogit_df, "yanit", ["x", "yok"])

    def test_single_category_raises(self):
        df = pd.DataFrame({"y": ["A"] * 40, "x": np.arange(40.0)})
        with pytest.raises(ValueError):
            multinomial_logistic_regression(df, "y", ["x"])

    def test_non_numeric_predictor_raises(self, mlogit_df):
        mlogit_df["kat"] = np.where(mlogit_df["x"] > 0, "h", "l")
        with pytest.raises(ValueError):
            multinomial_logistic_regression(mlogit_df, "yanit", ["kat"])


class TestOrdinalLogistic:
    @pytest.fixture
    def ologit_df(self):
        rng = np.random.default_rng(43)
        n = 250
        x = rng.normal(0, 1, n)
        latent = 1.0 * x + rng.logistic(0, 1, n)
        codes = np.digitize(latent, [-0.8, 0.8])
        y = pd.Series(codes).map({0: "düşük", 1: "orta", 2: "yüksek"})
        return pd.DataFrame({"verim": y, "x": x})

    def test_coefficient_positive_and_significant(self, ologit_df):
        result = ordinal_logistic_regression(ologit_df, "verim", ["x"])
        assert result["significant_at_005"] is True
        assert result["coefficients"]["x"] > 0.5  # gerçek değer 1.0'a yakın
        assert result["odds_ratios"]["x"] == pytest.approx(
            np.exp(result["coefficients"]["x"]))
        assert len(result["thresholds"]) == 2

    def test_thresholds_ascending(self, ologit_df):
        result = ordinal_logistic_regression(ologit_df, "verim", ["x"])
        theta = list(result["thresholds"].values())
        assert theta == sorted(theta)

    def test_two_categories_raises(self):
        rng = np.random.default_rng(44)
        df = pd.DataFrame({"y": rng.integers(0, 2, 60),
                           "x": rng.normal(size=60)})
        with pytest.raises(ValueError):
            ordinal_logistic_regression(df, "y", ["x"])

    def test_noise_predictor_not_significant(self):
        rng = np.random.default_rng(45)
        n = 300
        latent = rng.normal(0, 1, n)  # yalnızca gürültü
        df = pd.DataFrame({"y": np.digitize(latent, [-0.7, 0.7]),
                           "x": rng.normal(size=n)})
        result = ordinal_logistic_regression(df, "y", ["x"])
        assert result["significant_at_005"] is False


class TestDiscriminant:
    @pytest.fixture
    def separated_df(self):
        rng = np.random.default_rng(46)
        n = 60
        rows = []
        for i, (grp, mu) in enumerate([("G1", 0.0), ("G2", 4.0), ("G3", 8.0)]):
            rows.append(pd.DataFrame({
                "grup": grp,
                "v1": rng.normal(mu, 0.8, n),
                "v2": rng.normal(0.5 * mu, 0.8, n),
            }))
        return pd.concat(rows, ignore_index=True)

    def test_well_separated_groups(self, separated_df):
        result = discriminant_analysis(separated_df, "grup", ["v1", "v2"])
        assert result["significant_at_005"] is True
        assert result["wilks_lambda"] < 0.1
        assert result["overall_accuracy"] > 0.95
        assert len(result["eigenvalues"]) == 2       # min(g-1, p)
        assert result["degrees_of_freedom"] == 4     # p * (g - 1)
        assert sum(result["percent_of_variance"]) == pytest.approx(100.0, abs=1e-6)

    def test_canonical_correlations_bounded(self, separated_df):
        result = discriminant_analysis(separated_df, "grup", ["v1", "v2"])
        assert all(0 < r < 1 for r in result["canonical_correlations"])
        # İlk kanonik fonksiyon baskın
        assert result["eigenvalues"] == sorted(result["eigenvalues"], reverse=True)

    def test_overlapping_groups_not_significant(self):
        rng = np.random.default_rng(47)
        df = pd.DataFrame({
            "grup": ["A"] * 80 + ["B"] * 80,
            "x": rng.normal(0, 1, 160),
        })
        result = discriminant_analysis(df, "grup", ["x"])
        assert result["significant_at_005"] is False
        assert result["wilks_lambda"] > 0.9
        assert result["overall_accuracy"] < 0.65

    def test_centroids_match_group_means(self, separated_df):
        result = discriminant_analysis(separated_df, "grup", ["v1", "v2"])
        for grp in ("G1", "G2", "G3"):
            sub = separated_df[separated_df["grup"] == grp]
            assert result["group_centroids"][grp]["v1"] == pytest.approx(
                sub["v1"].mean())

    def test_single_group_raises(self):
        df = pd.DataFrame({"grup": ["A"] * 20, "x": np.arange(20.0)})
        with pytest.raises(ValueError):
            discriminant_analysis(df, "grup", ["x"])

    def test_insufficient_data_raises(self):
        df = pd.DataFrame({"grup": ["A", "A", "B", "B"],
                           "x1": [1.0, 2.0, 3.0, 4.0],
                           "x2": [1.0, 2.0, 3.0, 4.0],
                           "x3": [1.0, 2.0, 3.0, 4.0]})
        with pytest.raises(ValueError):
            discriminant_analysis(df, "grup", ["x1", "x2", "x3"])


class TestCorrespondence:
    @pytest.fixture
    def associated_df(self):
        # Çeşit x pazar tercihi: belirgin ilişki yapısı
        rows = (["c1"] * 40 + ["c2"] * 10) + (["c1"] * 10 + ["c2"] * 40)
        cols = (["m1"] * 40 + ["m2"] * 10) + (["m1"] * 10 + ["m2"] * 40)
        return pd.DataFrame({"cesit": rows, "pazar": cols})

    def test_inertia_matches_chi_square(self, associated_df):
        result = correspondence_analysis(associated_df, "cesit", "pazar")
        table = pd.crosstab(associated_df["cesit"], associated_df["pazar"])
        chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
        n = table.values.sum()
        assert result["chi_square"] == pytest.approx(chi2, rel=1e-9)
        assert result["total_inertia"] == pytest.approx(chi2 / n, rel=1e-9)
        assert result["degrees_of_freedom"] == dof
        assert result["p_value"] == pytest.approx(p, rel=1e-6)
        assert result["independence_rejected"] is True

    def test_explained_percent_sums_to_100(self, associated_df):
        result = correspondence_analysis(associated_df, "cesit", "pazar")
        assert sum(result["explained_percent"]) == pytest.approx(100.0, abs=1e-6)

    def test_coordinates_separate_associated_categories(self, associated_df):
        result = correspondence_analysis(associated_df, "cesit", "pazar")
        rc = result["row_coordinates"]
        cc = result["column_coordinates"]
        # c1-m1 ve c2-m2 ilişkili çiftler aynı bölgede, çapraz çiftler uzakta
        d_assoc_1 = np.linalg.norm(rc.loc["c1"] - cc.loc["m1"])
        d_cross_1 = np.linalg.norm(rc.loc["c1"] - cc.loc["m2"])
        assert d_assoc_1 < d_cross_1
        assert result["n_dimensions"] == 1  # 2x2 tablo: tek boyut

    def test_independent_table_not_rejected(self):
        rows = ["A"] * 50 + ["B"] * 100
        cols = ["x"] * 20 + ["y"] * 30 + ["x"] * 40 + ["y"] * 60
        df = pd.DataFrame({"satir": rows, "sutun": cols})
        result = correspondence_analysis(df, "satir", "sutun")
        assert result["independence_rejected"] is False

    def test_quality_bounded(self, associated_df):
        result = correspondence_analysis(associated_df, "cesit", "pazar")
        for q in list(result["row_quality"].values()) \
                + list(result["column_quality"].values()):
            assert 0 <= q <= 1.0 + 1e-9

    def test_3x3_table_two_dimensions(self):
        rng = np.random.default_rng(48)
        cats_r = rng.choice(["r1", "r2", "r3"], 200)
        cats_c = rng.choice(["c1", "c2", "c3"], 200)
        df = pd.DataFrame({"satir": cats_r, "sutun": cats_c})
        result = correspondence_analysis(df, "satir", "sutun", n_dims=5)
        # min(I,J)-1 = 2 boyutla sınırlanmalı
        assert result["n_dimensions"] == 2
        assert result["row_coordinates"].shape == (3, 2)
        assert result["column_coordinates"].shape == (3, 2)

    def test_single_category_raises(self):
        df = pd.DataFrame({"satir": ["a"] * 20, "sutun": ["x"] * 20})
        with pytest.raises(ValueError):
            correspondence_analysis(df, "satir", "sutun")
