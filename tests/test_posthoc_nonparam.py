"""P0 testleri — çoklu karşılaştırma, varsayım ve parametrik olmayan testler."""

import numpy as np
import pandas as pd
import pytest

from agrista.analysis import (
    posthoc_tukey,
    posthoc_duncan,
    normality_test,
    homogeneity_test,
    mann_whitney_u,
    kruskal_wallis,
    wilcoxon_test,
    friedman_test,
)
from agrista.experimental import latin_square_anova, factorial_anova


@pytest.fixture
def three_group_df():
    rng = np.random.default_rng(20)
    return pd.DataFrame({
        "grup": np.repeat(["A", "B", "C"], 20),
        "y": np.concatenate([
            rng.normal(5, 1, 20),
            rng.normal(6.5, 1, 20),
            rng.normal(9, 1, 20),
        ]),
    })


class TestPosthoc:
    def test_tukey_detects_differences(self, three_group_df):
        result = posthoc_tukey(three_group_df, "y", "grup")
        assert len(result["comparisons"]) == 3
        assert len(result["significant_pairs"]) == 3
        assert result["group_means"]["C"] > result["group_means"]["A"]

    def test_tukey_no_false_positive(self):
        rng = np.random.default_rng(21)
        df = pd.DataFrame({
            "grup": np.repeat(["A", "B"], 30),
            "y": rng.normal(5, 1, 60),
        })
        result = posthoc_tukey(df, "y", "grup")
        assert len(result["significant_pairs"]) == 0

    def test_duncan_letter_groups(self, three_group_df):
        result = posthoc_duncan(three_group_df, "y", "grup")
        labels = result["group_labels"]
        # Üç ayrık grup → üç farklı harf
        assert len({labels["A"], labels["B"], labels["C"]}) == 3
        assert result["sorted_means"]["C"] > result["sorted_means"]["A"]

    def test_duncan_identical_groups_share_letter(self):
        rng = np.random.default_rng(22)
        df = pd.DataFrame({
            "grup": np.repeat(["A", "B"], 30),
            "y": rng.normal(5, 1, 60),
        })
        result = posthoc_duncan(df, "y", "grup")
        assert result["group_labels"]["A"] == result["group_labels"]["B"]

    def test_duncan_chain_letters(self):
        # Zincir: A-B farkı ve B-C farkı önemsiz, A-C farkı anlamlı → B 'ab' alır
        rng = np.random.default_rng(1)
        df = pd.DataFrame({
            "grup": np.repeat(["A", "B", "C"], 30),
            "y": np.concatenate([
                rng.normal(5.00, 1, 30),
                rng.normal(5.25, 1, 30),
                rng.normal(5.50, 1, 30),
            ]),
        })
        result = posthoc_duncan(df, "y", "grup")
        labels = result["group_labels"]
        # B her iki harfi de taşımalı; A ve C ayrık
        assert set(labels["B"]) == {"a", "b"}
        assert labels["C"] == "a"
        assert labels["A"] == "b"

    def test_duncan_reports_anova_context(self, three_group_df):
        result = posthoc_duncan(three_group_df, "y", "grup")
        assert result["anova_p_value"] < 0.05
        assert result["anova_f_statistic"] > 0


class TestAssumptions:
    def test_shapiro_normal(self):
        rng = np.random.default_rng(23)
        result = normality_test(rng.normal(0, 1, 100))
        assert result["normal_at_alpha"] is True

    def test_shapiro_non_normal(self):
        rng = np.random.default_rng(24)
        mixed = np.concatenate([rng.normal(-3, 0.5, 50), rng.normal(3, 0.5, 50)])
        result = normality_test(mixed)
        assert result["normal_at_alpha"] is False

    def test_shapiro_too_small_raises(self):
        with pytest.raises(ValueError):
            normality_test([1.0, 2.0])

    def test_levene_homogeneous(self):
        rng = np.random.default_rng(25)
        result = homogeneity_test(rng.normal(0, 1, 50), rng.normal(5, 1, 50))
        assert result["homogeneous_at_alpha"] is True

    def test_levene_heterogeneous(self):
        rng = np.random.default_rng(26)
        result = homogeneity_test(rng.normal(0, 1, 80), rng.normal(0, 5, 80))
        assert result["homogeneous_at_alpha"] is False


class TestNonParametric:
    def test_mann_whitney_detects_shift(self):
        rng = np.random.default_rng(27)
        result = mann_whitney_u(rng.normal(5, 1, 40), rng.normal(7, 1, 40))
        assert result["significant_at_005"] is True
        assert result["group2_median"] > result["group1_median"]

    def test_kruskal_wallis_detects_difference(self):
        rng = np.random.default_rng(28)
        result = kruskal_wallis(
            rng.normal(5, 1, 30), rng.normal(5, 1, 30), rng.normal(9, 1, 30)
        )
        assert result["significant_at_005"] is True
        assert result["degrees_of_freedom"] == 2

    def test_wilcoxon_paired(self):
        rng = np.random.default_rng(29)
        before = rng.normal(10, 1, 25)
        after = before + rng.normal(1.5, 0.5, 25)
        result = wilcoxon_test(before, after)
        assert result["significant_at_005"] is True
        # fark = önce - sonra olduğundan negatif beklenir
        assert result["median_difference"] < 0

    def test_wilcoxon_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            wilcoxon_test([1, 2, 3], [1, 2])

    def test_kruskal_wallis_single_value_group(self):
        # Tek gözemli grup hata/NaN üretmemeli; sonuç sonlu olmalı
        result = kruskal_wallis([5.0, 6.0, 7.0], [9.0, 10.0, 11.0], [15.0])
        assert np.isfinite(result["h_statistic"])
        assert np.isfinite(result["p_value"])

    def test_friedman_detects_treatment(self):
        rng = np.random.default_rng(30)
        mat = rng.normal(0, 1, (15, 3)) + [0, 1.5, 3.0]
        result = friedman_test(mat)
        assert result["significant_at_005"] is True
        assert 0 <= result["kendall_w"] <= 1

    def test_friedman_no_difference(self):
        rng = np.random.default_rng(31)
        mat = rng.normal(0, 1, (20, 3))
        result = friedman_test(mat)
        assert result["significant_at_005"] is False


class TestLatinSquareAndFactorial:
    @pytest.fixture
    def latin_square_df(self):
        rng = np.random.default_rng(32)
        effects = {"T1": 0, "T2": 1, "T3": 2, "T4": 3}
        square = [
            ["T1", "T2", "T3", "T4"],
            ["T2", "T3", "T4", "T1"],
            ["T3", "T4", "T1", "T2"],
            ["T4", "T1", "T2", "T3"],
        ]
        rows = []
        for r in range(4):
            for c in range(4):
                rows.append({
                    "satir": f"R{r}",
                    "sutun": f"C{c}",
                    "uyg": square[r][c],
                    "verim": effects[square[r][c]] + 0.2 * r + 0.1 * c
                             + rng.normal(0, 0.05),
                })
        return pd.DataFrame(rows)

    def test_latin_square_detects_treatment(self, latin_square_df):
        result = latin_square_anova(latin_square_df, "verim", "satir", "sutun", "uyg")
        assert result["treatment_effect"]["p_value"] < 0.001
        assert result["treatment_statistics"]["T4"]["mean"] > \
               result["treatment_statistics"]["T1"]["mean"]

    def test_factorial_anova_main_and_interaction(self):
        rng = np.random.default_rng(33)
        # a ana etkisi yok, b ana etkisi var, etkileşim var
        rows = []
        for a in ["a1", "a2"]:
            for b in ["b1", "b2"]:
                base = 5 + (2 if b == "b2" else 0)
                if a == "a2" and b == "b2":
                    base += 3  # etkileşim
                rows.extend([{"a": a, "b": b, "y": base + rng.normal(0, 0.2)}
                             for _ in range(12)])
        df = pd.DataFrame(rows)
        result = factorial_anova(df, "y", ["a", "b"])
        assert result["main_effects"]["b"]["significant_at_005"] is True
        assert result["main_effects"]["a"]["significant_at_005"] is False
        assert result["interactions"]["a × b"]["significant_at_005"] is True
        assert result["n_obs"] == 48

    def test_factorial_single_factor_raises(self):
        df = pd.DataFrame({"a": ["x", "y"], "y": [1.0, 2.0]})
        with pytest.raises(ValueError):
            factorial_anova(df, "y", ["a"])
