"""P1 testleri — protection ve genetics modülleri."""

import numpy as np
import pandas as pd
import pytest

from agrista.protection import (
    abbott_efficiency,
    probit_dose_response,
    loglogistic_dose_response,
    audpc,
    disease_progress_fit,
)
from agrista.genetics import (
    path_analysis,
    pca_analysis,
    cluster_genotypes,
    mahalanobis_d2,
    heritability,
    ammi_analysis,
    gge_biplot,
    stability_indices,
)


class TestProtection:
    def test_abbott_formula(self):
        result = abbott_efficiency(80, 20)
        assert result["efficacy_pct"] == pytest.approx(75.0)
        assert result["interpretation"] == "Orta etkinlik"

    def test_abbott_zero_control_raises(self):
        with pytest.raises(ValueError):
            abbott_efficiency(0, 10)

    def test_probit_lc50_recovery(self):
        rng = np.random.default_rng(40)
        doses = np.array([1, 5, 10, 50, 100.0])
        p_true = 1 / (1 + np.exp(-(np.log10(doses) - 1) * 3))
        dead = rng.binomial(40, np.clip(p_true, 0.02, 0.98))
        result = probit_dose_response(doses, dead, [40] * 5)
        assert result["lc50"] == pytest.approx(10, rel=0.6)
        assert result["dose_significant"] is True
        assert result["lc90"] > result["lc50"]

    def test_loglogistic_gr50_recovery(self):
        rng = np.random.default_rng(41)
        dose = np.repeat([0, 1, 5, 10, 50, 100], 4)
        resp = np.repeat([95, 90, 60, 30, 8, 3], 4) + rng.normal(0, 2, 24)
        result = loglogistic_dose_response(dose, resp)
        assert result["gr50"] == pytest.approx(7, rel=0.6)
        assert result["r_squared"] > 0.95

    def test_audpc_trapezoid(self):
        # Sabit %100 şiddet, 10 gün → AUDPC = 1000
        result = audpc([0, 5, 10], [100, 100, 100])
        assert result["audpc"] == pytest.approx(1000)
        assert result["relative_audpc"] == pytest.approx(100)

    def test_audpc_too_few_raises(self):
        with pytest.raises(ValueError):
            audpc([1], [50])

    def test_disease_progress_fit_logistic(self):
        t = np.arange(0, 31, 2)
        severity = 100 / (1 + np.exp(-0.25 * (t - 15)))
        result = disease_progress_fit(t, severity, model_type="logistic")
        assert result["r_squared"] > 0.99
        assert result["rate_r"] == pytest.approx(0.25, rel=0.2)

    def test_disease_progress_unknown_model_raises(self):
        with pytest.raises(ValueError):
            disease_progress_fit([1, 2, 3, 4], [1, 2, 3, 4], model_type="üstel")


class TestGeneticsMultivariate:
    @pytest.fixture
    def causal_df(self):
        rng = np.random.default_rng(42)
        n = 150
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        y = 2.0 * x1 + 0.5 * x2 + rng.normal(0, 0.5, n)
        return pd.DataFrame({"x1": x1, "x2": x2, "y": y})

    def test_path_analysis_recovers_dominant_effect(self, causal_df):
        result = path_analysis(causal_df, "y", ["x1", "x2"])
        assert abs(result["direct_effects"]["x1"]) > abs(result["direct_effects"]["x2"])
        assert result["r_squared"] > 0.9

    def test_pca_structure(self, causal_df):
        result = pca_analysis(causal_df, ["x1", "x2", "y"])
        ratios = result["explained_variance_ratio"]
        assert list(ratios.values())[0] > 0.5
        assert result["loadings"].shape[0] == 3
        assert result["scores"].shape[0] == 150

    def test_cluster_recovers_two_groups(self):
        rng = np.random.default_rng(43)
        g1 = rng.normal(0, 0.5, (30, 2))
        g2 = rng.normal(6, 0.5, (30, 2))
        df = pd.DataFrame(np.vstack([g1, g2]), columns=["a", "b"])
        result = cluster_genotypes(df, ["a", "b"], n_clusters=2)
        assert result["n_clusters"] == 2
        labels = result["labels"].to_numpy()
        # İlk 30 aynı kümede, sonraki 30 diğer kümede
        assert len(set(labels[:30])) == 1
        assert len(set(labels[30:])) == 1
        assert labels[0] != labels[-1]

    def test_mahalanobis_d2_groups(self):
        rng = np.random.default_rng(44)
        df = pd.DataFrame({
            "g": ["A"] * 40 + ["B"] * 40,
            "x": np.concatenate([rng.normal(0, 1, 40), rng.normal(5, 1, 40)]),
            "y": rng.normal(0, 1, 80),
        })
        result = mahalanobis_d2(df, ["x", "y"], group_col="g")
        # Havuz kovaryansı ayrık kümelerde şişer; yine de belirgin fark beklenir
        assert result["d2_matrix"].loc["A", "B"] > 2.5
        assert result["max_pair"]["d2"] == pytest.approx(result["d2_matrix"].loc["A", "B"])

    def test_heritability_bounds(self):
        result = heritability(4.0, 2.0, n_reps=3)
        assert 0 < result["h2_broad_sense"] < 1
        assert result["h2_broad_sense"] == pytest.approx(4 / (4 + 2 / 3))

    def test_heritability_negative_raises(self):
        with pytest.raises(ValueError):
            heritability(-1.0, 2.0)


class TestGeneticsStability:
    @pytest.fixture
    def met_df(self):
        rng = np.random.default_rng(45)
        geno_eff = {"G1": 0.0, "G2": 1.0, "G3": 0.5, "G4": -0.5}
        env_eff = {"E1": 0.0, "E2": 1.0, "E3": 2.0}
        rows = []
        for g, gv in geno_eff.items():
            for e, ev in env_eff.items():
                inter = 1.0 if (g == "G1" and e == "E3") else 0.0
                for _ in range(2):
                    rows.append({"gen": g, "cev": e,
                                 "verim": 5 + gv + ev + inter + rng.normal(0, 0.1)})
        return pd.DataFrame(rows)

    def test_ammi_structure(self, met_df):
        result = ammi_analysis(met_df, "gen", "cev", "verim")
        assert result["n_genotypes"] == 4
        assert result["n_environments"] == 3
        assert result["anova"]["ss_genotype"] > 0
        # G1×E3 etkileşimi IPCA1'de baskın olmalı
        assert result["ipca_explained_variance"]["IPCA1"] > 0.5
        assert abs(result["genotype_scores"]["G1"]["IPCA1"]) > 0.1

    def test_gge_biplot_variance(self, met_df):
        result = gge_biplot(met_df, "gen", "cev", "verim")
        total = result["pc_explained_variance"]["PC1"] + result["pc_explained_variance"]["PC2"]
        assert total <= 1.0 + 1e-9
        assert result["pc_explained_variance"]["PC1"] > 0.5

    def test_ammi_unbalanced_raises(self):
        df = pd.DataFrame({
            "gen": ["G1", "G1", "G2"],
            "cev": ["E1", "E2", "E1"],
            "verim": [1.0, 2.0, 3.0],
        })
        with pytest.raises(ValueError):
            ammi_analysis(df, "gen", "cev", "verim")

    def test_stability_indices(self, met_df):
        result = stability_indices(met_df, "gen", "cev", "verim")
        assert result["method"] == "Finlay-Wilkinson"
        assert set(result["genotypes"].keys()) == {"G1", "G2", "G3", "G4"}
        assert result["most_stable"] in result["genotypes"]
