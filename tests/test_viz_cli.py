"""agrista.viz ve agrista.cli modülleri testleri."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from agrista.viz import AgristaPlotter
from agrista.cli import main as cli_main


@pytest.fixture(autouse=True)
def close_figures():
    yield
    AgristaPlotter.close()


class TestAgristaPlotter:
    @pytest.fixture
    def plotter(self):
        return AgristaPlotter()

    @pytest.fixture
    def numeric_df(self):
        rng = np.random.default_rng(11)
        return pd.DataFrame({
            "x": rng.normal(0, 1, 50),
            "y": rng.normal(0, 1, 50),
            "grup": np.random.choice(["A", "B"], 50),
        })

    def test_histogram_returns_figure(self, plotter):
        fig = plotter.histogram([1, 2, 2, 3, 3, 3], bins=5)
        assert fig is not None
        assert len(fig.axes) >= 1

    def test_scatter_with_regression_line(self, plotter):
        x = np.arange(20, dtype=float)
        fig = plotter.scatter(x, 2 * x + 1, regression_line=True)
        ax = fig.axes[0]
        # Nokta kümesi + regresyon çizgisi
        assert len(ax.lines) >= 1

    def test_scatter_insufficient_data_raises(self, plotter):
        with pytest.raises(ValueError):
            plotter.scatter([1.0], [2.0])

    def test_correlation_heatmap(self, plotter, numeric_df):
        fig = plotter.correlation_heatmap(numeric_df)
        assert fig is not None

    def test_bar_chart(self, plotter):
        fig = plotter.bar_chart(["A", "B"], [1.0, 2.0])
        assert fig is not None

    def test_boxplot_by_group(self, plotter, numeric_df):
        fig = plotter.boxplot(numeric_df, x_col="grup", y_col="x")
        assert fig is not None

    def test_save_specific_figure(self, plotter, tmp_path):
        fig1 = plotter.histogram([1, 2, 3], title="bir")
        fig2 = plotter.histogram([4, 5, 6], title="iki")
        assert fig1 is not None and fig2 is not None
        path = tmp_path / "bir.png"
        plotter.save(str(path), fig=fig1)
        assert path.exists() and path.stat().st_size > 0

    def test_time_series(self, plotter):
        fig = plotter.time_series([1, 3, 2, 5, 4, 6])
        assert fig is not None


class TestCli:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def csv_file(self, tmp_path):
        rng = np.random.default_rng(12)
        df = pd.DataFrame({
            "sulama": rng.uniform(3000, 12000, 40),
            "verim": rng.normal(5, 1, 40),
        })
        path = tmp_path / "veri.csv"
        df.to_csv(path, index=False)
        return str(path)

    def test_bare_launch_menu_and_exit(self, runner):
        result = runner.invoke(cli_main, [], input="0\n")
        assert result.exit_code == 0
        assert "AGRISTA — Ana Menü" in result.output
        assert "Betimsel İstatistikler" in result.output
        assert "Görüşürüz" in result.output

    def test_logo_banner_rendered(self, runner):
        result = runner.invoke(cli_main, [], input="0\n")
        # ASCII blok logo karakterleri çıktıda bulunmalı
        assert "█" in result.output and "╚" in result.output

    def test_explicit_menu_command(self, runner):
        result = runner.invoke(cli_main, ["menu"], input="0\n")
        assert result.exit_code == 0
        assert "Ana Menü" in result.output

    def test_menu_invalid_choice(self, runner):
        result = runner.invoke(cli_main, [], input="99\n0\n")
        assert "Geçersiz seçim" in result.output

    def test_menu_info_flow(self, runner, csv_file):
        # Kategori 1 (Dosya) → işlem 1 → dosya → çıkış
        result = runner.invoke(cli_main, [], input=f"1\n1\n{csv_file}\n0\n")
        assert "Satır sayısı: 40" in result.output
        assert "Görüşürüz" in result.output

    def test_menu_normality_flow(self, runner, csv_file):
        # Kategori 2 (Betimsel) → işlem 5 (normallik) → dosya → sütun
        result = runner.invoke(cli_main, [], input=f"2\n5\n{csv_file}\nverim\n0\n")
        assert "Shapiro-Wilk" in result.output

    def test_menu_correlation_flow_with_method(self, runner, csv_file):
        # Kategori 4 (Korelasyon) → işlem 1 → dosya → yöntem
        result = runner.invoke(cli_main, [],
                               input=f"4\n1\n{csv_file}\nspearman\n0\n")
        assert "Korelasyon Analizi (Spearman)" in result.output

    def test_premium_categories_present(self, runner):
        result = runner.invoke(cli_main, [], input="0\n")
        for kategori in ("Dosya", "Betimsel İstatistikler",
                         "Ortalamaların Karşılaştırılması", "Korelasyon",
                         "Dönüşüm (Transform)", "Ölçek (Scale)",
                         "Boyut İndirgeme", "Regresyon", "Sınıflandırma",
                         "Kestirim (Forecasting)",
                         "Yaşam Analizi (Survival)", "Kalite Kontrol",
                         "Bootstrapping", "ROC Eğrisi", "Loglinear",
                         "Uzman Branş Modülleri"):
            assert kategori in result.output

    def test_menu_pareto_flow(self, runner, tmp_path):
        cats = ["a"] * 30 + ["b"] * 15 + ["c"] * 5
        df = pd.DataFrame({"hata": cats})
        path = tmp_path / "hatalar.csv"
        df.to_csv(path, index=False)
        # Kategori 12 (Kalite Kontrol) → işlem 2 (Pareto)
        result = runner.invoke(cli_main, [], input=f"12\n2\n{path}\nhata\n0\n")
        assert "Pareto Analizi" in result.output
        assert "vital few" in result.output

    def test_menu_bootstrap_flow(self, runner, csv_file):
        # Kategori 13 (Bootstrapping) → işlem 1
        result = runner.invoke(cli_main, [], input=f"13\n1\n{csv_file}\nverim\n0\n")
        assert "Bootstrap Güven Aralığı" in result.output
        assert "%95 GA" in result.output

    def test_menu_transform_compute_flow(self, runner, csv_file, tmp_path):
        out_csv = tmp_path / "donusmus.csv"
        # Kategori 5 (Dönüşüm) → işlem 1 (compute) → dosya → ifade → ad → çıktı → çıkış
        inputs = f"5\n1\n{csv_file}\nsulama * 2\nsulama_cift\n{out_csv}\n0\n"
        result = runner.invoke(cli_main, [], input=inputs)
        assert "Kaydedildi" in result.output
        assert out_csv.exists()
        saved = pd.read_csv(out_csv)
        assert "sulama_cift" in saved.columns
        assert saved["sulama_cift"].iloc[0] == pytest.approx(saved["sulama"].iloc[0] * 2)

    def test_menu_paired_flow(self, runner, tmp_path):
        rng = np.random.default_rng(14)
        once = rng.normal(10, 1, 25)
        df = pd.DataFrame({"once": once, "sonra": once + 1.5})
        path = tmp_path / "eslesmis.csv"
        df.to_csv(path, index=False)
        # Kategori 3 → işlem 3 (eşleştirilmiş t)
        result = runner.invoke(cli_main, [],
                               input=f"3\n3\n{path}\nonce\nsonra\n0\n")
        assert "Eşleştirilmiş İki Örneklem T-Testi" in result.output
        assert "Anlamlı fark var" in result.output

    def test_menu_back_navigation(self, runner):
        result = runner.invoke(cli_main, [], input="1\nb\n0\n")
        assert result.exit_code == 0
        assert "Ana menüye dön" in result.output
        assert "Görüşürüz" in result.output
        # Analiz çalıştırılmamış olmalı
        assert "Satır sayısı" not in result.output

    def test_menu_missing_file_reprompts(self, runner, csv_file):
        result = runner.invoke(cli_main, [],
                               input=f"1\n1\n/olmayan.csv\n{csv_file}\n0\n")
        assert "Dosya bulunamadı" in result.output
        assert "Satır sayısı: 40" in result.output

    def test_version(self, runner):
        result = runner.invoke(cli_main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_info_command(self, runner, csv_file):
        result = runner.invoke(cli_main, ["info", csv_file])
        assert result.exit_code == 0
        assert "Satır sayısı: 40" in result.output

    def test_info_missing_file(self, runner):
        result = runner.invoke(cli_main, ["info", "/olmayan.csv"])
        assert "Dosya bulunamadı" in result.output

    def test_info_unsupported_format(self, runner, tmp_path):
        path = tmp_path / "veri.txt"
        path.write_text("a,b\n1,2")
        result = runner.invoke(cli_main, ["info", str(path)])
        assert "Desteklenmeyen" in result.output

    def test_describe_command(self, runner, csv_file):
        result = runner.invoke(cli_main, ["describe", csv_file])
        assert result.exit_code == 0
        assert "verim" in result.output

    def test_corr_command(self, runner, csv_file):
        result = runner.invoke(cli_main, ["corr", csv_file])
        assert result.exit_code == 0
        assert "Korelasyon Analizi" in result.output

    def test_corr_unsupported_format(self, runner, tmp_path):
        path = tmp_path / "veri.txt"
        path.write_text("a,b\n1,2")
        result = runner.invoke(cli_main, ["corr", str(path)])
        assert "Desteklenmeyen" in result.output

    @pytest.fixture
    def group_csv(self, tmp_path):
        rng = np.random.default_rng(13)
        df = pd.DataFrame({
            "grup": np.repeat(["A", "B", "C"], 20),
            "verim": np.concatenate([
                rng.normal(5, 1, 20), rng.normal(6.5, 1, 20), rng.normal(9, 1, 20)
            ]),
        })
        path = tmp_path / "gruplar.csv"
        df.to_csv(path, index=False)
        return str(path)

    def test_tukey_command(self, runner, group_csv):
        result = runner.invoke(cli_main, ["tukey", group_csv,
                                          "--yanit", "verim", "--grup", "grup"])
        assert result.exit_code == 0
        assert "Tukey HSD" in result.output
        assert "Anlamlı çiftler" in result.output

    def test_tukey_missing_file(self, runner):
        result = runner.invoke(cli_main, ["tukey", "/olmayan.csv",
                                          "--yanit", "y", "--grup", "g"])
        assert "Dosya bulunamadı" in result.output

    def test_audpc_command(self, runner, tmp_path):
        df = pd.DataFrame({"zaman": [0, 7, 14, 21], "siddet": [5, 20, 45, 70]})
        path = tmp_path / "hastalik.csv"
        df.to_csv(path, index=False)
        result = runner.invoke(cli_main, ["audpc", str(path)])
        assert result.exit_code == 0
        assert "AUDPC" in result.output
        assert "Bağıl AUDPC" in result.output

    def test_audpc_missing_column(self, runner, tmp_path):
        df = pd.DataFrame({"gun": [0, 7], "oran": [5, 20]})
        path = tmp_path / "hastalik2.csv"
        df.to_csv(path, index=False)
        result = runner.invoke(cli_main, ["audpc", str(path)])
        assert "Sütun bulunamadı" in result.output

    def test_dea_command(self, runner, tmp_path):
        inputs = pd.DataFrame({"isletme": ["C1", "C2", "C3", "C4"],
                               "arazi": [10, 20, 15, 12],
                               "emek": [5, 8, 6, 4]})
        outputs = pd.DataFrame({"isletme": ["C1", "C2", "C3", "C4"],
                                "urun": [100, 150, 140, 130]})
        in_path = tmp_path / "girdi.csv"
        out_path = tmp_path / "cikti.csv"
        inputs.to_csv(in_path, index=False)
        outputs.to_csv(out_path, index=False)
        result = runner.invoke(cli_main, ["dea", str(in_path), str(out_path)])
        assert result.exit_code == 0
        assert "DEA Etkinlik" in result.output
        assert "Ortalama etkinlik" in result.output

    def test_dea_bcc_option(self, runner, tmp_path):
        inputs = pd.DataFrame({"isletme": ["C1", "C2", "C3"], "g": [5, 8, 6]})
        outputs = pd.DataFrame({"isletme": ["C1", "C2", "C3"], "c": [60, 80, 90]})
        in_path = tmp_path / "g.csv"
        out_path = tmp_path / "c.csv"
        inputs.to_csv(in_path, index=False)
        outputs.to_csv(out_path, index=False)
        result = runner.invoke(cli_main, ["dea", str(in_path), str(out_path),
                                          "--model", "BCC"])
        assert result.exit_code == 0
        assert "BCC" in result.output

    @pytest.fixture
    def categorical_csv(self, tmp_path):
        df = pd.DataFrame({
            "grup": ["A"] * 20 + ["B"] * 10,
            "yanit": (["x"] * 20) + (["y"] * 10),
            "olcum": [1.0] * 30,
        })
        path = tmp_path / "kategorik.csv"
        df.to_csv(path, index=False)
        return str(path)

    def test_frequencies_command(self, runner, categorical_csv):
        result = runner.invoke(cli_main, ["frequencies", categorical_csv,
                                          "--kolonlar", "grup"])
        assert result.exit_code == 0
        assert "Frekans Tabloları" in result.output
        assert "Küm. %" in result.output

    def test_crosstabs_command(self, runner, categorical_csv):
        result = runner.invoke(cli_main, ["crosstabs", categorical_csv,
                                          "--satir", "grup", "--sutun", "yanit"])
        assert result.exit_code == 0
        assert "Ki-kare" in result.output
        assert "Bağımsızlık reddedildi" in result.output

    def test_onesample_command(self, runner, csv_file):
        result = runner.invoke(cli_main, ["onesample", csv_file,
                                          "--kolon", "verim", "--deger", "0"])
        assert result.exit_code == 0
        assert "Tek Örneklem T-Testi" in result.output
        assert "Anlamlı fark var" in result.output

    def test_paired_command(self, runner, tmp_path):
        rng = np.random.default_rng(15)
        once = rng.normal(10, 1, 25)
        df = pd.DataFrame({"once": once, "sonra": once + 1.5})
        path = tmp_path / "eslesmis2.csv"
        df.to_csv(path, index=False)
        result = runner.invoke(cli_main, ["paired", str(path),
                                          "--once", "once", "--sonra", "sonra"])
        assert result.exit_code == 0
        assert "Eşleştirilmiş" in result.output

    def test_onesample_missing_column(self, runner, csv_file):
        result = runner.invoke(cli_main, ["onesample", csv_file,
                                          "--kolon", "yok", "--deger", "0"])
        assert "bulunamadı" in result.output


class TestCliWave3:
    """Üçüncü dalga Premium Program komutları: multinom, ordlogit, discriminant,
    correspondence."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def model_csv(self, tmp_path):
        rng = np.random.default_rng(16)
        n = 180
        x = rng.normal(0, 1, n)
        kalite = np.where(x < -0.4, "dusuk",
                          np.where(x > 0.4, "yuksek", "orta"))
        kalite = np.where(rng.uniform(0, 1, n) < 0.1,
                          rng.choice(["dusuk", "orta", "yuksek"], n), kalite)
        df = pd.DataFrame({
            "kalite": kalite,
            "grup": np.where(x > 0, "A", "B"),
            "nem": x + rng.normal(0, 0.3, n),
            "protein": 0.6 * x + rng.normal(0, 0.5, n),
        })
        path = tmp_path / "model.csv"
        df.to_csv(path, index=False)
        return str(path)

    def test_multinom_command(self, runner, model_csv):
        result = runner.invoke(cli_main, ["multinom", model_csv,
                                          "--yanit", "kalite",
                                          "--degiskenler", "nem,protein"])
        assert result.exit_code == 0
        assert "Multinomial Lojistik" in result.output
        assert "Referans kategori" in result.output

    def test_ordlogit_command(self, runner, model_csv):
        result = runner.invoke(cli_main, ["ordlogit", model_csv,
                                          "--yanit", "kalite",
                                          "--degiskenler", "nem"])
        assert result.exit_code == 0
        assert "Ordinal Lojistik" in result.output
        assert "Eşikler" in result.output

    def test_discriminant_command(self, runner, model_csv):
        result = runner.invoke(cli_main, ["discriminant", model_csv,
                                          "--grup", "grup",
                                          "--degiskenler", "nem,protein"])
        assert result.exit_code == 0
        assert "Wilks lambda" in result.output
        assert "Genel doğruluk" in result.output

    def test_correspondence_command(self, runner, tmp_path):
        rows = ["c1"] * 40 + ["c2"] * 10 + ["c1"] * 10 + ["c2"] * 40
        cols = ["m1"] * 40 + ["m2"] * 10 + ["m1"] * 10 + ["m2"] * 40
        path = tmp_path / "capraz.csv"
        pd.DataFrame({"cesit": rows, "pazar": cols}).to_csv(path, index=False)
        result = runner.invoke(cli_main, ["correspondence", str(path),
                                          "--satir", "cesit", "--sutun", "pazar"])
        assert result.exit_code == 0
        assert "Uyuşum (Correspondence)" in result.output
        assert "Satır koordinatları" in result.output

    def test_multinom_missing_column(self, runner, model_csv):
        result = runner.invoke(cli_main, ["multinom", model_csv,
                                          "--yanit", "kalite",
                                          "--degiskenler", "yok"])
        assert "Hata" in result.output


class TestGlmCli:
    """GLM CLI komutu testleri: tek değişkenli ve tekrarlı ölçüm."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def glm_csv(self, tmp_path):
        path = tmp_path / "glm.csv"
        pd.DataFrame({"grup": ["A"] * 8 + ["B"] * 8,
                      "verim": [5, 6, 5, 6, 5, 6, 5, 6,
                                3, 4, 3, 4, 3, 4, 3, 4]}).to_csv(path, index=False)
        return str(path)

    @pytest.fixture
    def rm_csv(self, tmp_path):
        path = tmp_path / "rm.csv"
        pd.DataFrame({"denek": list(range(6)),
                      "t1": [1, 2, 3, 4, 5, 6], "t2": [2, 3, 4, 5, 6, 7],
                      "t3": [4, 5, 6, 7, 8, 9]}).to_csv(path, index=False)
        return str(path)

    def test_cli_glm_tek_faktor(self, runner, glm_csv):
        result = runner.invoke(cli_main, ["glm", glm_csv, "--yanit", "verim",
                                          "--faktorler", "grup", "--posthoc", "yok"])
        assert result.exit_code == 0
        assert "GLM" in result.output

    def test_cli_glm_tekrarli_olcum(self, runner, rm_csv):
        result = runner.invoke(cli_main, ["glm", rm_csv, "--yanit", "yok",
                                          "--faktorler", "denek",
                                          "--within", "t1,t2,t3",
                                          "--denek", "denek"])
        assert result.exit_code == 0
        assert "Mauchly" in result.output

    def test_cli_glm_hatali_sutun(self, runner, glm_csv):
        result = runner.invoke(cli_main, ["glm", glm_csv, "--yanit", "yok",
                                          "--faktorler", "grup"])
        assert result.exit_code != 0


class TestGeeCli:
    """GEE CLI komutu testleri: kümelenmiş veri için marjinal model."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def gee_csv(self, tmp_path):
        rng = np.random.default_rng(1)
        path = tmp_path / "gee.csv"
        pd.DataFrame({"grup": np.repeat(np.arange(15), 6),
                      "x": rng.normal(0, 1, 90),
                      "y": rng.normal(0, 1, 90)}).to_csv(path, index=False)
        return str(path)

    def test_cli_gee(self, runner, gee_csv):
        result = runner.invoke(cli_main, ["gee", gee_csv, "--yanit", "y",
                                          "--degiskenler", "x",
                                          "--grup", "grup",
                                          "--yapi", "exchangeable"])
        assert result.exit_code == 0
        assert "GEE" in result.output

    def test_cli_gee_hatali_aile(self, runner, gee_csv):
        result = runner.invoke(cli_main, ["gee", gee_csv, "--yanit", "y",
                                          "--degiskenler", "x",
                                          "--grup", "grup",
                                          "--aile", "negbin"])
        assert result.exit_code != 0


class TestGlmmCli:
    """GLMM CLI komutu testleri: gaussian REML + binomial/poisson PQL."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def _glmm_csv(self, tmp_path):
        rng = np.random.default_rng(2)
        rows = []
        for g in range(15):
            b = rng.normal(0, 1.0)
            for _ in range(8):
                x = rng.normal(0, 1)
                rows.append({"grup": g, "x": x,
                             "y": 2 + x + b + rng.normal(0, 0.5)})
        csv = tmp_path / "glmm.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)
        return str(csv)

    def test_cli_glmm(self, runner, _glmm_csv):
        result = runner.invoke(cli_main, ["glmm", _glmm_csv, "--yanit", "y",
                                          "--sabitler", "x", "--grup", "grup"])
        assert result.exit_code == 0
        assert "GLMM" in result.output

    def test_cli_glmm_binomial_hatali_yanit(self, runner, _glmm_csv):
        result = runner.invoke(cli_main, ["glmm", _glmm_csv, "--yanit", "y",
                                          "--sabitler", "x", "--grup", "grup",
                                          "--aile", "binomial"])
        assert result.exit_code != 0
