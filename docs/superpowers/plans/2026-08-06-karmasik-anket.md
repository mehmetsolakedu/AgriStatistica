# Plan: Karmaşık Anket (Complex Samples) — Taylor Doğrusallaştırması

**Spec:** `docs/superpowers/specs/2026-08-06-kalan-bosluklar-design.md` (§7, §8, §9, §10)

**Goal:** Tabakalı/çok aşamalı anket tasarımları için Taylor
doğrusallaştırmasına dayalı ortalama/toplam/oran tahminleri ve anket
lojistik regresyonunu yeni `agrista/survey` modülünde, CLI komutları,
yeni menü kategorisi [20] ve teslim güncellemeleriyle tamamlamak.

**Architecture:** Modül `agrista/survey/__init__.py`: `survey_design`
tasarım sözlüğü üretir; `_taylor_variance` lineer değişkenlerin PSU
toplamları üzerinden tabakalı varyansı hesaplar; `svy_mean`/`svy_total`/
`svy_ratio` uygun lineer değişkeni kurup bu çekirdeği kullanır;
`survey_logistic` ağırlıklı GLM + PSU-kümelenmiş sandwich kovaryans
(`cov_type="cluster"`) ile birinci derece Taylor denkliğini sağlar.
CLI: `agrista svymean/svyratio/svylogit`. Menüye YENİ [20] kategori eklenir
(19 → 20). Bu plan aynı zamanda teslim kapanışını taşır (README, sürüm
0.2.0, "Kısmi kalanlar" notunun kaldırılması).

**Tech Stack:** Python 3, numpy, pandas, scipy.stats (norm), statsmodels
GLM (var_weights, cluster kovaryans), click. Yeni bağımlılık yok.

**Global Constraints (spec'ten aynen):**
1. Yeni çekirdek bağımlılık YOK.
2. Her yerde yalnızca "Premium Program" adı; "SPSS" yasak.
3. Analiz fonksiyonları `dict` döner; sayısal alanlar `float`/`int` cast'li;
   Türkçe docstring; `_check_columns` deseniyle doğrulama.
4. CLI: click komutu `agrista/cli/__init__.py` içinde; Türkçe seçenekler;
   `_print_*_result` yazdırıcıları ayrı fonksiyon.
5. Menü handler'ları `_prompt_or_eof` kullanır; düz `click.prompt` yok.
6. TDD zorunlu; CLI testleri `CliRunner` ile; menü smoke testleri
   `tests/test_menu_flows.py` deseninde.
7. Alt sistem bitince `docs/02` log'una kayıt.
8. `pytest` yeşil + `flake8` temiz olmadan görev bitmez.

## Dosya Yapısı

| Dosya | Sorumluluk |
|---|---|
| `agrista/survey/__init__.py` (yeni) | `survey_design`, `_taylor_variance`, `svy_mean`, `svy_total`, `svy_ratio`, `survey_logistic` |
| `agrista/__init__.py` | `from agrista import survey` satırı |
| `agrista/cli/__init__.py` | `svymean`, `svyratio`, `svylogit` komutları, `_print_svy_result`, `_menu_svy*` handler'ları, menü [20] |
| `tests/test_premium_survey.py` | Survey birim testleri |
| `tests/test_viz_cli.py` | CliRunner testleri |
| `tests/test_menu_flows.py` | Kategori 20 smoke testleri |
| `README.md`, `pyproject.toml` | Teslim güncellemeleri |

---

## Task 1: `survey_design` + `_taylor_variance` çekirdeği (TDD)

**Files:** Test: `tests/test_premium_survey.py` (Create) · Create: `agrista/survey/__init__.py` · Modify: `agrista/__init__.py`
**Interfaces:** Produces tasarım sözlüğü {data, weight_col, id_col, strata_col, fpc_col, n_psu, n_strata}.

- [ ] **RED** — `tests/test_premium_survey.py` oluştur:

```python
"""Premium Program Karmaşık Örneklem (Taylor linearizasyonu) testleri."""
import numpy as np
import pandas as pd
import pytest

from agrista.survey import (survey_design, svy_mean, svy_total, svy_ratio,
                            survey_logistic)


def _hand_design():
    """Elle hesaplanabilir küçük tasarım: 2 tabaka × 2 PSU, ağırlık 1."""
    return pd.DataFrame({
        "tabaka": ["A", "A", "B", "B"],
        "psu": ["a1", "a2", "b1", "b2"],
        "y": [2.0, 4.0, 6.0, 8.0],
        "x": [1.0, 1.0, 2.0, 2.0],
    })


class TestSurveyDesign:
    def test_tasarim_ozeti(self):
        d = survey_design(_hand_design(), id_col="psu", strata_col="tabaka")
        assert d["n_psu"] == 4
        assert d["n_strata"] == 2

    def test_tek_psulu_tabaka_hatasi(self):
        df = _hand_design().iloc[[0, 2]]  # her tabakada tek PSU
        d = survey_design(df, id_col="psu", strata_col="tabaka")
        with pytest.raises(ValueError, match="Tek PSU"):
            svy_mean(d, "y")

    def test_eksik_sutun_hatasi(self):
        with pytest.raises(ValueError):
            survey_design(_hand_design(), id_col="yok")


class TestSvyMean:
    def test_elle_hesaplanan_deger(self):
        d = survey_design(_hand_design(), id_col="psu", strata_col="tabaka")
        res = svy_mean(d, "y")
        assert res["estimate"] == pytest.approx(5.0)
        # lin = w*(y-5)/4 -> [-0.75,-0.25,0.25,0.75]; var = 0.25+0.25 = 0.5
        assert res["std_err"] == pytest.approx(np.sqrt(0.5), rel=1e-9)
        assert res["ci_lower"] < 5.0 < res["ci_upper"]

    def test_deff_basit_sansimana_bir(self):
        rng = np.random.default_rng(12)
        df = pd.DataFrame({"y": rng.normal(10, 2, 40)})  # PSU yok -> SRS
        d = survey_design(df)
        res = svy_mean(d, "y")
        assert res["design_effect"] == pytest.approx(1.0, rel=1e-9)


class TestSvyTotalAndRatio:
    def test_toplam(self):
        d = survey_design(_hand_design(), id_col="psu", strata_col="tabaka")
        res = svy_total(d, "y")
        assert res["estimate"] == pytest.approx(20.0)
        assert res["std_err"] > 0
        assert res["design_effect"] is None

    def test_oran(self):
        d = survey_design(_hand_design(), id_col="psu", strata_col="tabaka")
        res = svy_ratio(d, numerator="y", denominator="x")
        assert res["estimate"] == pytest.approx(20.0 / 6.0)
        assert res["std_err"] >= 0

    def test_agirlikli_ortalama(self):
        df = _hand_design().assign(w=[2.0, 1.0, 1.0, 1.0])
        d = survey_design(df, weight_col="w", id_col="psu",
                          strata_col="tabaka")
        res = svy_mean(d, "y")
        beklenen = (2 * 2 + 1 * 4 + 1 * 6 + 1 * 8) / 5.0
        assert res["estimate"] == pytest.approx(beklenen)


class TestSurveyLogistic:
    def test_katsayilar_ve_cluster_se(self):
        rng = np.random.default_rng(8)
        psu = np.repeat(np.arange(30), 10)
        x = rng.normal(0, 1, 300)
        p = 1 / (1 + np.exp(-1.2 * x))
        y = (rng.uniform(size=300) < p).astype(int)
        df = pd.DataFrame({"psu": psu, "x": x, "y": y})
        d = survey_design(df, id_col="psu")
        res = survey_logistic(d, response="y", predictors=["x"])
        assert res["coefficients"]["x"]["coefficient"] == pytest.approx(1.2,
                                                                        abs=0.4)
        assert res["coefficients"]["x"]["std_err"] > 0
        assert res["n_psu"] == 30

    def test_psu_zorunlu(self):
        df = _hand_design().assign(y=[0, 1, 1, 0])
        d = survey_design(df)
        with pytest.raises(ValueError):
            survey_logistic(d, response="y", predictors=["x"])

    def test_ikili_olmayan_yanit(self):
        d = survey_design(_hand_design(), id_col="psu")
        with pytest.raises(ValueError):
            survey_logistic(d, response="y", predictors=["x"])
```

- [ ] Çalıştır: `python -m pytest tests/test_premium_survey.py -x -q` → `ModuleNotFoundError`.
- [ ] **GREEN** — `agrista/survey/__init__.py` oluştur:

```python
"""
Agrista Survey Module — Karmaşık Örneklem Analizleri
Tabakalı/çok aşamalı tasarımlar için Taylor doğrusallaştırması:
ortalama, toplam, oran tahminleri ve anket lojistik regresyonu.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from typing import Optional

_Z975 = float(stats.norm.ppf(0.975))


def survey_design(data: pd.DataFrame, weight_col: Optional[str] = None,
                  id_col: Optional[str] = None,
                  strata_col: Optional[str] = None,
                  fpc_col: Optional[str] = None) -> dict:
    """Anket tasarım tanımı (PSU, tabaka, ağırlık, FPC)."""
    for c in (weight_col, id_col, strata_col, fpc_col):
        if c is not None and c not in data.columns:
            raise ValueError(f"Eksik sütun: {c}")
    return {
        "data": data,
        "weight_col": weight_col,
        "id_col": id_col,
        "strata_col": strata_col,
        "fpc_col": fpc_col,
        "n_psu": int(data[id_col].nunique()) if id_col else int(len(data)),
        "n_strata": int(data[strata_col].nunique()) if strata_col else 1,
    }


def _taylor_variance(design: dict, lin: pd.Series) -> float:
    """Lineer değişkenin PSU toplamları üzerinden tabakalı Taylor varyansı.

    Tabaka h için: var_h = n_h/(n_h-1) * Σ_i (t_hi - t̄_h)²;
    FPC verilirse (1 - n_h/N_h) ile çarpılır.
    """
    data = design["data"]
    strata = data[design["strata_col"]] if design["strata_col"] \
        else pd.Series("_hepsi", index=data.index)
    psu = data[design["id_col"]] if design["id_col"] \
        else pd.Series(data.index, index=data.index)
    frame = pd.DataFrame({"strata": strata, "psu": psu, "lin": lin})
    psu_totals = frame.groupby(["strata", "psu"])["lin"].sum()

    fpc_N = None
    if design["fpc_col"]:
        fpc_N = frame.assign(N=data[design["fpc_col"]]).groupby(
            ["strata", "psu"])["N"].first()

    var = 0.0
    for h, sub in psu_totals.groupby(level="strata"):
        n_h = len(sub)
        if n_h < 2:
            raise ValueError("Tek PSU'lu tabakada Taylor varyansı hesaplanamaz")
        var_h = n_h / (n_h - 1) * float(((sub - sub.mean()) ** 2).sum())
        if fpc_N is not None:
            N_h = float(fpc_N.loc[h].iloc[0])
            var_h *= max(0.0, 1.0 - n_h / N_h)
        var += var_h
    return var


def _align(lin: pd.Series, design: dict) -> pd.Series:
    """Alt küme lineer değişkenini tasarım verisinin indeksine taşır."""
    return lin.reindex(design["data"].index, fill_value=0.0)


def _weights(work: pd.DataFrame, design: dict) -> pd.Series:
    wc = design["weight_col"]
    return work[wc] if wc else pd.Series(1.0, index=work.index)


def _report(estimate: float, var: float, design: dict,
            deff: Optional[float]) -> dict:
    se = float(np.sqrt(var))
    return {
        "estimate": float(estimate),
        "std_err": se,
        "ci_lower": float(estimate - _Z975 * se),
        "ci_upper": float(estimate + _Z975 * se),
        "design_effect": deff,
        "n_obs": int(len(design["data"].dropna())),
        "n_psu": design["n_psu"],
        "n_strata": design["n_strata"],
    }


def svy_mean(design: dict, var: str) -> dict:
    """Ağırlıklı anket ortalaması + Taylor SE, CI, tasarım etkisi (DEFF)."""
    data = design["data"]
    cols = [var] + ([design["weight_col"]] if design["weight_col"] else [])
    for c in cols:
        if c not in data.columns:
            raise ValueError(f"Eksik sütun: {c}")
    work = data[cols].dropna()
    w, y = _weights(work, design), work[var]
    w_sum = float(w.sum())
    est = float((w * y).sum() / w_sum)
    lin = _align(w * (y - est) / w_sum, design)
    var_est = _taylor_variance(design, lin)
    var_srs = float((w * (y - est) ** 2).sum()) / (w_sum * max(len(y) - 1, 1))
    deff = float(var_est / var_srs) if var_srs > 0 else None
    rep = _report(est, var_est, design, deff)
    rep["n_obs"] = int(len(work))
    return rep


def svy_total(design: dict, var: str) -> dict:
    """Ağırlıklı anket toplamı + Taylor SE ve CI."""
    data = design["data"]
    if var not in data.columns:
        raise ValueError(f"Eksik sütun: {var}")
    work = data[[var] + ([design["weight_col"]]
                         if design["weight_col"] else [])].dropna()
    w, y = _weights(work, design), work[var]
    est = float((w * y).sum())
    lin = _align(w * y, design)
    rep = _report(est, _taylor_variance(design, lin), design, None)
    rep["n_obs"] = int(len(work))
    return rep


def svy_ratio(design: dict, numerator: str, denominator: str) -> dict:
    """Anket oranı R = Σwy / Σwx; lineer değişken u = w(y - R x)/Σwx."""
    data = design["data"]
    for c in (numerator, denominator):
        if c not in data.columns:
            raise ValueError(f"Eksik sütun: {c}")
    work = data[[numerator, denominator]
                + ([design["weight_col"]]
                   if design["weight_col"] else [])].dropna()
    w, y, x = _weights(work, design), work[numerator], work[denominator]
    sum_x = float((w * x).sum())
    if sum_x == 0:
        raise ValueError("Payda toplamı sıfır; oran tanımsız")
    R = float((w * y).sum() / sum_x)
    lin = _align(w * (y - R * x) / sum_x, design)
    rep = _report(R, _taylor_variance(design, lin), design, None)
    rep["n_obs"] = int(len(work))
    return rep


def survey_logistic(design: dict, response: str, predictors: list) -> dict:
    """Anket lojistik regresyonu: ağırlıklı GLM + PSU-kümelenmiş sandwich.

    Birinci derece Taylor doğrusallaştırmasına denktir (skor toplamlarının
    PSU düzeyinde kümelenmiş kovaryansı).
    """
    data = design["data"]
    if design["id_col"] is None:
        raise ValueError("survey_logistic için id_col (PSU) gerekli")
    cols = [response] + list(predictors) \
        + ([design["weight_col"]] if design["weight_col"] else []) \
        + [design["id_col"]]
    for c in cols:
        if c not in data.columns:
            raise ValueError(f"Eksik sütun: {c}")
    work = data[cols].dropna()
    if not work[response].isin([0, 1]).all():
        raise ValueError("Anket lojistik için yanıt 0/1 olmalı")
    w = _weights(work, design)
    formula = f"{response} ~ {' + '.join(predictors)}"
    fitted = smf.glm(formula, work, family=sm.families.Binomial(),
                     var_weights=w).fit(cov_type="cluster",
                                        cov_kwds={"groups": work[design["id_col"]]})
    coefficients = {}
    for ad in fitted.params.index:
        coefficients[ad] = {
            "coefficient": float(fitted.params[ad]),
            "std_err": float(fitted.bse[ad]),
            "z_value": float(fitted.tvalues[ad]),
            "p_value": float(fitted.pvalues[ad]),
        }
    return {
        "model": "Survey Logistic",
        "coefficients": coefficients,
        "n_obs": int(len(work)),
        "n_psu": int(work[design["id_col"]].nunique()),
        "n_strata": design["n_strata"],
    }
```

- [ ] `agrista/__init__.py` modül import bloğuna ekle:

```python
from agrista import survey       # Premium Program Complex Samples: Taylor
```

- [ ] Çalıştır: `python -m pytest tests/test_premium_survey.py -q` → yeşil.
- [ ] `flake8 agrista/survey/__init__.py` temiz.
- [ ] Commit: `feat(survey): Taylor linearizasyonu — mean/total/ratio/logistic`

## Task 2: CLI komutları `svymean` / `svyratio` / `svylogit` (TDD)

**Files:** Test: `tests/test_viz_cli.py` (Modify) · Modify: `agrista/cli/__init__.py`
**Interfaces:** Consumes `agrista.survey` fonksiyonları · Produces stdout raporu.

- [ ] **RED** — `tests/test_viz_cli.py` sonuna ekle:

```python
def _svy_csv(tmp_path):
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "psu": np.repeat(np.arange(12), 5),
        "tabaka": np.repeat([0] * 6 + [1] * 6, 5),
        "w": rng.uniform(0.5, 2, 60),
        "y": rng.normal(50, 5, 60),
        "x": rng.uniform(1, 3, 60),
        "b": rng.integers(0, 2, 60),
    })
    csv = tmp_path / "svy.csv"
    df.to_csv(csv, index=False)
    return csv

def test_cli_svymean(tmp_path):
    csv = _svy_csv(tmp_path)
    result = runner.invoke(main, ["svymean", str(csv), "--degisken", "y",
                                  "--agirlik", "w", "--psu", "psu",
                                  "--tabaka", "tabaka"])
    assert result.exit_code == 0
    assert "Anket ortalaması" in result.output

def test_cli_svyratio(tmp_path):
    csv = _svy_csv(tmp_path)
    result = runner.invoke(main, ["svyratio", str(csv), "--pay", "y",
                                  "--payda", "x", "--psu", "psu"])
    assert result.exit_code == 0
    assert "oran" in result.output.lower()

def test_cli_svylogit(tmp_path):
    csv = _svy_csv(tmp_path)
    result = runner.invoke(main, ["svylogit", str(csv), "--yanit", "b",
                                  "--degiskenler", "y", "--psu", "psu",
                                  "--agirlik", "w"])
    assert result.exit_code == 0
    assert "Survey Logistic" in result.output

def test_cli_svymean_eksik_psu_hatasi(tmp_path):
    csv = _svy_csv(tmp_path)
    result = runner.invoke(main, ["svymean", str(csv), "--degisken", "y",
                                  "--psu", "yok"])
    assert result.exit_code != 0
```

- [ ] Çalıştır: `python -m pytest tests/test_viz_cli.py -k svy -q` → `No such command 'svymean'`.
- [ ] **GREEN** — `agrista/cli/__init__.py` içine ekle:

```python
def _print_svy_result(result: dict, baslik: str):
    click.echo(f"\n🧮 {baslik}")
    click.echo(f"   Tahmin = {result['estimate']:.4f}  "
               f"SE = {result['std_err']:.4f}")
    click.echo(f"   %95 CI [{result['ci_lower']:.4f}, "
               f"{result['ci_upper']:.4f}]")
    if result.get("design_effect") is not None:
        click.echo(f"   Tasarım etkisi (DEFF) = "
                   f"{result['design_effect']:.3f}")
    click.echo(f"   n = {result['n_obs']}, PSU = {result['n_psu']}, "
               f"tabaka = {result['n_strata']}")


def _print_svylogit_result(result: dict):
    click.echo(f"\n🧮 {result['model']}")
    click.echo("   Değişken        Katsayı     SE       z       p")
    for ad, c in result["coefficients"].items():
        click.echo(f"   {ad[:14]:<14} {c['coefficient']:9.4f} "
                   f"{c['std_err']:8.4f} {c['z_value']:7.3f} "
                   f"{c['p_value']:8.4f}")
    click.echo(f"   n = {result['n_obs']}, PSU = {result['n_psu']}")


def _build_svy_design(df, agirlik, psu, tabaka, fpc):
    from agrista.survey import survey_design
    return survey_design(df, weight_col=agirlik, id_col=psu,
                         strata_col=tabaka, fpc_col=fpc)


@main.command("svymean")
@click.argument("filepath")
@click.option("--degisken", required=True, help="Tahmin edilecek değişken")
@click.option("--agirlik", default=None, help="Ağırlık sütunu")
@click.option("--psu", default=None, help="PSU (birincil örnekleme birimi) sütunu")
@click.option("--tabaka", default=None, help="Tabaka sütunu")
@click.option("--fpc", default=None, help="FPC (tabaka popülasyon PSU sayısı) sütunu")
def svymean(filepath: str, degisken: str, agirlik: str, psu: str,
            tabaka: str, fpc: str):
    """Anket ortalaması (Taylor linearizasyonu)."""
    from agrista.survey import svy_mean
    df = _load_file(filepath)
    try:
        design = _build_svy_design(df, agirlik, psu, tabaka, fpc)
        result = svy_mean(design, degisken)
    except ValueError as e:
        raise click.ClickException(str(e))
    _print_svy_result(result, "Anket ortalaması")


@main.command("svyratio")
@click.argument("filepath")
@click.option("--pay", required=True, help="Pay değişkeni")
@click.option("--payda", required=True, help="Payda değişkeni")
@click.option("--agirlik", default=None, help="Ağırlık sütunu")
@click.option("--psu", default=None, help="PSU sütunu")
@click.option("--tabaka", default=None, help="Tabaka sütunu")
@click.option("--fpc", default=None, help="FPC sütunu")
def svyratio(filepath: str, pay: str, payda: str, agirlik: str, psu: str,
             tabaka: str, fpc: str):
    """Anket oranı (Taylor linearizasyonu)."""
    from agrista.survey import svy_ratio
    df = _load_file(filepath)
    try:
        design = _build_svy_design(df, agirlik, psu, tabaka, fpc)
        result = svy_ratio(design, numerator=pay, denominator=payda)
    except ValueError as e:
        raise click.ClickException(str(e))
    _print_svy_result(result, "Anket oranı")


@main.command("svylogit")
@click.argument("filepath")
@click.option("--yanit", required=True, help="0/1 yanıt değişkeni")
@click.option("--degiskenler", required=True, help="Virgülle ayrılmış açıklayıcılar")
@click.option("--agirlik", default=None, help="Ağırlık sütunu")
@click.option("--psu", required=True, help="PSU sütunu (zorunlu)")
@click.option("--tabaka", default=None, help="Tabaka sütunu")
@click.option("--fpc", default=None, help="FPC sütunu")
def svylogit(filepath: str, yanit: str, degiskenler: str, agirlik: str,
             psu: str, tabaka: str, fpc: str):
    """Anket lojistik regresyonu (kümelenmiş robust SE)."""
    from agrista.survey import survey_logistic
    df = _load_file(filepath)
    try:
        design = _build_svy_design(df, agirlik, psu, tabaka, fpc)
        result = survey_logistic(design, response=yanit,
                                 predictors=[c.strip()
                                             for c in degiskenler.split(",")])
    except ValueError as e:
        raise click.ClickException(str(e))
    _print_svylogit_result(result)
```

- [ ] Çalıştır: `python -m pytest tests/test_viz_cli.py -k svy -q` → yeşil.
- [ ] Commit: `feat(cli): agrista svymean/svyratio/svylogit`

## Task 3: Menü kategorisi [20] + smoke testler + log kaydı

**Files:** Modify: `agrista/cli/__init__.py` (`_build_menu_structure` sonuna
yeni kategori, üç handler), `tests/test_menu_flows.py`,
`docs/02_PREMIUM_PROGRAM_MENU_YAPISI_LOG.md`

- [ ] **RED** — `tests/test_menu_flows.py` içine mevcut desenle üç smoke
  testi ekle (kategori 20 → her alt öğe; stdin girdileri satır satır).
- [ ] **GREEN** — `_build_menu_structure` listesinin SONUNA yeni kategori ekle:

```python
("🧮 Karmaşık Örneklem (Complex Samples)", [
    ("Anket ortalaması/toplamı (Taylor)", _menu_svymean),
    ("Anket oranı (Taylor)", _menu_svyratio),
    ("Anket lojistik regresyonu", _menu_svylogit),
]),
```

  Handler'lar (`_prompt_or_eof` korumalı):

```python
def _menu_svymean():
    path = _ask_file("Veri dosyası yolu")
    df = _load_file(path)
    degisken = _ask_column(df, "Tahmin edilecek değişken")
    agirlik = _prompt_or_eof("Ağırlık sütunu (yoksa boş bırak)", default="")
    psu = _prompt_or_eof("PSU sütunu (yoksa boş bırak)", default="")
    tabaka = _prompt_or_eof("Tabaka sütunu (yoksa boş bırak)", default="")
    design = _build_svy_design(df, agirlik or None, psu or None,
                               tabaka or None, None)
    result = svy_mean(design, degisken)
    _print_svy_result(result, "Anket ortalaması")


def _menu_svyratio():
    path = _ask_file("Veri dosyası yolu")
    df = _load_file(path)
    pay = _ask_column(df, "Pay değişkeni")
    payda = _ask_column(df, "Payda değişkeni")
    psu = _prompt_or_eof("PSU sütunu (yoksa boş bırak)", default="")
    design = _build_svy_design(df, None, psu or None, None, None)
    result = svy_ratio(design, numerator=pay, denominator=payda)
    _print_svy_result(result, "Anket oranı")


def _menu_svylogit():
    path = _ask_file("Veri dosyası yolu")
    df = _load_file(path)
    yanit = _ask_column(df, "0/1 yanıt değişkeni")
    degiskenler = _prompt_or_eof("Açıklayıcılar (virgülle)")
    psu = _ask_column(df, "PSU sütunu")
    if not degiskenler:
        raise click.exceptions.Abort()
    design = _build_svy_design(df, None, psu, None, None)
    result = survey_logistic(design, response=yanit,
                             predictors=[c.strip()
                                         for c in degiskenler.split(",")])
    _print_svylogit_result(result)
```

  (Dosya başına `from agrista.survey import svy_mean, svy_ratio,
  survey_logistic` eklenir.)
- [ ] Çalıştır: `python -m pytest tests/test_menu_flows.py -q` → yeşil.
- [ ] `docs/02` log'una "Güncelleme 5" survey alt başlığını ekle;
  "Kısmi kalanlar" satırından "karmaşık anket Taylor doğrusallaştırması
  (2.18)" çıkarılır — üç kalem de kapandığı için satırın tamamı silinir ve
  yerine şu yazılır: **"Kısmi kalan: yok. Premium Program Base denkliği
  tamamlandı (Güncelleme 5)."**
- [ ] Commit: `feat(menu): [20] Karmaşık Örneklem menüsü + docs/02 Güncelleme 5 (survey)`

## Task 4: Teslim kapanışı — README, sürüm, son doğrulama

**Files:** Modify: `README.md`, `pyproject.toml`, `agrista/cli/__init__.py`
(version_option), `docs/02_PREMIUM_PROGRAM_MENU_YAPISI_LOG.md`

- [ ] `README.md` güncellemeleri:
  - Ana menü paragrafında "19 kategori" → "20 kategori"; kategori listesine
    **[20] Karmaşık Örneklem** eklenir.
  - Tek atımlık komut örnek bloğuna üç satır eklenir:
    `agrista svymean anket.csv --degisken gelir --agirlik w --psu psu`
    vb. + `agrista glm`, `agrista gee`, `agrista glmm` örnekleri
    (diğer planlarda eklenmediyse burada).
  - Özellik listesinde "İstatistiksel Analiz" satırına GLM/GEE/GLMM,
    yeni bir madde olarak karmaşık anket tahminleri eklenir.
- [ ] Sürüm artışı: `pyproject.toml` içinde `version = "0.1.0"` →
  `version = "0.2.0"`; `agrista/cli/__init__.py` içinde
  `@click.version_option(version="0.1.0", ...)` → `"0.2.0"`.
- [ ] Son doğrulama (taze kanıt, tam paket):
  `python -m pytest tests/ -q && flake8 agrista tests`
- [ ] Commit: `chore: v0.2.0 — README, sürüm ve docs/02 teslim güncellemeleri`
