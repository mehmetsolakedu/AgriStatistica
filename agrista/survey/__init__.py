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
