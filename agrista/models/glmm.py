"""
Agrista GLMM Module — Genelleştirilmiş Doğrusal Karışık Modeller
Gaussian: REML (statsmodels MixedLM) · Binomial/Poisson: PQL
(Breslow & Clayton 1993 penalized quasi-likelihood).

Not: statsmodels 0.14.6 MixedLM `freq_weights` desteklemez
("argument freq_weights not permitted for MixedLM initialization");
bu nedenle PQL'nin ağırlıklı LMM adımı `_weighted_lmm` içinde
profil REML (sabit çalışma ölçeği = 1) olarak doğrudan çözümlenir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrix
from scipy import stats
from scipy.linalg import solve_triangular
from scipy.optimize import minimize_scalar
from typing import Optional


def _weighted_lmm(z: np.ndarray, X: np.ndarray, groups: np.ndarray,
                  R: np.ndarray) -> dict:
    """Ağırlıklı rastgele etkili LMM (çalışma ölçeği 1'e sabitlenmiş REML).

    z ~ N(X β, ρ·(Z_g Z_g') + W^{-1}); W = diag(w) biliniyor, çalışma
    ölçeği PQL gereği 1'de sabitlenir (Breslow & Clayton 1993). β,
    standart hatalar, Wald z/p, grup BLUP'ları ve RE varyansı ρ döner.
    """
    n, p = X.shape
    uniq = np.unique(groups)
    idx = [np.where(groups == g)[0] for g in uniq]
    w = np.maximum(z[:, 1] ** 2, 1e-12)
    sw = np.sqrt(w)
    zr = z[:, 0]

    def neg_reml(log_rho: float) -> float:
        """Ölçek 1'e sabitlenmiş profil REML kriteri (yalnızca ρ serbest)."""
        rho = float(np.exp(log_rho))
        ld_v, rss, xtx = 0.0, 0.0, np.zeros((p, p))
        xty = np.zeros(p)
        for ix in idx:
            zg = R[ix] * sw[ix][:, None]
            v_g = rho * (zg @ zg.T) + np.eye(len(ix))
            chol = np.linalg.cholesky(v_g)
            ld_v += 2.0 * float(np.sum(np.log(np.diag(chol))))
            zt = solve_triangular(
                chol, zr[ix] * sw[ix], lower=True)
            xt = solve_triangular(
                chol, X[ix] * sw[ix][:, None], lower=True)
            rss += float(zt @ zt)
            xtx += xt.T @ xt
            xty += xt.T @ zt
        beta = np.linalg.solve(xtx, xty)
        rss -= float(beta @ xty)
        return ld_v + rss + float(np.linalg.slogdet(xtx)[1])

    res = minimize_scalar(neg_reml, bounds=(-16.0, 16.0), method="bounded")
    if not res.success:
        grid = np.linspace(-16.0, 16.0, 65)
        res.fun = min(neg_reml(t) for t in grid)
        res.x = min(grid, key=neg_reml)
    rho = float(np.exp(res.x))

    # rho sabitken β ve bilgi matrisi
    xtx = np.zeros((p, p))
    xty = np.zeros(p)
    for ix in idx:
        zg = R[ix] * sw[ix][:, None]
        v_g = rho * (zg @ zg.T) + np.eye(len(ix))
        chol = np.linalg.cholesky(v_g)
        xt = solve_triangular(
            chol, X[ix] * sw[ix][:, None], lower=True)
        zt = solve_triangular(
            chol, zr[ix] * sw[ix], lower=True)
        xtx += xt.T @ xt
        xty += xt.T @ zt
    beta = np.linalg.solve(xtx, xty)
    se = np.sqrt(np.maximum(np.diag(np.linalg.inv(xtx)), 0.0))

    # BLUP'lar: u_g = rho Z_g' V_g^{-1} (z_g - X_g beta) (dönüştürülmüş uzay)
    blups = {}
    for g, ix in zip(uniq, idx):
        zg = R[ix] * sw[ix][:, None]
        v_g = rho * (zg @ zg.T) + np.eye(len(ix))
        resid = zr[ix] * sw[ix] - (X[ix] * sw[ix][:, None]) @ beta
        blups[g] = rho * zg.T @ np.linalg.solve(v_g, resid)

    return {"beta": beta, "se": se, "rho": rho, "blups": blups,
            "names": list(range(p))}


def _mixed_aic(fitted) -> float:
    """AIC değeri; statsmodels 0.14.x REML uydurmalarında NaN döndürdüğü
    için gerekirse -2(llf - df) formülüyle hesaplanır."""
    val = float(fitted.aic)
    if np.isfinite(val):
        return val
    df = fitted.params.size + 1
    return float(-2 * (fitted.llf - df))


def _pql_fit(work: pd.DataFrame, response: str, rhs: str, groups_col: str,
             fam, re_formula: Optional[str], max_iter: int,
             tol: float) -> dict:
    """PQL döngüsü: pseudo-yanıt + ağırlıklı LMM + BLUP güncellemesi."""
    y = work[response].to_numpy(dtype=float)

    def _clip_mu(mu_arr: np.ndarray) -> np.ndarray:
        """Sayısal kararlılık için μ sınırlaması (poisson'da üst sınır taşmayı önler)."""
        if fam.__class__.__name__ == "Binomial":
            return np.clip(mu_arr, 1e-6, 1 - 1e-6)
        return np.clip(mu_arr, 1e-6, 1e6)

    glm0 = smf.glm(f"{response} ~ {rhs}", work, family=fam).fit()
    mu = _clip_mu(glm0.fittedvalues.to_numpy())
    converged = False
    n_iter = 0
    prev_beta = None
    fit = None
    X = dmatrix(rhs, work, return_type="dataframe")
    X_mat = X.to_numpy(dtype=float)
    groups = work[groups_col].to_numpy()

    # Rastgele etki tasarım bloğu (kesim + isteğe bağlı eğim)
    if re_formula is not None:
        rs_name = re_formula.lstrip("~").strip()
        r_col = work[rs_name].to_numpy(dtype=float)
        R = np.column_stack([np.ones(len(work)), r_col])
    else:
        R = np.ones((len(work), 1))

    for n_iter in range(1, max_iter + 1):
        g_prime = fam.link.deriv(mu)
        V_mu = fam.variance(mu)
        # PQL çalışma yanıtı: z = g(μ) + g'(μ)(y - μ) (Breslow & Clayton 1993)
        z_p = fam.link(mu) + (y - mu) * g_prime
        w = 1.0 / np.maximum(V_mu * g_prime ** 2, 1e-10)
        zp = np.column_stack([z_p, np.sqrt(w)])
        fit = _weighted_lmm(zp, X_mat, groups, R)
        beta = fit["beta"]
        if prev_beta is not None and \
                np.max(np.abs(beta - prev_beta)) < tol:
            converged = True
            break
        prev_beta = beta.copy()

        b_g = np.array([float(np.atleast_1d(fit["blups"][g])[0])
                        for g in groups])
        eta = X_mat @ beta + b_g
        mu = _clip_mu(fam.link.inverse(eta))

    fixed_effects = {}
    for j, ad in enumerate(X.design_info.column_names):
        coef = float(beta[j])
        se = float(fit["se"][j])
        z_val = coef / se if se > 0 else float("nan")
        fixed_effects[ad] = {
            "coefficient": coef, "std_err": se, "z_value": z_val,
            "p_value": float(2 * (1 - stats.norm.cdf(abs(z_val)))),
        }
    return {
        "method": "PQL", "converged": bool(converged),
        "n_iterations": int(n_iter), "fixed_effects": fixed_effects,
        "random_effects_variance": {"random_intercept": float(fit["rho"])},
        "aic": None,
    }


def glmm(data: pd.DataFrame, response: str, fixed_effects: list,
         groups_col: str, family: str = "gaussian",
         random_slope: Optional[str] = None,
         max_iter: int = 25, tol: float = 1e-5) -> dict:
    """Genelleştirilmiş doğrusal karışık model (GLMM).

    gaussian → REML (MixedLM); binomial/poisson → PQL yinelemesi.
    Dönüş şeması agrista.animal.mixed_model ile uyumludur.
    """
    cols = [response] + list(fixed_effects) + [groups_col] + \
           ([random_slope] if random_slope else [])
    eksik = [c for c in cols if c not in data.columns]
    if eksik:
        raise ValueError(f"Eksik sütunlar: {eksik}")
    work = data[cols].dropna()
    n_groups = work[groups_col].nunique()
    if n_groups < 2:
        raise ValueError("GLMM için en az 2 grup gerekli")
    if len(work) / n_groups < 2:
        raise ValueError("Grup başına en az 2 gözlem gerekli")

    rhs = " + ".join(fixed_effects)
    re_formula = f"~{random_slope}" if random_slope else None

    if family == "gaussian":
        model = smf.mixedlm(f"{response} ~ {rhs}", work,
                            groups=work[groups_col], re_formula=re_formula)
        fitted = model.fit(reml=True)
        cov_re = np.atleast_2d(np.asarray(fitted.cov_re))
        return {
            "model": "GLMM", "family": "gaussian", "method": "REML",
            "converged": bool(fitted.converged), "n_iterations": 1,
            "fixed_effects": {
                ad: {"coefficient": float(fitted.params[ad]),
                     "std_err": float(fitted.bse[ad]),
                     "z_value": float(fitted.tvalues[ad]),
                     "p_value": float(fitted.pvalues[ad])}
                for ad in fitted.params.index
            },
            "random_effects_variance": {"random_intercept": float(cov_re[0, 0])},
            "aic": _mixed_aic(fitted),
            "n_obs": int(len(work)), "n_groups": int(n_groups),
        }
    if family == "binomial":
        if not work[response].isin([0, 1]).all():
            raise ValueError("Binomial aile için yanıt 0/1 olmalı")
        fam = sm.families.Binomial()
    elif family == "poisson":
        fam = sm.families.Poisson()
    else:
        raise ValueError(f"Bilinmeyen aile: {family}")

    res = _pql_fit(work, response, rhs, groups_col, fam, re_formula,
                   max_iter, tol)
    res.update({"model": "GLMM", "family": family,
                "n_obs": int(len(work)), "n_groups": int(n_groups)})
    return res
