"""Agrista GUI analiz kaydı — bildirimsel analiz tanımları.

Her AnalysisSpec bir menü öğesini bir analiz fonksiyonuna bağlar;
formlar parametre şemasından otomatik üretilir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional  # noqa: F401

import pandas as pd


@dataclass
class Param:
    """Analiz parametresi şeması.

    kind: "column" | "columns" | "numeric" | "choice"
    """
    name: str
    label: str
    kind: str
    required: bool = True
    default: object = None
    choices: tuple = ()


@dataclass
class AnalysisSpec:
    """Bir analiz öğesinin menü + çalışma tanımı."""
    key: str
    menu_category: str
    label: str
    run: Callable[[pd.DataFrame, dict], dict]
    params: list = field(default_factory=list)


def _kolonlar(p: dict, anahtar: str) -> list:
    ham = p.get(anahtar) or ""
    return [c.strip() for c in str(ham).split(",") if c.strip()]


def _sutun_dogrula(df, adlar: list) -> None:
    eksik = [c for c in adlar if c not in df.columns]
    if eksik:
        raise ValueError(f"Sütun bulunamadı: {', '.join(eksik)}")


def _betimsel(df, p):
    from agrista.analysis import descriptive_stats
    kol = _kolonlar(p, "kolonlar")
    if kol:
        _sutun_dogrula(df, kol)
        return descriptive_stats(df[kol])
    return descriptive_stats(df)


def _frekans(df, p):
    from agrista.analysis import frequencies
    kol = _kolonlar(p, "kolonlar")
    return frequencies(df, columns=kol or None)


def _capraz(df, p):
    from agrista.analysis import crosstabs
    return crosstabs(df, p["satir"], p["sutun"])


def _oran(df, p):
    from agrista.analysis import ratio_statistics
    return ratio_statistics(df, p["pay"], p["payda"])


def _normallik(df, p):
    from agrista.analysis import normality_test
    return normality_test(df[p["kolon"]])


def _tek_orneklem_t(df, p):
    from agrista.analysis import one_sample_t_test
    return one_sample_t_test(df[p["kolon"]], test_value=float(p["deger"]))


def _bagimsiz_t(df, p):
    from agrista.analysis import t_test
    duzeyler = sorted(df[p["grup"]].dropna().unique(), key=str)
    if len(duzeyler) < 2:
        raise ValueError("Grup sütunu en az 2 düzey içermeli")
    g1 = df.loc[df[p["grup"]] == duzeyler[0], p["yanit"]]
    g2 = df.loc[df[p["grup"]] == duzeyler[1], p["yanit"]]
    return t_test(g1, g2)


def _eslestirilmis_t(df, p):
    from agrista.analysis import paired_t_test
    return paired_t_test(df[p["once"]], df[p["sonra"]])


def _anova_tukey(df, p):
    from agrista.analysis import anova_one_way, posthoc_tukey
    gruplar = [g[p["yanit"]].dropna().values
               for _, g in df.groupby(p["grup"], observed=True)]
    if len(gruplar) < 2:
        raise ValueError("ANOVA için en az 2 grup gerekli")
    return {"anova": anova_one_way(*gruplar),
            "tukey": posthoc_tukey(df, p["yanit"], p["grup"])}


def _glm_univariate(df, p):
    from agrista.analysis import glm_univariate
    return glm_univariate(df, response=p["yanit"],
                          between_factors=_kolonlar(p, "faktorler"),
                          posthoc=None)


def _glm_tekrarli(df, p):
    from agrista.analysis import glm_repeated_measures
    return glm_repeated_measures(df,
                                 response_cols=_kolonlar(p, "kosullar"),
                                 subject_col=p["denek"])


def _korelasyon(df, p):
    from agrista.analysis import correlation_analysis
    return correlation_analysis(df, method=p.get("yontem", "pearson"))


def _gee(df, p):
    from agrista.analysis import gee_model
    return gee_model(df, response=p["yanit"],
                     covariates=_kolonlar(p, "degiskenler"),
                     group_col=p["grup"])


def _roc(df, p):
    from agrista.analysis import roc_curve
    return roc_curve(df[p["gercek"]], df[p["skor"]])


def _kaplan_meier(df, p):
    from agrista.survival import kaplan_meier
    return kaplan_meier(df[p["zaman"]], df[p["olay"]])


def _audpc(df, p):
    from agrista.protection import audpc
    return audpc(df[p["zaman"]], df[p["siddet"]])


KAT_B = "📊 Betimsel İstatistikler"
KAT_O = "⚖️ Ortalamaların Karşılaştırılması"

REGISTRY: list = [
    AnalysisSpec("betimsel", KAT_B, "Betimsel özet tablosu", _betimsel,
                 [Param("kolonlar", "Sütunlar (boş = tüm sayısal)",
                        "columns", required=False)]),
    AnalysisSpec("frekans", KAT_B, "Frekans tabloları", _frekans,
                 [Param("kolonlar", "Kategorik sütunlar (virgülle)",
                        "columns", required=False)]),
    AnalysisSpec("capraz", KAT_B, "Çapraz tablolar (ki-kare)", _capraz,
                 [Param("satir", "Satır değişkeni", "column"),
                  Param("sutun", "Sütun değişkeni", "column")]),
    AnalysisSpec("oran", KAT_B, "Oran istatistikleri", _oran,
                 [Param("pay", "Pay değişkeni", "column"),
                  Param("payda", "Payda değişkeni", "column")]),
    AnalysisSpec("normallik", KAT_B, "Normallik testi (Shapiro-Wilk)",
                 _normallik, [Param("kolon", "Sayısal sütun", "column")]),
    AnalysisSpec("tek_orneklem_t", KAT_O, "Tek örneklem t-testi",
                 _tek_orneklem_t,
                 [Param("kolon", "Sayısal sütun", "column"),
                  Param("deger", "Karşılaştırılacak değer", "numeric",
                        default=0.0)]),
    AnalysisSpec("bagimsiz_t", KAT_O, "Bağımsız iki örneklem t-testi",
                 _bagimsiz_t,
                 [Param("yanit", "Yanıt değişkeni", "column"),
                  Param("grup", "Grup sütunu", "column")]),
    AnalysisSpec("eslestirilmis_t", KAT_O,
                 "Eşleştirilmiş iki örneklem t-testi",
                 _eslestirilmis_t,
                 [Param("once", "Önce ölçümü", "column"),
                  Param("sonra", "Sonra ölçümü", "column")]),
    AnalysisSpec("anova_tukey", KAT_O, "Tek yönlü ANOVA + Tukey HSD",
                 _anova_tukey,
                 [Param("yanit", "Yanıt değişkeni", "column"),
                  Param("grup", "Grup sütunu", "column")]),
    AnalysisSpec("glm_univariate", KAT_O,
                 "Genel Doğrusal Model (Tek Değişkenli)", _glm_univariate,
                 [Param("yanit", "Bağımlı değişken", "column"),
                  Param("faktorler", "Faktörler (virgülle)", "columns")]),
    AnalysisSpec("glm_tekrarli", KAT_O, "Tekrarlı Ölçümler (GLM)",
                 _glm_tekrarli,
                 [Param("kosullar", "Ölçüm sütunları (virgülle)",
                        "columns"),
                  Param("denek", "Denek sütunu", "column")]),
    AnalysisSpec("korelasyon", "🔗 Korelasyon",
                 "İki değişkenli korelasyon (Pearson/Spearman)",
                 _korelasyon,
                 [Param("yontem", "Yöntem", "choice", default="pearson",
                        choices=("pearson", "spearman"))]),
    AnalysisSpec("gee", "📈 Regresyon",
                 "GEE (Genelleştirilmiş Tahmin Denklemleri)", _gee,
                 [Param("yanit", "Bağımlı değişken", "column"),
                  Param("degiskenler", "Açıklayıcılar (virgülle)",
                        "columns"),
                  Param("grup", "Küme/grup sütunu", "column")]),
    AnalysisSpec("roc", "🎯 ROC Eğrisi", "ROC / AUC (Youden eşik)", _roc,
                 [Param("gercek", "Gerçek sınıf (0/1)", "column"),
                  Param("skor", "Skor sütunu", "column")]),
    AnalysisSpec("kaplan_meier", "⏳ Yaşam Analizi (Survival)",
                 "Kaplan-Meier sağkalım tablosu", _kaplan_meier,
                 [Param("zaman", "Zaman sütunu", "column"),
                  Param("olay", "Olay sütunu (0/1)", "column")]),
    AnalysisSpec("audpc", "🌿 Uzman Branş Modülleri",
                 "Bitki Koruma — AUDPC (hastalık ilerlemesi)", _audpc,
                 [Param("zaman", "Zaman sütunu", "column"),
                  Param("siddet", "Hastalık şiddeti sütunu", "column")]),
]


def format_result(obj, indent: int = 0) -> str:
    """dict/list sonuçlarını hiyerarşik metne çevirir."""
    girinti = "  " * indent
    if isinstance(obj, dict):
        satirlar = []
        for anahtar, deger in obj.items():
            if isinstance(deger, (dict, list)):
                satirlar.append(f"{girinti}{anahtar}:")
                satirlar.append(format_result(deger, indent + 1))
            else:
                satirlar.append(f"{girinti}{anahtar}: {_skaler(deger)}")
        return "\n".join(satirlar)
    if isinstance(obj, (list, tuple)):
        return "\n".join(f"{girinti}- {format_result(o, 0).strip()}"
                         if not isinstance(o, (dict, list))
                         else format_result(o, indent + 1) for o in obj)
    return f"{girinti}{_skaler(obj)}"


def _skaler(deger) -> str:
    if isinstance(deger, float):
        return f"{deger:.6g}"
    return str(deger)
