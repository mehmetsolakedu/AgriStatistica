"""
Agrista — Tarımsal İstatistik Yazılımı
Agricultural Statistical Analysis Toolkit
"""

__version__ = "0.1.0"
__author__ = "Agrista Team"

from agrista.data import AgristaData, load_csv, load_excel, load_json
from agrista.analysis import (
    descriptive_stats,
    correlation_analysis,
    t_test,
    anova_one_way,
    linear_regression,
    multiple_regression,
    chi_square_test,
)
from agrista.models import GrowthModel, YieldPredictionModel, RiskAnalysisModel
from agrista.viz import AgristaPlotter

# Alt branş modülleri (literatür logu: docs/01_ALT_BRANS_ISTATISTIK_LITERATUR_LOG.md)
from agrista import protection   # Bitki koruma: probit, Abbott, AUDPC
from agrista import genetics     # Tarla bitkileri: path, PCA, AMMI, GGE
from agrista import engineering  # Tarım makineleri: RSM, Taguchi
from agrista import animal       # Zootekni: karışık modeller, laktasyon
from agrista import sensory      # Bahçe bitkileri: duyusal analiz
from agrista import spatial      # Toprak bilimi: varyogram, IDW
from agrista import economics    # Tarım ekonomisi: DEA, logit
from agrista import transform    # Premium Program Transform: compute, recode, bin, impute
from agrista import forecasting  # Premium Program Forecasting: Holt-Winters, ARIMA, ayrıştırma
from agrista import survival     # Premium Program Survival: Kaplan-Meier, log-rank
from agrista import quality      # Premium Program Quality Control: X-bar/R, p, Pareto
from agrista import marketing    # Premium Program Direct Marketing: RFM, mailing, profil

__all__ = [
    "__version__",
    "AgristaData",
    "load_csv",
    "load_excel",
    "load_json",
    "descriptive_stats",
    "correlation_analysis",
    "t_test",
    "anova_one_way",
    "linear_regression",
    "multiple_regression",
    "chi_square_test",
    "GrowthModel",
    "YieldPredictionModel",
    "RiskAnalysisModel",
    "AgristaPlotter",
    "protection",
    "genetics",
    "engineering",
    "animal",
    "sensory",
    "spatial",
    "economics",
    "transform",
    "forecasting",
    "survival",
    "quality",
    "marketing",
]
