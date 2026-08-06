"""
Agrista Models Module — Tarımsal Modeller
Agricultural statistical and predictive models.
"""

from __future__ import annotations

import numpy as np

from agrista.models.glmm import glmm

__all__ = ["glmm", "GrowthModel", "YieldPredictionModel", "RiskAnalysisModel"]


class GrowthModel:
    """Bitki büyüme modelleri."""
    
    def __init__(self, model_type: str = "logistic"):
        self.model_type = model_type
        self.params = {}
    
    def logistic(self, t: float | np.ndarray, K: float, r: float, t0: float) -> float | np.ndarray:
        """Lojistik büyüme modeli.
        
        K: Taşıma kapasitesi (maksimum büyüklük)
        r: Büyüme oranı
        t0: Infleksiyon noktası
        """
        return K / (1 + np.exp(-r * (t - t0)))
    
    def gompertz(self, t: float | np.ndarray, A: float, b: float, r: float) -> float | np.ndarray:
        """Gompertz büyüme modeli."""
        return A * np.exp(-b * np.exp(-r * t))
    
    def von_bertalanffy(self, t: float | np.ndarray, L_inf: float, k: float, t0: float) -> float | np.ndarray:
        """Von Bertalanffy büyüme modeli."""
        return L_inf * (1 - np.exp(-k * (t - t0))) ** 3
    
    def monomolecular(self, t: float | np.ndarray, A: float, k: float, t0: float = 0.0) -> float | np.ndarray:
        """Monomoleküler büyüme modeli (fitopatolojide hastalık ilerlemesi için de kullanılır)."""
        return A * (1 - np.exp(-k * (t - t0)))
    
    def wood(self, t: float | np.ndarray, a: float, b: float, c: float) -> float | np.ndarray:
        """Wood (gamma) eğrisi — laktasyon ve büyüme hızı modellemesi."""
        t = np.asarray(t, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return a * np.power(np.where(t > 0, t, 1e-9), b) * np.exp(-c * t)
    
    def fit_logistic(self, time_points: np.ndarray, measurements: np.ndarray):
        """Lojistik model parametrelerini tahmin et."""
        from scipy.optimize import curve_fit
        
        def log_func(t, K, r, t0):
            return self.logistic(t, K, r, t0)
        
        time_points = np.asarray(time_points, dtype=float)
        measurements = np.asarray(measurements, dtype=float)
        mask = ~(np.isnan(time_points) | np.isnan(measurements))
        time_points, measurements = time_points[mask], measurements[mask]
        
        if len(time_points) < 4:
            raise ValueError("Lojistik model için en az 4 geçerli veri noktası gerekli")
        
        # Başlangıç tahminleri: K ~ gözlenen maksimum, t0 ~ zaman ortancası,
        # r ~ maksimum büyüme eğiminden türetilir
        K0 = float(np.max(measurements))
        if K0 <= 0:
            K0 = 1.0
        t0_guess = float(np.median(time_points))
        grad = np.gradient(measurements, time_points)
        r0 = float(4.0 * np.max(grad) / K0)
        if not np.isfinite(r0) or r0 <= 0:
            r0 = 0.1
        
        popt, _ = curve_fit(log_func, time_points, measurements, p0=[K0, r0, t0_guess], maxfev=10000)
        self.params = {"K": popt[0], "r": popt[1], "t0": popt[2]}
        return self.params
    
    def fit_monomolecular(self, time_points: np.ndarray, measurements: np.ndarray):
        """Monomoleküler model parametrelerini tahmin et."""
        from scipy.optimize import curve_fit
        
        time_points = np.asarray(time_points, dtype=float)
        measurements = np.asarray(measurements, dtype=float)
        mask = ~(np.isnan(time_points) | np.isnan(measurements))
        time_points, measurements = time_points[mask], measurements[mask]
        if len(time_points) < 4:
            raise ValueError("Monomoleküler model için en az 4 geçerli veri noktası gerekli")
        
        A0 = float(np.max(measurements)) or 1.0
        k0 = 0.05
        popt, _ = curve_fit(self.monomolecular, time_points, measurements,
                            p0=[A0, k0, 0.0], maxfev=10000)
        self.params = {"A": popt[0], "k": popt[1], "t0": popt[2]}
        self.model_type = "monomolecular"
        return self.params
    
    def predict(self, t: float | np.ndarray) -> float | np.ndarray:
        """Model ile tahmin yap."""
        if not self.params:
            raise ValueError("Model parametreleri henüz fit edilmemiş")
        
        if self.model_type == "logistic":
            return self.logistic(t, self.params["K"], self.params["r"], self.params["t0"])
        elif self.model_type == "gompertz":
            return self.gompertz(t, self.params["A"], self.params["b"], self.params["r"])
        elif self.model_type == "von_bertalanffy":
            return self.von_bertalanffy(t, self.params["L_inf"], self.params["k"], self.params["t0"])
        elif self.model_type == "monomolecular":
            return self.monomolecular(t, self.params["A"], self.params["k"], self.params.get("t0", 0.0))
        elif self.model_type == "wood":
            return self.wood(t, self.params["a"], self.params["b"], self.params["c"])
        else:
            raise ValueError(f"Desteklenmeyen model türü: {self.model_type}")


class YieldPredictionModel:
    """Verim tahmin modeli."""
    
    def __init__(self):
        self.factors = {}
        self.coefficients = {}
        self.intercept = 0.0
    
    def add_factor(self, name: str, coefficient: float):
        """Bir faktör ve katsayısı ekle."""
        self.factors[name] = True
        self.coefficients[name] = coefficient
    
    def predict(self, factor_values: dict) -> float:
        """Verim tahmini yap.
        
        factor_values: {faktör_adi: değer} formatında sözlük
        """
        prediction = self.intercept
        for factor, coeff in self.coefficients.items():
            if factor not in factor_values:
                raise KeyError(f"'{factor}' faktörü bulunamadı. Mevcut faktörler: {list(self.coefficients.keys())}")
            prediction += coeff * factor_values[factor]
        return float(prediction)
    
    def summary(self) -> dict:
        """Model özetini döndür."""
        return {
            "intercept": self.intercept,
            "factors": self.factors,
            "coefficients": self.coefficients,
        }


class RiskAnalysisModel:
    """Tarımsal risk analizi modeli."""
    
    def __init__(self):
        pass
    
    @staticmethod
    def value_at_risk(mean_yield: float, std_yield: float, confidence_level: float = 0.95) -> float:
        """VaR (Value at Risk) hesabı.
        
        Belirli bir güven düzeyinde maksimum kayıp miktarını tahmin eder.
        """
        from scipy import stats
        
        z_score = stats.norm.ppf(1 - confidence_level)
        var = mean_yield + z_score * std_yield
        return float(var)
    
    @staticmethod
    def coefficient_of_variation(data: np.ndarray | list) -> float:
        """Varyasyon katsayısı (CV, %)."""
        arr = np.asarray(data, dtype=float)
        mean = np.mean(arr)
        std = np.std(arr, ddof=1)
        if mean == 0:
            return float('inf')
        return float(std / abs(mean)) * 100
    
    @staticmethod
    def monte_carlo_yield(
        mean_yield: float,
        std_yield: float,
        n_simulations: int = 10000,
        confidence_levels: list[float] = None
    ) -> dict:
        """Monte Carlo simülasyonu ile verim risk analizi."""
        if confidence_levels is None:
            confidence_levels = [0.90, 0.95, 0.99]
        
        np.random.seed(42)
        simulated_yields = np.random.normal(mean_yield, std_yield, n_simulations)
        
        results = {
            "mean": float(np.mean(simulated_yields)),
            "std": float(np.std(simulated_yields)),
            "min": float(np.min(simulated_yields)),
            "max": float(np.max(simulated_yields)),
            "percentiles": {},
            "probability_below_mean": float(np.mean(simulated_yields < mean_yield)),
        }
        
        for cl in confidence_levels:
            var = float(np.percentile(simulated_yields, (1 - cl) * 100))
            results["percentiles"][f"{int(cl*100)}%"] = {
                "var": var,
                "probability_below_var": float(np.mean(simulated_yields < var)),
            }
        
        return results
