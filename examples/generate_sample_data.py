"""
Agrista Örnek Veri Üretici
Example data generator for Agrista testing and demos.
"""

import numpy as np
import pandas as pd


def generate_crop_yield_data(
    n_farms: int = 100,
    seed: int = 42
) -> pd.DataFrame:
    """Sentetik çiftlik verim verisi üretir."""
    np.random.seed(seed)
    
    data = {
        "farm_id": range(1, n_farms + 1),
        "crop_type": np.random.choice(["buğday", "arpa", "mısır", "pamuk"], n_farms),
        "area_hectares": np.round(np.random.uniform(5, 200, n_farms), 1),
        "irrigation_liters_per_ha": np.round(np.random.uniform(3000, 12000, n_farms), 0),
        "fertilizer_kg_per_ha": np.round(np.random.uniform(50, 400, n_farms), 1),
        "pesticide_applications": np.random.randint(0, 8, n_farms),
        "rainfall_mm": np.round(np.random.uniform(200, 800, n_farms), 1),
        "soil_ph": np.round(np.random.uniform(5.5, 8.0, n_farms), 1),
        "temperature_avg_c": np.round(np.random.uniform(15, 30, n_farms), 1),
    }
    
    df = pd.DataFrame(data)
    
    # Verim hesapla (gerçekçi bir model)
    base_yield = {
        "buğday": 4.5,
        "arpa": 3.8,
        "mısır": 7.2,
        "pamuk": 2.5,
    }
    
    yields = []
    for _, row in df.iterrows():
        base = base_yield[row["crop_type"]]
        yield_val = (
            base
            + 0.001 * row["irrigation_liters_per_ha"]
            + 0.005 * row["fertilizer_kg_per_ha"]
            - 0.3 * abs(row["soil_ph"] - 6.5)
            + np.random.normal(0, 0.8)
        )
        yields.append(max(0.5, round(yield_val, 2)))
    
    df["yield_tons_per_ha"] = yields
    
    return df


def generate_growth_data(
    n_time_points: int = 30,
    seed: int = 42
) -> pd.DataFrame:
    """Bitki büyüme zaman serisi üretir (lojistik model)."""
    np.random.seed(seed)
    
    t = np.linspace(0, n_time_points, n_time_points)
    K = 100  # Taşıma kapasitesi
    r = 0.3  # Büyüme oranı
    t0 = 15  # Infleksiyon noktası
    
    growth = K / (1 + np.exp(-r * (t - t0)))
    growth_noisy = growth + np.random.normal(0, 2, n_time_points)
    
    return pd.DataFrame({
        "day": t.astype(int),
        "biomass_kg_per_ha": np.round(np.maximum(growth_noisy, 0), 2),
    })


def generate_soil_analysis_data(
    n_samples: int = 50,
    seed: int = 42
) -> pd.DataFrame:
    """Toprak analiz verisi üretir."""
    np.random.seed(seed)
    
    return pd.DataFrame({
        "sample_id": range(1, n_samples + 1),
        "location": [f"Alan_{i+1}" for i in range(n_samples)],
        "ph": np.round(np.random.normal(6.5, 0.8, n_samples).clip(4.5, 8.5), 1),
        "organic_matter_pct": np.round(np.abs(np.random.normal(2.5, 1.0, n_samples)), 2),
        "nitrogen_kg_per_ha": np.round(np.abs(np.random.normal(80, 30, n_samples)), 1),
        "phosphorus_kg_per_ha": np.round(np.abs(np.random.normal(25, 10, n_samples)), 1),
        "potassium_kg_per_ha": np.round(np.abs(np.random.normal(150, 50, n_samples)), 1),
        "moisture_pct": np.round(np.abs(np.random.normal(18, 5, n_samples)), 1),
    })


if __name__ == "__main__":
    print("🌾 Agrista Örnek Veri Üretici")
    print("=" * 40)
    
    # Çiftlik verim verisi
    crop_df = generate_crop_yield_data(50)
    print("\n📊 Çiftlik Verim Verisi:")
    print(crop_df.head())
    print(f"\nToplam kayıt: {len(crop_df)}")
    
    # Büyüme verisi
    growth_df = generate_growth_data()
    print("\n🌱 Bitki Büyüme Verisi:")
    print(growth_df.head(10))
