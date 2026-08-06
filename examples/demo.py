"""
Agrista Demo — Kapsamlı Kullanım Örneği
Comprehensive usage demonstration of Agrista features.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from agrista.data import AgristaData
from agrista.analysis import (
    descriptive_stats,
    correlation_analysis,
    t_test,
    multiple_regression,
)
from agrista.models import GrowthModel, YieldPredictionModel, RiskAnalysisModel
from agrista.viz import AgristaPlotter
from examples.generate_sample_data import generate_crop_yield_data


def demo_1_data_loading():
    """Demo 1: Veri Yükleme ve Yönetimi"""
    print("\n" + "=" * 60)
    print("📊 DEMO 1: VERİ YÜKLEME VE YÖNETİMİ")
    print("=" * 60)
    
    # Sentetik veri üret
    df = generate_crop_yield_data(50)
    data = AgristaData(df)
    
    print(f"\n✅ Veri yüklendi: {data}")
    print("\n📋 Metadata:")
    for key, val in data.metadata.items():
        if isinstance(val, list):
            print(f"   • {key}: {len(val)} sütun")
        else:
            print(f"   • {key}: {val}")
    
    print("\n📄 İlk 5 satır:")
    print(data.head())


def demo_2_descriptive_stats():
    """Demo 2: Betimsel İstatistik"""
    print("\n" + "=" * 60)
    print("📈 DEMO 2: BETİMSSEL İSTATİSTİK")
    print("=" * 60)
    
    df = generate_crop_yield_data(100)
    
    # Verim verisi için betimsel istatistik
    results = descriptive_stats(df["yield_tons_per_ha"])
    
    print("\n🌾 Verim (ton/da) Betimsel İstatistikler:")
    for stat_name, value in results.items():
        if isinstance(value, dict):
            print(f"\n   {stat_name}:")
            for k, v in value.items():
                print(f"      • {k}: {v}")
        else:
            print(f"   • {stat_name}: {value}")


def demo_3_correlation():
    """Demo 3: Korelasyon Analizi"""
    print("\n" + "=" * 60)
    print("🔗 DEMO 3: KORELASYON ANALİZİ")
    print("=" * 60)
    
    df = generate_crop_yield_data(100)
    results = correlation_analysis(df[["irrigation_liters_per_ha", "fertilizer_kg_per_ha", 
                                       "rainfall_mm", "soil_ph", "yield_tons_per_ha"]])
    
    print("\n📊 Anlamlı Korelasyon Çiftleri:")
    for pair in results["significant_pairs"]:
        strength = "Güçlü" if abs(pair["correlation"]) > 0.5 else "Zayıf"
        direction = "Pozitif" if pair["correlation"] > 0 else "Negatif"
        print(f"\n   {pair['variable_1']} ↔ {pair['variable_2']}")
        print(f"      • Korelasyon: {pair['correlation']:.3f} ({strength} {direction})")
        print(f"      • p-değeri: {pair['p_value']:.4f}")


def demo_4_t_test():
    """Demo 4: T-Testi"""
    print("\n" + "=" * 60)
    print("🧪 DEMO 4: İKİ ÖRNEKLEM T-TESTİ")
    print("=" * 60)
    
    df = generate_crop_yield_data(100)
    
    # Buğday vs Mısır verim karşılaştırması
    buğday = df[df["crop_type"] == "buğday"]["yield_tons_per_ha"]
    mısır = df[df["crop_type"] == "mısır"]["yield_tons_per_ha"]
    
    if len(buğday) > 1 and len(mısır) > 1:
        result = t_test(buğday, mısır)
        
        print("\n🌾 Buğday vs 🌽 Mısır Verim Karşılaştırması:")
        print(f"   • Buğday ortalaması: {result['group1_mean']:.3f} ton/da")
        print(f"   • Mısır ortalaması: {result['group2_mean']:.3f} ton/da")
        print(f"   • t-değeri: {result['t_statistic']:.4f}")
        print(f"   • p-değeri: {result['p_value']:.6f}")
        print(f"   • Cohen's d (etki boyutu): {result['cohens_d']:.3f}")
        
        if result["significant_at_005"]:
            print("\n   ✅ İstatistiksel olarak anlamlı fark var (p < 0.05)")
        else:
            print("\n   ❌ İstatistiksel olarak anlamlı fark yok")


def demo_5_regression():
    """Demo 5: Regresyon Analizi"""
    print("\n" + "=" * 60)
    print("📉 DEMO 5: REGRESYON ANALİZİ")
    print("=" * 60)
    
    df = generate_crop_yield_data(100)
    
    # Çoklu regresyon: verim = f(sulama, gübre, yağış, toprak pH)
    result = multiple_regression(df, "yield_tons_per_ha", 
                                  ["irrigation_liters_per_ha", "fertilizer_kg_per_ha", 
                                   "rainfall_mm", "soil_ph"])
    
    print("\n🌾 Verim Tahmin Modeli:")
    print(f"   • R²: {result['r_squared']:.4f}")
    print(f"   • Düzeltilmiş R²: {result['adj_r_squared']:.4f}")
    print(f"   • F-değeri: {result['f_statistic']:.4f}")
    print("\n   Katsayılar:")
    
    # Intercept
    print(f"      • Sabit terim: {result['coefficients'].get('const', 0):.4f}")
    
    for factor, coeff in result["coefficients"].items():
        if factor == "const":
            continue
        if coeff is None:
            print(f"      • {factor}: model tarafından atlandı (çoklu bağlantı)")
            continue
        p_val = result["p_values"][factor]
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"      • {factor}: {coeff:.6f} (p={p_val:.4f} {sig})")


def demo_6_growth_model():
    """Demo 6: Büyüme Modeli"""
    print("\n" + "=" * 60)
    print("🌱 DEMO 6: BİTKİ BÜYÜME MODELİ (LOJİSTİK)")
    print("=" * 60)
    
    # Lojistik model oluştur ve fit et
    model = GrowthModel(model_type="logistic")
    
    # Sentetik büyüme verisi üret
    np.random.seed(42)
    t = np.linspace(0, 30, 30)
    K_true, r_true, t0_true = 100, 0.3, 15
    y_true = model.logistic(t, K_true, r_true, t0_true)
    y_noisy = y_true + np.random.normal(0, 2, len(t))
    
    # Modeli fit et
    params = model.fit_logistic(t, y_noisy)
    
    print("\n📊 Lojistik Büyüme Modeli Parametreleri:")
    print(f"   • Taşıma kapasitesi (K): {params['K']:.2f}")
    print(f"   • Büyüme oranı (r): {params['r']:.4f}")
    print(f"   • Infleksiyon noktası (t0): {params['t0']:.2f} gün")
    
    # Tahminler
    t_pred = np.linspace(0, 30, 100)
    y_pred = model.predict(t_pred)
    
    print("\n📈 Büyüme Eğrisi:")
    for day in [0, 5, 10, 15, 20, 25, 30]:
        idx = min(int(day * 100 / 30), len(y_pred) - 1)
        print(f"   Gün {day:2d}: {y_pred[idx]:.1f} kg/da")


def demo_7_yield_prediction():
    """Demo 7: Verim Tahmin Modeli"""
    print("\n" + "=" * 60)
    print("🎯 DEMO 7: VERİMD TAHMİN MODELİ")
    print("=" * 60)
    
    model = YieldPredictionModel()
    model.intercept = -5.0
    model.add_factor("sulama", 0.001)
    model.add_factor("gubre", 0.005)
    model.add_factor("yagis", 0.002)
    
    # Tahmin yap
    factor_values = {
        "sulama": 8000,   # litre/da
        "gubre": 200,     # kg/da
        "yagis": 500,     # mm
    }
    
    predicted_yield = model.predict(factor_values)
    
    print("\n🌾 Verim Tahmini:")
    print(f"   • Sulama: {factor_values['sulama']} litre/da")
    print(f"   • Gübre: {factor_values['gubre']} kg/da")
    print(f"   • Yağış: {factor_values['yagis']} mm")
    print(f"\n   📊 Tahmini Verim: {predicted_yield:.2f} ton/da")


def demo_8_risk_analysis():
    """Demo 8: Risk Analizi"""
    print("\n" + "=" * 60)
    print("⚠️ DEMO 8: TARIMSAL RİSK ANALİZİ (MONTE CARLO)")
    print("=" * 60)
    
    # Ortalama verim ve standart sapma
    mean_yield = 5.0  # ton/da
    std_yield = 1.2   # ton/da
    
    risk = RiskAnalysisModel()
    
    # VaR hesabı
    var_95 = risk.value_at_risk(mean_yield, std_yield, confidence_level=0.95)
    cv = risk.coefficient_of_variation([4.5, 5.2, 4.8, 6.1, 3.9, 5.5])
    
    print("\n📊 Risk Analizi:")
    print(f"   • Ortalama verim: {mean_yield} ton/da")
    print(f"   • Standart sapma: {std_yield} ton/da")
    print(f"   • Korelasyon Katsayısı (CV): {cv:.1f}%")
    
    # 95% güven ile minimum verim
    print(f"\n   ⚠️ 95% Güvenle Minimum Verim (VaR): {var_95:.2f} ton/da")
    print("      → %5 ihtimalle verim bu değerin altında kalır")
    
    # Monte Carlo simülasyonu
    mc_results = risk.monte_carlo_yield(mean_yield, std_yield)
    
    print("\n🎲 Monte Carlo Simülasyonu (10,000 deneme):")
    print(f"   • Ortalama: {mc_results['mean']:.2f} ton/da")
    print(f"   • Min: {mc_results['min']:.2f} ton/da")
    print(f"   • Max: {mc_results['max']:.2f} ton/da")
    
    for level, data in mc_results["percentiles"].items():
        print(f"   • {level} VaR: {data['var']:.2f} ton/da")


def demo_9_experimental_design():
    """Demo 9: Deneysel Tasarım"""
    print("\n" + "=" * 60)
    print("🔬 DEMO 9: DENEYSEL TASARIM")
    print("=" * 60)
    
    design = ExperimentalDesign()
    
    # RCBD tasarımı
    rcbd = design.random_complete_block(n_treatments=4, n_blocks=5)
    
    print("\n📋 Rastgele Tam Bloklama Deneyi (RCBD):")
    print(f"   • Uygulama sayısı: {rcbd['n_treatments']}")
    print(f"   • Blok sayısı: {rcbd['n_blocks']}")
    print(f"   • Toplam parsel: {rcbd['total_plots']}")
    
    print("\n📊 Rastgele Atamalar:")
    for block, assignments in rcbd["assignments"].items():
        print(f"   {block}: {' → '.join(assignments)}")
    
    # Faktöriyel tasarım
    factorial = design.factorial_design({
        "gubre_seviyesi": ["Düşük", "Orta", "Yüksek"],
        "sulama_seviyesi": ["Az", "Çok"],
    })
    
    print("\n📋 Faktöriyel Deney:")
    print(f"   • Tasarım: {factorial['design']}")
    print(f"   • Toplam işlem: {factorial['total_treatments']}")
    print(factorial["dataframe"].to_string(index=False))


def demo_10_visualization(save_plots: bool = False):
    """Demo 10: Görselleştirme"""
    print("\n" + "=" * 60)
    print("📊 DEMO 10: GÖRSELLEŞTİRME")
    print("=" * 60)
    
    df = generate_crop_yield_data(100)
    plotter = AgristaPlotter()
    
    # Histogram
    fig1 = plotter.histogram(df["yield_tons_per_ha"], title="Verim Dağılımı")
    print("   ✅ Histogram oluşturuldu")
    
    # Korelasyon ısı haritası
    fig2 = plotter.correlation_heatmap(
        df[["irrigation_liters_per_ha", "fertilizer_kg_per_ha", 
            "rainfall_mm", "soil_ph", "yield_tons_per_ha"]],
        title="Korelasyon Isı Haritası"
    )
    print("   ✅ Korelasyon ısı haritası oluşturuldu")
    
    # Saçılım grafiği (regresyon çizgisi ile)
    fig3 = plotter.scatter(
        df["fertilizer_kg_per_ha"],
        df["yield_tons_per_ha"],
        title="Gübre Miktarı vs Verim",
        xlabel="Gübre (kg/da)",
        ylabel="Verim (ton/da)",
        regression_line=True,
    )
    print("   ✅ Saçılım grafiği oluşturuldu")
    
    # Çubuk grafiği - ürün türlerine göre ortalama verim
    crop_means = df.groupby("crop_type")["yield_tons_per_ha"].mean()
    fig4 = plotter.bar_chart(
        crop_means.index.tolist(),
        crop_means.values,
        title="Ürün Türüne Göre Ortalama Verim",
        xlabel="Ürün Türü",
        ylabel="Ortalama Verim (ton/da)",
    )
    print("   ✅ Çubuk grafiği oluşturuldu")
    
    print("\n📁 Grafikler 'agrista_plots/' klasörüne kaydediliyor...")
    
    if save_plots:
        import os
        os.makedirs("agrista_plots", exist_ok=True)
        plotter.save("agrista_plots/histogram.png", fig=fig1)
        plotter.save("agrista_plots/correlation_heatmap.png", fig=fig2)
        plotter.save("agrista_plots/scatter_regression.png", fig=fig3)
        plotter.save("agrista_plots/bar_chart.png", fig=fig4)
        print("📁 Grafikler kaydedildi: agrista_plots/")


def demo_11_branch_modules():
    """Demo 11: Alt Branş Modülleri"""
    print("\n" + "=" * 60)
    print("🌿 DEMO 11: ALT BRANŞ MODÜLLERİ")
    print("=" * 60)
    
    from agrista.protection import audpc, abbott_efficiency
    from agrista.genetics import ammi_analysis
    from agrista.engineering import rsm_ccd, rsm_fit, find_optimum
    from agrista.economics import dea_efficiency
    
    # Bitki koruma: AUDPC
    zaman = [0, 7, 14, 21, 28]
    siddet = [2, 12, 35, 60, 78]
    a = audpc(zaman, siddet)
    print(f"\n🦠 [Bitki Koruma] AUDPC: {a['audpc']:.0f} "
          f"(bağıl: {a['relative_audpc']:.1f})")
    print(f"   Abbott etkinliği (kontrol %80, uygulama %15): "
          f"%{abbott_efficiency(80, 15)['efficacy_pct']:.1f}")
    
    # Tarla bitkileri: AMMI
    rng = np.random.default_rng(7)
    rows = []
    for g, gv in {"G1": 0.0, "G2": 0.8, "G3": -0.4}.items():
        for e, ev in {"E1": 0.0, "E2": 0.6, "E3": 1.2}.items():
            for _ in range(2):
                rows.append({"gen": g, "cev": e, "verim": 5 + gv + ev + rng.normal(0, 0.1)})
    am = ammi_analysis(pd.DataFrame(rows), "gen", "cev", "verim")
    print(f"\n🌾 [Tarla Bitkileri] AMMI IPCA1 açıklanan etkileşim varyansı: "
          f"%{am['ipca_explained_variance']['IPCA1']*100:.0f}")
    
    # Tarım makineleri: RSM optimizasyonu
    ccd = rsm_ccd({"hiz": (100, 300), "derinlik": (2, 8)})
    ccd["yakıt"] = ((ccd["hiz"] - 180) ** 2) / 800 + (ccd["derinlik"] - 4) ** 2 + 5 \
        + rng.normal(0, 0.1, len(ccd))
    fit = rsm_fit(ccd, "yakıt", ["hiz", "derinlik"])
    opt = find_optimum(fit, {"hiz": (100, 300), "derinlik": (2, 8)}, maximize=False)
    print(f"\n⚙️ [Tarım Makineleri] RSM optimum (min yakıt): "
          f"hız={opt['optimal_values']['hiz']:.0f}, "
          f"derinlik={opt['optimal_values']['derinlik']:.1f} "
          f"(R²={fit['r_squared']:.3f})")
    
    # Tarım ekonomisi: DEA
    inp = pd.DataFrame({"arazi": [10, 20, 15], "emek": [5, 8, 6]},
                       index=["Ç1", "Ç2", "Ç3"])
    out = pd.DataFrame({"ürün": [100, 150, 120]}, index=["Ç1", "Ç2", "Ç3"])
    dea = dea_efficiency(inp, out)
    print(f"\n💰 [Tarım Ekonomisi] DEA ortalama etkinlik: "
          f"{dea['mean_efficiency']:.2f} (etkin: {', '.join(dea['efficient_dmus'])})")


if __name__ == "__main__":
    from agrista.experimental import ExperimentalDesign
    
    print("\n" + "🌾" * 20)
    print("  AGRISTA — TARIMSAL İSTATİSTİK YAZILIMI v0.1.0")
    print("  Kapsamlı Kullanım Demo'su")
    print("🌾" * 20)
    
    demo_1_data_loading()
    demo_2_descriptive_stats()
    demo_3_correlation()
    demo_4_t_test()
    demo_5_regression()
    demo_6_growth_model()
    demo_7_yield_prediction()
    demo_8_risk_analysis()
    demo_9_experimental_design()
    demo_10_visualization(save_plots=True)
    demo_11_branch_modules()
    
    print("\n" + "=" * 60)
    print("✅ TÜM DEMO'LAR TAMAMLANDI!")
    print("=" * 60)
