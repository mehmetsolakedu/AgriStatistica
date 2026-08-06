"""
Agrista CLI — Komut Satırı Arayüzü
Command-line interface for Agrista.

Argümansız çalıştırıldığında (yalnızca `agrista`) etkileşimli ana menü açılır;
alt komutlar betik/otomasyon kullanımına yöneliktir.
"""

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd
import numpy as np

from agrista.models import glmm as glmm_fit


def _load_file(filepath: str, sheet: int = 0):
    """CSV veya Excel dosyasını AgristaData olarak yükler."""
    from agrista.data import load_csv, load_excel
    
    if filepath.endswith(".csv"):
        return load_csv(filepath)
    elif filepath.endswith((".xlsx", ".xls")):
        return load_excel(filepath, sheet_name=sheet)
    else:
        raise ValueError("Desteklenmeyen dosya formatı. CSV veya Excel kullanın.")


# ---------------------------------------------------------------------------
# Çıktı biçimlendiricileri (komutlar ve menü tarafından paylaşılır)
# ---------------------------------------------------------------------------


def _show_info(filepath: str, sheet: int = 0):
    data = _load_file(filepath, sheet)
    info = data.info()
    click.echo(f"📊 Dosya: {filepath}")
    click.echo(f"   Satır sayısı: {info['shape'][0]}")
    click.echo(f"   Sütun sayısı: {info['shape'][1]}")
    click.echo("\nSütunlar:")
    for col, dtype in info["dtypes"].items():
        null_count = info["null_counts"].get(col, 0)
        click.echo(f"   • {col} ({dtype}) — {null_count} boş değer")


def _show_corr(filepath: str, method: str = "pearson"):
    from agrista.analysis import correlation_analysis
    
    data = _load_file(filepath)
    result = correlation_analysis(data.dataframe, method=method)
    click.echo(f"📈 Korelasyon Analizi ({method.capitalize()})")
    click.echo("=" * 50)
    
    sig_pairs = result["significant_pairs"]
    if sig_pairs:
        for pair in sig_pairs[:10]:
            click.echo(f"   {pair['variable_1']} ↔ {pair['variable_2']}: "
                       f"r={pair['correlation']:.3f} (p={pair['p_value']:.4f})")
    else:
        click.echo("   Anlamlı korelasyon çifti bulunamadı.")


def _show_ttest(csv1: str, csv2: str):
    from agrista.analysis import t_test
    
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)
    
    num_cols1 = df1.select_dtypes(include=[np.number]).columns
    num_cols2 = df2.select_dtypes(include=[np.number]).columns
    
    if num_cols1.empty or num_cols2.empty:
        click.echo("Sayısal sütun bulunamadı.")
        return
    
    col = num_cols1[0]
    result = t_test(df1[col], df2[col])
    
    click.echo("📊 İki Örneklem T-Testi")
    click.echo("=" * 50)
    click.echo(f"   Grup 1 ortalaması: {result['group1_mean']:.3f}")
    click.echo(f"   Grup 2 ortalaması: {result['group2_mean']:.3f}")
    click.echo(f"   t-değeri: {result['t_statistic']:.4f}")
    click.echo(f"   p-değeri: {result['p_value']:.6f}")
    click.echo(f"   Cohen's d: {result['cohens_d']:.3f}")
    sig = "✅ Anlamlı (p < 0.05)" if result["significant_at_005"] else "❌ Anlamsız (p ≥ 0.05)"
    click.echo(f"   Sonuç: {sig}")


def _print_tukey_result(result: dict):
    click.echo("📊 Tukey HSD Sonuçları")
    click.echo("=" * 50)
    for name, mean in sorted(result["group_means"].items(), key=lambda kv: -kv[1]):
        click.echo(f"   {name}: {mean:.3f}")
    click.echo(f"\n   Anlamlı çiftler ({len(result['significant_pairs'])}):")
    for pair in result["significant_pairs"]:
        click.echo(f"   • {pair['group_1']} ↔ {pair['group_2']}: "
                   f"fark={pair['mean_diff']:.3f} (p={pair['p_value']:.4f})")
    if not result["significant_pairs"]:
        click.echo("   • Anlamlı fark bulunamadı")


def _print_audpc_result(result: dict):
    click.echo("🦠 AUDPC Sonuçları")
    click.echo("=" * 50)
    click.echo(f"   AUDPC: {result['audpc']:.1f}")
    click.echo(f"   Bağıl AUDPC: {result['relative_audpc']:.2f}")
    click.echo(f"   Gözlem sayısı: {result['n_observations']}")
    click.echo(f"   Maksimum şiddet: {result['max_severity']:.1f}")


def _print_dea_result(result: dict, model: str):
    click.echo(f"📈 DEA Etkinlik Analizi ({model}, girdi yönelimli)")
    click.echo("=" * 50)
    for name, eff in sorted(result["efficiencies"].items(), key=lambda kv: -kv[1]):
        durum = "✅ etkin" if eff >= 0.999 else f"⚠️ %{(1-eff)*100:.1f} girdi fazlası"
        click.echo(f"   {name}: {eff:.3f} — {durum}")
    click.echo(f"\n   Ortalama etkinlik: {result['mean_efficiency']:.3f}")


def _print_frequencies_result(results: dict):
    click.echo("📊 Frekans Tabloları")
    click.echo("=" * 50)
    for col, res in results.items():
        click.echo(f"\n  ▸ {col}  (n={res['n_total']}, eksik={res['n_missing']})")
        click.echo("   {:<20} {:>6} {:>9} {:>11} {:>8}".format(
            "Değer", "Sayı", "Yüzde", "Geçerli %", "Küm. %"))
        for row in res["table"]:
            click.echo("   {:<20} {:>6} {:>8.1f}% {:>10.1f}% {:>7.1f}%".format(
                str(row["value"])[:20], row["count"], row["percent"],
                row["valid_percent"], row["cumulative_percent"]))


def _print_crosstabs_result(result: dict):
    click.echo("📊 Çapraz Tablo + Ki-Kare Bağımsızlık Testi")
    click.echo("=" * 50)
    click.echo("   Gözlenen frekanslar:")
    for line in result["observed"].to_string().splitlines():
        click.echo(f"   {line}")
    click.echo(f"\n   Ki-kare: {result['chi_square']:.4f}  "
               f"(sd={result['degrees_of_freedom']})")
    click.echo(f"   p-değeri: {result['p_value']:.6f}")
    click.echo(f"   Cramer's V: {result['cramers_v']:.3f}  "
               f"Phi: {result['phi']:.3f}")
    if result["expected_cell_warning"]:
        click.echo(f"   ⚠️ {result['low_expected_cells']} hücrede beklenen "
                   "frekans < 5 (ki-kare geçerliliğine dikkat)")
    sig = "✅ Bağımsızlık reddedildi (ilişki var)" if result["significant_at_005"] \
        else "❌ Bağımsızlık reddedilemedi"
    click.echo(f"   Sonuç: {sig}")


def _print_one_sample_result(result: dict):
    click.echo("📊 Tek Örneklem T-Testi")
    click.echo("=" * 50)
    click.echo(f"   Ortalama: {result['mean']:.3f} (test değeri: {result['test_value']:.3f})")
    click.echo(f"   %95 GA: [{result['ci95_lower']:.3f}, {result['ci95_upper']:.3f}]")
    click.echo(f"   t-değeri: {result['t_statistic']:.4f}  "
               f"p-değeri: {result['p_value']:.6f}  (sd={result['degrees_of_freedom']})")
    click.echo(f"   Cohen's d: {result['cohens_d']:.3f}")
    sig = "✅ Anlamlı fark var" if result["significant_at_005"] else "❌ Anlamlı fark yok"
    click.echo(f"   Sonuç: {sig}")


def _print_paired_result(result: dict):
    click.echo("📊 Eşleştirilmiş İki Örneklem T-Testi")
    click.echo("=" * 50)
    click.echo(f"   Ortalama fark: {result['mean_difference']:.3f} "
               f"(ss={result['std_difference']:.3f}, n={result['n_pairs']})")
    click.echo(f"   %95 GA: [{result['ci95_lower']:.3f}, {result['ci95_upper']:.3f}]")
    click.echo(f"   t-değeri: {result['t_statistic']:.4f}  "
               f"p-değeri: {result['p_value']:.6f}  (sd={result['degrees_of_freedom']})")
    click.echo(f"   Cohen's d: {result['cohens_d']:.3f}")
    sig = "✅ Anlamlı fark var" if result["significant_at_005"] else "❌ Anlamlı fark yok"
    click.echo(f"   Sonuç: {sig}")


def _print_mlogit_result(result: dict):
    click.echo("📈 Multinomial Lojistik Regresyon")
    click.echo("=" * 50)
    click.echo(f"   Referans kategori: {result['reference_category']}  "
               f"(n={result['n']})")
    click.echo(f"   Olabilirlik oranı ki-kare: {result['lr_chi_square']:.4f} "
               f"(sd={result['degrees_of_freedom']})  p={result['p_value']:.6f}")
    click.echo(f"   Pseudo R²: {result['pseudo_r_squared']:.3f}  "
               f"AIC={result['aic']:.1f}")
    for cat, coefs in result["coefficients"].items():
        ors = result["odds_ratios"][cat]
        click.echo(f"\n  ▸ Kategori: {cat} vs {result['reference_category']}")
        click.echo("   {:<18} {:>10} {:>10}".format("Değişken", "Katsayı", "Odds"))
        for name, beta in coefs.items():
            click.echo("   {:<18} {:>10.4f} {:>10.3f}".format(
                name[:18], beta, ors[name]))
    click.echo(f"\n   Sınıflandırma doğruluğu: %{result['classification_accuracy'] * 100:.1f}")
    sig = "✅ Model anlamlı" if result["significant_at_005"] else "❌ Model anlamsız"
    click.echo(f"   Sonuç: {sig}")


def _print_ologit_result(result: dict):
    click.echo("📈 Ordinal Lojistik Regresyon (kümülatif logit)")
    click.echo("=" * 50)
    click.echo(f"   Kategoriler: {' < '.join(result['categories'])}  (n={result['n']})")
    click.echo(f"   Olabilirlik oranı ki-kare: {result['lr_chi_square']:.4f} "
               f"(sd={result['degrees_of_freedom']})  p={result['p_value']:.6f}")
    click.echo(f"   Pseudo R²: {result['pseudo_r_squared']:.3f}  "
               f"AIC={result['aic']:.1f}")
    click.echo("\n   Konum katsayıları:")
    click.echo("   {:<18} {:>10} {:>10}".format("Değişken", "Katsayı", "Odds"))
    for name, beta in result["coefficients"].items():
        click.echo("   {:<18} {:>10.4f} {:>10.3f}".format(
            name[:18], beta, result["odds_ratios"][name]))
    click.echo("\n   Eşikler:")
    for name, theta in result["thresholds"].items():
        click.echo(f"   {name}: {theta:.4f}")
    sig = "✅ Model anlamlı" if result["significant_at_005"] else "❌ Model anlamsız"
    click.echo(f"\n   Sonuç: {sig}")


def _print_discriminant_result(result: dict):
    click.echo("🗂️ Ayrımsama (Discriminant) Analizi")
    click.echo("=" * 50)
    click.echo(f"   Wilks lambda: {result['wilks_lambda']:.4f}  "
               f"Ki-kare: {result['chi_square']:.4f} "
               f"(sd={result['degrees_of_freedom']})  p={result['p_value']:.6f}")
    for i, (ev, r, pv) in enumerate(zip(result["eigenvalues"],
                                        result["canonical_correlations"],
                                        result["percent_of_variance"]), start=1):
        click.echo(f"   Fonksiyon {i}: özdeğer={ev:.3f}  "
                   f"kanonik r={r:.3f}  varyans %{pv:.1f}")
    click.echo("\n   Grup sentroidleri (orijinal değişken uzayı):")
    for grp, means in result["group_centroids"].items():
        text = "  ".join(f"{k}={v:.3f}" for k, v in means.items())
        click.echo(f"   • {grp}: {text}")
    click.echo("\n   Sınıflandırma:")
    for row in result["classification"]:
        click.echo(f"   • {row['group']}: {row['n_correct']}/{row['n']} doğru "
                   f"(%{row['percent_correct']:.1f})")
    click.echo(f"   Genel doğruluk: %{result['overall_accuracy'] * 100:.1f}")
    sig = "✅ Gruplar ayrışıyor" if result["significant_at_005"] \
        else "❌ Gruplar ayrışmıyor"
    click.echo(f"   Sonuç: {sig}")


def _print_correspondence_result(result: dict):
    click.echo("🧭 Uyuşum (Correspondence) Analizi")
    click.echo("=" * 50)
    click.echo(f"   Ki-kare: {result['chi_square']:.4f} "
               f"(sd={result['degrees_of_freedom']})  "
               f"p={result['p_value']:.6f}")
    click.echo(f"   Toplam eylemsizlik: {result['total_inertia']:.4f}")
    for i, (ev, pct) in enumerate(zip(result["eigenvalues"],
                                      result["explained_percent"]), start=1):
        click.echo(f"   Boyut {i}: eylemsizlik={ev:.4f}  açıklanan %{pct:.1f}")
    click.echo("\n   Satır koordinatları:")
    for line in result["row_coordinates"].round(3).to_string().splitlines():
        click.echo(f"   {line}")
    click.echo("\n   Sütun koordinatları:")
    for line in result["column_coordinates"].round(3).to_string().splitlines():
        click.echo(f"   {line}")
    sig = "✅ Bağımsızlık reddedildi (ilişki var)" if result["independence_rejected"] \
        else "❌ Bağımsızlık reddedilemedi"
    click.echo(f"\n   Sonuç: {sig}")


def _print_glm_result(result: dict):
    click.echo(f"\n📊 {result['model']}")
    if result["model"] == "GLM Univariate":
        click.echo(f"   R² = {result['r_squared']:.4f}  (n={result['n_obs']})")
        click.echo("   Kaynak                SS        df    F        p")
        for r in result["anova_table"]:
            f_txt = f"{r['f_value']:8.3f}" if r["f_value"] is not None else "     -"
            p_txt = f"{r['p_value']:.4f}" if r["p_value"] is not None else "-"
            click.echo(f"   {r['source'][:20]:<20} {r['ss']:9.3f} {r['df']:4d} {f_txt} {p_txt:>8}")
        for ad, eta in result["effect_sizes"].items():
            click.echo(f"   Kısmi η² [{ad[:24]}] = {eta:.3f}")
        if result["posthoc"]:
            _print_tukey_result(result["posthoc"])
    else:
        w = result["within_effect"]
        click.echo(f"   Within F({w['df1']:.0f},{w['df2']:.0f}) = "
                   f"{w['f_value']:.3f}, p = {w['p_value']:.4f}")
        m = result["mauchly"]
        click.echo(f"   Mauchly W = {m['w']:.3f}, χ² = {m['chi2']:.2f}, "
                   f"p = {m['p_value']:.4f}")
        for ad in ("greenhouse_geisser", "huynh_feldt"):
            c = result["corrected"][ad]
            click.echo(f"   {ad}: ε = {result['epsilon'][ad]:.3f}, "
                       f"p = {c['p_value']:.4f}")


def _print_gee_result(result: dict):
    click.echo(f"\n📈 {result['model']} — aile: {result['family']}, "
               f"yapı: {result['cov_struct']}")
    click.echo(f"   {result['n_groups']} grup, {result['n_obs']} gözlem, "
               f"yakınsama: {'evet' if result['converged'] else 'hayır'}")
    if result["qic"] is not None:
        click.echo(f"   QIC = {result['qic']:.2f}")
    click.echo("   Değişken        Katsayı     SE       z       p")
    for ad, c in result["coefficients"].items():
        click.echo(f"   {ad[:14]:<14} {c['coefficient']:9.4f} "
                   f"{c['std_err']:8.4f} {c['z_value']:7.3f} "
                   f"{c['p_value']:8.4f}")


def _print_glmm_result(result: dict):
    click.echo(f"\n🧬 {result['model']} — aile: {result['family']}, "
               f"yöntem: {result['method']}")
    click.echo(f"   {result['n_groups']} grup, {result['n_obs']} gözlem, "
               f"yakınsama: {'evet' if result['converged'] else 'hayır'}"
               + (f" ({result['n_iterations']} yineleme)"
                  if result["method"] == "PQL" else ""))
    click.echo("   Sabit etki      Katsayı     SE       z       p")
    for ad, c in result["fixed_effects"].items():
        click.echo(f"   {ad[:14]:<14} {c['coefficient']:9.4f} "
                   f"{c['std_err']:8.4f} {c['z_value']:7.3f} "
                   f"{c['p_value']:8.4f}")
    ri = result["random_effects_variance"]["random_intercept"]
    click.echo(f"   Rastgele kesim varyansı = {ri:.4f}")
    if result["aic"] is not None:
        click.echo(f"   AIC = {result['aic']:.2f}")


# ---------------------------------------------------------------------------
# Komut grubu ve alt komutlar
# ---------------------------------------------------------------------------


@click.group(invoke_without_command=True)
@click.version_option(version="0.1.0", prog_name="Agrista")
@click.pass_context
def main(ctx):
    """Agrista — Tarımsal İstatistik Yazılımı

    Argümansız çalıştırıldığında etkileşimli ana menü açılır.
    """
    if ctx.invoked_subcommand is None:
        run_interactive_menu()


@main.command()
def menu():
    """Etkileşimli ana menüyü başlat."""
    run_interactive_menu()


@main.command()
@click.argument("filepath")
@click.option("--sheet", type=int, default=0, help="Excel sayfa numarası")
def info(filepath: str, sheet: int):
    """Veri dosyası hakkında bilgi göster."""
    try:
        _show_info(filepath, sheet)
    except FileNotFoundError:
        click.echo(f"❌ Dosya bulunamadı: {filepath}")
    except ValueError as e:
        click.echo(f"⚠️ {e}")
    except Exception as e:
        click.echo(f"❌ Hata: {e}")


@main.command()
@click.argument("filepath")
def describe(filepath: str):
    """Veri setinin betimsel istatistiklerini göster."""
    try:
        data = _load_file(filepath)
        click.echo(data.describe_numeric().to_string())
    except ValueError as e:
        click.echo(f"⚠️ {e}")
    except Exception as e:
        click.echo(f"❌ Hata: {e}")


@main.command()
@click.argument("filepath")
@click.option("--method", default="pearson",
              type=click.Choice(["pearson", "spearman"]), help="Korelasyon yöntemi")
def corr(filepath: str, method: str):
    """Korelasyon analizi yap."""
    try:
        _show_corr(filepath, method)
    except ValueError as e:
        click.echo(f"⚠️ {e}")
    except Exception as e:
        click.echo(f"❌ Hata: {e}")


@main.command()
@click.argument("csv1")
@click.argument("csv2")
def ttest(csv1: str, csv2: str):
    """İki grup arasında t-testi yap."""
    try:
        _show_ttest(csv1, csv2)
    except Exception as e:
        click.echo(f"❌ Hata: {e}")


@main.command()
@click.argument("filepath")
@click.option("--yanit", required=True, help="Yanıt değişkeni sütunu")
@click.option("--grup", required=True, help="Grup sütunu")
def tukey(filepath: str, yanit: str, grup: str):
    """Tukey HSD çoklu karşılaştırma testi yap."""
    try:
        from agrista.analysis import posthoc_tukey
        
        data = _load_file(filepath)
        _print_tukey_result(posthoc_tukey(data.dataframe, yanit, grup))
    except FileNotFoundError:
        click.echo(f"❌ Dosya bulunamadı: {filepath}")
    except Exception as e:
        click.echo(f"❌ Hata: {e}")


@main.command()
@click.argument("filepath")
@click.option("--zaman", default="zaman", help="Zaman sütunu")
@click.option("--siddet", default="siddet", help="Hastalık şiddeti sütunu")
def audpc(filepath: str, zaman: str, siddet: str):
    """Hastalık ilerleme verisinden AUDPC hesapla."""
    try:
        from agrista.protection import audpc as audpc_calc
        
        data = _load_file(filepath)
        df = data.dataframe
        if zaman not in df.columns or siddet not in df.columns:
            click.echo(f"❌ Sütun bulunamadı. Mevcut sütunlar: {list(df.columns)}")
            return
        
        _print_audpc_result(audpc_calc(df[zaman], df[siddet]))
    except FileNotFoundError:
        click.echo(f"❌ Dosya bulunamadı: {filepath}")
    except Exception as e:
        click.echo(f"❌ Hata: {e}")


@main.command()
@click.argument("girdi_csv")
@click.argument("cikti_csv")
@click.option("--model", default="CCR", type=click.Choice(["CCR", "BCC"]), help="DEA modeli")
def dea(girdi_csv: str, cikti_csv: str, model: str):
    """Veri Zarflama Analizi (DEA) ile etkinlik ölç.
    
    Her iki CSV'de ilk sütun işletme (DMU) adı kabul edilir.
    """
    try:
        from agrista.economics import dea_efficiency
        
        inputs = pd.read_csv(girdi_csv, index_col=0)
        outputs = pd.read_csv(cikti_csv, index_col=0)
        _print_dea_result(dea_efficiency(inputs, outputs, model=model), model)
    except FileNotFoundError as e:
        click.echo(f"❌ Dosya bulunamadı: {e}")
    except Exception as e:
        click.echo(f"❌ Hata: {e}")


@main.command()
@click.argument("filepath")
@click.option("--kolonlar", default=None, help="Virgülle ayrılmış sütun adları (varsayılan: tüm kategorikler)")
def frequencies(filepath: str, kolonlar: str):
    """Frekans tabloları üret (Premium Program: Frequencies)."""
    try:
        from agrista.analysis import frequencies as freq_calc
        
        df = _load_file(filepath).dataframe
        cols = [c.strip() for c in kolonlar.split(",")] if kolonlar else None
        _print_frequencies_result(freq_calc(df, cols))
    except Exception as e:
        click.echo(f"❌ Hata: {e}")


@main.command()
@click.argument("filepath")
@click.option("--satir", required=True, help="Satır değişkeni")
@click.option("--sutun", required=True, help="Sütun değişkeni")
def crosstabs(filepath: str, satir: str, sutun: str):
    """Çapraz tablo + ki-kare bağımsızlık testi (Premium Program: Crosstabs)."""
    try:
        from agrista.analysis import crosstabs as crosstab_calc
        
        df = _load_file(filepath).dataframe
        _print_crosstabs_result(crosstab_calc(df, satir, sutun))
    except Exception as e:
        click.echo(f"❌ Hata: {e}")


@main.command()
@click.argument("filepath")
@click.option("--kolon", required=True, help="Test edilecek sayısal sütun")
@click.option("--deger", required=True, type=float, help="Karşılaştırılacak değer")
def onesample(filepath: str, kolon: str, deger: float):
    """Tek örneklem t-testi (Premium Program: One-Sample T Test)."""
    try:
        from agrista.analysis import one_sample_t_test
        
        df = _load_file(filepath).dataframe
        if kolon not in df.columns:
            click.echo(f"❌ '{kolon}' sütunu bulunamadı. Mevcut: {list(df.columns)}")
            return
        _print_one_sample_result(one_sample_t_test(df[kolon], deger))
    except Exception as e:
        click.echo(f"❌ Hata: {e}")


@main.command()
@click.argument("filepath")
@click.option("--once", required=True, help="Önce ölçümü sütunu")
@click.option("--sonra", required=True, help="Sonra ölçümü sütunu")
def paired(filepath: str, once: str, sonra: str):
    """Eşleştirilmiş t-testi (Premium Program: Paired-Samples T Test)."""
    try:
        from agrista.analysis import paired_t_test
        
        df = _load_file(filepath).dataframe
        for col in (once, sonra):
            if col not in df.columns:
                click.echo(f"❌ '{col}' sütunu bulunamadı. Mevcut: {list(df.columns)}")
                return
        _print_paired_result(paired_t_test(df[once], df[sonra]))
    except Exception as e:
        click.echo(f"❌ Hata: {e}")


@main.command()
@click.argument("filepath")
@click.option("--yanit", required=True, help="Bağımlı değişken sütunu (≥2 kategori)")
@click.option("--degiskenler", required=True,
              help="Virgülle ayrılmış sayısal açıklayıcı değişkenler")
def multinom(filepath: str, yanit: str, degiskenler: str):
    """Multinomial lojistik regresyon (Premium Program: Multinomial Logistic Regression)."""
    try:
        from agrista.analysis import multinomial_logistic_regression
        
        df = _load_file(filepath).dataframe
        predictors = [c.strip() for c in degiskenler.split(",") if c.strip()]
        _print_mlogit_result(
            multinomial_logistic_regression(df, yanit, predictors))
    except Exception as e:
        click.echo(f"❌ Hata: {e}")


@main.command()
@click.argument("filepath")
@click.option("--yanit", required=True, help="Bağımlı değişken sütunu (≥3 sıralı kategori)")
@click.option("--degiskenler", required=True,
              help="Virgülle ayrılmış sayısal açıklayıcı değişkenler")
def ordlogit(filepath: str, yanit: str, degiskenler: str):
    """Ordinal lojistik regresyon (Premium Program: Ordinal Regression / PLUM)."""
    try:
        from agrista.analysis import ordinal_logistic_regression
        
        df = _load_file(filepath).dataframe
        predictors = [c.strip() for c in degiskenler.split(",") if c.strip()]
        _print_ologit_result(ordinal_logistic_regression(df, yanit, predictors))
    except Exception as e:
        click.echo(f"❌ Hata: {e}")


@main.command("glm")
@click.argument("filepath")
@click.option("--yanit", required=True, help="Bağımlı değişken sütunu")
@click.option("--faktorler", required=True, help="Virgülle ayrılmış faktörler")
@click.option("--kovaryeteler", default=None, help="Virgülle ayrılmış kovaryeteler")
@click.option("--tip", default="3", type=click.Choice(["1", "2", "3"]), help="SS tipi")
@click.option("--posthoc", default="tukey", type=click.Choice(["tukey", "duncan", "yok"]))
@click.option("--within", default=None, help="Tekrarlı ölçüm sütunları (virgülle)")
@click.option("--denek", default=None, help="Denek sütunu (--within ile)")
def glm(filepath: str, yanit: str, faktorler: str, kovaryeteler: str,
        tip: str, posthoc: str, within: str, denek: str):
    """Genel Doğrusal Model (tek değişkenli / tekrarlı ölçüm)."""
    from agrista.analysis import glm_univariate, glm_repeated_measures
    df = _load_file(filepath).dataframe
    try:
        if within:
            if not denek:
                raise click.UsageError("--within ile --denek zorunludur")
            result = glm_repeated_measures(
                df, response_cols=[c.strip() for c in within.split(",")],
                subject_col=denek)
        else:
            kov = [c.strip() for c in kovaryeteler.split(",")] if kovaryeteler else None
            result = glm_univariate(
                df, response=yanit,
                between_factors=[f.strip() for f in faktorler.split(",")],
                covariates=kov, ss_type=int(tip),
                posthoc=None if posthoc == "yok" else posthoc)
    except ValueError as e:
        raise click.ClickException(str(e))
    _print_glm_result(result)


@main.command("gee")
@click.argument("filepath")
@click.option("--yanit", required=True, help="Bağımlı değişken sütunu")
@click.option("--degiskenler", required=True, help="Virgülle ayrılmış açıklayıcılar")
@click.option("--grup", required=True, help="Küme/grup sütunu")
@click.option("--aile", default="gaussian",
              type=click.Choice(["gaussian", "binomial", "poisson", "gamma"]))
@click.option("--yapi", default="independent",
              type=click.Choice(["independent", "exchangeable", "autoregressive"]))
@click.option("--zaman", default=None, help="Zaman sütunu (autoregressive için)")
def gee(filepath: str, yanit: str, degiskenler: str, grup: str,
        aile: str, yapi: str, zaman: str):
    """Genelleştirilmiş Tahmin Denklemleri (GEE)."""
    from agrista.analysis import gee_model
    df = _load_file(filepath).dataframe
    try:
        result = gee_model(df, response=yanit,
                           covariates=[c.strip() for c in degiskenler.split(",")],
                           group_col=grup, family=aile, cov_struct=yapi,
                           time_col=zaman)
    except ValueError as e:
        raise click.ClickException(str(e))
    _print_gee_result(result)


@main.command("glmm")
@click.argument("filepath")
@click.option("--yanit", required=True, help="Bağımlı değişken sütunu")
@click.option("--sabitler", required=True, help="Virgülle ayrılmış sabit etkiler")
@click.option("--grup", required=True, help="Rastgele etki grup sütunu")
@click.option("--aile", default="gaussian",
              type=click.Choice(["gaussian", "binomial", "poisson"]))
@click.option("--random-slope", default=None, help="Rastgele eğim değişkeni")
def glmm(filepath: str, yanit: str, sabitler: str, grup: str,
         aile: str, random_slope: str):
    """Genelleştirilmiş doğrusal karışık model (GLMM)."""
    df = _load_file(filepath).dataframe
    try:
        result = glmm_fit(df, response=yanit,
                          fixed_effects=[c.strip() for c in sabitler.split(",")],
                          groups_col=grup, family=aile,
                          random_slope=random_slope)
    except ValueError as e:
        raise click.ClickException(str(e))
    _print_glmm_result(result)


@main.command()
@click.argument("filepath")
@click.option("--grup", required=True, help="Grup sütunu")
@click.option("--degiskenler", required=True,
              help="Virgülle ayrılmış sayısal açıklayıcı değişkenler")
def discriminant(filepath: str, grup: str, degiskenler: str):
    """Ayrımsama analizi (Premium Program: Classify → Discriminant)."""
    try:
        from agrista.analysis import discriminant_analysis
        
        df = _load_file(filepath).dataframe
        predictors = [c.strip() for c in degiskenler.split(",") if c.strip()]
        _print_discriminant_result(discriminant_analysis(df, grup, predictors))
    except Exception as e:
        click.echo(f"❌ Hata: {e}")


@main.command()
@click.argument("filepath")
@click.option("--satir", required=True, help="Satır değişkeni")
@click.option("--sutun", required=True, help="Sütun değişkeni")
@click.option("--boyut", default=2, type=int, help="Boyut sayısı")
def correspondence(filepath: str, satir: str, sutun: str, boyut: int):
    """Uyuşum analizi (Premium Program: Dimension Reduction → Correspondence)."""
    try:
        from agrista.analysis import correspondence_analysis
        
        df = _load_file(filepath).dataframe
        _print_correspondence_result(
            correspondence_analysis(df, satir, sutun, n_dims=boyut))
    except Exception as e:
        click.echo(f"❌ Hata: {e}")


# ---------------------------------------------------------------------------
# Wave 4 komutları: Tablolar, Sınıflandırma, MDS, Survival, Direct Marketing
# ---------------------------------------------------------------------------


@main.command()
@click.argument("filepath")
@click.option("--satirlar", required=True, help="Satır değişkenleri (virgülle)")
@click.option("--sutunlar", default=None, help="Sütun değişkenleri (virgülle)")
@click.option("--deger", default=None, help="Özetlenecek sayısal sütun")
@click.option("--istatistikler", default="count,mean",
              help="count,mean,median,std,sum,percent")
def ctable(filepath: str, satirlar: str, sutunlar: str, deger: str,
           istatistikler: str):
    """Özel tablolar (Premium Program: Tables → Custom Tables)."""
    try:
        from agrista.analysis import custom_tables
        
        df = _load_file(filepath).dataframe
        rows = [c.strip() for c in satirlar.split(",") if c.strip()]
        cols = [c.strip() for c in sutunlar.split(",")] if sutunlar else None
        stats_list = [s.strip() for s in istatistikler.split(",") if s.strip()]
        result = custom_tables(df, rows=rows, columns=cols, values=deger,
                               statistics=stats_list)
        click.echo("📋 Özel Tablo (Custom Tables)")
        click.echo("=" * 50)
        for line in result["table"].round(3).to_string().splitlines():
            click.echo(f"   {line}")
        click.echo(f"\n   Vaka sayısı: {result['n_cases']}")
    except Exception as e:
        click.echo(f"❌ Hata: {e}")


@main.command()
@click.argument("filepath")
@click.option("--yanit", required=True, help="Sayısal yanıt sütunu")
@click.option("--gruplar", required=True, help="Grup sütunları (virgülle)")
def means(filepath: str, yanit: str, gruplar: str):
    """Ortalama raporu (Premium Program: Compare Means → Means)."""
    try:
        from agrista.analysis import means_report
        
        df = _load_file(filepath).dataframe
        cols = [c.strip() for c in gruplar.split(",") if c.strip()]
        result = means_report(df, yanit, cols)
        gt = result["grand_total"]
        click.echo("📊 Ortalama Raporu")
        click.echo("=" * 50)
        click.echo(f"   Genel: n={gt['n']}  ortalama={gt['mean']:.3f}  ss={gt['std']:.3f}")
        for col, layer in result["layers"].items():
            click.echo(f"\n  ▸ {col}")
            for name, s in layer.items():
                click.echo(f"   • {name}: n={s['n']}  ortalama={s['mean']:.3f}  ss={s['std']:.3f}")
    except Exception as e:
        click.echo(f"❌ Hata: {e}")


@main.command()
@click.argument("filepath")
@click.option("--pay", "pay_col", required=True, help="Pay sütunu")
@click.option("--payda", "payda_col", required=True, help="Payda sütunu")
@click.option("--grup", default=None, help="İsteğe bağlı grup sütunu (PRD için)")
def ratios(filepath: str, pay_col: str, payda_col: str, grup: str):
    """Oran istatistikleri (Premium Program: Descriptive Statistics → Ratios)."""
    try:
        from agrista.analysis import ratio_statistics
        
        df = _load_file(filepath).dataframe
        result = ratio_statistics(df, pay_col, payda_col, grup)
        click.echo("📊 Oran İstatistikleri")
        click.echo("=" * 50)
        click.echo(f"   Ortalama oran: {result['mean_ratio']:.4f}  "
                   f"Medyan: {result['median_ratio']:.4f}")
        click.echo(f"   COV: {result['cov']:.4f}  AAD: {result['aad']:.4f}")
        if "groups" in result:
            click.echo(f"   PRD: {result['prd']:.4f}")
            for name, g in result["groups"].items():
                click.echo(f"   • {name}: n={g['n']}  oran={g['mean_ratio']:.4f}")
    except Exception as e:
        click.echo(f"❌ Hata: {e}")


@main.command()
@click.argument("filepath")
@click.option("--grup", required=True, help="Grup sütunu")
@click.option("--degiskenler", required=True, help="Sayısal değişkenler (virgülle)")
@click.option("--k", default=3, type=int, help="Komşu sayısı")
def knn(filepath: str, grup: str, degiskenler: str, k: int):
    """En yakın komşu sınıflandırması (Premium Program: Classify → Nearest Neighbor)."""
    try:
        from agrista.analysis import nearest_neighbor_analysis
        
        df = _load_file(filepath).dataframe
        preds = [c.strip() for c in degiskenler.split(",") if c.strip()]
        result = nearest_neighbor_analysis(df, grup, preds, k=k)
        click.echo(f"🗂️ En Yakın Komşu (k={k}, LOO)")
        click.echo("=" * 50)
        for row in result["classification"]:
            click.echo(f"   • {row['group']}: {row['n_correct']}/{row['n']} doğru "
                       f"(%{row['percent_correct']:.1f})")
        click.echo(f"   Genel doğruluk: %{result['overall_accuracy'] * 100:.1f}")
    except Exception as e:
        click.echo(f"❌ Hata: {e}")


@main.command()
@click.argument("filepath")
@click.option("--kolonlar", required=True, help="Sayısal sütunlar (virgülle)")
@click.option("--boyut", default=2, type=int, help="Boyut sayısı")
@click.option("--olcut", default="euclidean",
              type=click.Choice(["euclidean", "manhattan", "cosine"]),
              help="Mesafe ölçütü")
def mds(filepath: str, kolonlar: str, boyut: int, olcut: str):
    """Çok boyutlu ölçekleme (Premium Program: Dimension Reduction → MDS)."""
    try:
        from agrista.analysis import distance_matrix, multidimensional_scaling
        
        df = _load_file(filepath).dataframe
        cols = [c.strip() for c in kolonlar.split(",") if c.strip()]
        dist = distance_matrix(df, cols, measure=olcut, between="cases")["distances"]
        result = multidimensional_scaling(dist, n_dims=boyut)
        click.echo("🧭 Çok Boyutlu Ölçekleme (klasik MDS)")
        click.echo("=" * 50)
        click.echo(f"   R² (uyum): {result['r_squared']:.4f}  "
                   f"stres vekili: {result['stress_proxy']:.4f}")
        for line in result["coordinates"].round(3).to_string().splitlines():
            click.echo(f"   {line}")
    except Exception as e:
        click.echo(f"❌ Hata: {e}")


@main.command()
@click.argument("filepath")
@click.option("--kolonlar", required=True, help="Sayısal sütunlar (virgülle)")
@click.option("--olcut", default="euclidean",
              type=click.Choice(["euclidean", "manhattan", "cosine", "correlation"]),
              help="Mesafe ölçütü")
@click.option("--arasinda", default="cases",
              type=click.Choice(["cases", "variables"]), help="Mesafe türü")
def distances(filepath: str, kolonlar: str, olcut: str, arasinda: str):
    """Mesafe matrisi (Premium Program: Correlate → Distances)."""
    try:
        from agrista.analysis import distance_matrix
        
        df = _load_file(filepath).dataframe
        cols = [c.strip() for c in kolonlar.split(",") if c.strip()]
        result = distance_matrix(df, cols, measure=olcut, between=arasinda)
        click.echo(f"🔗 Mesafe Matrisi ({olcut}, {arasinda})")
        click.echo("=" * 50)
        shown = result["distances"].iloc[:20, :20]
        for line in shown.round(3).to_string().splitlines():
            click.echo(f"   {line}")
        if result["n"] > 20:
            click.echo(f"   … ({result['n']} satırın ilki 20 gösterildi)")
    except Exception as e:
        click.echo(f"❌ Hata: {e}")


@main.command()
@click.argument("filepath")
@click.option("--kolonlar", required=True, help="Dikotomi sütunları (virgülle)")
@click.option("--deger", required=True, help="Sayılan değer (örn: 1)")
def multresp(filepath: str, kolonlar: str, deger: str):
    """Çoklu yanıt frekansları (Premium Program: Multiple Response)."""
    try:
        from agrista.analysis import multiple_response_frequencies
        
        df = _load_file(filepath).dataframe
        cols = [c.strip() for c in kolonlar.split(",") if c.strip()]
        counted = _coerce_number(deger)
        result = multiple_response_frequencies(df, cols, counted)
        click.echo("📊 Çoklu Yanıt Frekansları")
        click.echo("=" * 50)
        for row in result["table"]:
            click.echo(f"   • {row['category']:<16} {row['count']:>5}  "
                       f"yanıt %{row['percent_of_responses']:>5.1f}  "
                       f"vaka %{row['percent_of_cases']:>5.1f}")
        click.echo(f"   Toplam yanıt: {result['total_responses']}  "
                   f"vaka başına ortalama: {result['mean_responses_per_case']:.2f}")
    except Exception as e:
        click.echo(f"❌ Hata: {e}")


@main.command()
@click.argument("filepath")
@click.option("--zaman", required=True, help="Zaman sütunu")
@click.option("--olay", required=True, help="Olay sütunu (1=olay, 0=sansür)")
@click.option("--aralik", default=1.0, type=float, help="Aralık genişliği")
def lifetable(filepath: str, zaman: str, olay: str, aralik: float):
    """Yaşam tabloları (Premium Program: Survival → Life Tables)."""
    try:
        from agrista.survival import life_tables
        
        df = _load_file(filepath).dataframe
        result = life_tables(df[zaman], df[olay], interval_width=aralik)
        click.echo("⏳ Yaşam Tabloları (aktüeryal)")
        click.echo("=" * 50)
        tbl = result["table"][["interval_start", "n_entering", "n_withdrawn",
                               "n_terminated", "proportion_terminating", "survival"]]
        for line in tbl.round(4).to_string(index=False).splitlines():
            click.echo(f"   {line}")
        medyan = result["median_survival"]
        if medyan is not None:
            click.echo(f"\n   Medyan sağkalım: {medyan:.2f}")
        else:
            click.echo("\n   Medyan sağkalım: ulaşılamadı")
    except Exception as e:
        click.echo(f"❌ Hata: {e}")


@main.command()
@click.argument("filepath")
@click.option("--zaman", required=True, help="Zaman sütunu")
@click.option("--olay", required=True, help="Olay sütunu (1=olay, 0=sansür)")
@click.option("--degiskenler", required=True, help="Ortak değişkenler (virgülle)")
def cox(filepath: str, zaman: str, olay: str, degiskenler: str):
    """Cox orantılı tehlike regresyonu (Premium Program: Survival → Cox Regression)."""
    try:
        from agrista.survival import cox_regression
        
        df = _load_file(filepath).dataframe
        preds = [c.strip() for c in degiskenler.split(",") if c.strip()]
        result = cox_regression(df[zaman], df[olay], df[preds])
        click.echo("⏳ Cox Regresyonu (Breslow)")
        click.echo("=" * 50)
        click.echo("   {:<14} {:>9} {:>9} {:>9} {:>10}".format(
            "Değişken", "β", "exp(β)", "z", "p"))
        for i, name in enumerate(preds):
            click.echo("   {:<14} {:>9.4f} {:>9.4f} {:>9.3f} {:>10.4f}".format(
                name[:14], result["coefficients"][i], result["exp_coef"][i],
                result["wald_z"][i], result["p_values"][i]))
        click.echo(f"   LR ki-kare: {result['lr_chi_square']:.4f}  "
                   f"p={result['lr_p_value']:.6f}")
        click.echo(f"   Harrell C: {result['concordance_index']:.3f}  "
                   f"(olay={result['n_events']}, n={result['n']})")
    except Exception as e:
        click.echo(f"❌ Hata: {e}")


@main.command()
@click.argument("filepath")
@click.option("--musteri", required=True, help="Müşteri/işletme sütunu")
@click.option("--tarih", required=True, help="İşlem tarihi sütunu")
@click.option("--tutar", required=True, help="İşlem tutarı sütunu")
@click.option("--dilim", default=5, type=int, help="Kantil sayısı")
def rfm(filepath: str, musteri: str, tarih: str, tutar: str, dilim: int):
    """RFM analizi (Premium Program: Direct Marketing → RFM Analysis)."""
    try:
        from agrista.marketing import rfm_analysis
        
        df = _load_file(filepath).dataframe
        result = rfm_analysis(df, musteri, tarih, tutar, quantiles=dilim)
        click.echo("📣 RFM Analizi")
        click.echo("=" * 50)
        click.echo(f"   Müşteri: {result['n_customers']}  "
                   f"işlem: {result['n_transactions']}  "
                   f"referans: {result['reference_date']}")
        for seg, s in result["segment_summary"].items():
            click.echo(f"   • {seg}: {s['count']} müşteri  "
                       f"ortalama harcama={s['mean_monetary']:.1f}")
    except Exception as e:
        click.echo(f"❌ Hata: {e}")


@main.command()
@click.option("--kontrol-yanit", required=True, type=int)
@click.option("--kontrol-boyut", required=True, type=int)
@click.option("--uygulama-yanit", required=True, type=int)
@click.option("--uygulama-boyut", required=True, type=int)
def mailing(kontrol_yanit: int, kontrol_boyut: int,
            uygulama_yanit: int, uygulama_boyut: int):
    """Kampanya testi (Premium Program: Direct Marketing → Control vs Package)."""
    try:
        from agrista.marketing import mailing_test
        
        result = mailing_test(kontrol_yanit, kontrol_boyut,
                              uygulama_yanit, uygulama_boyut)
        click.echo("📣 Kampanya/Posta Testi")
        click.echo("=" * 50)
        click.echo(f"   Kontrol: %{result['control_rate'] * 100:.2f}  "
                   f"Uygulama: %{result['treatment_rate'] * 100:.2f}")
        click.echo(f"   Fark: {result['difference'] * 100:+.2f} puan  "
                   f"lift: {result['lift'] * 100:+.1f}%")
        click.echo(f"   z={result['z_statistic']:.3f}  p={result['p_value']:.6f}")
        click.echo(f"   {result['recommendation']}")
    except Exception as e:
        click.echo(f"❌ Hata: {e}")


@main.command()
@click.argument("filepath")
@click.option("--yanit", required=True, help="İkili yanıt sütunu")
@click.option("--degiskenler", required=True, help="Kategorik değişkenler (virgülle)")
def prospect(filepath: str, yanit: str, degiskenler: str):
    """Aday profilleri (Premium Program: Direct Marketing → Prospect Profiles)."""
    try:
        from agrista.marketing import prospect_profiles
        
        df = _load_file(filepath).dataframe
        preds = [c.strip() for c in degiskenler.split(",") if c.strip()]
        result = prospect_profiles(df, yanit, preds)
        click.echo("📣 Aday Profilleri")
        click.echo("=" * 50)
        click.echo(f"   Genel yanıt oranı: %{result['overall_response_rate'] * 100:.1f} "
                   f"(pozitif kategori: {result['positive_category']})")
        for col, rows in result["profiles"].items():
            click.echo(f"\n  ▸ {col}")
            for row in rows:
                click.echo(f"   • {row['category']:<14} n={row['n']:>4}  "
                           f"yanıt %{row['response_rate'] * 100:>5.1f}  "
                           f"lift {row['lift']:.2f}")
    except Exception as e:
        click.echo(f"❌ Hata: {e}")


@main.command()
@click.argument("filepath")
@click.option("--kolonlar", default=None, help="Sayısal sütunlar (varsayılan: tümü)")
@click.option("--maks-kume", default=8, type=int, help="Denenecek en çok küme sayısı")
def twostep(filepath: str, kolonlar: str, maks_kume: int):
    """İki aşamalı kümeleme (Premium Program: Classify → TwoStep Cluster)."""
    try:
        from agrista.analysis import twostep_cluster
        
        df = _load_file(filepath).dataframe
        cols = [c.strip() for c in kolonlar.split(",")] if kolonlar else None
        result = twostep_cluster(df, cols, max_clusters=maks_kume)
        click.echo("🗂️ İki Aşamalı Kümeleme (TwoStep)")
        click.echo("=" * 50)
        click.echo(f"   Seçilen küme sayısı: {result['n_clusters']} (BIC)")
        for k_, size in result["cluster_sizes"].items():
            click.echo(f"   • Küme {k_}: {size} vaka")
        for line in result["cluster_centroids"].round(3).to_string().splitlines():
            click.echo(f"   {line}")
    except Exception as e:
        click.echo(f"❌ Hata: {e}")


def _print_svy_result(result: dict, baslik: str):
    """Anket ortalama/toplam/oran tahminini yazdırır."""
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
    """Anket lojistik regresyon sonucunu yazdırır."""
    click.echo(f"\n🧮 {result['model']}")
    click.echo("   Değişken        Katsayı     SE       z       p")
    for ad, c in result["coefficients"].items():
        click.echo(f"   {ad[:14]:<14} {c['coefficient']:9.4f} "
                   f"{c['std_err']:8.4f} {c['z_value']:7.3f} "
                   f"{c['p_value']:8.4f}")
    click.echo(f"   n = {result['n_obs']}, PSU = {result['n_psu']}")


def _build_svy_design(df, agirlik, psu, tabaka, fpc):
    """CLI/menü seçeneklerinden anket tasarım sözlüğü kurar."""
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
    df = _load_file(filepath).dataframe
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
    df = _load_file(filepath).dataframe
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
    df = _load_file(filepath).dataframe
    try:
        design = _build_svy_design(df, agirlik, psu, tabaka, fpc)
        result = survey_logistic(design, response=yanit,
                                 predictors=[c.strip()
                                             for c in degiskenler.split(",")])
    except ValueError as e:
        raise click.ClickException(str(e))
    _print_svylogit_result(result)


# ---------------------------------------------------------------------------
# Etkileşimli ana menü
# ---------------------------------------------------------------------------


_LOGO = (
    " █████╗  ██████╗ ██████╗ ██╗███████╗████████╗ █████╗ \n"
    "██╔══██╗██╔════╝ ██╔══██╗██║██╔════╝╚══██╔══╝██╔══██╗\n"
    "███████║██║  ███╗██████╔╝██║███████╗   ██║   ███████║\n"
    "██╔══██║██║   ██║██╔══██╗██║╚════██║   ██║   ██╔══██║\n"
    "██║  ██║╚██████╔╝██║  ██║██║███████║   ██║   ██║  ██║\n"
    "╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝"
)


def _print_banner():
    """Açılış ekranı: ASCII logo + sürüm satırı."""
    click.echo("")
    for line in _LOGO.splitlines():
        click.secho(line, fg="green", bold=True)
    click.secho("      ── Tarımsal İstatistik Yazılımı · v0.1.0 ──",
                fg="yellow")
    click.echo("")


def _build_menu_structure():
    """Premium Program menü hiyerarşisiyle birebir uyumlu kategori → işlem ağacı."""
    return [
        ("📁 Dosya", [
            ("Veri dosyası bilgisi", _menu_info),
        ]),
        ("📊 Betimsel İstatistikler", [
            ("Betimsel özet tablosu", _menu_describe),
            ("Frekans tabloları", _menu_frequencies),
            ("Çapraz tablolar (ki-kare)", _menu_crosstabs),
            ("Oran istatistikleri", _menu_ratios),
            ("Normallik testi (Shapiro-Wilk)", _menu_normality),
            ("Q-Q grafiği", _menu_qqplot),
            ("P-P grafiği", _menu_ppplot),
        ]),
        ("⚖️ Ortalamaların Karşılaştırılması", [
            ("Tek örneklem t-testi", _menu_one_sample),
            ("Bağımsız iki örneklem t-testi", _menu_ttest),
            ("Eşleştirilmiş iki örneklem t-testi", _menu_paired),
            ("Tek yönlü ANOVA + Tukey HSD", _menu_oneway_anova),
            ("Ortalama raporu (Means)", _menu_means),
            ("Genel Doğrusal Model (Tek Değişkenli)", _menu_glm),
            ("Tekrarlı Ölçümler (GLM)", _menu_glm_rm),
        ]),
        ("🔗 Korelasyon", [
            ("İki değişkenli korelasyon (Pearson/Spearman)", _menu_corr),
            ("Mesafe matrisi", _menu_distances),
        ]),
        ("🔁 Dönüşüm (Transform)", [
            ("Değişken hesapla (Compute)", _menu_compute),
            ("Yeniden kodla (Recode)", _menu_recode),
            ("Kategorile (Binning)", _menu_bin),
            ("Vakaları sırala (Rank Cases)", _menu_rank),
            ("Eksik değerleri tamamla", _menu_impute),
            ("Zaman serisi değişkeni oluştur", _menu_timeseries),
        ]),
        ("🔬 Ölçek (Scale)", [
            ("Güvenirlik analizi (Cronbach alfa)", _menu_cronbach),
        ]),
        ("🧭 Boyut İndirgeme", [
            ("Faktör analizi (varimax)", _menu_factor),
            ("Uyuşum analizi (Correspondence)", _menu_correspondence),
            ("Çok boyutlu ölçekleme (MDS)", _menu_mds),
        ]),
        ("📈 Regresyon", [
            ("Multinomial lojistik regresyon", _menu_multinom),
            ("Ordinal lojistik regresyon (PLUM)", _menu_ordlogit),
            ("GEE (Genelleştirilmiş Tahmin Denklemleri)", _menu_gee),
            ("GLMM (Genelleştirilmiş Karışık Model)", _menu_glmm),
        ]),
        ("🗂️ Sınıflandırma", [
            ("Ayrımsama analizi (Discriminant)", _menu_discriminant),
            ("En yakın komşu (k-NN)", _menu_knn),
            ("İki aşamalı kümeleme (TwoStep)", _menu_twostep),
        ]),
        ("🔮 Kestirim (Forecasting)", [
            ("Holt-Winters mevsimsel kestirim", _menu_holtwinters),
            ("Mevsimsel ayrıştırma", _menu_decomposition),
            ("ARIMA kestirimi", _menu_arima),
        ]),
        ("⏳ Yaşam Analizi (Survival)", [
            ("Kaplan-Meier sağkalım tablosu", _menu_kaplan),
            ("Log-rank testi (iki grup)", _menu_logrank),
            ("Yaşam tabloları (aktüeryal)", _menu_lifetable),
            ("Cox orantılı tehlike regresyonu", _menu_cox),
        ]),
        ("📊 Kalite Kontrol", [
            ("X̄-R kontrol grafikleri", _menu_xbar),
            ("Pareto analizi", _menu_pareto),
        ]),
        ("🥾 Bootstrapping", [
            ("Bootstrap güven aralığı (ortalama)", _menu_bootstrap),
        ]),
        ("🎯 ROC Eğrisi", [
            ("ROC / AUC (Youden eşik)", _menu_roc),
        ]),
        ("🧬 Loglinear", [
            ("Bağımsızlık modeli (olabilirlik oranı)", _menu_loglinear),
        ]),
        ("📋 Tablolar ve Raporlar", [
            ("Özel tablolar (Custom Tables)", _menu_ctable),
            ("Çoklu yanıt frekansları", _menu_multresp),
            ("Vaka özetleri (Case Summaries)", _menu_casesummaries),
        ]),
        ("📣 Doğrudan Pazarlama", [
            ("RFM analizi", _menu_rfm),
            ("Kampanya testi (Control vs Package)", _menu_mailing),
            ("Aday profilleri (Prospect Profiles)", _menu_prospect),
        ]),
        ("🗃️ Veri Yönetimi", [
            ("Vakaları sırala (Sort Cases)", _menu_sortcases),
            ("Toplu özet (Aggregate)", _menu_aggregate),
            ("Vaka ağırlıklandırma (Weight Cases)", _menu_weightcases),
            ("Dosyayı böl (Split File)", _menu_splitfile),
        ]),
        ("🌿 Uzman Branş Modülleri", [
            ("Bitki Koruma — AUDPC (hastalık ilerlemesi)", _menu_audpc),
            ("Tarım Ekonomisi — DEA etkinlik analizi", _menu_dea),
        ]),
    ]


def _print_main_menu(structure):
    click.echo("─" * 54)
    click.secho("  AGRISTA — Ana Menü", fg="cyan", bold=True)
    click.echo("─" * 54)
    for idx, (title, items) in enumerate(structure, start=1):
        click.echo(f"  [{idx}] {title}  ({len(items)} işlem)")
    click.echo("  [0] 🚪 Çıkış")


def _print_submenu(title: str, items):
    click.echo("")
    click.secho(f"  ▸ {title}", fg="cyan", bold=True)
    click.echo("  " + "─" * 44)
    for idx, (label, _) in enumerate(items, start=1):
        click.echo(f"    [{idx}] {label}")
    click.echo("    [b] ← Ana menüye dön")


def _prompt_or_eof(prompt_text: str, default=None, **kwargs):
    """EOF'ta sessiz sonsuz döngüye düşmemek için click.prompt sarmalayıcısı.

    Giriş akışı tükendiğinde None döndürür (etkileşimli oturumda Ctrl-D
    ile aynı etki).
    """
    import io
    import sys
    
    if default is None and not sys.stdin.isatty():
        pos = None
        try:
            pos = sys.stdin.tell()
            peek = sys.stdin.read(1)
            sys.stdin.seek(pos)
        except (OSError, io.UnsupportedOperation, ValueError):
            peek = "x"  # aranabilir akış değilse normal davran
        if peek == "":
            return None
    return click.prompt(prompt_text, default=default, **kwargs)


def _ask_file(prompt_text: str) -> str:
    """Var olan bir dosya yolu girilene kadar tekrar sorar."""
    while True:
        raw = _prompt_or_eof(prompt_text)
        if raw is None:
            raise click.exceptions.Abort()
        path = Path(raw).expanduser()
        if path.exists():
            return str(path)
        click.echo(f"   ❌ Dosya bulunamadı: {raw} — yeniden deneyin.")


def _ask_column(df: pd.DataFrame, prompt_text: str, default: str = None) -> str:
    """Geçerli bir sütun adı girilene kadar tekrar sorar."""
    while True:
        value = _prompt_or_eof(prompt_text, default=default)
        if value is None:
            raise click.exceptions.Abort()
        if value in df.columns:
            return value
        click.echo(f"   ❌ '{value}' sütunu yok. Mevcut sütunlar: {list(df.columns)}")


def _menu_info():
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    sheet = 0
    if path.endswith((".xlsx", ".xls")):
        sheet = click.prompt("Sayfa numarası", default=0, type=int)
    _show_info(path, sheet)


def _menu_describe():
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    data = _load_file(path)
    click.echo(data.describe_numeric().to_string())


def _menu_corr():
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    method = click.prompt("Yöntem",
                          type=click.Choice(["pearson", "spearman"],
                                            case_sensitive=False),
                          default="pearson")
    _show_corr(path, method.lower())


def _menu_ttest():
    path1 = _ask_file("Grup 1 CSV dosyası")
    path2 = _ask_file("Grup 2 CSV dosyası")
    _show_ttest(path1, path2)


def _menu_oneway_anova():
    from agrista.analysis import anova_one_way, posthoc_tukey
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    yanit = _ask_column(df, "Yanıt değişkeni sütunu")
    grup = _ask_column(df, "Grup sütunu")
    
    groups = [g[yanit].dropna().values for _, g in df.groupby(grup)]
    if len(groups) < 2:
        raise ValueError("En az 2 grup gerekli")
    
    result = anova_one_way(*groups)
    click.echo("📊 Tek Yönlü ANOVA")
    click.echo("=" * 50)
    click.echo(f"   F-değeri: {result['f_statistic']:.4f}  "
               f"p-değeri: {result['p_value']:.6f}")
    click.echo(f"   Eta-kare (etki boyutu): {result['eta_squared']:.3f}")
    sig = "✅ Gruplar arası fark anlamlı" if result["significant_at_005"] \
        else "❌ Gruplar arası fark anlamsız"
    click.echo(f"   Sonuç: {sig}")
    
    if result["significant_at_005"]:
        click.echo("")
        _print_tukey_result(posthoc_tukey(df, yanit, grup))


def _menu_glm():
    from agrista.analysis import glm_univariate

    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    yanit = _ask_column(df, "Bağımlı değişken sütunu")
    faktorler = _prompt_or_eof("Faktörler (virgülle ayrılmış)", default="")
    if not faktorler:
        raise click.exceptions.Abort()
    result = glm_univariate(
        df, response=yanit,
        between_factors=[f.strip() for f in faktorler.split(",")])
    _print_glm_result(result)


def _menu_glm_rm():
    from agrista.analysis import glm_repeated_measures

    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    within = _prompt_or_eof("Tekrarlı ölçüm sütunları (virgülle ayrılmış)")
    if not within:
        raise click.exceptions.Abort()
    denek = _ask_column(df, "Denek sütunu")
    result = glm_repeated_measures(
        df, response_cols=[c.strip() for c in within.split(",")],
        subject_col=denek)
    _print_glm_result(result)


def _menu_normality():
    from agrista.analysis import normality_test
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    col = _ask_column(df, "Test edilecek sayısal sütun")
    result = normality_test(df[col])
    verdict = ("✅ normal dağılıma uygun (H0 reddedilemedi)"
               if result["normal_at_alpha"] else "❌ normal dağılımdan sapma var")
    click.echo(f"   Shapiro-Wilk: W={result['statistic']:.4f}, "
               f"p={result['p_value']:.4f} (n={result['n']})")
    click.echo(f"   Sonuç: {verdict}")


def _menu_audpc():
    from agrista.protection import audpc as audpc_calc
    
    path = _ask_file("Hastalık ilerleme CSV dosyası")
    df = _load_file(path).dataframe
    zaman = _ask_column(df, "Zaman sütunu",
                        default="zaman" if "zaman" in df.columns else None)
    siddet = _ask_column(df, "Şiddet sütunu",
                         default="siddet" if "siddet" in df.columns else None)
    _print_audpc_result(audpc_calc(df[zaman], df[siddet]))


def _menu_dea():
    from agrista.economics import dea_efficiency
    
    girdi = _ask_file("Girdi CSV (ilk sütun = işletme adı)")
    cikti = _ask_file("Çıktı CSV (ilk sütun = işletme adı)")
    model = click.prompt("Model", type=click.Choice(["CCR", "BCC"]), default="CCR")
    inputs = pd.read_csv(girdi, index_col=0)
    outputs = pd.read_csv(cikti, index_col=0)
    _print_dea_result(dea_efficiency(inputs, outputs, model=model), model)


def _default_output(path: str, suffix: str = "_donusmus") -> str:
    p = Path(path)
    return str(p.with_name(p.stem + suffix + p.suffix))


def _save_output(df: pd.DataFrame, default_path: str):
    out = click.prompt("Kaydedilecek çıktı dosyası", default=default_path)
    df.to_csv(out, index=False)
    click.echo(f"   ✅ Kaydedildi: {out} ({len(df)} satır, {len(df.columns)} sütun)")


def _coerce_number(text: str):
    """Mümkünse sayıya çevir, değilse metin olarak döndür."""
    try:
        f = float(text)
        return int(f) if f.is_integer() else f
    except ValueError:
        return text


def _parse_recode_mapping(text: str) -> tuple:
    """'eski=yeni;eski=yeni;ELSE=x' biçimini (mapping, varsayılan) çiftine çevirir."""
    mapping, default = {}, np.nan
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Geçersiz eşleme parçası: '{part}' (biçim: 'eski=yeni')")
        old, new = (s.strip() for s in part.split("=", 1))
        if old.upper() in ("ELSE", "OTHER", "DIGER", "DİĞER"):
            default = _coerce_number(new)
        else:
            mapping[_coerce_number(old)] = _coerce_number(new)
    if not mapping:
        raise ValueError("En az bir 'eski=yeni' eşlemesi gerekli")
    return mapping, default


def _menu_frequencies():
    from agrista.analysis import frequencies as freq_calc
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    cat_cols = list(df.select_dtypes(include=["object", "category"]).columns)
    col = _ask_column(df, "Frekansı alınacak sütun",
                      default=cat_cols[0] if cat_cols else None)
    _print_frequencies_result(freq_calc(df, [col]))


def _menu_crosstabs():
    from agrista.analysis import crosstabs as crosstab_calc
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    satir = _ask_column(df, "Satır değişkeni")
    sutun = _ask_column(df, "Sütun değişkeni")
    _print_crosstabs_result(crosstab_calc(df, satir, sutun))


def _menu_qqplot():
    import matplotlib
    matplotlib.use("Agg")
    import os
    from agrista.viz import AgristaPlotter
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    col = _ask_column(df, "Sayısal sütun")
    plotter = AgristaPlotter()
    fig = plotter.qq_plot(df[col], title=f"Q-Q: {col}")
    os.makedirs("agrista_plots", exist_ok=True)
    out = f"agrista_plots/qq_{col}.png"
    plotter.save(out, fig=fig)


def _menu_one_sample():
    from agrista.analysis import one_sample_t_test
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    col = _ask_column(df, "Test edilecek sayısal sütun")
    test_value = click.prompt("Karşılaştırılacak test değeri", type=float)
    _print_one_sample_result(one_sample_t_test(df[col], test_value))


def _menu_paired():
    from agrista.analysis import paired_t_test
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    once = _ask_column(df, "Önce ölçümü sütunu")
    sonra = _ask_column(df, "Sonra ölçümü sütunu")
    _print_paired_result(paired_t_test(df[once], df[sonra]))


def _menu_compute():
    from agrista.transform import compute
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    click.echo(f"   Mevcut sütunlar: {list(df.columns)}")
    expr = click.prompt("İfade (örn: sulama * 0.001 + gubre * 0.005)")
    new_col = click.prompt("Yeni sütun adı")
    result = compute(df, new_col, expr)
    click.echo(f"   '{new_col}' hesaplandı: ort={result[new_col].mean():.3f}")
    _save_output(result, _default_output(path))


def _menu_recode():
    from agrista.transform import recode
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    col = _ask_column(df, "Kodlanacak sütun")
    click.echo("   Biçim: eski=yeni;eski=yeni;ELSE=diger  (örn: 5=1;6=1;7=2;ELSE=3)")
    mapping_text = click.prompt("Eşlemeler")
    mapping, default = _parse_recode_mapping(mapping_text)
    new_col = click.prompt("Yeni sütun adı", default=f"{col}_recoded")
    result = recode(df, col, mapping, new_column=new_col, default=default)
    _save_output(result, _default_output(path))


def _menu_bin():
    from agrista.transform import bin_variable
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    col = _ask_column(df, "Kategorilecek sayısal sütun")
    n_bins = click.prompt("Kategori sayısı", default=4, type=int)
    method = click.prompt("Yöntem",
                          type=click.Choice(["equal_width", "equal_freq"],
                                            case_sensitive=False),
                          default="equal_width").lower()
    result = bin_variable(df, col, bins=n_bins, method=method)
    counts = result[f"{col}_binned"].value_counts()
    for interval, count in counts.items():
        click.echo(f"   {interval}: {count} vaka")
    _save_output(result, _default_output(path))


def _menu_rank():
    from agrista.transform import rank_cases
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    col = _ask_column(df, "Sıralanacak sütun")
    result = rank_cases(df, col)
    _save_output(result, _default_output(path))


def _menu_impute():
    from agrista.transform import replace_missing
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    cols_text = click.prompt("Sütunlar (boş bırakılırsa tüm sayısal sütunlar)",
                             default="")
    cols = [c.strip() for c in cols_text.split(",") if c.strip()] or None
    method = click.prompt("Yöntem",
                          type=click.Choice(["mean", "median", "ffill", "interpolate"],
                                            case_sensitive=False),
                          default="mean").lower()
    result, report = replace_missing(df, cols, method=method)
    completed = {k: v for k, v in report.items() if v > 0}
    if completed:
        for col_name, n in completed.items():
            click.echo(f"   • {col_name}: {n} eksik tamamlandı")
    else:
        click.echo("   Eksik değer bulunamadı.")
    _save_output(result, _default_output(path))


def _menu_cronbach():
    from agrista.sensory import cronbach_alpha
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    click.echo(f"   Mevcut sütunlar: {list(df.columns)}")
    cols_text = click.prompt("Madde sütunları (virgülle ayrılmış)")
    cols = [c.strip() for c in cols_text.split(",") if c.strip()]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Sütunlar bulunamadı: {missing}")
    result = cronbach_alpha(df[cols])
    click.echo("🔬 Güvenirlik Analizi (Cronbach alfa)")
    click.echo("=" * 50)
    click.echo(f"   Cronbach alfa: {result['cronbach_alpha']:.4f} "
               f"({result['interpretation']})")
    click.echo(f"   Madde sayısı: {result['n_items']}  "
               f"Denek sayısı: {result['n_subjects']}")
    if result["alpha_if_item_deleted"]:
        click.echo("   Madde silindiğinde alfa:")
        for item, alpha in result["alpha_if_item_deleted"].items():
            click.echo(f"   • {item}: {alpha:.4f}")


def _menu_holtwinters():
    from agrista.forecasting import holt_winters
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    col = _ask_column(df, "Zaman serisi sütunu")
    period = click.prompt("Mevsim dönemi (ör: aylık veride 12)", type=int)
    horizon = click.prompt("Kestirim ufku", default=3, type=int)
    result = holt_winters(df[col], period=period, horizon=horizon)
    click.echo("🔮 Holt-Winters Additive Kestirim")
    click.echo("=" * 50)
    click.echo(f"   RMSE: {result['rmse']:.3f}  (n={result['n']})")
    for h, fc in enumerate(result["forecasts"], start=1):
        click.echo(f"   t+{h}: {fc:.3f}")


def _menu_decomposition():
    from agrista.forecasting import seasonal_decomposition
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    col = _ask_column(df, "Zaman serisi sütunu")
    period = click.prompt("Mevsim dönemi", type=int)
    result = seasonal_decomposition(df[col], period=period)
    click.echo("🔮 Mevsimsel Ayrıştırma (additive)")
    click.echo("=" * 50)
    click.echo("   Mevsim indisleri: "
               + ", ".join(f"{v:.2f}" for v in result["seasonal_indices"]))
    click.echo(f"   Artık ss: {result['residual_std']:.3f}")


def _menu_arima():
    from agrista.forecasting import arima_forecast
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    col = _ask_column(df, "Zaman serisi sütunu")
    order = click.prompt("ARIMA (p,d,q) — ör: 1,1,1", default="1,1,1")
    p, d, q = (int(x) for x in order.split(","))
    horizon = click.prompt("Kestirim ufku", default=3, type=int)
    result = arima_forecast(df[col], order=(p, d, q), horizon=horizon)
    click.echo(f"🔮 {result['model']} Kestirimi (AIC={result['aic']:.1f})")
    click.echo("=" * 50)
    for i, fc in enumerate(result["forecasts"], start=1):
        click.echo(f"   t+{i}: {fc:.3f}  "
                   f"[%95 GA: {result['ci95_lower'][i-1]:.3f}, "
                   f"{result['ci95_upper'][i-1]:.3f}]")


def _menu_kaplan():
    from agrista.survival import kaplan_meier
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    time_col = _ask_column(df, "Zaman sütunu")
    event_col = _ask_column(df, "Olay sütunu (1=olay, 0=sansür)")
    result = kaplan_meier(df[time_col], df[event_col])
    click.echo("⏳ Kaplan-Meier Sağkalım Tablosu")
    click.echo("=" * 50)
    click.echo("   {:>8} {:>8} {:>8} {:>9} {:>9}".format(
        "Zaman", "Risk", "Olay", "S(t)", "SH"))
    for i, t in enumerate(result["time"][:20]):
        click.echo("   {:>8.1f} {:>8} {:>8} {:>9.4f} {:>9.4f}".format(
            t, result["n_risk"][i], result["n_events"][i],
            result["survival"][i], result["std_error"][i]))
    medyan = result["median_survival"]
    if medyan is not None:
        click.echo(f"   Medyan sağkalım: {medyan:.2f}")
    else:
        click.echo("   Medyan sağkalım: ulaşılamadı")


def _menu_logrank():
    from agrista.survival import log_rank_test
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    time_col = _ask_column(df, "Zaman sütunu")
    event_col = _ask_column(df, "Olay sütunu")
    group_col = _ask_column(df, "Grup sütunu (tam 2 grup)")
    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        raise ValueError(f"Log-rank için tam 2 grup gerekli (bulunan: {len(groups)})")
    g1 = df[df[group_col] == groups[0]]
    g2 = df[df[group_col] == groups[1]]
    result = log_rank_test(g1[time_col], g1[event_col], g2[time_col], g2[event_col])
    click.echo("⏳ Log-Rank Testi")
    click.echo("=" * 50)
    click.echo(f"   {groups[0]} vs {groups[1]}")
    click.echo(f"   Ki-kare: {result['chi_square']:.4f}  "
               f"p-değeri: {result['p_value']:.6f}")
    sig = "✅ Sağkalım eğrileri farklı" if result["significant_at_005"] \
        else "❌ Fark bulunamadı"
    click.echo(f"   Sonuç: {sig}")


def _menu_xbar():
    from agrista.quality import xbar_r_chart
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    col = _ask_column(df, "Ölçüm sütunu")
    size = click.prompt("Alt grup büyüklüğü (2-10)", default=5, type=int)
    result = xbar_r_chart(df[col], subgroup_size=size)
    click.echo("📊 X̄-R Kontrol Grafikleri")
    click.echo("=" * 50)
    xl = result["xbar_limits"]
    rl = result["r_limits"]
    click.echo(f"   X̄ grafiği: LCL={xl['lcl']:.3f}  merkez={xl['center']:.3f}  UCL={xl['ucl']:.3f}")
    click.echo(f"   R grafiği:  LCL={rl['lcl']:.3f}  merkez={rl['center']:.3f}  UCL={rl['ucl']:.3f}")
    durum = "✅ Süreç kontrol altında" if result["in_control"] \
        else f"⚠️ Kontrol dışı: X̄'de {result['xbar_out_of_control']}, R'de {result['r_out_of_control']} alt grup"
    click.echo(f"   Durum: {durum}")


def _menu_pareto():
    from agrista.quality import pareto_analysis
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    col = _ask_column(df, "Kategori sütunu")
    result = pareto_analysis(df[col])
    click.echo("📊 Pareto Analizi")
    click.echo("=" * 50)
    for cat, count, pct, cum in zip(result["categories"], result["counts"],
                                    result["percent"], result["cumulative_percent"]):
        click.echo(f"   {cat:<20} {count:>5}  %{pct:>5.1f}  küm. %{cum:>5.1f}")
    click.echo(f"   Kritik azınlık (vital few): {', '.join(result['vital_few'])}")


def _menu_bootstrap():
    from agrista.analysis import bootstrap_statistic
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    col = _ask_column(df, "Sayısal sütun")
    result = bootstrap_statistic(df[col], seed=42)
    click.echo("🥾 Bootstrap Güven Aralığı (ortalama)")
    click.echo("=" * 50)
    click.echo(f"   Nokta tahmini: {result['point_estimate']:.4f}")
    click.echo(f"   %95 GA: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
    click.echo(f"   Bootstrap ss: {result['bootstrap_std']:.4f}  "
               f"({result['n_bootstrap']} yineleme)")


def _menu_roc():
    from agrista.analysis import roc_curve
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    actual = _ask_column(df, "Gerçek sınıf sütunu (0/1)")
    score = _ask_column(df, "Skor/olasılık sütunu")
    result = roc_curve(df[actual], df[score])
    click.echo("🎯 ROC Eğrisi Analizi")
    click.echo("=" * 50)
    click.echo(f"   AUC: {result['auc']:.4f} ({result['interpretation']})")
    click.echo(f"   Optimum eşik (Youden J={result['youden_j']:.3f}): "
               f"{result['optimal_threshold']:.4f}")
    click.echo(f"   Duyarlılık: {result['sensitivity_at_optimal']:.3f}  "
               f"Özgüllük: {result['specificity_at_optimal']:.3f}")


def _menu_factor():
    from agrista.genetics import factor_analysis
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    click.echo(f"   Mevcut sütunlar: {list(df.columns)}")
    cols_text = click.prompt("Değişken sütunları (virgülle ayrılmış)")
    cols = [c.strip() for c in cols_text.split(",") if c.strip()]
    n_factors = click.prompt("Faktör sayısı", default=2, type=int)
    result = factor_analysis(df, cols, n_factors=n_factors)
    click.echo("🧭 Faktör Analizi (varimax döndürmeli)")
    click.echo("=" * 50)
    click.echo(f"   Toplam açıklanan varyans: %{result['total_explained_pct']:.1f}")
    for line in result["loadings"].round(3).to_string().splitlines():
        click.echo(f"   {line}")


def _menu_loglinear():
    from agrista.analysis import loglinear_analysis
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    satir = _ask_column(df, "Satır değişkeni")
    sutun = _ask_column(df, "Sütun değişkeni")
    result = loglinear_analysis(df, satir, sutun)
    click.echo("🧬 Log-Linear Model (bağımsızlık)")
    click.echo("=" * 50)
    click.echo(f"   Olabilirlik oranı ki-kare: {result['likelihood_ratio_chi2']:.4f} "
               f"(sd={result['degrees_of_freedom']})")
    click.echo(f"   p-değeri: {result['p_value']:.6f}")
    sig = "✅ Bağımsızlık reddedildi" if result["independence_rejected"] \
        else "❌ Bağımsızlık reddedilemedi"
    click.echo(f"   Sonuç: {sig}")


def _ask_predictors(df: pd.DataFrame, prompt_text: str = "Açıklayıcı değişkenler (virgülle ayrılmış)") -> list:
    """Virgülle ayrılmış geçerli değişken adları girilene kadar tekrar sorar."""
    while True:
        click.echo(f"   Mevcut sütunlar: {list(df.columns)}")
        text = _prompt_or_eof(prompt_text)
        if text is None:
            raise click.exceptions.Abort()
        cols = [c.strip() for c in text.split(",") if c.strip()]
        missing = [c for c in cols if c not in df.columns]
        if cols and not missing:
            return cols
        if missing:
            click.echo(f"   ❌ Sütunlar bulunamadı: {missing}")


def _menu_multinom():
    from agrista.analysis import multinomial_logistic_regression
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    yanit = _ask_column(df, "Bağımlı değişken sütunu (≥2 kategori)")
    predictors = _ask_predictors(df)
    _print_mlogit_result(multinomial_logistic_regression(df, yanit, predictors))


def _menu_ordlogit():
    from agrista.analysis import ordinal_logistic_regression
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    yanit = _ask_column(df, "Bağımlı değişken sütunu (≥3 sıralı kategori)")
    predictors = _ask_predictors(df)
    _print_ologit_result(ordinal_logistic_regression(df, yanit, predictors))


def _menu_gee():
    from agrista.analysis import gee_model

    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    yanit = _ask_column(df, "Bağımlı değişken sütunu")
    degiskenler = _prompt_or_eof("Açıklayıcı değişkenler (virgülle)")
    if not degiskenler:
        raise click.exceptions.Abort()
    grup = _ask_column(df, "Küme/grup sütunu")
    result = gee_model(df, response=yanit,
                       covariates=[c.strip() for c in degiskenler.split(",")],
                       group_col=grup)
    _print_gee_result(result)


def _menu_glmm():
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    yanit = _ask_column(df, "Bağımlı değişken")
    sabitler = _prompt_or_eof("Sabit etkiler (virgülle)")
    if not sabitler:
        raise click.exceptions.Abort()
    grup = _ask_column(df, "Rastgele etki grup sütunu")
    aile = _prompt_or_eof("Aile (gaussian/binomial/poisson)",
                          default="gaussian")
    try:
        result = glmm_fit(df, response=yanit,
                          fixed_effects=[c.strip() for c in sabitler.split(",")],
                          groups_col=grup, family=aile)
    except ValueError as e:
        raise click.ClickException(str(e))
    _print_glmm_result(result)


def _menu_discriminant():
    from agrista.analysis import discriminant_analysis
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    grup = _ask_column(df, "Grup sütunu")
    predictors = _ask_predictors(df)
    _print_discriminant_result(discriminant_analysis(df, grup, predictors))


def _menu_correspondence():
    from agrista.analysis import correspondence_analysis
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    satir = _ask_column(df, "Satır değişkeni")
    sutun = _ask_column(df, "Sütun değişkeni")
    n_dims = click.prompt("Boyut sayısı", default=2, type=int)
    _print_correspondence_result(
        correspondence_analysis(df, satir, sutun, n_dims=n_dims))


def _menu_ppplot():
    import matplotlib
    matplotlib.use("Agg")
    import os
    from agrista.viz import AgristaPlotter
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    col = _ask_column(df, "Sayısal sütun")
    plotter = AgristaPlotter()
    fig = plotter.pp_plot(df[col], title=f"P-P: {col}")
    os.makedirs("agrista_plots", exist_ok=True)
    out = f"agrista_plots/pp_{col}.png"
    plotter.save(out, fig=fig)


def _menu_ratios():
    from agrista.analysis import ratio_statistics
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    pay = _ask_column(df, "Pay sütunu")
    payda = _ask_column(df, "Payda sütunu")
    result = ratio_statistics(df, pay, payda)
    click.echo(f"   Ortalama oran: {result['mean_ratio']:.4f}  "
               f"COV: {result['cov']:.4f}  AAD: {result['aad']:.4f}")


def _menu_means():
    from agrista.analysis import means_report
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    yanit = _ask_column(df, "Sayısal yanıt sütunu")
    grup = _ask_column(df, "Grup sütunu")
    result = means_report(df, yanit, [grup])
    gt = result["grand_total"]
    click.echo(f"   Genel: n={gt['n']}  ortalama={gt['mean']:.3f}")
    for name, s in result["layers"][grup].items():
        click.echo(f"   • {name}: n={s['n']}  ortalama={s['mean']:.3f}")


def _menu_distances():
    from agrista.analysis import distance_matrix
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    cols = _ask_predictors(df, "Sayısal sütunlar (virgülle ayrılmış)")
    result = distance_matrix(df, cols, measure="euclidean", between="cases")
    shown = result["distances"].iloc[:10, :10]
    for line in shown.round(3).to_string().splitlines():
        click.echo(f"   {line}")


def _menu_mds():
    from agrista.analysis import distance_matrix, multidimensional_scaling
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    cols = _ask_predictors(df, "Sayısal sütunlar (virgülle ayrılmış)")
    n_dims = click.prompt("Boyut sayısı", default=2, type=int)
    dist = distance_matrix(df, cols, measure="euclidean", between="cases")["distances"]
    result = multidimensional_scaling(dist, n_dims=n_dims)
    click.echo(f"   R² (uyum): {result['r_squared']:.4f}")
    for line in result["coordinates"].head(15).round(3).to_string().splitlines():
        click.echo(f"   {line}")


def _menu_knn():
    from agrista.analysis import nearest_neighbor_analysis
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    grup = _ask_column(df, "Grup sütunu")
    preds = _ask_predictors(df)
    k = click.prompt("Komşu sayısı (k)", default=3, type=int)
    result = nearest_neighbor_analysis(df, grup, preds, k=k)
    for row in result["classification"]:
        click.echo(f"   • {row['group']}: {row['n_correct']}/{row['n']} doğru")
    click.echo(f"   Genel doğruluk: %{result['overall_accuracy'] * 100:.1f}")


def _menu_lifetable():
    from agrista.survival import life_tables
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    zaman = _ask_column(df, "Zaman sütunu")
    olay = _ask_column(df, "Olay sütunu (1=olay, 0=sansür)")
    aralik = click.prompt("Aralık genişliği", default=1.0, type=float)
    result = life_tables(df[zaman], df[olay], interval_width=aralik)
    tbl = result["table"][["interval_start", "n_entering", "n_terminated", "survival"]]
    for line in tbl.head(20).round(4).to_string(index=False).splitlines():
        click.echo(f"   {line}")


def _menu_cox():
    from agrista.survival import cox_regression
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    zaman = _ask_column(df, "Zaman sütunu")
    olay = _ask_column(df, "Olay sütunu")
    preds = _ask_predictors(df, "Ortak değişkenler (virgülle ayrılmış)")
    result = cox_regression(df[zaman], df[olay], df[preds])
    for i, name in enumerate(preds):
        click.echo(f"   • {name}: β={result['coefficients'][i]:.4f}  "
                   f"exp(β)={result['exp_coef'][i]:.4f}  p={result['p_values'][i]:.4f}")
    click.echo(f"   Harrell C: {result['concordance_index']:.3f}")


def _menu_ctable():
    from agrista.analysis import custom_tables
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    rows_text = click.prompt("Satır değişkenleri (virgülle ayrılmış)")
    rows = [c.strip() for c in rows_text.split(",") if c.strip()]
    deger = click.prompt("Özetlenecek sayısal sütun (boş = yalnız sayı)", default="")
    deger = deger or None
    stats_list = [s.strip() for s in click.prompt(
        "İstatistikler", default="count,mean").split(",") if s.strip()]
    result = custom_tables(df, rows=rows, values=deger, statistics=stats_list)
    for line in result["table"].round(3).to_string().splitlines():
        click.echo(f"   {line}")


def _menu_multresp():
    from agrista.analysis import multiple_response_frequencies
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    cols_text = click.prompt("Dikotomi sütunları (virgülle ayrılmış)")
    cols = [c.strip() for c in cols_text.split(",") if c.strip()]
    deger = _coerce_number(click.prompt("Sayılan değer", default="1"))
    result = multiple_response_frequencies(df, cols, deger)
    for row in result["table"]:
        click.echo(f"   • {row['category']}: {row['count']}  "
                   f"vaka %{row['percent_of_cases']:.1f}")


def _menu_casesummaries():
    from agrista.analysis import case_summaries
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    n = click.prompt("Gösterilecek vaka sayısı", default=20, type=int)
    result = case_summaries(df, n_cases=n)
    for line in result["cases"].to_string().splitlines():
        click.echo(f"   {line}")
    click.echo(f"   Toplam vaka: {result['n_total']}")


def _menu_rfm():
    from agrista.marketing import rfm_analysis
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    musteri = _ask_column(df, "Müşteri/işletme sütunu")
    tarih = _ask_column(df, "İşlem tarihi sütunu")
    tutar = _ask_column(df, "İşlem tutarı sütunu")
    result = rfm_analysis(df, musteri, tarih, tutar)
    for seg, s in result["segment_summary"].items():
        click.echo(f"   • {seg}: {s['count']} müşteri")


def _menu_mailing():
    from agrista.marketing import mailing_test
    
    kr = _prompt_or_eof("Kontrol yanıt sayısı", type=int)
    kn = _prompt_or_eof("Kontrol grup büyüklüğü", type=int)
    ur = _prompt_or_eof("Uygulama yanıt sayısı", type=int)
    un = _prompt_or_eof("Uygulama grup büyüklüğü", type=int)
    if None in (kr, kn, ur, un):
        raise click.exceptions.Abort()
    result = mailing_test(kr, kn, ur, un)
    click.echo(f"   Kontrol %{result['control_rate'] * 100:.2f}  "
               f"Uygulama %{result['treatment_rate'] * 100:.2f}  "
               f"p={result['p_value']:.6f}")
    click.echo(f"   {result['recommendation']}")


def _menu_prospect():
    from agrista.marketing import prospect_profiles
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    yanit = _ask_column(df, "İkili yanıt sütunu")
    cols_text = click.prompt("Kategorik değişkenler (virgülle ayrılmış)")
    preds = [c.strip() for c in cols_text.split(",") if c.strip()]
    result = prospect_profiles(df, yanit, preds)
    click.echo(f"   Genel yanıt oranı: %{result['overall_response_rate'] * 100:.1f}")
    for seg in result["top_segments"][:3]:
        click.echo(f"   • {seg['variable']}={seg['category']}: "
                   f"lift {seg['lift']:.2f}")


def _menu_sortcases():
    from agrista.data import sort_cases
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    cols = _ask_predictors(df, "Sıralama sütunları (virgülle ayrılmış)")
    result = sort_cases(df, cols)
    _save_output(result, _default_output(path, suffix="_sirali"))


def _menu_aggregate():
    from agrista.data import aggregate_data
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    grup = _ask_column(df, "Grup sütunu")
    deger = _ask_column(df, "Özetlenecek sayısal sütun")
    islev = click.prompt("İşlev",
                         type=click.Choice(["mean", "median", "sum", "count",
                                            "min", "max"], case_sensitive=False),
                         default="mean").lower()
    result = aggregate_data(df, [grup], {deger: islev})
    for line in result.to_string(index=False).splitlines():
        click.echo(f"   {line}")
    _save_output(result, _default_output(path, suffix="_ozet"))


def _menu_weightcases():
    from agrista.data import weight_cases
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    agirlik = _ask_column(df, "Ağırlık sütunu")
    result = weight_cases(df, agirlik)
    click.echo(f"   Toplam ağırlık: {result['sum_of_weights']:.1f}  "
               f"eşdeğer n: {result['equivalent_n']:.1f}")
    for col, wm in list(result["weighted_means"].items())[:5]:
        click.echo(f"   • {col}: ağırlıklı ortalama={wm:.3f}")


def _menu_splitfile():
    from agrista.data import split_file
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    grup = _ask_column(df, "Grup sütunu")
    parts = split_file(df, grup)
    for name, sub in parts.items():
        click.echo(f"   • {name}: {len(sub)} vaka")


def _menu_twostep():
    from agrista.analysis import twostep_cluster
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    cols = _ask_predictors(df, "Sayısal sütunlar (virgülle ayrılmış)")
    result = twostep_cluster(df, cols)
    click.echo(f"   Seçilen küme sayısı: {result['n_clusters']}")
    for k_, size in result["cluster_sizes"].items():
        click.echo(f"   • Küme {k_}: {size} vaka")


def _menu_timeseries():
    from agrista.transform import create_time_series
    
    path = _ask_file("Veri dosyası yolu (CSV/Excel)")
    df = _load_file(path).dataframe
    col = _ask_column(df, "Zaman serisi sütunu")
    islev = click.prompt("İşlev",
                         type=click.Choice(["lag", "difference",
                                            "seasonal_difference",
                                            "moving_average"],
                                           case_sensitive=False),
                         default="lag").lower()
    periods = click.prompt("Periyot", default=1, type=int)
    result = create_time_series(df, col, function=islev, periods=periods)
    _save_output(result, _default_output(path, suffix="_seri"))


def run_interactive_menu():
    """Etkileşimli arayüz döngüsü (Premium Program tarzı kategori → işlem)."""
    structure = _build_menu_structure()
    _print_banner()
    while True:
        _print_main_menu(structure)
        try:
            choice = click.prompt("\nKategori", default="0").strip()
        except (click.exceptions.Abort, KeyboardInterrupt):
            click.echo("")
            break
        if choice == "0":
            break
        if not choice.isdigit() or not (1 <= int(choice) <= len(structure)):
            click.echo(f"   ⚠️ Geçersiz seçim. 0-{len(structure)} arası bir numara girin.")
            continue
        
        title, items = structure[int(choice) - 1]
        while True:
            _print_submenu(title, items)
            try:
                sub = click.prompt("\nİşlem", default="b").strip().lower()
            except (click.exceptions.Abort, KeyboardInterrupt):
                break
            if sub in ("b", "geri"):
                break
            if not sub.isdigit() or not (1 <= int(sub) <= len(items)):
                click.echo(f"   ⚠️ Geçersiz seçim. 1-{len(items)} veya 'b' girin.")
                continue
            
            label, handler = items[int(sub) - 1]
            click.echo("")
            click.secho(f"  ▶ {label}", fg="cyan")
            try:
                handler()
            except (click.exceptions.Abort, KeyboardInterrupt):
                click.echo("\n⚠️ İşlem iptal edildi.")
            except ValueError as e:
                click.echo(f"   ⚠️ {e}")
            except Exception as e:
                click.echo(f"   ❌ Hata: {e}")
            # Premium Program davranışı: sonuç gösterildi, ana menü açık kalır
            break
    click.echo("\n👋 Görüşürüz!")


if __name__ == "__main__":
    main()
