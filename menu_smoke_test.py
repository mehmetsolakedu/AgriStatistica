"""Agrista etkileşimli menü duman testi — 19 kategorinin tüm işlemleri
CliRunner ile betikli girdi üzerinden uçtan uca çalıştırılır."""

import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
from click.testing import CliRunner

from agrista.cli import main as cli_main

TMP = tempfile.mkdtemp(prefix="agrista_menu_")
OUT = os.path.join(TMP, "out")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(123)

# ---------------------------------------------------------------------------
# Veri setleri
# ---------------------------------------------------------------------------
farm = pd.DataFrame({
    "sulama": rng.uniform(3000, 12000, 60),
    "verim": rng.normal(5, 1, 60),
    "alan": rng.uniform(1, 10, 60),
})
farm.to_csv(f"{TMP}/farm.csv", index=False)

cat = pd.DataFrame({
    "grup": ["A"] * 30 + ["B"] * 30,
    "yanit": (["x"] * 27 + ["y"] * 3) + (["x"] * 5 + ["y"] * 25),
})
cat.to_csv(f"{TMP}/cat.csv", index=False)

grp_verim = np.concatenate([rng.normal(5, 0.8, 20), rng.normal(6.5, 0.8, 20),
                            rng.normal(9, 0.8, 20)])
group = pd.DataFrame({
    "grup": np.repeat(["A", "B", "C"], 20),
    "verim": grp_verim,
    "agirlik": rng.uniform(0.5, 2, 60),
})
group.to_csv(f"{TMP}/group.csv", index=False)

once = rng.normal(10, 1, 25)
pd.DataFrame({"once": once, "sonra": once + 1.5}).to_csv(
    f"{TMP}/paired.csv", index=False)
pd.DataFrame({"v": rng.normal(5, 1, 20)}).to_csv(f"{TMP}/g1.csv", index=False)
pd.DataFrame({"v": rng.normal(6.5, 1, 20)}).to_csv(f"{TMP}/g2.csv", index=False)

f1, f2 = rng.normal(0, 1, 80), rng.normal(0, 1, 80)
items = pd.DataFrame({
    "m1": f1 + rng.normal(0, 0.4, 80),
    "m2": f1 + rng.normal(0, 0.4, 80),
    "m3": f1 + rng.normal(0, 0.4, 80),
})
items.to_csv(f"{TMP}/items.csv", index=False)
factor = pd.DataFrame({
    "v1": f1 + rng.normal(0, 0.3, 80),
    "v2": f1 + rng.normal(0, 0.3, 80),
    "v3": f2 + rng.normal(0, 0.3, 80),
    "v4": f2 + rng.normal(0, 0.3, 80),
})
factor.to_csv(f"{TMP}/factor.csv", index=False)

ca_rows = ["c1"] * 40 + ["c2"] * 10 + ["c1"] * 10 + ["c2"] * 40
ca_cols = ["m1"] * 40 + ["m2"] * 10 + ["m1"] * 10 + ["m2"] * 40
pd.DataFrame({"cesit": ca_rows, "pazar": ca_cols}).to_csv(
    f"{TMP}/ca.csv", index=False)

n = 200
x = rng.normal(0, 1, n)
kalite = np.where(x + rng.normal(0, 0.6, n) < -0.5, "dusuk",
                  np.where(x + rng.normal(0, 0.6, n) > 0.5, "yuksek", "orta"))
mlogit = pd.DataFrame({
    "kalite": kalite,
    "grup": np.where(x > 0, "A", "B"),
    "nem": x + rng.normal(0, 0.4, n),
    "protein": 0.6 * x + rng.normal(0, 0.5, n),
})
mlogit.to_csv(f"{TMP}/mlogit.csv", index=False)

t_idx = np.arange(28)
ts = pd.DataFrame({"deger": 20 + 0.2 * t_idx
                   + np.tile([3.0, -2.0, 1.0, -2.0], 7)
                   + rng.normal(0, 0.3, 28)})
ts.to_csv(f"{TMP}/ts.csv", index=False)

surv = pd.DataFrame({
    "zaman": rng.exponential(8, 120),
    "olay": rng.binomial(1, 0.75, 120),
    "grup": np.repeat(["A", "B"], 60),
    "x": rng.normal(0, 1, 120),
})
surv.to_csv(f"{TMP}/surv.csv", index=False)

qc = pd.DataFrame({
    "olcum": rng.normal(50, 0.5, 50),
    "hata": ["cizik"] * 25 + ["kirik"] * 15 + ["renk"] * 7 + ["diger"] * 3,
})
qc.to_csv(f"{TMP}/qc.csv", index=False)

roc = pd.DataFrame({
    "actual": [0] * 50 + [1] * 50,
    "score": list(rng.uniform(0.1, 0.6, 50)) + list(rng.uniform(0.4, 0.95, 50)),
})
roc.to_csv(f"{TMP}/roc.csv", index=False)

pd.DataFrame({
    "s1": [1, 1, 0, 1, 0, 1],
    "s2": [1, 0, 0, 1, 1, 0],
    "s3": [0, 0, 1, 0, 1, 0],
}).to_csv(f"{TMP}/mr.csv", index=False)

rfm_rows = []
for i in range(25):
    for _ in range(int(rng.integers(1, 5))):
        rfm_rows.append({
            "musteri": f"M{i}",
            "tarih": (pd.Timestamp("2025-12-31")
                      - pd.Timedelta(days=int(rng.integers(1, 350)))
                      ).strftime("%Y-%m-%d"),
            "tutar": float(rng.uniform(20, 600)),
        })
pd.DataFrame(rfm_rows).to_csv(f"{TMP}/rfm.csv", index=False)

kanal = rng.choice(["posta", "sms"], 300)
pd.DataFrame({
    "yanit": rng.binomial(1, np.where(kanal == "sms", 0.6, 0.2)),
    "kanal": kanal,
}).to_csv(f"{TMP}/prospect.csv", index=False)

pd.DataFrame({"zaman": [0, 7, 14, 21], "siddet": [5, 20, 45, 70]}).to_csv(
    f"{TMP}/audpc.csv", index=False)

rm_taban = rng.normal(0, 1, 10)
rm_df = pd.DataFrame({
    "denek": [f"d{i}" for i in range(10)],
    "t1": rm_taban + rng.normal(0, 0.3, 10),
    "t2": rm_taban + 0.5 + rng.normal(0, 0.3, 10),
    "t3": rm_taban + 1.0 + rng.normal(0, 0.3, 10),
})
rm_df.to_csv(f"{TMP}/rm.csv", index=False)

pd.DataFrame({"isletme": ["C1", "C2", "C3", "C4"],
              "arazi": [10, 20, 15, 12], "emek": [5, 8, 6, 4]}).to_csv(
    f"{TMP}/dea_in.csv", index=False)
pd.DataFrame({"isletme": ["C1", "C2", "C3", "C4"],
              "urun": [100, 150, 140, 130]}).to_csv(
    f"{TMP}/dea_out.csv", index=False)

# ---------------------------------------------------------------------------
# Menü akışları: (kategori, işlem, başlık, girdi, beklenen çıktı)
# ---------------------------------------------------------------------------


def out(name):
    return os.path.join(OUT, name)


FLOWS = [
    ("[1] Dosya", "Veri dosyası bilgisi",
     f"1\n1\n{TMP}/farm.csv\n0\n", "Satır sayısı: 60"),

    ("[2] Betimsel", "Betimsel özet tablosu",
     f"2\n1\n{TMP}/farm.csv\n0\n", "verim"),
    ("[2] Betimsel", "Frekans tabloları",
     f"2\n2\n{TMP}/cat.csv\ngrup\n0\n", "Frekans Tabloları"),
    ("[2] Betimsel", "Çapraz tablolar (ki-kare)",
     f"2\n3\n{TMP}/cat.csv\ngrup\nyanit\n0\n", "Ki-kare"),
    ("[2] Betimsel", "Oran istatistikleri",
     f"2\n4\n{TMP}/farm.csv\nverim\nalan\n0\n", "Ortalama oran"),
    ("[2] Betimsel", "Normallik testi",
     f"2\n5\n{TMP}/farm.csv\nverim\n0\n", "Shapiro-Wilk"),
    ("[2] Betimsel", "Q-Q grafiği",
     f"2\n6\n{TMP}/farm.csv\nverim\n0\n", ""),
    ("[2] Betimsel", "P-P grafiği",
     f"2\n7\n{TMP}/farm.csv\nverim\n0\n", ""),

    ("[3] Ortalamalar", "Tek örneklem t-testi",
     f"3\n1\n{TMP}/farm.csv\nverim\n5\n0\n", "Tek Örneklem T-Testi"),
    ("[3] Ortalamalar", "Bağımsız iki örneklem t-testi",
     f"3\n2\n{TMP}/g1.csv\n{TMP}/g2.csv\n0\n", "İki Örneklem T-Testi"),
    ("[3] Ortalamalar", "Eşleştirilmiş t-testi",
     f"3\n3\n{TMP}/paired.csv\nonce\nsonra\n0\n", "Eşleştirilmiş"),
    ("[3] Ortalamalar", "ANOVA + Tukey",
     f"3\n4\n{TMP}/group.csv\nverim\ngrup\n0\n", "Tukey"),
    ("[3] Ortalamalar", "Ortalama raporu",
     f"3\n5\n{TMP}/group.csv\nverim\ngrup\n0\n", "Genel:"),
    ("[3] Ortalamalar", "Genel Doğrusal Model (Tek Değişkenli)",
     f"3\n6\n{TMP}/group.csv\nverim\ngrup\n0\n", "GLM"),
    ("[3] Ortalamalar", "Tekrarlı Ölçümler (GLM)",
     f"3\n7\n{TMP}/rm.csv\nt1,t2,t3\ndenek\n0\n", "Mauchly"),

    ("[4] Korelasyon", "İki değişkenli korelasyon",
     f"4\n1\n{TMP}/farm.csv\npearson\n0\n", "Korelasyon Analizi"),
    ("[4] Korelasyon", "Mesafe matrisi",
     f"4\n2\n{TMP}/farm.csv\nverim,alan\n0\n", "0.0"),

    ("[5] Dönüşüm", "Compute",
     f"5\n1\n{TMP}/farm.csv\nverim * 2\nverim2\n{out('compute.csv')}\n0\n",
     "Kaydedildi"),
    ("[5] Dönüşüm", "Recode",
     f"5\n2\n{TMP}/cat.csv\ngrup\nA=1;ELSE=2\n\n{out('recode.csv')}\n0\n",
     "Kaydedildi"),
    ("[5] Dönüşüm", "Binning",
     f"5\n3\n{TMP}/farm.csv\nverim\n3\nequal_width\n{out('bin.csv')}\n0\n",
     "Kaydedildi"),
    ("[5] Dönüşüm", "Rank Cases",
     f"5\n4\n{TMP}/farm.csv\nverim\n{out('rank.csv')}\n0\n",
     "Kaydedildi"),
    ("[5] Dönüşüm", "Eksik değerleri tamamla",
     f"5\n5\n{TMP}/farm.csv\n\nmean\n{out('impute.csv')}\n0\n",
     "Kaydedildi"),
    ("[5] Dönüşüm", "Zaman serisi değişkeni",
     f"5\n6\n{TMP}/ts.csv\ndeger\nlag\n1\n{out('seri.csv')}\n0\n",
     "Kaydedildi"),

    ("[6] Ölçek", "Cronbach alfa",
     f"6\n1\n{TMP}/items.csv\nm1,m2,m3\n0\n", "Cronbach alfa"),

    ("[7] Boyut İndirgeme", "Faktör analizi",
     f"7\n1\n{TMP}/factor.csv\nv1,v2,v3,v4\n2\n0\n", "Faktör Analizi"),
    ("[7] Boyut İndirgeme", "Uyuşum analizi",
     f"7\n2\n{TMP}/ca.csv\ncesit\npazar\n2\n0\n", "Uyuşum"),
    ("[7] Boyut İndirgeme", "MDS",
     f"7\n3\n{TMP}/farm.csv\nverim,alan\n2\n0\n", "R² (uyum)"),

    ("[8] Regresyon", "Multinomial lojistik",
     f"8\n1\n{TMP}/mlogit.csv\nkalite\nnem,protein\n0\n", "Multinomial"),
    ("[8] Regresyon", "Ordinal lojistik",
     f"8\n2\n{TMP}/mlogit.csv\nkalite\nnem\n0\n", "Ordinal"),

    ("[9] Sınıflandırma", "Ayrımsama analizi",
     f"9\n1\n{TMP}/mlogit.csv\ngrup\nnem,protein\n0\n", "Wilks lambda"),
    ("[9] Sınıflandırma", "k-NN",
     f"9\n2\n{TMP}/mlogit.csv\ngrup\nnem,protein\n3\n0\n", "Genel doğruluk"),
    ("[9] Sınıflandırma", "TwoStep",
     f"9\n3\n{TMP}/farm.csv\nverim,alan\n0\n", "küme sayısı"),

    ("[10] Kestirim", "Holt-Winters",
     f"10\n1\n{TMP}/ts.csv\ndeger\n4\n2\n0\n", "Holt-Winters"),
    ("[10] Kestirim", "Mevsimsel ayrıştırma",
     f"10\n2\n{TMP}/ts.csv\ndeger\n4\n0\n", "Mevsim indisleri"),
    ("[10] Kestirim", "ARIMA",
     f"10\n3\n{TMP}/ts.csv\ndeger\n1,1,1\n2\n0\n", "ARIMA"),

    ("[11] Yaşam Analizi", "Kaplan-Meier",
     f"11\n1\n{TMP}/surv.csv\nzaman\nolay\n0\n", "Kaplan-Meier"),
    ("[11] Yaşam Analizi", "Log-rank",
     f"11\n2\n{TMP}/surv.csv\nzaman\nolay\ngrup\n0\n", "Log-Rank"),
    ("[11] Yaşam Analizi", "Yaşam tabloları",
     f"11\n3\n{TMP}/surv.csv\nzaman\nolay\n2\n0\n", "interval_start"),
    ("[11] Yaşam Analizi", "Cox regresyonu",
     f"11\n4\n{TMP}/surv.csv\nzaman\nolay\nx\n0\n", "Harrell C"),

    ("[12] Kalite Kontrol", "X̄-R grafikleri",
     f"12\n1\n{TMP}/qc.csv\nolcum\n5\n0\n", "Kontrol Grafikleri"),
    ("[12] Kalite Kontrol", "Pareto",
     f"12\n2\n{TMP}/qc.csv\nhata\n0\n", "Pareto Analizi"),

    ("[13] Bootstrapping", "Bootstrap GA",
     f"13\n1\n{TMP}/farm.csv\nverim\n0\n", "Bootstrap Güven Aralığı"),

    ("[14] ROC", "ROC / AUC",
     f"14\n1\n{TMP}/roc.csv\nactual\nscore\n0\n", "ROC Eğrisi Analizi"),

    ("[15] Loglinear", "Bağımsızlık modeli",
     f"15\n1\n{TMP}/ca.csv\ncesit\npazar\n0\n", "Log-Linear"),

    ("[16] Tablolar", "Özel tablolar",
     f"16\n1\n{TMP}/group.csv\ngrup\nverim\ncount,mean\n0\n", "count"),
    ("[16] Tablolar", "Çoklu yanıt frekansları",
     f"16\n2\n{TMP}/mr.csv\ns1,s2,s3\n1\n0\n", "vaka %"),
    ("[16] Tablolar", "Vaka özetleri",
     f"16\n3\n{TMP}/farm.csv\n5\n0\n", "Toplam vaka"),

    ("[17] Pazarlama", "RFM",
     f"17\n1\n{TMP}/rfm.csv\nmusteri\ntarih\ntutar\n0\n", "müşteri"),
    ("[17] Pazarlama", "Kampanya testi",
     "17\n2\n20\n1000\n40\n1000\n0\n", "Kontrol %"),
    ("[17] Pazarlama", "Aday profilleri",
     f"17\n3\n{TMP}/prospect.csv\nyanit\nkanal\n0\n", "Genel yanıt oranı"),

    ("[18] Veri Yönetimi", "Sort Cases",
     f"18\n1\n{TMP}/farm.csv\nverim\n{out('sort.csv')}\n0\n", "Kaydedildi"),
    ("[18] Veri Yönetimi", "Aggregate",
     f"18\n2\n{TMP}/group.csv\ngrup\nverim\nmean\n{out('agg.csv')}\n0\n",
     "Kaydedildi"),
    ("[18] Veri Yönetimi", "Weight Cases",
     f"18\n3\n{TMP}/group.csv\nagirlik\n0\n", "Toplam ağırlık"),
    ("[18] Veri Yönetimi", "Split File",
     f"18\n4\n{TMP}/group.csv\ngrup\n0\n", "vaka"),

    ("[19] Uzman Branş", "AUDPC",
     f"19\n1\n{TMP}/audpc.csv\nzaman\nsiddet\n0\n", "AUDPC"),
    ("[19] Uzman Branş", "DEA",
     f"19\n2\n{TMP}/dea_in.csv\n{TMP}/dea_out.csv\nCCR\n0\n", "DEA Etkinlik"),
]

# ---------------------------------------------------------------------------
# Çalıştırma
# ---------------------------------------------------------------------------


def run_flows() -> list:
    """Tüm menü akışlarını çalıştırır; başarısız olan (menü, işlem, çıktı)
    üçlülerini döndürür. pytest ve betik giriş noktası tarafından paylaşılır."""
    runner = CliRunner()
    failures = []
    current_menu = None
    menu_status = {}

    for menu, item, inputs, expect in FLOWS:
        result = runner.invoke(cli_main, [], input=inputs)
        ok = result.exit_code == 0
        output = result.output or ""
        if ok and expect:
            ok = expect in output
        if not ok and result.exception is not None:
            import traceback
            output += "\n" + "".join(
                traceback.format_exception(type(result.exception),
                                           result.exception,
                                           result.exception.__traceback__))
        mark = "PASS" if ok else "FAIL"
        if menu != current_menu:
            current_menu = menu
            menu_status[menu] = True
            print(f"\n=== {menu} ===")
        print(f"  [{mark}] {item}")
        if not ok:
            menu_status[menu] = False
            failures.append((menu, item, output[-2000:]))

    print("\n" + "=" * 60)
    print("ÖZET")
    print("=" * 60)
    for menu, ok in menu_status.items():
        print(f"  {'OK ' if ok else 'HATA'} {menu}")
    print(f"\nToplam: {len(FLOWS)} akış, {len(failures)} hata")
    for menu, item, tail_out in failures:
        print(f"\n--- HATA: {menu} / {item} ---")
        print(tail_out[-1200:])
    return failures


if __name__ == "__main__":
    failures = run_flows()
    shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(1 if failures else 0)
