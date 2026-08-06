# Agrista — Tarımsal İstatistik Yazılımı

Agrista, tarım verilerini toplamak, analiz etmek ve görselleştirmek için tasarlanmış kapsamlı bir istatistik yazılımıdır.

## Özellikler

- 📊 **Veri Yönetimi**: Çiftlik verileri, toprak analizi, hava durumu, ürün verimleri; sıralama, birleştirme, ağırlıklandırma, yeniden yapılandırma, karşılaştırma
- 📈 **İstatistiksel Analiz**: Betimsel istatistik, regresyon (OLS, WLS, 2SLS, multinomial/ordinal lojistik), GLM (tek değişkenli, tekrarlı ölçümler), GEE, GLMM, ANOVA, Tukey/Duncan, parametrik olmayan testler, özel tablolar, çoklu yanıt frekansları
- 🧮 **Karmaşık Anket Tahminleri**: Taylor doğrusallaştırması ile ağırlıklı ortalama/toplam/oran, tasarım etkisi (DEFF), anket lojistik regresyonu (PSU, tabaka, ağırlık, FPC desteği)
- 🗂️ **Sınıflandırma ve Boyut İndirgeme**: Ayrımsama analizi, TwoStep kümeleme, k-NN, MDS, faktör analizi, uyuşum analizi
- ⏳ **Yaşam Analizi**: Kaplan-Meier, log-rank, yaşam tabloları, Cox regresyonu
- 🔮 **Kestirim**: Hareketli ortalama, üstel düzleştirme, Holt-Winters, mevsimsel ayrıştırma, ARIMA
- 📣 **Doğrudan Pazarlama**: RFM segmentasyonu, kampanya testi (control vs package), prospect profilleri
- 🌾 **Tarımsal Modeller**: Verim tahmini, büyüme modelleri (lojistik, Gompertz, monomoleküler, Wood), risk analizi
- 📉 **Görselleştirme**: 16 grafik tipi (violin, ridge, raincloud, pair grid, orman, Bland-Altman, ROC, sağkalım, kontrol, artık, hexbin, yığılmış alan, büyüme eğrisi, eğim ve temel grafikler), 4 tema (agrista/yayın/minimal/karanlık), PNG/SVG/HTML dışa aktarım, plotly etkileşimli dashboard, auto-EDA (değişken türüne göre akıllı grafik önerisi + HTML keşif raporu)
- 🔬 **Deneysel Tasarım**: Tarla denemeleri, RCBD/Latin kare/faktöriyel tasarımlar ve analizleri

### Alt Branş Modülleri

| Modül | Branş | İçerik |
|---|---|---|
| `agrista.protection` | Bitki Koruma | Probit doz-yanıt (LC50), log-logistik (GR50), Abbott etkinliği, AUDPC, hastalık ilerleme eğrileri |
| `agrista.genetics` | Tarla Bitkileri / Islah | Yol analizi, PCA, kümeleme, Mahalanobis D², kalıtım derecesi, AMMI, GGE biplot, Finlay-Wilkinson stabilite |
| `agrista.engineering` | Tarım Makineleri | RSM (CCD, Box-Behnken), optimum bulma, Taguchi L9, S/N oranı |
| `agrista.animal` | Zootekni | Karışık modeller (LMM), Wood laktasyon eğrisi, laktasyon özeti |
| `agrista.sensory` | Bahçe Bitkileri | Kendall W uyum, hedonik özet + Friedman, panel ANOVA |
| `agrista.spatial` | Toprak Bilimi | Yarıvaryogram, mekânsal bağımlılık, IDW enterpolasyon |
| `agrista.economics` | Tarım Ekonomisi | DEA (CCR/BCC), logit/probit benimseme modelleri, kısmi bütçe |

## Kurulum

```bash
pip install -e .
```

## Masaüstü Uygulaması

Agrista, PySide6 tabanlı masaüstü uygulaması olarak da kullanılabilir:

```bash
pip install "agrista[gui]"
agrista-gui
```

Uygulama: 21 kategorili menü, veri tablosu görünümü, 16 bağlı analiz
(otomatik formlar), 12 grafik tipi (gömülü canvas, tema desteği),
uygulama içi güncelleme denetimi. macOS/Windows kurulum paketleri
GitHub Releases'ta yayınlanır (`Agrista-<sürüm>-macOS.dmg`,
`Agrista-<sürüm>-Setup.exe`).

## Kullanım

```python
import agrista as ag

# Veri yükleme
data = ag.load_csv("veriler.csv")

# Betimsel istatistik
print(ag.descriptive_stats(data.dataframe))

# Çoklu regresyon analizi
model = ag.multiple_regression(data.dataframe, target="verim", predictors=["sulama", "gubre"])
print(model["r_squared"], model["coefficients"])

# Branş modülleri
from agrista.protection import audpc, probit_dose_response
from agrista.genetics import ammi_analysis, path_analysis
from agrista.engineering import rsm_ccd, rsm_fit, find_optimum
from agrista.animal import mixed_model, fit_wood
```

Komut satırından:

```bash
agrista            # Etkileşimli ana menü (Premium Program tarzı kategoriler)
agrista menu       # Aynı menü açıkça başlatılır

# Tek atımlık alt komutlar (betik/otomasyon için)
agrista info veriler.csv
agrista describe veriler.csv
agrista frequencies veriler.csv --kolonlar urun,bolge
agrista crosstabs veriler.csv --satir grup --sutun yanit
agrista onesample veriler.csv --kolon verim --deger 5.0
agrista ttest grup1.csv grup2.csv
agrista paired veriler.csv --once olcum1 --sonra olcum2
agrista corr veriler.csv --method spearman
agrista tukey veriler.csv --yanit verim --grup uygulama
agrista audpc hastalik.csv
agrista dea girdi.csv cikti.csv --model BCC
agrista glm veriler.csv --yanit verim --faktorler grup
agrista gee panel.csv --yanit y --degiskenler x --grup isletme
agrista glmm veriler.csv --yanit verim --sabitler gubre --grup parsel
agrista svymean anket.csv --degisken gelir --agirlik w --psu psu
agrista svyratio anket.csv --pay harcama --payda gelir --psu psu
agrista svylogit anket.csv --yanit yanit --degiskenler gelir,yas --psu psu --agirlik w
agrista plot veri.csv --tip scatter --x sulama --y verim --cikti s.png
agrista dashboard veri.csv
agrista autoeda veri.csv
```

Ana menü kategorileri (Premium Program menü hiyerarşisiyle birebir, 21 kategori):
**[1] Dosya**, **[2] Betimsel İstatistikler** (özet, frekans, çapraz tablo,
oran istatistikleri, normallik, Q-Q, P-P), **[3] Ortalamaların Karşılaştırılması**
(tek örneklem, bağımsız, eşleştirilmiş t-testleri; ANOVA + Tukey; ortalama raporu),
**[4] Korelasyon** (korelasyon, mesafe matrisi), **[5] Dönüşüm/Transform**
(compute, recode, binning, rank, eksik tamamlama, zaman serisi değişkeni),
**[6] Ölçek/Scale** (Cronbach alfa), **[7] Boyut İndirgeme** (faktör analizi +
varimax, uyuşum analizi, MDS), **[8] Regresyon** (multinomial lojistik, ordinal
lojistik/PLUM), **[9] Sınıflandırma** (ayrımsama, TwoStep kümeleme, k-NN),
**[10] Kestirim/Forecasting** (Holt-Winters, mevsimsel ayrıştırma, ARIMA),
**[11] Yaşam Analizi/Survival** (Kaplan-Meier, log-rank, yaşam tabloları, Cox),
**[12] Kalite Kontrol** (X̄-R, Pareto), **[13] Bootstrapping**, **[14] ROC Eğrisi**,
**[15] Loglinear**, **[16] Tablolar ve Raporlar** (özel tablolar, vaka özetleri,
çoklu yanıt), **[17] Doğrudan Pazarlama** (RFM, kampanya testi, prospect profilleri),
**[18] Veri Yönetimi** (sıralama, toplulaştırma, ağırlıklandırma, birleştirme,
bölme, yineleme, devrik, yeniden yapılandırma, karşılaştırma, ölçüm düzeyi),
**[19] Uzman Branş Modülleri** (Bitki Koruma, Tarım Ekonomisi vb.),
**[20] Karmaşık Örneklem (Complex Samples)** (anket ortalaması/toplamı,
anket oranı — Taylor doğrusallaştırması; anket lojistik regresyonu),
**[21] Grafikler** (hızlı grafik, dağılım/tanı/model grafikleri,
etkileşimli dashboard, otomatik keşif Auto-EDA).

## Lisans

MIT

## Araştırma Logları

- [Alt Branşlara Göre İstatistiksel Yöntemler Literatür Araştırması](docs/01_ALT_BRANS_ISTATISTIK_LITERATUR_LOG.md) — Bitki koruma, tarla bitkileri, bahçe bitkileri, zootekni, tarım makineleri, toprak bilimi ve tarım ekonomisinde en çok kullanılan analizler + uygulama yol haritası.
- [Premium Program Menü Yapısı Envanteri](docs/02_PREMIUM_PROGRAM_MENU_YAPISI_LOG.md) — Premium Program'ın 11 ana menüsü ve Analyze'ın 22 alt menüsünün tam dökümü + Agrista kapsam eşlemesi ve boşluk analizi.
