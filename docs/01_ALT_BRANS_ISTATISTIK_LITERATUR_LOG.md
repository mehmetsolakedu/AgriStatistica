# Agrista — Alt Branşlara Göre İstatistiksel Yöntemler Literatür Araştırması

> **Log türü:** Literatür araştırması → uygulama yol haritası
> **Tarih:** 2026-08-05
> **Amaç:** Tarım bilimlerinin alt branşlarında en sık kullanılan istatistiksel
> analizleri tespit etmek ve Agrista'ya kazandırılacak modülleri önceliklendirmek.

---

## Özet Tablo

| Branş | En kritik yöntemler | Agrista'da var mı? |
|---|---|---|
| Bitki Koruma | Probit/Logit doz-yanıt (LC50), Abbott düzeltmesi, AUDPC | ❌ Yok |
| Tarla Bitkileri | AMMI, GGE biplot, stabilite analizi, yol (path) analizi | ⚠️ Kısmen (RCBD, ANOVA) |
| Bahçe Bitkileri | Duyusal analiz, hasat sonrası kalite indeksleri, tekrarlı ölçüm | ❌ Yok |
| Zootekni | Karışık modeller, BLUP, kalıtım derecesi, süt eğrileri | ❌ Yok |
| Tarım Makineleri | RSM (CCD, Box-Behnken), Taguchi, faktöriyel DOE | ⚠️ Kısmen (faktöriyel tasarım) |
| Toprak Bilimi | Jeoistatistik (varyogram, kriging), mekânsal analiz | ❌ Yok |
| Tarım Ekonomisi | DEA, Stokastik Sınır (SFA), logit/probit benimseme | ❌ Yok |

---

## 1. Bitki Koruma (Entomoloji, Fitopatoloji, Herboloji)

### 1.1 Doz-yanıt / Biyodeney analizi
- **Probit ve logit analizi:** LC50/LD50/ED50 tahmini için standart yöntem.
  İnsektisit/fungisit biyodeneylerinde doz-mortalite ilişkisi probit dönüşümüyle
  doğrusallaştırılır; güven aralıkları ve göreli etkinlik (relative potency)
  karşılaştırmaları yapılır.
  - Kaynak: FAO Chemical Control Methods (FAO, x5048E): *"Data are analysed
    using probit analysis and compounds are compared with standards using
    relative potency analyses."* — https://www.fao.org/4/x5048e/x5048E0m.htm
- **Uygulama:** `statsmodels` GLM (binom aile, probit/logit link) ile doğrudan
  gerçeklenebilir. Çıktı: LC50 ± %95 GA, eğim, ki-kare uyum testi.

### 1.2 Abbott düzeltmesi ve etkinlik hesabı
- **Abbott formülü:** `E = (C - T) / C × 100` — kontrol mortalitesine göre
  düzeltilmiş etkinlik. Biyodeney değerlendirmesinin endüstri standardı.
  - Piepho (2024), *Journal of Plant Diseases and Protection*: Abbott
    denklemini yeniden ele alıp **genelleştirilmiş doğrusal karışık modeller
    (GLMM)** üzerinden alternatif analiz yolu öneriyor — oran verisinde ham
    yüzde yerine GLM/GLMM'nin üstün olduğunu vurguluyor.
    https://link.springer.com/article/10.1007/s41348-024-00968-0
  - Rosenheim & Hoy (1989), *J. Econ. Entomol.*: Abbott düzeltmesi için
    güven aralıkları. https://rosenheim.faculty.ucdavis.edu/
- **Uygulama:** `efficacy_abbott(control, treated)` + binom GLM seçeneği.

### 1.3 Epidemiyoloji — Hastalık ilerleme eğrileri
- **AUDPC** (Area Under the Disease Progress Curve): Trapez yöntemiyle
  hesaplanan, hastalık şiddetinin zaman içindeki toplam yükünü özetleyen
  standart epidemiyolojik indeks. Tedavi karşılaştırmalarında birincil yanıt
  değişkeni.
  - Kaynaklar: Chandran (2025) *"Modeling Plant Disease Epidemics"* (JOSTA);
    R paketi PDIndex (CRAN) — trapez AUDPC gerçeklemesi.
    https://cran.r-project.org/web/packages/PDIndex/PDIndex.pdf
- **Eğri uydurma:** Hastalık ilerleme eğrilerine lojistik/Gompertz/monomoleküler
  modeller uydurulur; enfeksiyon oranı (r) karşılaştırılır. Agrista'daki
  `GrowthModel` doğrudan bu amaca da hizmet eder (logistik zaten var;
  monomoleküler eklenebilir).
- **Uygulama:** `AUDPC(zaman, şiddet)` + `disease_progress_fit()` → r, K, model
  karşılaştırma (AIC).

### 1.4 Yabancı ot bilimi
- Doz-yanıt çalışmalarında **log-logistik** model (3/4 parametreli) ve
  GR50/ED90 tahminleri standarttır. R `drc` paketi fiili standart.
  Agrista'da `dose_response()` fonksiyonu log-logistik aileyi kapsamalı.

---

## 2. Tarla Bitkileri (Agronomi + Islah/Biyometri)

### 2.1 Genotip × Çevre etkileşimi ve stabilite
- **AMMI** (Additive Main effects and Multiplicative Interaction): MET (çok
  lokasyonlu deneme) verilerinde G×E'yi ANOVA + PCA ile ayrıştıran baskın
  yöntem. AMMI biplot ile adaptasyon/stabilite görselleştirilir.
  - Mullualem (2024), *Heliyon*: 12 genotip arasında AMMI biplot ile adaptif
    genotip seçimi. https://www.sciencedirect.com/science/article/pii/S2405844024089497
  - Derbew (2024), *Agronomy Journal (agg2.20565)*: Arpada AMMI + GGE biplot.
  - Masoodi (2025), *Scientific Reports*: Stabilite için AMMI + GGE biplot.
- **GGE biplot** ("Genotype + G×E"): Hangi genotipin hangi çevrede üstün
  olduğunu gösteren "which-won-where" analizi; ıslah literatüründe AMMI ile
  birlikte en sık kullanılan görsel/istatistiksel araç.
- **Finlay-Wilkinson regresyonu** ve **Eberhart-Russell** stabilite
  parametreleri (bi, S²d): Çevresel indekse karşı genotip regresyonu.
- **Uygulama:** `ammi_analysis(df, genotip, cevre, ozellik)` → IPCA skorları,
  biplot koordinatları; `gge_biplot()`; `stability_indices()` (bi, S²d,
  Shukla varyansı). Bu, Tarla Bitkileri bölümünün en ayırt edici ihtiyacı.

### 2.2 Çok değişkenli ıslah analizleri (neredeyse her makalede var)
- **Korelasyon + Yol (Path) analizi:** Özelliklerin verime doğrudan/dolaylı
  etkilerini ayırır; ıslah makalelerinin imza yöntemi.
  - Mubai (2020), *Cogent Food & Agriculture*: Fenotipik korelasyon, yol
    katsayısı ve çok değişkenli analiz.
    https://www.tandfonline.com/doi/full/10.1080/23311932.2020.1823591
- **PCA (Temel Bileşenler):** Genotip gruplandırma ve özellik azaltma.
- **Kümeleme** (Ward/Euclidean, UPGMA) + **Mahalanobis D²:** Ebeveyn seçimi
  için genetik uzaklık. Roka (2024) derlemesi bu dörtlüyü (PCA, küme, D²,
  path) ıslah çalışmalarının çekirdeği olarak listeler.
  https://aesacademy.org/journals/index.php/aaes/article/view/09-03-029
- **Uygulama:** `path_analysis(df, hedef, açıklayıcılar)` (korelasyon
  matrisinin tersi ile standartlaştırılmış regresyon — scipy/numpy ile kolay),
  `pca_analysis()`, `cluster_genotypes()` (scipy linkage), `mahalanobis_d2()`.

### 2.3 Genetik parametreler
- **Kalıtım derecesi (heritability)** ve **genetik ilerleme (genetic advance)**:
  Varyans bileşenlerinden tahmin; Schmidt (2019, *Heredity*) giriş-farkı
  temelinde kalıtım derecesi tartışması.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6707473/
- **Uygulama:** `heritability(Vg, Ve, n_blok)` — varyans bileşeni girişine dayalı
  basit ve ANOVA tabanlı tahmin.

---

## 3. Bahçe Bitkileri (Meyve, Sebze, Süs Bitkileri, Bağ)

### 3.1 Duyusal (sensory) analiz
- Islah ve kalite çalışmalarının ayırt edici yöntemi: **hedonik ölçek**
  (9 noktalı), tercih/ikili karşılaştırma testleri, panel değerlendirmesi.
  - Moreira (2025), *HortScience*: Üzüm ıslahında duyusal değerlendirmenin
    eleme kararlarına entegrasyonu.
    http://journals.ashs.org/view/journals/hortsci/60/6/article-p1003.xml
  - "Integration of Sensory Analysis into Plant Breeding: A Review"
    (ResearchGate, 2019).
- **İstatistiksel arka plan:** Panelist × örnek ANOVA'sı, Friedman (parametrik
  olmayan), Kruskal-Wallis, çoklu karşılaştırma; panelistler arası uyum
  (Kendall W uyum katsayısı).
- **Uygulama:** `sensory_analysis(df, panelist, örnek, puan)` → Friedman +
  Kendall W + post-hoc; `hedonic_summary()`.

### 3.2 Hasat sonrası ve kalite
- Depolama denemeleri **tekrarlı ölçüm** yapısındadır (aynı örnekler zaman
  içinde ölçülür): tekrarlı ölçüm ANOVA / karışık model ihtiyacı.
- Kalite indeksi birleştirme (Brix/asitlik oranı, renk L*a*b*, sertlik).
  - Gunness (2009), *Postharvest Biology and Technology*: Çilekte meyveden
    meyveye değişimin duyusal + enstrümental ölçümü.
- **Uygulama:** `quality_index()`, tekrarlı ölçüm ANOVA (`repeated_measures_anova`).

---

## 4. Zootekni (Hayvan Yetiştirme, Besleme, Biyometri)

### 4.1 Karışık modeller ve tekrarlı ölçümler
- Zootekni denemelerinin tartışmasız standardi: **karışık modeller (LMM)** ve
  tekrarlı ölçüm analizi. Hayvan = tekrarlı ölçüm birimi; blok/ahır/dönem
  rastgele etki.
  - Goonewardene (2004), *J. Anim. Sci.*: Tekrarlı ölçümlü hayvan
    denemelerinde karışık modeller — 600+ atıf.
    https://www.researchgate.net/publication/236847290
- **BLUP / damızlık değeri:** Henderson karışık model denklemleri; seleksiyon
  kararlarının temeli. VSNI "BLUEs, BLUPs and Breeding Values" kaynağı.
  https://vsni.co.uk/blups_blues_breeding_values/
- **Uygulama:** `statsmodels.MixedLM` sarmalayıcısı:
  `mixed_model(df, yanıt, sabit_efkiler, rastgele_efki, gruplar)`.
  Tam BLUP için ileri aşamada `pyreffects`/özel MME gerçeklemesi gerekebilir;
  ilk sürümde MixedLM üzerinden EBV yaklaşımı yeterli.

### 4.2 Besleme denemeleri tasarımları
- **Latin kare ve crossover (çapraz) tasarımlar:** Sindirilebilirlik ve rasyon
  denemelerinde hayvan sayısını minimize etmek için standart.
  - Kim (2009), *"Balanced Latin Square design"* (134 atıf).
  - Zanton (2019), *J. Dairy Sci.*: Tasarım seçiminin sonuçlara etkisi.
  - CIPAV kaynağı: Besleme denemelerinde istatistiksel tasarım ve yorumlama.
- Agrista'da Latin kare üretimi VAR; **Latin kare ANOVA analizi** eksik.
- **Uygulama:** `latin_square_anova(df, yanıt, dönem, hayvan, uygulama)` —
  üç faktörlü varyans ayrıştırması.

### 4.3 Verim eğrileri
- **Laktasyon eğrisi modelleri:** Wood (gamma), Cobby-Le Du, Wilmink —
  doğrusal olmayan regresyonla uyum; pik verim, kalıcılık parametreleri.
- **Uygulama:** `GrowthModel`'in `fit_*` deseninin genişletilmesi:
  `wood_model(t, a, b, c) = a·t^b·exp(-c·t)` + `fit_wood()`.

---

## 5. Tarım Makineleri (Makine ve Teknoloji Mühendisliği)

### 5.1 Yanıt yüzey yöntemi (RSM)
- **Central Composite Design (CCD)** ve **Box-Behnken Design (BBD):**
  İşleme parametreleri (hız, ilerleme, derinlik), ekim makinesi ayarları,
  kurutma koşulları gibi çok faktörlü optimizasyonların standardı.
  2. derece polinom yüzey + ANOVA ile katsayı anlamlılığı + durma noktası
  (stationary point) optimizasyonu.
  - Sri (2025), PMC: RSM tasarımlarının karşılaştırması (BBD %96, CCD %98
    optimizasyon doğruluğu). https://pmc.ncbi.nlm.nih.gov/articles/PMC11937516/
- **Uygulama:** `rsm_ccd(factor_ranges)`, `rsm_fit(df, yanıt, terimler)`
  (kuadratik model, statsmodels OLS ile), `find_optimum()`.

### 5.2 Taguchi yöntemi
- **Ortogonal diziler (L9, L16...)** + **S/N oranı** (smaller/larger/nominal
  the better) ile sağlam (robust) tasarım; deneme sayısını minimize eder.
  - ScienceDirect Tarımsal ve Biyolojik Bilimler başlığı altında Taguchi
    özeti; Toapanta (2024) faktöriyel/Taguchi/RSM doğrulama karşılaştırması.
    https://www.mdpi.com/2073-4360/16/14/2051
- **Uygulama:** `taguchi_design(n_seviye_sözlüğü)`, `sn_ratio(ölçümler, hedef)`,
  `taguchi_analyze()` → S/N tablosu + ana etki sıralaması.

### 5.3 Mevcut durum
- Faktöriyel tasarım üretimi Agrista'da VAR; eksik olan **analiz tarafı**
  (faktöriyel ANOVA, ana etki + etkileşim grafikleri) ve RSM/Taguchi'nin tamamı.

---

## 6. Toprak Bilimi ve Bitki Besleme

### 6.1 Jeoistatistik
- **Yarıvaryogram (semivariogram)** modelleme (spherical, exponential,
  Gaussian) → **Kriging** (ordinary, regression) ile toprak özelliklerinin
  mekânsal haritalaması; IDW ile karşılaştırma.
  - Goovaerts (1999), *Geoderma*: "Geostatistics in soil science:
    state-of-the-art" — 1900+ atıf. (PDF: slu.se arşivi)
  - AbdelRahman (2020), *Sustainability*: Ordinary kriging + regression
    kriging + IDW karşılaştırması.
  - Selmy (2022): Semivariogram + ordinary kriging ile toprak özellikleri.
- **Uygulama:** Orta-uzun vade; `scipy` ile varyogram gerçeklemesi yapılabilir
  (veya `pykrige` bağımlılığı). İlk aşama: `spatial_summary()` (mesafe
  sınıflarına göre semivaryans).

---

## 7. Tarım Ekonomisi ve İşletmecilik

### 7.1 Etkinlik analizi
- **DEA** (Veri Zarflama Analizi, parametrik olmayan) ve **SFA/Stokastik
  Sınır** (parametrik): Çiftlik düzeyinde teknik etkinlik ölçümünün iki
  rakip standardı; literatürde sık sık birlikte kullanılırlar.
  - Theodoridis (2011), çiftlik düzeyi DEA vs SFA karşılaştırması (85 atıf).
    https://www.jstor.org/stable/23215265
- **Uygulama:** Uzun vade; DEA'nın LP çekirdeği `scipy.optimize.linprog` ile
  gerçeklenebilir (CCR/BCC modelleri).

### 7.2 Anket/benimseme analizleri
- **Logit/Probit benimseme modelleri** (teknoloji benimseme),
  **çoklu yanıt regresyonu**, Likert ölçek analizleri.
- **Uygulama:** `statsmodels` GLM/Logit sarmalayıcısı — Bitki Koruma'daki
  probit altyapısıyla ortak `binary_response_model()` fonksiyonuna birleşir.

---

## 8. Ortak/Branşlar Arası İhtiyaçlar

Literatürde her branşta tekrar eden ve Agrista'da eksik olan taban araçlar:

| Yöntem | Açıklama | Öncelik |
|---|---|---|
| Çoklu karşılaştırma | Tukey HSD, Duncan, LSD — ANOVA sonrası standart | **P0** |
| Varsayım testleri | Shapiro-Wilk (normallik), Levene (varyans homojenliği) | **P0** |
| Parametrik olmayan testler | Mann-Whitney U, Kruskal-Wallis, Friedman, Wilcoxon | **P0** |
| Bölünmüş parseller | Split-plot, strip-plot tasarımı ve analizi | P1 |
| Tekrarlı ölçüm ANOVA | Depolama/laktasyon/zaman serisi denemeleri | P1 |
| Kendall W uyum katsayısı | Panelist/jüri uyumu | P1 |
| Doğrusal olmayan eğri kütüphanesi | Lojistik ✓, Gompertz ✓, v. Bertalanffy ✓; eksik: monomoleküler, Wood, log-logistik | P1 |

---

## 9. Agrista Uygulama Yol Haritası

> **Durum (2026-08-05): P0, P1 ve P2'nin tamamı uygulandı ve test edildi
> (140+ test). Kriging (pykrige) ve SFA bilinçli olarak ertelendi.**

Mevcut yapı: `analysis/` (t-test, ANOVA, regresyon, ki-kare, Tukey/Duncan,
parametrik olmayan testler), `models/` (büyüme, verim, risk), `experimental/`
(RCBD, Latin kare üretimi + ANOVA'sı, faktöriyel tasarım + ANOVA'sı),
`viz/`, `cli/` (info, describe, corr, ttest, tukey, audpc, dea).

### P0 — ✅ Tamamlandı
1. `agrista/analysis`: `posthoc_tukey()`, `posthoc_duncan()` (gerçek Duncan,
   harf gruplamalı), `normality_test()`, `homogeneity_test()`,
   `mann_whitney_u()`, `kruskal_wallis()`, `wilcoxon_test()`, `friedman_test()`
2. `experimental`: `latin_square_anova()`, `factorial_anova()`

### P1 — ✅ Tamamlandı
3. `agrista/protection/`: `probit_dose_response()`, `loglogistic_dose_response()`,
   `abbott_efficiency()`, `audpc()`, `disease_progress_fit()`
4. `agrista/genetics/`: `path_analysis()`, `pca_analysis()`, `cluster_genotypes()`,
   `mahalanobis_d2()`, `heritability()`
5. `agrista/engineering/`: `rsm_ccd()`, `rsm_bbd()`, `rsm_fit()`, `find_optimum()`,
   `taguchi_design()`, `sn_ratio()`, `taguchi_analyze()`
6. `agrista/animal/`: `mixed_model()` (MixedLM), `fit_wood()`, `lactation_summary()`
7. `models/GrowthModel`: monomoleküler + Wood eğrileri eklendi

### P2 — ✅ Tamamlandı (kriging ve SFA hariç)
8. `agrista/genetics/`: `ammi_analysis()`, `gge_biplot()`, `stability_indices()`
9. `agrista/sensory/`: `kendall_w()`, `hedonic_summary()`, `panel_anova()`
10. `agrista/spatial/`: `semivariogram()`, `idw_interpolation()`, `spatial_summary()`
    (Tam kriging için pykrige entegrasyonu açık kalan tek madde)
11. `agrista/economics/`: `dea_efficiency()` (CCR/BCC, linprog),
    `adoption_logit()` (logit/probit), `partial_budget()` (SFA ertelendi)

### Mimari notlar
- Her yeni analiz fonksiyonu mevcut sözleşmeyi izlemeli: `dict` döndürme,
  Türkçe hata mesajları, `significant_at_005` alanları.
- Branş modülleri `agrista/__init__.py`'ye tembel (lazy) import ile eklenmeli;
  ağır bağımlılıklar (pykrige vb.) opsiyonel ekstra olarak tanımlanmalı.
- Her yeni fonksiyon için `tests/` içine karşılık test sınıfı eklenmeli
  (mevcut örüntü: deterministik `np.random.default_rng(seed)` kullanımı).

---

## Kaynak Listesi

1. Piepho H.P. (2024). Efficacy assessment in crop protection: a tutorial on
   the use of Abbott's formula. *J Plant Dis Prot.* — springer.com/article/10.1007/s41348-024-00968-0
2. Rosenheim J.A., Hoy M.A. (1989). Confidence Intervals for the Abbott's
   Formula Correction. *J. Econ. Entomol.*
3. FAO. Chemical control methods — probit analysis & relative potency.
   fao.org/4/x5048e/x5048E0m.htm
4. Chandran (2025). Modeling Plant Disease Epidemics: A Comprehensive Review.
   *JOSTA* — jostapubs.com
5. PDIndex R package (CRAN) — AUDPC trapez gerçeklemesi.
6. Mullualem (2024). Genotype-by-environment interaction and stability
   analysis. *Heliyon* — sciencedirect.com/pii/S2405844024089497
7. Derbew (2024). AMMI and GGE biplot analysis for barley. *agg2.20565*
8. Masoodi (2025). Yield stability and GE interaction. *Sci Rep* —
   nature.com/articles/s41598-025-07621-2
9. Mubai (2020). Phenotypic correlation, path coefficient and multivariate
   analysis. *Cogent Food Agric* — tandfonline.com/10.1080/23311932.2020.1823591
10. Roka (2024). A review on genetic parameters estimation and trait
    association. *AAES* — aesacademy.org
11. Schmidt (2019). Heritability in Plant Breeding on a Genotype-Difference
    Basis. *Heredity* — pmc.ncbi.nlm.nih.gov/articles/PMC6707473
12. Moreira (2025). Sensory Analysis to Inform Breeding Decisions. *HortScience*
13. Gunness (2009). Postharvest Biology and Technology — strawberry variation.
14. Goonewardene (2004). The use of MIXED models in the analysis of animal
    experiments with repeated measures data. *J. Anim. Sci.*
15. VSNI. BLUEs, BLUPs and Breeding Values. vsni.co.uk/blups_blues_breeding_values
16. Kim (2009). Balanced Latin Square design. dialnet.unirioja.es
17. Zanton (2019). Effect of experimental design on responses. *J. Dairy Sci.*
18. CIPAV. The design and interpretation of animal feeding trials.
19. Sri (2025). Comparison between response surface methodology designs. PMC —
    pmc.ncbi.nlm.nih.gov/articles/PMC11937516
20. Toapanta (2024). Validation of DOE Factorial/Taguchi/RSM. *Polymers* —
    mdpi.com/2073-4360/16/14/2051
21. Goovaerts (1999). Geostatistics in soil science: state-of-the-art. *Geoderma*
22. AbdelRahman (2020). Soil Spatial Variability through Geostatistics.
    *Sustainability* — mdpi.com/2071-1050/13/1/194
23. Selmy (2022). Characterizing, predicting, and mapping soil spatial
    variability. sciencedirect.com/pii/S1658077X21001508
24. Theodoridis (2011). Farm efficiency: DEA vs SFA comparison. *JSTOR* —
    jstor.org/stable/23215265
25. Abukari (2019). Deterministic and stochastic approaches to farm
    efficiency. *AgriSe* — agrise.ub.ac.id
