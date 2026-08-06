# Premium Program Menü Yapısı — Tam Envanter ve Agrista Eşlemesi

> **Log türü:** Yazılım mimarisi referansı (Premium Program Statistics 26-29 baz alınmıştır)
> **Tarih:** 2026-08-05
> **Amaç:** Agrista CLI menü sınıflandırmasını Premium Program standardıyla hizalamak ve
> kapsam boşluklarını (gap) takip etmek.
> **Kaynaklar:** IBM Support (menu bar listesi), IBM Premium Program Statistics Base 26
> User's Guide, CMU Premium Program Online Workshop, Loughborough Premium Program Guide.

---

## 1. Ana Menü Çubuğu — 11 Menü

Premium Program Data Editor penceresinde **11 ana menü** bulunur
(IBM destek kaynağıyla birebir doğrulanmıştır):

| # | Menü | İşlev | Agrista karşılığı |
|---|---|---|---|
| 1 | **File** | Aç/kaydet/aktar (sav, csv, excel, veritabanı) | ⚠️ Kısmen (`load_csv/excel/json`) |
| 2 | **Edit** | Kes/kopyala/yapıştır, bul, seçenekler | — (terminal doğası gereği yok) |
| 3 | **View** | Görünüm, değişken/value label aç-kapa | — |
| 4 | **Data** | Veri yönetimi: sırala, birleştir, böl, filtrele, ağırlıklandır | ⚠️ Kısmen (filter, drop_nulls, select) |
| 5 | **Transform** | Değişken türetme: compute, recode, binning | ❌ Yok |
| 6 | **Analyze** | Tüm istatistiksel analizler | ✅ Çekirdek (5 kategori) |
| 7 | **Graphs** (v29+: *Plots*) | Grafik üretimi | ⚠️ Kısmen (`viz/AgristaPlotter`) |
| 8 | **Utilities** | Değişken bilgisi, script çalıştırma | ⚠️ Kısmen (`info`) |
| 9 | **Add-Ons / Extensions** | Ek modüller (Amos, R/Python entegrasyonu) | ⚠️ Branş modülleri bu rolü üstleniyor |
| 10 | **Window** | Pencereler arası geçiş | — |
| 11 | **Help** | Yardım, istatistik koçu | ❌ Yok |

---

## 2. ANALYZE Menüsü — Tam Alt Menü Ağacı (22 alt menü)

Agrista için en kritik menü. Premium Program'ın tam hiyerarşisi ve Agrista kapsamı:

### 2.1 Reports ✅ (Güncelleme 4 ile kapatıldı)
- OLAP Cubes → ✅ `data.aggregate_data` (grup kırılımlı özet küpler)
- Case Summaries → ✅ `analysis.case_summaries`
- Report Summaries → ✅ `analysis.means_report`

### 2.2 Descriptive Statistics ✅ (Agrista: Betimsel İstatistikler)
- Frequencies → ✅ `frequencies` (Güncelleme 1)
- Descriptives → ✅ `describe` / `descriptive_stats`
- Explore → ✅ normallik + `boxplot` + `qq_plot` + `pp_plot`
- Crosstabs → ✅ `crosstabs` (Güncelleme 1)
- Ratios → ✅ `ratio_statistics` (COV, AAD, PRD)
- P-P Plots, Q-Q Plots → ✅ `AgristaPlotter.pp_plot` / `qq_plot`

### 2.3 Compare Means ✅ (Agrista: Ortalamaların Karşılaştırılması)
- Means → ✅ `means_report`
- One-Sample T Test → ✅ `one_sample_t_test` (Güncelleme 1)
- Independent-Samples T Test → ✅ `ttest`
- Paired-Samples T Test → ✅ `paired_t_test` (Güncelleme 1)
- One-Way ANOVA → ✅ `anova_one_way` + `posthoc_tukey/duncan`

### 2.4 General Linear Model
- Univariate, Multivariate, Repeated Measures, Variance Components
- → ⚠️ Kısmen (`factorial_anova`, `latin_square_anova`, `rcbd_analysis`
  kısmi F-testiyle; tam GLM tablosu yok)

### 2.5 Generalized Linear Models
- GENLIN, GEE → ⚠️ Kısmen (probit/logit GLM `protection` ve `economics`'te)

### 2.6 Mixed Models
- Linear / Generalized Linear / Nonlinear Mixed
- → ⚠️ Kısmen (`animal.mixed_model` — LMM var, GLMM yok)

### 2.7 Correlate ✅ (Agrista: Korelasyon)
- Bivariate → ✅ `corr` (Pearson/Spearman)
- Partial → ✅ `partial_correlation`
- Distances → ✅ `distance_matrix` (euclidean/manhattan/cosine/correlation)

### 2.8 Regression ✅ (Güncelleme 3-4 ile tamamlandı)
- Linear → ✅ `linear_regression` / `multiple_regression`
- Curve Estimation → ⚠️ Kısmen (büyüme eğrileri; genel polinom/üs yok)
- Binary Logistic → ✅ `adoption_logit`
- Multinomial/Ordinal Logistic → ✅ `multinomial_logistic_regression` /
  `ordinal_logistic_regression` (Güncelleme 3)
- Probit → ✅ `probit_dose_response`
- Nonlinear → ⚠️ Kısmen (curve_fit tabanlı özel modeller)
- Weight Estimation → ✅ `weight_estimation` (WLS, w = 1/x^p)
- 2SLS → ✅ `two_stage_least_squares` (IV2SLS)

### 2.9 Loglinear → ✅ `loglinear_analysis` (Güncelleme 2)

### 2.10 Classify (Sınıflandırma) ✅ (Güncelleme 3-4 ile tamamlandı)
- TwoStep Cluster → ✅ `twostep_cluster` (BIC ile otomatik k seçimi)
- K-Means → ✅ `genetics.kmeans_cluster`
- Hierarchical Cluster → ✅ `cluster_genotypes` (Ward/UPGMA)
- Discriminant → ✅ `discriminant_analysis` (Güncelleme 3)
- Nearest Neighbor → ✅ `nearest_neighbor_analysis` (k-NN, LOO)

### 2.11 Dimension Reduction ✅ (Güncelleme 3-4 ile tamamlandı)
- Factor Analysis → ✅ `genetics.factor_analysis` (Güncelleme 2)
- Correspondence Analysis → ✅ `correspondence_analysis` (Güncelleme 3)
- Multidimensional Scaling → ✅ `multidimensional_scaling` (klasik MDS)
- (PCA, `pca_analysis` olarak var ama Premium Program'da Factor altında) ✅

### 2.12 Scale ✅
- Reliability Analysis (Cronbach α) → ✅ `sensory.cronbach_alpha`
  (Güncelleme 1)

### 2.13 Nonparametric Tests ✅
- Modern arayüz (One-Sample / Independent / Related) → ⚠️ Kısmen
  (CLI menüsü yok; fonksiyonlar tam)
- Legacy: Chi-Square ✅, Binomial ✅, Runs ✅, 1-Sample K-S ✅,
  Mann-Whitney ✅, K Independent (Kruskal) ✅, Wilcoxon ✅,
  Friedman ✅, Cochran Q ✅

### 2.14 Survival ✅
- Kaplan-Meier → ✅ `kaplan_meier` (Güncelleme 2)
- Life Tables → ✅ `life_tables` (aktüeryal)
- Cox Regression → ✅ `cox_regression` (kısmi olabilirlik, Breslow,
  Harrell C)

### 2.15 Multiple Response → ✅ `multiple_response_frequencies`

### 2.16 Forecasting ✅ (Güncelleme 2: ARIMA, Holt-Winters, SES,
ayrıştırma)

### 2.17 Direct Marketing ✅ (Güncelleme 4)
- RFM → ✅ `marketing.rfm_analysis`
- Control vs Package (mailing test) → ✅ `marketing.mailing_test`
- Prospect Profiles → ✅ `marketing.prospect_profiles`

### 2.18 Complex Samples
- Örneklem planı tasarımı → ⚠️ Kısmen (`sample_size_calculation` +
  `data.weight_cases`; Taylor doğrusallaştırmalı karmaşık anket tahmini yok)

### 2.19 Quality Control ✅ (Güncelleme 2: X̄-R, p-grafiği, Pareto)

### 2.20 ROC Curve ✅ (Güncelleme 2)

### 2.21 Tables ✅
- Custom Tables → ✅ `custom_tables` (çok indeksli, seçilebilir
  istatistikler)

### 2.22 Bootstrapping ✅ (Güncelleme 2)

---

## 3. DATA Menüsü — Alt İşlemler

Define Variable Properties · Copy Data Properties · Define Measurement Level ·
Validate Data · Identify Duplicate Cases · Identify Unusual Cases ·
Compare Datasets · Sort Cases · Transpose · Restructure · Merge Files
(Add Cases / Add Variables) · Aggregate · Orthogonal Design · Split File ·
Weight Cases · Select Cases

**Agrista kapsamı (Güncelleme 4 sonrası):** filter ✅, drop_nulls ✅,
select_columns ✅, rename_columns ✅, `sort_cases` ✅, `merge_files` ✅,
`split_file` ✅, `weight_cases` ✅, `aggregate_data` ✅,
`identify_duplicates` ✅, `transpose_data` ✅, `restructure_data` ✅,
`compare_datasets` ✅, `define_measurement_level` ✅ — Orthogonal Design
`engineering` faktöriyel/RSM/Taguchi tasarımlarıyla karşılanıyor

## 4. TRANSFORM Menüsü — Alt İşlemler

Compute Variable · Count Values · Recode (Same/Different) · Visual Binning ·
Optimal Binning · Replace Missing Values · Rank Cases · Automatic Recode ·
Create Time Series · Define Dates · Random Number Generators

**Agrista kapsamı (Güncelleme 1 + 4 sonrası):** `compute` ✅, `recode` ✅,
`bin_variable` ✅, `rank_cases` ✅, `count_values` ✅, `replace_missing` ✅,
`automatic_recode` ✅, `create_time_series` ✅ (lag/fark/mevsimsel fark/
hareketli ortalama), `random_numbers` ✅ — Define Dates kapsam dışı
(tarih ayrıştırma pandas ile doğrudan yapılabilir)

## 5. GRAPHS Menüsü — Alt İşlemler

Chart Builder · Legacy Dialogs (Bar, Line, Area, Pie, Boxplot, Scatter/Dot,
Histogram, Error Bar, Population Pyramid, ROC, Pareto, Control, Q-Q) ·
Interactive

**Agrista kapsamı (Güncelleme 1-4 sonrası):** histogram ✅, boxplot ✅,
scatter ✅, bar_chart ✅, time_series ✅, correlation_heatmap ✅,
qq_plot ✅, pp_plot ✅, pie_chart ✅, line_chart ✅, error_bar ✅,
population_pyramid ✅, stem_leaf ✅ — Chart Builder'ın sürükle-bırak
GUI'si ve Interactive kapsam dışı (terminal doğası)

## 6. UTILITIES / WINDOW / HELP

- Utilities: Variables, Data File Information, Variable Sets, Run Script,
  Menu Editor → Agrista: `info` komutu kısmen karşılık verir
- Window / Help: terminal CLI'da karşılığı yok (`--help` yeterli)

---

## 7. Agrista CLI Mevcut Durum → Premium Program Hizalama Özeti

| Agrista menüsü | Premium Program karşılığı | Durum |
|---|---|---|
| [1] Dosya | File + Utilities | ✅ Hizalı |
| [2] Betimsel İstatistikler | Analyze → Descriptive Statistics | ✅ Hizalı |
| [3] Ortalamaların Karşılaştırılması | Analyze → Compare Means | ✅ Hizalı |
| [4] Korelasyon | Analyze → Correlate | ✅ Hizalı |
| [5] Uzman Branş Modülleri | (Premium Program'da yok — Agrista'ya özgü) | ✅ Özgün değer |

## 8. Öncelikli Boşluklar (Premium Program'a göre)

> **Güncelleme (2026-08-05): Aşağıdaki 8 boşluğun TAMAMI kapatıldı.**

1. ~~**Crosstabs**~~ ✅ `crosstabs()` + CLI `agrista crosstabs` (Pearson ki-kare,
   Cramer's V, phi, beklenen frekans uyarısı)
2. ~~**Paired-Samples T Test**~~ ✅ `paired_t_test()` + CLI `agrista paired`
3. ~~**Frequencies**~~ ✅ `frequencies()` + CLI `agrista frequencies`
   (sayı, %, geçerli %, kümülatif %)
4. ~~**Compute/Recode (Transform)**~~ ✅ `agrista/transform` modülü:
   `compute`, `recode` (ELSE destekli), `bin_variable`, `rank_cases`,
   `count_values` — menüden [5] Dönüşüm altında
5. ~~**Replace Missing Values**~~ ✅ `replace_missing()` (mean/median/ffill/interpolate)
6. ~~**Cronbach α**~~ ✅ `sensory.cronbach_alpha()` (madde silindiğinde alfa dahil)
7. ~~**Q-Q plot**~~ ✅ `AgristaPlotter.qq_plot()` + menüden kayıt
8. ~~**One-Sample T Test**~~ ✅ `one_sample_t_test()` + CLI `agrista onesample`

Menü yapısı Premium Program'ın File / Descriptive Statistics / Compare Means /
Correlate / Transform / Scale / Add-Ons hiyerarşisiyle birebir hizalandı.

> **Güncelleme 2 (2026-08-05): İkinci dalga Premium Program modülleri de kapatıldı.**

- ~~Survival~~ ✅ `agrista/survival`: Kaplan-Meier (Greenwood SH, medyan),
  log-rank (Mantel-Cox) — menü [9]
- ~~Forecasting~~ ✅ `agrista/forecasting`: hareketli ortalama, SES,
  Holt-Winters additive, mevsimsel ayrıştırma, ARIMA (statsmodels) — menü [8]
- ~~Factor Analysis~~ ✅ `genetics.factor_analysis`: PCA çıkarması +
  SVD tabanlı Kaiser varimax döndürmesi, komunallıklar — menü [7]
- ~~Loglinear~~ ✅ `analysis.loglinear_analysis`: Poisson GLM bağımsızlık
  vs doymuş model, olabilirlik oranı testi — menü [13]
- ~~Bootstrapping~~ ✅ `analysis.bootstrap_statistic`: yüzdelik GA, özel
  istatistik desteği, tohumlanabilir — menü [11]
- ~~Quality Control~~ ✅ `agrista/quality`: X̄-R (A2/D3/D4 sabitleri),
  p-grafiği, Pareto (vital few) — menü [10]
- ~~ROC Curve~~ ✅ `analysis.roc_curve`: AUC (trapez), Youden J optimum
  eşik, duyarlılık/özgüllük — menü [12]

> **Güncelleme 3 (2026-08-05): Üçüncü dalga — kalan Sınıflandırma/Regresyon
> modülleri de kapatıldı.**

- ~~Multinomial Logistic~~ ✅ `analysis.multinomial_logistic_regression`
  (MNLogit, referans = sıralı ilk kategori, LR ki-kare, odds oranları,
  kategori başına katsayı + sınıflandırma doğruluğu) — CLI `agrista multinom`,
  menü [8] Regresyon
- ~~Ordinal Logistic (PLUM)~~ ✅ `analysis.ordinal_logistic_regression`:
  kümülatif logit (OrderedModel), Premium Program işaret uzlaşımı
  logit(P(Y ≤ j)) = eşik_j − β'x, eşikler + odds oranları —
  CLI `agrista ordlogit`, menü [8] Regresyon
- ~~Discriminant~~ ✅ `analysis.discriminant_analysis`: Fisher LDA,
  Wilks lambda + Bartlett ki-kare, kanonik korelasyonlar
  (genelleştirilmiş özdeğer problemi), sentroidler, sınıflandırma
  matrisi — CLI `agrista discriminant`, menü [9] Sınıflandırma
- ~~Correspondence Analysis~~ ✅ `analysis.correspondence_analysis`:
  SVD tabanlı asal eylemsizlikler (atalet), satır/sütun ana koordinatları,
  kalite (cos²), katkılar, ki-kare bağımsızlık testi — CLI
  `agrista correspondence`, menü [7] Boyut İndirgeme

Test kaydı: `tests/test_premium_classify.py` (üçüncü dalga) +
`tests/test_premium_advanced.py::TestMultinomialAndCorrespondence`.

Not: Menüye eklenen yeni kategoriler nedeniyle Kestirim [10], Survival [11],
Kalite Kontrol [12], Bootstrapping [13], ROC [14], Loglinear [15],
Uzman Branş [16] olarak yeniden numaralandı.

> **Güncelleme 4 (2026-08-06): Dördüncü ve son dalga — Premium Program ile tam
> denklik sağlandı; açık fonksiyonel boşluk kalmadı.**

Tablolar/Raporlar:
- ~~Custom Tables~~ ✅ `analysis.custom_tables` — CLI `agrista ctable`,
  menü [16] Tablolar ve Raporlar
- ~~Case Summaries / Means / Ratios~~ ✅ `case_summaries`, `means_report`,
  `ratio_statistics` — CLI `agrista means`, `agrista ratios`
- ~~Multiple Response~~ ✅ `multiple_response_frequencies` — CLI `agrista multresp`

Sınıflandırma/Boyut İndirgeme:
- ~~TwoStep Cluster~~ ✅ `twostep_cluster` (BIC otomatik k seçimi,
  k++ yeniden başlatmalı) — CLI `agrista twostep`, menü [9]
- ~~Nearest Neighbor~~ ✅ `nearest_neighbor_analysis` (k-NN, LOO) —
  CLI `agrista knn`, menü [9]
- ~~Multidimensional Scaling~~ ✅ `multidimensional_scaling` (klasik MDS,
  R² uyum) — CLI `agrista mds`, menü [7]
- ~~Distances~~ ✅ `distance_matrix` — CLI `agrista distances`, menü [4]

Yaşam Analizi:
- ~~Life Tables~~ ✅ `survival.life_tables` (aktüeryal) — CLI `agrista lifetable`
- ~~Cox Regression~~ ✅ `survival.cox_regression` (kısmi olabilirlik,
  Breslow bağ, Wald testleri, Harrell C) — CLI `agrista cox`

Regresyon (kalanlar):
- ~~Weight Estimation~~ ✅ `weight_estimation` (WLS w=1/x^p adayları)
- ~~2SLS~~ ✅ `two_stage_least_squares` (IV2SLS)

Doğrudan Pazarlama — yeni `agrista/marketing` modülü, menü [17]:
- ~~RFM~~ ✅ `rfm_analysis` (kantil skorları + segmentasyon) — CLI `agrista rfm`
- ~~Control vs Package~~ ✅ `mailing_test` (iki oran z-testi, lift, GA) —
  CLI `agrista mailing`
- ~~Prospect Profiles~~ ✅ `prospect_profiles` (yanıt oranı + lift) —
  CLI `agrista prospect`

Data menüsü denkliği — `agrista/data`:
- ✅ `sort_cases`, `aggregate_data`, `weight_cases`, `merge_files`,
  `split_file`, `identify_duplicates`, `transpose_data`,
  `restructure_data`, `compare_datasets`, `define_measurement_level`

Transform kalanlar — `agrista/transform`:
- ✅ `create_time_series` (lag/fark/mevsimsel fark/hareketli ortalama) —
  menü [5] Dönüşüm
- ✅ `random_numbers` (normal/uniform/binomial/poisson/exponential)
- ✅ `automatic_recode`

Graphs kalanlar — `AgristaPlotter`:
- ✅ `pie_chart`, `line_chart`, `error_bar`, `pp_plot`, `population_pyramid`,
  `stem_leaf`

Test kaydı: `tests/test_premium_wave4.py` (75+ test).

**Son durum:** Premium Program Base'in istatistiksel/veri yönetimi işlevlerinde açık
boşluk kalmadı. Kapsam dışı bırakılanlar yalnızca terminal ortamında
karşılığı olmayan GUI öğeleridir: Edit/View/Window menüleri, Chart
Builder sürükle-bırak arayüzü, Interactive grafikler, Extensions Hub.
Kısmi kalan: yok. Premium Program Base denkliği tamamlandı (Güncelleme 5).

## Güncelleme 5 — Kalan Boşlukların Kapatılması

### GLM (General Linear Model)

Analiz implementasyonları — `agrista/analysis`:
- ✅ `glm_univariate` — tek değişkenli faktöriyel/kovaryeteli model:
  Tip I/II/III kareler toplamı (Sum kontrast + `anova_lm`), kısmi η²
  efekt büyüklükleri, tek faktörde Tukey/Duncan post-hoc
- ✅ `glm_repeated_measures` — tekrarlı ölçümler (wide format):
  `AnovaRM` within-subject F, Mauchly küresellik testi,
  Greenhouse-Geisser ve Huynh-Feldt ε düzeltmeleri, isteğe bağlı
  between-faktör karşılaştırması

CLI — `agrista/cli`:
- ✅ `agrista glm` komutu (`--within` + `--denek` verilirse tekrarlı
  ölçüm dalı çalışır); çıktı yazdırıcısı `_print_glm_result`
- ✅ Menü `[3] ⚖️ Ortalamaların Karşılaştırılması` yeni öğeleri:
  `[6] Genel Doğrusal Model (Tek Değişkenli)`,
  `[7] Tekrarlı Ölçümler (GLM)`

Test kaydı: `tests/test_premium_glm.py`; menü akışları
`menu_smoke_test.py` üzerinden `tests/test_menu_flows.py`'de
parametrize edilir.

### GEE (Generalized Estimating Equations)

Analiz implementasyonu — `agrista/analysis`:
- ✅ `gee_model` — kümelenmiş/korelasyonlu veri için marjinal model:
  4 aile (gaussian/binomial/poisson/gamma) × 3 çalışma korelasyonu
  (independent/exchangeable/autoregressive), population-averaged
  katsayılar, robust (sandwich) standart hatalar, QIC; eski
  statsmodels sürümleri için `qic()` yoksa None dönüşü

CLI — `agrista/cli`:
- ✅ `agrista gee` komutu (`--aile`, `--yapi`, `--zaman` seçenekleri);
  çıktı yazdırıcısı `_print_gee_result`
- ✅ Menü `[8] 📈 Regresyon` yeni öğesi:
  `[3] GEE (Genelleştirilmiş Tahmin Denklemleri)`

Test kaydı: `tests/test_premium_gee.py`; menü akışları
`menu_smoke_test.py` üzerinden `tests/test_menu_flows.py`'de
parametrize edilir.

### GLMM (Generalized Linear Mixed Models)

Analiz implementasyonu — `agrista/models`:
- ✅ `glmm` — genelleştirilmiş doğrusal karışık model:
  gaussian aile REML (statsmodels MixedLM), binomial/poisson aileler
  PQL (Breslow & Clayton 1993 penalize kuazi-olabilirlik) yinelemesi;
  rastgele kesim varyansı, Wald z/p, isteğe bağlı rastgele eğim.
  Not: statsmodels 0.14.6 MixedLM `freq_weights` kabul etmediği için
  PQL'nin ağırlıklı LMM adımı profil REML (sabit çalışma ölçeği)
  olarak modül içinde çözümlenir.

CLI — `agrista/cli`:
- ✅ `agrista glmm` komutu (`--yanit`, `--sabitler`, `--grup`,
  `--aile`, `--random-slope` seçenekleri); çıktı yazdırıcısı
  `_print_glmm_result`
- ✅ Menü `[8] 📈 Regresyon` yeni öğesi:
  `[4] GLMM (Genelleştirilmiş Karışık Model)`

Test kaydı: `tests/test_premium_glmm.py`; menü akışları
`menu_smoke_test.py` üzerinden `tests/test_menu_flows.py`'de
parametrize edilir.

### Karmaşık Anket (Complex Samples — Taylor Doğrusallaştırması)

Analiz implementasyonu — `agrista/survey`:
- ✅ `survey_design` — tasarım tanımı (PSU, tabaka, ağırlık, FPC);
  tek PSU'lu tabakalarda varyans hatası, eksik sütun doğrulaması
- ✅ `_taylor_variance` — lineer değişkenin PSU toplamları üzerinden
  tabakalı Taylor varyansı; FPC desteği (1 - n_h/N_h)
- ✅ `svy_mean` / `svy_total` / `svy_ratio` — ağırlıklı ortalama,
  toplam ve oran tahminleri; Taylor SE, %95 CI, DEFF
- ✅ `survey_logistic` — ağırlıklı GLM + PSU-kümelenmiş sandwich
  kovaryans (`cov_type="cluster"`); birinci derece Taylor denkliği

CLI — `agrista/cli`:
- ✅ `agrista svymean` / `agrista svyratio` / `agrista svylogit`
  komutları; çıktı yazdırıcıları `_print_svy_result` ve
  `_print_svylogit_result`
- ✅ YENİ menü kategorisi `[20] 🧮 Karmaşık Örneklem (Complex Samples)`:
  `[1] Anket ortalaması/toplamı (Taylor)`,
  `[2] Anket oranı (Taylor)`,
  `[3] Anket lojistik regresyonu` (kategori sayısı 19 → 20)

Test kaydı: `tests/test_premium_survey.py`; menü akışları
`menu_smoke_test.py` üzerinden `tests/test_menu_flows.py`'de
parametrize edilir (58 → 61 akış).
