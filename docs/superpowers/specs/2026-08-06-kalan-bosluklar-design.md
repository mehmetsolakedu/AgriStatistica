# Tasarım: Premium Program Kalan Boşluklarının Kapatılması (GLM, GEE, GLMM, Karmaşık Anket)

**Tarih:** 2026-08-06
**Durum:** Onaylandı (Aşama 1 — Brainstorming)
**Referans:** `docs/02_PREMIUM_PROGRAM_MENU_YAPISI_LOG.md` son durum notu

## 1. Amaç

`docs/02` log'unda "kısmi kalanlar" olarak listelenen son üç kalemi —
**GLM/GEE (2.5)**, **GLMM (2.6)** ve **karmaşık anket Taylor doğrusallaştırması
(2.18)** — proje misyonuyla (Premium Program Base ile tam fonksiyonel denklik)
tutarlı biçimde tamamlayarak istatistiksel kapsamda açık boşluk bırakmamak.

Kapsam dışı: GUI öğeleri (Edit/View/Window, Chart Builder, Interactive
grafikler, Extensions Hub) — terminal karşılığı yok, log'da zaten dışarıda.

## 2. Global Kısıtlar (tüm görevler için geçerli)

1. **Yeni çekirdek bağımlılık YOK.** Mevcut yığın (numpy, pandas, scipy,
   statsmodels, click) içinde kalınır. `pyproject.toml` bağımlılıkları
   değişmez.
2. **Adlandırma:** Referans paketi için her yerde yalnızca "Premium Program"
   adı kullanılır; referans paketinin eski adı hiçbir dosyada yer alamaz.
3. **Fonksiyon şeması:** Analiz fonksiyonları `dict` döner; sayısal alanlar
   `float`/`int` cast'li; Türkçe docstring; girdi doğrulama için mevcut
   `_check_columns` deseni kullanılır.
4. **CLI:** `agrista/cli/__init__.py` içinde click komutları; Türkçe
   seçenek adları (`--yanit`, `--grup`...); çıktı yazdırıcıları
   `_print_*_result` yardımcıları olarak ayrı fonksiyonlar.
5. **Menü:** Etkileşimli menü handler'ları `_prompt_or_eof` koruması
   kullanır; `click.prompt` doğrudan kullanılmaz.
6. **Test:** Her fonksiyon için `tests/test_premium_*.py` altında TDD ile
   yazılmış testler; CLI komutları `click.testing.CliRunner` ile sınanır;
   menü akışları `tests/test_menu_flows.py` desenine eklenir.
7. **Kayıt:** Her alt sistem tamamlandığında `docs/02_PREMIUM_PROGRAM_MENU_YAPISI_LOG.md`
   dosyasına "Güncelleme 5" bölümü eklenir.
8. **Kalite kapısı:** Tüm paket `pytest` yeşil + `flake8` temiz olmadan
   görev tamamlanmış sayılmaz.

## 3. Mimari ve Dosya Yerleşimi

| Alt sistem | Kod yeri | Test dosyası | CLI komutları | Menü yeri |
|---|---|---|---|---|
| GLM (tek değişkenli + tekrarlı ölçüm) | `agrista/analysis/__init__.py` | `tests/test_premium_glm.py` | `agrista glm` | [3] Ortalamaların Karşılaştırılması |
| GEE | `agrista/analysis/__init__.py` | `tests/test_premium_gee.py` | `agrista gee` | [8] Regresyon |
| GLMM | yeni `agrista/models/glmm.py` + `models/__init__.py` re-export | `tests/test_premium_glmm.py` | `agrista glmm` | [8] Regresyon |
| Karmaşık Anket | yeni `agrista/survey/__init__.py` | `tests/test_premium_survey.py` | `agrista svymean`, `agrista svyratio`, `agrista svylogit` | yeni kategori [20] Karmaşık Örneklem |

Üst düzey paket `agrista/__init__.py` güncellenir: `from agrista import
survey` eklenir; `models.glmm` fonksiyonları `agrista.models` üzerinden
erişilebilir olur.

## 4. Alt Sistem 1 — GLM (General Linear Model)

### 4.1 `glm_univariate`

```python
def glm_univariate(
    data: pd.DataFrame,
    response: str,
    between_factors: list[str],
    covariates: list[str] | None = None,
    ss_type: int = 3,
    posthoc: str | None = "tukey",   # "tukey" | "duncan" | None
    alpha: float = 0.05,
) -> dict
```

Davranış:
- Formül: `response ~ C(f1) + C(f2) + cov1 + ...`; Tip III için faktörler
  Sum-kodlama (`- Sum` contrast) ile kurulur.
- `statsmodels.stats.anova.anova_lm(typ=ss_type)` ile SS tablosu.
- Kısmi eta-kare: `SS_efekt / (SS_efekt + SS_artik)`.

Dönüş şeması:
```python
{
  "model": "GLM Univariate",
  "ss_type": 3,
  "anova_table": [{"source", "ss", "df", "ms", "f_value", "p_value"}...],
  "effect_sizes": {"f1": 0.42, ...},          # kısmi eta-kare
  "r_squared": float,
  "posthoc": {"tukey": [...]} | None,         # yalnız tek faktör istendi
  "n_obs": int,
}
```

Hata durumları: sütun yoksa `ValueError` (mevcut `_check_columns` deseni);
kalıntı df ≤ 0 ise `ValueError`.

### 4.2 `glm_repeated_measures`

```python
def glm_repeated_measures(
    data: pd.DataFrame,
    response_cols: list[str],   # her zaman noktası bir sütun (wide format)
    subject_col: str,
    between_factor: str | None = None,
    alpha: float = 0.05,
) -> dict
```

Davranış:
- Wide → long dönüşümü (`pd.melt`), `statsmodels.stats.anova.AnovaRM`
  ile within-subject F testi.
- **Mauchly küresellik testi:** koşul kovaryans matrisinin ortogonal
  kontrastlara dönüşümünün özdeğerlerinden W istatistiği ve ki-kare p.
- **Düzeltmeler:** Greenhouse-Geisser ve Huynh-Feldt epsilonları; düzeltilmiş
  p-değerleri F dağılımından df çarpılarak hesaplanır.

Dönüş şeması:
```python
{
  "model": "GLM Repeated Measures",
  "within_effect": {"f_value", "df1", "df2", "p_value"},
  "mauchly": {"w": float, "chi2": float, "p_value": float},
  "epsilon": {"greenhouse_geisser": float, "huynh_feldt": float},
  "corrected": {
      "greenhouse_geisser": {"df1", "df2", "p_value"},
      "huynh_feldt": {"df1", "df2", "p_value"},
  },
  "between_effect": {...} | None,
  "n_subjects": int,
}
```

Hata durumları: < 3 tekrar koşulu → `ValueError`; < 2 denek → `ValueError`.

## 5. Alt Sistem 2 — GEE (Generalized Estimating Equations)

```python
def gee_model(
    data: pd.DataFrame,
    response: str,
    covariates: list[str],
    group_col: str,
    family: str = "gaussian",      # gaussian | binomial | poisson | gamma
    cov_struct: str = "independent",  # independent | exchangeable | autoregressive
    time_col: str | None = None,   # autoregressive için gerekli
) -> dict
```

Davranış:
- `statsmodels.formula.api.gee` ile marjinal model; bağlantı fonksiyonu
  ailenin kanonik bağlantısı (override edilmez).
- Robust (sandwich) standart hatalar sonuç şemasına taşınır.
- QIC: `GEEResults.qic()` mevcutsa; eski statsmodels sürümlerinde `None`.

Dönüş şeması:
```python
{
  "model": "GEE",
  "family": "gaussian", "cov_struct": "exchangeable",
  "coefficients": {name: {"coefficient", "std_err", "z_value", "p_value"}},
  "qic": float | None,
  "n_groups": int, "n_obs": int,
  "converged": bool,
}
```

Hata durumları: `autoregressive` + `time_col` yok → `ValueError`; tek
gruplu veri → `ValueError`; binomial için 0/1 dışı yanıt → `ValueError`.

## 6. Alt Sistem 3 — GLMM (Generalized Linear Mixed Models)

Yeni dosya `agrista/models/glmm.py`; `agrista/models/__init__.py`
`from agrista.models.glmm import glmm` ile re-export eder.

```python
def glmm(
    data: pd.DataFrame,
    response: str,
    fixed_effects: list[str],
    groups_col: str,
    family: str = "gaussian",      # gaussian | binomial | poisson
    random_slope: str | None = None,
    max_iter: int = 25,
    tol: float = 1e-5,
) -> dict
```

Davranış:
- **gaussian:** `smf.mixedlm` (REML) — `animal.mixed_model` ile aynı şema.
- **binomial / poisson:** Penalized Quasi-Likelihood (Breslow & Clayton 1993):
  1. Rastgele etkiyi yok sayan GLM ile başlat → `mu`, `eta`.
  2. Pseudo-yanıt `z = eta + (y - mu) / g'(mu)`; ağırlık
     `w = 1 / (V(mu) · g'(mu)²)`.
  3. Ağırlıklı LMM (`smf.mixedlm`, `freq_weights=w`) ile beta + BLUP'lar.
  4. `eta = X·beta + Z·b` güncelle; beta ve varyans bileşenleri `tol`
     içinde sabitlenene dek yinele.
  5. Wald z/p son iterasyonun ağırlıklı LMM kovaryansından.

Dönüş şeması (`animal.mixed_model` ile uyumlu + ek alanlar):
```python
{
  "model": "GLMM",
  "family": "binomial",
  "method": "PQL" | "REML",
  "converged": bool,
  "n_iterations": int,
  "fixed_effects": {name: {"coefficient", "std_err", "z_value", "p_value"}},
  "random_effects_variance": {"random_intercept": float},
  "aic": float | None,      # gaussian'da REML AIC; PQL'de None
  "n_obs": int, "n_groups": int,
}
```

Hata durumları: bilinmeyen family → `ValueError`; grup başına ortalama
gözlem < 2 → `ValueError` (karışık model anlamsız); yakınsama olmazsa
`converged: False` döner, hata fırlatılmaz.

## 7. Alt Sistem 4 — Karmaşık Anket (Complex Samples)

Yeni modül `agrista/survey/__init__.py`.

### 7.1 Tasarım nesnesi

```python
def survey_design(
    data: pd.DataFrame,
    weight_col: str | None = None,
    id_col: str | None = None,       # PSU (birincil örnekleme birimi)
    strata_col: str | None = None,
    fpc_col: str | None = None,
) -> dict    # "data", "weight_col", "id_col", "strata_col", "fpc_col", "n_psu", "n_strata"
```

### 7.2 Taylor linearizasyonu

Çok aşamalı, tabakalı tasarım için varyans genel formu (PSU toplamları
üzerinden, tabaka içi `n_h/(n_h-1)` düzeltmesi):

- `svy_mean(design, var) -> dict`: ağırlıklı ortalama, Taylor SE, %95 CI,
  tasarım etkisi (DEFF = `var_svy / var_srs_ağırlıklı`).
- `svy_total(design, var) -> dict`: toplam tahmini + SE + CI.
- `svy_ratio(design, numerator, denominator) -> dict`: oran tahmini;
  linearize değişken `u_i = y_i - R·x_i` üzerinden varyans.
- Tek tabakada tek PSU → varyans hesaplanamaz → `ValueError`
  ("Tek PSU'lu tabakada Taylor varyansı hesaplanamaz").

### 7.3 `survey_logistic`

```python
def survey_logistic(
    design: dict,
    response: str,
    predictors: list[str],
) -> dict
```

Ağırlıklı logit (statsmodels GLM binomial, `freq_weights` yuvarlanmış
ağırlık DEĞİL — `var_weights` ile) + PSU-kümelenmiş sandwich kovaryans
(skormatrisi toplamı tabaka/PSU düzeyinde). Çıktı: katsayılar, robust SE,
Wald z/p, tasarım etkileri.

### 7.4 Dönüş şemaları (mean/total/ratio ortak)

```python
{
  "estimate": float, "std_err": float,
  "ci_lower": float, "ci_upper": float,
  "design_effect": float | None,   # yalnız mean/total
  "n_obs": int, "n_psu": int, "n_strata": int,
}
```

## 8. CLI ve Menü Entegrasyonu

### 8.1 CLI komutları (`agrista/cli/__init__.py`)

| Komut | Seçenekler |
|---|---|
| `agrista glm FILE` | `--yanit` (zorunlu), `--faktorler` (zorunlu, virgül ayraçlı), `--kovaryeteler`, `--tip {1,2,3}` varsayılan 3, `--posthoc {tukey,duncan,yok}` varsayılan tukey, `--within` (tekrarlı ölçüm sütunları), `--denek` (denek sütunu; `--within` ile zorunlu) |
| `agrista gee FILE` | `--yanit`, `--degiskenler`, `--grup`, `--aile {gaussian,binomial,poisson,gamma}` varsayılan gaussian, `--yapi {independent,exchangeable,autoregressive}` varsayılan independent, `--zaman` |
| `agrista glmm FILE` | `--yanit`, `--sabitler`, `--grup`, `--aile {gaussian,binomial,poisson}` varsayılan gaussian, `--random-slope` |
| `agrista svymean FILE` | `--degisken`, `--agirlik`, `--psu`, `--tabaka`, `--fpc` |
| `agrista svyratio FILE` | `--pay`, `--payda`, tasarım seçenekleri yukarıdaki gibi |
| `agrista svylogit FILE` | `--yanit`, `--degiskenler`, tasarım seçenekleri yukarıdaki gibi |

Her komut için `_print_*_result` yazdırıcısı eklenir.

### 8.2 Menü güncellemeleri

- **[3] Ortalamaların Karşılaştırılması** → yeni alt öğeler:
  "Genel Doğrusal Model (Tek Değişkenli)", "Tekrarlı Ölçümler (GLM)".
- **[8] Regresyon** → yeni alt öğeler: "GEE (Genelleştirilmiş Tahmin
  Denklemleri)", "GLMM (Genelleştirilmiş Karışık Model)".
- **[20] Karmaşık Örneklem** → YENİ kategori: "Anket Ortalaması/Toplamı",
  "Anket Oranı", "Anket Lojistik Regresyonu". Kategori sayısı 19 → 20 olur;
  README ve `docs/02` log'u buna göre güncellenir.
- Tüm handler'lar `_prompt_or_eof` kullanır.

## 9. Test Stratejisi

- **TDD zorunlu:** her fonksiyon önce başarısız test ile başlar.
- Deterministik veri: `np.random.default_rng(seed)` + `conftest.make_sample_df`
  deseni; karışık/korelasyonlu veri gereken yerlerde sabit seed'li üreteçler.
- Doğrulama çıpaları: bilinen küçük veri setlerinde değerler elle/numpy ile
  bağımsız hesaplanır (örn. tek faktörlü GLM = `anova_one_way` sonucu ile
  birebir aynı F ve p); PQL sonuçları `animal.mixed_model` ile gaussian
  durumda çapraz doğrulanır.
- CLI: her yeni komut için en az bir `CliRunner` başarı + bir hata testi.
- Menü: `tests/test_menu_flows.py` içinde her yeni akış için smoke testi.
- Hedef: toplam kapsam %88 altına düşmez; yeni kod satır kapsamı ≥ %90.

## 10. Dokümantasyon ve Teslim

1. Her alt sistem bitince `docs/02` log'una "Güncelleme 5" bölümü:
   implementasyon listesi, CLI komutları, test dosyası kaydı.
2. README: yeni CLI komutları örnek blok + menü kategorisi listesi
   (19 → 20) + özellik listesi satırları güncellenir.
3. Sürüm: `pyproject.toml` ve CLI `version_option` **0.1.0 → 0.2.0**.
4. Tüm paket yeşilken teslim seçenekleri sunulur (birleştir / PR / dal koru).

## 11. Başarı Kriterleri

- [ ] `glm_univariate`, `glm_repeated_measures`, `gee_model`, `glmm`,
      `survey_design`, `svy_mean`, `svy_total`, `svy_ratio`,
      `survey_logistic` implementeleri ve testleri tamam.
- [ ] 9 yeni CLI komutu/alt komutu çalışıyor; menüde 3 yeni akış grubu.
- [ ] `pytest tests/` 0 hata; `flake8` 0 ihlal.
- [ ] `docs/02` log'unda "Kısmi kalanlar" notu kaldırılmış; README güncel.
- [ ] Versiyon 0.2.0; tek teslim commit'i (veya alt sistem başına commit).
