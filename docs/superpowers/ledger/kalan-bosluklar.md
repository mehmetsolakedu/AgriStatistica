# Ledger — Premium Program Kalan Boşluklar (GLM, GEE, GLMM, Karmaşık Anket)

Planlar:
- docs/superpowers/plans/2026-08-06-glm.md
- docs/superpowers/plans/2026-08-06-gee.md
- docs/superpowers/plans/2026-08-06-glmm.md
- docs/superpowers/plans/2026-08-06-karmasik-anket.md

Spec: docs/superpowers/specs/2026-08-06-kalan-bosluklar-design.md
Dal: feat/premium-kalan-bosluklar
Yöntem: alt-ajan güdümlü, iki aşamalı inceleme (spec uyumu → kod kalitesi)

## Durum

| Görev | Plan | Durum | Commit | Not |
|---|---|---|---|---|
| GLM Task 1: glm_univariate | glm.md | complete | 6c3bd14 | Sapma: anova tablosunda Intercept satırı da atlanıyor (gerekli) |
| GLM Task 2: glm_repeated_measures | glm.md | complete | 4feb408 | Plan kodu olduğu gibi geçti; import satırı birleştirildi |
| GLM Task 3: CLI `agrista glm` | glm.md | complete | c8d3100 | `_load_file(...).dataframe`, `--tip` default str, testler sınıf idiyomuna uyarlandı |
| GLM Task 4: Menü + smoke + log | glm.md | complete | 4a609ec | FLOWS formatı (başlık, başlık, inputs, expect); rm.csv eklendi; 56 akış |
| GEE Task 1: gee_model | gee.md | complete | ba7bcb8 | cov_struct örnekleri (statsmodels 0.14.6), AR grid=True |
| GEE Task 2: CLI `agrista gee` | gee.md | complete | 0ab22df | TestGeeCli sınıf idiyomu |
| GEE Task 3: Menü + smoke + log | gee.md | complete | d857396 | 57 akış; docs/02 kısmi kalanlar güncellendi |
| GLMM Task 1: glmm (PQL) | glmm.md | complete | 7173033 | `_weighted_lmm` profil-REML (freq_weights yok); AIC yedeği |
| GLMM Task 2: CLI `agrista glmm` | glmm.md | complete | 7db5cc9 | TestGlmmCli sınıf idiyomu |
| GLMM Task 3: Menü + smoke + log | glmm.md | complete | 53d3335 | 58 akış; kısmi kalanlar yalnız survey |
| Survey Task 1: modül + Taylor çekirdeği | karmasik-anket.md | complete | e805b20 | Elle hesaplı testler ilk denemede yeşil |
| Survey Task 2: CLI svymean/svyratio/svylogit | karmasik-anket.md | complete | a71173f | TestSurveyCli sınıf idiyomu |
| Survey Task 3: Menü [20] + smoke + log | karmasik-anket.md | complete | 2c02556 | 61 akış; kısmi kalan: yok |
| Survey Task 4: Teslim kapanışı (README, v0.2.0) | karmasik-anket.md | complete | 769fc11 | __version__/banner düzeltmesi ayrıca eklendi |

## İnceleme Kayıtları

(her görev sonrası: spec uyumu + kod kalitesi bulguları)

### GLM Task 1 (6c3bd14)
- Spec uyumu: dönüş şeması §4.1 ile birebir; Intercept satırı atlaması gerekli sapma. ✅
- Kod kalitesi: mevcut fonksiyon-içi import ve dict şema deseniyle tutarlı; kritik bulgu yok. ✅
- Taze kanıt: 7 passed; flake8 temiz.

### GLM Task 2 (4feb408)
- Spec uyumu: dönüş şeması §4.2 ile birebir (within/mauchly/epsilon/corrected/between/n_subjects). ✅
- Kod kalitesi: Mauchly/HF formülleri clip korumalı; kritik bulgu yok. ✅
- Taze kanıt: 12 passed (toplam GLM); regresyon 28 passed; flake8 temiz.

### GLM Task 3 (c8d3100)
- Spec uyumu: CLI seçenekleri §8.1 ile uyumlu; `glm --help` çalışıyor. ✅
- Kod kalitesi: mevcut komut deseniyle tutarlı; `.dataframe` düzeltmesi gerekli (diğer komutlarda da böyle). ✅
- Taze kanıt: 3 glm testi + 48 toplam viz/cli testi geçti; flake8 temiz.
- ÇAPRAZ NOT: Task 4 menü handler'larında da `_load_file(path).dataframe` kullanılmalı.

### GLM Task 4 (4a609ec) + GLM plan kapanışı (a2507f6)
- Spec uyumu: menü [3]'e 2 öğe, smoke akışları yeşil, docs/02 Güncelleme 5 GLM bölümü eklendi. ✅
- Kod kalitesi: handler'lar `_prompt_or_eof` korumalı, mevcut desenle tutarlı. ✅
- Taze kanıt: TAM PAKET 419 passed; flake8 temiz (GLM planı kapanış doğrulaması).

### GEE Task 1-3 (ba7bcb8, 0ab22df, d857396) + GEE plan kapanışı
- Spec uyumu: §5 şeması birebir (coefficients/qic/n_groups/n_obs/converged); menü [8], CLI, docs/02 GEE bölümü tamam. ✅
- Kod kalitesi: statsmodels 0.14.6 cov_struct örnek uyarlaması planın öngördüğü kapsamda; kritik bulgu yok. ✅
- Not: ilk alt ajan çağrısı ağ hatasıyla kesildi; çalışma ağacında kalan kod plan ile birebir doğrulandı. ✅
- Taze kanıt: TAM PAKET 430 passed; flake8 temiz.

### GLMM Task 1-3 (7173033, 7db5cc9, 53d3335) + GLMM plan kapanışı
- Spec uyumu: §6 şeması korunuyor; PQL yakınsama ve katsayı testleri plan toleranslarıyla yeşil. ✅
- Kod kalitesi: freq_weights engeli profil-REML çözücüyle aşılmış (Breslow & Clayton'a uygun); animal.mixed_model AIC NaN yedeği mevcut testlerle korunmuş (36 passed). Sapmalar gerekli ve belgelenmiş. ✅
- Taze kanıt: TAM PAKET 441 passed; flake8 temiz.

### Survey Task 1-4 (e805b20, a71173f, 2c02556, 769fc11) + Survey plan kapanışı
- Spec uyumu: §7 şemaları birebir; elle hesaplı Taylor değerleri tuttu; menü [20] (20 kategori), README, sürüm 0.2.0. ✅
- Kod kalitesi: kritik bulgu yok; `cluster` kovaryans 0.14.6'da sorunsuz. ✅
- Kapanış düzeltmeleri (ana inceleme): `__version__` ve menü banner'ı 0.2.0'a çekildi; spec/plan metinlerindeki eski paket adı kaldırıldı (0 eşleşme). ✅
- Taze kanıt (Aşama 6 doğrulama kapısı): TAM PAKET 459 passed; flake8 temiz; `agrista --version` → 0.2.0.

## SONUÇ
Tüm görevler complete. Premium Program Base denkliği: kısmi kalan yok.
