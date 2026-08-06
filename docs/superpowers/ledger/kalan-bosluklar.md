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
| GLM Task 4: Menü + smoke + log | glm.md | pending | | |
| GEE Task 1: gee_model | gee.md | pending | | |
| GEE Task 2: CLI `agrista gee` | gee.md | pending | | |
| GEE Task 3: Menü + smoke + log | gee.md | pending | | |
| GLMM Task 1: glmm (PQL) | glmm.md | pending | | |
| GLMM Task 2: CLI `agrista glmm` | glmm.md | pending | | |
| GLMM Task 3: Menü + smoke + log | glmm.md | pending | | |
| Survey Task 1: modül + Taylor çekirdeği | karmasik-anket.md | pending | | |
| Survey Task 2: CLI svymean/svyratio/svylogit | karmasik-anket.md | pending | | |
| Survey Task 3: Menü [20] + smoke + log | karmasik-anket.md | pending | | |
| Survey Task 4: Teslim kapanışı (README, v0.2.0) | karmasik-anket.md | pending | | |

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
