# Ledger — Masaüstü Dağıtım Wave 1

Planlar:
- docs/superpowers/plans/2026-08-06-masaustu-1-iskelet.md
- docs/superpowers/plans/2026-08-06-masaustu-2-analiz-kosucu.md
- docs/superpowers/plans/2026-08-06-masaustu-3-grafik-paneli.md
- docs/superpowers/plans/2026-08-06-masaustu-4-paketleme.md
- docs/superpowers/plans/2026-08-06-masaustu-5-guncelleyici-kapanis.md

Spec: docs/superpowers/specs/2026-08-06-masaustu-dagitim-design.md
Dal: feat/masaustu-dagitim
Yöntem: alt-ajan güdümlü, iki aşamalı inceleme (spec uyumu → kod kalitesi)

## Durum

| Görev | Plan | Durum | Commit | Not |
|---|---|---|---|---|
| P1 T1: pyproject + tema + DataFrameModel | 1-iskelet | complete | 938b8f0 | PySide6 6.10.3 |
| P1 T2: kayıt şeması + format_result | 1-iskelet | complete | 16027c5 | |
| P1 T3: MainWindow + menü + veri açma | 1-iskelet | complete | aa16257 | QMenu GC düzeltmesi |
| P1 T4: CI gui job'ı | 1-iskelet | complete | ba97560 | offscreen job |
| P2 T1: adaptörler + REGISTRY (16) | 2-analiz-kosucu | complete | 4bac155 | imzalar doğrulandı |
| P2 T2: AnalysisDialog | 2-analiz-kosucu | complete | 17a1d23 | |
| P2 T3: pencere entegrasyonu | 2-analiz-kosucu | complete | 630d7f8 | eski Plan-1 testi güncellendi |
| P3 T1: ChartPanel (12 tip) | 3-grafik-paneli | complete | 8fa27e8 | teardown close eklendi |
| P3 T2: pencere entegrasyonu | 3-grafik-paneli | complete | b93c659 | placeholder kaldırıldı |
| P4 T1: make_latest_json | 4-paketleme | complete | c160ad4 | 3 test |
| P4 T2: PyInstaller spec + macOS betiği | 4-paketleme | complete | 1c503cd | yerel build OK: app + 110 MB DMG |
| P4 T3: Windows + NSIS + release.yml | 4-paketleme | complete | 52014e3 | YAML geçerli |
| P5 T1: güncelleyici saf fonksiyonlar | 5-guncelleyici | complete | 910e34f | 8 test, ağ mock'lu |
| P5 T2: menü denetim akışı | 5-guncelleyici | complete | e64bd90 | |
| P5 T3: teslim kapanışı v0.4.0 | 5-guncelleyici | complete | 8456250 | README + docs/02 Güncelleme 7 |

## İnceleme Kayıtları

(her görev sonrası: spec uyumu + kod kalitesi bulguları)

### Plan 1 (938b8f0, 16027c5, aa16257, ba97560)
- Spec uyumu: GUI iskeleti, DataFrameModel, tema, menü denkliği tamam. ✅
- Kod kalitesi: PySide6 QMenu GC sorunu açık parent + referans tutma ile çözülmüş (kritik ve doğru); conftest __future__ sıralama zorunluluğu; CLI'nin '📁 Dosya' kategorisi GUI menüsünde de görünüyor (denklik gereği, küçük kozmetik çoğulluk). ✅
- Taze kanıt: TAM PAKET 542 passed; flake8 temiz.

### Plan 2 (4bac155, 17a1d23, 630d7f8)
- Spec uyumu: 16 analiz kayıtta, menü denklik testi yeşil, diyalog + uçtan uca akış tamam. ✅
- Kod kalitesi: KeyError→ValueError doğrulama yardımcısı gerekli sapma; KAT_B/KAT_O adlandırması E741 için; eski boş-kayıt testi spec kısıtı 4'e göre güncellendi. ✅
- Taze kanıt: TAM PAKET 563 passed; flake8 temiz.

### Plan 3 (8fa27e8, b93c659)
- Spec uyumu: 12 grafik tipi, gömülü canvas, kaydetme, veri senkronu tamam. ✅
- Kod kalitesi: figür sızıntısı için fixture teardown; kullanılmayan import'lar temizlendi. ✅
- Taze kanıt: TAM PAKET 573 passed; flake8 temiz.

### Plan 4 (c160ad4, 1c503cd, 52014e3)
- Spec uyumu: latest.json üretici testli; macOS build yerelde DOĞRULANDI (dist/Agrista.app ad-hoc imzalı + Agrista-0.3.0-macOS.dmg ~110 MB); Windows/NSIS/release.yml hazır. ✅
- Kod kalitesi: PyInstaller 6.x spec-yolu kök nedeni doğru çözüldü (../ öneki). ✅
- Taze kanıt: TAM PAKET 576 passed; flake8 temiz.

### Plan 5 (910e34f, e64bd90, 8456250)
- Spec uyumu: güncelleyici (saf + ağ katmanı), menü akışı, v0.4.0 dört yerde, README + docs/02 Güncelleme 7 tamam. ✅
- Kod kalitesi: kritik bulgu yok. ✅
- Taze kanıt: TAM PAKET 586 passed; flake8 temiz; `agrista --version` 0.4.0.

### Aşama 6 doğrulama kapısı (bağımsız)
- 586 passed; flake8 agrista tests packaging temiz; sürüm 0.4.0.
- GUI offscreen açıldı: 23 menü (Dosya + Görünüm + 21 kategori), 16 bağlı analiz, 12 grafik tipi.
- dist/Agrista-0.3.0-macOS.dmg (~110 MB) mevcut (sürüm artışı önce build; release CI v0.4.0 DMG üretecek).
- Eski ad taraması: 0 eşleşme.

## SONUÇ
Tüm görevler complete. Masaüstü Dağıtım Wave 1 (v0.4.0) teslim edildi.

## Dağıtım Günlüğü (yayın)
- PR'lar #1, #2, #3 main'e birleştirildi; depo PUBLIC yapıldı.
- v0.4.0 etiketi main'de; release elle açıldı: dist/Agrista-0.4.0-macOS.dmg
  (yerel build, ad-hoc imzalı, hdiutil verify VALID, uygulama build'den
  çalıştırılıp doğrulandı) + latest.json.
- Güncelleme kanalı anonim kanıtlandı: releases/latest/download/latest.json
  HTTP 200; `check_update('0.3.0')` → 0.4.0 algıladı; DMG HTTP 200.
- AÇIK KALEM: Windows kurulumcusu (release.yml matrix) GitHub Actions
  olay işleme hattı public geçişi sonrası yanıt vermiyor (release.yml
  indekslenmedi, ~2 saat). Hat toparlanınca etiket/elle tetikleme ile
  exe aynı release'e eklenecek.

## DAĞITIM TAMAMLANDI (final)
- Actions hattı toparlandı; CI hataları kök nedenleriyle giderildi:
  pytest-qt dev→gui ekstrasi, importorskip korumaları, libegl1,
  build betikleri .venv varsayımı, makensis yol bulma, NSIS DIST_DIR
  mutlak yol, Section "Uninstall", notarizasyon sırası, pandas 3
  StringDtype uyumu (is_numeric_dtype).
- Release koşusu 31128846583 TAM YEŞİL: macOS + Windows + release job.
- Release v0.4.0 varlıkları: Agrista-0.4.0-macOS.dmg (116 MB),
  Agrista-0.4.0-Setup.exe (139 MB), latest.json (424 bayt, doğru not).
- Anonim kanal: releases/latest/download/latest.json HTTP 200;
  check_update('0.3.0') → 0.4.0 algılıyor, her iki platform URL'si
  HTTP 200. DMG hdiutil verify VALID; .app build'den çalıştırıldı.
- CI main'de TAM YEŞİL (test 3.9 + test 3.11 + gui), run 31129416759.
- SONUÇ: macOS/Windows kurulum paketleri yayında, güncelleme kanalı
  çalışıyor; dağıtım hedefi tamamlandı.
