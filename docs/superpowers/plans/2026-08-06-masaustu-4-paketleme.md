# Plan: Masaüstü Wave 1 — Paketleme Hattı (PyInstaller + DMG/NSIS + Release)

**Spec:** `docs/superpowers/specs/2026-08-06-masaustu-dagitim-design.md` (§8)

**Goal:** macOS ve Windows için tekrarlanabilir build hattı kurmak:
PyInstaller spesifikasyonu, platform build betikleri, etiket tetiklemeli
GitHub Actions release workflow'u ve test edilebilir `latest.json` üretici.

**Architecture:** `packaging/` klasörü: `agrista-gui.spec` (PyInstaller),
`build_macos.sh` (app + ad-hoc/imzalı codesign + DMG), `build_windows.ps1`
(exe + NSIS), `agrista.nsi`, `make_latest_json.py` (saf, testli).
`.github/workflows/release.yml` etiket bazlı matrix build yapar.

**Tech Stack:** PyInstaller (build bağımlılığı, CI'da kurulur), hdiutil,
codesign, NSIS (`makensis`), GitHub Actions. Ön koşul: Plan 1-3 tamamlanmış.

**Global Constraints (spec'ten aynen):**
1. PySide6 opsiyonel ekstra; build araçları yalnız CI/build ortamına girer.
2. Yalnızca "Premium Program" adı; eski ad yasak.
3. GUI Türkçe; betiklerdeki kullanıcıya dönük metinler Türkçe.
4. İmza: macOS ad-hoc varsayılan; `MACOS_SIGNING_IDENTITY` secret'ı
   tanımlıysa gerçek imza + notarizasyon adımı etkinleşir (betikte koşul).
5. Test: TDD zorunlu; `make_latest_json.py` saf fonksiyonları birim testli;
   build betikleri yerelde (macOS) çalıştırılarak doğrulanır.
6. Sürüm bu planda değişmez.
7. `pytest` tam paket yeşil + `flake8` temiz olmadan görev bitmez.

## Dosya Yapısı

| Dosya | Sorumluluk |
|---|---|
| `packaging/agrista-gui.spec` (yeni) | PyInstaller build tanımı |
| `packaging/build_macos.sh` (yeni) | macOS .app + codesign + DMG |
| `packaging/build_windows.ps1` (yeni) | Windows exe + NSIS |
| `packaging/agrista.nsi` (yeni) | NSIS kurulum betiği |
| `packaging/make_latest_json.py` (yeni) | `latest.json` üretici (saf) |
| `tests/test_packaging.py` (yeni) | latest.json + sürüm okuma testleri |
| `.github/workflows/release.yml` (yeni) | etiket tetiklemeli release |

---

## Task 1: make_latest_json (TDD)

**Files:** Test: `tests/test_packaging.py` (Create) · Create: `packaging/make_latest_json.py`
**Interfaces:** `build_latest_payload(version, notes, macos_url, windows_url) -> dict`; `main(argv)`.

- [ ] **RED** — `tests/test_packaging.py` oluştur:

```python
"""Paketleme hattı testleri (latest.json üretimi)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "packaging"))

from make_latest_json import build_latest_payload


class TestLatestPayload:
    def test_sema(self):
        p = build_latest_payload("0.4.0", "ilk sürüm",
                                 "https://x/Agrista.dmg",
                                 "https://x/Agrista-Setup.exe")
        assert p["version"] == "0.4.0"
        assert p["assets"]["macos"].endswith(".dmg")
        assert p["assets"]["windows"].endswith(".exe")
        assert p["notes"] == "ilk sürüm"

    def test_gecersiz_sürüm(self):
        import pytest
        with pytest.raises(ValueError):
            build_latest_payload("abc", "", "u1", "u2")

    def test_json_serileştirilebilir(self):
        p = build_latest_payload("0.4.0", "", "u1", "u2")
        assert json.loads(json.dumps(p))["version"] == "0.4.0"
```

- [ ] Çalıştır: `.venv/bin/python -m pytest tests/test_packaging.py -x -q`
      → ModuleNotFoundError.
- [ ] **GREEN** — `packaging/make_latest_json.py` oluştur:

```python
"""Release `latest.json` üretici — güncelleme denetiminin veri kaynağı."""

from __future__ import annotations

import argparse
import json
import re

SURUM_RE = re.compile(r"^\d+\.\d+\.\d+$")


def build_latest_payload(version: str, notes: str, macos_url: str,
                         windows_url: str) -> dict:
    """latest.json içeriğini üretir (saf, doğrulamalı)."""
    if not SURUM_RE.match(version):
        raise ValueError(f"Geçersiz sürüm: {version}")
    return {
        "version": version,
        "notes": notes,
        "assets": {"macos": macos_url, "windows": windows_url},
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="latest.json üret")
    ap.add_argument("--version", required=True)
    ap.add_argument("--notes", default="")
    ap.add_argument("--macos-url", required=True)
    ap.add_argument("--windows-url", required=True)
    ap.add_argument("--out", default="latest.json")
    a = ap.parse_args(argv)
    payload = build_latest_payload(a.version, a.notes, a.macos_url,
                                   a.windows_url)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"Yazıldı: {a.out}")


if __name__ == "__main__":
    main()
```

- [ ] Çalıştır → yeşil. Commit: `feat(packaging): latest.json üretici`

## Task 2: PyInstaller spec + macOS betiği + yerel doğrulama

**Files:** Create: `packaging/agrista-gui.spec`, `packaging/build_macos.sh`
**Interfaces:** `packaging/build_macos.sh` → `dist/Agrista.app` + `dist/Agrista-<sürüm>-macOS.dmg`.

- [ ] `packaging/agrista-gui.spec` oluştur:

```python
# -*- mode: python ; coding: utf-8 -*-
"""Agrista GUI PyInstaller spesifikasyonu (onedir, pencere modu)."""
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

a = Analysis(
    ["agrista/gui/main.py"],
    pathex=["."],
    hiddenimports=collect_submodules("agrista"),
    datas=collect_data_files("matplotlib"),
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True,
    name="Agrista", console=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="Agrista")

if sys.platform == "darwin":
    BUNDLE(
        coll, name="Agrista.app",
        bundle_identifier="com.agrista.desktop",
        info_plist={"CFBundleShortVersionString": "0.4.0",
                    "NSHighResolutionCapable": True},
    )
```

- [ ] `packaging/build_macos.sh` oluştur:

```bash
#!/usr/bin/env bash
# Agrista macOS build: .app + codesign + DMG
set -euo pipefail
cd "$(dirname "$0")/.."

SURUM=$(.venv/bin/python -c "import agrista; print(agrista.__version__)")
echo "== Agrista ${SURUM} macOS build =="

.venv/bin/python -m PyInstaller packaging/agrista-gui.spec --noconfirm --clean

KIMLIK="${MACOS_SIGNING_IDENTITY:--}"   # '-' = ad-hoc
codesign --force --deep --sign "$KIMLIK" "dist/Agrista.app"

if [ -n "${MACOS_NOTARY_KEYCHAIN_PROFILE:-}" ]; then
  xcrun notarytool submit "dist/Agrista-${SURUM}-macOS.dmg" \
    --keychain-profile "$MACOS_NOTARY_KEYCHAIN_PROFILE" --wait || true
fi

hdiutil create -volname "Agrista ${SURUM}" \
  -srcfolder "dist/Agrista.app" -ov -format UDZO \
  "dist/Agrista-${SURUM}-macOS.dmg"
echo "== Tamam: dist/Agrista-${SURUM}-macOS.dmg =="
```

- [ ] PyInstaller kurulumu: `.venv/bin/pip install pyinstaller -q`.
- [ ] Yerel doğrulama: `bash packaging/build_macos.sh` → çıkış 0;
      `dist/Agrista.app` var; DMG var (`ls -la dist/` kanıtını not et).
      (Build uzun sürer; hata çıkarsa kök neden spec hiddenimports/datas
      ayarlarında aranır, 5 deneme sınırına uy.)
- [ ] Commit: `feat(packaging): PyInstaller spec + macOS build betiği`

## Task 3: Windows betiği + NSIS + release workflow

**Files:** Create: `packaging/build_windows.ps1`, `packaging/agrista.nsi`,
`.github/workflows/release.yml`
**Interfaces:** workflow: `v*` etiketi veya `workflow_dispatch` → 2 build job + latest job.

- [ ] `packaging/agrista.nsi` oluştur:

```nsis
!include "MUI2.nsh"
Name "Agrista"
OutFile "dist\Agrista-Setup.exe"
InstallDir "$PROGRAMFILES64\Agrista"
RequestExecutionLevel admin

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_LANGUAGE "Turkish"

Section "Kurulum"
  SetOutPath "$INSTDIR"
  File /r "dist\Agrista\*.*"
  CreateShortcut "$SMPROGRAMS\Agrista.lnk" "$INSTDIR\Agrista.exe"
  CreateShortcut "$DESKTOP\Agrista.lnk" "$INSTDIR\Agrista.exe"
  WriteUninstaller "$INSTDIR\Kaldir.exe"
SectionEnd

Section "Kaldırıcı"
  Delete "$INSTDIR\Kaldir.exe"
  RMDir /r "$INSTDIR"
SectionEnd
```

- [ ] `packaging/build_windows.ps1` oluştur:

```powershell
# Agrista Windows build: exe + NSIS kurulumcu
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$surum = & .venv\Scripts\python.exe -c "import agrista; print(agrista.__version__)"
Write-Host "== Agrista $surum Windows build =="

& .venv\Scripts\python.exe -m PyInstaller packaging\agrista-gui.spec --noconfirm --clean

makensis packaging\agrista.nsi
Move-Item dist\Agrista-Setup.exe "dist\Agrista-$surum-Setup.exe" -Force
Write-Host "== Tamam: dist\Agrista-$surum-Setup.exe =="
```

- [ ] `.github/workflows/release.yml` oluştur:

```yaml
name: Release

on:
  push:
    tags: ["v*"]
  workflow_dispatch:

jobs:
  build:
    strategy:
      fail-fast: false
      matrix:
        include:
          - os: macos-latest
            script: bash packaging/build_macos.sh
            artifact: "dist/Agrista-*.dmg"
          - os: windows-latest
            setup: choco install nsis -y
            script: powershell -File packaging/build_windows.ps1
            artifact: "dist/Agrista-*-Setup.exe"
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install package + gui + build tools
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[gui]" pyinstaller
      - name: Platform setup
        if: ${{ matrix.setup != '' }}
        run: ${{ matrix.setup }}
      - name: Build
        env:
          MACOS_SIGNING_IDENTITY: ${{ secrets.MACOS_SIGNING_IDENTITY }}
          MACOS_NOTARY_KEYCHAIN_PROFILE: ${{ secrets.MACOS_NOTARY_PROFILE }}
        run: ${{ matrix.script }}
      - uses: actions/upload-artifact@v4
        with:
          name: installer-${{ matrix.os }}
          path: ${{ matrix.artifact }}

  release:
    needs: build
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4
      - uses: actions/download-artifact@v4
        with:
          path: varliklar
      - name: Flatten artifacts
        run: |
          mkdir -p release_assets
          find varliklar -type f \( -name "*.dmg" -o -name "*.exe" \) \
            -exec mv {} release_assets/ \;
          ls -la release_assets
      - name: Build latest.json
        run: |
          SURUM=${GITHUB_REF_NAME#v}
          TABAN="https://github.com/${{ github.repository }}/releases/download/${GITHUB_REF_NAME}"
          DMG=$(ls release_assets | grep '\.dmg$' | head -1)
          EXE=$(ls release_assets | grep '\.exe$' | head -1)
          python packaging/make_latest_json.py \
            --version "$SURUM" --notes "${GITHUB_REF_NAME} sürümü" \
            --macos-url "$TABAN/$DMG" --windows-url "$TABAN/$EXE" \
            --out release_assets/latest.json
      - uses: softprops/action-gh-release@v2
        with:
          files: release_assets/*
```

- [ ] Workflow sözdizimi: `python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/release.yml'))"` → hata yok (pyyaml kurulu değilse `.venv/bin/pip install pyyaml -q`).
- [ ] Tam doğrulama: `.venv/bin/python -m pytest tests/ -q` ve
      `.venv/bin/python -m flake8 agrista tests packaging` → temiz
      (`packaging/make_latest_json.py` flake8 kapsamına girer).
- [ ] Commit: `feat(packaging): Windows build + NSIS + release workflow`
