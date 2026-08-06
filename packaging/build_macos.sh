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
