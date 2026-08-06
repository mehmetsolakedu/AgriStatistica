# Agrista Windows build: exe + NSIS kurulumcu
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

# Yerel geliştirmede .venv, CI'da PATH'teki python kullanılır
if (Test-Path .venv\Scripts\python.exe) {
    $py = ".venv\Scripts\python.exe"
} else {
    $py = "python"
}

$surum = & $py -c "import agrista; print(agrista.__version__)"
Write-Host "== Agrista $surum Windows build (python: $py) =="

& $py -m PyInstaller packaging\agrista-gui.spec --noconfirm --clean

makensis packaging\agrista.nsi
Move-Item dist\Agrista-Setup.exe "dist\Agrista-$surum-Setup.exe" -Force
Write-Host "== Tamam: dist\Agrista-$surum-Setup.exe =="
