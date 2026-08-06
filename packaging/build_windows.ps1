# Agrista Windows build: exe + NSIS kurulumcu
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$surum = & .venv\Scripts\python.exe -c "import agrista; print(agrista.__version__)"
Write-Host "== Agrista $surum Windows build =="

& .venv\Scripts\python.exe -m PyInstaller packaging\agrista-gui.spec --noconfirm --clean

makensis packaging\agrista.nsi
Move-Item dist\Agrista-Setup.exe "dist\Agrista-$surum-Setup.exe" -Force
Write-Host "== Tamam: dist\Agrista-$surum-Setup.exe =="
