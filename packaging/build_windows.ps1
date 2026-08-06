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

$nsisCandidates = @(
    (Get-Command makensis -ErrorAction SilentlyContinue).Source,
    "$env:ProgramFiles\NSIS\makensis.exe",
    "${env:ProgramFiles(x86)}\NSIS\makensis.exe",
    "C:\ProgramData\chocolatey\bin\makensis.exe"
)
$makensis = $nsisCandidates | Where-Object { $_ -and (Test-Path $_) } |
    Select-Object -First 1
if (-not $makensis) {
    choco install nsis -y
    $makensis = "${env:ProgramFiles(x86)}\NSIS\makensis.exe"
    if (-not (Test-Path $makensis)) {
        $makensis = "$env:ProgramFiles\NSIS\makensis.exe"
    }
}
Write-Host "NSIS: $makensis"
Write-Host "dist içeriği:"
Get-ChildItem dist | ForEach-Object { Write-Host $_.Name }

$repoRoot = (Get-Location).Path
& $makensis "/DDIST_DIR=$repoRoot\dist" packaging\agrista.nsi
Move-Item dist\Agrista-Setup.exe "dist\Agrista-$surum-Setup.exe" -Force
Write-Host "== Tamam: dist\Agrista-$surum-Setup.exe =="
