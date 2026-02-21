# Cenario installer for Windows
# Creates a self-contained installation with venv, app code, model cache, and launchers.

param(
    [string]$InstallDir = "$env:USERPROFILE\cenario",
    [switch]$CpuOnly,
    [switch]$Help
)

$ErrorActionPreference = "Stop"
$RepoDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

function Info($msg)  { Write-Host "[INFO]  $msg" -ForegroundColor Cyan }
function Warn($msg)  { Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function Fatal($msg) { Write-Host "[ERROR] $msg" -ForegroundColor Red; exit 1 }

if ($Help) {
    Write-Host "Usage: .\install.ps1 [-InstallDir <path>] [-CpuOnly]"
    Write-Host ""
    Write-Host "  -InstallDir <path>  Installation directory (default: ~\cenario)"
    Write-Host "  -CpuOnly            Skip CUDA detection, install CPU-only PyTorch"
    exit 0
}

# ---- Check Python ----
$Python = $null
foreach ($cmd in @("python3", "python")) {
    try {
        $ver = & $cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($ver) {
            $parts = $ver.Split(".")
            if ([int]$parts[0] -eq 3 -and [int]$parts[1] -ge 10) {
                $Python = $cmd
                break
            }
        }
    } catch {}
}
if (-not $Python) { Fatal "Python >= 3.10 not found. Install Python 3.10+ from python.org and try again." }
$pyVer = & $Python --version
Info "Using Python: $Python ($pyVer)"

# ---- Detect CUDA ----
$TorchIndex = "cpu"
if (-not $CpuOnly) {
    try {
        $nvOut = & nvidia-smi 2>$null
        if ($nvOut) {
            $match = [regex]::Match(($nvOut | Out-String), 'CUDA Version:\s*(\d+)\.(\d+)')
            if ($match.Success) {
                $cudaMajor = [int]$match.Groups[1].Value
                $cudaMinor = [int]$match.Groups[2].Value
                $cudaVer = "$cudaMajor.$cudaMinor"
                if ($cudaMajor -gt 12 -or ($cudaMajor -eq 12 -and $cudaMinor -ge 4)) {
                    $TorchIndex = "cu124"
                } elseif ($cudaMajor -eq 12 -and $cudaMinor -ge 1) {
                    $TorchIndex = "cu121"
                } elseif ($cudaMajor -eq 11 -and $cudaMinor -ge 8) {
                    $TorchIndex = "cu118"
                } else {
                    Warn "CUDA $cudaVer detected but too old for GPU PyTorch. Falling back to CPU."
                }
                if ($TorchIndex -ne "cpu") { Info "Detected CUDA $cudaVer -> using PyTorch index: $TorchIndex" }
            }
        }
    } catch {
        Info "No NVIDIA GPU detected. Installing CPU-only PyTorch."
    }
} else {
    Info "CPU-only mode requested."
}

# ---- bitsandbytes warning ----
if ($TorchIndex -ne "cpu") {
    Warn "bitsandbytes has limited Windows support. 4-bit LLM quantization may not work."
    Warn "Summarization will still work with 8-bit or no quantization."
}

# ---- Create directory structure ----
Info "Installing to: $InstallDir"
foreach ($sub in @("app", "models", "data")) {
    $p = Join-Path $InstallDir $sub
    if (-not (Test-Path $p)) { New-Item -ItemType Directory -Path $p -Force | Out-Null }
}

# ---- Copy app source ----
Info "Copying application source..."
foreach ($item in @("cenario.py", "updater.py")) {
    $dest = Join-Path (Join-Path $InstallDir "app") $item
    Copy-Item (Join-Path $RepoDir $item) -Destination $dest -Force
}
foreach ($dir in @("core", "gui")) {
    $dest = Join-Path (Join-Path $InstallDir "app") $dir
    if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
    Copy-Item (Join-Path $RepoDir $dir) -Destination $dest -Recurse -Force
}

# Remove __pycache__ from copied source
Get-ChildItem -Path (Join-Path $InstallDir "app") -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# ---- Create / update venv ----
$VenvDir = Join-Path $InstallDir "venv"
if (-not (Test-Path $VenvDir)) {
    Info "Creating virtual environment..."
    & $Python -m venv $VenvDir
} else {
    Info "Virtual environment already exists, reusing."
}

$Pip = Join-Path (Join-Path $VenvDir "Scripts") "pip.exe"
$PythonVenv = Join-Path (Join-Path $VenvDir "Scripts") "python.exe"

Info "Upgrading pip..."
& $PythonVenv -m pip install --upgrade pip --quiet

# ---- Install PyTorch ----
Info "Installing PyTorch ($TorchIndex)..."
& $Pip install torch torchaudio --index-url "https://download.pytorch.org/whl/$TorchIndex" --quiet

# ---- Install remaining dependencies ----
Info "Installing dependencies..."
& $Pip install -r (Join-Path (Join-Path $RepoDir "installer") "requirements-pip.txt") --quiet

# ---- Install bitsandbytes ----
Info "Installing bitsandbytes..."
& $Pip install bitsandbytes==0.49.2 --quiet

# ---- Generate launcher scripts ----
Info "Creating launcher scripts..."

# GUI launcher (.bat)
$guiBat = @(
    '@echo off',
    'set "SCRIPT_DIR=%~dp0"',
    'set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"',
    'set "HF_HOME=%SCRIPT_DIR%\models"',
    'set "CENARIO_DATA_DIR=%SCRIPT_DIR%\data"',
    '"%SCRIPT_DIR%\venv\Scripts\python.exe" "%SCRIPT_DIR%\app\gui\app.py" %*'
) -join "`r`n"
Set-Content -Path (Join-Path $InstallDir "cenario-gui.bat") -Value $guiBat -Encoding ASCII

# CLI launcher (.bat)
$cliBat = @(
    '@echo off',
    'set "SCRIPT_DIR=%~dp0"',
    'set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"',
    'set "HF_HOME=%SCRIPT_DIR%\models"',
    'set "CENARIO_DATA_DIR=%SCRIPT_DIR%\data"',
    '"%SCRIPT_DIR%\venv\Scripts\python.exe" "%SCRIPT_DIR%\app\cenario.py" %*'
) -join "`r`n"
Set-Content -Path (Join-Path $InstallDir "cenario.bat") -Value $cliBat -Encoding ASCII

# ---- Create .env template if not present ----
$envFile = Join-Path $InstallDir ".env"
if (-not (Test-Path $envFile)) {
    $envContent = @(
        '# Cenario configuration',
        '# Uncomment and set your HuggingFace token to enable speaker diarization:',
        '# HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
    ) -join "`r`n"
    Set-Content -Path $envFile -Value $envContent -Encoding UTF8
    Info "Created .env template at $envFile"
}

# ---- Done ----
Write-Host ""
Info "Installation complete!"
Write-Host ""
Write-Host "  Install directory:  $InstallDir"
Write-Host "  PyTorch variant:    $TorchIndex"
Write-Host ""
Write-Host "  Start the GUI:      $InstallDir\cenario-gui.bat"
Write-Host "  Use the CLI:        $InstallDir\cenario.bat --help"
Write-Host ""
Write-Host "  To enable speaker diarization, add your HuggingFace token to:"
Write-Host "    $envFile"
Write-Host ""
