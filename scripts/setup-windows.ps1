# ─────────────────────────────────────────────────────────────────────────────
# whisper-live · Windows setup
#
# This script:
#   1. Validates the environment (Windows 10+, Python 3.10–3.13, git)
#   2. Creates a temporary virtual-env for heavy conversion dependencies
#   3. Installs torch + openai-whisper + numpy (one-time, for .pt → ggml)
#   4. Runs scripts/setup.py (downloads from OpenAI Azure CDN only)
#   5. Tears down the temporary venv
#
# After setup completes, the runtime needs ZERO Python dependencies —
# it's pure whisper.cpp (C/C++).
#
# Everything is stored under <project_root>\local\ (gitignored).
#
# *** No HuggingFace URLs are ever contacted. ***
# ─────────────────────────────────────────────────────────────────────────────
#Requires -Version 5.1

[CmdletBinding()]
param(
    [ValidateSet(
        "tiny.en", "tiny",
        "base.en", "base",
        "small.en", "small",
        "medium.en", "medium",
        "large-v1", "large-v2", "large-v3", "large"
    )]
    [string]$Model = "tiny.en",

    [ValidateSet("q8_0", "q5_0", "q5_1", "q4_0", "q4_1", "")]
    [string]$Quantize = "",

    [switch]$ForceBuild,

    [switch]$Help
)

$ErrorActionPreference = "Stop"

# ── Colour helpers ───────────────────────────────────────────────────────────
function Write-Step  { param([string]$Msg) Write-Host "`n> $Msg" -ForegroundColor White }
function Write-Ok    { param([string]$Msg) Write-Host "  + $Msg" -ForegroundColor Green }
function Write-Warn  { param([string]$Msg) Write-Host "  ! $Msg" -ForegroundColor Yellow }
function Write-Err   { param([string]$Msg) Write-Host "  x $Msg" -ForegroundColor Red }

function Exit-Fatal {
    param([string]$Msg)
    Write-Err $Msg
    exit 1
}

# ── Help ─────────────────────────────────────────────────────────────────────
if ($Help) {
    @"

Usage: .\setup-windows.ps1 [OPTIONS]

Options:
  -Model NAME        Whisper model name (default: tiny.en)
                     Choices: tiny.en tiny base.en base small.en small
                              medium.en medium large-v1 large-v2 large-v3 large
  -Quantize TYPE     Quantization type (q8_0, q5_0, q5_1, q4_0, q4_1)
  -ForceBuild        Force building from source instead of downloading prebuilt
  -Help              Show this help and exit

Everything is installed under <project_root>\local\ (gitignored).

Examples:
  .\setup-windows.ps1
  .\setup-windows.ps1 -Model base.en -Quantize q8_0

"@
    exit 0
}

# ── Locate project root ─────────────────────────────────────────────────────
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$SetupPy     = Join-Path $ScriptDir "setup.py"

if (-not (Test-Path $SetupPy)) {
    Exit-Fatal "Cannot find setup.py at $SetupPy"
}

# ── Validate: OS ─────────────────────────────────────────────────────────────
Write-Step "Checking environment ..."

# Allow PowerShell Core on Windows
if ($PSVersionTable.PSEdition -eq "Core" -and -not $IsWindows) {
    Exit-Fatal "This script is for Windows only. Use setup-mac.sh on macOS/Linux."
}

$osVersion = [System.Environment]::OSVersion.Version
Write-Ok "Windows $($osVersion.Major).$($osVersion.Minor) (Build $($osVersion.Build))"

# ── Validate: git ────────────────────────────────────────────────────────────
$gitCmd = Get-Command git -ErrorAction SilentlyContinue
if (-not $gitCmd) {
    Exit-Fatal @"
git is not installed or not on PATH.
Install from: https://git-scm.com/download/win
Or via winget: winget install Git.Git
"@
}
$gitVersion = & git --version 2>$null
Write-Ok "git found: $gitVersion"

# ── Validate: Python 3.10-3.13 ──────────────────────────────────────────────
$PythonExe = $null
$MaxMinor = 13

# Try versioned pythons first (most compatible with torch), then generic
$candidates = @(
    "python3.12", "python3.13", "python3.11", "python3.10",
    "python", "python3"
)

foreach ($candidate in $candidates) {
    $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
    if ($cmd) {
        try {
            $pyVersionStr = & $candidate -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
            if ($pyVersionStr -match '^(\d+)\.(\d+)$') {
                $pyMajor = [int]$Matches[1]
                $pyMinor = [int]$Matches[2]
                if ($pyMajor -eq 3 -and $pyMinor -ge 10 -and $pyMinor -le $MaxMinor) {
                    $PythonExe = $candidate
                    break
                }
            }
        } catch {
            # Try next candidate
        }
    }
}

# Also try the Python launcher (py -3.12, py -3.13, etc.)
if (-not $PythonExe) {
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        foreach ($minor in @(12, 13, 11, 10)) {
            try {
                $pyVersionStr = & py "-3.$minor" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
                if ($pyVersionStr -match '^3\.(\d+)$') {
                    $detectedMinor = [int]$Matches[1]
                    if ($detectedMinor -ge 10 -and $detectedMinor -le $MaxMinor) {
                        $PythonExe = "py"
                        # We'll use py -3.$minor later
                        $PyLauncherVersion = "-3.$minor"
                        break
                    }
                }
            } catch {
                # Try next
            }
        }
    }
}

if (-not $PythonExe) {
    Exit-Fatal @"
Python 3.10-3.$MaxMinor is required but not found on PATH.
Install from: https://www.python.org/downloads/
Or via winget: winget install Python.Python.3.12
Make sure to check 'Add python.exe to PATH' during installation.
(Python 3.14+ is not yet supported by torch/openai-whisper)
"@
}

# Build the actual python command to use
if ($PyLauncherVersion) {
    $PythonCmd = @("py", $PyLauncherVersion)
    $pyFullVersion = & py $PyLauncherVersion --version 2>$null
} else {
    $PythonCmd = @($PythonExe)
    $pyFullVersion = & $PythonExe --version 2>$null
}
Write-Ok "Python: $pyFullVersion"

# ── Create temporary venv ────────────────────────────────────────────────────
Write-Step "Creating temporary Python venv for conversion dependencies ..."

$VenvDir = Join-Path $env:TEMP "whisper-setup-venv-$(Get-Random)"

function Remove-TempVenv {
    if (Test-Path $VenvDir) {
        Write-Ok "Cleaning up temporary venv at $VenvDir ..."
        Remove-Item -Recurse -Force $VenvDir -ErrorAction SilentlyContinue
    }
}

# Register cleanup on script exit
$null = Register-EngineEvent -SourceIdentifier PowerShell.Exiting -Action {
    if (Test-Path $using:VenvDir) {
        Remove-Item -Recurse -Force $using:VenvDir -ErrorAction SilentlyContinue
    }
} -ErrorAction SilentlyContinue

try {
    & @PythonCmd -m venv $VenvDir
} catch {
    Exit-Fatal "Failed to create virtual environment: $_"
}

$VenvPython  = Join-Path $VenvDir "Scripts" "python.exe"
$VenvPip     = Join-Path $VenvDir "Scripts" "pip.exe"

if (-not (Test-Path $VenvPython)) {
    Exit-Fatal "Venv python.exe not found at $VenvPython. Venv creation may have failed."
}

Write-Ok "Temporary venv -> $VenvDir"

# ── Install conversion dependencies ─────────────────────────────────────────
Write-Step "Installing conversion dependencies (torch, numpy, openai-whisper) ..."
Write-Ok "This is a one-time cost. The runtime needs no Python."

# Upgrade pip
& $VenvPython -m pip install --quiet --upgrade pip wheel 2>$null

# Install numpy
& $VenvPip install --quiet "numpy>=1.24"
if ($LASTEXITCODE -ne 0) { Remove-TempVenv; Exit-Fatal "Failed to install numpy." }
Write-Ok "numpy installed"

# Install torch (CPU-only for smaller download)
Write-Ok "Installing torch (CPU-only) ... this may take a few minutes"
& $VenvPip install --quiet "torch>=2.0" --index-url "https://download.pytorch.org/whl/cpu" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Warn "CPU-only torch failed, trying default index ..."
    & $VenvPip install --quiet "torch>=2.0"
    if ($LASTEXITCODE -ne 0) { Remove-TempVenv; Exit-Fatal "Failed to install torch." }
}
Write-Ok "torch installed"

# Install openai-whisper
& $VenvPip install --quiet "openai-whisper"
if ($LASTEXITCODE -ne 0) { Remove-TempVenv; Exit-Fatal "Failed to install openai-whisper." }
Write-Ok "openai-whisper installed"

# ── Verify no HuggingFace leakage ───────────────────────────────────────────
Write-Step "Verifying dependency integrity ..."

$setupContent = Get-Content $SetupPy -Raw
if ($setupContent -match "(?i)https?://[^ ]*huggingface|https?://[^ ]*hf\.co|https?://[^ ]*hf-mirror") {
    Remove-TempVenv
    Exit-Fatal "INTEGRITY CHECK FAILED: setup.py contains HuggingFace download URLs!"
}
Write-Ok "No HuggingFace download URLs found in setup script"

# ── Build setup.py arguments ────────────────────────────────────────────────
$SetupArgs = @(
    $SetupPy,
    "--model", $Model
)

if (-not [string]::IsNullOrWhiteSpace($Quantize)) {
    $SetupArgs += @("--quantize", $Quantize)
}

if ($ForceBuild) {
    $SetupArgs += "--force-build"
}

# ── Run the main setup ──────────────────────────────────────────────────────
Write-Step "Running setup.py ..."
Write-Ok "Arguments: $($SetupArgs -join ' ')"
Write-Host ""

& $VenvPython @SetupArgs
$setupExit = $LASTEXITCODE

if ($setupExit -ne 0) {
    Remove-TempVenv
    Exit-Fatal "setup.py exited with code $setupExit"
}

# ── Cleanup ──────────────────────────────────────────────────────────────────
Remove-TempVenv

# ── Final summary ───────────────────────────────────────────────────────────
Write-Host ""
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "  Windows setup complete!" -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Quick start:"
Write-Host "    cd $ProjectRoot"
Write-Host "    .\run-live.ps1"
Write-Host ""
Write-Host "  Ultra-low-latency:"
Write-Host "    .\run-live-fast.ps1"
Write-Host ""
Write-Host "  Or use the .bat file:"
Write-Host "    run-live.bat"
Write-Host ""
Write-Host "  If microphone capture fails, check:"
Write-Host "    Settings -> Privacy -> Microphone -> Allow apps to access"
Write-Host ""
Write-Host "  The temporary venv has been removed."
Write-Host ""
