# ===========================================
# build_all.ps1 — Build CPU + GPU exe versions
# Supports CUDA 12.6, 12.8, 12.9
# Automatically restores GPU or CPU environment after build
# ===========================================

$ErrorActionPreference = "Stop"

# ---- Configuration ----
$AppName = "app"
$EntryFile = "main.py"
$DistDir = "dist"
$BuildDir = "build"
# Note: Separate CUDA builds no longer needed - runtime installer handles all environments

# ---- Helpers ----
function Get-TorchPath {
    return (uv run python -c "import torch, os; print(os.path.dirname(torch.__file__))")
}

function Clean-PreviousBuilds {
    if (Test-Path $BuildDir) { Remove-Item -Recurse -Force $BuildDir }
}

# Heavy dependencies (torch, audio-separator) are now installed at runtime

function Build-App {
    param(
        [string]$Name,
        [string]$TorchPath
    )

    uv run pyinstaller `
        --onefile `
        --noconfirm `
        --name $Name `
        --distpath $DistDir `
        --workpath $BuildDir `
        --exclude-module "torch" `
        --exclude-module "torchvision" `
        --exclude-module "torchaudio" `
        --exclude-module "audio_separator" `
        $EntryFile
}

function Has-GPU {
    try {
        $gpuName = uv run python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
        return -not [string]::IsNullOrWhiteSpace($gpuName)
    } catch {
        return $false
    }
}

# ---- Ensure dist folder exists ----
if (-not (Test-Path $DistDir)) {
    New-Item -ItemType Directory -Path $DistDir | Out-Null
}

# ---- UNIVERSAL BUILD ----
Write-Host "`nBuilding universal version..." -ForegroundColor Green
Clean-PreviousBuilds
# Heavy dependencies (torch, audio-separator) are now installed at runtime based on detected environment
Build-App -Name "$AppName" -TorchPath ""
Write-Host "Universal build complete: $DistDir\$AppName.exe" -ForegroundColor Green

# ---- RESTORE DEV ENVIRONMENT ----
Write-Host "`nEnvironment restoration note:" -ForegroundColor Cyan
Write-Host "Heavy dependencies (torch, audio-separator) are now installed at runtime" -ForegroundColor Yellow
Write-Host "No environment restoration needed for build-time dependencies" -ForegroundColor Yellow

if (Has-GPU) {
    Write-Host "GPU detected — runtime will auto-install GPU versions when needed" -ForegroundColor Green
} else {
    Write-Host "No GPU detected — runtime will auto-install CPU versions when needed" -ForegroundColor Yellow
}

Write-Host "`nBuild completed successfully!"
Write-Host "  Universal exe: $DistDir\$AppName.exe"
Write-Host "`nRuntime will automatically detect and install appropriate dependencies (CPU/CUDA/ROCm)." -ForegroundColor Green
