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

# Heavy dependencies (torch, audio-separator, ML packages, utilities) are now installed at runtime for smaller binary size

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
        --add-data "config;config" `
        --add-data ".env;.env" `
        --hidden-import "convert" `
        --hidden-import "convert.mp4_wav" `
        --hidden-import "convert.separate_audio" `
        --hidden-import "transcription" `
        --hidden-import "transcription.whisper" `
        --hidden-import "transcription.emotion" `
        --hidden-import "translate" `
        --hidden-import "translate.openAi" `
        --hidden-import "tts" `
        --hidden-import "tts.orchestrator" `
        --hidden-import "tts.edge_tts" `
        --hidden-import "tts.F5" `
        --hidden-import "tts.xtts" `
        --hidden-import "tts.utils" `
        --hidden-import "tts.config" `
        --hidden-import "dub" `
        --hidden-import "dub.mixer" `
        --hidden-import "utils" `
        --hidden-import "utils.logger" `
        --hidden-import "utils.metadata" `
        --hidden-import "utils.srt_export" `
        --hidden-import "utils.burn_subtitles" `
        --exclude-module "torch" `
        --exclude-module "torchvision" `
        --exclude-module "torchaudio" `
        --exclude-module "audio_separator" `
        --exclude-module "openai-whisper" `
        --exclude-module "whisper" `
        --exclude-module "coqui-tts" `
        --exclude-module "f5-tts" `
        --exclude-module "pyannote-audio" `
        --exclude-module "pyannote" `
        --exclude-module "transformers" `
        --exclude-module "vocos" `
        --exclude-module "silero_vad" `
        --exclude-module "huggingface_hub" `
        --exclude-module "edge_tts" `
        --exclude-module "coqui-tts" `
        --exclude-module "silero-vad" `
        --exclude-module "openai" `
        --exclude-module "pydub" `
        --exclude-module "soundfile" `
        --exclude-module "tqdm" `
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
# Heavy dependencies (torch, audio-separator, ML packages, utilities) are now installed at runtime based on detected environment for smaller binary size
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
Write-Host "`nRuntime will automatically detect and install appropriate dependencies (CPU/CUDA/ROCm) for smaller binary size." -ForegroundColor Green
