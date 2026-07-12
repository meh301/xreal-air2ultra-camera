# Export the ZipDepth NPU checkpoint to ONNX and stage it as the app asset.
# Needs the ZipDepth repo cloned at <root>\ZipDepth (gitignored) with its
# checkpoints, and a Python env with its requirements (torch, onnx, onnxsim).
# The ONNX is the SAME file for CPU or NPU -- the execution provider (QNN /
# NeuroPilot / CPU) decides where it runs, not the file.
#
#   powershell -File android\build_zipdepth_onnx.ps1
param(
    [int]$Width = 384,
    [int]$Height = 384          # 288x384 preserves the 3:4 rectified aspect;
                                # if you change these, set ZD_IN_W/H in xr_zipdepth.c
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$zip = Join-Path $root "ZipDepth"
if (-not (Test-Path (Join-Path $zip "scripts\export.py"))) {
    throw "ZipDepth repo not found at $zip (git clone https://github.com/fabiotosi92/ZipDepth)"
}
$env:PYTHONUTF8 = "1"
Push-Location $zip
try {
    python scripts/export.py --ckpt checkpoints/zipdepth_base_npu.pth `
        --format onnx --npu --width $Width --height $Height
} finally { Pop-Location }

$onnx = Join-Path $zip ("checkpoints\zipdepth_base_{0}x{1}.onnx" -f $Width, $Height)
$dst = Join-Path $PSScriptRoot "app\src\xfeat\assets\zipdepth.onnx"
Copy-Item $onnx $dst -Force
Write-Host "Staged $onnx -> $dst ($((Get-Item $dst).Length) bytes)"
