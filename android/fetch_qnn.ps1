# OPTIONAL. The -Pqnn build already gets version-matched QNN backend libs from
# Maven (onnxruntime-android-qnn -> com.qualcomm.qti:qnn-runtime), so you do NOT
# need this for a normal Snapdragon build. Use it only to OVERRIDE those with a
# specific QAIRT SDK version -- e.g. a SoC whose Hexagon arch is newer than the
# matched qnn-runtime ships (a V81 part). Match the QAIRT version to the ORT
# release or the EP/backend ABI can mismatch. jniLibs is gitignored and wins
# over the Maven libs during the native-lib merge.
#
#   powershell -File android\fetch_qnn.ps1 -Sdk "F:\v2.48.0.260626\qairt\2.48.0.260626"
param(
    [string]$Sdk = "F:\v2.48.0.260626\qairt\2.48.0.260626"
)
$ErrorActionPreference = "Stop"
$dst = Join-Path $PSScriptRoot "app\src\main\jniLibs\arm64-v8a"
New-Item -ItemType Directory -Force -Path $dst | Out-Null

$android = Join-Path $Sdk "lib\aarch64-android"
if (-not (Test-Path $android)) { throw "QAIRT android libs not found at $android -- check -Sdk" }

# CPU-side: the HTP backend, the system lib, the graph-prepare lib, a CPU
# fallback backend, and the per-Hexagon-arch stubs. Bundling all stubs lets one
# APK cover SD888 (V68) through the newest parts (V79/V81); the runtime loads
# the one matching the device.
$cpuSide = @(
    "libQnnHtp.so", "libQnnSystem.so", "libQnnHtpPrepare.so", "libQnnCpu.so",
    "libQnnHtpV68Stub.so", "libQnnHtpV73Stub.so", "libQnnHtpV75Stub.so",
    "libQnnHtpV79Stub.so", "libQnnHtpV81Stub.so"
)
foreach ($f in $cpuSide) {
    $src = Join-Path $android $f
    if (Test-Path $src) { Copy-Item $src $dst -Force; Write-Host "  + $f" }
    else { Write-Host "  ! missing $f (skipped)" }
}

# DSP-side skels (run on the Hexagon): lib\hexagon-v68\unsigned\libQnnHtpV68Skel.so, ...
foreach ($v in @("v68", "v73", "v75", "v79", "v81")) {
    $V = $v.ToUpper()
    $src = Join-Path $Sdk "lib\hexagon-$v\unsigned\libQnnHtp${V}Skel.so"
    if (Test-Path $src) { Copy-Item $src $dst -Force; Write-Host "  + libQnnHtp${V}Skel.so" }
    else { Write-Host "  ! missing libQnnHtp${V}Skel.so (skipped)" }
}

Write-Host ""
Write-Host "Staged QAIRT QNN libs -> $dst (these OVERRIDE the Maven qnn-runtime)."
Write-Host "Build with -Pqnn. Remove this jniLibs dir to go back to the matched Maven libs."
