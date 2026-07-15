# Build the on-device (arm64 Android) replay binaries, one per dataset
# resolution. Output: bench/replay/obj/<res>/xr_replay
#   powershell -File bench/build_replay.ps1
$ErrorActionPreference = "Stop"
$Ndk = Join-Path $env:USERPROFILE "Android\toolchains\android-ndk-r27c"
$NdkBuild = Join-Path $Ndk "ndk-build.cmd"
if (-not (Test-Path $NdkBuild)) { throw "ndk-build not found at $NdkBuild" }
$Replay = Join-Path $PSScriptRoot "replay"

$variants = @(
    @{name="euroc";  w=752; h=480},
    @{name="tumvi";  w=512; h=512},
    @{name="msdmoo"; w=640; h=480},
    @{name="device"; w=480; h=640}
)
foreach ($v in $variants) {
    Write-Host "== xr_replay $($v.name) ($($v.w)x$($v.h)) =="
    $out = Join-Path $Replay "obj\$($v.name)"
    & $NdkBuild -C $Replay NDK_PROJECT_PATH=. `
        APP_BUILD_SCRIPT=jni/Android.mk NDK_APPLICATION_MK=jni/Application.mk `
        NDK_OUT="obj\$($v.name)" NDK_LIBS_OUT="libs\$($v.name)" `
        "XR_OW=$($v.w)" "XR_OH=$($v.h)" | Select-Object -Last 3
    if ($LASTEXITCODE -ne 0) { throw "build failed: $($v.name)" }
}
Write-Host "binaries under bench/replay/libs/<variant>/arm64-v8a/xr_replay"