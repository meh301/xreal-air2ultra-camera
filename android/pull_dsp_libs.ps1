# Pull the Qualcomm FastRPC / DSP libraries THIS device needs for the QNN HTP
# (Hexagon NPU) ZipDepth path, into jniLibs/arm64-v8a (bundled by the -Pqnn
# build; jniLibs is gitignored). Run ONCE with the target phone on adb.
#
# WHY: QNN's libQnnHtp.so must dlopen libcdsprpc.so (the FastRPC client) + its
# DSP/system-lib closure to reach the Hexagon. Those live in /vendor/lib64 and
# /system/lib64, which are NOT on an Android app's linker namespace -- so they
# are never found and QnnDevice_create fails with QNN_DEVICE_ERROR_INVALID_CONFIG
# (the device is refused BEFORE any DSP contact, which is why no EP option,
# backend_path or QNN version ever moved it). Placing them in the app's native
# lib dir (nativeLibraryDir, via jniLibs) puts them on the SAME namespace as
# libQnnHtp.so, so its dlopen resolves the whole closure. (setenv LD_LIBRARY_PATH
# alone is NOT reliably honored by the app namespace -- nativeLibraryDir is.)
#
# These libs are DEVICE-SPECIFIC: pull them from the same phone the -Pqnn APK
# will run on. Proven set: DakeQQ/YOLO-Depth-Estimation-for-Android, ORT #21214.
#
#   powershell -File android\pull_dsp_libs.ps1
#   android\gradlew.bat assembleDebug -Pqnn
#
# NOTE: this only ADDS files to jniLibs/arm64-v8a; it never removes the prebuilt
# libbasalt.so / libtbb.so / libc++_shared.so already there.
$ErrorActionPreference = "Stop"

if (-not (Get-Command adb -ErrorAction SilentlyContinue)) {
    throw "adb not found on PATH. Install platform-tools or add it to PATH."
}
$state = (& adb get-state) 2>&1
if ($LASTEXITCODE -ne 0 -or "$state" -notmatch "device") {
    throw "No device over adb (get-state='$state'). Connect the phone + enable USB debugging."
}

$dst = Join-Path $PSScriptRoot "app\src\main\jniLibs\arm64-v8a"
New-Item -ItemType Directory -Force -Path $dst | Out-Null

# The /vendor/lib64 ones are the app-namespace-invisible vendor libs (the real
# fix); the /system/lib64 ones are libcdsprpc's platform-private closure (not in
# the NDK, so also invisible to an app). ld-android.so / libdl_android.so are
# intentionally omitted: they are linker-owned and never loaded from a file.
$libs = @(
    "/vendor/lib64/libcdsprpc.so",
    "/vendor/lib64/vendor.qti.hardware.dsp@1.0.so",
    "/vendor/lib64/libvmmem.so",
    "/system/lib64/libhidlbase.so",
    "/system/lib64/libhardware.so",
    "/system/lib64/libutils.so",
    "/system/lib64/libcutils.so",
    "/system/lib64/libdmabufheap.so",
    "/system/lib64/libc++.so",
    "/system/lib64/libbase.so",
    "/system/lib64/libvndksupport.so"
)

$ok = 0
foreach ($l in $libs) {
    $name = Split-Path $l -Leaf
    $out = Join-Path $dst $name
    if (Test-Path $out) { Remove-Item $out -Force }
    try { & adb pull $l $out 2>&1 | Out-Null } catch {}
    if (Test-Path $out) { Write-Host "  + $name"; $ok++ }
    else { Write-Host "  ! missing $l (skipped)" }
}

Write-Host ""
if (Test-Path (Join-Path $dst "libcdsprpc.so")) {
    Write-Host "Pulled $ok/$($libs.Count) DSP libs -> $dst"
    Write-Host "Now (re)build the NPU APK:  android\gradlew.bat assembleDebug -Pqnn"
} else {
    throw "libcdsprpc.so was NOT pulled -- the HTP device cannot be created without it. Confirm this is a Snapdragon and /vendor/lib64/libcdsprpc.so exists."
}
