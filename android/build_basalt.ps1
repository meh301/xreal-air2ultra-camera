# build_basalt.ps1 — build libbasalt.so (Monado's Basalt VIO fork) for
# Android arm64-v8a and drop it into app/src/main/jniLibs/.
#
# The app dlopen's the library at runtime; without it the research app still
# runs on the built-in tracker. Everything lands under $ToolDir (~4 GB with
# the NDK), nothing outside this repo is modified besides that directory.
#
# Requires: git, curl (both ship with Git for Windows). Run from anywhere:
#   powershell -ExecutionPolicy Bypass -File android\build_basalt.ps1

$ErrorActionPreference = "Stop"
$RepoAndroid = Split-Path -Parent $MyInvocation.MyCommand.Path
$ThirdParty = Join-Path $RepoAndroid "app\src\main\cpp\third_party"
$JniLibs = Join-Path $RepoAndroid "app\src\main\jniLibs\arm64-v8a"
$ToolDir = Join-Path $env:USERPROFILE "Android\toolchains"

New-Item -ItemType Directory -Force $ToolDir | Out-Null
New-Item -ItemType Directory -Force $JniLibs | Out-Null

# ---- toolchain: NDK r27c + CMake + Ninja --------------------------------------
$Ndk = Join-Path $ToolDir "android-ndk-r27c"
if (-not (Test-Path $Ndk)) {
    Write-Host "Downloading NDK r27c (~650 MB)..."
    $zip = Join-Path $ToolDir "ndk.zip"
    curl.exe -sL -o $zip https://dl.google.com/android/repository/android-ndk-r27c-windows.zip
    Expand-Archive $zip -DestinationPath $ToolDir
    Remove-Item $zip
}
$CMakeDir = Join-Path $ToolDir "cmake-3.29.6-windows-x86_64"
if (-not (Test-Path $CMakeDir)) {
    Write-Host "Downloading CMake..."
    $zip = Join-Path $ToolDir "cmake.zip"
    curl.exe -sL -o $zip https://github.com/Kitware/CMake/releases/download/v3.29.6/cmake-3.29.6-windows-x86_64.zip
    Expand-Archive $zip -DestinationPath $ToolDir
    Remove-Item $zip
}
$NinjaDir = Join-Path $ToolDir "ninja"
if (-not (Test-Path $NinjaDir)) {
    Write-Host "Downloading Ninja..."
    $zip = Join-Path $ToolDir "ninja.zip"
    curl.exe -sL -o $zip https://github.com/ninja-build/ninja/releases/download/v1.12.1/ninja-win.zip
    Expand-Archive $zip -DestinationPath $NinjaDir
    Remove-Item $zip
}
$CMake = Join-Path $CMakeDir "bin\cmake.exe"
$Ninja = Join-Path $NinjaDir "ninja.exe"
$Toolchain = Join-Path $Ndk "build\cmake\android.toolchain.cmake"
$Prefix = Join-Path $ToolDir "prefix-arm64"

$CommonArgs = @(
    "-G", "Ninja", "-DCMAKE_MAKE_PROGRAM=$Ninja",
    "-DCMAKE_TOOLCHAIN_FILE=$Toolchain",
    "-DANDROID_ABI=arm64-v8a", "-DANDROID_PLATFORM=android-24",
    "-DCMAKE_BUILD_TYPE=Release"
)

# ---- sources -------------------------------------------------------------------
$Basalt = Join-Path $ThirdParty "basalt"
if (-not (Test-Path $Basalt)) {
    Write-Host "Cloning basalt (Monado fork, ~600 MB with submodules)..."
    git clone --depth 1 --recurse-submodules --shallow-submodules -j8 `
        https://gitlab.freedesktop.org/mateosss/basalt.git $Basalt
}
# oneTBB's config package instead of basalt's pre-oneTBB find module
$cml = Join-Path $Basalt "CMakeLists.txt"
(Get-Content $cml -Raw) -replace "  find_package\(TBB REQUIRED\)", "  find_package(TBB CONFIG REQUIRED)" |
    Set-Content $cml -NoNewline -Encoding utf8
# local source patches (cam-calib optional, non-blocking VIT pushes)
python (Join-Path $RepoAndroid "patch_basalt.py")
if ($LASTEXITCODE) { throw "patch_basalt.py failed" }

$OneTbb = Join-Path $ToolDir "onetbb"
if (-not (Test-Path $OneTbb)) {
    git clone --depth 1 --branch v2021.13.0 https://github.com/uxlfoundation/oneTBB.git $OneTbb
}
$Fmt = Join-Path $ToolDir "fmt"
if (-not (Test-Path $Fmt)) {
    git clone --depth 1 --branch 9.1.0 https://github.com/fmtlib/fmt.git $Fmt
}
$OpenCv = Join-Path $ToolDir "OpenCV-android-sdk"
if (-not (Test-Path $OpenCv)) {
    Write-Host "Downloading OpenCV Android SDK (~250 MB)..."
    $zip = Join-Path $ToolDir "opencv.zip"
    curl.exe -sL -o $zip https://github.com/opencv/opencv/releases/download/4.10.0/opencv-4.10.0-android-sdk.zip
    Expand-Archive $zip -DestinationPath $ToolDir
    Remove-Item $zip
}

# ---- oneTBB (tbbmalloc's version scripts don't link on Android — skip it) -----
& $CMake -B "$ToolDir\build-tbb" -S $OneTbb @CommonArgs `
    "-DTBB_TEST=OFF" "-DTBB_EXAMPLES=OFF" "-DTBB_STRICT=OFF" `
    "-DTBBMALLOC_BUILD=OFF" "-DTBBMALLOC_PROXY_BUILD=OFF" `
    "-DCMAKE_INSTALL_PREFIX=$Prefix"
& $CMake --build "$ToolDir\build-tbb"; if ($LASTEXITCODE) { throw "tbb build failed" }
& $CMake --install "$ToolDir\build-tbb"

# ---- fmt -----------------------------------------------------------------------
& $CMake -B "$ToolDir\build-fmt" -S $Fmt @CommonArgs `
    "-DFMT_TEST=OFF" "-DFMT_DOC=OFF" "-DCMAKE_INSTALL_PREFIX=$Prefix"
& $CMake --build "$ToolDir\build-fmt"; if ($LASTEXITCODE) { throw "fmt build failed" }
& $CMake --install "$ToolDir\build-fmt"

# ---- basalt --------------------------------------------------------------------
$PrefixFwd = $Prefix -replace "\\", "/"
$OpenCvJni = (Join-Path $OpenCv "sdk\native\jni") -replace "\\", "/"
& $CMake -B "$ToolDir\build-basalt" -S $Basalt @CommonArgs `
    "-DANDROID_STL=c++_shared" `
    "-DBASALT_BUILD_SHARED_LIBRARY_ONLY=ON" `
    "-DBASALT_INSTANTIATIONS_DOUBLE=OFF" `
    "-DCXX_MARCH=armv8.2-a+fp16+dotprod" `
    "-DCMAKE_FIND_ROOT_PATH=$PrefixFwd;$OpenCvJni" `
    "-DTBB_DIR=$PrefixFwd/lib/cmake/TBB" `
    "-Dfmt_DIR=$PrefixFwd/lib/cmake/fmt" `
    "-DOpenCV_DIR=$OpenCvJni" `
    "-DSKIP_PERFORMANCE_COMPARISON=ON" "-DBUILD_TESTS=OFF" `
    "-DBUILD_SANDBOX=OFF" "-DBUILD_DOC=OFF"
& $CMake --build "$ToolDir\build-basalt" --target basalt
if ($LASTEXITCODE) { throw "basalt build failed" }

# ---- strip + package ------------------------------------------------------------
$Strip = Join-Path $Ndk "toolchains\llvm\prebuilt\windows-x86_64\bin\llvm-strip.exe"
& $Strip --strip-unneeded -o (Join-Path $JniLibs "libbasalt.so") (Join-Path $ToolDir "build-basalt\libbasalt.so")
& $Strip --strip-unneeded -o (Join-Path $JniLibs "libtbb.so") (Join-Path $Prefix "lib\libtbb.so")
Copy-Item (Join-Path $Ndk "toolchains\llvm\prebuilt\windows-x86_64\sysroot\usr\lib\aarch64-linux-android\libc++_shared.so") $JniLibs

Write-Host "Done:"
Get-ChildItem $JniLibs | Format-Table Name, Length
