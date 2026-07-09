# Clones the native dependencies (pinned tags) into app/src/main/cpp/third_party.
# Run once before the first Gradle build.
$ErrorActionPreference = "Stop"
$dest = Join-Path $PSScriptRoot "app\src\main\cpp\third_party"
New-Item -ItemType Directory -Force $dest | Out-Null

$deps = @(
    @{ name = "libusb"; url = "https://github.com/libusb/libusb.git"; tag = "v1.0.27" },
    @{ name = "libuvc"; url = "https://github.com/libuvc/libuvc.git"; tag = "v0.0.7" }
)
foreach ($d in $deps) {
    $path = Join-Path $dest $d.name
    if (Test-Path (Join-Path $path ".git")) {
        Write-Host "$($d.name) already present, skipping"
        continue
    }
    git clone --depth 1 --branch $d.tag $d.url $path
}
Write-Host "Done. Native deps are in $dest"
