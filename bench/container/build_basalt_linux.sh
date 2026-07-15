#!/usr/bin/env bash
# Build the BENCH libbasalt.so for Linux x86_64 (remote container) + the
# replay harness. Mirrors android/build_basalt.ps1's configuration minus the
# Android toolchain and minus the realtime drop patches (see
# patch_basalt_bench.py). Run from the repo root inside the container.
#
# Container needs: git cmake ninja-build build-essential libtbb-dev
#                  libeigen3-dev libopencv-dev libfmt-dev python3
# (Basalt vendors most other deps; check its README if configure complains.)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="${BASALT_WORK:-$ROOT/bench/container/basalt-linux}"
JOBS="$(nproc)"

if [ ! -d "$WORK/.git" ]; then
    git clone --recursive https://gitlab.freedesktop.org/mateosss/basalt.git "$WORK"
fi
python3 "$ROOT/bench/container/patch_basalt_bench.py" "$WORK"

# oneTBB config-mode find_package (same tweak as build_basalt.ps1:66)
sed -i 's/find_package(TBB REQUIRED)/find_package(TBB CONFIG REQUIRED)/' \
    "$WORK/CMakeLists.txt" || true

cmake -S "$WORK" -B "$WORK/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBASALT_INSTANTIATIONS_DOUBLE=OFF \
    -DBASALT_BUILD_SHARED_LIBRARY_ONLY=ON \
    -DBUILD_TESTS=OFF
cmake --build "$WORK/build" -j "$JOBS" --target basalt

LIB="$(find "$WORK/build" -name 'libbasalt.so' | head -1)"
[ -n "$LIB" ] || { echo "libbasalt.so not produced"; exit 1; }
mkdir -p "$ROOT/bench/container/lib"
cp "$LIB" "$ROOT/bench/container/lib/"
echo "bench libbasalt.so -> $ROOT/bench/container/lib/"

# replay harness at the three dataset resolutions
cd "$ROOT/bench/replay"
make -f Makefile.linux XR_OW=752 XR_OH=480 OUT=xr_replay_euroc
make -f Makefile.linux XR_OW=512 XR_OH=512 OUT=xr_replay_tumvi
make -f Makefile.linux XR_OW=640 XR_OH=480 OUT=xr_replay_msdmoo
echo "harness built: xr_replay_{euroc,tumvi,msdmoo}"
echo "run: LD_LIBRARY_PATH=$ROOT/bench/container/lib ./xr_replay_euroc --pack ... --out ..."
