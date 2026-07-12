# ZipDepth on the NPU — setup

Monocular metric depth (ZipDepth) run on the Qualcomm Hexagon NPU via ONNX
Runtime's QNN Execution Provider, made metric by calibrating against Basalt VIO
landmarks + a sparse-stereo grid (`xr_depthcal` / `xr_sgrid` / `xr_zipdepth`).

## The three layers (they are independent)

1. **The ONNX** — one portable file. The *same* file runs on CPU or any NPU;
   it does **not** get built per-device. `assets/zipdepth.onnx` (input
   `image (1,3,384,384)` NCHW RGB/255, output `depth`).
2. **The Execution Provider** — a *runtime library* that decides CPU vs NPU.
   - Stock `onnxruntime-android` (Maven) = **CPU only**.
   - **NPU needs a QNN-enabled `libonnxruntime.so`** (built `--use_qnn`) + the
     Qualcomm backend `.so`s. No prebuilt Android package exists — build it.
3. **On-device compile** — with the QNN EP present, ORT compiles the ONNX to
   the specific Hexagon at session load (cacheable). `xr_zipdepth.c` requests
   `backend_path=libQnnHtp.so`, `htp_performance_mode=burst`.

## Steps

**1. ONNX** — done (`android/build_zipdepth_onnx.ps1` regenerates it; the
`--npu` flag is required for the npu checkpoint). Lands in
`app/src/xfeat/assets/zipdepth.onnx` (full-build only; gitignored for size).

**2. Build ORT with the QNN EP** (a machine with a host C++ compiler + NDK;
this repo's host has none). On Windows:
```bat
git clone --recursive https://github.com/microsoft/onnxruntime
cd onnxruntime
.\build.bat --android --use_qnn ^
    --qnn_home "F:\v2.48.0.260626\qairt\2.48.0.260626" ^
    --android_sdk_path "%LOCALAPPDATA%\Android\Sdk" ^
    --android_ndk_path "%LOCALAPPDATA%\Android\Sdk\ndk\27.0.12077973" ^
    --android_abi arm64-v8a --android_api 24 ^
    --config Release --build_shared_lib --parallel --skip_tests
```
Use a recent NDK (r26/r27) just for this ORT build — the resulting
`build\Android\Release\libonnxruntime.so` is ABI-compatible with our NDK-21 app.
Copy it to `app/src/main/jniLibs/arm64-v8a/`.

**3. Stage the QNN backend libs** from the QAIRT SDK:
```
powershell -File android\fetch_qnn.ps1 -Sdk "F:\v2.48.0.260626\qairt\2.48.0.260626"
```
Copies `libQnnHtp.so`, `libQnnSystem.so`, the per-arch `libQnnHtpV*Stub.so`
(V68 = SD888 … V81) and DSP-side `libQnnHtpV*Skel.so` into `jniLibs/arm64-v8a/`.
`libcdsprpc.so` is a device system lib — not bundled.

**4. Build the app for the NPU:**
```
gradlew assembleDebug -Pqnn
```
`-Pqnn` drops the Maven CPU ORT (jniLibs supplies the QNN one), arm64-only.

## Verify on device

`adb logcat -s xrealcam` should show, when depth is enabled:
- `ZipDepth: QNN (Hexagon HTP, burst) EP` (not `CPU EP` → EP loaded)
- `ZipDepth ready: … (384x384, in=image out=depth)`
- first session load is slow (QNN compiles the graph); enable context caching
  later to skip it.

Then check the metric quality: the landmark anchors should agree with the
stereo grid (validates `xr_lm_anchors`), and near/far depth should track the
scene. If depth reads inverted (near↔far), flip the one invert in the worker
(`dinv = 1/dd`). INT8 quantization (calibration set) is the follow-up speed win.

## MediaTek

MediaTek's core NeuroPilot SDK is hard to obtain; the intended path there is
**LiteRT (TFLite) + the Neuron delegate** — a different runtime and a TFLite
export of the model, not ORT. That is a separate integration (a `xr_zipdepth`
LiteRT backend), tracked for later; the calibration/anchor layers are reused
as-is.
