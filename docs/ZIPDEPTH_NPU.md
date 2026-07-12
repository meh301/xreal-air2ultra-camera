# ZipDepth on the NPU — setup

Monocular metric depth (ZipDepth) run on the Qualcomm Hexagon NPU via ONNX
Runtime's QNN Execution Provider, made metric by calibrating against Basalt VIO
landmarks + a sparse-stereo grid (`xr_depthcal` / `xr_sgrid` / `xr_zipdepth`).

## The three layers (independent)

1. **The ONNX** — one portable file. The *same* file runs on CPU or any NPU; it
   does not get built per-device. `assets/zipdepth.onnx`, input
   `image (1,3,384,384)` NCHW RGB/255, output `depth`.
2. **The Execution Provider** — a runtime library that decides CPU vs NPU.
   Stock `onnxruntime-android` = CPU. `onnxruntime-android-qnn` = the QNN EP for
   the Hexagon NPU, and it pulls the Qualcomm backend libs
   (`com.qualcomm.qti:qnn-runtime`) **transitively and version-matched** from
   Maven.
3. **On-device compile** — the QNN EP compiles the ONNX to the specific Hexagon
   at session load (cacheable). `xr_zipdepth.c` requests
   `backend_path=libQnnHtp.so`, `htp_performance_mode=burst`.

## Build it — two commands

```
# 1. ONNX (already staged; regenerate if the checkpoint changes). --npu is
#    required for the npu checkpoint.
powershell -File android\build_zipdepth_onnx.ps1

# 2. NPU build. -Pqnn swaps in onnxruntime-android-qnn (ORT + matched QNN
#    backend libs, from Maven), arm64-only.
gradlew assembleDebug -Pqnn
```

That's the whole NPU build — **no QAIRT SDK, no from-source ORT build.** The
resulting APK (~165 MB) carries `libonnxruntime.so` (QNN EP), the QNN 2.33
backend + `libQnnHtpV68Skel.so` (V68 = SD888), and `zipdepth.onnx`.

The default `gradlew assembleDebug` (no `-Pqnn`) uses the stock CPU ORT, so the
exact same pipeline runs on the CPU for testing without the NPU stack.

## Verify on device

`adb logcat -s xrealcam`, with depth enabled:
- `ZipDepth: QNN (Hexagon HTP, burst) EP` (not `CPU EP` → the NPU loaded)
- `ZipDepth ready: … (384x384, in=image out=depth)`
- first session load is slow (QNN compiles the graph on-device).

Then check metric quality: the landmark anchors should agree with the stereo
grid (validates `xr_lm_anchors`), and near/far should track the scene. If depth
reads inverted, flip the one invert in the worker (`dinv = 1/dd`).

## Optional / later

- **Override the QNN version** (e.g. a SoC whose Hexagon arch is newer than the
  matched `qnn-runtime` ships, like a V81 part): `android/fetch_qnn.ps1 -Sdk
  <QAIRT path>` stages backend libs from the QAIRT SDK into jniLibs, overriding
  the Maven ones. Match the QAIRT version to the ORT release to avoid an
  EP/backend ABI mismatch.
- **Shrink the APK**: it bundles every Hexagon arch (V68..V79) + the 52 MB
  `libQnnHtpPrepare.so`. For a single target, keep only that arch's Stub/Skel
  and, if you pre-generate a QNN context binary offline (QAIRT tools), drop
  `libQnnHtpPrepare` entirely.
- **INT8 quantization** (calibration set) — the real HTP speed win over FP.

## MediaTek

MediaTek's core NeuroPilot SDK is impractical to obtain; the path there is
**LiteRT (TFLite) + the Neuron delegate** — a different runtime and a TFLite
export of the model, not ORT. Separate integration (a LiteRT backend in
`xr_zipdepth`), tracked for later; the calibration + anchor layers reuse as-is.
