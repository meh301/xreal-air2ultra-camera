package org.air2ultra.stereocam

import android.app.Activity
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbDeviceConnection
import android.hardware.usb.UsbManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors
import java.util.zip.Deflater
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/**
 * xreal_datarecorder — a sibling of MainActivity that captures a device-specific
 * VIO/SLAM dataset from the glasses. It opens the UVC stereo feed + the 1 kHz
 * IMU (but deliberately never calls nativeSetAlignment, so Basalt/SLAM/depth
 * never start — no tracking, minimal battery) and streams a benchmark-style
 * "replay pack" to SHARED device storage (/sdcard/XrealDatasets), so a capture
 * is one `adb pull` away, browsable in the Files app, and survives uninstall.
 * App-private storage is only the fallback when all-files access is declined.
 *
 * Capture format (a self-describing ZIP, laid out like a bench replay pack so a
 * capture is a drop-in benchmark sequence — see bench/host/pack_common.py):
 *
 *   meta.txt     dataset seq W H fps imu_hz compressed n_frames  (key=value)
 *   calib.txt    model kb4 + per-eye pinhole/dist/q_xyzw/p + IMU noises, the
 *                EXACT bench field format (built natively from the factory blob
 *                via the SAME fisheye624->kb4 refit the live VIO uses)
 *   frames.csv   "ts_ns,idx" per stereo pair (header-less)
 *   frames.raw   L8 gray pairs [left W*H][right W*H] per frames.csv line — the
 *                exact bytes the VIO consumed (no lossy re-encode)
 *   imu.bin      32-byte LE records: i64 ts_ns, f32 gyro_dps[3], f32 accel_g[3]
 *   intrinsics.json  the same calibration as structured JSON (per-eye fx/fy/
 *                cx/cy, fisheye624 distortion, cam->IMU q/p, IMU biases +
 *                noises, geometry and rates) — for tooling that shouldn't
 *                have to parse bench text
 *   config.json  raw factory calibration blob (provenance / re-derivation)
 *   README.txt   this format + the timestamp-sync note
 *
 * TIMESTAMP SYNC (the crux of a usable VIO dataset): frames.csv ts_ns and
 * imu.bin ts_ns are in ONE clock. Each frame's ts is the camera exposure stamp
 * unwrapped against the IMU's 64-bit ns clock (native frame_cb), and the IMU
 * samples carry that same clock — so a host can integrate IMU between frames
 * with no cross-clock guesswork. left = physical left camera (cam1).
 *
 * The only bench-pack file NOT written is gt.tum (6-DoF ground truth): a
 * headset has no external mocap. Packs without it still replay; they just can't
 * be ATE-scored. dataset is "xreal" (see README for using it with the strict
 * bench tooling).
 */
class RecorderActivity : Activity() {

    companion object {
        private const val XREAL_VID = 0x3318
        private const val XREAL_PID = 0x0426
        private const val ACTION_USB_PERMISSION = "org.air2ultra.stereocam.RECORDER_USB_PERMISSION"
        private const val FRAME_INTERVAL_MS = 33L       // ~30 fps preview poll
        // Shared storage, NOT app-private: a dataset is meant to be pulled off
        // the device (`adb pull /sdcard/XrealDatasets`), opened in the Files
        // app, and to outlive an app uninstall/reinstall cycle.
        private const val DATASET_DIR = "XrealDatasets" // /sdcard/XrealDatasets
        private const val CAPTURE_DIR = "captures"      // app-private fallback
        private const val POLL_MS = 4L                  // writer drain cadence
        private const val IMU_BATCH = 512               // samples per IMU drain
        private const val ALIGN_VARIANT = 2             // XR_ALIGN_VARIANT_DEFAULT
        // USB access to the composite (UVC+audio) device needs these on 10+.
        private val RUNTIME_PERMS = arrayOf(
            android.Manifest.permission.CAMERA,
            android.Manifest.permission.RECORD_AUDIO
        )
    }

    private lateinit var usbManager: UsbManager
    private lateinit var imageView: ImageView
    private lateinit var recButton: Button
    private lateinit var statusText: TextView

    private var connection: UsbDeviceConnection? = null
    private var streaming = false
    private var bitmap: Bitmap? = null

    // Recording state. `capturing` is read by the writer thread, so @Volatile.
    @Volatile private var capturing = false
    // Keeps an important status message (Saved / failed) on screen instead of
    // letting the ~30 fps preview loop overwrite it immediately.
    @Volatile private var statusHoldUntil = 0L
    // One all-files prompt per launch: re-prompting on every resume would trap
    // a user who deliberately declined.
    private var storageAsked = false

    // Preview buffer (composed side-by-side RGBA, same as the main app).
    private val previewBuffer =
        ByteBuffer.allocateDirect(1280 * 640 * 4).order(ByteOrder.nativeOrder())
    private val handler = Handler(Looper.getMainLooper())
    private val recExec = Executors.newSingleThreadExecutor()

    private val usbReceiver = object : BroadcastReceiver() {
        override fun onReceive(ctx: Context, intent: Intent) {
            when (intent.action) {
                ACTION_USB_PERMISSION -> {
                    val dev: UsbDevice? = intent.getParcelableExtra(UsbManager.EXTRA_DEVICE)
                    if (intent.getBooleanExtra(UsbManager.EXTRA_PERMISSION_GRANTED, false) && dev != null)
                        openDevice(dev)
                    else status("USB permission denied")
                }
                UsbManager.ACTION_USB_DEVICE_DETACHED -> stopStreaming("Glasses unplugged")
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)  // long walks: don't sleep
        usbManager = getSystemService(Context.USB_SERVICE) as UsbManager
        buildUi()
        ensureStorageAccess()   // before the first capture, not during one
        val filter = IntentFilter(ACTION_USB_PERMISSION).apply {
            addAction(UsbManager.ACTION_USB_DEVICE_DETACHED)
        }
        if (Build.VERSION.SDK_INT >= 33)
            registerReceiver(usbReceiver, filter, Context.RECEIVER_NOT_EXPORTED)
        else
            @Suppress("UnspecifiedRegisterReceiverFlag") registerReceiver(usbReceiver, filter)
        connect(intent)
    }

    override fun onNewIntent(intent: Intent?) {
        super.onNewIntent(intent)
        connect(intent)     // relaunched by USB_DEVICE_ATTACHED (singleTask)
    }

    override fun onResume() {
        super.onResume()
        // Returning from the all-files Settings screen lands here; refresh the
        // banner so the user can see whether the grant took.
        if (!capturing && System.currentTimeMillis() > statusHoldUntil && !haveSharedStorage())
            status("Storage access not granted — captures go to app-private storage")
    }

    // ---- shared storage ------------------------------------------------------

    /** True when captures can be written to /sdcard/XrealDatasets. */
    private fun haveSharedStorage(): Boolean =
        Build.VERSION.SDK_INT < 30 || Environment.isExternalStorageManager()

    /**
     * Ask once per launch for all-files access. Scoped storage offers no other
     * way to own a top-level public folder AND stream multi-gigabyte captures
     * into it by real path; MediaStore would work but hands back opaque content
     * URIs, which is exactly the "hard to get the data out" problem this avoids.
     */
    private fun ensureStorageAccess() {
        if (haveSharedStorage() || storageAsked) return
        storageAsked = true
        val tries = listOf(
            Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                   Uri.parse("package:$packageName")),
            Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION)
        )
        for (i in tries) {
            try { startActivity(i); return } catch (_: Exception) {}
        }
    }

    /**
     * Where this capture is written. Shared storage when granted, otherwise
     * app-private external — a denied grant must never lose a walk, it just
     * makes the ZIP less convenient to retrieve.
     */
    private fun captureDir(): File {
        if (haveSharedStorage()) {
            val pub = File(Environment.getExternalStorageDirectory(), DATASET_DIR)
            if (pub.isDirectory || pub.mkdirs()) return pub
        }
        return File(getExternalFilesDir(null) ?: filesDir, CAPTURE_DIR).apply { mkdirs() }
    }

    private fun buildUi() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.BLACK)
        }
        statusText = TextView(this).apply {
            setTextColor(Color.WHITE); textSize = 15f; setPadding(28, 28, 28, 12)
        }
        imageView = ImageView(this).apply { scaleType = ImageView.ScaleType.FIT_CENTER }
        recButton = Button(this).apply {
            text = "START"; textSize = 22f; isEnabled = false
            setOnClickListener { toggleRecord() }
        }
        root.addView(statusText, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        root.addView(imageView, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        root.addView(recButton, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
            setMargins(36, 12, 36, 48)
        })
        setContentView(root)
        status("Plug in the glasses…")
    }

    private fun connect(intent: Intent?) {
        if (streaming) return
        val missing = RUNTIME_PERMS.filter {
            checkSelfPermission(it) != PackageManager.PERMISSION_GRANTED
        }
        if (missing.isNotEmpty()) {
            status("Grant camera + mic permission (needed for USB access)…")
            requestPermissions(missing.toTypedArray(), 1)   // resumes in onRequestPermissionsResult
            return
        }
        val fromIntent: UsbDevice? = intent?.getParcelableExtra(UsbManager.EXTRA_DEVICE)
        val device = fromIntent?.takeIf { isXreal(it) }
            ?: usbManager.deviceList.values.firstOrNull { isXreal(it) }
        if (device == null) { status("Plug in the glasses…"); return }
        if (usbManager.hasPermission(device)) {
            openDevice(device)
        } else {
            status("Waiting for USB permission…")
            val flags = if (Build.VERSION.SDK_INT >= 31)
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_MUTABLE
            else PendingIntent.FLAG_UPDATE_CURRENT
            val pi = PendingIntent.getBroadcast(
                this, 0, Intent(ACTION_USB_PERMISSION).setPackage(packageName), flags)
            usbManager.requestPermission(device, pi)
        }
    }

    private fun isXreal(d: UsbDevice) = d.vendorId == XREAL_VID && d.productId == XREAL_PID

    private fun openDevice(device: UsbDevice) {
        val conn = usbManager.openDevice(device) ?: run { status("USB open failed"); return }
        connection = conn                        // keep open while native streams on its fd
        val rc = XrealNative.nativeStart(conn.fileDescriptor)
        if (rc != 0) { status("native start error $rc"); conn.close(); connection = null; return }
        streaming = true
        XrealNative.nativeSetPaneMode(1)         // phone pane = LEFT | RIGHT cameras (raw pair)
        // NB: intentionally NOT calling nativeSetAlignment -> SLAM/depth never start.
        recButton.isEnabled = true
        status("Streaming — press START to record")
        handler.post(pollFrame)
    }

    /** Preview loop only — recording runs on its own thread off the native tap. */
    private val pollFrame = object : Runnable {
        override fun run() {
            val packed = XrealNative.nativeGrabFrame(previewBuffer)
            if (packed != 0L) {
                val w = (packed ushr 48).toInt()
                val h = ((packed ushr 32) and 0xFFFF).toInt()
                var bm = bitmap
                if (bm == null || bm.width != w || bm.height != h) {
                    bm = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
                    bitmap = bm; imageView.setImageBitmap(bm)
                }
                previewBuffer.position(0)
                bm.copyPixelsFromBuffer(previewBuffer)
                imageView.invalidate()
                // The writer thread owns the status line while capturing, and a
                // held message (Saved / failed) stays put until it expires.
                if (!capturing && System.currentTimeMillis() > statusHoldUntil)
                    status("Streaming  ${w}x$h  — press START")
            }
            if (streaming) handler.postDelayed(this, FRAME_INTERVAL_MS)
        }
    }

    private fun toggleRecord() {
        if (!capturing) {
            capturing = true
            recButton.text = "STOP"
            XrealNative.nativeSetRecordTap(true)
            recExec.execute { runCapture() }
        } else {
            // Stop producing; the writer flushes the ring, finalizes the zip,
            // then re-enables the button from its own thread.
            capturing = false
            XrealNative.nativeSetRecordTap(false)
            recButton.isEnabled = false
            recButton.text = "finalizing…"
        }
    }

    // ---- capture (writer thread) --------------------------------------------

    /** Runs entirely on [recExec]: owns all zip I/O so no locking is needed. */
    private fun runCapture() {
        val dims = XrealNative.nativeRawDims()
        val w = dims ushr 16
        val h = dims and 0xFFFF
        val pairBytes = 2 * w * h

        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val seq = "xreal_$stamp"
        val dir = captureDir()
        val zipFile = File(dir, "$seq.zip")
        // Scratch files stay in app cache: they are deleted on finalize and have
        // no business appearing in the user's dataset folder mid-capture.
        val tmpDir = cacheDir
        val imuTmp = File(tmpDir, "$seq.imu.tmp")
        val csvTmp = File(tmpDir, "$seq.csv.tmp")

        // Reusable buffers (allocated once per capture).
        val pairDirect = ByteBuffer.allocateDirect(pairBytes).order(ByteOrder.nativeOrder())
        val pairHeap = ByteArray(pairBytes)
        // imu.bin records are little-endian; native writes native-order bytes
        // (LE on the device). LE order here also makes getLong() read ts_ns.
        val imuDirect = ByteBuffer.allocateDirect(IMU_BATCH * 32).order(ByteOrder.LITTLE_ENDIAN)
        val imuHeap = ByteArray(IMU_BATCH * 32)

        var frameIdx = 0
        var frameFirstTs = 0L
        var frameLastTs = 0L
        var imuCount = 0L
        var imuFirstTs = 0L
        var imuLastTs = 0L
        var lastStatusMs = 0L

        var zip: ZipOutputStream? = null
        var imuOut: OutputStream? = null
        var csvOut: OutputStream? = null
        try {
            // Grab the factory calibration up front (fetched over the IMU
            // channel during nativeStart; ready by the time recording starts).
            val configBytes: ByteArray? = XrealNative.nativeGetConfig()
            val calibText: String? = configBytes?.let { buildCalibText(it) }

            // NO_COMPRESSION: L8 camera noise barely deflates, and we want cheap
            // CPU + a frames.raw that extracts byte-identical. The pack's own
            // "compressed=0" flag still holds (frames.raw content is raw L8).
            val z = ZipOutputStream(BufferedOutputStream(FileOutputStream(zipFile), 1 shl 20))
            z.setLevel(Deflater.NO_COMPRESSION)
            zip = z
            val iout = BufferedOutputStream(FileOutputStream(imuTmp), 1 shl 16)
            imuOut = iout
            val cout = BufferedOutputStream(FileOutputStream(csvTmp), 1 shl 16)
            csvOut = cout

            // frames.raw is the one huge stream: open it as the first entry and
            // write each pair into it live so a long capture never buffers in
            // RAM. Small entries (csv/imu/calib/meta/…) are folded in at stop.
            z.putNextEntry(ZipEntry("frames.raw"))

            // Drain both streams until the user stops; then one final flush so
            // the last frames still queued in the native ring are captured.
            while (capturing) {
                imuCount = drainImu(imuDirect, imuHeap, iout,
                    imuCount, { imuFirstTs = it }, { imuLastTs = it })
                frameIdx = drainFrames(pairDirect, pairHeap, pairBytes, z, cout, frameIdx,
                    { if (frameFirstTs == 0L) frameFirstTs = it }, { frameLastTs = it })

                val now = System.currentTimeMillis()
                if (now - lastStatusMs >= 250) {
                    lastStatusMs = now
                    val mb = frameIdx.toLong() * pairBytes / (1L shl 20)
                    status("● REC  ${w}x$h  pairs=$frameIdx  imu=$imuCount  ${mb}MB")
                }
                try { Thread.sleep(POLL_MS) } catch (_: InterruptedException) {}
            }
            // Final flush of whatever is still queued after the tap was cut.
            imuCount = drainImu(imuDirect, imuHeap, iout,
                imuCount, { imuFirstTs = it }, { imuLastTs = it })
            frameIdx = drainFrames(pairDirect, pairHeap, pairBytes, z, cout, frameIdx,
                { if (frameFirstTs == 0L) frameFirstTs = it }, { frameLastTs = it })
            z.closeEntry()   // frames.raw complete

            iout.flush(); iout.close(); imuOut = null
            cout.flush(); cout.close(); csvOut = null

            // Remaining pack entries (order is irrelevant to pack readers).
            copyFileEntry(z, "frames.csv", csvTmp)
            copyFileEntry(z, "imu.bin", imuTmp)
            if (calibText != null) writeTextEntry(z, "calib.txt", calibText)
            if (configBytes != null) writeBytesEntry(z, "config.json", configBytes)

            val fps = rateHz(frameIdx.toLong(), frameFirstTs, frameLastTs)
            val imuHz = rateHz(imuCount, imuFirstTs, imuLastTs)
            val drops = XrealNative.nativeRecordDrops()
            // Machine-readable intrinsics: same numbers as calib.txt, but keyed
            // and typed so a dataset consumer never has to parse bench text.
            configBytes?.let { cb ->
                buildIntrinsicsJson(cb, seq, w, h, fps, imuHz, frameIdx, imuCount)
                    ?.let { writeTextEntry(z, "intrinsics.json", it) }
            }
            writeTextEntry(z, "meta.txt", buildMeta(seq, w, h, fps, imuHz, frameIdx))
            writeTextEntry(z, "README.txt",
                buildReadme(seq, w, h, fps, imuHz, frameIdx, imuCount, drops,
                    calibText != null))
            z.close(); zip = null

            imuTmp.delete(); csvTmp.delete()
            val finalIdx = frameIdx
            val finalImu = imuCount
            val where = if (haveSharedStorage()) "/sdcard/$DATASET_DIR" else "app storage"
            handler.post {
                statusHoldUntil = System.currentTimeMillis() + 8000
                status("Saved $seq.zip -> $where  ($finalIdx pairs, $finalImu imu" +
                    (if (drops > 0) ", $drops dropped" else "") + ")")
                onCaptureFinished()
            }
        } catch (e: Exception) {
            try { zip?.close() } catch (_: Exception) {}
            try { imuOut?.close() } catch (_: Exception) {}
            try { csvOut?.close() } catch (_: Exception) {}
            imuTmp.delete(); csvTmp.delete()
            handler.post {
                statusHoldUntil = System.currentTimeMillis() + 8000
                status("Capture failed: ${e.message}")
                onCaptureFinished()
            }
        }
    }

    private fun onCaptureFinished() {
        capturing = false
        recButton.text = "START"
        recButton.isEnabled = streaming
    }

    /** Drain queued raw stereo pairs into the open frames.raw entry. Returns
     *  the next frame index. */
    private inline fun drainFrames(
        pairDirect: ByteBuffer, pairHeap: ByteArray, pairBytes: Int,
        zip: ZipOutputStream, csvOut: OutputStream, startIdx: Int,
        onFirstTs: (Long) -> Unit, onLastTs: (Long) -> Unit
    ): Int {
        var idx = startIdx
        while (true) {
            pairDirect.clear()
            val ts = XrealNative.nativeGrabRawPair(pairDirect)
            if (ts == 0L) break
            pairDirect.position(0)
            pairDirect.get(pairHeap, 0, pairBytes)
            zip.write(pairHeap, 0, pairBytes)
            csvOut.write("$ts,$idx\n".toByteArray(Charsets.US_ASCII))
            onFirstTs(ts); onLastTs(ts)
            idx++
        }
        return idx
    }

    /** Drain queued IMU samples (raw 32-byte records) into imu.bin. Returns the
     *  new cumulative sample count. */
    private inline fun drainImu(
        imuDirect: ByteBuffer, imuHeap: ByteArray, imuOut: OutputStream,
        startCount: Long, onFirstTs: (Long) -> Unit, onLastTs: (Long) -> Unit
    ): Long {
        var count = startCount
        while (true) {
            imuDirect.clear()
            val k = XrealNative.nativeGrabImuBatch(imuDirect)
            if (k <= 0) break
            val bytes = k * 32
            imuDirect.position(0)
            imuDirect.get(imuHeap, 0, bytes)
            imuOut.write(imuHeap, 0, bytes)
            if (count == 0L) onFirstTs(imuDirect.getLong(0))
            onLastTs(imuDirect.getLong((k - 1) * 32))
            count += k
            if (k < IMU_BATCH) break   // ring drained
        }
        return count
    }

    // ---- pack text builders --------------------------------------------------

    /** Parse the factory calibration JSON (mirrors MainActivity.setupAlignment)
     *  and hand it to native for exact bench-format calib.txt. */
    private fun buildCalibText(raw: ByteArray): String? {
        val params = try {
            val cfg = JSONObject(String(raw))
            val disp = cfg.getJSONObject("display")
            val cams = cfg.getJSONObject("SLAM_camera")
            val p = FloatArray(82)
            var o = 0
            for (eye in arrayOf("left", "right")) {
                // left eye pairs with device_1 (= cam1, the physical left camera)
                val cam = cams.getJSONObject(if (eye == "left") "device_1" else "device_2")
                for (arr in arrayOf(disp.getJSONArray("k_${eye}_display"),
                                    disp.getJSONArray("target_q_${eye}_display"),
                                    cam.getJSONArray("imu_q_cam"),
                                    cam.getJSONArray("fc"),
                                    cam.getJSONArray("cc"),
                                    cam.getJSONArray("kc"),
                                    cam.getJSONArray("imu_p_cam"))) {
                    for (i in 0 until arr.length()) p[o++] = arr.getDouble(i).toFloat()
                }
            }
            if (o != 72) return null
            try {
                val imu = cfg.getJSONObject("IMU").getJSONObject("device_1")
                for (arr in arrayOf(imu.getJSONArray("gyro_bias"),
                                    imu.getJSONArray("accel_bias"),
                                    imu.getJSONArray("imu_noises"))) {
                    for (i in 0 until arr.length()) p[o++] = arr.getDouble(i).toFloat()
                }
                if (o != 82) o = 72
            } catch (e: Exception) {
                o = 72   // vision-only calib still valid; noises fall back to 0
            }
            p.copyOf(o)
        } catch (e: Exception) {
            return null
        }
        return XrealNative.nativeFormatCalib(params, ALIGN_VARIANT)
    }

    /**
     * intrinsics.json — the calibration as structured data.
     *
     * Read straight from the factory blob's own field names rather than from
     * the flattened float array [buildCalibText] builds, so the values keep
     * their provenance and no array-layout assumption can silently corrupt
     * them. Distortion is the native fisheye624 (calib.txt carries the kb4
     * refit the on-device VIO actually runs); both describe the same camera.
     */
    private fun buildIntrinsicsJson(raw: ByteArray, seq: String, w: Int, h: Int,
                                    fps: Double, imuHz: Double, nFrames: Int,
                                    nImu: Long): String? = try {
        val cfg = JSONObject(String(raw))
        val cams = cfg.getJSONObject("SLAM_camera")
        val out = JSONObject()
        out.put("dataset", "xreal")
        out.put("seq", seq)
        out.put("device", "XREAL Air 2 Ultra")
        out.put("width", w)
        out.put("height", h)
        out.put("fps", fps)
        out.put("n_frames", nFrames)
        out.put("n_imu", nImu)
        out.put("camera_model", "fisheye624")
        out.put("extrinsics_convention", "p_imu = R(imu_q_cam) * p_cam + imu_p_cam")
        out.put("quaternion_order", "xyzw")

        val camsOut = JSONArray()
        // left = device_1 = physical left camera = first half of each frames.raw
        // pair; keep that mapping explicit, it is the one thing a consumer
        // cannot recover from the data itself.
        for ((name, dev) in listOf("left" to "device_1", "right" to "device_2")) {
            val c = cams.getJSONObject(dev)
            val fc = c.getJSONArray("fc")
            val cc = c.getJSONArray("cc")
            camsOut.put(JSONObject().apply {
                put("name", name)
                put("device", dev)
                put("fx", fc.getDouble(0))
                put("fy", fc.getDouble(1))
                put("cx", cc.getDouble(0))
                put("cy", cc.getDouble(1))
                put("distortion", c.getJSONArray("kc"))
                put("imu_q_cam", c.getJSONArray("imu_q_cam"))
                put("imu_p_cam", c.getJSONArray("imu_p_cam"))
            })
        }
        out.put("cameras", camsOut)

        // IMU block is optional: some units ship a vision-only calibration.
        try {
            val imu = cfg.getJSONObject("IMU").getJSONObject("device_1")
            out.put("imu", JSONObject().apply {
                put("rate_hz", imuHz)
                put("gyro_units", "deg/s")
                put("accel_units", "g")
                put("gyro_bias", imu.getJSONArray("gyro_bias"))
                put("accel_bias", imu.getJSONArray("accel_bias"))
                put("noises", imu.getJSONArray("imu_noises"))
                put("noises_order", "[gyro_n, gyro_bias_rw, accel_n, accel_bias_rw] (SI)")
            })
        } catch (_: Exception) {
            out.put("imu", JSONObject.NULL)
        }
        out.toString(2)
    } catch (e: Exception) {
        null
    }

    private fun buildMeta(seq: String, w: Int, h: Int, fps: Double, imuHz: Double,
                          nFrames: Int): String =
        "dataset=xreal\n" +
        "seq=$seq\n" +
        "W=$w\n" +
        "H=$h\n" +
        "fps=${round3(fps)}\n" +
        "imu_hz=${round3(imuHz)}\n" +
        "compressed=0\n" +
        "n_frames=$nFrames\n"

    private fun buildReadme(seq: String, w: Int, h: Int, fps: Double, imuHz: Double,
                            nFrames: Int, nImu: Long, drops: Int, haveCalib: Boolean): String =
        """xreal_datarecorder capture: $seq

This is a benchmark-style replay pack (see bench/host/pack_common.py) captured
from XREAL Air 2 Ultra glasses.

  meta.txt     dataset/seq/W/H/fps/imu_hz/compressed/n_frames (key=value)
  calib.txt    ${if (haveCalib) "model kb4, per-eye pinhole/dist/q_xyzw/p + IMU noises"
                 else "ABSENT (factory calibration blob was not available)"}
  frames.csv   header-less "ts_ns,idx", one line per stereo pair
  frames.raw   L8 gray pairs [left ${w}x${h}][right ${w}x${h}] per frames.csv line
  imu.bin      32-byte LE records: i64 ts_ns, f32 gyro_dps[3], f32 accel_g[3]
  intrinsics.json  ${if (haveCalib) "per-eye fx/fy/cx/cy + fisheye624 distortion, cam->IMU q/p,\n               IMU biases/noises, geometry and measured rates"
                     else "ABSENT (factory calibration blob was not available)"}
  config.json  raw factory calibration blob (provenance / re-derivation)

Geometry: W=$w H=$h, $nFrames pairs @ ${round3(fps)} fps, $nImu imu @ ${round3(imuHz)} Hz.
${if (drops > 0) "WARNING: $drops stereo pairs were dropped (writer fell behind).\n" else ""}
TIMESTAMPS: frames.csv ts_ns and imu.bin ts_ns share ONE clock (the glasses'
64-bit IMU ns clock). Each frame ts is the camera exposure stamp unwrapped
against that clock, so IMU can be integrated between frames with no cross-clock
alignment. left = physical left camera (cam1); frames.raw and calib.txt agree.

UNITS: gyro deg/s, accel g (= 9.80665 m/s^2) — identical to a pack's imu.bin.
calib.txt: fisheye624 refit to Kannala-Brandt kb4 (the model the on-device VIO
uses); q_xyzw/p are camera->IMU (p_imu = R(q)*p_cam + p); noises are SI
[gyro_n, gyro_bias_rw, accel_n, accel_bias_rw]. intrinsics.json carries the
SAME calibration but with the factory's native fisheye624 distortion and its
original field names — use calib.txt to reproduce the on-device VIO, and
intrinsics.json to build your own pipeline.

frames.raw are the conditioned L8 frames the VIO actually consumed (descrambled,
FPN/vignette-corrected, contrast-stretched) — i.e. exactly what a replay feeds
xr_slam_push_pair, matching the semantics of an EuRoC/TUM/MSD pack's frames.raw.

BENCHMARK USE: this pack has no gt.tum (a headset has no external mocap), so it
replays but is not ATE-scored. dataset=xreal is not in pack_common.DATASETS;
either add "xreal" to that tuple (imu_hz/fps math is generic) or drop these
files into an msd-typed pack dir. Everything else is already 1:1 with a pack.
"""

    private fun round3(v: Double): String =
        if (v.isFinite()) String.format(Locale.US, "%.3f", v) else "0.000"

    private fun rateHz(n: Long, firstTs: Long, lastTs: Long): Double =
        if (n > 1 && lastTs > firstTs) (n - 1) * 1e9 / (lastTs - firstTs) else 0.0

    private fun writeTextEntry(zip: ZipOutputStream, name: String, text: String) {
        zip.putNextEntry(ZipEntry(name))
        zip.write(text.toByteArray(Charsets.US_ASCII))
        zip.closeEntry()
    }

    private fun writeBytesEntry(zip: ZipOutputStream, name: String, bytes: ByteArray) {
        zip.putNextEntry(ZipEntry(name))
        zip.write(bytes)
        zip.closeEntry()
    }

    private fun copyFileEntry(zip: ZipOutputStream, name: String, src: File) {
        zip.putNextEntry(ZipEntry(name))
        FileInputStream(src).use { it.copyTo(zip, 1 shl 16) }
        zip.closeEntry()
    }

    // ---- lifecycle -----------------------------------------------------------

    private fun stopStreaming(msg: String) {
        if (!streaming) return
        streaming = false
        // Stop producing capture data before tearing down the native stream;
        // the writer thread flushes the ring and finalizes the zip.
        capturing = false
        XrealNative.nativeSetRecordTap(false)
        handler.removeCallbacks(pollFrame)
        XrealNative.nativeStop()
        connection?.close(); connection = null
        recButton.text = "START"; recButton.isEnabled = false
        status(msg)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        connect(null)
    }

    override fun onDestroy() {
        super.onDestroy()
        try { unregisterReceiver(usbReceiver) } catch (_: Exception) {}
        stopStreaming("Stopped")
        recExec.shutdown()   // lets an in-flight finalize complete
    }

    private fun status(s: String) = runOnUiThread { statusText.text = s }
}
