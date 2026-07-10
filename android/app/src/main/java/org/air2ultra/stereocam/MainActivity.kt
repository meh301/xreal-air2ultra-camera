package org.air2ultra.stereocam

import android.Manifest
import android.app.Activity
import android.app.PendingIntent
import android.app.Presentation
import android.content.pm.PackageManager
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Bitmap
import android.hardware.display.DisplayManager
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbDeviceConnection
import android.hardware.usb.UsbManager
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.view.Display
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Fullscreen surface on the glasses' display (external/presentation display).
 * Native code renders composed stereo pairs straight into the Surface from
 * the UVC callback thread — no UI polling or bitmap copies on this path,
 * which keeps passthrough latency to a minimum. With the glasses in 3D/SBS
 * mode the left half lands on the left eye.
 */
class GlassesPresentation(context: Context, display: Display) :
    Presentation(context, display) {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val surfaceView = SurfaceView(context)
        surfaceView.holder.addCallback(object : SurfaceHolder.Callback {
            override fun surfaceCreated(holder: SurfaceHolder) {
                XrealNative.nativeSetSurface(holder.surface)
            }

            override fun surfaceChanged(holder: SurfaceHolder, format: Int,
                                        width: Int, height: Int) {
                XrealNative.nativeSetSurface(holder.surface)
            }

            override fun surfaceDestroyed(holder: SurfaceHolder) {
                XrealNative.nativeSetSurface(null)
            }
        })
        setContentView(surfaceView)
    }
}

/**
 * vSLAM research app for the XREAL Air 2 Ultra (research branch).
 *
 * Android exposes no camera API for external UVC devices, so the app opens
 * the glasses through the USB host API and hands the raw file descriptor to
 * native code (libusb + libuvc), which streams, descrambles and denoises the
 * frames, then runs the SLAM front end (rectification, feature tracking,
 * stereo depth — see docs/VSLAM.md). Requires the current glasses firmware
 * (MCU 12.1.00.498+ — update at https://ota.xreal.com/ultra-update?version=1).
 *
 * Phone UI: tracking pane | depth pane on top, the 3D pose/map below.
 * The glasses get the world-aligned passthrough with the tracked features
 * overlaid, rendered natively (front buffer + IMU timewarp, always on).
 */
class MainActivity : Activity() {

    companion object {
        const val XREAL_VID = 0x3318
        const val XREAL_PID = 0x0426
        const val ACTION_USB_PERMISSION = "org.air2ultra.stereocam.USB_PERMISSION"
        const val FRAME_INTERVAL_MS = 16L   // phone-UI poll; glasses get frames
                                            // pushed natively, not polled
        const val REQ_RUNTIME_PERMS = 1
        // Android 10+ refuses the USB grant for a device containing a camera
        // and microphones unless the app holds these runtime permissions
        val RUNTIME_PERMS = arrayOf(Manifest.permission.CAMERA,
                                    Manifest.permission.RECORD_AUDIO)
    }

    private lateinit var usbManager: UsbManager
    private lateinit var imageView: ImageView
    private lateinit var statusView: TextView
    private lateinit var poseMap: PoseMapView

    private var connection: UsbDeviceConnection? = null
    private var streaming = false
    private var stereoMode = true       // stereo+aligned vs plain SBS mirror
    private var showPoints = true
    private var depthOn = true
    private var eyeMode = 0             // glasses: 0 cam, 1 depth, 2 AR, 3 off
    private var paneMode = 0            // phone: 0 = L|depth, 1 = L|R
    private var bitmap: Bitmap? = null
    private var presentation: GlassesPresentation? = null
    private lateinit var displayManager: DisplayManager
    private var lastFrameAt = 0L        // watchdog: last time a frame arrived
    private var lastReconnectAt = 0L
    private var alignReady = false
    private val frameBuffer: ByteBuffer = ByteBuffer.allocateDirect(1280 * 640 * 4)
    private val poseBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(40).order(ByteOrder.nativeOrder())
    private val poseQ = FloatArray(4)
    private val poseP = FloatArray(3)
    private val imuBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(56).order(ByteOrder.nativeOrder())
    private val imuG = FloatArray(3)
    private val imuA = FloatArray(3)
    private val handler = Handler(Looper.getMainLooper())

    private val pollFrame = object : Runnable {
        override fun run() {
            // watchdog: a mid-stream device hiccup (e.g. DP alt-mode
            // renegotiation right after plug-in) silently kills every
            // transfer; a full reopen recovers without replugging
            val now = System.currentTimeMillis()
            if (streaming && now - lastFrameAt > 4000 &&
                now - lastReconnectAt > 6000) {
                lastReconnectAt = now
                lastFrameAt = 0L
                statusView.text = getString(R.string.status_reconnecting)
                stopStreaming()
                connectToGlasses(null)
                return   // connect flow restarts the poll loop
            }

            // live IMU readout for the frame diagnosis (factory frame)
            if (XrealNative.nativeGrabImu(imuBuffer)) {
                imuBuffer.position(8)              // skip u64 timestamp
                for (i in 0..2) imuG[i] = imuBuffer.float
                for (i in 0..2) imuA[i] = imuBuffer.float
                poseMap.setImuDebug(imuA, imuG)
            }

            // pose/SLAM state feeds both the 3D map and the status line
            var tracked = 0
            var depthMs = 0f
            if (XrealNative.nativeGrabPose(poseBuffer)) {
                poseBuffer.position(0)
                for (i in 0..3) poseQ[i] = poseBuffer.float
                for (i in 0..2) poseP[i] = poseBuffer.float
                tracked = poseBuffer.int
                depthMs = poseBuffer.float
                val flags = poseBuffer.int
                poseMap.update(poseQ, poseP, tracked, depthMs, flags)
            }

            val packed = XrealNative.nativeGrabFrame(frameBuffer)
            if (packed != 0L) {
                lastFrameAt = now
                val w = (packed ushr 48).toInt()
                val h = ((packed ushr 32) and 0xFFFF).toInt()
                val fps = ((packed ushr 16) and 0xFFFF).toInt() / 10.0f
                var bm = bitmap
                if (bm == null || bm.width != w || bm.height != h) {
                    bm = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
                    bitmap = bm
                    imageView.setImageBitmap(bm)
                }
                frameBuffer.position(0)
                bm.copyPixelsFromBuffer(frameBuffer)
                imageView.invalidate()
                statusView.text = getString(
                    R.string.status_streaming, tracked, fps, depthMs
                ) + when {
                    presentation == null -> "  [no ext display]"
                    !stereoMode -> "  [glasses sbs]"
                    alignReady -> "  [glasses stereo]"
                    else -> "  [glasses uncal]"
                }
            }
            if (streaming) handler.postDelayed(this, FRAME_INTERVAL_MS)
        }
    }

    private val displayListener = object : DisplayManager.DisplayListener {
        override fun onDisplayAdded(displayId: Int) = updatePresentation()
        override fun onDisplayRemoved(displayId: Int) = updatePresentation()
        override fun onDisplayChanged(displayId: Int) {}
    }

    /** Show the passthrough on the first external/presentation display. */
    private fun updatePresentation() {
        for (d in displayManager.displays) {
            android.util.Log.i("xrealcam",
                "display ${d.displayId}: '${d.name}' flags=0x${d.flags.toString(16)}")
        }
        val display = displayManager
            .getDisplays(DisplayManager.DISPLAY_CATEGORY_PRESENTATION)
            .firstOrNull()
        if (display == null) {
            presentation?.dismiss()
            presentation = null
            return
        }
        if (presentation?.display?.displayId == display.displayId) return
        presentation?.dismiss()
        presentation = GlassesPresentation(this, display).also { it.show() }
    }

    private val usbReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            when (intent.action) {
                ACTION_USB_PERMISSION -> {
                    val device: UsbDevice? = intent.getParcelableExtra(UsbManager.EXTRA_DEVICE)
                    if (intent.getBooleanExtra(UsbManager.EXTRA_PERMISSION_GRANTED, false)
                        && device != null
                    ) {
                        openDevice(device)
                    } else {
                        statusView.text = getString(R.string.status_permission_denied)
                    }
                }
                UsbManager.ACTION_USB_DEVICE_DETACHED -> {
                    val device: UsbDevice? = intent.getParcelableExtra(UsbManager.EXTRA_DEVICE)
                    if (device != null && isXreal(device)) {
                        stopStreaming()
                        statusView.text = getString(R.string.status_unplugged)
                    }
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        usbManager = getSystemService(Context.USB_SERVICE) as UsbManager
        imageView = findViewById(R.id.preview)
        statusView = findViewById(R.id.status)
        poseMap = findViewById(R.id.pose_map)

        findViewById<Button>(R.id.slam_reset).setOnClickListener {
            XrealNative.nativeSlamReset()
            poseMap.reset()
        }
        val ptsButton = findViewById<Button>(R.id.toggle_points)
        ptsButton.setOnClickListener {
            showPoints = !showPoints
            XrealNative.nativeSetShowPoints(showPoints)
            ptsButton.text =
                getString(if (showPoints) R.string.pts_on else R.string.pts_off)
        }
        val depButton = findViewById<Button>(R.id.toggle_depth)
        depButton.setOnClickListener {
            depthOn = !depthOn
            XrealNative.nativeSetDepth(depthOn)
            depButton.text =
                getString(if (depthOn) R.string.dep_on else R.string.dep_off)
        }
        val eyeLabels = intArrayOf(R.string.eye_cam, R.string.eye_dep,
                                   R.string.eye_ar, R.string.eye_off)
        val eyeButton = findViewById<Button>(R.id.eye_mode)
        eyeButton.setOnClickListener {
            // glasses view: camera -> depth -> AR points-only -> off
            eyeMode = (eyeMode + 1) % 4
            XrealNative.nativeSetEyeMode(eyeMode)
            if (eyeMode == 1 && !depthOn) {      // depth view forces SGM on
                depthOn = true
                findViewById<Button>(R.id.toggle_depth).text = getString(R.string.dep_on)
            }
            eyeButton.text = getString(eyeLabels[eyeMode])
        }
        val paneButton = findViewById<Button>(R.id.pane_mode)
        paneButton.setOnClickListener {
            paneMode = 1 - paneMode
            XrealNative.nativeSetPaneMode(paneMode)
            paneButton.text =
                getString(if (paneMode == 0) R.string.pane_ldep else R.string.pane_lr)
        }
        val modeButton = findViewById<Button>(R.id.view_mode)
        modeButton.setOnClickListener {
            // Stereo = glasses in per-eye SBS display mode + calibrated
            // world-aligned warp; SBS = mirror display mode, raw framebuffer
            stereoMode = !stereoMode
            XrealNative.nativeSetStereoMode(stereoMode)
            modeButton.text =
                getString(if (stereoMode) R.string.mode_sbs else R.string.mode_stereo)
        }
        findViewById<Button>(R.id.snapshot).setOnClickListener { saveSnapshot() }
        setupSlamConfig()

        displayManager = getSystemService(Context.DISPLAY_SERVICE) as DisplayManager
        displayManager.registerDisplayListener(displayListener, handler)
        updatePresentation()

        val filter = IntentFilter(ACTION_USB_PERMISSION).apply {
            addAction(UsbManager.ACTION_USB_DEVICE_DETACHED)
        }
        if (Build.VERSION.SDK_INT >= 33) {
            registerReceiver(usbReceiver, filter, Context.RECEIVER_NOT_EXPORTED)
        } else {
            @Suppress("UnspecifiedRegisterReceiverFlag")
            registerReceiver(usbReceiver, filter)
        }

        connectToGlasses(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        connectToGlasses(intent) // plugged in while the app was already open
    }

    override fun onRequestPermissionsResult(requestCode: Int,
                                            permissions: Array<out String>,
                                            grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode != REQ_RUNTIME_PERMS) return
        if (grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
            connectToGlasses(null)
        } else {
            statusView.text = getString(R.string.status_permissions_denied)
        }
    }

    private fun isXreal(d: UsbDevice) = d.vendorId == XREAL_VID && d.productId == XREAL_PID

    private fun connectToGlasses(intent: Intent?) {
        if (streaming) return
        val missing = RUNTIME_PERMS.filter {
            checkSelfPermission(it) != PackageManager.PERMISSION_GRANTED
        }
        if (missing.isNotEmpty()) {
            statusView.text = getString(R.string.status_need_permissions)
            requestPermissions(missing.toTypedArray(), REQ_RUNTIME_PERMS)
            return   // continues from onRequestPermissionsResult
        }
        // launched by the USB_DEVICE_ATTACHED intent filter? then permission
        // is already implicitly granted
        val fromIntent: UsbDevice? = intent?.getParcelableExtra(UsbManager.EXTRA_DEVICE)
        val device = fromIntent?.takeIf { isXreal(it) }
            ?: usbManager.deviceList.values.firstOrNull { isXreal(it) }
        if (device == null) {
            statusView.text = getString(R.string.status_plug_in)
            return
        }
        if (usbManager.hasPermission(device)) {
            openDevice(device)
        } else {
            statusView.text = getString(R.string.status_waiting_permission)
            val flags = if (Build.VERSION.SDK_INT >= 31) {
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_MUTABLE
            } else {
                PendingIntent.FLAG_UPDATE_CURRENT
            }
            val pi = PendingIntent.getBroadcast(
                this, 0, Intent(ACTION_USB_PERMISSION).setPackage(packageName), flags
            )
            usbManager.requestPermission(device, pi)
        }
    }

    private fun openDevice(device: UsbDevice) {
        val conn = usbManager.openDevice(device)
        if (conn == null) {
            statusView.text = getString(R.string.status_open_failed)
            return
        }
        connection = conn // must stay open while native code streams on its fd
        val rc = XrealNative.nativeStart(conn.fileDescriptor)
        if (rc != 0) {
            statusView.text = getString(R.string.status_native_error, rc)
            conn.close()
            connection = null
            return
        }
        streaming = true
        lastFrameAt = System.currentTimeMillis()   // arm the watchdog from start
        statusView.text = getString(R.string.status_starting)
        setupAlignment()
        handler.post(pollFrame)
    }

    /** Write Basalt's config files into the app dir: the phone-tuned
     * VioConfig (from assets, HMD baseline + realtime enforcement) and the
     * unified config that points at it and caps the TBB worker count so
     * the VIO shares the SoC with capture/depth/render. */
    private fun setupSlamConfig() {
        try {
            val vio = java.io.File(filesDir, "basalt_vio_config.json")
            assets.open("basalt_vio_config.json").use { src ->
                vio.outputStream().use { dst -> src.copyTo(dst) }
            }
            val toml = java.io.File(filesDir, "basalt.toml")
            toml.writeText(
                "show-gui=0\n" +
                "cam-calib=\"\"\n" +                 // calibration stays programmatic
                "config-path=\"${vio.absolutePath}\"\n" +
                "marg-data=\"\"\n" +
                "print-queue=0\n" +
                "use-double=0\n" +
                "deterministic=0\n" +
                "num-threads=3\n")
            XrealNative.nativeSetSlamConfig(toml.absolutePath)
            android.util.Log.i("xrealcam", "Basalt config staged: ${toml.absolutePath}")
        } catch (e: Exception) {
            android.util.Log.e("xrealcam", "Basalt config staging failed: $e")
        }
    }

    /** Parse the on-device factory calibration and enable the world-aligned
     * per-eye passthrough plus the stereo rectification (each eye is its own
     * calibrated virtual display; the camera positions give the baseline). */
    private fun setupAlignment() {
        alignReady = false
        val raw = XrealNative.nativeGetConfig() ?: return
        try {
            val cfg = org.json.JSONObject(String(raw))
            val disp = cfg.getJSONObject("display")
            val cams = cfg.getJSONObject("SLAM_camera")
            val params = FloatArray(82)
            var o = 0
            for (eye in arrayOf("left", "right")) {
                // left eye pairs with device_1 (= cam1, the left camera)
                val cam = cams.getJSONObject(if (eye == "left") "device_1" else "device_2")
                for (arr in arrayOf(disp.getJSONArray("k_${eye}_display"),
                                    disp.getJSONArray("target_q_${eye}_display"),
                                    cam.getJSONArray("imu_q_cam"),
                                    cam.getJSONArray("fc"),
                                    cam.getJSONArray("cc"),
                                    cam.getJSONArray("kc"),
                                    cam.getJSONArray("imu_p_cam"))) {
                    for (i in 0 until arr.length()) params[o++] = arr.getDouble(i).toFloat()
                }
            }
            if (o != 72) throw IllegalStateException("unexpected calibration shape ($o)")
            // factory IMU calibration: the raw stream is uncorrected, Basalt
            // needs the biases (rad/s, m/s^2) and the noise densities. If
            // this block is missing the vision-side calibration still loads.
            try {
                val imu = cfg.getJSONObject("IMU").getJSONObject("device_1")
                for (arr in arrayOf(imu.getJSONArray("gyro_bias"),
                                    imu.getJSONArray("accel_bias"),
                                    imu.getJSONArray("imu_noises"))) {
                    for (i in 0 until arr.length()) params[o++] = arr.getDouble(i).toFloat()
                }
                if (o != 82) throw IllegalStateException("shape $o")
            } catch (e: Exception) {
                android.util.Log.e("xrealcam", "IMU calibration block missing/odd: $e")
                o = 72
            }
            XrealNative.nativeSetAlignment(params.copyOf(o))
            alignReady = true
            android.util.Log.i("xrealcam", "alignment + rectification + IMU calibration enabled")
        } catch (e: Exception) {
            android.util.Log.e("xrealcam", "calibration parse failed: $e")
        }
    }

    private fun stopStreaming() {
        if (streaming) {
            streaming = false
            handler.removeCallbacks(pollFrame)
            XrealNative.nativeStop()
        }
        connection?.close()
        connection = null
    }

    private fun saveSnapshot() {
        val bm = bitmap ?: return
        val name = "xreal_" +
            SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date()) + ".png"
        val where: String
        if (Build.VERSION.SDK_INT >= 29) {
            // MediaStore: lands in the gallery, no storage permission needed
            val values = android.content.ContentValues().apply {
                put(android.provider.MediaStore.Images.Media.DISPLAY_NAME, name)
                put(android.provider.MediaStore.Images.Media.MIME_TYPE, "image/png")
                put(android.provider.MediaStore.Images.Media.RELATIVE_PATH,
                    Environment.DIRECTORY_PICTURES + "/XREAL")
            }
            val uri = contentResolver.insert(
                android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            if (uri == null) {
                Toast.makeText(this, R.string.snapshot_failed, Toast.LENGTH_LONG).show()
                return
            }
            contentResolver.openOutputStream(uri)?.use {
                bm.compress(Bitmap.CompressFormat.PNG, 100, it)
            }
            where = "Pictures/XREAL/$name"
        } else {
            // pre-Android-10 fallback: classic public directory
            val dir = File(Environment.getExternalStoragePublicDirectory(
                Environment.DIRECTORY_PICTURES), "XREAL")
            dir.mkdirs()
            val file = File(dir, name)
            FileOutputStream(file).use { bm.compress(Bitmap.CompressFormat.PNG, 100, it) }
            android.media.MediaScannerConnection.scanFile(
                this, arrayOf(file.path), null, null)
            where = file.path
        }
        Toast.makeText(this, getString(R.string.snapshot_saved, where),
                       Toast.LENGTH_LONG).show()
    }

    override fun onDestroy() {
        super.onDestroy()
        stopStreaming()
        presentation?.dismiss()
        presentation = null
        displayManager.unregisterDisplayListener(displayListener)
        unregisterReceiver(usbReceiver)
    }
}
