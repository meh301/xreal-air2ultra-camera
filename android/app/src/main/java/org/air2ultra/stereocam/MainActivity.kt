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
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
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
import android.view.View
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
 * Draws the stereo pair onto the glasses' own display: the left half of the
 * source bitmap (left camera) aspect-fit into the left half of the screen,
 * the right half into the right half. With the glasses in 3D/SBS mode each
 * half lands on one eye — camera passthrough.
 */
class PassthroughView(context: Context) : View(context) {
    var bitmap: Bitmap? = null
    var swap = false
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

    override fun onDraw(canvas: Canvas) {
        canvas.drawColor(Color.BLACK)
        val bm = bitmap ?: return
        val srcHalf = bm.width / 2
        val dstHalf = width / 2f
        for (eye in 0..1) {
            val srcIdx = if (swap) 1 - eye else eye
            val src = Rect(srcIdx * srcHalf, 0, (srcIdx + 1) * srcHalf, bm.height)
            val scale = minOf(dstHalf / srcHalf, height.toFloat() / bm.height)
            val dw = srcHalf * scale
            val dh = bm.height * scale
            val ox = eye * dstHalf + (dstHalf - dw) / 2
            val oy = (height - dh) / 2
            canvas.drawBitmap(bm, src, RectF(ox, oy, ox + dw, oy + dh), paint)
        }
    }
}

/** Fullscreen [PassthroughView] on an external (presentation) display. */
class GlassesPresentation(context: Context, display: Display) :
    Presentation(context, display) {
    val view = PassthroughView(context)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(view)
    }
}

/**
 * Live stereo preview of the XREAL Air 2 Ultra tracking cameras.
 *
 * Android exposes no camera API for external UVC devices, so the app opens
 * the glasses through the USB host API and hands the raw file descriptor to
 * native code (libusb + libuvc), which streams, descrambles and denoises the
 * frames. Requires the current glasses firmware (MCU 12.1.00.498+ — update at
 * https://ota.xreal.com/ultra-update?version=1). See docs/PROTOCOL.md.
 *
 * When the glasses' display is available (DisplayPort alt-mode), the clean
 * stereo pair is additionally rendered onto it as SBS passthrough.
 */
class MainActivity : Activity() {

    companion object {
        const val XREAL_VID = 0x3318
        const val XREAL_PID = 0x0426
        const val ACTION_USB_PERMISSION = "org.air2ultra.stereocam.USB_PERMISSION"
        const val FRAME_INTERVAL_MS = 33L
        const val REQ_RUNTIME_PERMS = 1
        // Android 10+ refuses the USB grant for a device containing a camera
        // and microphones unless the app holds these runtime permissions
        val RUNTIME_PERMS = arrayOf(Manifest.permission.CAMERA,
                                    Manifest.permission.RECORD_AUDIO)
    }

    private lateinit var usbManager: UsbManager
    private lateinit var imageView: ImageView
    private lateinit var statusView: TextView
    private lateinit var gyroGraph: ImuGraphView
    private lateinit var accelGraph: ImuGraphView
    private lateinit var toggleButton: Button

    private var connection: UsbDeviceConnection? = null
    private var streaming = false
    private var showClean = true
    private var swapEyes = false
    private var bitmap: Bitmap? = null
    private var presentation: GlassesPresentation? = null
    private lateinit var displayManager: DisplayManager
    private val frameBuffer: ByteBuffer = ByteBuffer.allocateDirect(1280 * 640 * 4)
    private val imuBatch: ByteBuffer =                 // 256 samples x 32 B
        ByteBuffer.allocateDirect(256 * 32).order(ByteOrder.nativeOrder())
    private val gyroTriples = FloatArray(256 * 3)
    private val accelTriples = FloatArray(256 * 3)
    private var imuSampleCount = 0
    private var imuRateT0 = 0L
    private val handler = Handler(Looper.getMainLooper())

    private val pollFrame = object : Runnable {
        override fun run() {
            val packed = XrealNative.nativeGrabFrame(frameBuffer)
            if (packed != 0L) {
                val w = (packed ushr 48).toInt()
                val h = ((packed ushr 32) and 0xFFFF).toInt()
                val fps = ((packed ushr 16) and 0xFFFF).toInt() / 10.0
                val counter = (packed and 0xFF).toInt()
                var bm = bitmap
                if (bm == null || bm.width != w || bm.height != h) {
                    bm = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
                    bitmap = bm
                    imageView.setImageBitmap(bm)
                }
                frameBuffer.position(0)
                bm.copyPixelsFromBuffer(frameBuffer)
                imageView.invalidate()
                presentation?.view?.let {
                    it.bitmap = bm
                    it.invalidate()
                }
                statusView.text = getString(
                    R.string.status_streaming,
                    counter, fps, if (showClean) "CLEAN" else "SCRAMBLED"
                )
            }
            val n = XrealNative.nativeGrabImuBatch(imuBatch)
            if (n > 0) {
                imuBatch.position(0)
                for (i in 0 until n) {
                    imuBatch.long                      // ts_ns (graphs don't need it)
                    for (c in 0..2) gyroTriples[i * 3 + c] = imuBatch.float
                    for (c in 0..2) accelTriples[i * 3 + c] = imuBatch.float
                }
                gyroGraph.addSamples(gyroTriples, n)
                accelGraph.addSamples(accelTriples, n)

                imuSampleCount += n
                val now = System.nanoTime()
                if (imuRateT0 == 0L) imuRateT0 = now
                if (now - imuRateT0 >= 1_000_000_000L) {
                    val rate = imuSampleCount * 1e9f / (now - imuRateT0)
                    gyroGraph.rateHz = rate
                    accelGraph.rateHz = rate
                    imuSampleCount = 0
                    imuRateT0 = now
                }
                gyroGraph.invalidate()
                accelGraph.invalidate()
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
        presentation = GlassesPresentation(this, display).also {
            it.view.swap = swapEyes
            it.show()
        }
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
        gyroGraph = findViewById(R.id.gyro_graph)
        accelGraph = findViewById(R.id.accel_graph)
        gyroGraph.title = "gyro"
        gyroGraph.unit = "deg/s"
        accelGraph.title = "accel"
        accelGraph.unit = "g"
        toggleButton = findViewById(R.id.toggle)

        toggleButton.setOnClickListener {
            showClean = !showClean
            XrealNative.nativeSetClean(showClean)
            toggleButton.text =
                getString(if (showClean) R.string.show_raw else R.string.show_clean)
        }
        findViewById<Button>(R.id.snapshot).setOnClickListener { saveSnapshot() }
        findViewById<Button>(R.id.swap).setOnClickListener {
            swapEyes = !swapEyes
            presentation?.view?.let {
                it.swap = swapEyes
                it.invalidate()
            }
        }

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
        statusView.text = getString(R.string.status_starting)
        handler.post(pollFrame)
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
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val dir = getExternalFilesDir(Environment.DIRECTORY_PICTURES) ?: filesDir
        val file = File(dir, "xreal_$stamp.png")
        FileOutputStream(file).use { bm.compress(Bitmap.CompressFormat.PNG, 100, it) }
        Toast.makeText(this, getString(R.string.snapshot_saved, file.path), Toast.LENGTH_LONG)
            .show()
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
