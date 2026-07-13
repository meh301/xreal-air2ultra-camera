package org.air2ultra.stereocam

import android.app.Activity
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbDeviceConnection
import android.hardware.usb.UsbManager
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger

/**
 * xreal_datarecorder — a STRIPPED-DOWN sibling of MainActivity for gathering IR
 * stereo-pair training data. It only opens the glasses' camera and shows/saves
 * the raw LEFT|RIGHT pair. It deliberately never calls nativeSetAlignment, so
 * Basalt/SLAM/depth never start (xr_slam_start is gated behind alignment) — no
 * tracking, minimal battery. One button toggles auto-snapshot every 2 s into
 * Pictures/xreal_datarecorder. Shares libxrealcam.so + XrealNative with the main
 * app but touches none of its code.
 */
class RecorderActivity : Activity() {

    companion object {
        private const val XREAL_VID = 0x3318
        private const val XREAL_PID = 0x0426
        private const val ACTION_USB_PERMISSION = "org.air2ultra.stereocam.RECORDER_USB_PERMISSION"
        private const val SNAP_INTERVAL_MS = 2000L      // auto-snapshot cadence (1-5 s window)
        private const val FRAME_INTERVAL_MS = 33L       // ~30 fps preview poll
        private const val ALBUM = "xreal_datarecorder"  // Pictures/<ALBUM>
        // USB access to the composite (UVC+audio) device needs these granted on 10+
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
    private var recording = false
    private var bitmap: Bitmap? = null
    private var lastSnapAt = 0L
    private val snapCount = AtomicInteger(0)

    private val frameBuffer =
        ByteBuffer.allocateDirect(1280 * 640 * 4).order(ByteOrder.nativeOrder())
    private val handler = Handler(Looper.getMainLooper())
    private val saveExec = Executors.newSingleThreadExecutor()

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

    private val pollFrame = object : Runnable {
        override fun run() {
            val packed = XrealNative.nativeGrabFrame(frameBuffer)
            if (packed != 0L) {
                val w = (packed ushr 48).toInt()
                val h = ((packed ushr 32) and 0xFFFF).toInt()
                var bm = bitmap
                if (bm == null || bm.width != w || bm.height != h) {
                    bm = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
                    bitmap = bm; imageView.setImageBitmap(bm)
                }
                frameBuffer.position(0)
                bm.copyPixelsFromBuffer(frameBuffer)
                imageView.invalidate()
                val now = System.currentTimeMillis()
                if (recording && now - lastSnapAt >= SNAP_INTERVAL_MS) {
                    lastSnapAt = now
                    val snap = bm.copy(Bitmap.Config.ARGB_8888, false)   // detach from the live buffer
                    saveExec.execute { savePng(snap) }
                }
                status(if (recording) "● REC  ${w}x$h  saved ${snapCount.get()}"
                       else "Streaming  ${w}x$h  — press START")
            }
            if (streaming) handler.postDelayed(this, FRAME_INTERVAL_MS)
        }
    }

    private fun toggleRecord() {
        recording = !recording
        recButton.text = if (recording) "STOP" else "START"
        if (recording) lastSnapAt = 0L          // snap immediately on start
    }

    private fun savePng(bm: Bitmap) {
        val name = "xreal_" + SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date()) +
            "_%04d.png".format(snapCount.get())
        try {
            if (Build.VERSION.SDK_INT >= 29) {
                val values = ContentValues().apply {
                    put(MediaStore.Images.Media.DISPLAY_NAME, name)
                    put(MediaStore.Images.Media.MIME_TYPE, "image/png")
                    put(MediaStore.Images.Media.RELATIVE_PATH,
                        Environment.DIRECTORY_PICTURES + "/" + ALBUM)
                }
                val uri = contentResolver.insert(
                    MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return
                contentResolver.openOutputStream(uri)?.use {
                    bm.compress(Bitmap.CompressFormat.PNG, 100, it)
                }
            } else {
                val dir = File(Environment.getExternalStoragePublicDirectory(
                    Environment.DIRECTORY_PICTURES), ALBUM).apply { mkdirs() }
                FileOutputStream(File(dir, name)).use {
                    bm.compress(Bitmap.CompressFormat.PNG, 100, it)
                }
            }
            snapCount.incrementAndGet()
        } catch (e: Exception) {
            runOnUiThread { status("save failed: ${e.message}") }
        } finally {
            bm.recycle()
        }
    }

    private fun stopStreaming(msg: String) {
        if (!streaming) return
        streaming = false; recording = false
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
        saveExec.shutdown()
    }

    private fun status(s: String) = runOnUiThread { statusText.text = s }
}
