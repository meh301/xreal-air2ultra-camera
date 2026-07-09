package org.air2ultra.stereocam

import android.app.Activity
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Bitmap
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbDeviceConnection
import android.hardware.usb.UsbManager
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.Looper
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
import kotlin.math.asin
import kotlin.math.atan2

/**
 * Live stereo preview of the XREAL Air 2 Ultra tracking cameras.
 *
 * Android exposes no camera API for external UVC devices, so the app opens
 * the glasses through the USB host API and hands the raw file descriptor to
 * native code (libusb + libuvc), which streams, descrambles and denoises the
 * frames. See docs/PROTOCOL.md in the repository root.
 */
class MainActivity : Activity() {

    companion object {
        const val XREAL_VID = 0x3318
        const val XREAL_PID = 0x0426
        const val ACTION_USB_PERMISSION = "org.air2ultra.stereocam.USB_PERMISSION"
        const val FRAME_INTERVAL_MS = 33L
    }

    private lateinit var usbManager: UsbManager
    private lateinit var imageView: ImageView
    private lateinit var statusView: TextView
    private lateinit var imuStatusView: TextView
    private lateinit var toggleButton: Button

    private var connection: UsbDeviceConnection? = null
    private var streaming = false
    private var showClean = true
    private var bitmap: Bitmap? = null
    private val frameBuffer: ByteBuffer = ByteBuffer.allocateDirect(1280 * 640 * 4)
    private val imuBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(64).order(ByteOrder.nativeOrder())
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
                statusView.text = getString(
                    R.string.status_streaming,
                    counter, fps, if (showClean) "CLEAN" else "SCRAMBLED"
                )
            }
            if (XrealNative.nativeGrabImu(imuBuffer)) {
                imuBuffer.position(0)
                val ts = imuBuffer.long
                val g = FloatArray(3) { imuBuffer.float }
                val a = FloatArray(3) { imuBuffer.float }
                val q = FloatArray(4) { imuBuffer.float }
                val rate = imuBuffer.float
                val hasQuat = imuBuffer.int != 0
                imuStatusView.text = if (hasQuat) {
                    val (yaw, pitch, roll) = eulerDeg(q)
                    String.format(
                        Locale.US,
                        "IMU %4.0f Hz  gyro(%+6.1f,%+6.1f,%+6.1f)deg/s  " +
                            "acc(%+5.2f,%+5.2f,%+5.2f)g  ypr(%+5.0f,%+4.0f,%+5.0f)deg",
                        rate, g[0], g[1], g[2], a[0], a[1], a[2], yaw, pitch, roll
                    )
                } else {
                    String.format(
                        Locale.US,
                        "IMU %4.0f Hz  capturing gyro bias - hold still... ts=%d",
                        rate, ts
                    )
                }
            }
            if (streaming) handler.postDelayed(this, FRAME_INTERVAL_MS)
        }
    }

    private fun eulerDeg(q: FloatArray): Triple<Double, Double, Double> {
        val (w, x, y, z) = q.map { it.toDouble() }
        val yaw = Math.toDegrees(atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        val pitch = Math.toDegrees(asin((2 * (w * y - z * x)).coerceIn(-1.0, 1.0)))
        val roll = Math.toDegrees(atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
        return Triple(yaw, pitch, roll)
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
        imuStatusView = findViewById(R.id.imu_status)
        toggleButton = findViewById(R.id.toggle)

        toggleButton.setOnClickListener {
            showClean = !showClean
            XrealNative.nativeSetClean(showClean)
            toggleButton.text =
                getString(if (showClean) R.string.show_raw else R.string.show_clean)
        }
        findViewById<Button>(R.id.snapshot).setOnClickListener { saveSnapshot() }

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

    private fun isXreal(d: UsbDevice) = d.vendorId == XREAL_VID && d.productId == XREAL_PID

    private fun connectToGlasses(intent: Intent?) {
        if (streaming) return
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
        unregisterReceiver(usbReceiver)
    }
}
