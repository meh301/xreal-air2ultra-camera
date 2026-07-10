package org.air2ultra.stereocam

import android.view.Surface
import java.nio.ByteBuffer

/** JNI bindings to libxrealcam.so (libusb + libuvc + descrambler). */
object XrealNative {
    init {
        System.loadLibrary("xrealcam")
    }

    /**
     * Start streaming from an already-opened USB device. The fd comes from
     * [android.hardware.usb.UsbDeviceConnection.getFileDescriptor]; the
     * connection object must stay open for as long as the stream runs.
     * Returns 0 on success, a negative libuvc error code otherwise.
     */
    external fun nativeStart(fd: Int): Int

    external fun nativeStop()

    /** Swap the glasses' two eye views (debug). */
    external fun nativeSetSwap(swap: Boolean)

    /** Drop all tracked features and restart the SLAM front end. */
    external fun nativeSlamReset()

    /** Show/hide the tracked features (phone pane + glasses overlay). */
    external fun nativeSetShowPoints(on: Boolean)

    /** Enable/disable stereo depth computation (tracking keeps running). */
    external fun nativeSetDepth(on: Boolean)

    /**
     * Show/hide the passthrough camera image on the glasses (stereo mode).
     * Off = pure AR: only the tracked features, floating over the real
     * world seen through the lenses.
     */
    external fun nativeSetShowCam(on: Boolean)

    /**
     * Copy the newest pose/SLAM state into [buf] (direct ByteBuffer >= 40
     * bytes, native order): f32 quat_wxyz[4], f32 pos_m[3], i32 tracked
     * features, f32 depth_ms, u32 flags (bit0 depth on, bit1 rectification
     * ready, bit2 orientation valid). Position stays zero until the Basalt
     * backend lands. Returns false while no orientation exists yet.
     */
    external fun nativeGrabPose(buf: ByteBuffer): Boolean

    /**
     * true: glasses in per-eye stereo display mode with the calibrated
     * world-aligned warp; false: mirror display mode showing the plain
     * side-by-side framebuffer. Switches the display over the MCU channel.
     */
    external fun nativeSetStereoMode(stereo: Boolean)

    /**
     * Attach the glasses' [Surface]; native code then presents each composed
     * stereo pair directly from the capture thread (lowest latency). Pass
     * null before the surface is destroyed.
     */
    external fun nativeSetSurface(surface: Surface?)

    /** The factory calibration JSON fetched from the glasses, or null. */
    external fun nativeGetConfig(): ByteArray?

    /**
     * Enable the world-aligned per-eye passthrough and the stereo
     * rectification. 72 floats — per eye (left, then right): display K[9],
     * display quaternion[4], camera quaternion[4], fc[2], cc[2], kc[12],
     * p_cam[3], all from the calibration JSON.
     */
    external fun nativeSetAlignment(params: FloatArray)

    /** Cycle the rotation-convention variant (0..3) used by the alignment. */
    external fun nativeSetAlignVariant(variant: Int)

    /**
     * Enable/disable the IMU timewarp (default on): every present
     * counter-rotates the stereo view by the head rotation since the
     * frame's exposure, and the newest frame is re-warped at display rate
     * between camera frames (asynchronous timewarp).
     */
    external fun nativeSetTimewarp(on: Boolean)

    /**
     * Copy the newest composed side-by-side RGBA frame into [buf] (a direct
     * ByteBuffer, capacity >= 1280*640*4). Returns 0 when there is no new
     * frame, else (w shl 48) or (h shl 32) or (fps*10 shl 16) or counter.
     */
    external fun nativeGrabFrame(buf: ByteBuffer): Long

    /**
     * Copy the newest IMU state into [buf] (direct ByteBuffer >= 56 bytes,
     * native order): u64 ts_ns, f32 gyro_dps[3], f32 accel_g[3],
     * f32 quat_wxyz[4] (host-side Madgwick), f32 rate_hz, u32 has_quat.
     * Returns false when nothing new arrived since the last call.
     */
    external fun nativeGrabImu(buf: ByteBuffer): Boolean

    /**
     * Drain pending IMU samples (the full 1 kHz stream) into [buf] (direct
     * ByteBuffer, native order), 32 bytes per sample: u64 ts_ns,
     * f32 gyro_dps[3], f32 accel_g[3]. Returns the number of samples
     * written; call every UI tick to keep up.
     */
    external fun nativeGrabImuBatch(buf: ByteBuffer): Int
}
