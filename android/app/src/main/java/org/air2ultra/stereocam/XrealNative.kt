package org.air2ultra.stereocam

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

    /** true = descrambled/denoised view, false = raw scrambled view. */
    external fun nativeSetClean(clean: Boolean)

    /** Horizontal per-eye flip of the composed view (also affects the
     * glasses passthrough). */
    external fun nativeSetMirror(mirror: Boolean)

    /** Swap the two panes of the composed view (also affects the glasses
     * passthrough and snapshots). */
    external fun nativeSetSwap(swap: Boolean)

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
