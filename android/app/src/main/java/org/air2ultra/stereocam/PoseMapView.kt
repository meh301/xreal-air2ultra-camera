package org.air2ultra.stereocam

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * 3D pose/map view: a fixed isometric look at the world frame with a ground
 * grid, the headset drawn as a camera frustum at its current pose, and a
 * breadcrumb trail of past positions.
 *
 * Orientation comes from the on-device AHRS; position is zero until the
 * Basalt VIO backend lands (docs/VSLAM.md), at which point the same update()
 * starts moving the frustum and growing the trail — the view needs no change.
 */
class PoseMapView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    private val q = floatArrayOf(1f, 0f, 0f, 0f)   // wxyz, body -> world
    private val p = floatArrayOf(0f, 0f, 0f)       // meters, world
    private var tracked = 0
    private var depthMs = 0f
    private var flags = 0
    private var haveState = false

    private val trail = ArrayList<FloatArray>()    // sampled positions

    // accumulated landmark cloud (world xyz triplets)
    private var cloud = FloatArray(0)
    private var cloudN = 0

    /** [pts] = xyz triplets, [n] = point count. Call from the UI thread. */
    fun setCloud(pts: FloatArray, n: Int) {
        if (cloud.size < n * 3) cloud = FloatArray(n * 3)
        pts.copyInto(cloud, 0, 0, n * 3)
        cloudN = n
    }

    // live IMU readout (factory frame) for on-device frame diagnosis
    private val imuA = FloatArray(3)
    private val imuG = FloatArray(3)
    private var haveImu = false

    /** Latest remapped IMU sample: accel in g, gyro in deg/s. */
    fun setImuDebug(a: FloatArray, g: FloatArray) {
        a.copyInto(imuA)
        g.copyInto(imuG)
        haveImu = true
    }

    private var kfCount = 0
    private var trackedR = 0

    /** [quat] wxyz body->world, [pos] meters. Call from the UI thread. */
    fun update(quat: FloatArray, pos: FloatArray, trackedCount: Int,
               depthMillis: Float, stateFlags: Int, keyframes: Int = 0,
               trackedRight: Int = 0) {
        quat.copyInto(q)
        pos.copyInto(p)
        tracked = trackedCount
        depthMs = depthMillis
        flags = stateFlags
        kfCount = keyframes
        trackedR = trackedRight
        haveState = true
        val last = trail.lastOrNull()
        val moved = last == null || run {
            val dx = pos[0] - last[0]; val dy = pos[1] - last[1]; val dz = pos[2] - last[2]
            sqrt(dx * dx + dy * dy + dz * dz) > 0.02f
        }
        if (moved) {
            trail.add(pos.copyOf())
            if (trail.size > 512) trail.removeAt(0)
        }
        invalidate()
    }

    fun reset() {
        trail.clear()
        invalidate()
    }

    // interactive camera: orbit with one finger, pan/zoom with two
    private var viewYaw = -35.0                    // deg, about world up
    private var viewElev = 25.0                    // deg, downward tilt
    private var viewZoom = 1f
    private var panX = 0f
    private var panY = 0f
    private var yawS = 0f; private var yawC = 1f
    private var elS = 0f; private var elC = 1f

    /** World (z up) -> screen px, orthographic. The camera sits ABOVE and
     * behind, pitched down by viewElev — same viewing hemisphere as the
     * python scope. (The first version had the y-term sign flipped, which
     * is still a proper camera but one looking up from BELOW the ground
     * plane: with no depth cues the triad then reads left-handed — "right
     * points left".) Screen y grows downward, hence the leading minus. */
    private fun project(wx: Float, wy: Float, wz: Float, out: FloatArray) {
        val x1 = wx * yawC - wy * yawS
        val y1 = wx * yawS + wy * yawC
        val s = height / 9f * viewZoom             // ~9 m of world at zoom 1
        out[0] = width / 2f + panX + x1 * s
        out[1] = height * 0.55f + panY - (y1 * elS + wz * elC) * s
    }

    // touch state
    private var lastX = 0f
    private var lastY = 0f
    private var lastSpan = 0f
    private var lastMidX = 0f
    private var lastMidY = 0f
    private var twoFinger = false

    override fun onTouchEvent(event: MotionEvent): Boolean {
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                lastX = event.x; lastY = event.y
                twoFinger = false
            }
            MotionEvent.ACTION_POINTER_DOWN -> if (event.pointerCount == 2) {
                twoFinger = true
                lastSpan = hypot(event.getX(1) - event.getX(0),
                                 event.getY(1) - event.getY(0))
                lastMidX = (event.getX(0) + event.getX(1)) * 0.5f
                lastMidY = (event.getY(0) + event.getY(1)) * 0.5f
            }
            MotionEvent.ACTION_MOVE -> {
                if (event.pointerCount >= 2) {
                    // two fingers: pinch to zoom, drag the midpoint to pan
                    val span = hypot(event.getX(1) - event.getX(0),
                                     event.getY(1) - event.getY(0))
                    val midX = (event.getX(0) + event.getX(1)) * 0.5f
                    val midY = (event.getY(0) + event.getY(1)) * 0.5f
                    if (lastSpan > 10f) {
                        viewZoom = (viewZoom * (span / lastSpan))
                            .coerceIn(0.15f, 12f)
                    }
                    panX += midX - lastMidX
                    panY += midY - lastMidY
                    lastSpan = span; lastMidX = midX; lastMidY = midY
                    invalidate()
                } else if (!twoFinger) {
                    // one finger: orbit
                    viewYaw += (event.x - lastX) * 0.45
                    viewElev = (viewElev + (event.y - lastY) * 0.3)
                        .coerceIn(-5.0, 89.0)
                    lastX = event.x; lastY = event.y
                    invalidate()
                }
            }
            MotionEvent.ACTION_POINTER_UP -> if (event.pointerCount == 2) {
                // keep the remaining finger from orbiting with a jump
                val keep = 1 - event.actionIndex
                lastX = event.getX(keep); lastY = event.getY(keep)
            }
            MotionEvent.ACTION_UP -> {
                if (!twoFinger && event.eventTime - event.downTime < 200 &&
                    hypot(event.x - lastX, event.y - lastY) < 8f) {
                    performClick()
                }
                twoFinger = false
            }
        }
        return true
    }

    override fun performClick(): Boolean {
        super.performClick()
        return true
    }

    /** rotate a body vector into the world frame by q (Hamilton wxyz) */
    private fun rotate(v: FloatArray, out: FloatArray) {
        val w = q[0]; val x = q[1]; val y = q[2]; val z = q[3]
        out[0] = (1 - 2 * (y * y + z * z)) * v[0] + 2 * (x * y - w * z) * v[1] + 2 * (x * z + w * y) * v[2]
        out[1] = 2 * (x * y + w * z) * v[0] + (1 - 2 * (x * x + z * z)) * v[1] + 2 * (y * z - w * x) * v[2]
        out[2] = 2 * (x * z - w * y) * v[0] + 2 * (y * z + w * x) * v[1] + (1 - 2 * (x * x + y * y)) * v[2]
    }

    private val gridPaint = Paint().apply {
        color = Color.rgb(0, 70, 40); strokeWidth = 1f
    }
    private val axisPaint = Paint().apply { strokeWidth = 3f }
    private val frustumPaint = Paint().apply {
        color = Color.rgb(0, 255, 102); strokeWidth = 3f
        style = Paint.Style.STROKE; isAntiAlias = true
    }
    private val trailPaint = Paint().apply {
        color = Color.rgb(120, 120, 120); strokeWidth = 2f; isAntiAlias = true
    }
    private val cloudPaint = Paint().apply {
        color = Color.rgb(90, 200, 255); strokeWidth = 2f
        strokeCap = Paint.Cap.ROUND
    }
    private val textPaint = Paint().apply {
        color = Color.rgb(0, 255, 102); textSize = 24f
        typeface = android.graphics.Typeface.MONOSPACE; isAntiAlias = true
    }
    private val dimTextPaint = Paint(textPaint).apply {
        color = Color.rgb(110, 110, 110)
    }

    private val pa = FloatArray(2)
    private val pb = FloatArray(2)
    private val tmp = FloatArray(3)
    private val tmpW = FloatArray(3)

    override fun onDraw(canvas: Canvas) {
        yawS = sin(Math.toRadians(viewYaw)).toFloat()
        yawC = cos(Math.toRadians(viewYaw)).toFloat()
        elS = sin(Math.toRadians(viewElev)).toFloat()
        elC = cos(Math.toRadians(viewElev)).toFloat()
        canvas.drawColor(Color.rgb(8, 10, 8))

        // ground grid, 1 m cells on the world horizontal plane
        for (i in -4..4) {
            project(i.toFloat(), -4f, 0f, pa); project(i.toFloat(), 4f, 0f, pb)
            canvas.drawLine(pa[0], pa[1], pb[0], pb[1], gridPaint)
            project(-4f, i.toFloat(), 0f, pa); project(4f, i.toFloat(), 0f, pb)
            canvas.drawLine(pa[0], pa[1], pb[0], pb[1], gridPaint)
        }
        // world axes at the origin (x red, y green, z blue), 0.5 m
        project(0f, 0f, 0f, pa)
        project(0.5f, 0f, 0f, pb)
        axisPaint.color = Color.rgb(200, 60, 60)
        canvas.drawLine(pa[0], pa[1], pb[0], pb[1], axisPaint)
        project(0f, 0.5f, 0f, pb)
        axisPaint.color = Color.rgb(60, 200, 60)
        canvas.drawLine(pa[0], pa[1], pb[0], pb[1], axisPaint)
        project(0f, 0f, 0.5f, pb)
        axisPaint.color = Color.rgb(80, 120, 255)
        canvas.drawLine(pa[0], pa[1], pb[0], pb[1], axisPaint)

        // landmark cloud (Basalt's estimator landmarks, accumulated)
        cloudPaint.strokeWidth = (2f * viewZoom).coerceIn(1.5f, 5f)
        for (i in 0 until cloudN) {
            project(cloud[i * 3], cloud[i * 3 + 1], cloud[i * 3 + 2], pa)
            if (pa[0] < -20 || pa[0] > width + 20 ||
                pa[1] < -20 || pa[1] > height + 20) continue
            canvas.drawPoint(pa[0], pa[1], cloudPaint)
        }

        // trail
        for (i in 1 until trail.size) {
            val a = trail[i - 1]; val b = trail[i]
            project(a[0], a[1], a[2], pa)
            project(b[0], b[1], b[2], pb)
            canvas.drawLine(pa[0], pa[1], pb[0], pb[1], trailPaint)
        }

        // headset frustum: body +z is the look direction (camera forward)
        if (haveState) {
            val d = 0.7f   // frustum depth, m
            val corners = arrayOf(
                floatArrayOf(-0.35f * d, -0.25f * d, d),
                floatArrayOf(0.35f * d, -0.25f * d, d),
                floatArrayOf(0.35f * d, 0.25f * d, d),
                floatArrayOf(-0.35f * d, 0.25f * d, d)
            )
            val scr = Array(4) { FloatArray(2) }
            project(p[0], p[1], p[2], pa)
            for (i in 0..3) {
                rotate(corners[i], tmpW)
                project(p[0] + tmpW[0], p[1] + tmpW[1], p[2] + tmpW[2], scr[i])
                canvas.drawLine(pa[0], pa[1], scr[i][0], scr[i][1], frustumPaint)
            }
            for (i in 0..3) {
                val j = (i + 1) and 3
                canvas.drawLine(scr[i][0], scr[i][1], scr[j][0], scr[j][1], frustumPaint)
            }
            // "up" tick on the frustum's top edge
            tmp[0] = 0f; tmp[1] = -0.35f * d; tmp[2] = d
            rotate(tmp, tmpW)
            project(p[0] + tmpW[0], p[1] + tmpW[1], p[2] + tmpW[2], pb)
            canvas.drawCircle(pb[0], pb[1], 5f, frustumPaint)

            // labeled body axes (factory frame): R = its RIGHT (red),
            // D = its DOWN (green), F = its FORWARD (blue). The labels
            // remove the viewing-side ambiguity of a wireframe frustum —
            // apparent yaw/roll direction flips when seen from the front,
            // which once poisoned a whole diagnostic round.
            project(p[0], p[1], p[2], pa)
            val bodyAxes = arrayOf(
                floatArrayOf(0.45f, 0f, 0f), floatArrayOf(0f, 0.45f, 0f),
                floatArrayOf(0f, 0f, 0.6f))
            val axisColors = intArrayOf(
                Color.rgb(255, 80, 80), Color.rgb(80, 255, 80),
                Color.rgb(100, 140, 255))
            val axisNames = arrayOf("R", "D", "F")
            for (i in 0..2) {
                rotate(bodyAxes[i], tmpW)
                project(p[0] + tmpW[0], p[1] + tmpW[1], p[2] + tmpW[2], pb)
                axisPaint.color = axisColors[i]
                canvas.drawLine(pa[0], pa[1], pb[0], pb[1], axisPaint)
                textPaint.color = axisColors[i]
                canvas.drawText(axisNames[i], pb[0] + 4f, pb[1] - 4f, textPaint)
            }
            textPaint.color = Color.rgb(0, 255, 102)
        }

        // state text
        val rect = if (flags and 2 != 0) "rect✓" else "rect…"
        val dep = if (flags and 1 != 0) "%.0fms".format(depthMs) else "off"
        canvas.drawText("trk=%d|%d  map=%d  kf=%d  depth=%s  %s".format(
            tracked, trackedR, cloudN, kfCount, dep, rect), 12f, 30f, textPaint)
        val backend = if (flags and 8 != 0)
            "pose: Basalt VIO (6-DoF)  p=[%.2f %.2f %.2f]m".format(p[0], p[1], p[2])
        else
            "pose: AHRS — Basalt not active"
        canvas.drawText(backend, 12f, 58f, dimTextPaint)
        if (haveImu) {
            canvas.drawText(
                "a=(%+.2f %+.2f %+.2f)g".format(imuA[0], imuA[1], imuA[2]),
                12f, 86f, textPaint)
            canvas.drawText(
                "w=(%+6.1f %+6.1f %+6.1f)°/s".format(imuG[0], imuG[1], imuG[2]),
                12f, 114f, textPaint)
        }
    }
}
