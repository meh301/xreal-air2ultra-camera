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

    private val trail = ArrayList<FloatArray>()    // sampled positions (~2 cm gate)
    private val TRAIL_MAX = 4096                    // ~80 m+ of path history
    private val trailScreen = FloatArray(TRAIL_MAX * 4)  // segment endpoints, batched draw

    // accumulated landmark cloud (world xyz triplets)
    private var cloud = FloatArray(0)
    private var cloudN = 0
    private var cloudScreen = FloatArray(0)        // projected xy pairs, batched draw

    /** [pts] = xyz triplets, [n] = point count. Call from the UI thread. */
    fun setCloud(pts: FloatArray, n: Int) {
        if (cloud.size < n * 3) cloud = FloatArray(n * 3)
        if (cloudScreen.size < n * 2) cloudScreen = FloatArray(n * 2)
        pts.copyInto(cloud, 0, 0, n * 3)
        cloudN = n
    }

    // live IMU readout (factory frame) for on-device frame diagnosis
    private val imuA = FloatArray(3)
    private val imuG = FloatArray(3)
    private var haveImu = false

    // general log line (fps, connection state, thermal), drawn along the
    // bottom edge — fed by MainActivity
    private var statusLine = ""

    fun setStatusLine(s: String) {
        if (s != statusLine) {
            statusLine = s
            invalidate()
        }
    }

    /** Latest remapped IMU sample: accel in g, gyro in deg/s. */
    fun setImuDebug(a: FloatArray, g: FloatArray) {
        a.copyInto(imuA)
        g.copyInto(imuG)
        haveImu = true
    }

    private var kfCount = 0
    private var trackedR = 0
    private var loopCount = 0
    private val loopAt = FloatArray(3)
    private var loopFlashMs = 0L        // wall time of the latest loop event

    private var verPairs = 0
    private var verInliers = 0
    private var verOutcome = 0

    /** [quat] wxyz body->world, [pos] meters. Call from the UI thread. */
    fun update(quat: FloatArray, pos: FloatArray, trackedCount: Int,
               depthMillis: Float, stateFlags: Int, keyframes: Int = 0,
               trackedRight: Int = 0, loops: Int = 0,
               loopPosition: FloatArray? = null,
               vPairs: Int = 0, vInliers: Int = 0, vOutcome: Int = 0) {
        verPairs = vPairs
        verInliers = vInliers
        verOutcome = vOutcome
        quat.copyInto(q)
        pos.copyInto(p)
        tracked = trackedCount
        depthMs = depthMillis
        flags = stateFlags
        kfCount = keyframes
        trackedR = trackedRight
        if (loops > loopCount && loopPosition != null) {
            loopPosition.copyInto(loopAt)
            loopFlashMs = System.currentTimeMillis()
        }
        loopCount = loops
        haveState = true
        if (follow) pos.copyInto(pivot)
        val last = trail.lastOrNull()
        val moved = last == null || run {
            val dx = pos[0] - last[0]; val dy = pos[1] - last[1]; val dz = pos[2] - last[2]
            sqrt(dx * dx + dy * dy + dz * dz) > 0.02f
        }
        if (moved) {
            trail.add(pos.copyOf())
            if (trail.size > TRAIL_MAX) trail.removeAt(0)
        }
        invalidate()
    }

    fun reset() {
        trail.clear()
        pivot.fill(0f)
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

    // follow mode: the orbit/zoom pivot passively tracks the user's
    // position (translation only — orbit, pan and zoom still work), so a
    // long walk never leaves the frustum stranded far from the origin
    private var follow = true
    private val pivot = FloatArray(3)
    private val btnRect = android.graphics.RectF()

    /** World (z up) -> screen px, orthographic. The camera sits ABOVE and
     * behind, pitched down by viewElev — same viewing hemisphere as the
     * python scope. (The first version had the y-term sign flipped, which
     * is still a proper camera but one looking up from BELOW the ground
     * plane: with no depth cues the triad then reads left-handed — "right
     * points left".) Screen y grows downward, hence the leading minus. */
    private fun project(wx: Float, wy: Float, wz: Float, out: FloatArray) {
        val dx = wx - pivot[0]
        val dy = wy - pivot[1]
        val dz = wz - pivot[2]
        val x1 = dx * yawC - dy * yawS
        val y1 = dx * yawS + dy * yawC
        val s = height / 9f * viewZoom             // ~9 m of world at zoom 1
        out[0] = width / 2f + panX + x1 * s
        out[1] = height * 0.55f + panY - (y1 * elS + dz * elC) * s
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
                    if (btnRect.contains(event.x, event.y)) {
                        follow = !follow
                        if (follow) {
                            p.copyInto(pivot)
                            panX = 0f; panY = 0f
                        }
                        invalidate()
                    } else {
                        performClick()
                    }
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
    private val loopPaint = Paint().apply {
        color = Color.rgb(255, 210, 60); strokeWidth = 3f
        style = Paint.Style.STROKE; isAntiAlias = true
    }
    private val textPaint = Paint().apply {
        color = Color.rgb(0, 255, 102); textSize = 24f
        typeface = android.graphics.Typeface.MONOSPACE; isAntiAlias = true
    }
    private val btnPaint = Paint().apply { isAntiAlias = true }
    private val btnTextPaint = Paint(textPaint).apply { textSize = 25f }

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

        // ground grid, 1 m cells on the world horizontal plane, centered
        // near the pivot (snapped to whole meters so lines never swim)
        val gcx = kotlin.math.floor(pivot[0])
        val gcy = kotlin.math.floor(pivot[1])
        for (i in -4..4) {
            project(gcx + i, gcy - 4f, 0f, pa); project(gcx + i, gcy + 4f, 0f, pb)
            canvas.drawLine(pa[0], pa[1], pb[0], pb[1], gridPaint)
            project(gcx - 4f, gcy + i, 0f, pa); project(gcx + 4f, gcy + i, 0f, pb)
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

        // landmark cloud (keyframe-derived map). Project into the screen-xy
        // scratch and draw in ONE batched call — per-point drawPoint chugged
        // the UI thread once the cap grew past a few thousand.
        cloudPaint.strokeWidth = (2f * viewZoom).coerceIn(1.5f, 5f)
        var pn = 0
        for (i in 0 until cloudN) {
            project(cloud[i * 3], cloud[i * 3 + 1], cloud[i * 3 + 2], pa)
            if (pa[0] < -20 || pa[0] > width + 20 ||
                pa[1] < -20 || pa[1] > height + 20) continue
            cloudScreen[pn++] = pa[0]
            cloudScreen[pn++] = pa[1]
        }
        if (pn > 0) canvas.drawPoints(cloudScreen, 0, pn, cloudPaint)

        // trail — batched polyline (one drawLines call for the whole path)
        if (trail.size > 1) {
            var tn = 0
            project(trail[0][0], trail[0][1], trail[0][2], pa)
            for (i in 1 until trail.size) {
                project(trail[i][0], trail[i][1], trail[i][2], pb)
                if (tn + 4 <= trailScreen.size) {
                    trailScreen[tn++] = pa[0]; trailScreen[tn++] = pa[1]
                    trailScreen[tn++] = pb[0]; trailScreen[tn++] = pb[1]
                }
                pa[0] = pb[0]; pa[1] = pb[1]
            }
            if (tn > 0) canvas.drawLines(trailScreen, 0, tn, trailPaint)
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

        // state text: two lines, up to ~3/4 width (the Follow button owns
        // the top-right corner)
        val rect = if (flags and 2 != 0) "rect✓" else "rect…"
        val dep = if (flags and 1 != 0) "%.0fms".format(depthMs) else "off"
        val src = if (flags and 8 != 0) "" else "  AHRS-only"
        // last verification attempt: names the stage that blocked (or
        // allowed) a snap, without needing adb
        val ver = when (verOutcome) {
            1 -> "ver=match<bar"
            2 -> "ver=pairs:%d".format(verPairs)
            3 -> "ver=inliers:%d/%d".format(verInliers, verPairs)
            4 -> "ver=CAPPED %d/%d".format(verInliers, verPairs)
            5 -> "ver=SNAP %d/%d".format(verInliers, verPairs)
            6 -> "ver=aligned %d/%d".format(verInliers, verPairs)
            7 -> "ver=confirm? %d/%d".format(verInliers, verPairs)
            else -> "ver=—"
        }
        // recovery lifecycle (flags bits 4-5): storage stays frozen while
        // LOST until a closure is verified — not a fixed shake timer
        val rec = when ((flags shr 4) and 3) {
            1 -> "  LOST"
            2 -> "  REC↺"
            else -> ""
        }
        canvas.drawText("trk=%d|%d  map=%d  kf=%d  loop=%d  %s%s".format(
            tracked, trackedR, cloudN, kfCount, loopCount, ver, rec),
            12f, 30f, textPaint)
        canvas.drawText("p=[%+.2f %+.2f %+.2f]m  depth=%s  %s%s".format(
            p[0], p[1], p[2], dep, rect, src), 12f, 58f, textPaint)

        // loop/reloc event: pulsing ring at the matched keyframe (~3 s)
        val since = System.currentTimeMillis() - loopFlashMs
        if (loopFlashMs > 0 && since < 3000) {
            project(loopAt[0], loopAt[1], loopAt[2], pa)
            val t = since / 3000f
            loopPaint.alpha = ((1f - t) * 255).toInt()
            canvas.drawCircle(pa[0], pa[1], 10f + t * 40f, loopPaint)
            canvas.drawCircle(pa[0], pa[1], 4f, loopPaint)
        }
        // follow toggle, top right
        btnRect.set(width - 146f, 8f, width - 8f, 54f)
        btnPaint.color = if (follow) Color.rgb(22, 66, 38) else Color.rgb(30, 34, 30)
        canvas.drawRoundRect(btnRect, 10f, 10f, btnPaint)
        btnTextPaint.color =
            if (follow) Color.rgb(0, 255, 102) else Color.rgb(150, 150, 150)
        canvas.drawText(if (follow) "Follow✓" else "Follow✗",
                        btnRect.left + 13f, btnRect.centerY() + 9f, btnTextPaint)

        // general log line along the bottom (full viewer width)
        if (statusLine.isNotEmpty())
            canvas.drawText(statusLine, 12f, height - 12f, textPaint)
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
