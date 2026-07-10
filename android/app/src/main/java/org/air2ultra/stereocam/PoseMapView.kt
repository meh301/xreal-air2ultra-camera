package org.air2ultra.stereocam

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import kotlin.math.cos
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

    /** [quat] wxyz body->world, [pos] meters. Call from the UI thread. */
    fun update(quat: FloatArray, pos: FloatArray, trackedCount: Int,
               depthMillis: Float, stateFlags: Int) {
        quat.copyInto(q)
        pos.copyInto(p)
        tracked = trackedCount
        depthMs = depthMillis
        flags = stateFlags
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

    // fixed isometric view: yaw about the up axis, then a downward tilt
    private val yawS = sin(Math.toRadians(-35.0)).toFloat()
    private val yawC = cos(Math.toRadians(-35.0)).toFloat()
    private val elS = sin(Math.toRadians(25.0)).toFloat()
    private val elC = cos(Math.toRadians(25.0)).toFloat()

    /** world (x fwd-ish, y left-ish, z up) -> screen px, orthographic */
    private fun project(wx: Float, wy: Float, wz: Float, out: FloatArray) {
        val x1 = wx * yawC - wy * yawS
        val y1 = wx * yawS + wy * yawC
        val s = height / 9f                        // ~9 m of world vertically
        out[0] = width / 2f + x1 * s
        out[1] = height * 0.55f + (y1 * elS - wz * elC) * s
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
        }

        // state text
        val rect = if (flags and 2 != 0) "rect✓" else "rect…"
        val dep = if (flags and 1 != 0) "%.0fms".format(depthMs) else "off"
        canvas.drawText("trk=%d  depth=%s  %s".format(tracked, dep, rect),
            12f, 30f, textPaint)
        val backend = if (flags and 8 != 0)
            "pose: Basalt VIO (6-DoF)  p=[%.2f %.2f %.2f]m".format(p[0], p[1], p[2])
        else
            "pose: AHRS orientation — Basalt not loaded"
        canvas.drawText(backend, 12f, 58f, dimTextPaint)
    }
}
