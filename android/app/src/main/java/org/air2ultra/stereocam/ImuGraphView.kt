package org.air2ultra.stereocam

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.util.AttributeSet
import android.view.View
import java.util.Locale
import kotlin.math.abs
import kotlin.math.max

/**
 * Rolling 3-channel oscilloscope panel for IMU data (the Android sibling of
 * python/xreal_imu_scope.py). Feed it triplets at 1 kHz via [addSamples];
 * it keeps the last [capacity] samples and autoscales symmetrically.
 */
class ImuGraphView(context: Context, attrs: AttributeSet? = null) : View(context, attrs) {

    var title = ""
    var unit = ""
    var rateHz = 0f

    private val capacity = 4000                     // ~4 s at 1 kHz
    private val data = FloatArray(capacity * 3)
    private var head = 0                            // next write slot
    private var count = 0

    private val channelColors =
        intArrayOf(Color.rgb(255, 80, 80), Color.rgb(80, 220, 80), Color.rgb(60, 200, 255))
    private val channelNames = arrayOf("x", "y", "z")
    private val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 2f
    }
    private val gridPaint = Paint().apply { color = Color.rgb(60, 60, 60); strokeWidth = 1f }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(160, 160, 160)
        textSize = 28f
        typeface = android.graphics.Typeface.MONOSPACE
    }
    private val path = Path()

    /** Append `n` triplets from `src` (x0,y0,z0, x1,y1,z1, ...). */
    fun addSamples(src: FloatArray, n: Int) {
        for (i in 0 until n) {
            val base = (head % capacity) * 3
            data[base] = src[i * 3]
            data[base + 1] = src[i * 3 + 1]
            data[base + 2] = src[i * 3 + 2]
            head++
        }
        count = minOf(count + n, capacity)
    }

    override fun onDraw(canvas: Canvas) {
        canvas.drawColor(Color.rgb(24, 24, 24))
        val w = width.toFloat()
        val h = height.toFloat()
        val mid = h / 2

        if (count >= 2) {
            var lim = 1e-3f
            val start = head - count
            for (i in start until head) {
                val b = (i % capacity) * 3
                lim = max(lim, max(abs(data[b]), max(abs(data[b + 1]), abs(data[b + 2]))))
            }
            lim *= 1.15f
            val scale = (h / 2 - 8) / lim

            canvas.drawLine(0f, mid, w, mid, gridPaint)
            for (frac in floatArrayOf(0.5f, 1f)) {
                canvas.drawLine(0f, mid - frac * lim * scale, w, mid - frac * lim * scale, gridPaint)
                canvas.drawLine(0f, mid + frac * lim * scale, w, mid + frac * lim * scale, gridPaint)
            }

            // decimate to at most one point per pixel column
            val step = max(1, count / max(1, width))
            for (ch in 0..2) {
                path.rewind()
                var first = true
                var i = start
                while (i < head) {
                    val v = data[(i % capacity) * 3 + ch]
                    val x = (i - start).toFloat() / (count - 1) * (w - 1)
                    val y = (mid - v * scale).coerceIn(0f, h - 1)
                    if (first) { path.moveTo(x, y); first = false } else path.lineTo(x, y)
                    i += step
                }
                linePaint.color = channelColors[ch]
                canvas.drawPath(path, linePaint)
            }

            val b = ((head - 1) % capacity) * 3
            val header = String.format(
                Locale.US, "%s  %4.0f Hz  (%+8.2f, %+8.2f, %+8.2f) %s   ±%.3g",
                title, rateHz, data[b], data[b + 1], data[b + 2], unit, lim
            )
            textPaint.color = Color.rgb(0, 255, 102)
            canvas.drawText(header, 10f, 34f, textPaint)
            var lx = 10f
            for (ch in 0..2) {
                textPaint.color = channelColors[ch]
                canvas.drawText(channelNames[ch], lx, h - 12f, textPaint)
                lx += 40f
            }
        } else {
            textPaint.color = Color.rgb(160, 160, 160)
            canvas.drawText("$title — waiting for data…", 10f, 34f, textPaint)
        }
    }
}
