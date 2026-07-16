/* SLAM Benchmark Explorer — zero-dependency SPA.
 * Data: data/{results,baselines,published,ledger,meta}.json + data/traj/<seq>_<arm>.json */
"use strict";

const ARMS = ["bad", "vpr", "megaloc", "xfeat", "xvpr", "xmegaloc"];
const ARM_LABEL = {
  bad: "BAD/TEBLID", vpr: "BAD + EigenPlaces", megaloc: "BAD + MegaLoc",
  xfeat: "XFeat", xvpr: "XFeat + EigenPlaces", xmegaloc: "XFeat + MegaLoc",
  "bad-vio": "VIO only", "xfeat-vio": "VIO only (XFeat)",
  okvis2lc0: "OKVIS2", okvis2lc1: "OKVIS2+LC",
  orb3lc0: "ORB-SLAM3", orb3lc1: "ORB-SLAM3+LC", openvinslc0: "OpenVINS",
  xdense: "XFeat-dense + MegaLoc", xdenselg6: "XFeat-dense + MegaLoc + LGlue",
  lmdesc: "dense+LGlue + landmark bank", clip15: "dense+LGlue, clip-15 probes",
};
/* deterministic color for arms without a CSS variable (new A/B arms) */
function armColor(a) {
  const c = col(a);
  if (c && c !== "#888") return c;
  let h = 0;
  for (const ch of a) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return `hsl(${h % 360},62%,50%)`;
}
/* every plot key the trajectory panel may offer, in display order */
const TRAJ_KEYS = ["bad-vio", "xfeat-vio", ...ARMS,
  "okvis2lc0", "okvis2lc1", "orb3lc0", "orb3lc1", "openvinslc0"];
const TRAJ_COLOR = {
  "bad-vio": "vio", "xfeat-vio": "vio",
  okvis2lc0: "base", okvis2lc1: "base", orb3lc0: "xvpr", orb3lc1: "xvpr",
  openvinslc0: "pub",
};
const GROUPS = [
  ["overview", "Overview"], ["systems", "vs. Systems"], ["trajectories", "Trajectories"],
  ["reloc", "Reloc"],
  ["euroc", "EuRoC"], ["rooms", "TUM-VI rooms"], ["long", "TUM-VI long"],
  ["msd", "MSD (headset)"], ["table", "Full table"], ["method", "Method"],
];
const GROUP_TITLE = Object.fromEntries(GROUPS);

const S = {
  runs: [], baselines: [], published: [], ledger: [], meta: {},
  tab: "overview", armOn: Object.fromEntries(ARMS.map(a => [a, true])),
  useRte: false,
};

const $ = s => document.querySelector(s);
const el = (t, a = {}, html) => {
  const e = document.createElement(t);
  for (const [k, v] of Object.entries(a)) e.setAttribute(k, v);
  if (html != null) e.innerHTML = html;
  return e;
};
const median = a => {
  const v = a.filter(x => x != null).sort((x, y) => x - y);
  if (!v.length) return null;
  return v.length % 2 ? v[(v.length - 1) / 2] : (v[v.length / 2 - 1] + v[v.length / 2]) / 2;
};
const fmt = v => v == null ? "—" : v >= 99.5 ? v.toFixed(0) : v >= 9.95 ? v.toFixed(1) : v.toFixed(2);
const col = a => getComputedStyle(document.documentElement).getPropertyValue(`--c-${a}`).trim() || "#888";
const shortName = s => s.replace("dataset-", "").replace("_512_16", "")
    .replace(/_easy|_medium|_difficult/, "");
const group_of = s => /^(MH_|V1_|V2_)/.test(s) ? "euroc"
    : /^(MOO|MIO|MIP|MGO)/.test(s) ? "msd" : /room/.test(s) ? "rooms" : "long";

async function loadJSON(p, fb) {
  try { const r = await fetch(p, { cache: "no-cache" }); if (!r.ok) throw 0; return await r.json(); }
  catch { return fb; }
}

/* ---------- aggregation ---------- */
function medians(runs, metric) {
  const acc = {};
  for (const r of runs) {
    const v = metric === "rte" ? r.rte_cm : r.ate_cm;
    (((acc[r.seq] ??= {})[r.arm] ??= { vio: [], map: [] })[r.track] ||= []).push(v);
  }
  const out = {};
  for (const [seq, arms] of Object.entries(acc)) {
    out[seq] = {};
    for (const [arm, t] of Object.entries(arms))
      out[seq][arm] = { vio: median(t.vio), map: median(t.map),
                        n: Math.max(t.vio.length, t.map.length) };
  }
  return out;
}
const seqsIn = g => [...new Set(S.runs.filter(r => r.group === g).map(r => r.seq))].sort();
function armsFor(seq) { return ARMS.filter(a => S.runs.some(r => r.seq === seq && r.arm === a)); }
function aggBaselines() {
  const acc = {};
  for (const r of S.baselines) (acc[`${r.seq}|${r.arm}|${r.track}`] ??= []).push(r);
  return Object.entries(acc).map(([k, rr]) => {
    const [seq, sys, lc] = k.split("|");
    return { seq, sys, lc: lc.replace("lc", ""),
             ate: median(rr.map(r => r.ate_cm)), rte: median(rr.map(r => r.rte_cm)) };
  });
}

/* ---------- bar chart (click-to-filter legend) ---------- */
function barChart({ title, cats, series, refs = [], note = "", onBar,
                    unit = "cm", higherBetter = false }) {
  series.forEach(s => { if (s.on === undefined) s.on = !s.defaultOff; });
  const refGroups = [...new Set(refs.map(r => r.cls || "pub"))];
  const refOn = Object.fromEntries(refGroups.map(g => [g, true]));
  const card = el("div", { class: "card" });
  card.append(el("h3", {}, title));
  const vsHost = el("div", { class: "vs-strip" });   // dynamic ranking of enabled series
  card.append(vsHost);
  const holder = el("div");
  card.append(holder);

  function drawVs() {
    const act = series.filter(s => s.on);
    const rank = act.map(s => {
      const v = s.values.filter(x => x != null);
      return v.length ? { label: s.label, color: s.color, med: median(v),
                          mean: v.reduce((a, b) => a + b, 0) / v.length, n: v.length } : null;
    }).filter(Boolean).sort((a, b) => higherBetter ? b.med - a.med : a.med - b.med);
    vsHost.innerHTML = !rank.length ? "" :
      `<span class="hint">median over sequences (mean on hover) · ${higherBetter ? "higher" : "lower"} is better:</span>` +
      rank.map((r, i) =>
        `<span class="vs-item${i === 0 ? " best" : ""}" title="mean ${fmt(r.mean)} ${unit} over ${r.n} sequences">` +
        `<span class="dot" style="background:${r.color}"></span>` +
        `${r.label} <b>${fmt(r.med)}</b><small>·${r.n} seq</small></span>`
      ).join(`<span class="vs-gt">${higherBetter ? "&gt;" : "&lt;"}</span>`);
  }

  function draw() {
    const act = series.filter(s => s.on);
    const aRefs = refs.filter(r => refOn[r.cls || "pub"]);
    const W = 1180, mt = 26, mb = cats.length > 13 ? 96 : 56, ml = 48, mr = 10;
    const H = 320 + (cats.length > 13 ? 44 : 0);
    const pw = W - ml - mr, ph = H - mt - mb;
    const vals = act.flatMap(s => s.values).filter(v => v != null).concat(aRefs.map(r => r.v));
    const ymax = Math.max(1, ...vals) * 1.14;
    const Y = v => mt + ph - Math.min(v, ymax) / ymax * ph;
    const gw = pw / Math.max(1, cats.length), bw = Math.min(24, gw * 0.82 / Math.max(1, act.length));
    let s = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">`;
    for (let g = 0; g <= 4; g++) {
      const v = ymax * g / 4;
      s += `<line x1="${ml}" x2="${W - mr}" y1="${Y(v).toFixed(1)}" y2="${Y(v).toFixed(1)}" class="grid-l"/>`
         + `<text x="${ml - 6}" y="${(Y(v) + 4).toFixed(1)}" class="tick" text-anchor="end">${v.toFixed(v < 10 ? 1 : 0)}</text>`;
    }
    cats.forEach((c, i) => {
      const x0 = ml + i * gw + (gw - bw * act.length - 2 * (act.length - 1)) / 2;
      act.forEach((sr, j) => {
        const v = sr.values[i]; if (v == null) return;
        const x = x0 + j * (bw + 2), y = Y(v);
        s += `<rect class="bar" data-cat="${i}" x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${bw.toFixed(1)}" height="${(mt + ph - y).toFixed(1)}" rx="3" fill="${sr.color}"><title>${c} — ${sr.label}: ${fmt(v)} ${unit}${v > ymax ? " (clipped)" : ""}</title></rect>`;
        if (act.length * cats.length <= 60)
          s += `<text x="${(x + bw / 2).toFixed(1)}" y="${(y - 3).toFixed(1)}" class="vlab" text-anchor="middle">${fmt(v)}</text>`;
      });
      const cx = ml + i * gw + gw / 2;
      const rot = cats.length > 13 ? ` transform="rotate(40 ${cx} ${H - mb + 14})"` : "";
      s += `<text class="cat" data-cat="${i}" x="${cx}" y="${H - mb + 14}" text-anchor="${cats.length > 13 ? "start" : "middle"}"${rot}>${c}</text>`;
    });
    for (const r of aRefs) {
      const x0 = ml + r.i * gw + gw * 0.06, x1 = ml + r.i * gw + gw * 0.94;
      const c = r.cls === "base" ? "var(--c-base)" : "var(--c-pub)";
      s += `<line x1="${x0.toFixed(1)}" x2="${x1.toFixed(1)}" y1="${Y(r.v).toFixed(1)}" y2="${Y(r.v).toFixed(1)}" class="refline" stroke="${c}"><title>${r.label}: ${fmt(r.v)} cm</title></line>`;
    }
    // empty state: keep the axes frame, just note there's nothing to plot
    if (!vals.length)
      s += `<text x="${(ml + pw / 2).toFixed(0)}" y="${(mt + ph / 2).toFixed(0)}" class="tick" text-anchor="middle" font-size="13">no data for the selected series</text>`;
    s += "</svg>";
    holder.innerHTML = s;
    drawVs();
    if (onBar) holder.querySelectorAll(".bar,.cat").forEach(b => b.onclick = () => onBar(+b.dataset.cat));
  }
  const leg = el("div", { class: "legend" });
  for (const sr of series) {
    const c = el("span", { class: "chip" + (sr.on ? "" : " off"), title: "toggle series" }, `<span class="dot" style="background:${sr.color}"></span>${sr.label}`);
    c.onclick = () => { sr.on = !sr.on; c.classList.toggle("off"); draw(); };
    leg.append(c);
  }
  for (const g of refGroups) {
    const lbl = g === "base" ? "baselines (ours-run)" : "published refs";
    const c = el("span", { class: "chip", title: "toggle" }, `<span class="dot" style="background:${g === "base" ? "var(--c-base)" : "var(--c-pub)"}"></span>${lbl}`);
    c.onclick = () => { refOn[g] = !refOn[g]; c.classList.toggle("off"); draw(); };
    leg.append(c);
  }
  card.append(leg);
  if (note) card.append(el("div", { class: "note" }, note));
  draw();
  return card;
}

/* ---------- 3D orbit + error timeline trajectory panel ---------- */
/* Orbit renderer with a PERSISTENT camera (cam survives data swaps). Grid
 * bounds come from `g` (stable per sequence, so pivot/grid don't jump when
 * arms toggle). .set(trajs) redraws with new data, camera untouched. */
function orbitView(cv, cam, g) {   // g bounds must already exclude null gap-markers
  const ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  cam.target = g.ctr.slice();                 // retarget to this scene's center
  if (!cam.init) { cam.theta = 0.7; cam.phi = 1.0; cam.zoom = 1; cam.init = true; }
  let trajs = [], B;
  const errCol = e => { const t = Math.max(0, Math.min(e / 50, 1));
    return `hsl(${(1 - t) * 214},78%,${52 + t * 6}%)`; };
  function basis() {   // spherical orbit camera, world +Z up (three.js OrbitControls)
    const ct = Math.cos(cam.theta), st = Math.sin(cam.theta),
          cf = Math.cos(cam.phi), sf = Math.sin(cam.phi);
    B = { right: [-st, ct, 0], up: [-cf * ct, -cf * st, sf],
          sc: (Math.min(W, H) * 0.4 * cam.zoom) / g.span };
  }
  function proj(p) {
    const dx = (p[0] || 0) - cam.target[0], dy = (p[1] || 0) - cam.target[1], dz = (p[2] || 0) - cam.target[2];
    return [W / 2 + (dx * B.right[0] + dy * B.right[1] + dz * B.right[2]) * B.sc,
            H / 2 - (dx * B.up[0] + dy * B.up[1] + dz * B.up[2]) * B.sc];
  }
  function seg(a, b, style, w) {
    const [x0, y0] = proj(a), [x1, y1] = proj(b);
    ctx.strokeStyle = style; ctx.lineWidth = w;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
  }
  function polyline(pts, style, w) {   // null entries = pen-up (gap breaks)
    ctx.strokeStyle = style; ctx.lineWidth = w; ctx.beginPath();
    let pen = false;
    for (const p of pts) {
      if (!p) { pen = false; continue; }
      const [x, y] = proj(p);
      pen ? ctx.lineTo(x, y) : ctx.moveTo(x, y); pen = true;
    }
    ctx.stroke();
  }
  function drawGrid() {
    const gc = col("grid");
    ctx.globalAlpha = 0.75;
    for (let x = g.gx0; x <= g.gx1 + 1e-6; x += g.step) seg([x, g.gy0, 0], [x, g.gy1, 0], gc, Math.abs(x) < 1e-6 ? 1.4 : 0.7);
    for (let y = g.gy0; y <= g.gy1 + 1e-6; y += g.step) seg([g.gx0, y, 0], [g.gx1, y, 0], gc, Math.abs(y) < 1e-6 ? 1.4 : 0.7);
    ctx.globalAlpha = 1;
    const ax = g.step * 1.15;
    seg([0, 0, 0], [ax, 0, 0], "#d83b3b", 2.2); seg([0, 0, 0], [0, ax, 0], "#1baf7a", 2.2);
    seg([0, 0, 0], [0, 0, ax], "#2a78d6", 2.2);
    const [ox, oy] = proj([0, 0, 0]);
    ctx.fillStyle = col("ink2"); ctx.font = "11px system-ui";
    ctx.fillText("0,0,0", ox + 6, oy - 4);
    ctx.fillText(`grid ${g.step >= 1 ? g.step + " m" : (g.step * 100).toFixed(0) + " cm"}`, 10, 16);
  }
  function draw() {
    basis();
    ctx.clearRect(0, 0, W, H); ctx.lineJoin = "round";
    drawGrid();
    const noGt = col("gt");
    for (const tr of trajs) {
      if (tr.err) {   // single-arm: error-color where GT exists, neutral where not
        ctx.lineWidth = 1.9;
        for (let i = 1; i < tr.pts.length; i++) {
          const e = tr.err[i];
          const [x0, y0] = proj(tr.pts[i - 1]), [x1, y1] = proj(tr.pts[i]);
          ctx.strokeStyle = (e == null) ? noGt : errCol(e);
          ctx.globalAlpha = (e == null) ? 0.5 : 1;
          ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
        }
        ctx.globalAlpha = 1;
      } else polyline(tr.pts, tr.color, tr.w || 1.8);
    }
  }
  let drag = null;
  cv.onpointerdown = e => { drag = { x: e.clientX, y: e.clientY, pan: e.shiftKey || e.button === 2 }; cv.setPointerCapture(e.pointerId); };
  cv.oncontextmenu = e => e.preventDefault();
  cv.onpointermove = e => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (drag.pan) {
      const r = cv.getBoundingClientRect(), s = W / r.width;
      const kx = -dx * s / B.sc, ky = dy * s / B.sc;
      for (let k = 0; k < 3; k++) cam.target[k] += kx * B.right[k] + ky * B.up[k];
    } else {
      cam.theta -= dx * 0.01;
      cam.phi = Math.max(0.05, Math.min(Math.PI - 0.05, cam.phi - dy * 0.01));
    }
    drag.x = e.clientX; drag.y = e.clientY; draw();
  };
  cv.onpointerup = () => drag = null;
  cv.onwheel = e => { e.preventDefault(); cam.zoom *= e.deltaY > 0 ? 1 / 1.15 : 1.15; draw(); };
  return { set(t) { trajs = t; draw(); } };
}

/* multi-series error-over-time; drag-zoom shared across all series.
 * opts.onRange(x0, x1|null) fires when the zoom window changes (null = reset);
 * opts.range = [x0, x1] restores a prior window (survives arm toggles). */
function timelineMulti(cv, series, opts = {}) {   // series = [{t, err, color, label}]
  const ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height, ml = 44, mb = 26, mt = 10;
  const tmax = Math.max(1, ...series.map(s => s.t[s.t.length - 1] || 0));
  let x0 = opts.range ? opts.range[0] : 0, x1 = opts.range ? opts.range[1] : tmax, sel = null;
  // median dt per series → break the line when a gap exceeds ~8× it (no-GT holes)
  series.forEach(s => {
    const dts = []; for (let i = 1; i < s.t.length; i++) if (s.err[i] != null && s.err[i - 1] != null) dts.push(s.t[i] - s.t[i - 1]);
    dts.sort((a, b) => a - b); s._gap = (dts[Math.floor(dts.length / 2)] || 1) * 8 + 0.5;
  });
  function draw() {
    ctx.clearRect(0, 0, W, H);
    const ymax = Math.max(5, ...series.flatMap(s => s.t.map((t, i) => t >= x0 && t <= x1 && s.err[i] != null ? s.err[i] : 0))) * 1.12;
    const X = t => ml + (t - x0) / (x1 - x0 || 1) * (W - ml - 10);
    const Y = v => mt + (1 - v / ymax) * (H - mt - mb);
    ctx.strokeStyle = col("grid"); ctx.fillStyle = col("ink2"); ctx.font = "10px system-ui"; ctx.lineWidth = 1;
    for (let gg = 0; gg <= 3; gg++) { const v = ymax * gg / 3;
      ctx.beginPath(); ctx.moveTo(ml, Y(v)); ctx.lineTo(W - 10, Y(v)); ctx.stroke(); ctx.fillText(v.toFixed(0), 8, Y(v) + 3); }
    for (const s of series) {
      ctx.strokeStyle = s.color; ctx.lineWidth = 1.5; ctx.beginPath();
      let pen = false, pt = null;
      s.t.forEach((t, i) => {
        const e = s.err[i];
        if (t < x0 || t > x1 || e == null) { pen = false; return; }
        if (pen && (t - pt) > s._gap) pen = false;   // break across a data gap
        const x = X(t), y = Y(e); pen ? ctx.lineTo(x, y) : ctx.moveTo(x, y); pen = true; pt = t;
      });
      ctx.stroke();
    }
    ctx.fillStyle = col("ink2"); ctx.fillText("error [cm] vs time [s] — drag to zoom, double-click resets", ml, H - 8);
    if (sel) { ctx.fillStyle = "rgba(90,120,214,.2)"; ctx.fillRect(Math.min(sel.a, sel.b), mt, Math.abs(sel.b - sel.a), H - mt - mb); }
  }
  draw();
  const tAt = px => x0 + (px - ml) / (W - ml - 10) * (x1 - x0);
  const px = e => { const r = cv.getBoundingClientRect(); return (e.clientX - r.left) * W / r.width; };
  cv.onpointerdown = e => { sel = { a: px(e), b: px(e) }; cv.setPointerCapture(e.pointerId); };
  cv.onpointermove = e => { if (sel) { sel.b = px(e); draw(); } };
  cv.onpointerup = () => {
    if (sel && Math.abs(sel.b - sel.a) > 10) {
      x0 = tAt(Math.min(sel.a, sel.b)); x1 = tAt(Math.max(sel.a, sel.b));
      opts.onRange?.([x0, x1]);
    }
    sel = null; draw();
  };
  cv.ondblclick = () => { x0 = 0; x1 = tmax; draw(); opts.onRange?.(null); };
}

/* sequence selector + MULTI-arm chips + persistent-camera 3D orbit + timeline.
 * Toggling arms never resets the camera (orbit.set only swaps data). */
function trajPanel({ group, seq, compact } = {}) {
  const card = el("div", { class: "card" });
  const seqs = group ? seqsIn(group) : [...new Set(S.runs.map(r => r.seq))].sort();
  seq = seq && seqs.includes(seq) ? seq : seqs[0];
  card.append(el("h3", {}, "Trajectory vs ground truth"));
  const controls = el("div", { class: "traj-controls" });
  const seqSel = el("select");
  seqs.forEach(sq => { const o = el("option", { value: sq }, shortName(sq)); if (sq === seq) o.selected = true; seqSel.append(o); });
  controls.append(el("label", {}, "sequence "), seqSel);
  const armBox = el("span", { class: "traj-arms" });
  controls.append(el("span", { class: "flabel", style: "margin-left:8px" }, "overlay arms"), armBox);
  card.append(controls);
  const vb = el("div", { class: "viewbox" });
  const cv = el("canvas", { width: 1000, height: compact ? 460 : 560 });
  vb.append(cv); card.append(vb);
  const legendHost = el("div", { class: "legend-line" });
  card.append(legendHost);
  const tlWrap = el("div");
  card.append(tlWrap);

  const cam = {};                         // PERSISTENT across arm toggles
  let orbit = null, gInfo = null, gtData = null, token = 0;
  let availArms = [];                     // arms that actually HAVE a trajectory
  let timeRange = null;                   // timeline brush [t0,t1] — arms only
  const sel = new Set();                  // selected arms

  // clip an arm's trajectory to the brushed time window (GT is never clipped)
  function clipArm(d) {
    if (!timeRange || !d.t) return d;
    const [a, b] = timeRange;
    const idx = d.t.map((t, i) => t >= a && t <= b ? i : -1).filter(i => i >= 0);
    return { est: idx.map(i => d.est[i]), err: d.err ? idx.map(i => d.err[i]) : null, t: idx.map(i => d.t[i]) };
  }

  function gridInfo(gt) {
    // gt may contain null gap-markers (no-mocap holes) — bounds skip them
    const withOrigin = gt.filter(Boolean).concat([[0, 0, 0]]);
    const lo = [0, 1, 2].map(k => Math.min(...withOrigin.map(p => p[k] || 0)));
    const hi = [0, 1, 2].map(k => Math.max(...withOrigin.map(p => p[k] || 0)));
    const ctr = [0, 1, 2].map(k => (lo[k] + hi[k]) / 2);
    const span = Math.max(1e-3, Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]));
    const raw = span / 6, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = (raw / mag >= 5 ? 5 : raw / mag >= 2 ? 2 : 1) * mag;
    return { lo, hi, ctr, span, step,
      gx0: Math.floor(lo[0] / step) * step - step, gx1: Math.ceil(hi[0] / step) * step + step,
      gy0: Math.floor(lo[1] / step) * step - step, gy1: Math.ceil(hi[1] / step) * step + step };
  }
  const jsonCache = {};
  async function getArm(a) {
    if (!(a in jsonCache)) jsonCache[a] = await loadJSON(`data/traj/${seq}_${a}.json`, null);
    return jsonCache[a];
  }
  async function updateOrbit() {
    if (!orbit) return;
    const active = [...sel];
    const loaded = await Promise.all(active.map(getArm));
    const trajs = [{ pts: gtData, color: col("gt"), w: 1.6 }];   // GT always full
    const singleErr = active.length === 1 && loaded[0]?.err;
    loaded.forEach((d, i) => {
      if (!d?.est) return;
      const c = clipArm(d);
      trajs.push(singleErr ? { pts: c.est, err: c.err }
                           : { pts: c.est, color: col(TRAJ_COLOR[active[i]] || active[i]) });
    });
    orbit.set(trajs);
    legendHost.innerHTML = `<span><i style="background:var(--c-gt)"></i>ground truth</span>` +
      (singleErr ? `<span><i style="background:linear-gradient(90deg,#2a78d6,#d83b3b)"></i>error 0→50 cm</span>`
                 : active.map(a => `<span><i style="background:${col(TRAJ_COLOR[a] || a)}"></i>${ARM_LABEL[a] || a}</span>`).join("")) +
      (timeRange ? `<span class="hint">showing ${timeRange[0].toFixed(0)}–${timeRange[1].toFixed(0)} s (arms only) — double-click the timeline to reset</span>`
                 : `<span class="hint">drag = orbit · wheel = zoom · shift/right-drag = pan · grid on z=0 through origin</span>`);
  }
  async function redraw() {
    if (!orbit) return;
    await updateOrbit();
    const active = [...sel];
    const loaded = await Promise.all(active.map(getArm));
    const singleErr = active.length === 1 && loaded[0]?.err;
    tlWrap.innerHTML = "";
    const ts = active.map((a, i) => loaded[i]?.err && loaded[i]?.t ? { t: loaded[i].t, err: loaded[i].err, color: singleErr ? col("base") : col(TRAJ_COLOR[a] || a), label: ARM_LABEL[a] || a } : null).filter(Boolean);
    if (ts.length) {
      const tcv = el("canvas", { width: 1000, height: 180 });
      const w = el("div", { class: "viewbox", style: "margin-top:10px" });
      w.append(tcv); tlWrap.append(w);
      timelineMulti(tcv, ts, { range: timeRange,
        onRange: r => { timeRange = r; updateOrbit(); } });
    }
  }
  function buildArmChips() {
    armBox.innerHTML = "";
    if (!availArms.length) return;   // no arm has data — panel shows GT/grid only
    if (!sel.size) { const first = availArms.find(a => S.armOn[a]) || availArms[0]; if (first) sel.add(first); }
    [...sel].forEach(a => { if (!availArms.includes(a)) sel.delete(a); });
    if (!sel.size) sel.add(availArms[0]);
    for (const a of availArms) {
      const c = el("span", { class: "chip" + (sel.has(a) ? "" : " off") }, `<span class="dot" style="background:${col(TRAJ_COLOR[a] || a)}"></span>${ARM_LABEL[a] || a}`);
      c.onclick = () => { if (sel.has(a)) { if (sel.size > 1) sel.delete(a); } else sel.add(a); buildArmChips(); redraw(); };
      armBox.append(c);
    }
  }
  async function loadSeq() {
    const my = ++token;
    for (const k in jsonCache) delete jsonCache[k];
    // probe EVERY plot key (arms, VIO tracks, baselines); only keys with
    // real data become chips
    const loaded = await Promise.all(TRAJ_KEYS.map(async a => [a, await getArm(a)]));
    if (my !== token) return;
    availArms = loaded.filter(([, d]) => d?.est?.length).map(([a]) => a);
    const withGt = loaded.find(([, d]) => d?.gt?.length);
    if (!withGt) { cv.getContext("2d").clearRect(0, 0, cv.width, cv.height);
      legendHost.innerHTML = `<span class="hint">no trajectory export for this sequence yet.</span>`;
      armBox.innerHTML = ""; tlWrap.innerHTML = ""; return; }
    gtData = withGt[1].gt; gInfo = gridInfo(gtData); orbit = orbitView(cv, cam, gInfo);
    buildArmChips(); redraw();
  }
  // keep the arm selection across sequence changes (buildArmChips prunes any
  // arm the new sequence lacks); only the time brush resets — new time axis
  seqSel.onchange = () => { seq = seqSel.value; timeRange = null; loadSeq(); };
  loadSeq();
  return card;
}

/* ---------- views ---------- */
const BASE_COLORS = { "okvis2_lc0": "#d83b3b", "okvis2_lc1": "#a01f1f",
  "orb3_lc0": "#e0930a", "orb3_lc1": "#a86a00", "openvins_lc0": "#8a897f" };
const BASE_LABEL = { "okvis2_lc0": "OKVIS2", "okvis2_lc1": "OKVIS2+LC",
  "orb3_lc0": "ORB-SLAM3", "orb3_lc1": "ORB-SLAM3+LC", "openvins_lc0": "OpenVINS" };

function datasetBar(group, view) {
  const seqs = seqsIn(group);
  const med = medians(S.runs.filter(r => r.group === group), S.useRte ? "rte" : "ate");
  const armsOn = ARMS.filter(a => S.armOn[a]);
  const series = [
    { label: "VIO only", color: col("vio"), values: seqs.map(s => med[s]?.bad?.vio) },
    ...armsOn.map(a => ({ label: `+map ${ARM_LABEL[a]}`, color: col(a),
      values: seqs.map(s => med[s]?.[a]?.map) })),
  ];
  // baseline systems as REAL bar series (same machine, same causal protocol)
  const base = aggBaselines().filter(b => group_of(b.seq) === group);
  const sysKeys = [...new Set(base.map(b => `${b.sys}_lc${b.lc}`))].sort();
  for (const k of sysKeys) series.push({
    label: BASE_LABEL[k] || k.replace("_", " "),
    color: BASE_COLORS[k] || "#888",
    defaultOff: k.startsWith("orb3"),   // causal snaps make ORB3 dominate the scale
    values: seqs.map(s => { const b = base.find(x => x.seq === s && `${x.sys}_lc${x.lc}` === k); return b ? (S.useRte ? b.rte : b.ate) : null; }),
  });
  let curSeq = seqs[0];
  const panelHost = el("div");
  const chart = barChart({
    title: `${GROUP_TITLE[group]} — causal ${S.useRte ? "RTE" : "ATE"} medians [cm]`,
    cats: seqs.map(shortName), series,
    note: "All bars are systems we ran ourselves: same machine, same causal protocol. Click a legend chip to toggle a series, or a bar to load its trajectory below.",
    onBar: i => { curSeq = seqs[i]; panelHost.innerHTML = ""; panelHost.append(trajPanel({ group, seq: curSeq, compact: true })); panelHost.scrollIntoView({ behavior: "smooth", block: "nearest" }); },
  });
  view.append(chart);
  panelHost.append(trajPanel({ group, seq: curSeq, compact: true }));
  view.append(panelHost);
}

function overviewView() {
  const view = $("#view");
  const med = medians(S.runs, "ate");
  const gmed = g => { const v = seqsIn(g).map(s => med[s]?.vpr?.map ?? med[s]?.bad?.map).filter(x => x != null); return v.length ? median(v) : null; };
  const tiles = el("div", { class: "tiles" });
  const defs = [
    [new Set(S.runs.map(r => r.seq)).size, "sequences"],
    [(S.runs.length / 2).toFixed(0), "scored runs (ours)"],
    [S.baselines.length ? (S.baselines.length) : "—", "baseline runs"],
    [fmt(gmed("euroc")), "EuRoC med ATE +map"],
    [fmt(gmed("msd")), "MSD med ATE +map"],
    [new Set(S.runs.map(r => r.arm)).size, "arms"],
  ];
  for (const [v, l] of defs) tiles.append(el("div", { class: "tile" }, `<b>${v}</b><span>${l}</span>`));
  view.append(tiles);
  for (const g of ["euroc", "rooms", "long", "msd"]) datasetBar(g, view);
}

function systemsView() {
  const view = $("#view");
  const base = aggBaselines();
  if (!base.length) {
    view.append(el("div", { class: "card" }, "<h3>Baseline runs scoring…</h3><p class='note'>OKVIS2 (LC on/off), ORB-SLAM3 (LC on/off) and OpenVINS finished on the compute container; their causal trajectories are being scored in. Auto-refreshes.</p>"));
    return;
  }
  const seqs = [...new Set(base.map(b => b.seq))].sort();
  const med = medians(S.runs.filter(r => seqs.includes(r.seq)), S.useRte ? "rte" : "ate");
  const sysArms = [...new Set(base.map(b => `${b.sys}_lc${b.lc}`))].sort();
  const bestArm = s => { const v = ARMS.map(a => med[s]?.[a]?.map).filter(x => x != null); return v.length ? Math.min(...v) : null; };
  const series = [
    { label: "ours: VIO", color: col("vio"), values: seqs.map(s => med[s]?.bad?.vio) },
    { label: "ours: +map (best arm)", color: col("vpr"), values: seqs.map(bestArm) },
    ...sysArms.map(sa => ({ label: BASE_LABEL[sa] || sa.replace("_", " "), color: BASE_COLORS[sa] || "#888",
      defaultOff: sa.startsWith("orb3"),
      values: seqs.map(s => { const b = base.find(x => x.seq === s && `${x.sys}_lc${x.lc}` === sa); return b ? (S.useRte ? b.rte : b.ate) : null; }) })),
  ];
  view.append(barChart({
    title: `Ours vs systems we ran — causal ${S.useRte ? "RTE" : "ATE"} medians [cm]`,
    cats: seqs.map(shortName), series,
    note: "Every row: same machine, same causal protocol (pose at first estimate). Whole-run causal ATE penalizes loop-closure re-anchoring for ALL systems — see Method for the tail-window caveat.",
    onBar: i => { document.getElementById("sys-traj").innerHTML = ""; document.getElementById("sys-traj").append(trajPanel({ seq: seqs[i], compact: true })); },
  }));
  view.append(el("div", { id: "sys-traj" }));
  view.querySelector("#sys-traj").append(trajPanel({ seq: seqs[0], compact: true }));
}

function trajectoriesView() {
  $("#view").append(trajPanel({}));
}

/* ---------- Reloc tab: cold-probe relocalization results ---------- */
async function relocView() {
  const view = $("#view");
  const data = await loadJSON("data/reloc.json", { entries: [] });
  if (!data.entries.length) {
    view.append(el("div", { class: "card" },
      "<h3>Relocalization</h3><p class='note'>No reloc runs exported yet. " +
      "Produce with <code>xr_replay --reloc N</code>, then export with " +
      "<code>--reloc &lt;logdir&gt;</code>.</p>"));
    return;
  }
  const entries = data.entries;
  const seqs = [...new Set(entries.map(e => e.seq))].sort();
  const arms = [...new Set(entries.map(e => e.arm))].sort();
  const M = {};
  entries.forEach(e => M[`${e.seq}|${e.arm}`] = e);

  /* summary bar charts — same barChart component as every other tab,
   * arms as toggleable series, sequences as categories */
  const mk = (title, get, unit, higherBetter, note) => barChart({
    title, cats: seqs.map(shortName), unit, higherBetter, note,
    series: arms.map(a => ({ label: ARM_LABEL[a] || a, color: armColor(a),
      values: seqs.map(s => { const e = M[`${s}|${a}`]; return e ? get(e) : null; }) })),
  });
  view.append(mk("Relocalization recall [%]", e => e.recall * 100, "%", true,
    "Cold probes (seeded-random frames, no VIO context) against the run's own finished map. Click a legend chip to toggle an arm."));
  view.append(mk("Recall @25 cm [%]", e => e.r25 * 100, "%", true, ""));
  view.append(mk("Median landing error [cm] (verified probes)",
    e => e.med >= 0 ? e.med * 100 : null, "cm", false, ""));

  /* spatial panel — trajPanel-style comparative plotting: sequence
   * selector + multi-arm chips + persistent-camera orbit. Arms overlay in
   * their own colors; single-arm selection switches to error coloring. */
  const pseqs = seqs.filter(s => arms.some(a => M[`${s}|${a}`]?.probes?.some(p => p.exp)));
  if (!pseqs.length) {
    view.append(el("div", { class: "note" },
      "spatial plot needs probe positions (runs made after the exp/got extension)"));
  } else {
    const pc = el("div", { class: "card" });
    pc.append(el("h3", {}, "Where relocalization landed"));
    const ctr = el("div", { class: "traj-controls" });
    const seqSel = el("select");
    pseqs.forEach(sq => seqSel.append(el("option", { value: sq }, shortName(sq))));
    ctr.append(el("label", {}, "sequence "), seqSel);
    const armBox = el("span", { class: "traj-arms" });
    ctr.append(el("span", { class: "flabel", style: "margin-left:8px" }, "overlay arms"), armBox);
    pc.append(ctr);
    const vb = el("div", { class: "viewbox" });
    const cv = el("canvas", { width: 1000, height: 520 });
    vb.append(cv); pc.append(vb);
    const legendHost = el("div", { class: "legend-line" });
    pc.append(legendHost);
    view.append(pc);

    const cam = {};                       // persistent across arm toggles
    let orbit = null, curSeq = pseqs[0];
    const sel = new Set();
    const dim = c => c.startsWith("hsl(") ? c.replace(")", ",0.45)").replace("hsl", "hsla") : c;
    const errCol = v => { const t2 = Math.min(Math.max(v, 0) / 1.0, 1); return `hsl(${(1 - t2) * 214},80%,55%)`; };

    const armsWithProbes = sq => arms.filter(a => M[`${sq}|${a}`]?.probes?.some(p => p.exp));
    function gridFor(sq) {
      const e = M[`${sq}|${armsWithProbes(sq)[0]}`];
      const base = (e.traj || e.probes.filter(p => p.exp).map(p => p.exp)).filter(Boolean);
      const withO = base.concat([[0, 0, 0]]);
      const lo = [0, 1, 2].map(k => Math.min(...withO.map(p => p[k] || 0)));
      const hi = [0, 1, 2].map(k => Math.max(...withO.map(p => p[k] || 0)));
      const g = { lo, hi, ctr: [0, 1, 2].map(k => (lo[k] + hi[k]) / 2),
        span: Math.max(1e-3, Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])) };
      const raw = g.span / 6, mag = Math.pow(10, Math.floor(Math.log10(raw)));
      g.step = (raw / mag >= 5 ? 5 : raw / mag >= 2 ? 2 : 1) * mag;
      g.gx0 = Math.floor(lo[0] / g.step) * g.step - g.step;
      g.gx1 = Math.ceil(hi[0] / g.step) * g.step + g.step;
      g.gy0 = Math.floor(lo[1] / g.step) * g.step - g.step;
      g.gy1 = Math.ceil(hi[1] / g.step) * g.step + g.step;
      return g;
    }
    function redraw() {
      if (!orbit) return;
      const act = [...sel];
      const single = act.length === 1;
      const trajs = [];
      const s = 0.012 * gridFor(curSeq).span;   // failed-probe X size
      for (const a of act) {
        const e = M[`${curSeq}|${a}`];
        if (!e) continue;
        const c = armColor(a);
        if (e.traj) trajs.push({ pts: e.traj, color: single ? col("gt") : dim(c), w: 1.2 });
        for (const p of e.probes) {
          if (!p.exp) continue;
          if (p.ok && p.got) {
            trajs.push({ pts: [p.exp, p.got], color: single ? errCol(p.err) : c, w: 2.4 });
          } else {
            trajs.push({ pts: [[p.exp[0] - s, p.exp[1] - s, p.exp[2]], [p.exp[0] + s, p.exp[1] + s, p.exp[2]]], color: single ? "var(--c-base)" : dim(c), w: 1.6 });
            trajs.push({ pts: [[p.exp[0] - s, p.exp[1] + s, p.exp[2]], [p.exp[0] + s, p.exp[1] - s, p.exp[2]]], color: single ? "var(--c-base)" : dim(c), w: 1.6 });
          }
        }
      }
      orbit.set(trajs);
      legendHost.innerHTML =
        (single
          ? `<span><i style="background:var(--c-gt)"></i>session trajectory</span>` +
            `<span><i style="background:linear-gradient(90deg,#2a78d6,#d83b3b)"></i>expected→landed (color = error 0→1 m)</span>` +
            `<span><i style="background:var(--c-base)"></i>✕ failed probe (at expected)</span>`
          : act.map(a => `<span><i style="background:${armColor(a)}"></i>${ARM_LABEL[a] || a}</span>`).join("") +
            `<span class="hint">solid = landed vector · faint = trajectory / failed ✕</span>`) +
        `<span class="hint">drag = orbit · wheel = zoom · shift-drag = pan</span>`;
    }
    function buildChips() {
      armBox.innerHTML = "";
      const avail = armsWithProbes(curSeq);
      [...sel].forEach(a => { if (!avail.includes(a)) sel.delete(a); });
      if (!sel.size && avail.length) sel.add(avail[0]);
      for (const a of avail) {
        const c = el("span", { class: "chip" + (sel.has(a) ? "" : " off") },
          `<span class="dot" style="background:${armColor(a)}"></span>${ARM_LABEL[a] || a}`);
        c.onclick = () => { if (sel.has(a)) { if (sel.size > 1) sel.delete(a); } else sel.add(a); buildChips(); redraw(); };
        armBox.append(c);
      }
    }
    function loadSeq() {
      orbit = orbitView(cv, cam, gridFor(curSeq));
      buildChips();
      redraw();
    }
    seqSel.onchange = () => { curSeq = seqSel.value; loadSeq(); };
    loadSeq();
  }

  // full summary table (below the charts)
  const t = el("table");
  t.innerHTML = "<tr><th>sequence</th><th>arm</th><th>probes</th>" +
    "<th>recall</th><th>r@25cm</th><th>r@10cm</th><th>median err [m]</th></tr>" +
    entries.map(e =>
      `<tr><td>${shortName(e.seq)}</td><td>${ARM_LABEL[e.arm] || e.arm}</td>` +
      `<td>${e.n}</td><td>${(e.recall * 100).toFixed(0)}%</td>` +
      `<td>${(e.r25 * 100).toFixed(0)}%</td><td>${(e.r10 * 100).toFixed(0)}%</td>` +
      `<td>${e.med >= 0 ? e.med.toFixed(2) : "—"}</td></tr>`).join("");
  const card = el("div", { class: "card" });
  card.append(el("h3", {}, "All runs"), t);
  card.append(el("div", { class: "note" },
    "Probes are seeded-random frames re-fed against the run's own finished map " +
    "with no VIO context (gravity from the IMU, as deployment would have). " +
    "Error = landed pose vs the run's map-track pose at that frame. OKVIS2 " +
    "public code lacks loadMap — cross-system comparison uses the blackout " +
    "protocol (pending); ORB-SLAM3 atlas-reload pending; OpenVINS N/A."));
  view.append(card);
}

function tableView() {
  const view = $("#view");
  const med = medians(S.runs, "ate"), medR = medians(S.runs, "rte");
  const armsOn = ARMS.filter(a => S.armOn[a]);
  const t = el("table");
  t.innerHTML = `<tr><th>sequence</th><th>VIO ATE</th><th>VIO RTE</th>` +
    armsOn.map(a => `<th>+${ARM_LABEL[a]}</th>`).join("") + `</tr>`;
  for (const g of ["euroc", "rooms", "long", "msd"]) {
    t.innerHTML += `<tr class="ghead"><td colspan="${3 + armsOn.length}">${GROUP_TITLE[g]}</td></tr>`;
    for (const s of seqsIn(g)) {
      const vals = armsOn.map(a => med[s]?.[a]?.map);
      const best = Math.min(...vals.filter(v => v != null).concat(med[s]?.bad?.vio ?? Infinity));
      t.innerHTML += `<tr><td>${shortName(s)}</td>` +
        `<td class="${med[s]?.bad?.vio === best ? "best" : ""}">${fmt(med[s]?.bad?.vio)}</td><td>${fmt(medR[s]?.bad?.vio)}</td>` +
        armsOn.map((a, k) => `<td class="${vals[k] === best ? "best" : ""}">${fmt(vals[k])}</td>`).join("") + `</tr>`;
    }
  }
  view.append(el("div", { class: "overflow" }, "")).append(t);
}

function methodView() {
  $("#view").append(el("div", { class: "card" }, `
    <h3>Protocol</h3>
    <p>${S.meta.protocol || "causal, SE(3)-Umeyama ATE, RTE Δ=6 frames"}. One replay emits both tracks: raw VIO pose and map-corrected session pose (the map layer has no feedback into VIO). Arms differ only in descriptor (BAD/TEBLID vs XFeat) and retrieval model (none / EigenPlaces-512 / MegaLoc-8448).</p>
    <h3>Comparing to other systems, honestly</h3>
    <p>OKVIS2, ORB-SLAM3 and OpenVINS were built and run by us on the same container under the same causal protocol (see "vs. Systems"). Whole-run causal ATE penalizes any loop-closure system at the instant a correction re-anchors the live pose (past poses stay in the old frame) — ORB-SLAM3 shows metre-scale steps at closures while its tail-window ATE is centimetres. Divergence therefore gates on ATE only (10 m; 1 m for MSD): an RTE gate would structurally disqualify correction-based systems, since one re-anchor step is a giant frame-pair error. RTE is still computed and reported for every run — correction snaps are plainly visible there.</p>
    <h3>Reproduce</h3>
    <p><code>node bench/site/server.js</code> serves this page · <code>bench/host/export_site_data.py</code> regenerates <code>data/*.json</code> · <code>bench/replay</code> is the harness · <code>bench/PROGRAM.md</code> is the plan.</p>`));
}

/* ---------- shell ---------- */
function render() {
  $("#view").innerHTML = "";
  ({ overview: overviewView, systems: systemsView, trajectories: trajectoriesView,
     reloc: relocView,
     table: tableView, method: methodView }[S.tab] || (() => datasetBar(S.tab, $("#view"))))();
}
function buildShell() {
  const tabs = $("#tabs");
  tabs.innerHTML = "";
  for (const [id, label] of GROUPS) {
    const b = el("button", {}, label);
    if (id === S.tab) b.classList.add("on");
    b.onclick = () => { S.tab = id; tabs.querySelectorAll("button").forEach(x => x.classList.remove("on")); b.classList.add("on"); render(); };
    tabs.append(b);
  }
  const chips = $("#arm-chips"); chips.innerHTML = "";
  for (const a of ARMS) {
    const c = el("span", { class: "chip" + (S.armOn[a] ? "" : " off") }, `<span class="dot" style="background:${col(a)}"></span>${ARM_LABEL[a]}`);
    c.onclick = () => { S.armOn[a] = !S.armOn[a]; c.classList.toggle("off"); render(); };
    chips.append(c);
  }
  const seg = $("#theme-seg");
  const applyTheme = t => {
    if (t === "auto") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = t;
    localStorage.setItem("bench-theme", t);
    seg.querySelectorAll("button").forEach(x => x.classList.toggle("on", x.dataset.t === t));
  };
  applyTheme(localStorage.getItem("bench-theme") || "auto");
  seg.querySelectorAll("button").forEach(b => b.onclick = () => { applyTheme(b.dataset.t); render(); });
  $("#metric-rte").onchange = e => { S.useRte = e.target.checked; render(); };
}

let lastGen = null;
async function refresh(first) {
  const [res, base, led, meta] = await Promise.all([
    loadJSON("data/results.json", { runs: [] }), loadJSON("data/baselines.json", { runs: [] }),
    loadJSON("data/ledger.json", []), loadJSON("data/meta.json", {}),
  ]);
  const changed = (res.generated !== lastGen) || (S.baselines.length !== (base.runs || []).length);
  lastGen = res.generated;
  S.runs = res.runs; S.baselines = base.runs || []; S.ledger = led; S.meta = meta;
  $("#meta-line").innerHTML = `<span id="live-dot">●</span> LIVE · ${S.runs.length} run-scores · ${S.baselines.length} baseline runs · data ${res.generated || "?"}`;
  if (first || changed) render();
}
(async () => { await refresh(true); buildShell(); render(); setInterval(() => refresh(false), 45000); })();
