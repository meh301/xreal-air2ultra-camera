/* SLAM Benchmark Explorer — zero-dependency SPA.
 * Data: data/{results,baselines,published,ledger,meta}.json + data/traj/<seq>_<arm>.json */
"use strict";

const ARMS = ["bad", "vpr", "megaloc", "xfeat", "xvpr", "xmegaloc"];
const ARM_LABEL = {
  bad: "BAD/TEBLID", vpr: "BAD + EigenPlaces", megaloc: "BAD + MegaLoc",
  xfeat: "XFeat", xvpr: "XFeat + EigenPlaces", xmegaloc: "XFeat + MegaLoc",
};
const GROUPS = [
  ["overview", "Overview"], ["systems", "vs. Systems"], ["trajectories", "Trajectories"],
  ["euroc", "EuRoC"], ["rooms", "TUM-VI rooms"], ["long", "TUM-VI long"],
  ["msd", "MSD (headset)"], ["table", "Full table"], ["method", "Method"],
];
const GROUP_TITLE = Object.fromEntries(GROUPS);

const S = {
  runs: [], baselines: [], published: [], ledger: [], meta: {},
  tab: "overview", armOn: Object.fromEntries(ARMS.map(a => [a, true])),
  showPub: true, showBase: true, useRte: false,
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
function barChart({ title, cats, series, refs = [], note = "", onBar }) {
  series.forEach(s => { if (s.on === undefined) s.on = true; });
  const refGroups = [...new Set(refs.map(r => r.cls || "pub"))];
  const refOn = Object.fromEntries(refGroups.map(g => [g, true]));
  const card = el("div", { class: "card" });
  card.append(el("h3", {}, title));
  const holder = el("div");
  card.append(holder);

  function draw() {
    const act = series.filter(s => s.on);
    const aRefs = refs.filter(r => refOn[r.cls || "pub"]);
    const W = 1180, mt = 26, mb = cats.length > 13 ? 96 : 56, ml = 48, mr = 10;
    const H = 320 + (cats.length > 13 ? 44 : 0);
    const pw = W - ml - mr, ph = H - mt - mb;
    const vals = act.flatMap(s => s.values).filter(v => v != null).concat(aRefs.map(r => r.v));
    const ymax = Math.max(1, ...vals) * 1.14;
    const Y = v => mt + ph - Math.min(v, ymax) / ymax * ph;
    const gw = pw / cats.length, bw = Math.min(24, gw * 0.82 / Math.max(1, act.length));
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
        s += `<rect class="bar" data-cat="${i}" x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${bw.toFixed(1)}" height="${(mt + ph - y).toFixed(1)}" rx="3" fill="${sr.color}"><title>${c} — ${sr.label}: ${fmt(v)} cm${v > ymax ? " (clipped)" : ""}</title></rect>`;
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
    s += "</svg>";
    holder.innerHTML = s;
    if (onBar) holder.querySelectorAll(".bar,.cat").forEach(b => b.onclick = () => onBar(+b.dataset.cat));
  }
  const leg = el("div", { class: "legend" });
  for (const sr of series) {
    const c = el("span", { class: "chip", title: "toggle series" }, `<span class="dot" style="background:${sr.color}"></span>${sr.label}`);
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
function orbitCanvas(cv, data, armColor) {
  const ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  const all = data.gt.concat(data.est);
  const dim = all[0].length;
  // include the world origin in bounds so the reference grid always shows it
  const lo = [0, 1, 2].map(k => Math.min(0, ...all.map(p => p[k] || 0)));
  const hi = [0, 1, 2].map(k => Math.max(0, ...all.map(p => p[k] || 0)));
  const ctr = [0, 1, 2].map(k => (lo[k] + hi[k]) / 2);
  const span = Math.max(1e-3, Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]));
  const raw = span / 6, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = (raw / mag >= 5 ? 5 : raw / mag >= 2 ? 2 : 1) * mag;
  const gx0 = Math.floor(lo[0] / step) * step - step, gx1 = Math.ceil(hi[0] / step) * step + step;
  const gy0 = Math.floor(lo[1] / step) * step - step, gy1 = Math.ceil(hi[1] / step) * step + step;

  const target = ctr.slice();           // world look-at point; pan moves it
  let yaw = 0.6, pitch = 0.9, zoom = 1;
  const errCol = e => { const t = Math.max(0, Math.min(e / 50, 1));
    return `hsl(${(1 - t) * 214},78%,${52 + t * 6}%)`; };
  let B;
  function basis() {
    // orbit camera about the world up-axis (+Z): forward f = (cp·cy, cp·sy, sp);
    // right = normalize(f × Zup) = (sy, -cy, 0); camUp = right × f has +Z
    // component (cos pitch) so world-up always projects to screen-up.
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    B = { right: [sy, -cy, 0], up: [-cy * sp, -sy * sp, cp],
          sc: (Math.min(W, H) * 0.4 * zoom) / span };
  }
  function proj(p) {
    const dx = (p[0] || 0) - target[0], dy = (p[1] || 0) - target[1], dz = (p[2] || 0) - target[2];
    const X = dx * B.right[0] + dy * B.right[1] + dz * B.right[2];
    const Y = dx * B.up[0] + dy * B.up[1] + dz * B.up[2];
    return [W / 2 + X * B.sc, H / 2 - Y * B.sc];
  }
  function seg(a, b, style, w) {
    const [x0, y0] = proj(a), [x1, y1] = proj(b);
    ctx.strokeStyle = style; ctx.lineWidth = w;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
  }
  function drawGrid() {
    const gc = col("grid");
    ctx.globalAlpha = 0.75;
    for (let x = gx0; x <= gx1 + 1e-6; x += step) seg([x, gy0, 0], [x, gy1, 0], gc, Math.abs(x) < 1e-6 ? 1.4 : 0.7);
    for (let y = gy0; y <= gy1 + 1e-6; y += step) seg([gx0, y, 0], [gx1, y, 0], gc, Math.abs(y) < 1e-6 ? 1.4 : 0.7);
    ctx.globalAlpha = 1;
    const ax = step * 1.15;             // origin axes: X red, Y green, Z blue
    seg([0, 0, 0], [ax, 0, 0], "#d83b3b", 2.2);
    seg([0, 0, 0], [0, ax, 0], "#1baf7a", 2.2);
    seg([0, 0, 0], [0, 0, ax], "#2a78d6", 2.2);
    const [ox, oy] = proj([0, 0, 0]);
    ctx.fillStyle = col("ink2"); ctx.font = "11px system-ui";
    ctx.fillText("0,0,0", ox + 6, oy - 4);
    ctx.fillText(`grid ${step >= 1 ? step + " m" : (step * 100).toFixed(0) + " cm"}`, 10, 16);
  }
  function draw() {
    basis();
    ctx.clearRect(0, 0, W, H);
    ctx.lineJoin = "round";
    drawGrid();
    ctx.strokeStyle = col("gt"); ctx.lineWidth = 1.6;
    ctx.beginPath();
    data.gt.forEach((p, i) => { const [x, y] = proj(p); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();
    if (data.err) {
      ctx.lineWidth = 1.9;
      for (let i = 1; i < data.est.length; i++) {
        const [x0, y0] = proj(data.est[i - 1]), [x1, y1] = proj(data.est[i]);
        ctx.strokeStyle = errCol(data.err[i] ?? 0);
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
      }
    } else {
      ctx.strokeStyle = armColor; ctx.lineWidth = 1.9; ctx.beginPath();
      data.est.forEach((p, i) => { const [x, y] = proj(p); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.stroke();
    }
  }
  draw();
  let drag = null;
  cv.onpointerdown = e => { drag = { x: e.clientX, y: e.clientY, pan: e.shiftKey || e.button === 2 }; cv.setPointerCapture(e.pointerId); };
  cv.oncontextmenu = e => e.preventDefault();
  cv.onpointermove = e => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (drag.pan) {   // pan moves the look-at target -> orbit pivots with it
      const r = cv.getBoundingClientRect(), s = W / r.width;
      const kx = -dx * s / B.sc, ky = dy * s / B.sc;
      for (let k = 0; k < 3; k++) target[k] += kx * B.right[k] + ky * B.up[k];
    } else {   // grab-and-turn: content follows the cursor
      yaw -= dx * 0.01; pitch = Math.max(-1.55, Math.min(1.55, pitch + dy * 0.01));
    }
    drag.x = e.clientX; drag.y = e.clientY; draw();
  };
  cv.onpointerup = () => drag = null;
  cv.onwheel = e => { e.preventDefault(); zoom *= e.deltaY > 0 ? 1 / 1.15 : 1.15; draw(); };
  return { redraw: draw, is3d: dim >= 3 };
}

function timelineCanvas(cv, data) {
  const ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height, ml = 44, mb = 26, mt = 10;
  let x0 = 0, x1 = data.t[data.t.length - 1] || 1, sel = null;
  function draw() {
    ctx.clearRect(0, 0, W, H);
    const pts = data.t.map((t, i) => [t, data.err[i]]).filter(p => p[0] >= x0 && p[0] <= x1);
    const ymax = Math.max(5, ...pts.map(p => p[1])) * 1.12;
    const X = t => ml + (t - x0) / (x1 - x0 || 1) * (W - ml - 10);
    const Y = v => mt + (1 - v / ymax) * (H - mt - mb);
    ctx.strokeStyle = col("grid"); ctx.fillStyle = col("ink2");
    ctx.font = "10px system-ui"; ctx.lineWidth = 1;
    for (let g = 0; g <= 3; g++) { const v = ymax * g / 3;
      ctx.beginPath(); ctx.moveTo(ml, Y(v)); ctx.lineTo(W - 10, Y(v)); ctx.stroke();
      ctx.fillText(v.toFixed(0), 8, Y(v) + 3); }
    ctx.strokeStyle = col("base"); ctx.lineWidth = 1.5; ctx.beginPath();
    pts.forEach((p, i) => { i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1])); });
    ctx.stroke();
    ctx.fillText("error [cm] vs time [s] — drag to zoom, double-click resets", ml, H - 8);
    if (sel) { ctx.fillStyle = "rgba(90,120,214,.2)";
      ctx.fillRect(Math.min(sel.a, sel.b), mt, Math.abs(sel.b - sel.a), H - mt - mb); }
  }
  draw();
  const tAt = px => x0 + (px - ml) / (W - ml - 10) * (x1 - x0);
  const px = e => { const r = cv.getBoundingClientRect(); return (e.clientX - r.left) * W / r.width; };
  cv.onpointerdown = e => { sel = { a: px(e), b: px(e) }; cv.setPointerCapture(e.pointerId); };
  cv.onpointermove = e => { if (sel) { sel.b = px(e); draw(); } };
  cv.onpointerup = () => { if (sel && Math.abs(sel.b - sel.a) > 10) {
      const a = tAt(Math.min(sel.a, sel.b)), b = tAt(Math.max(sel.a, sel.b)); x0 = a; x1 = b; }
    sel = null; draw(); };
  cv.ondblclick = () => { x0 = 0; x1 = data.t[data.t.length - 1] || 1; draw(); };
}

/* reusable: sequence + arm selectors, big 3D orbit, error timeline */
function trajPanel({ group, seq, compact } = {}) {
  const card = el("div", { class: "card" });
  const seqs = group ? seqsIn(group) : [...new Set(S.runs.map(r => r.seq))].sort();
  seq = seq && seqs.includes(seq) ? seq : seqs[0];
  card.append(el("h3", {}, "Trajectory vs ground truth"));
  const controls = el("div", { class: "traj-controls" });
  const seqSel = el("select");
  seqs.forEach(sq => { const o = el("option", { value: sq }, shortName(sq)); if (sq === seq) o.selected = true; seqSel.append(o); });
  const armSel = el("select");
  const armWrap = el("label", {}, "arm ");
  armWrap.append(armSel);
  controls.append(el("label", {}, "sequence "), seqSel, armWrap);
  card.append(controls);
  const vb = el("div", { class: "viewbox" });
  const cv = el("canvas", { width: 1000, height: compact ? 460 : 560 });
  vb.append(cv);
  card.append(vb);
  const tlWrap = el("div");
  card.append(tlWrap);
  card.append(el("div", { class: "legend-line" },
    `<span><i style="background:var(--c-gt)"></i>ground truth</span>` +
    `<span><i style="background:linear-gradient(90deg,#2a78d6,#d83b3b)"></i>estimate, colored by error 0→50 cm</span>` +
    `<span class="hint">drag = orbit (pivots on look-at) · wheel = zoom · shift/right-drag = pan · grid on z=0 through origin</span>`));

  let token = 0;
  function fillArms() {
    const av = armsFor(seq).filter(a => S.armOn[a]);
    armSel.innerHTML = "";
    (av.length ? av : armsFor(seq)).forEach(a => armSel.append(el("option", { value: a }, ARM_LABEL[a])));
  }
  async function load() {
    const my = ++token;
    const arm = armSel.value || "bad";
    const d = await loadJSON(`data/traj/${seq}_${arm}.json`, null);
    if (my !== token) return;
    if (!d || !d.est || !d.est.length) {
      cv.getContext("2d").clearRect(0, 0, cv.width, cv.height);
      tlWrap.innerHTML = `<div class="hint">no trajectory export for this run yet.</div>`;
      return;
    }
    orbitCanvas(cv, d, col(arm));
    tlWrap.innerHTML = "";
    if (d.err && d.t) {
      const tcv = el("canvas", { width: 1000, height: 180 });
      const w = el("div", { class: "viewbox", style: "margin-top:10px" });
      w.append(tcv); tlWrap.append(w);
      timelineCanvas(tcv, d);
    }
  }
  seqSel.onchange = () => { seq = seqSel.value; fillArms(); load(); };
  armSel.onchange = load;
  fillArms(); load();
  return card;
}

/* ---------- views ---------- */
function datasetBar(group, view) {
  const seqs = seqsIn(group);
  const med = medians(S.runs.filter(r => r.group === group), S.useRte ? "rte" : "ate");
  const armsOn = ARMS.filter(a => S.armOn[a]);
  const series = [
    { label: "VIO only", color: col("vio"), values: seqs.map(s => med[s]?.bad?.vio) },
    ...armsOn.map(a => ({ label: `+map ${ARM_LABEL[a]}`, color: col(a),
      values: seqs.map(s => med[s]?.[a]?.map) })),
  ];
  const refs = [];
  if (S.showPub) for (const p of S.published.filter(p => p.group === group)) {
    const i = seqs.indexOf(p.seq); if (i >= 0 && !S.useRte) refs.push({ i, v: p.ate_cm, cls: "pub", label: `${p.system} (${p.regime})` }); }
  if (S.showBase) for (const b of aggBaselines().filter(b => group_of(b.seq) === group)) {
    const i = seqs.indexOf(b.seq); if (i >= 0) refs.push({ i, v: S.useRte ? b.rte : b.ate, cls: "base", label: `${b.sys} lc${b.lc} (ours-run causal)` }); }
  let curSeq = seqs[0];
  const panelHost = el("div");
  const chart = barChart({
    title: `${GROUP_TITLE[group]} — causal ${S.useRte ? "RTE" : "ATE"} medians [cm]`,
    cats: seqs.map(shortName), series, refs,
    note: "Bars = our arms. Gray dashes = published refs (regime on hover); red dashes = baselines we ran on the same machine. Click a bar to load its trajectory below.",
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
  const palette = ["#d83b3b", "#8a897f", "#a06be0", "#e0930a", "#0f8a3c"];
  const bestArm = s => { const v = ARMS.map(a => med[s]?.[a]?.map).filter(x => x != null); return v.length ? Math.min(...v) : null; };
  const series = [
    { label: "ours: VIO", color: col("vio"), values: seqs.map(s => med[s]?.bad?.vio) },
    { label: "ours: +map (best arm)", color: col("vpr"), values: seqs.map(bestArm) },
    ...sysArms.map((sa, i) => ({ label: sa.replace("_", " "), color: palette[i % palette.length],
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
    <p>OKVIS2, ORB-SLAM3 and OpenVINS were built and run by us on the same container under the same causal protocol (see "vs. Systems"). Whole-run causal ATE penalizes any loop-closure system at the instant a correction re-anchors the live pose (past poses stay in the old frame) — ORB-SLAM3 shows metre-scale steps at closures while its tail-window ATE is centimetres. Published rows carry their regime (causal?, IMU?, LC?, compression) in the hover title; several are non-causal with full-batch BA and are not directly comparable to a causal column.</p>
    <h3>Reproduce</h3>
    <p><code>node bench/site/server.js</code> serves this page · <code>bench/host/export_site_data.py</code> regenerates <code>data/*.json</code> · <code>bench/replay</code> is the harness · <code>bench/PROGRAM.md</code> is the plan.</p>`));
}

/* ---------- shell ---------- */
function render() {
  $("#view").innerHTML = "";
  ({ overview: overviewView, systems: systemsView, trajectories: trajectoriesView,
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
  $("#show-pub").onchange = e => { S.showPub = e.target.checked; render(); };
  $("#show-base").onchange = e => { S.showBase = e.target.checked; render(); };
  $("#metric-rte").onchange = e => { S.useRte = e.target.checked; render(); };
}

let lastGen = null;
async function refresh(first) {
  const [res, base, pub, led, meta] = await Promise.all([
    loadJSON("data/results.json", { runs: [] }), loadJSON("data/baselines.json", { runs: [] }),
    loadJSON("data/published.json", []), loadJSON("data/ledger.json", []), loadJSON("data/meta.json", {}),
  ]);
  const changed = res.generated !== lastGen; lastGen = res.generated;
  S.runs = res.runs; S.baselines = base.runs || []; S.published = pub; S.ledger = led; S.meta = meta;
  $("#meta-line").innerHTML = `<span id="live-dot">●</span> LIVE · ${S.runs.length} run-scores · ${S.baselines.length} baseline runs · data ${res.generated || "?"}`;
  if (first || changed) render();
}
(async () => { await refresh(true); buildShell(); render(); setInterval(() => refresh(false), 45000); })();
