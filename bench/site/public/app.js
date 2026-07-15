/* SLAM Benchmark Explorer — zero-dependency SPA.
 * Data: data/results.json (run-level), data/baselines.json, data/published.json,
 * data/ledger.json, data/traj/<seq>_<arm>.json, data/meta.json. */
"use strict";

const ARMS = ["bad", "vpr", "megaloc", "xfeat", "xvpr", "xmegaloc"];
const ARM_LABEL = {
  bad: "BAD/TEBLID", vpr: "BAD + EigenPlaces", megaloc: "BAD + MegaLoc",
  xfeat: "XFeat", xvpr: "XFeat + EigenPlaces", xmegaloc: "XFeat + MegaLoc",
};
const GROUPS = [
  ["overview", "Overview"], ["euroc", "EuRoC"], ["rooms", "TUM-VI rooms"],
  ["long", "TUM-VI long"], ["msd", "MSD (headset)"],
  ["systems", "vs. Systems"], ["table", "Full table"], ["method", "Method"],
];

const S = {
  runs: [], baselines: [], published: [], ledger: [], meta: {},
  tab: "overview", armOn: Object.fromEntries(ARMS.map(a => [a, true])),
  showPub: true, showBase: true, useRte: false,
};

const $ = s => document.querySelector(s);
const el = (t, attrs = {}, html = "") => {
  const e = document.createElement(t);
  Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
  if (html) e.innerHTML = html;
  return e;
};
const median = a => {
  const v = a.filter(x => x != null).sort((x, y) => x - y);
  if (!v.length) return null;
  return v.length % 2 ? v[(v.length - 1) / 2]
                      : (v[v.length / 2 - 1] + v[v.length / 2]) / 2;
};
const fmt = v => v == null ? "—" : v >= 99.5 ? v.toFixed(0) : v >= 9.95 ? v.toFixed(1) : v.toFixed(2);
const cssColor = a => getComputedStyle(document.documentElement)
    .getPropertyValue(`--c-${a}`).trim() || "#888";

async function loadJSON(path, fallback) {
  try { const r = await fetch(path); if (!r.ok) throw 0; return await r.json(); }
  catch { return fallback; }
}

/* ---------- aggregation ---------- */
function medians(runs, metric) {
  // -> map seq -> arm -> {vio, map}
  const acc = {};
  for (const r of runs) {
    const v = metric === "rte" ? r.rte_cm : r.ate_cm;
    ((acc[r.seq] ??= {})[r.arm] ??= { vio: [], map: [] })[r.track]?.push(v);
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
const seqsIn = group => [...new Set(S.runs.filter(r => r.group === group)
    .map(r => r.seq))].sort();
const shortName = s => s.replace("dataset-", "").replace("_512_16", "")
    .replace("_easy", "").replace("_medium", "").replace("_difficult", "");

/* ---------- SVG bar chart with click-to-filter legend ---------- */
function barChart({ title, cats, series, refs = [], note = "", onBar }) {
  series.forEach(s => { if (s.on === undefined) s.on = true; });
  const refGroups = [...new Set(refs.map(r => r.cls || "refline"))];
  const refOn = Object.fromEntries(refGroups.map(g => [g, true]));

  const card = el("div", { class: "chart-card" });
  card.append(el("h3", {}, title));
  const holder = el("div");
  card.append(holder);

  function draw() {
    const act = series.filter(s => s.on);
    const actRefs = refs.filter(r => refOn[r.cls || "refline"]);
    const W = 1080, mb = 58, mt = 26, ml = 46, mr = 8;
    const H = 300 + (cats.length > 14 ? 40 : 0);
    const pw = W - ml - mr, ph = H - mt - mb;
    const vals = act.flatMap(s => s.values).filter(v => v != null)
        .concat(actRefs.map(r => r.v));
    const ymax = Math.max(1, ...vals) * 1.14;
    const Y = v => mt + ph - Math.min(v, ymax) / ymax * ph;
    const gw = pw / cats.length;
    const bw = Math.min(22, gw * 0.8 / Math.max(1, act.length));
    let s = `<svg viewBox="0 0 ${W} ${H}">`;
    for (let g = 0; g <= 4; g++) {
      const v = ymax * g / 4;
      s += `<line x1="${ml}" x2="${W - mr}" y1="${Y(v)}" y2="${Y(v)}" class="grid-l"/>` +
           `<text x="${ml - 5}" y="${Y(v) + 4}" class="tick" text-anchor="end">${v.toFixed(0)}</text>`;
    }
    cats.forEach((c, i) => {
      const x0 = ml + i * gw + (gw - bw * act.length - 2 * (act.length - 1)) / 2;
      act.forEach((sr, j) => {
        const v = sr.values[i];
        if (v == null) return;
        const x = x0 + j * (bw + 2), y = Y(v);
        s += `<rect class="bar" data-cat="${i}" x="${x.toFixed(1)}" y="${y.toFixed(1)}"` +
             ` width="${bw.toFixed(1)}" height="${(mt + ph - y).toFixed(1)}" rx="3"` +
             ` fill="${sr.color}"><title>${c} — ${sr.label}: ${fmt(v)} cm` +
             `${v > ymax ? " (clipped)" : ""}</title></rect>`;
        if (act.length * cats.length < 70)
          s += `<text x="${(x + bw / 2).toFixed(1)}" y="${(y - 3).toFixed(1)}"` +
               ` class="vlab" text-anchor="middle">${fmt(v)}</text>`;
      });
      const rot = cats.length > 14 ? ` transform="rotate(38 ${ml + i * gw + gw / 2} ${H - mb + 14})"` : "";
      s += `<text class="cat" data-cat="${i}" x="${ml + i * gw + gw / 2}" y="${H - mb + 14}"` +
           ` text-anchor="${cats.length > 14 ? "start" : "middle"}"${rot}>${c}</text>`;
    });
    for (const r of actRefs) {
      const x0 = ml + r.i * gw + gw * 0.06, x1 = ml + r.i * gw + gw * 0.94;
      s += `<line x1="${x0}" x2="${x1}" y1="${Y(r.v)}" y2="${Y(r.v)}"` +
           ` class="${r.cls || "refline"}"><title>${r.label}: ${fmt(r.v)} cm</title></line>`;
    }
    s += "</svg>";
    holder.innerHTML = s;
    if (onBar) holder.querySelectorAll(".bar,.cat").forEach(b =>
        b.addEventListener("click", () => onBar(+b.dataset.cat)));
  }

  const leg = el("div", { class: "legend" });
  for (const sr of series) {
    const chip = el("span", { class: "chip", title: "click to toggle this series" },
        `<span class="dot" style="background:${sr.color}"></span>${sr.label}`);
    chip.onclick = () => { sr.on = !sr.on; chip.classList.toggle("off"); draw(); };
    leg.append(chip);
  }
  for (const g of refGroups) {
    const lbl = g === "refline" ? "published refs" : "baselines (ours-run)";
    const col = g === "refline" ? "var(--c-ref)" : "var(--c-base)";
    const chip = el("span", { class: "chip", title: "click to toggle" },
        `<span class="dot" style="background:${col}"></span>${lbl}`);
    chip.onclick = () => { refOn[g] = !refOn[g]; chip.classList.toggle("off"); draw(); };
    leg.append(chip);
  }
  card.append(leg);
  if (note) card.append(el("div", { class: "note" }, note));
  draw();
  return card;
}

/* ---------- zoomable trajectory viewer ---------- */
function trajViewer(data, armColor) {
  const W = 860, H = 520;
  const pts = data.gt.concat(data.est);
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  let x0 = Math.min(...xs), x1 = Math.max(...xs);
  let y0 = Math.min(...ys), y1 = Math.max(...ys);
  const pad = Math.max(x1 - x0, y1 - y0) * 0.06;
  x0 -= pad; x1 += pad; y0 -= pad; y1 += pad;
  const span = Math.max(x1 - x0, (y1 - y0) * W / H);
  let vb = { x: x0, y: y0, w: span, h: span * H / W };
  const path = a => a.map((p, i) =>
      `${i ? "L" : "M"}${p[0].toFixed(3)},${(-p[1]).toFixed(3)}`).join("");
  const wrap = el("div", { class: "traj-wrap" });
  wrap.innerHTML =
    `<svg id="tv" viewBox="${vb.x} ${-vb.y - vb.h} ${vb.w} ${vb.h}">` +
    `<path d="${path(data.gt)}" fill="none" stroke="var(--c-gt)" stroke-width="${span / 400}"/>` +
    `<path d="${path(data.est)}" fill="none" stroke="${armColor}" stroke-width="${span / 400}"/>` +
    `</svg>`;
  const svg = wrap.firstChild;
  const apply = () => svg.setAttribute("viewBox",
      `${vb.x} ${-vb.y - vb.h} ${vb.w} ${vb.h}`);
  wrap.addEventListener("wheel", e => {
    e.preventDefault();
    const k = e.deltaY > 0 ? 1.18 : 1 / 1.18;
    const r = wrap.getBoundingClientRect();
    const fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height;
    const nx = vb.x + fx * vb.w * (1 - k), nyTop = (-vb.y - vb.h) + fy * vb.h * (1 - k);
    vb = { x: nx, w: vb.w * k, h: vb.h * k, y: -(nyTop) - vb.h * k };
    apply();
  }, { passive: false });
  let drag = null;
  wrap.addEventListener("pointerdown", e => { drag = { x: e.clientX, y: e.clientY }; wrap.setPointerCapture(e.pointerId); });
  wrap.addEventListener("pointermove", e => {
    if (!drag) return;
    const r = wrap.getBoundingClientRect();
    vb.x -= (e.clientX - drag.x) / r.width * vb.w;
    vb.y += (e.clientY - drag.y) / r.height * vb.h;
    drag = { x: e.clientX, y: e.clientY };
    apply();
  });
  wrap.addEventListener("pointerup", () => drag = null);
  return wrap;
}

/* ---------- 3D orbit trajectory viewer (canvas) ---------- */
function orbitViewer(data, armColor) {
  const W = 880, H = 540;
  const wrap = el("div", { class: "traj-wrap" });
  const cv = el("canvas", { width: W, height: H, style: "width:100%;display:block" });
  wrap.append(cv);
  const ctx = cv.getContext("2d");
  const pts = data.gt.concat(data.est);
  const c = [0, 1, 2].map(k => pts.reduce((s, p) => s + p[k], 0) / pts.length);
  const span = Math.max(...pts.map(p => Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2]))) || 1;
  let yaw = 0.7, pitch = 0.5, zoom = 1, panX = 0, panY = 0;
  const errCol = e => {
    const t = Math.min(e / 60, 1);   // 0..60cm ramp
    return `hsl(${(1 - t) * 210},80%,55%)`;
  };
  function proj(p) {
    const x = p[0] - c[0], y = p[1] - c[1], z = p[2] - c[2];
    const cy = Math.cos(yaw), sy = Math.sin(yaw);
    const cp = Math.cos(pitch), sp = Math.sin(pitch);
    const x1 = x * cy - y * sy, y1 = x * sy + y * cy;
    const y2 = y1 * cp - z * sp, z2 = y1 * sp + z * cp;
    const s = (H * 0.42 * zoom) / span;
    return [W / 2 + x1 * s + panX, H / 2 - z2 * s + panY, y2];
  }
  function draw() {
    ctx.clearRect(0, 0, W, H);
    ctx.lineWidth = 1.2;
    // ground-truth: neutral
    ctx.strokeStyle = "#8888";
    ctx.beginPath();
    data.gt.forEach((p, i) => {
      const [x, y] = proj(p);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
    // estimate: error-colored segments when err available, else arm color
    for (let i = 1; i < data.est.length; i++) {
      const [x0, y0] = proj(data.est[i - 1]);
      const [x1, y1] = proj(data.est[i]);
      ctx.strokeStyle = data.err ? errCol(data.err[i] ?? 0) : armColor;
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    }
    ctx.fillStyle = "#999"; ctx.font = "11px system-ui";
    ctx.fillText("drag = orbit · wheel = zoom · shift-drag = pan · color = error (blue→red 0–60 cm)", 10, H - 10);
  }
  let drag = null;
  cv.addEventListener("pointerdown", e => { drag = { x: e.clientX, y: e.clientY, pan: e.shiftKey }; cv.setPointerCapture(e.pointerId); });
  cv.addEventListener("pointermove", e => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (drag.pan) { panX += dx; panY += dy; }
    else { yaw += dx * 0.008; pitch = Math.max(-1.55, Math.min(1.55, pitch + dy * 0.008)); }
    drag.x = e.clientX; drag.y = e.clientY;
    requestAnimationFrame(draw);
  });
  cv.addEventListener("pointerup", () => drag = null);
  wrap.addEventListener("wheel", e => {
    e.preventDefault();
    zoom *= e.deltaY > 0 ? 1 / 1.15 : 1.15;
    requestAnimationFrame(draw);
  }, { passive: false });
  draw();
  return wrap;
}

/* ---------- error-over-time strip (drag to zoom, dblclick reset) ---------- */
function errTimeline(data) {
  if (!data.err || !data.t) return null;
  const W = 880, H = 170, ml = 42, mb = 24, mt = 8;
  const wrap = el("div", { class: "traj-wrap" });
  const cv = el("canvas", { width: W, height: H, style: "width:100%;display:block" });
  wrap.append(cv);
  const ctx = cv.getContext("2d");
  let x0 = 0, x1 = data.t[data.t.length - 1];
  let sel = null;
  function draw() {
    ctx.clearRect(0, 0, W, H);
    const inRange = data.t.map((t, i) => [t, data.err[i]]).filter(p => p[0] >= x0 && p[0] <= x1);
    const ymax = Math.max(10, ...inRange.map(p => p[1])) * 1.1;
    const X = t => ml + (t - x0) / (x1 - x0 || 1) * (W - ml - 8);
    const Y = v => mt + (1 - v / ymax) * (H - mt - mb);
    ctx.strokeStyle = "#8884"; ctx.fillStyle = "#999"; ctx.font = "10px system-ui";
    for (let g = 0; g <= 3; g++) {
      const v = ymax * g / 3;
      ctx.beginPath(); ctx.moveTo(ml, Y(v)); ctx.lineTo(W - 8, Y(v)); ctx.stroke();
      ctx.fillText(v.toFixed(0), 6, Y(v) + 3);
    }
    ctx.strokeStyle = "#e34948"; ctx.lineWidth = 1.4; ctx.beginPath();
    inRange.forEach((p, i) => {
      i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1]));
    });
    ctx.stroke();
    ctx.fillText("position error [cm] vs time [s] — drag to zoom, double-click to reset",
                 ml, H - 8);
    if (sel) {
      ctx.fillStyle = "rgba(42,120,214,.18)";
      ctx.fillRect(Math.min(sel.a, sel.b), mt, Math.abs(sel.b - sel.a), H - mt - mb);
    }
  }
  const tAt = px => x0 + (px - ml) / (W - ml - 8) * (x1 - x0);
  cv.addEventListener("pointerdown", e => {
    const r = cv.getBoundingClientRect();
    sel = { a: (e.clientX - r.left) * W / r.width, b: (e.clientX - r.left) * W / r.width };
    cv.setPointerCapture(e.pointerId);
  });
  cv.addEventListener("pointermove", e => {
    if (!sel) return;
    const r = cv.getBoundingClientRect();
    sel.b = (e.clientX - r.left) * W / r.width;
    requestAnimationFrame(draw);
  });
  cv.addEventListener("pointerup", () => {
    if (sel && Math.abs(sel.b - sel.a) > 12) {
      const [a, b] = [tAt(Math.min(sel.a, sel.b)), tAt(Math.max(sel.a, sel.b))];
      x0 = a; x1 = b;
    }
    sel = null;
    draw();
  });
  cv.addEventListener("dblclick", () => { x0 = 0; x1 = data.t[data.t.length - 1]; draw(); });
  draw();
  return wrap;
}

/* ---------- drill-down ---------- */
async function drill(seq) {
  $("#drill-title").textContent = shortName(seq);
  const body = $("#drill-body");
  body.innerHTML = "";
  const runs = S.runs.filter(r => r.seq === seq);
  // run-level scatter table
  const t = el("table");
  t.innerHTML = "<tr><th>arm</th><th>track</th>" +
      [1, 2, 3, 4, 5].map(i => `<th>r${i}</th>`).join("") + "<th>median</th></tr>";
  for (const arm of ARMS.filter(a => S.armOn[a])) {
    for (const track of ["vio", "map"]) {
      const rr = runs.filter(r => r.arm === arm && r.track === track)
          .sort((a, b) => a.run - b.run);
      if (!rr.length) continue;
      const cells = [1, 2, 3, 4, 5].map(i => {
        const r = rr.find(x => x.run === i);
        return `<td>${r ? fmt(S.useRte ? r.rte_cm : r.ate_cm) : "—"}</td>`;
      }).join("");
      t.innerHTML += `<tr><td><span class="dot" style="background:${cssColor(arm)};` +
          `display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px">` +
          `</span>${ARM_LABEL[arm]}</td><td>${track}</td>${cells}` +
          `<td><b>${fmt(median(rr.map(x => S.useRte ? x.rte_cm : x.ate_cm)))}</b></td></tr>`;
    }
  }
  body.append(el("h3", {}, "runs (ATE cm)"), t);
  // trajectory: 3D orbit + error timeline + 2D top view
  for (const arm of ARMS.filter(a => S.armOn[a])) {
    const d = await loadJSON(`data/traj/${seq}_${arm}.json`, null);
    if (!d) continue;
    const is3d = d.est[0] && d.est[0].length >= 3;
    body.append(el("h3", {}, `trajectory — ${ARM_LABEL[arm]} (map track, r1) vs ground truth`));
    if (is3d) {
      body.append(orbitViewer(d, cssColor(arm)));
      const et = errTimeline(d);
      if (et) body.append(et);
    } else {
      body.append(trajViewer(d, cssColor(arm)),
                  el("div", { class: "hint" }, "wheel = zoom · drag = pan · gray = ground truth"));
    }
    break;
  }
  $("#drill").classList.remove("hidden");
}

/* ---------- views ---------- */
function datasetView(group) {
  const seqs = seqsIn(group);
  const med = medians(S.runs.filter(r => r.group === group), S.useRte ? "rte" : "ate");
  const view = $("#view");
  const armsOn = ARMS.filter(a => S.armOn[a]);
  const series = [
    { label: "VIO only", color: cssColor("bad"), values: seqs.map(s => med[s]?.bad?.vio) },
    ...armsOn.map(a => ({
      label: `+map ${ARM_LABEL[a]}`, color: cssColor(a),
      values: seqs.map(s => med[s]?.[a]?.map),
    })),
  ];
  const refs = [];
  if (S.showPub)
    for (const p of S.published.filter(p => p.group === group)) {
      const i = seqs.indexOf(p.seq);
      if (i >= 0 && !S.useRte)
        refs.push({ i, v: p.ate_cm, label: `${p.system} (${p.regime})` });
    }
  if (S.showBase)
    for (const b of aggBaselines().filter(b => group_of_(b.seq) === group)) {
      const i = seqs.indexOf(b.seq);
      if (i >= 0)
        refs.push({ i, v: S.useRte ? b.rte : b.ate, cls: "baseline-mark",
                    label: `${b.sys} lc${b.lc} (ours-run, causal)` });
    }
  view.append(barChart({
    title: `${GROUPS.find(g => g[0] === group)[1]} — causal ${S.useRte ? "RTE" : "ATE"} medians [cm]`,
    cats: seqs.map(shortName), series, refs,
    note: "click a bar or label for run-level detail + trajectory. Yellow dashes = " +
          "published references (regime annotated on hover); red dashes = baselines we ran.",
    onBar: i => drill(seqs[i]),
  }));
}
const group_of_ = seq => /^(MH_|V1_|V2_)/.test(seq) ? "euroc"
    : /^(MOO|MIO|MIP|MGO)/.test(seq) ? "msd" : /room/.test(seq) ? "rooms" : "long";
function aggBaselines() {
  const acc = {};
  for (const r of S.baselines) {
    const key = `${r.seq}|${r.arm}|${r.track}`;
    (acc[key] ??= []).push(r);
  }
  return Object.entries(acc).map(([k, rr]) => {
    const [seq, sys, lc] = k.split("|");
    return { seq, sys, lc: lc.replace("lc", ""),
             ate: median(rr.map(r => r.ate_cm)), rte: median(rr.map(r => r.rte_cm)) };
  });
}

function overviewView() {
  const view = $("#view");
  const med = medians(S.runs, "ate");
  const groupAgg = g => {
    const seqs = seqsIn(g);
    const v = seqs.map(s => med[s]?.vpr?.map ?? med[s]?.bad?.map).filter(x => x != null);
    return v.length ? median(v) : null;
  };
  const tiles = el("div", { class: "tiles" });
  const nRuns = S.runs.length / 2;
  const tileDefs = [
    [new Set(S.runs.map(r => r.seq)).size, "sequences"],
    [nRuns.toFixed(0), "scored runs"],
    [fmt(groupAgg("euroc")), "EuRoC med ATE cm (+map)"],
    [fmt(groupAgg("msd")), "MSD med ATE cm (+map)"],
    [fmt(groupAgg("long")), "TUM-VI long med cm (+map)"],
    [new Set(S.runs.map(r => r.arm)).size, "arms measured"],
  ];
  for (const [v, l] of tileDefs)
    tiles.append(el("div", { class: "tile" }, `<b>${v}</b><span>${l}</span>`));
  view.append(tiles);
  for (const g of ["euroc", "rooms", "long", "msd"]) datasetViewInto(g, view);
}
function datasetViewInto(group, view) {
  const old = S.tab; S.tab = group;
  const tmp = $("#view");
  datasetView(group);
  S.tab = old;
}

function systemsView() {
  const view = $("#view");
  const base = aggBaselines();
  const seqs = [...new Set(base.map(b => b.seq))].sort();
  const med = medians(S.runs.filter(r => seqs.includes(r.seq)), S.useRte ? "rte" : "ate");
  const sysArms = [...new Set(base.map(b => `${b.sys}_lc${b.lc}`))].sort();
  const colors = ["#e34948", "#eda100", "#8a2be2", "#1baf7a", "#2a78d6"];
  const series = [
    { label: "ours: VIO", color: cssColor("bad"), values: seqs.map(s => med[s]?.bad?.vio) },
    { label: "ours: +map (best arm)", color: cssColor("vpr"),
      values: seqs.map(s => {
        const v = ARMS.map(a => med[s]?.[a]?.map).filter(x => x != null);
        return v.length ? Math.min(...v) : null;
      }) },
    ...sysArms.map((sa, i) => ({
      label: sa.replace("_", " "), color: colors[i % colors.length],
      values: seqs.map(s => {
        const b = base.find(x => x.seq === s && `${x.sys}_lc${x.lc}` === sa);
        return b ? (S.useRte ? b.rte : b.ate) : null;
      }),
    })),
  ];
  view.append(barChart({
    title: `Ours vs systems we ran — causal ${S.useRte ? "RTE" : "ATE"} medians [cm]`,
    cats: seqs.map(shortName), series,
    note: "All rows same machine, same causal protocol. Note: whole-run causal ATE " +
          "penalizes loop-closure re-anchoring jumps for ALL systems; see Method.",
    onBar: i => drill(seqs[i]),
  }));
}

function tableView() {
  const view = $("#view");
  const med = medians(S.runs, "ate");
  const medR = medians(S.runs, "rte");
  const t = el("table");
  const armsOn = ARMS.filter(a => S.armOn[a]);
  t.innerHTML = `<tr><th>sequence</th><th>VIO ATE</th><th>VIO RTE</th>` +
      armsOn.map(a => `<th>+map ${ARM_LABEL[a]}</th>`).join("") + `</tr>`;
  for (const g of ["euroc", "rooms", "long", "msd"]) {
    t.innerHTML += `<tr class="ghead"><td colspan="${3 + armsOn.length}">${g}</td></tr>`;
    for (const s of seqsIn(g)) {
      t.innerHTML += `<tr><td>${shortName(s)}</td>` +
        `<td>${fmt(med[s]?.bad?.vio)}</td><td>${fmt(medR[s]?.bad?.vio)}</td>` +
        armsOn.map(a => `<td>${fmt(med[s]?.[a]?.map)}</td>`).join("") + `</tr>`;
    }
  }
  view.append(t);
}

function methodView() {
  $("#view").append(el("div", { class: "chart-card" }, `
    <h3>Protocol</h3>
    <p>${S.meta.protocol || ""}. One replay emits both tracks: raw VIO pose and
    map-corrected session pose (the map layer has no feedback into VIO). Arms
    differ only in descriptor (BAD/TEBLID vs XFeat) and retrieval model
    (none / EigenPlaces-512 / MegaLoc-8448).</p>
    <h3>Honest-comparison notes</h3>
    <p>Whole-run causal ATE penalizes any loop-closure system at the moment a
    correction re-anchors the live pose (past poses stay in the old frame) —
    ORB-SLAM3 shows metre-scale steps at closures while its tail-window ATE is
    centimetres. Compare like-for-like: causal columns against causal, and see
    tail-window metrics where reported. Published rows carry their regime
    (causal?, IMU?, LC?, compression) in the hover title.</p>
    <h3>Reproduce</h3>
    <p><code>bench/replay</code> (harness) · <code>bench/host</code> (prep/score/export)
    · <code>node bench/site/server.js</code> (this site) · data regenerated via
    <code>export_site_data.py</code>.</p>`));
}

/* ---------- shell ---------- */
function render() {
  $("#view").innerHTML = "";
  ({ overview: overviewView, systems: systemsView, table: tableView,
     method: methodView }[S.tab] || (() => datasetView(S.tab)))();
}
function buildShell() {
  const tabs = $("#tabs");
  for (const [id, label] of GROUPS) {
    const b = el("button", {}, label);
    b.onclick = () => { S.tab = id; document.querySelectorAll("nav button")
        .forEach(x => x.classList.remove("on")); b.classList.add("on"); render(); };
    if (id === S.tab) b.classList.add("on");
    tabs.append(b);
  }
  const chips = $("#arm-chips");
  for (const a of ARMS) {
    const c = el("span", { class: "chip" },
        `<span class="dot" style="background:${cssColor(a)}"></span>${ARM_LABEL[a]}`);
    c.onclick = () => { S.armOn[a] = !S.armOn[a]; c.classList.toggle("off"); render(); };
    chips.append(c);
  }
  $("#show-pub").onchange = e => { S.showPub = e.target.checked; render(); };
  $("#show-base").onchange = e => { S.showBase = e.target.checked; render(); };
  $("#metric-rte").onchange = e => { S.useRte = e.target.checked; render(); };
  $("#drill-close").onclick = () => $("#drill").classList.add("hidden");
  $("#drill").onclick = e => { if (e.target.id === "drill") $("#drill").classList.add("hidden"); };
}

let lastGenerated = null;
async function refreshData(first) {
  const [res, base, pub, led, meta] = await Promise.all([
    loadJSON("data/results.json", { runs: [] }),
    loadJSON("data/baselines.json", { runs: [] }),
    loadJSON("data/published.json", []),
    loadJSON("data/ledger.json", []),
    loadJSON("data/meta.json", {}),
  ]);
  const changed = res.generated !== lastGenerated;
  lastGenerated = res.generated;
  S.runs = res.runs; S.baselines = base.runs; S.published = pub;
  S.ledger = led; S.meta = meta;
  $("#meta-line").innerHTML =
    `<span id="live-dot" title="auto-refreshing">●</span> LIVE · ` +
    `${S.runs.length} run-scores · data ${res.generated || "?"} · causal protocol`;
  if (first || changed) render();
}
(async () => {
  await refreshData(true);
  buildShell();
  render();
  setInterval(() => refreshData(false), 45000);   // live: poll every 45 s
})();
