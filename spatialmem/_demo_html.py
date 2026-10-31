"""Self-contained HTML/JS viewer template for spatialmem.demo (no network, no libraries). `__DATA__` is replaced by the payload JSON."""

TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Spatial map viewer</title>
<style>
:root{--bg:#14161a;--panel:#1d2026;--fg:#e6e8eb;--dim:#9aa1ab;--line:#2c3038;--ok:#2fb344;--bad:#e5484d;--miss:#f5a524}
html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);font:13px/1.4 system-ui,Segoe UI,sans-serif}
#app{display:flex;height:100%}
#side{width:290px;flex:none;background:var(--panel);border-right:1px solid var(--line);padding:12px 14px;overflow:auto;box-sizing:border-box}
#view{flex:1;position:relative;min-width:0}
canvas{position:absolute;inset:0;width:100%;height:100%;display:block;cursor:grab}
canvas.drag{cursor:grabbing}
h1{font-size:15px;margin:0 0 2px}h2{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--dim);margin:14px 0 5px}
.sub{color:var(--dim);font-size:12px}
table{border-collapse:collapse;width:100%}td{padding:1px 0}td+td{text-align:right;font-variant-numeric:tabular-nums}
label{display:block;margin:2px 0;cursor:pointer}select,button{background:#262a32;color:var(--fg);border:1px solid var(--line);border-radius:4px;padding:3px 7px;font:inherit}
button{cursor:pointer}input[type=range]{width:100%}
.sw{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:6px;vertical-align:-1px}
#tip{position:absolute;pointer-events:none;background:rgba(10,12,15,.92);border:1px solid var(--line);padding:4px 7px;border-radius:4px;display:none;white-space:pre;font-size:12px}
#help{position:absolute;left:10px;bottom:8px;color:var(--dim);font-size:11px;pointer-events:none}
</style></head><body><div id="app">
<div id="side">
<h1 id="title"></h1><div class="sub" id="sub"></div>
<h2>Score @ IoU 0.25 (project scorer)</h2><table id="score"></table>
<h2>Show</h2>
<div>View <select id="view"><option value="map">Map (ConceptGraphs)</option><option value="gt">Ground truth</option><option value="both">Both</option></select></div>
<div style="margin-top:4px">Colour <select id="mode"><option value="obj">per object</option><option value="match">match</option></select></div>
<label><input type="checkbox" id="cPts" checked> points</label>
<label><input type="checkbox" id="cBox" checked> boxes</label>
<label><input type="checkbox" id="cLab"> labels (always)</label>
<label><input type="checkbox" id="cRel" checked> relations on / in</label>
<label><input type="checkbox" id="cNext"> relations next_to</label>
<label><input type="checkbox" id="cStruct"> structural (wall/floor/ceiling)</label>
<div style="margin-top:4px">Point size <input type="range" id="ps" min="1" max="5" value="2"></div>
<div style="margin-top:6px"><button id="bTop">Top</button> <button id="bReset">Reset view</button></div>
<h2>Legend</h2><div id="legend"></div>
</div>
<div id="view"><canvas id="cv"></canvas><div id="tip"></div>
<div id="help">drag: orbit &middot; wheel: zoom &middot; shift-drag / right-drag: pan &middot; hover: object info</div></div>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>
"use strict";
const D = JSON.parse(document.getElementById("data").textContent);
const $ = id => document.getElementById(id);
const REL = {on:"#ff6b6b", "in":"#4dabf7", next_to:"#868e96"};
const ST = {m:[47,179,68], x:[229,72,77], u:[245,165,36], i:[110,116,125]};
const STN = {m:"matched", x:"extra prediction", u:"missed GT", i:"structural (ignored by scorer)"};
const hex = c => "#" + c.map(v => v.toString(16).padStart(2, "0")).join("");
function hsl(h, s, l){ // -> [r,g,b]
  const a = s * Math.min(l, 1 - l), f = n => { const k = (n + h * 12) % 12; return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1)); };
  return [f(0) * 255 | 0, f(8) * 255 | 0, f(4) * 255 | 0];
}
const objCol = i => hsl(((i * 0.61803398875) % 1), 0.65, 0.58);

// ---- data
const P = D.pred, G = D.gt;
const N = P.length, np_ = D.pts.off;
const raw = Uint8Array.from(atob(D.pts.b64), c => c.charCodeAt(0));
const q = new Int16Array(raw.buffer, raw.byteOffset, raw.byteLength >> 1);
const NP = q.length / 3, X = new Float32Array(NP * 3), owner = new Int32Array(NP);
for (let i = 0; i < NP; i++) for (let k = 0; k < 3; k++) X[i * 3 + k] = D.pts.lo[k] + (q[i * 3 + k] + 32768) * D.pts.sc[k];
for (let o = 0; o < N; o++) for (let i = np_[o]; i < np_[o + 1]; i++) owner[i] = o;
const pById = new Map(P.map(o => [o.id, o])), gById = new Map(G.map(o => [o.id, o]));
const cen = o => [(o.min[0] + o.max[0]) / 2, (o.min[1] + o.max[1]) / 2, (o.min[2] + o.max[2]) / 2];
const vol = o => (o.max[0] - o.min[0]) * (o.max[1] - o.min[1]) * (o.max[2] - o.min[2]);

// ---- side panel
$("title").textContent = "HSSD " + D.scene;
$("sub").textContent = `${D.n_pred_total} map objects (${D.n_pred} scored), ${D.n_gt} GT objects, ${D.n_points_total.toLocaleString()} map points (${NP.toLocaleString()} shown)`;
const f2 = v => (v === null ? "n/a" : v.toFixed(3));
$("score").innerHTML = [
  ["objects: pred / GT / matched", `${D.score.n_pred} / ${D.score.n_gt} / ${D.score.tp}`],
  ["precision", f2(D.score.p)], ["recall", f2(D.score.r)], ["F1", f2(D.score.f1)], ["label accuracy", f2(D.score.label_acc)],
  ["relation P / R / F1", `${f2(D.rel.p)} / ${f2(D.rel.r)} / ${f2(D.rel.f1)}`]
].map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join("");
const cnt = k => P.filter(o => o.st === k).length;
$("legend").innerHTML =
  ["m","x","u"].map(k => `<div><span class="sw" style="background:${hex(ST[k])}"></span>${STN[k]} (${k === "u" ? G.filter(o => o.st === "u").length : cnt(k)})</div>`).join("") +
  `<div><span class="sw" style="background:${hex(ST.i)}"></span>${STN.i}</div>` +
  `<div style="margin-top:4px">relations: <span style="color:${REL.on}">on</span> &middot; <span style="color:${REL["in"]}">in</span> &middot; <span style="color:${REL.next_to}">next_to</span> (solid = map, dashed = GT)</div>` +
  `<div class="sub" style="margin-top:4px">per-object colour: arbitrary, one per map object. Match mode: matched/extra = map objects, missed = GT objects. Up axis +y.</div>`;

// ---- camera
const cv = $("cv"), ctx = cv.getContext("2d"), tip = $("tip");
let W = 0, H = 0, dpr = 1, img = null, buf = null, zb = null;
const lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9];
for (const o of P.concat(G)) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], o.min[k]); hi[k] = Math.max(hi[k], o.max[k]); }
const ctr0 = [0, 1, 2].map(k => (lo[k] + hi[k]) / 2), diag = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
const cam = {t: ctr0.slice(), d: diag * 1.1, yaw: 0.6, pitch: 0.75};
function resetCam(){ cam.t = ctr0.slice(); cam.d = diag * 1.1; cam.yaw = 0.6; cam.pitch = 0.75; draw(); }
function topCam(){ cam.pitch = 1.5707; cam.yaw = 0; cam.t = ctr0.slice(); cam.d = diag * 1.5; draw(); }
let V; // view basis
function setView(){
  const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
  const e = [cam.t[0] + cam.d * cp * Math.sin(cam.yaw), cam.t[1] + cam.d * sp, cam.t[2] + cam.d * cp * Math.cos(cam.yaw)];
  let f = [cam.t[0] - e[0], cam.t[1] - e[1], cam.t[2] - e[2]]; const fl = Math.hypot(...f); f = f.map(v => v / fl);
  let r = [-f[2], 0, f[0]]; const rl = Math.hypot(...r) || 1; r = r.map(v => v / rl);
  const u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]];
  V = {e, r, u, f, focal: 0.5 * Math.min(W, H) / Math.tan(0.4)};
}
function camSpace(p){
  const a = p[0] - V.e[0], b = p[1] - V.e[1], c = p[2] - V.e[2];
  return [a * V.r[0] + b * V.r[1] + c * V.r[2], a * V.u[0] + b * V.u[1] + c * V.u[2], a * V.f[0] + b * V.f[1] + c * V.f[2]];
}
const scr = c => [W / 2 + V.focal * c[0] / c[2], H / 2 - V.focal * c[1] / c[2]];
const NEAR = 0.05;

// ---- state
const S = () => ({view: $("view").value, mode: $("mode").value, pts: $("cPts").checked, box: $("cBox").checked, lab: $("cLab").checked,
  rel: $("cRel").checked, nxt: $("cNext").checked, str: $("cStruct").checked, ps: +$("ps").value});
let hover = null;
function colOf(o, side, mode){
  if (mode === "match") return ST[o.st];
  return side === "gt" ? [200, 205, 212] : objCol(o.id);
}
const EDGES = [[0,1],[2,3],[4,5],[6,7],[0,2],[1,3],[4,6],[5,7],[0,4],[1,5],[2,6],[3,7]];
function corners(o){ const c = []; for (let i = 0; i < 8; i++) c.push([i & 1 ? o.max[0] : o.min[0], i & 2 ? o.max[1] : o.min[1], i & 4 ? o.max[2] : o.min[2]]); return c; }
function line(a, b){ // camera-space points, clip to near plane, draw
  if (a[2] < NEAR && b[2] < NEAR) return;
  if (a[2] < NEAR) a = lerpNear(a, b); else if (b[2] < NEAR) b = lerpNear(b, a);
  const p = scr(a), q2 = scr(b); ctx.moveTo(p[0], p[1]); ctx.lineTo(q2[0], q2[1]);
}
function lerpNear(a, b){ const t = (NEAR - a[2]) / (b[2] - a[2]); return [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), NEAR]; }

function visible(o, s){ return s.str || o.st !== "i"; }
function items(s){ // {o, side}
  const out = [];
  if (s.view !== "gt") for (const o of P) if (visible(o, s)) out.push({o, side: "map"});
  if (s.view !== "map") for (const o of G) if (visible(o, s)) out.push({o, side: "gt"});
  return out;
}

function drawPoints(s){
  const bg = 0xff1a1614; buf.fill(bg); zb.fill(1e30);
  if (!s.pts || s.view === "gt") { ctx.putImageData(img, 0, 0); return; }
  const colr = new Uint32Array(N), show = new Uint8Array(N);
  for (let o = 0; o < N; o++) { const c = s.mode === "match" ? ST[P[o].st] : objCol(o); colr[o] = 0xff000000 | (c[2] << 16) | (c[1] << 8) | c[0]; show[o] = visible(P[o], s); }
  const ps = Math.max(1, Math.round(s.ps * dpr * 0.8)), foc = V.focal;
  const [rx, ry, rz] = V.r, [ux, uy, uz] = V.u, [fx, fy, fz] = V.f, [ex, ey, ez] = V.e;
  for (let i = 0; i < NP; i++) {
    const o = owner[i]; if (!show[o]) continue;
    const a = X[i * 3] - ex, b = X[i * 3 + 1] - ey, c = X[i * 3 + 2] - ez;
    const z = a * fx + b * fy + c * fz; if (z < NEAR) continue;
    const x = (W / 2 + foc * (a * rx + b * ry + c * rz) / z) | 0, y = (H / 2 - foc * (a * ux + b * uy + c * uz) / z) | 0;
    for (let dy = 0; dy < ps; dy++) { const yy = y + dy; if (yy < 0 || yy >= H) continue;
      for (let dx = 0; dx < ps; dx++) { const xx = x + dx; if (xx < 0 || xx >= W) continue;
        const k = yy * W + xx; if (z < zb[k]) { zb[k] = z; buf[k] = colr[o]; } } }
  }
  ctx.putImageData(img, 0, 0);
}

function draw(){
  if (!W) return; setView(); const s = S();
  drawPoints(s);
  const its = items(s), cs = new Map();
  const cenOf = (o, side) => { const k = side + o.id; if (!cs.has(k)) cs.set(k, camSpace(cen(o))); return cs.get(k); };
  // relations
  if (s.rel || s.nxt) {
    ctx.lineWidth = dpr;
    for (const side of ["map", "gt"]) {
      if ((side === "map" && s.view === "gt") || (side === "gt" && s.view === "map")) continue;
      const byId = side === "map" ? pById : gById, rels = side === "map" ? D.prel : D.grel;
      ctx.setLineDash(side === "gt" && s.view === "both" ? [5 * dpr, 4 * dpr] : []);
      for (const t of ["on", "in", "next_to"]) {
        if (t === "next_to" ? !s.nxt : !s.rel) continue;
        ctx.strokeStyle = REL[t]; ctx.globalAlpha = t === "next_to" ? 0.45 : 0.85; ctx.beginPath();
        for (const [a, b, p] of rels) { if (p !== t) continue; const oa = byId.get(a), ob = byId.get(b); if (!oa || !ob || !visible(oa, s) || !visible(ob, s)) continue;
          line(cenOf(oa, side), cenOf(ob, side)); }
        ctx.stroke();
      }
    }
    ctx.setLineDash([]); ctx.globalAlpha = 1;
  }
  // boxes
  if (s.box) {
    for (const {o, side} of its) {
      const hv = hover && hover.o === o && hover.side === side;
      const col = colOf(o, side, s.mode); ctx.strokeStyle = hv ? "#fff" : hex(col); ctx.lineWidth = (hv ? 2.6 : 1.1) * dpr;
      ctx.globalAlpha = hv ? 1 : (side === "gt" && s.view === "both" ? 0.7 : 0.9);
      ctx.setLineDash(side === "gt" && s.view === "both" ? [5 * dpr, 4 * dpr] : []);
      const c = corners(o).map(camSpace); ctx.beginPath(); for (const [i, j] of EDGES) line(c[i], c[j]); ctx.stroke();
    }
    ctx.setLineDash([]); ctx.globalAlpha = 1;
  }
  // labels
  ctx.font = `${11 * dpr}px system-ui,sans-serif`; ctx.textBaseline = "middle"; ctx.textAlign = "center";
  const placed = [];
  const lab = (o, side, force) => {
    const c = camSpace([(o.min[0] + o.max[0]) / 2, o.max[1], (o.min[2] + o.max[2]) / 2]); if (c[2] < NEAR) return;
    const [x, y] = scr(c), t = o.label, w = ctx.measureText(t).width + 6 * dpr, h = 14 * dpr, r = [x - w / 2, y - h / 2 - 8 * dpr, w, h];
    if (!force && placed.some(p => r[0] < p[0] + p[2] && p[0] < r[0] + r[2] && r[1] < p[1] + p[3] && p[1] < r[1] + r[3])) return;
    placed.push(r); ctx.fillStyle = "rgba(10,12,15,.78)"; ctx.fillRect(r[0], r[1], r[2], r[3]);
    ctx.fillStyle = force ? "#fff" : hex(side === "gt" && s.mode === "obj" ? [200, 205, 212] : colOf(o, side, s.mode)); ctx.fillText(t, x, r[1] + h / 2);
  };
  if (hover) lab(hover.o, hover.side, true);
  if (s.lab) for (const {o, side} of its.slice().sort((a, b) => vol(b.o) - vol(a.o))) if (!(hover && hover.o === o)) lab(o, side, false);
}

// ---- hover
function pick(mx, my){
  const s = S(); let best = null, bd = 18 * dpr;
  for (const {o, side} of items(s)) {
    const c = camSpace(cen(o)); if (c[2] < NEAR) continue; const [x, y] = scr(c);
    const d = Math.hypot(x - mx, y - my); if (d < bd) { bd = d; best = {o, side}; }
  }
  return best;
}
cv.addEventListener("mousemove", e => {
  if (drag) return; const r = cv.getBoundingClientRect(), h = pick((e.clientX - r.left) * dpr, (e.clientY - r.top) * dpr);
  const changed = (h && hover && h.o === hover.o && h.side === hover.side) ? false : (h || hover) ? true : false;
  hover = h; if (changed) draw();
  if (!h) { tip.style.display = "none"; return; }
  const o = h.o, mate = h.side === "map" ? (o.mate !== null ? gById.get(o.mate) : null) : (o.mate !== null ? pById.get(o.mate) : null);
  tip.textContent = `${h.side === "map" ? "map" : "GT"} #${o.id}  ${o.label}\n${STN[o.st]}` + (mate ? `\nIoU ${o.iou.toFixed(2)} with ${h.side === "map" ? "GT" : "map"} #${mate.id} ${mate.label}` : "") +
    `\nsize ${[0,1,2].map(k => (o.max[k] - o.min[k]).toFixed(2)).join(" x ")} m`;
  tip.style.display = "block"; tip.style.left = Math.min(e.clientX - r.left + 14, r.width - 220) + "px"; tip.style.top = (e.clientY - r.top + 14) + "px";
});
cv.addEventListener("mouseleave", () => { if (hover) { hover = null; draw(); } tip.style.display = "none"; });

// ---- camera interaction
let drag = null;
cv.addEventListener("contextmenu", e => e.preventDefault());
cv.addEventListener("mousedown", e => { drag = {x: e.clientX, y: e.clientY, pan: e.shiftKey || e.button === 2 || e.button === 1}; cv.classList.add("drag"); tip.style.display = "none"; });
window.addEventListener("mouseup", () => { drag = null; cv.classList.remove("drag"); });
window.addEventListener("mousemove", e => {
  if (!drag) return; const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY;
  if (drag.pan) { setView(); const k = cam.d / V.focal * dpr; for (let i = 0; i < 3; i++) cam.t[i] += (-dx * V.r[i] + dy * V.u[i]) * k; }
  else { cam.yaw -= dx * 0.008; cam.pitch = Math.max(-1.5707, Math.min(1.5707, cam.pitch + dy * 0.008)); }
  draw();
});
cv.addEventListener("wheel", e => { e.preventDefault(); cam.d = Math.max(diag * 0.02, Math.min(diag * 6, cam.d * Math.exp(e.deltaY * 0.001))); draw(); }, {passive: false});
for (const id of ["view","mode","cPts","cBox","cLab","cRel","cNext","cStruct","ps"]) $(id).addEventListener("input", draw);
$("bTop").onclick = topCam; $("bReset").onclick = resetCam;

function resize(){
  dpr = window.devicePixelRatio || 1; const r = cv.getBoundingClientRect(); W = Math.max(1, Math.round(r.width * dpr)); H = Math.max(1, Math.round(r.height * dpr));
  cv.width = W; cv.height = H; img = ctx.createImageData(W, H); buf = new Uint32Array(img.data.buffer); zb = new Float32Array(W * H); draw();
}
window.addEventListener("resize", resize); resize();
</script></body></html>
"""
