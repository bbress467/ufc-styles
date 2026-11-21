/* Interactive UFC style map. Data: data/fighters.json (written by src/export_site_data.py). */
(function () {
  "use strict";

  const chartEl = document.getElementById("chart");
  const messageEl = document.getElementById("chart-message");
  const tooltipEl = document.getElementById("tooltip");
  const input = document.getElementById("search-input");
  const clearBtn = document.getElementById("search-clear");
  const list = document.getElementById("search-suggestions");
  const statusEl = document.getElementById("search-status");

  const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                  "November", "December"];
  const normalize = (s) => s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();

  let DATA = null;
  let byId = new Map();
  let selectedId = null;
  let state = null;          // per-render geometry: scales, screen positions, delaunay, svg layers
  let stickyTooltip = false; // true while a tapped or searched fighter's tooltip should stay open

  fetch("data/fighters.json")
    .then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then((data) => { DATA = data; init(); })
    .catch(() => {
      messageEl.classList.add("error");
      messageEl.innerHTML = "The map data couldn't be loaded. If you opened this file directly, serve the <code>site/</code> " +
        "folder with a local web server instead (see the project README).";
    });

  function init() {
    messageEl.remove();
    DATA.fighters.forEach((f) => { f.key = normalize(f.name); byId.set(f.id, f); });
    const quad = {};
    DATA.meta.quadrants.forEach((q) => { quad[q.name] = q; });
    DATA.quad = quad;
    const [y, m, d] = DATA.meta.data_through.split("-");
    document.getElementById("data-through").textContent = `${MONTHS[+m - 1]} ${y}`;
    // Dateline under the title, from the dataset's first year and last fight date (updates when the data is refreshed)
    document.getElementById("data-range").textContent =
      `Based on UFC fights from ${DATA.meta.data_from.slice(0, 4)} through ${MONTHS[+m - 1]} ${+d}, ${y}`;
    document.getElementById("source-link").href = DATA.meta.source.url;
    document.getElementById("repo-link").href = DATA.meta.repo_url;
    document.getElementById("legend-all").textContent =
      `all ${DATA.meta.n_fighters.toLocaleString("en-US")} fighters with ${DATA.meta.min_fights}+ UFC fights since 2000`;
    document.getElementById("note").textContent = DATA.meta.note;
    render();
    let raf = null;
    new ResizeObserver(() => { cancelAnimationFrame(raf); raf = requestAnimationFrame(render); }).observe(chartEl);
    setupSearch();
  }

  // ------------------------------------------------------------------ rendering
  function render() {
    const width = Math.max(300, chartEl.clientWidth);
    if (state && state.width === width) return;
    const narrow = width < 640;
    const height = Math.round(narrow ? Math.min(width * 1.55, 720) : Math.min(width * 0.68, 760));
    const margin = { top: narrow ? 26 : 30, right: 1, bottom: narrow ? 50 : 58, left: 1 };
    const fs = narrow ? { name: 12.5, label: 12, end: 12.5 } : { name: 17, label: 13.5, end: 15 };

    d3.select(chartEl).select("svg").remove();
    hideTooltip(true);
    const svg = d3.select(chartEl).insert("svg", ":first-child")
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("role", "img")
      .attr("aria-label", "Scatter plot of UFC fighters by style: grappling vs. striking volume across, and clinch and " +
        "body work vs. head strikes at range up and down, split into four shaded quadrants.");

    const F = DATA.fighters;
    const xs = F.map((f) => f.axis_1), ys = F.map((f) => f.axis_2);
    const xDom = [d3.min(xs) - 0.6, d3.max(xs) + 1.0];
    const yDom = [d3.min(ys) - 0.75, d3.max(ys) + 0.7];
    const plot = { x0: margin.left, x1: width - margin.right, y0: margin.top, y1: height - margin.bottom };
    const x = d3.scaleLinear().domain(xDom).range([plot.x0, plot.x1]);
    const y = d3.scaleLinear().domain(yDom).range([plot.y1, plot.y0]);
    const cx = x(0), cy = y(0);

    // Quadrant tints: each fades from its outer corner (full tint) to white at the centre.
    // objectBoundingBox radial gradient centred on the quadrant's inner corner, radius sqrt(2):
    // offset t = normalised distance from the centre, opacity t^1.6 (same profile as src/make_figures.py).
    const defs = svg.append("defs");
    const quads = DATA.meta.quadrants.map((q) => ({
      ...q,
      rx: q.x === "left" ? plot.x0 : cx, ry: q.y === "top" ? plot.y0 : cy,
      rw: q.x === "left" ? cx - plot.x0 : plot.x1 - cx, rh: q.y === "top" ? cy - plot.y0 : plot.y1 - cy,
      gcx: q.x === "left" ? 1 : 0, gcy: q.y === "top" ? 1 : 0,
    }));
    quads.forEach((q, i) => {
      const g = defs.append("radialGradient").attr("id", `quad-grad-${i}`)
        .attr("gradientUnits", "objectBoundingBox").attr("cx", q.gcx).attr("cy", q.gcy).attr("r", Math.SQRT2)
        .attr("fx", q.gcx).attr("fy", q.gcy);
      d3.range(0, 1.0001, 0.1).forEach((t) =>
        g.append("stop").attr("offset", t).attr("stop-color", q.tint).attr("stop-opacity", Math.pow(t, 1.6)));
    });
    const gBack = svg.append("g");
    gBack.append("rect").attr("x", plot.x0).attr("y", plot.y0).attr("width", plot.x1 - plot.x0)
      .attr("height", plot.y1 - plot.y0).attr("fill", "#fff");
    quads.forEach((q, i) => gBack.append("rect").attr("x", q.rx).attr("y", q.ry).attr("width", q.rw).attr("height", q.rh)
      .attr("fill", `url(#quad-grad-${i})`));
    gBack.append("line").attr("x1", plot.x0).attr("x2", plot.x1).attr("y1", cy).attr("y2", cy).attr("stroke", "#8a8a8a").attr("stroke-width", 0.9);
    gBack.append("line").attr("x1", cx).attr("x2", cx).attr("y1", plot.y0).attr("y2", plot.y1).attr("stroke", "#8a8a8a").attr("stroke-width", 0.9);
    gBack.append("rect").attr("x", plot.x0 + 0.5).attr("y", plot.y0 + 0.5).attr("width", plot.x1 - plot.x0 - 1)
      .attr("height", plot.y1 - plot.y0 - 1).attr("fill", "none").attr("stroke", "#bbbbbb");

    // Background dots for every fighter
    const pts = F.map((f) => [x(f.axis_1), y(f.axis_2)]);
    const rBg = narrow ? 1.5 : 1.8;
    svg.append("g").selectAll("circle").data(F).join("circle").attr("class", "bg-dot")
      .attr("cx", (_, i) => pts[i][0]).attr("cy", (_, i) => pts[i][1]).attr("r", rBg);

    // Quadrant names in the corners (wrapped onto two lines if a corner is too narrow)
    const gNames = svg.append("g");
    const nameBoxes = [];
    quads.forEach((q) => {
      const pad = narrow ? 8 : 12;
      const tx = q.x === "left" ? plot.x0 + pad : plot.x1 - pad;
      const t = gNames.append("text").attr("class", "quadrant-name").attr("fill", q.name_color)
        .attr("font-size", fs.name).attr("text-anchor", q.x === "left" ? "start" : "end");
      const maxW = q.rw - 2 * pad;
      const words = q.name.split(" ");
      let lines = [q.name];
      t.text(q.name);
      if (t.node().getComputedTextLength() > maxW && words.length > 1) {
        const mid = Math.ceil(words.length / 2);
        lines = [words.slice(0, mid).join(" "), words.slice(mid).join(" ")];
      }
      t.text(null);
      const lh = fs.name * 1.15;
      const baseY = q.y === "top" ? plot.y0 + pad + fs.name * 0.9 : plot.y1 - pad - (lines.length - 1) * lh;
      lines.forEach((ln, k) => t.append("tspan").attr("x", tx).attr("y", baseY + k * lh).text(ln));
      const b = t.node().getBBox();
      nameBoxes.push(expand({ x0: b.x, y0: b.y, x1: b.x + b.width, y1: b.y + b.height }, 4, 3));
    });

    // Plain-language axis ends
    const gEnds = svg.append("g").attr("class", "axis-ends");
    const endText = (tx, ty, anchor, text) => gEnds.append("text").attr("class", "axis-end").attr("x", tx).attr("y", ty)
      .attr("text-anchor", anchor).attr("font-size", fs.end).text(text);
    endText(cx, plot.y0 - 8, "middle", "↑ More clinch and body work");
    endText(cx, plot.y1 + fs.end + 4, "middle", "↓ More head strikes at range");
    endText(plot.x0, plot.y1 + 2 * fs.end + 14, "start", "← More striking volume");
    endText(plot.x1, plot.y1 + 2 * fs.end + 14, "end", "More takedowns and control →");

    // Average-fighter marker
    const d = narrow ? 5.5 : 6.5;
    svg.append("path").attr("d", `M${cx},${cy - d} L${cx + d},${cy} L${cx},${cy + d} L${cx - d},${cy} Z`)
      .attr("fill", "#fff").attr("stroke", "#222").attr("stroke-width", 1.4);

    // Labelled fighters
    const labelled = DATA.meta.labelled.map((n) => F.find((f) => f.name === n)).filter(Boolean);
    const gLab = svg.append("g");
    const gLabels = svg.append("g");
    const labPts = labelled.map((f) => [x(f.axis_1), y(f.axis_2)]);

    const gSel = svg.append("g");
    const gHover = svg.append("g");
    state = { width, height, narrow, x, y, plot, pts, labelled, labPts, nameBoxes, cx, cy, fs, svg,
              gLabels, gSel, gHover, placed: [], segments: [], delaunay: d3.Delaunay.from(pts) };

    // Most crowded labels first (as in src/make_figures.py). On narrow screens a label that can't satisfy every
    // rule is left out (its fighter is drawn as an ordinary dot and can still be found by search); on wider
    // screens the best available position is used.
    const crowd = (i) => labPts.filter((p, j) => j !== i && Math.hypot(p[0] - labPts[i][0], p[1] - labPts[i][1]) < 120).length;
    const priority = DATA.meta.label_priority || [];
    const rank = (i) => { const k = priority.indexOf(labelled[i].name); return k < 0 ? priority.length : k; };
    const order = labelled.map((_, i) => i).sort((a, b) => rank(a) - rank(b) || crowd(b) - crowd(a));
    const shown = [];
    order.forEach((i) => {
      const required = priority.includes(labelled[i].name);
      if (placeLabel(labelled[i].name, labPts[i], false, !narrow || required)) shown.push(i);
    });
    state.shownLabelled = new Set(shown.map((i) => labelled[i].name));
    gLab.selectAll("circle").data(shown).join("circle").attr("class", "lab-dot")
      .attr("fill", DATA.meta.labelled_dot_color).attr("cx", (i) => labPts[i][0]).attr("cy", (i) => labPts[i][1])
      .attr("r", narrow ? 4 : 4.5);
    if (selectedId) showSelected(byId.get(selectedId), false);

    // Hover and tap: nearest fighter within a radius (larger for touch)
    svg.on("pointermove", (ev) => {
      if (ev.pointerType === "touch" || stickyTooltip) return;
      const [mx, my] = d3.pointer(ev);
      const i = nearest(mx, my, 18);
      if (i === null) hideTooltip(); else showHover(F[i], pts[i]);
    });
    svg.on("pointerleave", (ev) => { if (ev.pointerType !== "touch" && !stickyTooltip) hideTooltip(); });
    svg.on("click", (ev) => {
      const [mx, my] = d3.pointer(ev);
      const i = nearest(mx, my, ev.pointerType === "touch" || state.narrow ? 26 : 18);
      if (i === null) { stickyTooltip = false; hideTooltip(); if (selectedId) showSelected(byId.get(selectedId), true); return; }
      stickyTooltip = true;
      showHover(F[i], pts[i]);
    });
  }

  function nearest(mx, my, radius) {
    const i = state.delaunay.find(mx, my);
    if (i === undefined || i < 0) return null;
    const [px, py] = state.pts[i];
    return Math.hypot(px - mx, py - my) <= radius ? i : null;
  }

  // ------------------------------------------------------------------ label placement
  // Same rules as src/make_figures.py: inside the plot, no overlaps with labels or quadrant names, no covering a
  // labelled point or the average marker, nearest to its own point (15% margin), off the split lines when possible.
  function placeLabel(text, [px, py], selected, allowFallback = true) {
    const s = state;
    const t = s.gLabels.append("text").attr("class", "fighter-label" + (selected ? " selected" : "")).text(text);
    const others = s.labPts.filter(([ax, ay]) => ax !== px || ay !== py);
    const avoidPts = s.labPts.concat([[s.cx, s.cy]]);
    let best = null, fallback = null;
    const radii = s.narrow ? [10, 15, 22, 30, 40, 52] : [12, 18, 26, 36, 48, 62];
    outer:
    for (const avoidSplit of [true, false]) {
      for (const r of radii) {
        for (let a = 0; a < 360; a += 15) {
          const dx = r * Math.cos(a * Math.PI / 180), dy = -r * Math.sin(a * Math.PI / 180);
          const anchor = dx > 3 ? "start" : dx < -3 ? "end" : "middle";
          t.attr("x", px + dx).attr("y", py + dy).attr("text-anchor", anchor)
            .attr("dominant-baseline", dy < -3 ? "auto" : dy > 3 ? "hanging" : "middle");
          const bb = t.node().getBBox();
          const b = expand({ x0: bb.x, y0: bb.y, x1: bb.x + bb.width, y1: bb.y + bb.height }, 2, 1.5);
          const p = s.plot;
          let penalty = 0;
          if (b.x0 < p.x0 || b.x1 > p.x1 || b.y0 < p.y0 || b.y1 > p.y1) continue;
          if (s.placed.some((o) => overlaps(b, o)) || s.nameBoxes.some((o) => overlaps(b, o))) penalty += 100;
          if (avoidPts.some(([ax, ay]) => rectDist(b, ax, ay) < 6)) penalty += 50;
          const own = rectDist(b, px, py);
          const other = Math.min(...others.map(([ax, ay]) => rectDist(b, ax, ay)), Infinity);
          if (other < 1.15 * own + 4) penalty += 30;
          const seg = leaderEnds([px, py], b);
          if (seg && (s.placed.some((o) => segHits(seg, o)) || s.nameBoxes.some((o) => segHits(seg, o)))) penalty += 20;
          if (s.segments.some((sg) => segHits(sg, b))) penalty += 20;
          const crossesSplit = (b.x0 < s.cx && s.cx < b.x1) || (b.y0 < s.cy && s.cy < b.y1);
          if (avoidSplit && crossesSplit) continue;
          const cand = { dx, dy, anchor, b, seg, penalty: penalty + r * 0.1 };
          if (penalty === 0) { best = cand; break outer; }
          if (!fallback || cand.penalty < fallback.penalty) fallback = cand;
        }
      }
    }
    const c = best || (allowFallback ? fallback : null);
    if (!c) { t.remove(); return null; }
    t.attr("x", px + c.dx).attr("y", py + c.dy).attr("text-anchor", c.anchor)
      .attr("dominant-baseline", c.dy < -3 ? "auto" : c.dy > 3 ? "hanging" : "middle");
    s.placed.push(c.b);
    if (c.seg) {
      s.segments.push(c.seg);
      s.gLabels.insert("line", ":first-child").attr("class", "leader")
        .attr("x1", c.seg[0][0]).attr("y1", c.seg[0][1]).attr("x2", c.seg[1][0]).attr("y2", c.seg[1][1]);
    }
    return t;
  }

  const expand = (b, px, py) => ({ x0: b.x0 - px, y0: b.y0 - py, x1: b.x1 + px, y1: b.y1 + py });
  const overlaps = (a, b) => a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1;
  function rectDist(b, x, y) {
    const dx = Math.max(b.x0 - x, 0, x - b.x1), dy = Math.max(b.y0 - y, 0, y - b.y1);
    return Math.hypot(dx, dy);
  }
  function leaderEnds([x0, y0], b, gap = 5) {
    const ex = Math.min(Math.max(x0, b.x0), b.x1), ey = Math.min(Math.max(y0, b.y0), b.y1);
    const d = Math.hypot(ex - x0, ey - y0);
    if (d <= gap) return null;
    return [[x0 + (ex - x0) * gap / d, y0 + (ey - y0) * gap / d], [ex, ey]];
  }
  function segHits(seg, b) {
    if (!seg) return false;
    const [[x1, y1], [x2, y2]] = seg;
    for (let k = 0; k <= 24; k++) {
      const t = (k / 24) * 0.96, xx = x1 + t * (x2 - x1), yy = y1 + t * (y2 - y1);
      if (b.x0 < xx && xx < b.x1 && b.y0 < yy && yy < b.y1) return true;
    }
    return false;
  }

  // ------------------------------------------------------------------ tooltip
  function tooltipHtml(f) {
    const q = DATA.quad[f.quadrant];
    const years = f.first_year === f.last_year ? `${f.first_year}` : `${f.first_year}–${f.last_year}`;
    return `<div class="name">${escapeHtml(f.name)}</div>` +
      `<div>${escapeHtml(f.division || "—")}</div>` +
      `<div>${f.fights} UFC fights since 2000 (${years})</div>` +
      `<div class="q"><span class="swatch" style="background:${q.tint}"></span>${escapeHtml(f.quadrant)}</div>`;
  }

  function showHover(f, [px, py], avoid) {
    state.gHover.selectAll("*").remove();
    state.gHover.append("circle").attr("class", "hover-dot").attr("cx", px).attr("cy", py).attr("r", state.narrow ? 6 : 6.5);
    tooltipEl.innerHTML = tooltipHtml(f);
    tooltipEl.hidden = false;
    positionTooltip(px, py, avoid);
  }

  // Place the tooltip beside the dot: try the four diagonal positions and use the first that fits in the chart
  // and doesn't cover `avoid` (a label box in SVG coordinates, e.g. the searched fighter's label).
  function positionTooltip(px, py, avoid) {
    const svgBox = state.svg.node().getBoundingClientRect();
    const scale = svgBox.width / state.width;
    const sx = px * scale, sy = py * scale;
    const tw = tooltipEl.offsetWidth, th = tooltipEl.offsetHeight, W = svgBox.width, H = svgBox.height;
    const a = avoid && { x0: avoid.x0 * scale, y0: avoid.y0 * scale, x1: avoid.x1 * scale, y1: avoid.y1 * scale };
    const cands = [[sx + 14, sy - th - 12], [sx - tw - 14, sy - th - 12], [sx + 14, sy + 16], [sx - tw - 14, sy + 16]];
    const fits = ([l, t]) => l >= 4 && t >= 4 && l + tw <= W - 4 && t + th <= H - 4;
    const clear = ([l, t]) => !a || !overlaps({ x0: l, y0: t, x1: l + tw, y1: t + th }, a);
    const pick = cands.find((c) => fits(c) && clear(c)) || cands.find(fits) ||
      [Math.max(4, Math.min(W - tw - 4, sx - tw / 2)), Math.min(H - th - 4, sy + 16)];
    tooltipEl.style.left = `${pick[0]}px`;
    tooltipEl.style.top = `${pick[1]}px`;
  }

  function hideTooltip(force) {
    if (state && state.gHover) state.gHover.selectAll("*").remove();
    tooltipEl.hidden = true;
    if (force) stickyTooltip = false;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  // ------------------------------------------------------------------ search
  function showSelected(f, withTooltip) {
    const s = state;
    s.gSel.selectAll("*").remove();
    s.gLabels.selectAll(".fighter-label.selected").remove();
    if (!f) return;
    const px = s.x(f.axis_1), py = s.y(f.axis_2);
    s.gSel.append("circle").attr("class", "selected-ring").attr("cx", px).attr("cy", py).attr("r", 9);
    s.gSel.append("circle").attr("fill", DATA.meta.labelled_dot_color).attr("stroke", "#fff").attr("cx", px).attr("cy", py).attr("r", 4.5);
    const alreadyLabelled = s.shownLabelled.has(f.name);
    const label = alreadyLabelled ? null : placeLabel(f.name, [px, py], true);
    let avoid = null;
    if (label) {
      const bb = label.node().getBBox();
      avoid = { x0: bb.x, y0: bb.y, x1: bb.x + bb.width, y1: bb.y + bb.height };
    }
    if (withTooltip) { stickyTooltip = true; showHover(f, [px, py], avoid); }
  }

  function setupSearch() {
    let matches = [], active = -1;

    function update() {
      const q = normalize(input.value.trim());
      clearBtn.hidden = !input.value;
      if (!q) { closeList(); return; }
      const starts = [], contains = [];
      for (const f of DATA.fighters) {
        if (f.key.startsWith(q) || f.key.split(/[\s-]+/).some((w) => w.startsWith(q))) starts.push(f);
        else if (f.key.includes(q)) contains.push(f);
      }
      matches = starts.concat(contains).slice(0, 8);
      active = matches.length ? 0 : -1;
      list.innerHTML = matches.length
        ? matches.map((f, i) => `<li role="option" id="opt-${i}" data-i="${i}" aria-selected="${i === active}">` +
            `<span>${escapeHtml(f.name)}</span><span class="div">${escapeHtml(f.division || "")}</span></li>`).join("")
        : `<li class="empty" role="option" aria-disabled="true">No fighter with 5+ UFC fights since 2000 matches</li>`;
      list.hidden = false;
      input.setAttribute("aria-expanded", "true");
      input.setAttribute("aria-activedescendant", active >= 0 ? `opt-${active}` : "");
    }

    function closeList() {
      list.hidden = true; list.innerHTML = ""; matches = []; active = -1;
      input.setAttribute("aria-expanded", "false");
      input.removeAttribute("aria-activedescendant");
    }

    function choose(f) {
      selectedId = f.id;
      input.value = f.name;
      clearBtn.hidden = false;
      closeList();
      showSelected(f, true);
      statusEl.textContent = `${f.name}: ${f.division || "no division"}, ${f.fights} UFC fights since 2000, ${f.quadrant}.`;
    }

    function setActive(i) {
      active = i;
      [...list.children].forEach((li, k) => li.setAttribute("aria-selected", String(k === active)));
      if (active >= 0) {
        input.setAttribute("aria-activedescendant", `opt-${active}`);
        list.children[active].scrollIntoView({ block: "nearest" });
      }
    }

    input.addEventListener("input", update);
    input.addEventListener("focus", () => { if (input.value && !selectedId) update(); });
    input.addEventListener("keydown", (e) => {
      if (list.hidden && e.key === "ArrowDown") { update(); return; }
      if (e.key === "ArrowDown" && matches.length) { e.preventDefault(); setActive((active + 1) % matches.length); }
      else if (e.key === "ArrowUp" && matches.length) { e.preventDefault(); setActive((active - 1 + matches.length) % matches.length); }
      else if (e.key === "Enter" && active >= 0 && matches[active]) { e.preventDefault(); choose(matches[active]); }
      else if (e.key === "Escape") { closeList(); }
    });
    list.addEventListener("pointerdown", (e) => {
      const li = e.target.closest("li[data-i]");
      if (li) { e.preventDefault(); choose(matches[+li.dataset.i]); }
    });
    document.addEventListener("pointerdown", (e) => { if (!e.target.closest(".search-box")) closeList(); });
    clearBtn.addEventListener("click", () => {
      input.value = ""; selectedId = null; clearBtn.hidden = true; closeList(); statusEl.textContent = "";
      state.gSel.selectAll("*").remove();
      state.gLabels.selectAll(".fighter-label.selected").remove();
      hideTooltip(true);
      input.focus();
    });
  }
})();
