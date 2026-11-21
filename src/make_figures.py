"""Regenerate the README figures in docs/figures/ (run after the pipeline: features, style_axes, elo, matchups).

    venv/bin/python src/make_figures.py

* style_map.png        fighters on style axis 1 vs axis 2, with well-known fighters labelled   (notebook 06)
* style_evolution.png  pace and distance share by year since 2000, with 95% bands             (notebook 04)
* matchup_results.png  2019+ test log loss, Elo alone vs Elo + style axes, with 95% CIs        (notebook 07)

Figures are drawn for README width: large fonts, plain-language labels, no overlapping labels.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import features as F  # noqa: E402
import matchups as M  # noqa: E402
import style_axes as A  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "figures"
DPI = 150
BLUE, RED, GREY = "#2F6DB3", "#C44E52", "#B8B8B8"

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": DPI, "savefig.dpi": DPI,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})

# Fighters labelled on the style map. Label positions are chosen automatically by place_labels().
STYLE_MAP_FIGHTERS = [
    "Khabib Nurmagomedov", "Jailton Almeida", "Merab Dvalishvili", "Demian Maia", "Georges St-Pierre", "Amanda Nunes",
    "Kamaru Usman", "Valentina Shevchenko", "Zhang Weili", "Alexander Volkanovski", "Charles Oliveira",
    "Demetrious Johnson", "Jon Jones", "Anderson Silva", "Nate Diaz", "Joanna Jedrzejczyk", "Conor McGregor",
    "Francis Ngannou", "Jose Aldo", "Israel Adesanya", "Max Holloway", "Sean Strickland",
]
LABEL_FONTSIZE = 10.5
MARKER_RADIUS_PX = 6          # labelled-point marker radius used for collision checks (display pixels, incl. margin)
NEAREST_MARGIN = 1.15         # a label's own point must be this much closer than any other labelled point


def _rect_point_distance(b, x, y) -> float:
    """Distance in display pixels from point (x, y) to bounding box b (0 if inside)."""
    dx = max(b.x0 - x, 0, x - b.x1)
    dy = max(b.y0 - y, 0, y - b.y1)
    return float(np.hypot(dx, dy))


def _leader_endpoints(point_px, box, gap_px: float = 5.0):
    """Display-pixel endpoints of the leader from just outside the point to the nearest edge of the box (or None)."""
    x0, y0 = point_px
    ex, ey = min(max(x0, box.x0), box.x1), min(max(y0, box.y0), box.y1)
    d = np.hypot(ex - x0, ey - y0)
    if d <= gap_px:
        return None
    return (x0 + (ex - x0) * gap_px / d, y0 + (ey - y0) * gap_px / d), (ex, ey)


def _segment_hits_box(seg, box, n: int = 25) -> bool:
    """True if the segment passes through the box (sampled; endpoints on the box edge don't count)."""
    if seg is None:
        return False
    (x1, y1), (x2, y2) = seg
    for t in np.linspace(0.0, 0.96, n):                    # stop just short of the label end
        x, y = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
        if box.x0 < x < box.x1 and box.y0 < y < box.y1:
            return True
    return False


def place_labels(ax, points: dict[str, tuple[float, float]], fontsize: float = LABEL_FONTSIZE,
                 obstacles: list | None = None, keep_clear: list[tuple[float, float]] | None = None) -> dict:
    """Place one label per point with a thin leader line, avoiding ambiguity and overlaps.

    For each label (most crowded points first), candidate positions are tried on rings around the point,
    nearest first. A candidate is accepted only if the label box (a) stays inside the axes, (b) doesn't overlap
    an already placed label or any `obstacles` box (e.g. quadrant names), (c) doesn't cover any labelled point
    or `keep_clear` point (e.g. the average-fighter marker), and (d) is closer to its own point than to any other
    labelled point by NEAREST_MARGIN. Raises if a label can't be placed, rather than drawing a bad figure.
    Returns {name: (text, leader line)} for checks.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_px = ax.transData.transform
    px = {n: to_px(p) for n, p in points.items()}
    axes_box = ax.get_window_extent(renderer)

    def crowding(n):
        return sum(np.hypot(*(px[n] - px[m])) < 120 for m in px if m != n)

    zx, zy = to_px((0.0, 0.0))                            # zero lines, in display pixels

    def crosses_zero(b):
        return b.x0 < zx < b.x1 or b.y0 < zy < b.y1

    clear_px = [to_px(p) for p in (keep_clear or [])]
    placed, out, segments = list(obstacles or []), {}, []
    for name in sorted(points, key=crowding, reverse=True):
        x0, y0 = px[name]
        best = None
        # Pass 1: nearby positions that don't sit on a zero line. Pass 2: any valid position.
        tries = [(r, True) for r in (12, 18, 26)] + [(r, False) for r in (12, 18, 26, 36, 48, 62)]
        for radius, avoid_zero in tries:
            for ang in np.deg2rad(np.arange(0, 360, 15)):
                dx, dy = radius * np.cos(ang), radius * np.sin(ang)
                ha = "left" if dx > 3 else "right" if dx < -3 else "center"
                va = "bottom" if dy > 3 else "top" if dy < -3 else "center"
                t = ax.annotate(name, points[name], xytext=(dx, dy), textcoords="offset points",
                                ha=ha, va=va, fontsize=fontsize, zorder=5)
                b = t.get_window_extent(renderer).expanded(1.06, 1.45)   # small gap around every label
                t.remove()
                if not (axes_box.x0 <= b.x0 and b.x1 <= axes_box.x1 and axes_box.y0 <= b.y0 and b.y1 <= axes_box.y1):
                    continue
                if any(b.overlaps(o) for o in placed):
                    continue
                seg = _leader_endpoints((x0, y0), b)
                if any(_segment_hits_box(seg, o) for o in placed):          # my leader through another label
                    continue
                if any(_segment_hits_box(sg, b) for sg in segments):       # another leader through my label
                    continue
                if any(_rect_point_distance(b, *px[m]) < MARKER_RADIUS_PX for m in px):
                    continue
                if any(_rect_point_distance(b, *c) < MARKER_RADIUS_PX for c in clear_px):
                    continue
                own = _rect_point_distance(b, x0, y0)
                other = min(_rect_point_distance(b, *px[m]) for m in px if m != name)
                if other < NEAREST_MARGIN * own + 4:
                    continue
                if avoid_zero and crosses_zero(b):
                    continue
                best = (dx, dy, ha, va, b)            # offsets are in points, as drawn
                break
            if best:
                break
        if best is None:
            raise RuntimeError(f"could not place label for {name} without ambiguity or overlap")
        dxp, dyp, ha, va, b = best
        t = ax.annotate(name, points[name], xytext=(dxp, dyp), textcoords="offset points", ha=ha, va=va,
                        fontsize=fontsize, zorder=5)
        placed.append(b)
        segments.append(_leader_endpoints((x0, y0), t.get_window_extent(renderer)))
        out[name] = (t, _leader_line(ax, (x0, y0), t.get_window_extent(renderer)))
    return out


def _leader_line(ax, point_px, box, gap_px: float = 5.0):
    """Thin line from just outside the point to the nearest edge of the label box (drawn in data coordinates)."""
    x0, y0 = point_px
    ex, ey = min(max(x0, box.x0), box.x1), min(max(y0, box.y0), box.y1)     # nearest point on the box
    d = np.hypot(ex - x0, ey - y0)
    if d <= gap_px:
        return None
    sx, sy = x0 + (ex - x0) * gap_px / d, y0 + (ey - y0) * gap_px / d
    inv = ax.transData.inverted()
    (ax_, ay_), (bx_, by_) = inv.transform([(sx, sy), (ex, ey)])
    line, = ax.plot([ax_, bx_], [ay_, by_], color="#777777", lw=0.7, zorder=4, solid_capstyle="butt")
    return line


def check_labels(ax, labels: dict, points: dict, obstacles: dict | None = None,
                 keep_clear: list[tuple[float, float]] | None = None) -> None:
    """Assert the rendered labels are unambiguous: non-overlapping (with each other and with any `obstacles`,
    e.g. quadrant names: {name: artist}), not covering `keep_clear` points, and nearest to their own point."""
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    px = {n: ax.transData.transform(p) for n, p in points.items()}
    boxes = {n: t.get_window_extent(renderer) for n, (t, _) in labels.items()}
    for n, (_, line) in labels.items():
        assert line is not None, f"label {n} has no leader line"
    obstacle_boxes = {k: o.get_window_extent(renderer) for k, o in (obstacles or {}).items()}
    for n, b in boxes.items():
        for k, ob in obstacle_boxes.items():
            assert not b.overlaps(ob), f"label {n} overlaps {k!r}"
        for c in keep_clear or []:
            assert _rect_point_distance(b, *ax.transData.transform(c)) >= MARKER_RADIUS_PX / 2, f"label {n} covers {c}"
    names = list(boxes)
    segs = {n: _leader_endpoints(px[n], boxes[n]) for n in names}
    for n in names:
        for m in names:
            if m != n:
                assert not _segment_hits_box(segs[n], boxes[m]), f"leader line of {n} passes through label {m}"
        for k, ob in obstacle_boxes.items():
            assert not _segment_hits_box(segs[n], ob), f"leader line of {n} passes through {k!r}"
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not boxes[a].overlaps(boxes[b]), f"labels overlap: {a} / {b}"
        own = _rect_point_distance(boxes[a], *px[a])
        nearest = min(names, key=lambda m: _rect_point_distance(boxes[a], *px[m]))
        assert nearest == a, f"label {a} is nearer to {nearest}'s point than its own"
        assert own < 80, f"label {a} is far from its point ({own:.0f}px)"


QUADRANT_NAMES = {   # (x side, y side): name
    ("left", "top"): "Clinch and body strikers", ("right", "top"): "Clinch grapplers",
    ("left", "bottom"): "Range strikers", ("right", "bottom"): "Wrestlers and ground grapplers",
}
SHADE_RGB = (0.42, 0.42, 0.42)     # one neutral gray for all four quadrants
SHADE_MAX_ALPHA = 0.30             # opacity in the outer corners, fading to ~0 at the centre
NOTE = ("Styles are a continuum: fighters near the center have no strong lean either way.\n"
        "Positions are relative to fighters in the same division and era.")


def _quadrant_shading(ax, xlim, ylim, n: int = 600) -> None:
    """Gray shading per quadrant: darkest in each outer corner, fading to nearly nothing toward the centre."""
    xs, ys = np.linspace(*xlim, n), np.linspace(*ylim, n)
    X, Y = np.meshgrid(xs, ys)
    u = np.where(X >= 0, X / xlim[1], X / xlim[0])          # 0 at the split, 1 at the outer edge of each quadrant
    v = np.where(Y >= 0, Y / ylim[1], Y / ylim[0])
    r = np.clip(np.hypot(u, v) / np.sqrt(2), 0, 1)          # 1 in the outer corner
    rgba = np.zeros((n, n, 4))
    rgba[..., :3] = SHADE_RGB
    rgba[..., 3] = SHADE_MAX_ALPHA * r ** 1.6
    ax.imshow(rgba, extent=(*xlim, *ylim), origin="lower", aspect="auto", interpolation="bilinear", zorder=0)


def style_map(path: Path) -> None:
    profiles = F.build_profiles(None)
    model_path = A.MODEL_DIR / f"style_axes_m{A.CHOSEN_AXES}.joblib"
    axes = A.StyleAxes.load(model_path) if model_path.exists() else A.fit_style_axes(profiles, A.CHOSEN_AXES)
    s = axes.transform(profiles).set_index("fighter_name")
    points = {n: (s.at[n, "axis_1"], s.at[n, "axis_2"]) for n in STYLE_MAP_FIGHTERS if n in s.index}

    fig, ax = plt.subplots(figsize=(10, 8.4))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.89, bottom=0.2)
    # Extra room past the data so the quadrant names sit in clear corners.
    xlim = (s["axis_1"].min() - 0.6, s["axis_1"].max() + 1.0)
    ylim = (s["axis_2"].min() - 0.75, s["axis_2"].max() + 0.7)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    _quadrant_shading(ax, xlim, ylim)
    ax.axhline(0, color="#8a8a8a", lw=0.9, zorder=1)
    ax.axvline(0, color="#8a8a8a", lw=0.9, zorder=1)
    bg = ax.scatter(s["axis_1"], s["axis_2"], s=9, color="#7f7f7f", alpha=0.75, lw=0, zorder=2)  # visible over shading
    xy = np.array(list(points.values()))
    lab = ax.scatter(xy[:, 0], xy[:, 1], s=36, color=BLUE, edgecolor="white", lw=0.6, zorder=3)
    avg = ax.scatter([0], [0], marker="D", s=46, facecolor="white", edgecolor="#222222", lw=1.3, zorder=4)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("#bbbbbb")
        sp.set_linewidth(0.8)

    # Quadrant names in the corners: larger than fighter labels, darker gray.
    names = {}
    for (xs_, ys_), text in QUADRANT_NAMES.items():
        x = 0.015 if xs_ == "left" else 0.985
        y = 0.975 if ys_ == "top" else 0.025
        names[text] = ax.text(x, y, text, transform=ax.transAxes, ha=xs_, va=ys_, fontsize=14, fontweight="bold",
                              color="#4a4a4a", zorder=6)

    # Plain-language axis ends: horizontal below the plot's corners, vertical above/below the vertical split.
    pos = ax.get_position()
    x_zero = pos.x0 + (0 - xlim[0]) / (xlim[1] - xlim[0]) * (pos.x1 - pos.x0)
    style = dict(fontsize=11.5, color="#222222")
    ends = {
        "left": fig.text(pos.x0, pos.y0 - 0.012, "← More striking volume", ha="left", va="top", **style),
        "right": fig.text(pos.x1, pos.y0 - 0.012, "More takedowns and control →", ha="right", va="top", **style),
        "bottom": fig.text(x_zero, pos.y0 - 0.012, "More head strikes at range", ha="center", va="top", **style),
        "top": fig.text(x_zero, pos.y1 + 0.008, "More clinch and body work", ha="center", va="bottom", **style),
    }
    leg = fig.legend([bg, lab, avg], [f"all {len(s):,} fighters with 5+ UFC fights since 2000", "labelled fighters",
                                      "average fighter"],
                     loc="upper left", bbox_to_anchor=(pos.x0, pos.y0 - 0.055), ncol=3, frameon=False, fontsize=10.5,
                     handletextpad=0.3, columnspacing=1.6, markerscale=1.0)
    note = fig.text(pos.x0, pos.y0 - 0.105, NOTE, ha="left", va="top", fontsize=10.5, color="#555555", linespacing=1.4)
    fig.suptitle("UFC style map: four broad styles", fontsize=15, y=0.975)

    fig.canvas.draw()
    name_boxes = [o.get_window_extent(fig.canvas.get_renderer()).expanded(1.06, 1.3) for o in names.values()]
    labels = place_labels(ax, points, obstacles=name_boxes, keep_clear=[(0.0, 0.0)])
    check_labels(ax, labels, points, obstacles=names, keep_clear=[(0.0, 0.0)])
    _check_no_overlap({**{f"axis end ({k})": t for k, t in ends.items()}, "legend": leg, "note": note,
                       **{f"quadrant name {k!r}": t for k, t in names.items()}}, fig)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def _check_no_overlap(artists: dict, fig) -> None:
    """Assert that none of the given text/legend artists overlap each other (figure furniture check)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = {k: a.get_window_extent(r) for k, a in artists.items()}
    keys = list(boxes)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            assert not boxes[a].overlaps(boxes[b]), f"{a} overlaps {b}"


def style_evolution(path: Path, n_boot: int = 1000, seed: int = 0, last_year: int = 2025) -> None:
    ff = F.load_fighter_fights()
    ff["year"] = ff["event_date"].str[:4].astype(int)
    ff = ff[ff["year"] <= last_year]                                     # 2026 is a partial year
    fights = ff.groupby(["year", "fight_id"])[["sig_att", "distance_att", "time_s"]].sum().reset_index()
    rng = np.random.default_rng(seed)
    rows = []
    for year, g in fights.groupby("year"):
        W = rng.multinomial(len(g), np.full(len(g), 1 / len(g)), size=n_boot)
        pace = 60 * g["sig_att"].sum() / g["time_s"].sum()
        pace_b = 60 * (W @ g["sig_att"].to_numpy()) / (W @ g["time_s"].to_numpy())
        dist = g["distance_att"].sum() / g["sig_att"].sum()
        dist_b = (W @ g["distance_att"].to_numpy()) / (W @ g["sig_att"].to_numpy())
        rows.append((year, pace, *np.quantile(pace_b, [.025, .975]), dist, *np.quantile(dist_b, [.025, .975])))
    t = np.array(rows)
    years = t[:, 0]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    a1.fill_between(years, t[:, 2], t[:, 3], color=BLUE, alpha=0.2, lw=0, label="95% confidence band")
    a1.plot(years, t[:, 1], color=BLUE, marker="o", ms=4, label="all UFC fights")
    a1.set_title("Pace more than doubled")
    a1.set_ylabel("Significant strikes attempted\nper fighter per minute")
    a1.set_ylim(0, None)
    a1.legend(frameon=False, loc="lower right", fontsize=10.5)
    a2.fill_between(years, 100 * t[:, 5], 100 * t[:, 6], color=BLUE, alpha=0.2, lw=0)
    a2.plot(years, 100 * t[:, 4], color=BLUE, marker="o", ms=4)
    a2.set_title("Fights moved to range")
    a2.set_ylabel("Share of significant strikes\nthrown at distance (%)")
    a2.set_ylim(0, 100)
    for a in (a1, a2):
        a.set_xlabel("Year")
        a.set_xticks(range(2000, last_year + 1, 5))
        a.grid(axis="y", color="#e5e5e5", lw=0.8)
    fig.suptitle(f"How UFC striking changed, 2000–{last_year}", fontsize=15, y=1.02)
    fig.text(0.5, -0.04, "Pooled over all UFC fights each year. Bands: bootstrap over fights. Early years have few fights (about 40 a year before 2005).",
             ha="center", fontsize=9.5, color="#555555")
    fig.savefig(path)
    plt.close(fig)


def matchup_results(path: Path, min_prior: int = M.MIN_PRIOR) -> None:
    r = M.evaluate(M.build_dataset(), min_prior, n_boot=2000)
    met, d = r["metrics"], r["vs_elo"].loc["elo+style"]
    rows = [("Elo rating alone", "elo", BLUE), ("Elo + the four style axes", "elo+style", RED)]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
    for i, (label, key, col) in enumerate(rows):
        m = met.loc[key]
        a1.errorbar(m["log_loss"], i, xerr=[[m["log_loss"] - m["ll_lo"]], [m["ll_hi"] - m["log_loss"]]],
                    fmt="o", color=col, ms=8, capsize=5, lw=2)
        a1.annotate(f"{m['log_loss']:.4f}", (m["log_loss"], i), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=10.5)
    a1.axvline(np.log(2), color="#777777", ls="--", lw=1)
    a1.text(np.log(2), 1.45, "coin flip ", ha="right", va="center", fontsize=10, color="#555555")
    a1.set_yticks([0, 1], [r_[0] for r_ in rows])
    a1.set_ylim(-0.6, 1.7)
    a1.invert_yaxis()
    a1.set_xlim(0.655, 0.70)
    a1.set_xticks(np.arange(0.66, 0.7001, 0.01))
    a1.set_xlabel("Test log loss on 2019–2026 fights (lower = better)")
    a1.set_title("Prediction error, with 95% CIs")

    a2.errorbar(d["d_log_loss"], 0, xerr=[[d["d_log_loss"] - d["dll_lo"]], [d["dll_hi"] - d["d_log_loss"]]],
                fmt="o", color=RED, ms=8, capsize=5, lw=2)
    a2.axvline(0, color="#333333", lw=1)
    a2.set_yticks([])
    a2.set_ylim(-1, 1)
    lim = 1.25 * max(abs(d["dll_lo"]), abs(d["dll_hi"]))
    a2.set_xlim(-lim, lim)
    a2.text(-lim * 0.95, 0.6, "← style helps", fontsize=10.5, color="#555555")
    a2.text(lim * 0.95, 0.6, "style hurts →", fontsize=10.5, color="#555555", ha="right")
    a2.annotate(f"{d['d_log_loss']:+.4f}  (95% CI {d['dll_lo']:+.4f} to {d['dll_hi']:+.4f})", (d["d_log_loss"], 0),
                xytext=(0, -28), textcoords="offset points", ha="center", fontsize=10.5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))
    a2.set_xlabel("Change in log loss from adding style (negative = better)")
    verdict = ("no measurable gain" if d["dll_lo"] < 0 < d["dll_hi"] else
               "a measurable gain" if d["dll_hi"] < 0 else "worse predictions")
    a2.set_title(f"Adding style: {verdict}")
    fig.suptitle(f"Do style axes predict winners beyond Elo? ({r['n_test']:,} test fights)", fontsize=15, y=1.03)
    fig.text(0.5, -0.06, f"Models trained on 2006–2018 fights; both fighters had {min_prior}+ prior UFC fights. "
             "The interval crossing 0 means no detectable improvement.", ha="center", fontsize=9.5, color="#555555")
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, fn in (("style_map.png", style_map), ("style_evolution.png", style_evolution), ("matchup_results.png", matchup_results)):
        fn(OUT_DIR / name)
        print(f"wrote {(OUT_DIR / name).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
