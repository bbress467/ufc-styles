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


def place_labels(ax, points: dict[str, tuple[float, float]], fontsize: float = LABEL_FONTSIZE) -> dict:
    """Place one label per point with a thin leader line, avoiding ambiguity and overlaps.

    For each label (most crowded points first), candidate positions are tried on rings around the point,
    nearest first. A candidate is accepted only if the label box (a) stays inside the axes, (b) doesn't overlap
    an already placed label, (c) doesn't cover any labelled point, and (d) is closer to its own point than to
    any other labelled point by NEAREST_MARGIN. Raises if a label can't be placed, rather than drawing a bad figure.
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

    placed, out = [], {}
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
                b = t.get_window_extent(renderer).expanded(1.04, 1.12)
                t.remove()
                if not (axes_box.x0 <= b.x0 and b.x1 <= axes_box.x1 and axes_box.y0 <= b.y0 and b.y1 <= axes_box.y1):
                    continue
                if any(b.overlaps(o) for o in placed):
                    continue
                if any(_rect_point_distance(b, *px[m]) < MARKER_RADIUS_PX for m in px):
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


def check_labels(ax, labels: dict, points: dict) -> None:
    """Assert the rendered labels are unambiguous: inside the axes, non-overlapping, nearest to their own point."""
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    px = {n: ax.transData.transform(p) for n, p in points.items()}
    boxes = {n: t.get_window_extent(renderer) for n, (t, _) in labels.items()}
    for n, (_, line) in labels.items():
        assert line is not None, f"label {n} has no leader line"
    names = list(boxes)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not boxes[a].overlaps(boxes[b]), f"labels overlap: {a} / {b}"
        own = _rect_point_distance(boxes[a], *px[a])
        nearest = min(names, key=lambda m: _rect_point_distance(boxes[a], *px[m]))
        assert nearest == a, f"label {a} is nearer to {nearest}'s point than its own"
        assert own < 80, f"label {a} is far from its point ({own:.0f}px)"


def style_map(path: Path) -> None:
    profiles = F.build_profiles(None)
    model_path = A.MODEL_DIR / f"style_axes_m{A.CHOSEN_AXES}.joblib"
    axes = A.StyleAxes.load(model_path) if model_path.exists() else A.fit_style_axes(profiles, A.CHOSEN_AXES)
    s = axes.transform(profiles).set_index("fighter_name")
    points = {n: (s.at[n, "axis_1"], s.at[n, "axis_2"]) for n in STYLE_MAP_FIGHTERS if n in s.index}

    fig, ax = plt.subplots(figsize=(10, 7.8))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.885, bottom=0.165)
    ax.scatter(s["axis_1"], s["axis_2"], s=9, color=GREY, alpha=0.6, lw=0,
               label=f"all {len(s):,} fighters with 5+ UFC fights since 2000")
    xy = np.array(list(points.values()))
    ax.scatter(xy[:, 0], xy[:, 1], s=34, color=BLUE, zorder=3, label="labelled fighters")
    ax.set_xlim(s["axis_1"].min() - 0.3, s["axis_1"].max() + 0.6)
    ax.set_ylim(s["axis_2"].min() - 0.4, s["axis_2"].max() + 0.3)
    ax.axhline(0, color="#555555", lw=0.6, zorder=1)
    ax.axvline(0, color="#555555", lw=0.6, zorder=1)
    ax.set_ylabel("Axis 2: clinch and body work\n(higher = more strikes in the clinch and to the body)")
    ax.set_xlabel("Axis 1: grappler vs. volume striker", labelpad=6)
    # Axis-end descriptions on their own line just below the axis title (measured), then the footnote.
    fig.canvas.draw()
    to_fig = fig.transFigure.inverted()
    label_bottom = to_fig.transform((0, ax.xaxis.label.get_window_extent().y0))[1]
    pos = ax.get_position()
    y_ends = label_bottom - 0.012
    fig.text(pos.x0, y_ends, "← more striking volume", ha="left", va="top", fontsize=10.5, color="#333333")
    fig.text(pos.x1, y_ends, "more takedowns, control and ground work →", ha="right", va="top", fontsize=10.5, color="#333333")
    fig.text(pos.x1, y_ends - 0.045, "Scores in standard deviations, compared with fighters in the same division and era.",
             ha="right", va="top", fontsize=9.5, color="#555555")
    fig.suptitle("UFC style map: grappling vs. striking volume, and clinch/body work", fontsize=14, y=0.985)
    # Legend above the plot area, so it can't cross the zero lines or cover points.
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=2, frameon=False, fontsize=10.5,
              handletextpad=0.3, columnspacing=1.5, borderaxespad=0)
    labels = place_labels(ax, points)
    check_labels(ax, labels, points)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


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
