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

# Fighters labelled on the style map: label offset in points (dx, dy) and horizontal alignment, chosen so no labels overlap.
STYLE_MAP_LABELS = {
    "Khabib Nurmagomedov": (-4, -17, "right"), "Jailton Almeida": (0, -17, "center"), "Merab Dvalishvili": (0, 9, "center"),
    "Demian Maia": (8, -4, "left"), "Georges St-Pierre": (8, -12, "left"), "Amanda Nunes": (8, -4, "left"),
    "Kamaru Usman": (8, 2, "left"), "Valentina Shevchenko": (0, 9, "center"), "Zhang Weili": (8, 4, "left"),
    "Alexander Volkanovski": (-6, -15, "right"), "Charles Oliveira": (8, 2, "left"), "Demetrious Johnson": (8, 2, "left"),
    "Jon Jones": (8, -2, "left"), "Anderson Silva": (8, 4, "left"), "Nate Diaz": (8, 4, "left"),
    "Joanna Jedrzejczyk": (-6, 6, "right"), "Conor McGregor": (8, 4, "left"), "Francis Ngannou": (8, -12, "left"),
    "Jose Aldo": (8, -4, "left"), "Israel Adesanya": (8, -4, "left"), "Max Holloway": (8, -13, "left"),
    "Sean Strickland": (8, -4, "left"),
}


def style_map(path: Path) -> None:
    profiles = F.build_profiles(None)
    model_path = A.MODEL_DIR / f"style_axes_m{A.CHOSEN_AXES}.joblib"
    axes = A.StyleAxes.load(model_path) if model_path.exists() else A.fit_style_axes(profiles, A.CHOSEN_AXES)
    s = axes.transform(profiles).set_index("fighter_name")

    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.scatter(s["axis_1"], s["axis_2"], s=9, color=GREY, alpha=0.6, lw=0, label=f"all {len(s):,} fighters with 5+ UFC fights since 2000")
    for name, (dx, dy, ha) in STYLE_MAP_LABELS.items():
        if name not in s.index:
            continue
        x, y = s.at[name, "axis_1"], s.at[name, "axis_2"]
        ax.scatter([x], [y], s=34, color=BLUE, zorder=3)
        ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=10.5, ha=ha, zorder=4)
    ax.axhline(0, color="#555555", lw=0.6)
    ax.axvline(0, color="#555555", lw=0.6)
    ax.set_xlabel("Axis 1: grappler vs. volume striker", labelpad=8)
    ax.text(0.0, -0.105, "← more striking volume", transform=ax.transAxes, ha="left", fontsize=10.5, color="#333333")
    ax.text(1.0, -0.105, "more takedowns, control and ground work →", transform=ax.transAxes, ha="right", fontsize=10.5, color="#333333")
    ax.set_ylabel("Axis 2: clinch and body work\n(higher = more strikes in the clinch and to the body)")
    ax.set_title("UFC style map: grappling vs. striking volume, and clinch/body work")
    ax.legend(loc="upper left", frameon=False, fontsize=10.5)
    ax.text(1.0, -0.155, "Scores in standard deviations, compared with fighters in the same division and era.",
            transform=ax.transAxes, ha="right", fontsize=9.5, color="#555555")
    fig.savefig(path)
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
