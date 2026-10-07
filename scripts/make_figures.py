"""Draw the report figures (PDF for LaTeX; fig1 also as PNG for the README).

    python scripts/make_figures.py

Colours follow the treatment, never its rank: A blue, B orange, C aqua
(a palette checked for colour-vision deficiency). Every series is labelled
directly, so colour is never the only cue.
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from treatment_bayes import PRIORS, compare, exact, load_trial, posterior  # noqa: E402

COLOR = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e7e6e1"
OUT = ROOT / "figures"


def style() -> None:
    family = "Inter" if any(f.name == "Inter" for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": family, "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlecolor": INK, "axes.labelcolor": INK2,
        "axes.edgecolor": MUTED, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "savefig.facecolor": "white", "figure.facecolor": "white", "axes.facecolor": "white",
        "pdf.fonttype": 3,
    })


def save(fig, name: str, png: bool = False) -> None:
    OUT.mkdir(exist_ok=True)
    # metadata pinned so rebuilding gives byte-identical files
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    if png:  # only the figure shown in the README
        fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def fig_posteriors(posts) -> None:
    x = np.linspace(0.45, 0.88, 1200)
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    for k, p in posts.items():
        y = p.pdf(x)
        lo, hi = p.interval(0.95)
        band = (x >= lo) & (x <= hi)
        ax.fill_between(x[band], 0, y[band], color=COLOR[k], alpha=0.13, lw=0)
        ax.plot(x, y, color=COLOR[k], lw=2)
        ax.plot([p.mean, p.mean], [0, p.pdf(p.mean)], color=COLOR[k], lw=1, ls=(0, (2, 2)))
    # direct labels on the outer flanks / above the middle peak so they never collide
    place = {"A": (-0.075, (-5, 0), "right", "center"), "B": (0.075, (5, 0), "left", "center"),
             "C": (0.0, (0, 13), "center", "bottom")}
    for k, p in posts.items():
        dx, off, ha, va = place[k]
        xk = p.mode + dx
        ax.annotate(f"{k}: Beta({int(p.alpha)}, {int(p.beta)})", xy=(xk, p.pdf(xk)), xytext=off,
                    textcoords="offset points", ha=ha, va=va, color=INK, fontsize=8.5)
    ax.set_xlim(0.45, 0.88)
    ax.set_ylim(0, max(p.pdf(p.mode) for p in posts.values()) * 1.18)
    ax.set_xlabel("success probability θ")
    ax.set_ylabel("posterior density")
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.text(0.01, 0.97, "shaded: 95% credible interval\ndashed: posterior mean",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color=INK2)
    save(fig, "fig1_posteriors", png=True)


def fig_differences(posts) -> None:
    pairs = [("B", "A"), ("B", "C"), ("C", "A")]
    exacts = {k: exact.BetaExact(int(p.alpha), int(p.beta)) for k, p in posts.items()}
    d = np.linspace(-0.3, 0.42, 721)
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.25), sharey=True)
    ymax = 0.0
    dens = {}
    for i, j in pairs:
        dens[(i, j)] = compare.diff_pdf(posts[i], posts[j], d)
        ymax = max(ymax, dens[(i, j)].max())
    for ax, (i, j) in zip(axes, pairs):
        y = dens[(i, j)]
        pos = d >= 0
        ax.fill_between(d[pos], 0, y[pos], color=COLOR[i], alpha=0.22, lw=0)
        ax.plot(d, y, color=INK2, lw=1.6)
        ax.axvline(0, color=MUTED, lw=0.9)
        p = float(exact.prob_greater(exacts[i], exacts[j]))
        ax.text(0.97, 0.95, f"P(θ{i} > θ{j})\n= {p:.3f}", transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=INK)
        ax.set_title(f"θ{i} − θ{j}")
        ax.set_xlim(d[0], d[-1])
        ax.set_ylim(0, ymax * 1.12)
        ax.set_xticks([-0.2, 0.0, 0.2, 0.4])
        ax.grid(True, axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("posterior density")
    axes[1].set_xlabel("difference in success probability")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig2_differences")


def fig_best(posts) -> None:
    exacts = {k: exact.BetaExact(int(p.alpha), int(p.beta)) for k, p in posts.items()}
    names = list(posts)
    pbest = [float(exact.prob_best(exacts, k)) for k in names]
    loss = [100 * float(exact.expected_loss(exacts, k)) for k in names]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.3))
    for ax, vals, title, fmt, top in ((a1, pbest, "P(treatment is best)", "{:.3f}", 1.0),
                                      (a2, loss, "Expected loss of choosing it (percentage points)", "{:.2f}", None)):
        bars = ax.bar(names, vals, color=[COLOR[k] for k in names], width=0.62, edgecolor="white", linewidth=1.5)
        for b, v in zip(bars, vals):
            ax.annotate(fmt.format(v), xy=(b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.set_title(title)
        ax.set_ylim(0, (top or max(vals)) * 1.15)
        ax.grid(True, axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig3_best")


def fig_more_data(arms) -> None:
    sizes = [100, 200, 300, 400, 500, 600, 800, 1000]
    pb, pbc = [], []
    for N in sizes:
        es = {}
        for k, a in arms.items():
            s = Fraction(a.s, a.n) * N
            if s.denominator != 1:
                raise ValueError(f"N={N} does not give whole successes for {k}")
            es[k] = exact.BetaExact(int(s) + 1, N - int(s) + 1)
        pb.append(float(exact.prob_best(es, "B")))
        pbc.append(float(exact.prob_greater(es["B"], es["C"])))
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.plot(sizes, pbc, color=INK2, lw=1.6, marker="s", ms=4, label="P(θB > θC)")
    ax.plot(sizes, pb, color=COLOR["B"], lw=2, marker="o", ms=4.5, label="P(B is best)")
    ax.axhline(0.95, color=MUTED, lw=0.9, ls=(0, (3, 3)))
    ax.text(sizes[-1], 0.947, "0.95", fontsize=7.5, color=INK2, va="top", ha="right")
    ax.legend(loc="lower right", fontsize=8, handlelength=2.2)
    ax.set_xlabel("patients per arm, if the observed success rates held exactly")
    ax.set_ylabel("posterior probability")
    ax.set_ylim(0.6, 1.0)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "fig4_more_data")


def main() -> None:
    style()
    arms = load_trial()
    posts = {k: posterior(a, PRIORS["uniform"]) for k, a in arms.items()}
    fig_posteriors(posts)
    fig_differences(posts)
    fig_best(posts)
    fig_more_data(arms)
    print("figures written to", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
