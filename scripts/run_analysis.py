"""Run the whole Project 2 analysis and write every reported number.

    python scripts/run_analysis.py

Outputs
  results/results.json     all quantities (exact rationals as "p/q" strings too)
  report/numbers.tex       LaTeX macros read by the report, so no number in it
                           is typed by hand

Main prior: independent Beta(1,1) for each treatment. Sensitivity priors:
Jeffreys Beta(1/2,1/2), Beta(2,2), Beta(10,10).
"""

from __future__ import annotations

import json
import platform
import sys
from fractions import Fraction
from itertools import permutations
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from treatment_bayes import PRIORS, load_trial, posterior  # noqa: E402
from treatment_bayes import approx, classical, compare, exact, montecarlo, predictive  # noqa: E402
from treatment_bayes.data import Arm  # noqa: E402
from treatment_bayes.decision import decide  # noqa: E402

SEED = 20261008
N_DRAWS = 2_000_000
PAIRS = [("B", "A"), ("B", "C"), ("C", "A")]
MARGIN = Fraction(1, 20)  # 5 percentage points
FUTURE_PATIENTS = 100
SCENARIO_SIZES = [100, 200, 300, 500, 1000]


def frac(x: Fraction) -> str:
    return f"{x.numerator}/{x.denominator}"


def main() -> None:
    arms = load_trial()
    names = list(arms)
    prior = PRIORS["uniform"]
    posts = {k: posterior(a, prior) for k, a in arms.items()}
    exacts = {k: exact.BetaExact(int(p.alpha), int(p.beta)) for k, p in posts.items()}

    out: dict = {
        "data": {k: {"n": a.n, "s": a.s} for k, a in arms.items()},
        "prior": {"name": prior.name, "alpha": prior.alpha, "beta": prior.beta},
        "settings": {"seed": SEED, "n_draws": N_DRAWS, "margin": float(MARGIN),
                     "future_patients": FUTURE_PATIENTS},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    }

    # ---- 1 and 3: estimates and uncertainty per treatment -------------------
    per = {}
    for k, a in arms.items():
        p, e = posts[k], exacts[k]
        c = classical.proportion(a)
        lo, hi = p.interval(0.95)
        pred = predictive.predictive(p, FUTURE_PATIENTS)
        per[k] = {
            "alpha": int(p.alpha), "beta": int(p.beta),
            "mle": c.mle, "wald_se": c.se, "wald_low": c.wald_low, "wald_high": c.wald_high,
            "mean": float(e.mean), "mean_exact": frac(e.mean),
            "mode": float(e.mode), "mode_exact": frac(e.mode),
            "median": p.median, "sd": p.sd, "var_exact": frac(e.var),
            "cri_low": lo, "cri_high": hi,
            "pred_next": float(e.mean),
            "pred_m": pred.m, "pred_mean": pred.mean, "pred_sd": pred.sd,
            "pred_low": pred.low, "pred_high": pred.high, "pred_coverage": pred.coverage,
        }
    out["per_treatment"] = per

    # ---- 2 and 5: pairwise differences ---------------------------------------
    pairs = {}
    for i, j in PAIRS:
        pi, pj = posts[i], posts[j]
        cl = classical.difference(arms[i], arms[j])
        lo, hi = compare.diff_interval(pi, pj, 0.95)
        gt = exact.prob_greater(exacts[i], exacts[j])
        gt_margin = exact.prob_diff_greater(exacts[i], exacts[j], MARGIN)
        pairs[f"{i}-{j}"] = {
            "mean": pi.mean - pj.mean, "sd": (pi.var + pj.var) ** 0.5,
            "cri_low": lo, "cri_high": hi,
            "prob_greater": float(gt), "prob_greater_exact": frac(gt),
            "prob_greater_margin": float(gt_margin), "prob_greater_margin_exact": frac(gt_margin),
            "classical_estimate": cl.estimate, "classical_se": cl.se,
            "wald_low": cl.wald_low, "wald_high": cl.wald_high,
            "z_pooled": cl.z_pooled, "p_one_sided": cl.p_one_sided,
        }
    out["pairs"] = pairs

    # ---- 6 and 7: best treatment and the decision ---------------------------
    pbest = {k: exact.prob_best(exacts, k) for k in names}
    assert sum(pbest.values()) == 1, "P(best) must sum to exactly 1"
    emax = exact.expected_max(exacts)
    eloss = {k: exact.expected_loss(exacts, k) for k in names}
    dec = decide({k: float(v) for k, v in eloss.items()}, {k: float(v) for k, v in pbest.items()})
    # probability of each full ranking (best, middle, worst), by symmetry from P(best) structure
    out["best"] = {
        "prob_best": {k: float(v) for k, v in pbest.items()},
        "prob_best_exact": {k: frac(v) for k, v in pbest.items()},
        "expected_max": float(emax), "expected_max_exact": frac(emax),
        "expected_loss": {k: float(v) for k, v in eloss.items()},
        "expected_loss_exact": {k: frac(v) for k, v in eloss.items()},
        "decision_by_expected_loss": dec.by_expected_loss,
        "decision_by_prob_best": dec.by_prob_best,
    }

    # ---- Monte Carlo cross-check ----------------------------------------------
    rng = np.random.default_rng(SEED)
    draws = montecarlo.draw(posts, N_DRAWS, rng)
    mc = {"pairs": {}, "best": {}}
    for i, j in PAIRS:
        est = montecarlo.mc_prob_greater(draws, i, j)
        mc["pairs"][f"{i}-{j}"] = {"value": est.value, "se": est.se, "z": est.z_against(pairs[f"{i}-{j}"]["prob_greater"]),
                                   "diff_quantiles": montecarlo.mc_diff_quantiles(draws, i, j)}
    for k, est in montecarlo.mc_prob_best(draws).items():
        mc["best"][k] = {"value": est.value, "se": est.se, "z": est.z_against(float(pbest[k]))}
    em = montecarlo.mc_expected_max(draws)
    mc["expected_max"] = {"value": em.value, "se": em.se, "z": em.z_against(float(emax))}
    out["monte_carlo"] = mc

    # ---- prior sensitivity (quadrature: works for half-integer priors too) ------
    sens = {}
    for key, pr in PRIORS.items():
        ps = {k: posterior(a, pr) for k, a in arms.items()}
        pb = {k: compare.prob_best(ps, k) for k in names}
        el = {k: compare.expected_loss(ps, k) for k in names}
        d = decide(el, pb)
        sens[key] = {
            "name": pr.name, "alpha": pr.alpha, "beta": pr.beta,
            "mean": {k: ps[k].mean for k in names},
            "prob_B_gt_A": compare.prob_greater(ps["B"], ps["A"]),
            "prob_B_gt_C": compare.prob_greater(ps["B"], ps["C"]),
            "prob_best": pb, "expected_loss": el,
            "decision_by_expected_loss": d.by_expected_loss, "decision_by_prob_best": d.by_prob_best,
        }
    out["sensitivity"] = sens

    # ---- Module 5 approximations against the exact conjugate answers -------------
    m5 = {}
    for k, a in arms.items():
        e = exacts[k]
        row = {}
        for which, ex in (("mean", e.mean), ("odds", e.odds_mean())):
            li = approx.binomial_lindley(a.n, a.s, which)
            tk = approx.binomial_tierney_kadane(a.n, a.s, which)
            row[which] = {"exact": float(ex), "exact_frac": frac(ex), "lindley": li, "tk": tk.value,
                          "lindley_err": li - float(ex), "tk_err": tk.value - float(ex)}
        m5[k] = row
    out["module5"] = m5

    # ---- scenario: same observed rates, more patients per arm (illustration) --------
    rates = {k: Fraction(a.s, a.n) for k, a in arms.items()}
    scen = {}
    for N in SCENARIO_SIZES:
        succ = {k: rates[k] * N for k in names}
        if any(v.denominator != 1 for v in succ.values()):
            raise ValueError(f"N={N}: observed rates do not give whole numbers of successes")
        ex_n = {k: exact.BetaExact(int(succ[k]) + 1, N - int(succ[k]) + 1) for k in names}
        scen[str(N)] = {"prob_best_B": float(exact.prob_best(ex_n, "B")),
                        "prob_B_gt_A": float(exact.prob_greater(ex_n["B"], ex_n["A"])),
                        "prob_B_gt_C": float(exact.prob_greater(ex_n["B"], ex_n["C"]))}
    # smallest per-arm size (multiples of 50, whole successes) at which P(B best) >= 0.95
    cross = None
    for N in range(100, 2001, 50):
        succ = {k: rates[k] * N for k in names}
        if any(v.denominator != 1 for v in succ.values()):
            continue
        ex_n = {k: exact.BetaExact(int(succ[k]) + 1, N - int(succ[k]) + 1) for k in names}
        if exact.prob_best(ex_n, "B") >= Fraction(95, 100):
            cross = N
            break
    scen["crossing_095"] = cross
    out["scenario_more_data"] = scen

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "results.json").write_text(json.dumps(out, indent=2) + "\n")
    write_macros(out, ROOT / "report" / "numbers.tex")
    print(f"wrote results/results.json and report/numbers.tex "
          f"(decision: {dec.by_expected_loss} by expected loss, {dec.by_prob_best} by P(best))")


# ---------------------------------------------------------------------------
# LaTeX macros
# ---------------------------------------------------------------------------

def f3(x: float) -> str:
    return f"{x:.3f}"


def p3(x: float) -> str:
    """A probability to 3 decimals, never rounded to a misleading 0.000 or 1.000."""
    if x >= 0.9995:
        return r"{>}\,0.999"
    if x < 0.0005:
        return r"{<}\,0.001"
    return f"{x:.3f}"


def sci(x: float, digits: int = 1) -> str:
    """LaTeX scientific notation; values below 1e-12 in size are exactly 0 up to rounding."""
    if abs(x) < 1e-12:
        return "0"
    mant, exp = f"{x:.{digits}e}".split("e")
    return rf"{mant}\times 10^{{{int(exp)}}}"


def f4(x: float) -> str:
    return f"{x:.4f}"


def write_macros(r: dict, path: Path) -> None:
    m: dict[str, str] = {}
    L = {"A": "A", "B": "B", "C": "C"}
    for k, v in r["per_treatment"].items():
        t = L[k]
        m[f"alpha{t}"] = str(v["alpha"]); m[f"beta{t}"] = str(v["beta"])
        m[f"mle{t}"] = f"{v['mle']:.2f}"
        m[f"waldse{t}"] = f4(v["wald_se"])
        m[f"waldlo{t}"] = f3(v["wald_low"]); m[f"waldhi{t}"] = f3(v["wald_high"])
        m[f"mean{t}"] = f4(v["mean"]); m[f"meanfrac{t}"] = rf"\tfrac{{{v['alpha']}}}{{{v['alpha'] + v['beta']}}}"
        m[f"median{t}"] = f4(v["median"]); m[f"mode{t}"] = f4(v["mode"]); m[f"sd{t}"] = f4(v["sd"])
        m[f"crilo{t}"] = f3(v["cri_low"]); m[f"crihi{t}"] = f3(v["cri_high"])
        m[f"predmean{t}"] = f"{v['pred_mean']:.1f}"; m[f"predsd{t}"] = f"{v['pred_sd']:.1f}"
        m[f"predlo{t}"] = str(v["pred_low"]); m[f"predhi{t}"] = str(v["pred_high"])
        m[f"predcov{t}"] = f3(v["pred_coverage"])
    for key, v in r["pairs"].items():
        t = key.replace("-", "")  # BA, BC, CA
        m[f"dmean{t}"] = f3(v["mean"]); m[f"dsd{t}"] = f3(v["sd"])
        m[f"dlo{t}"] = f3(v["cri_low"]); m[f"dhi{t}"] = f3(v["cri_high"])
        m[f"pgt{t}"] = p3(v["prob_greater"]); m[f"pgtlong{t}"] = f"{v['prob_greater']:.12f}"
        m[f"pmargin{t}"] = p3(v["prob_greater_margin"])
        m[f"cwaldlo{t}"] = f3(v["wald_low"]); m[f"cwaldhi{t}"] = f3(v["wald_high"])
        m[f"cse{t}"] = f4(v["classical_se"])
        m[f"zpool{t}"] = f"{v['z_pooled']:.2f}"; m[f"pval{t}"] = f3(v["p_one_sided"])
        m[f"onemp{t}"] = f3(1 - v["p_one_sided"])
    b = r["best"]
    for k in ("A", "B", "C"):
        m[f"pbest{k}"] = p3(b["prob_best"][k]); m[f"pbestlong{k}"] = f"{b['prob_best'][k]:.12f}"
        m[f"eloss{k}"] = f4(b["expected_loss"][k])
    m["emax"] = f4(b["expected_max"])
    for k in ("A", "B", "C"):
        m[f"elosspts{k}"] = f"{100 * b['expected_loss'][k]:.2f}"
    n_b = r["data"]["B"]["n"]
    m["shrinkw"] = f"{n_b / (n_b + 2):.3f}"
    mc = r["monte_carlo"]
    m["mcdraws"] = f"{r['settings']['n_draws']:,}".replace(",", "{,}")
    m["mcseed"] = str(r["settings"]["seed"])
    m["mcpgtBA"] = f"{mc['pairs']['B-A']['value']:.4f}"; m["mcseBA"] = f"{mc['pairs']['B-A']['se']:.5f}"
    m["mczBA"] = f"{mc['pairs']['B-A']['z']:.2f}"
    m["mcpbestB"] = f"{mc['best']['B']['value']:.4f}"; m["mcsebestB"] = f"{mc['best']['B']['se']:.5f}"
    m["mczbestB"] = f"{mc['best']['B']['z']:.2f}"
    zs = [abs(v["z"]) for v in mc["pairs"].values()] + [abs(v["z"]) for v in mc["best"].values()] + [abs(mc["expected_max"]["z"])]
    m["mcmaxabsz"] = f"{max(zs):.2f}"
    names = {"uniform": "Uni", "jeffreys": "Jef", "beta22": "Btt", "beta1010": "Bten"}
    for key, v in r["sensitivity"].items():
        t = names[key]
        m[f"s{t}meanB"] = f3(v["mean"]["B"])
        m[f"s{t}pBA"] = f3(v["prob_B_gt_A"]); m[f"s{t}pBC"] = f3(v["prob_B_gt_C"])
        for k in ("A", "B", "C"):
            m[f"s{t}pbest{k}"] = f3(v["prob_best"][k])
        m[f"s{t}elossB"] = f4(v["expected_loss"]["B"])
        m[f"s{t}dec"] = v["decision_by_expected_loss"] if v["decision_by_expected_loss"] == v["decision_by_prob_best"] else "split"
    for k, v in r["module5"].items():
        for which, w in v.items():
            t = "mean" if which == "mean" else "odds"
            m[f"m{t}ex{k}"] = f"{w['exact']:.6f}"; m[f"m{t}li{k}"] = f"{w['lindley']:.6f}"; m[f"m{t}tk{k}"] = f"{w['tk']:.6f}"
            m[f"m{t}lierr{k}"] = sci(w["lindley_err"]); m[f"m{t}tkerr{k}"] = sci(w["tk_err"])
    m["scencross"] = str(r["scenario_more_data"]["crossing_095"])
    for N, v in r["scenario_more_data"].items():
        if N == "crossing_095":
            continue
        t = {"100": "a", "200": "b", "300": "c", "500": "d", "1000": "e"}[N]
        m[f"scen{t}pbest"] = p3(v["prob_best_B"]); m[f"scen{t}pBC"] = p3(v["prob_B_gt_C"]); m[f"scen{t}pBA"] = p3(v["prob_B_gt_A"])

    lines = ["% Generated by scripts/run_analysis.py. Do not edit by hand.",
             "% Use numeric macros inside math mode so minus signs render correctly."]
    for name in sorted(m):
        if not name.isalpha():
            raise ValueError(f"macro name {name!r} must be letters only")
        lines.append(rf"\newcommand{{\{name}}}{{{m[name]}}}")
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
