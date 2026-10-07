"""Audit the report: every generated number recomputed independently, every hand-written claim checked.

Generated numbers live in report/numbers.tex. Here each one is recomputed
with code that does not use src/ (closed-form sums, scipy quadrature and
distributions, formulas by hand) and compared with the printed string.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from fractions import Fraction

import numpy as np
from scipy import integrate, optimize, stats

from _common import DATA, ROOT, assert_close, prob_greater_closed_form

TEX = (ROOT / "report" / "report.tex").read_text()
MACROS = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}", (ROOT / "report" / "numbers.tex").read_text()))
POST = {k: (s + 1, n - s + 1) for k, (n, s) in DATA.items()}
B = {k: stats.beta(*ab) for k, ab in POST.items()}
Q = dict(epsabs=1e-14, epsrel=1e-12, limit=500)


def f3(x):
    return f"{x:.3f}"


def f4(x):
    return f"{x:.4f}"


def p3(x):
    return r"{>}\,0.999" if x >= 0.9995 else (r"{<}\,0.001" if x < 0.0005 else f"{x:.3f}")


def check(name, expected):
    assert name in MACROS, f"macro \\{name} missing"
    assert MACROS[name] == expected, f"\\{name} = {MACROS[name]!r}, independent value {expected!r}"


def quad(f, lo=0.0, hi=1.0):
    return integrate.quad(f, lo, hi, **Q)[0]


def p_best(k, dists):
    others = [d for j, d in dists.items() if j != k]
    return quad(lambda t: dists[k].pdf(t) * others[0].cdf(t) * others[1].cdf(t))


def test_per_treatment_numbers():
    z = stats.norm.ppf(0.975)
    for k, (n, s) in DATA.items():
        a, b = POST[k]
        d = B[k]
        check(f"alpha{k}", str(a)); check(f"beta{k}", str(b))
        check(f"mle{k}", f"{s / n:.2f}")
        check(f"mean{k}", f4(a / (a + b))); check(f"median{k}", f4(d.median()))
        check(f"mode{k}", f4((a - 1) / (a + b - 2))); check(f"sd{k}", f4(d.std()))
        check(f"crilo{k}", f3(d.ppf(0.025))); check(f"crihi{k}", f3(d.ppf(0.975)))
        se = math.sqrt(s / n * (1 - s / n) / n)
        check(f"waldse{k}", f4(se)); check(f"waldlo{k}", f3(s / n - z * se)); check(f"waldhi{k}", f3(s / n + z * se))
        check(f"meanfrac{k}", rf"\tfrac{{{a}}}{{{a + b}}}")
        bb = stats.betabinom(100, a, b)
        lo, hi = int(bb.ppf(0.025)), int(bb.ppf(0.975))
        check(f"predmean{k}", f"{bb.mean():.1f}"); check(f"predsd{k}", f"{bb.std():.1f}")
        check(f"predlo{k}", str(lo)); check(f"predhi{k}", str(hi))
        check(f"predcov{k}", f3(bb.cdf(hi) - bb.cdf(lo - 1)))
    check("shrinkw", f"{100 / 102:.3f}")


def test_pairwise_numbers():
    z = stats.norm.ppf(0.975)
    for i, j in [("B", "A"), ("B", "C"), ("C", "A")]:
        t = i + j
        (ni, si), (nj, sj) = DATA[i], DATA[j]
        ai, bi = POST[i]; aj, bj = POST[j]
        pg = prob_greater_closed_form(ai, bi, aj, bj)
        check(f"pgt{t}", p3(float(pg)))
        check(f"pgtlong{t}", f"{float(pg):.12f}")
        check(f"pmargin{t}", p3(quad(lambda u: B[i].pdf(u) * B[j].cdf(u - 0.05), 0.05, 1.0)))
        check(f"dmean{t}", f3(B[i].mean() - B[j].mean()))
        check(f"dsd{t}", f3(math.sqrt(B[i].var() + B[j].var())))
        cdf = lambda d: quad(lambda u: B[i].pdf(u) * (1 - B[j].cdf(u - d)))
        check(f"dlo{t}", f3(optimize.brentq(lambda d: cdf(d) - 0.025, -0.9, 0.9, xtol=1e-12)))
        check(f"dhi{t}", f3(optimize.brentq(lambda d: cdf(d) - 0.975, -0.9, 0.9, xtol=1e-12)))
        pi_, pj_ = si / ni, sj / nj
        se = math.sqrt(pi_ * (1 - pi_) / ni + pj_ * (1 - pj_) / nj)
        check(f"cwaldlo{t}", f3(pi_ - pj_ - z * se)); check(f"cwaldhi{t}", f3(pi_ - pj_ + z * se))
        pooled = (si + sj) / (ni + nj)
        zst = (pi_ - pj_) / math.sqrt(pooled * (1 - pooled) * (1 / ni + 1 / nj))
        check(f"zpool{t}", f"{zst:.2f}"); check(f"pval{t}", f3(stats.norm.sf(zst))); check(f"onemp{t}", f3(stats.norm.cdf(zst)))


def test_best_and_decision_numbers():
    pb = {k: p_best(k, B) for k in B}
    emax = quad(lambda t: 1 - B["A"].cdf(t) * B["B"].cdf(t) * B["C"].cdf(t))
    for k in B:
        check(f"pbest{k}", p3(pb[k]))
        check(f"pbestlong{k}", f"{pb[k]:.12f}")
        check(f"eloss{k}", f4(emax - B[k].mean()))
        check(f"elosspts{k}", f"{100 * (emax - B[k].mean()):.2f}")
    check("emax", f4(emax))


def test_sensitivity_numbers():
    tags = {"Uni": (1, 1), "Jef": (0.5, 0.5), "Btt": (2, 2), "Bten": (10, 10)}
    for tag, (a0, b0) in tags.items():
        d = {k: stats.beta(s + a0, n - s + b0) for k, (n, s) in DATA.items()}
        pb = {k: p_best(k, d) for k in d}
        emax = quad(lambda t: 1 - d["A"].cdf(t) * d["B"].cdf(t) * d["C"].cdf(t))
        check(f"s{tag}meanB", f3(d["B"].mean()))
        check(f"s{tag}pBA", p3(quad(lambda t: d["B"].pdf(t) * d["A"].cdf(t))))
        check(f"s{tag}pBC", p3(quad(lambda t: d["B"].pdf(t) * d["C"].cdf(t))))
        for k in d:
            check(f"s{tag}pbest{k}", p3(pb[k]))
        check(f"s{tag}elossB", f4(emax - d["B"].mean()))
        loss = {k: emax - d[k].mean() for k in d}
        assert min(loss, key=loss.get) == max(pb, key=pb.get) == MACROS[f"s{tag}dec"] == "B"


def test_module5_numbers():
    for k, (n, s) in DATA.items():
        a, b = POST[k]
        check(f"mmeanex{k}", f"{a / (a + b):.6f}")
        check(f"moddsex{k}", f"{a / (b - 1):.6f}")
        check(f"mmeanli{k}", f"{s / n + (n - 2 * s) / n**2:.6f}")
        check(f"moddsli{k}", f"{(s + 1) / (n - s):.6f}")
        assert MACROS[f"moddslierr{k}"] == "0"
        mant, exp = MACROS[f"mmeanlierr{k}"].split(r"\times 10^{")
        printed = float(mant) * 10 ** int(exp.rstrip("}"))
        assert_close(printed, s / n + (n - 2 * s) / n**2 - a / (a + b), rtol=0.06, what=f"{k} Lindley error")


def test_scenario_and_monte_carlo_numbers():
    for N, t in ((100, "a"), (200, "b"), (300, "c"), (500, "d"), (1000, "e")):
        ab = {k: (s * N // n + 1, N - s * N // n + 1) for k, (n, s) in DATA.items()}
        check(f"scen{t}pBA", p3(float(prob_greater_closed_form(*ab["B"], *ab["A"]))))
        check(f"scen{t}pBC", p3(float(prob_greater_closed_form(*ab["B"], *ab["C"]))))
    r = json.loads((ROOT / "results" / "results.json").read_text())
    z = [abs(v["z"]) for v in r["monte_carlo"]["pairs"].values()] + [abs(v["z"]) for v in r["monte_carlo"]["best"].values()]
    z.append(abs(r["monte_carlo"]["expected_max"]["z"]))
    check("mcmaxabsz", f"{max(z):.2f}")
    assert max(z) < 4


def test_crossing_point_independently():
    def pbest_B(N):
        d = {k: stats.beta(s * N // n + 1, N - s * N // n + 1) for k, (n, s) in DATA.items()}
        return p_best("B", d)
    assert pbest_B(700) < 0.95 <= pbest_B(750)
    check("scencross", "750")
    assert "seven and a half" in TEX and 750 / 100 == 7.5


def test_hand_written_claims_in_the_text():
    sds = [B[k].std() for k in B]
    assert 0.045 <= min(sds) and max(sds) <= 0.048 and "0.045$--$0.048" in TEX
    half = [(B[k].ppf(0.975) - B[k].ppf(0.025)) / 2 for k in B]
    assert all(0.085 <= h <= 0.095 for h in half) and r"\pm9" in TEX
    assert r"\sum_{j=63}^{101}\binom{101}{j}" in TEX and POST["A"] == (63, 39)
    assert "20 imaginary patients" in TEX
    for k, (n, s) in DATA.items():  # Table 1 rows
        assert re.search(rf"{k} & {n} & {s} & {n - s} &", TEX), f"Table 1 row {k}"
    assert "Lindley $1.9965$" in TEX and "Tierney--Kadane $1.995138$" in TEX  # values checked in test_module5
    worst = 0.0
    for k, (n, s) in DATA.items():  # "accurate to about 1e-4"
        a, b = POST[k]
        worst = max(worst, abs(s / n + (n - 2 * s) / n**2 - a / (a + b)))
    assert worst < 2e-4


def test_quadrature_agreement_claim():
    """'Numerical quadrature agrees to 10^-12' for P(B > A)."""
    exact = float(prob_greater_closed_form(*POST["B"], *POST["A"]))
    assert abs(quad(lambda t: B["B"].pdf(t) * B["A"].cdf(t)) - exact) < 1e-12


def test_report_pdf_length_if_built():
    pdf = ROOT / "report" / "report.pdf"
    if not pdf.exists() or shutil.which("pdfinfo") is None:
        return
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    pages = int(re.search(r"Pages:\s+(\d+)", out).group(1))
    assert 5 <= pages <= 8, f"report has {pages} pages; the brief asks for 5-8"


def test_readme_numbers_match_the_analysis():
    readme = (ROOT / "README.md").read_text()
    for k in ("A", "B", "C"):
        row = rf"\| {k} \| {MACROS['mean' + k]} \| {MACROS['crilo' + k]} – {MACROS['crihi' + k]} \| {MACROS['pbest' + k]} \|"
        assert re.search(row, readme), f"README result row for {k}"
        n, s = DATA[k]
        assert re.search(rf"\| {k} \| {n} \| {s} \|", readme), f"README data row for {k}"
        a, b = POST[k]
        assert f"θ_{k} ~ Beta({a}, {b})" in readme
    for text in (f"P(θ_B > θ_A | data) = {MACROS['pgtBA']}", f"P(B is best) = {MACROS['pbestB']}",
                 f"{MACROS['elossptsB']} percentage points", f"{MACROS['elossptsC']} for C", f"{MACROS['elossptsA']} for A",
                 f"P(θ_B > θ_C) = {MACROS['pgtBC']}", "Lindley 1.9965", "Tierney–Kadane 1.995138"):
        assert text in readme, f"README claim {text!r}"
    assert "(7 pages)" in readme
    pdf = ROOT / "report" / "report.pdf"
    if pdf.exists() and shutil.which("pdfinfo"):
        out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
        assert int(re.search(r"Pages:\s+(\d+)", out).group(1)) == 7, "README says 7 pages"


GROUP = {"Aditya Singh": "250003007", "Hiransh Anand": "250041018", "Parth Pawar": "250041030",
         "Tirthanker Singh": "250003081", "Gulam Abbas": "250041016"}


def test_group_members_everywhere():
    authors = (ROOT / "report" / "authors.tex").read_text()
    readme = (ROOT / "README.md").read_text()
    for name, roll in GROUP.items():
        assert f"{name} ({roll})" in authors, f"authors.tex: {name}"
        assert f"{name} ({roll})" in readme, f"README: {name}"
        assert name in authors.split(r"\projectauthorsplain")[1], f"PDF metadata names: {name}"
    pdf = ROOT / "report" / "report.pdf"
    if pdf.exists() and shutil.which("pdftotext"):
        text = subprocess.run(["pdftotext", "-l", "1", str(pdf), "-"], capture_output=True, text=True).stdout
        for name, roll in GROUP.items():
            assert f"{name} ({roll})" in text, f"report title page: {name} ({roll})"
        meta = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
        assert all(name in meta for name in GROUP), "PDF Author field"


def test_pdf_fonts_are_clean_if_built():
    """Every font embedded, no bitmap fonts, and no font-type mismatch in the report or the figures."""
    if shutil.which("pdffonts") is None:
        return
    pdfs = [ROOT / "report" / "report.pdf"] + sorted((ROOT / "figures").glob("*.pdf"))
    for pdf in pdfs:
        if not pdf.exists():
            continue
        res = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True)
        assert "Mismatch" not in res.stderr, f"{pdf.name}: font type does not match the embedded font file"
        for line in res.stdout.splitlines()[2:]:
            cols = line.split()
            emb, sub, uni = cols[-5], cols[-4], cols[-3]
            assert emb == "yes", f"{pdf.name}: font not embedded: {line}"
            assert not (" Type 3 " in line and uni == "no"), f"{pdf.name}: bitmap Type 3 font (no text mapping): {line}"
