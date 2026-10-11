"""Statistics shared by the CPU stage-A experiments (standard library only, no numpy).

Student-t quantiles by bisection of the exact CDF, exact paired one-sided t bounds, and the Monte-Carlo helpers the
power analyses use. A unit test pins the quantiles to published table values. Not imported by any harness."""

from __future__ import annotations

import math
import random
from decimal import Decimal, getcontext
from fractions import Fraction

getcontext().prec = 40


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the regularised incomplete beta function (Numerical Recipes, Lentz)."""
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def t_cdf(t: float, df: int) -> float:
    x = df / (df + t * t)
    ln_beta = math.lgamma(df / 2) + math.lgamma(0.5) - math.lgamma(df / 2 + 0.5)
    front = math.exp(0.5 * df * math.log(x) + 0.5 * math.log(1 - x) - ln_beta)
    ib = (
        front * _betacf(df / 2, 0.5, x) / (df / 2)
        if x < (df / 2 + 1) / (df / 2 + 0.5 + 2)
        else 1 - front * _betacf(0.5, df / 2, 1 - x) / 0.5
    )
    return 1 - 0.5 * ib if t > 0 else 0.5 * ib


def t_quantile(p: float, df: int) -> float:
    """The p-quantile (0.5 < p < 1) of Student's t with `df` degrees of freedom, by bisection."""
    lo, hi = 0.0, 50.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if t_cdf(mid, df) < p else (lo, mid)
    return (lo + hi) / 2


def quantile_string(p: float, df: int) -> str:
    """The pinned spelling of a t quantile: six decimals, as the criteria files store it."""
    return f"{t_quantile(p, df):.6f}"


class PairedError(Exception):
    """A paired comparison that cannot be computed from the seeds' results."""


def paired_bound(differences: list[Fraction], t_quantile_value: Decimal) -> dict:
    """Mean, sample standard deviation and the one-sided t bounds (lower and upper) of paired differences: exact fractions in,
    Decimal arithmetic, floats out. The quantile is the pinned one for n - 1 degrees of freedom."""
    n = len(differences)
    if n < 2:
        raise PairedError("a paired bound needs at least two seeds")
    mean = sum(differences, Fraction(0)) / n
    variance = sum(((d - mean) ** 2 for d in differences), Fraction(0)) / (n - 1)
    sd = (Decimal(variance.numerator) / Decimal(variance.denominator)).sqrt()
    mean_d = Decimal(mean.numerator) / Decimal(mean.denominator)
    half = t_quantile_value * sd / Decimal(n).sqrt()
    return {
        "mean_permille": float(mean_d * 1000),
        "sd_permille": float(sd * 1000),
        "lower_bound_permille": float((mean_d - half) * 1000),
        "upper_bound_permille": float((mean_d + half) * 1000),
        "n": n,
    }


def median(values: list[Fraction]) -> Fraction:
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


# -- Monte-Carlo helpers of the power analyses --------------------------------------------------------

def bounds_of(xs: list[float], t_value: float) -> tuple[float, float]:
    """(mean, standard error * t) of a sample, floats."""
    n = len(xs)
    mean = sum(xs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1))
    return mean, t_value * sd / math.sqrt(n)


def covariance(rows: list[list[float]]) -> tuple[list[float], list[list[float]]]:
    """Mean vector and sample covariance (n - 1) of equal-length vectors."""
    n, k = len(rows), len(rows[0])
    mean = [sum(r[j] for r in rows) / n for j in range(k)]
    return mean, [[sum((r[a] - mean[a]) * (r[b] - mean[b]) for r in rows) / (n - 1) for b in range(k)] for a in range(k)]


def cholesky(matrix: list[list[float]], jitter: float = 1e-6) -> list[list[float]]:
    """Lower-triangular factor of a (semi-)definite matrix; `jitter` keeps a singular covariance factorisable."""
    k = len(matrix)
    low = [[0.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(i + 1):
            total = matrix[i][j] + (jitter if i == j else 0.0) - sum(low[i][m] * low[j][m] for m in range(j))
            low[i][j] = math.sqrt(max(total, 1e-12)) if i == j else total / low[j][j]
    return low


def draw_mvn(rng: random.Random, mean: list[float], low: list[list[float]]) -> list[float]:
    z = [rng.gauss(0.0, 1.0) for _ in mean]
    return [mean[j] + sum(low[j][m] * z[m] for m in range(j + 1)) for j in range(len(mean))]


def outcome_probabilities(means, cov, n, margin, t_support, t_refute, trials=20000, seed=0, p_invalid=0.0):
    """Monte-Carlo probabilities (COMPLETED, NOT_SHOWN, FALSIFIED, INVALID) of the stage-A decision rule for judged contrasts drawn per seed from
    N(means, cov) (n seeds per trial): COMPLETED iff every one-sided lower bound (t_support) reaches the margin; FALSIFIED (refuted) iff some
    Bonferroni upper bound (t_refute) is below it; NOT_SHOWN otherwise. `p_invalid` is the probability that a guard or control makes the run INVALID
    (independent of the contrasts; estimated from the pilots by the caller). The structural verdicts and the integrity counters are not modelled."""
    rng = random.Random(seed)
    low = cholesky(cov)
    counts = {"COMPLETED": 0, "NOT_SHOWN": 0, "FALSIFIED": 0, "INVALID": 0}
    for _ in range(trials):
        if p_invalid > 0 and rng.random() < p_invalid:
            counts["INVALID"] += 1
            continue
        rows = [draw_mvn(rng, list(means), low) for _ in range(n)]
        bounds = [bounds_of([r[j] for r in rows], 1.0) for j in range(len(means))]
        if all(m - t_support * se >= margin for m, se in bounds):
            counts["COMPLETED"] += 1
        elif any(m + t_refute * se < margin for m, se in bounds):
            counts["FALSIFIED"] += 1
        else:
            counts["NOT_SHOWN"] += 1
    return tuple(counts[k] / trials for k in ("COMPLETED", "NOT_SHOWN", "FALSIFIED", "INVALID"))
