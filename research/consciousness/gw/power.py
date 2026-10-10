"""Student-t quantiles and a Monte-Carlo power estimate for C001's decision rule (standard library only).

Used to derive the pinned t quantiles of criteria.toml and the power statement of the preregistration (section 4); a unit test pins
the quantiles to published table values. Not imported by the harness.

    python research/consciousness/gw/power.py        # prints the tables of section 4 (final design: three pairwise comparators, margin 30)  (output saved in pilots/power.txt)

Two Monte-Carlo models of the decision rule: `decision_probabilities` (independent normal paired differences, three outcomes: the
design-stage estimate) and `decision_outcomes` (round 6: per-seed vectors of the three contrasts and the two sensitivity-control
deficits drawn from a multivariate normal fitted to the pilot seeds, so that the INVALID outcome of an unresolved control is part of the
estimate, with the graded and gross controls, the convergence-guard allowance and the order of `aggregate.decide`).
"""

from __future__ import annotations

import importlib.util
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rules  # noqa: E402  (the harness's own definition of "the workspace trained")


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
    ib = front * _betacf(df / 2, 0.5, x) / (df / 2) if x < (df / 2 + 1) / (df / 2 + 0.5 + 2) else 1 - front * _betacf(0.5, df / 2, 1 - x) / 0.5
    return 1 - 0.5 * ib if t > 0 else 0.5 * ib


def t_quantile(p: float, df: int) -> float:
    """The p-quantile (0.5 < p < 1) of Student's t with `df` degrees of freedom, by bisection."""
    lo, hi = 0.0, 50.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if t_cdf(mid, df) < p else (lo, mid)
    return (lo + hi) / 2


def decision_probabilities(means, sds, n, margin, t_support, t_refute, pairwise, trials=20000, seed=0):
    """Monte-Carlo probabilities of the three outcomes for independent normal paired differences (a conservative assumption:
    the real differences share the workspace and are positively correlated). `means`/`sds` are per comparator (permille), the
    first `pairwise` of which are the pairwise comparators the refutation rule looks at. Returns (supported, refuted, not_shown)."""
    rng = random.Random(seed)
    supported = refuted = 0
    for _ in range(trials):
        lowers, uppers = [], []
        for m, s in zip(means, sds):
            xs = [rng.gauss(m, s) for _ in range(n)]
            mean = sum(xs) / n
            sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1))
            lowers.append(mean - t_support * sd / math.sqrt(n))
            uppers.append(mean + t_refute * sd / math.sqrt(n))
        if all(v >= margin for v in lowers):
            supported += 1
        elif any(v < margin for v in uppers[:pairwise]):
            refuted += 1
    return supported / trials, refuted / trials, 1 - (supported + refuted) / trials


def covariance(rows):
    """Sample covariance (n - 1) of a list of equal-length vectors."""
    n, k = len(rows), len(rows[0])
    mean = [sum(r[j] for r in rows) / n for j in range(k)]
    return mean, [[sum((r[a] - mean[a]) * (r[b] - mean[b]) for r in rows) / (n - 1) for b in range(k)] for a in range(k)]


def cholesky(matrix, jitter=1e-6):
    """Lower-triangular factor of a (semi-)definite matrix; `jitter` keeps a singular sample covariance factorisable."""
    k = len(matrix)
    low = [[0.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(i + 1):
            total = matrix[i][j] + (jitter if i == j else 0.0) - sum(low[i][m] * low[j][m] for m in range(j))
            low[i][j] = math.sqrt(max(total, 1e-12)) if i == j else total / low[j][j]
    return low


# Order of the five per-seed quantities of `decision_outcomes`: the three workspace advantages over (all_to_all, bw_all_to_all,
# sparse_all_to_all), the graded deficit (all_to_all minus the weak control) and the gross deficit (all_to_all minus crippled).
QUANTITIES = ("ws-a2a", "ws-bw", "ws-sparse", "graded", "gross")


def decision_outcomes(means, cov, n, margin, t_support, t_refute, graded_min, gross_min, p_unconverged=0.0, max_unconverged=2,
                      p_workspace_fail=0.0, fail_contrasts=(0.0, 0.0, 0.0), fail_counted=True, max_untrained=4, trials=20000, seed=0):
    """Monte-Carlo probabilities (COMPLETED, NOT_SHOWN, FALSIFIED, INVALID) of the final decision rule. Per seed a vector of the five
    `QUANTITIES` is drawn from N(means, cov) (n seeds per trial); the verdict follows `aggregate.decide`: INVALID when the graded
    control's one-sided lower bound does not exceed `graded_min`, the gross control's does not reach `gross_min`, or more than
    `max_unconverged` seeds have an unconverged guarded arm (each seed independently with probability `p_unconverged`), or the workspace
    arm did not train on more than `max_untrained` seeds (each seed independently with probability `p_workspace_fail`; on such a seed
    the three contrasts are `fail_contrasts` instead of the drawn ones, as the paired differences keep every seed; the seed counts toward the guard only
    when `fail_counted`, i.e. when the harness rule `rules.workspace_trained` flags it); otherwise
    FALSIFIED when some Bonferroni upper bound (t_refute) of the three contrasts is below the margin, COMPLETED when all three lower
    bounds (t_support) reach it, NOT_SHOWN else. The comparator band guard is not modelled (no pilot seed came near it)."""
    rng = random.Random(seed)
    low = cholesky(cov)
    k = len(means)
    counts = {"COMPLETED": 0, "NOT_SHOWN": 0, "FALSIFIED": 0, "INVALID": 0}
    for _ in range(trials):
        columns = [[] for _ in range(k)]
        untrained = 0
        for _ in range(n):
            z = [rng.gauss(0.0, 1.0) for _ in range(k)]
            failed = p_workspace_fail > 0.0 and rng.random() < p_workspace_fail
            untrained += failed and fail_counted
            for j in range(k):
                drawn = means[j] + sum(low[j][m] * z[m] for m in range(j + 1))
                columns[j].append(fail_contrasts[j] if failed and j < 3 else drawn)
        stats = []
        for column in columns:
            mean = sum(column) / n
            sd = math.sqrt(sum((x - mean) ** 2 for x in column) / (n - 1))
            stats.append((mean, sd / math.sqrt(n)))
        unconverged = sum(1 for _ in range(n) if rng.random() < p_unconverged)
        graded_ok = stats[3][0] - t_support * stats[3][1] > graded_min
        gross_ok = stats[4][0] - t_support * stats[4][1] >= gross_min
        if not (graded_ok and gross_ok) or unconverged > max_unconverged or untrained > max_untrained:
            counts["INVALID"] += 1
        elif any(m + t_refute * e < margin for m, e in stats[:3]):
            counts["FALSIFIED"] += 1
        elif all(m - t_support * e >= margin for m, e in stats[:3]):
            counts["COMPLETED"] += 1
        else:
            counts["NOT_SHOWN"] += 1
    return tuple(counts[name] / trials for name in ("COMPLETED", "NOT_SHOWN", "FALSIFIED", "INVALID"))


def graded_control_failure(mean, sd, n, graded_min, t_support, trials=20000, seed=0):
    """P(the graded control does not resolve over n seeds) for per-seed deficits N(mean, sd): the lower one-sided t bound of the
    n-seed mean is not above `graded_min`."""
    rng = random.Random(seed)
    failures = 0
    for _ in range(trials):
        xs = [rng.gauss(mean, sd) for _ in range(n)]
        m = sum(xs) / n
        s = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
        failures += m - t_support * s / math.sqrt(n) <= graded_min
    return failures / trials


def bootstrap_graded_control_failure(deficits, n, graded_min, t_support, trials=20000, seed=0):
    """Like `graded_control_failure`, but the n per-seed deficits are drawn with replacement from the observed ones (no normality
    assumption, heavy tails and outliers stay)."""
    rng = random.Random(seed)
    failures = 0
    for _ in range(trials):
        xs = [rng.choice(deficits) for _ in range(n)]
        m = sum(xs) / n
        s = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
        failures += m - t_support * s / math.sqrt(n) <= graded_min
    return failures / trials


def load_pilot_table():
    """seed -> {arm: 16-module accuracy per mille} from the raw pilot outputs (pilots/summarise.py:seed_table)."""
    path = Path(__file__).resolve().parent / "pilots" / "summarise.py"
    spec = importlib.util.spec_from_file_location("c001_pilot_summarise", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.seed_table(), module

# The final design (v3): the three pairwise comparators, margin 30 per mille, twenty seeds. The means and standard deviations are the
# pilot ones of the paired differences at the pinned settings (all_to_all, bw_all_to_all, sparse_all_to_all; pilot seeds 1004-1008,
# each comparator at its selected learning rate, including the lr 0.006 extension; the second extension, lr 0.012, won for no arm and changed nothing: see pilots/summarise.py).
PILOT_MEANS = (31.4, 46.8, 89.2)
PILOT_SDS = (24.4, 42.0, 64.4)
MARGIN = 30
N = 20


def scenarios() -> list[tuple[str, tuple[float, ...]]]:
    return [
        ("pilot means (31.4, 46.8, 89.2)", PILOT_MEANS),
        ("pilot means halved (winner's curse)", tuple(m / 2 for m in PILOT_MEANS)),
        ("zero effect", (0.0, 0.0, 0.0)),
        ("effect = margin (30 everywhere)", (30.0, 30.0, 30.0)),
        ("effect 45 everywhere", (45.0, 45.0, 45.0)),
        ("effect 80 everywhere", (80.0, 80.0, 80.0)),
    ]


STALL_BELOW = 450  # descriptive: a pilot seed whose workspace 16-module accuracy is below this "stalled" (chance 333; the next lowest seed is 767). The GUARD uses rules.workspace_trained instead


def pilot_vectors(table, seeds, trained=None):
    """The per-seed vectors of `QUANTITIES` for the seeds that have every arm (workspace, the three pairwise comparators, the graded
    control weak_all_to_all and the gross control crippled_all_to_all). `trained` True keeps only the seeds on which the workspace did not stall (16-module accuracy at least STALL_BELOW), False only the stalled
    ones, None all. (Whether the guard counts a stalled seed is `rules.workspace_trained` on its in-distribution accuracies, `harness_counts`.)"""
    need = ("workspace", "all_to_all", "bw_all_to_all", "sparse_all_to_all", "weak_all_to_all", "crippled_all_to_all")
    rows = []
    for seed in seeds:
        t = table.get(seed, {})
        if all(arm in t for arm in need) and (trained is None or (t["workspace"] >= STALL_BELOW) == trained):
            rows.append([t["workspace"] - t["all_to_all"], t["workspace"] - t["bw_all_to_all"], t["workspace"] - t["sparse_all_to_all"],
                         t["all_to_all"] - t["weak_all_to_all"], t["all_to_all"] - t["crippled_all_to_all"]])
    return rows


def harness_untrained(summarise, seeds):
    """The pilot seeds on which the harness rule `rules.workspace_trained(workspace ID, no_comm ID, trained_margin)` is 0 (the guard's definition)."""
    import tomllib
    margin = tomllib.loads((Path(__file__).resolve().parents[3] / "experiments/consciousness/C001-global-workspace/config.toml").read_text(encoding="utf-8"))["pinned_thresholds"]["trained_margin_permille"]
    ids = summarise.id_table()
    return [seed for seed in seeds if rules.workspace_trained(ids[seed]["workspace"], ids[seed]["no_comm"], margin) == 0], ids, margin


def main() -> None:
    t_support, t_refute = t_quantile(0.95, N - 1), t_quantile(1 - 0.05 / 3, N - 1)
    print(f"n = {N}, margin = {MARGIN}, t_support = {t_support:.6f}, t_refute = {t_refute:.6f}")
    print()
    print("Design-stage estimate (round 3-5: independent normal paired differences, pilot means 31.4 / 46.8 / 89.2, sds 24.4 / 42.0 / 64.4, no INVALID branch):")
    print(f"{'scenario':52s} P(COMPLETED)  P(NOT_SHOWN)  P(FALSIFIED)")
    for name, means in scenarios():
        supported, refuted, not_shown = decision_probabilities(
            means, PILOT_SDS, N, MARGIN, t_support, t_refute, len(means), trials=20000, seed=0
        )
        print(f"{name:52s} {supported:12.3f}  {not_shown:12.3f}  {refuted:12.3f}")

    table, summarise = load_pilot_table()
    tuning, dry, fresh = summarise.TUNING_SEEDS, summarise.DRY_SEEDS, tuple(range(1011, 1021))
    pooled = tuning + dry + fresh
    groups = {"tuning seeds 1004-1008": tuning, "dry seeds 1009-1010": dry, "fresh seeds 1011-1020": fresh, "all pilot seeds": pooled}
    print()
    print("Pilot per-seed quantities at the pinned cells (mean / sd / n; per mille of 16-module accuracy; 'trained' = the workspace trained):")
    print(f"{'seeds':34s} " + " ".join(f"{q:>16s}" for q in QUANTITIES))
    stats = {}
    for label, seeds in groups.items():
        for kind, trained in (("", None), (" (trained)", True)):
            rows = pilot_vectors(table, seeds, trained)
            if len(rows) < 2 or (trained and len(rows) == len(pilot_vectors(table, seeds, None))):
                continue
            mean, cov = covariance(rows)
            stats[label + kind] = (mean, cov, len(rows))
            sds = [math.sqrt(cov[j][j]) for j in range(5)]
            print(f"{label + kind:34s} " + " ".join(f"{mean[j]:6.1f}/{sds[j]:4.1f}/{len(rows):2d}" for j in range(5)))
    failures = pilot_vectors(table, pooled, False)
    n_all = len(pilot_vectors(table, pooled, None))
    p_fail = len(failures) / n_all
    fail_contrasts = tuple(sum(r[j] for r in failures) / len(failures) for j in range(3)) if failures else (0.0, 0.0, 0.0)
    counted, ids, trained_margin = harness_untrained(summarise, pooled)
    print(f"\nStalled workspace (16-module accuracy below {STALL_BELOW}): {len(failures)} of {n_all} pilot seeds; its contrasts there: "
          f"{tuple(round(v) for v in fail_contrasts)}; p_stall = {p_fail:.3f}")
    print(f"Harness rule (workspace ID >= no_comm ID + {trained_margin}) flags as untrained: {counted} of {len(pooled)} pilot seeds (the guard counts these).")
    for seed in (1011,):
        print(f"  seed {seed}: workspace ID {ids[seed]['workspace']}, no_comm ID {ids[seed]['no_comm']}, difference {ids[seed]['workspace'] - ids[seed]['no_comm']} (the rule needs {trained_margin}): not flagged")
    fail_counted = bool(counted)
    from math import comb
    print("Tails of the guard (P(more than k of 20 seeds flagged)) at flag rates 0.059 (1/17), 0.12, 0.18 (upper 95% end for 0 of 17), 0.29:")
    for k in (2, 4):
        print(f"  k = {k}: " + "  ".join(f"{q:.2f}: {sum(comb(20, m) * q**m * (1 - q)**(20 - m) for m in range(k + 1, 21)):.4f}" for q in (1 / 17, 0.12, 0.18, 0.29)))
    print()
    print("Graded control over 20 seeds, P(the lower one-sided 95% bound does not exceed 15):")
    for label, seeds in groups.items():
        rows = [r[3] for r in pilot_vectors(table, seeds, None)]
        if len(rows) >= 5:
            m = sum(rows) / len(rows)
            sd = math.sqrt(sum((x - m) ** 2 for x in rows) / (len(rows) - 1))
            print(f"  {label:34s} n {len(rows):2d} mean {m:6.1f} sd {sd:5.1f}  normal {graded_control_failure(m, sd, N, 15, t_support):.4f}  bootstrap {bootstrap_graded_control_failure(rows, N, 15, t_support):.4f}")
    trained_mean, trained_cov, _ = stats["all pilot seeds (trained)"]
    print()
    print("Outcome probabilities (INVALID = graded or gross control unresolved, more than 4 seeds flagged by the harness rule rules.workspace_trained, more than 2 unconverged seeds).")
    print("The contrast means are those of seeds on which the workspace trained; covariance of the trained seeds; seeds on which the workspace stalls")
    print("are drawn with probability p_stall and carry the contrasts of the stalled pilot seed (see variants S and G below).")
    print(f"{'scenario':50s} {'var':>4s} {'p_stall':>6s}  P(COMPLETED)  P(NOT_SHOWN)  P(FALSIFIED)  P(INVALID)")
    controls = trained_mean[3:]
    cases = [
        ("tuning-seed pilot means (5 seeds)", stats["tuning seeds 1004-1008"][0][:3], stats["tuning seeds 1004-1008"][0][3:]),
        ("fresh-seed means 1011-1020 (trained seeds)", stats["fresh seeds 1011-1020 (trained)"][0][:3], stats["fresh seeds 1011-1020 (trained)"][0][3:]),
        ("means pooled over all pilot seeds (trained)", trained_mean[:3], controls),
        ("zero effect", (0.0, 0.0, 0.0), controls),
        ("effect = margin (30 everywhere)", (30.0, 30.0, 30.0), controls),
        ("effect 45 everywhere", (45.0, 45.0, 45.0), controls),
        ("effect 80 everywhere", (80.0, 80.0, 80.0), controls),
    ]
    print("variant S = stall seeds drawn with probability p_stall and NOT counted by the guard (what the harness rule does to a seed like 1011);")
    print("variant G = the same stall seeds counted by the guard (round 7 table, if such seeds fell under the rule).")
    for name, contrasts, ctl in cases:
        for variant, pf, counted_flag in (("none", 0.0, False), ("S", p_fail, fail_counted), ("G", p_fail, True)):
            outcome = decision_outcomes(list(contrasts) + list(ctl), trained_cov, N, MARGIN, t_support, t_refute, 15, 50,
                                        p_workspace_fail=pf, fail_contrasts=fail_contrasts, fail_counted=counted_flag, trials=20000, seed=0)
            print(f"{name:50s} {variant:>4s} {pf:6.3f}  {outcome[0]:12.3f}  {outcome[1]:12.3f}  {outcome[2]:12.3f}  {outcome[3]:10.3f}")
    print()
    print("Sensitivity to the stall rate (pooled trained means), stalls not counted (S) and counted (G) by the guard, and to unconverged guarded arms:")
    for label, pf, pu, cf in (("S", 0.0, 0.0, False), ("S", 0.03, 0.0, False), ("S", p_fail, 0.0, False), ("S", 0.12, 0.0, False), ("G", 0.03, 0.0, True),
                              ("G", p_fail, 0.0, True), ("G", 0.12, 0.0, True), ("S", p_fail, 0.02, False)):
        outcome = decision_outcomes(list(trained_mean), trained_cov, N, MARGIN, t_support, t_refute, 15, 50, p_unconverged=pu,
                                    p_workspace_fail=pf, fail_contrasts=fail_contrasts, fail_counted=cf, trials=20000, seed=0)
        print(f"  {label} p_stall {pf:5.3f}, p_unconverged {pu:4.2f}: {outcome[0]:12.3f}  {outcome[1]:12.3f}  {outcome[2]:12.3f}  {outcome[3]:10.3f}")


if __name__ == "__main__":
    main()
