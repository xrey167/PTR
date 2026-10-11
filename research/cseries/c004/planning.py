"""C004 planning model: the margin and the number of seeds, derived BEFORE any pilot from stated assumptions (standard library only).

    python research/cseries/c004/planning.py

Chronology (disclosed in the preregistration, section 6): this file was written, and the margin and the seed count pinned in
config.toml from its output, **after one development run of the harness on pilot seed 1001** (the first task design, a word problem in the
group S3, on which the feed-forward stack did not learn; those numbers were seen) and before any other pilot. The planning model is task-agnostic
(an accuracy margin and a between-seed standard deviation taken from the range of C001's contrasts) and was not changed afterwards. The pilots (seeds
1001 and up) tune learning rates (and, in the fix round, the stack's depth) on the primary metric for every arm and check that the arms train; they do
not move the margin, the seed count or the planning assumptions, which stay as written here. Whatever the pilots then say about the effect is reported as a
second power table next to this one.

Planning assumptions (priors, not measurements of this experiment):
* per-seed accuracy of one arm at the primary length has a between-seed (training) standard deviation of 70 per mille: the
  paired-difference standard deviations of the sister experiment C001 ranged from 24 to 168 per mille across contrasts; 70 is its
  middle. Two arms on one seed share the data and the schedule: correlation 0.3.
* evaluation noise: binomial on 2048 sequences per arm, at most 11 per mille.
* so the standard deviation of a per-seed paired difference is sigma_d = sqrt(2 (1 - 0.3) 70^2 + 2 11^2) = 84 per mille.
* margin rule: the advantage that justifies a carried recurrent state is a cost judgement (a sequential, non-parallel computation
  in training; a 100 per mille gain at twice the trained length is the smallest we would pay that for), and the margin must not
  sit inside the seed noise: margin >= 1.2 sigma_d = 101. Both give 100 per mille.
* the planning alternative (an effect worth building for): the recurrent arm 250 per mille above each comparator at the primary
  length (for instance 0.85 against 0.60). The planning null: no effect (0).
* seeds: the smallest n whose power at the alternative is at least 0.9 for COMPLETED and whose refutation power at the null is at
  least 0.8 for FALSIFIED (kind refuted), with the false-support rate at the null below 0.01, AND which keeps the power at the
  alternative at least 0.8 if the true standard deviation is twice the assumed one (the priors are uncertain); n at most 20 (the
  CPU budget).

FIX ROUND (written after the pilots of the first version, after the independent review that found that a competent comparator could not yield a refutation).
The margin (100), the seed count (14) and the sampling model are unchanged; what is added is a table that discriminates: the outcome probabilities when the
feed-forward stack is competent (its 12-step accuracy `a` while the recurrent arm sits at 975, so that the paired mean is 975 - a, the reset-state ablation staying
far below), from a stack at the margin to one that extrapolates. With this table the experiment has a refutation path that the first version's guard (a ceiling of 800)
closed: the saturation guard is now a ceiling of 980 (a stack at or above it leaves no room for the margin and the experiment is uninformative).
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from stats import bounds_of, t_quantile  # noqa: E402

SIGMA_TRAIN, RHO, SIGMA_EVAL = 70.0, 0.3, 11.0
SIGMA_D = math.sqrt(2 * (1 - RHO) * SIGMA_TRAIN**2 + 2 * SIGMA_EVAL**2)
MARGIN = 100
ALTERNATIVE, NULL = 250.0, 0.0
COMPARATORS = 2  # the feed-forward stack and the reset-state ablation


def outcome_probabilities(means, n, margin=MARGIN, sigma=SIGMA_D, trials=20000, seed=0):
    """Monte-Carlo (COMPLETED, NOT_SHOWN, FALSIFIED) for independent normal paired differences against each comparator
    (`means` per comparator), the decision rule of the aggregator: COMPLETED iff every one-sided 95% lower bound reaches the margin,
    FALSIFIED iff some Bonferroni upper bound (alpha / 2) is below it. Guards and controls are not modelled here, and what they do in each branch of
    the decision is: the saturation guard and the sensitivity and null controls make a seed set INVALID before the contrast is read; the extrapolation-removed control
    is read ONLY when the contrast is supported (it can turn a COMPLETED into INVALID, never a FALSIFIED or NOT_SHOWN into anything), so the probabilities below
    for FALSIFIED and NOT_SHOWN are those of the aggregator, and the COMPLETED probability is an upper bound by the probability that the control holds. The
    ceiling of 980 and the control share of one half are not derived from this model."""
    rng = random.Random(seed)
    t_s, t_r = t_quantile(0.95, n - 1), t_quantile(1 - 0.05 / COMPARATORS, n - 1)
    counts = [0, 0, 0]
    for _ in range(trials):
        stats = [bounds_of([rng.gauss(m, sigma) for _ in range(n)], 1.0) for m in means]
        mean_se = [(mu, se) for mu, se in stats]
        if all(mu - t_s * se >= margin for mu, se in mean_se):
            counts[0] += 1
        elif any(mu + t_r * se < margin for mu, se in mean_se):
            counts[2] += 1
        else:
            counts[1] += 1
    return tuple(c / trials for c in counts)


def competent_baseline_rows(n: int = 14, recurrent: float = 975.0, reset: float = 175.0) -> list[tuple[float, tuple]]:
    """(stack accuracy, (COMPLETED, NOT_SHOWN, FALSIFIED)) when the stack reaches that accuracy at the primary length and the recurrent arm `recurrent`."""
    return [(a, outcome_probabilities((recurrent - a, recurrent - reset), n)) for a in (980.0, 940.0, 900.0, 875.0, 850.0, 800.0, 600.0, 250.0)]


def main() -> None:
    print(f"sigma_d = {SIGMA_D:.1f}; margin rule 1.2 sigma_d = {1.2 * SIGMA_D:.1f}; margin = {MARGIN}")
    print(f"{'n':>3s} {'alt: COMPLETED':>15s} {'null: COMPLETED':>16s} {'null: FALSIFIED':>16s} {'margin: NOT_SHOWN':>18s} {'alt, 2x sd':>11s}")
    chosen = None
    for n in (5, 6, 8, 10, 12, 14, 16, 20):
        alt = outcome_probabilities((ALTERNATIVE,) * COMPARATORS, n)
        null = outcome_probabilities((NULL,) * COMPARATORS, n)
        edge = outcome_probabilities((float(MARGIN),) * COMPARATORS, n)
        wide = outcome_probabilities((ALTERNATIVE,) * COMPARATORS, n, sigma=2 * SIGMA_D)
        ok = alt[0] >= 0.9 and null[2] >= 0.8 and null[0] < 0.01 and wide[0] >= 0.8
        chosen = chosen or (n if ok else None)
        print(f"{n:3d} {alt[0]:15.3f} {null[0]:16.3f} {null[2]:16.3f} {edge[1]:18.3f} {wide[0]:11.3f}{'  <- smallest n meeting every condition' if chosen == n and ok else ''}")
    print(f"smallest n meeting the rules: {chosen}")
    print("a competent baseline (n = 14; recurrent 975, reset-state 175; the stack's 12-step accuracy in the first column): COMPLETED / NOT_SHOWN / FALSIFIED")
    for a, row in competent_baseline_rows():
        print(f"  stack {a:5.0f}: contrast {975 - a:5.0f}: {row[0]:.3f} / {row[1]:.3f} / {row[2]:.3f}")
    print("rows near the margin (n = 14, both comparators at the stated mean): COMPLETED / NOT_SHOWN / FALSIFIED")
    for mean in (0.0, 50.0, 75.0, 100.0, 125.0, 150.0, 250.0):
        row = outcome_probabilities((mean,) * COMPARATORS, 14)
        print(f"  mean {mean:5.0f}: {row[0]:.3f} / {row[1]:.3f} / {row[2]:.3f}")


if __name__ == "__main__":
    main()
