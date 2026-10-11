"""C007 planning model: the margin and the number of seeds, derived BEFORE any pilot from stated assumptions (standard library only).

    python research/cseries/c002a/planning.py

Chronology (disclosed in the preregistration, section 6): this file was written, and the margin and the seed count pinned in config.toml
from its output, before the first pilot run of C007. (Before it, the harness was smoke-tested once at a tiny scale, 300 classifier steps
and 200 head steps on pilot seed 1001, to see that it runs; those numbers were looked at and played no part here.) The pilots (seeds 1001
and up) tune the heads' learning rate and the size of the graded noise control and check that the arms train; they do not move the margin,
the seed count or the planning assumptions, which stay as written here. What the pilots then say about the effect is reported as a second
power table.

Planning assumptions (priors, not measurements of this experiment):
* the Brier score of one predictor of the classifier's errors on the shifted split (4000 inputs) is about 0.2 (error rate about a third); its
  between-seed (world, classifier and training) standard deviation is 3 per mille of Brier, and two predictors trained on the same data from
  the same initial weights are correlated 0.5.
* evaluation noise of a paired Brier difference on 4000 inputs: at most 1 per mille per arm.
* so the standard deviation of a per-seed paired difference is sigma_d = sqrt(2 (1 - 0.5) 3^2 + 2 1^2) = 3.3 per mille.
* margin rule: the improvement that justifies a separate self-model (a second network, a trace channel and a second training objective) over a
  probe that already reads the same trace is at least 2 per cent of the Brier score (about 4 per mille), and the margin must not sit inside the
  seed noise: margin >= 1.2 sigma_d = 4.0. Both give 4 per mille.
* the planning alternative (an effect worth building for): the self-model 8 per mille below each comparator (4 per cent of the Brier score). The
  planning null: no effect.
* seeds: the smallest n whose power at the alternative is at least 0.9 for COMPLETED and whose refutation power at the null is at least 0.8
  for FALSIFIED (kind refuted), with the false-support rate at the null below 0.01 (10 seeds). The priors are uncertain, so the rule asks in
  addition that the power at the alternative stay at least 0.8 if the true standard deviation is twice the assumed one; a modest effect makes
  that unreachable within the cap (0.69 at 20 seeds). So the seed count is the cap, 20: a run costs minutes, and more seeds are the only lever on
  the robustness column. The expected outcome probabilities are stated honestly in the preregistration: with the sd as assumed COMPLETED is
  likely at the alternative, with twice the sd it is not, and (the stronger statement) COMPLETED needs a real effect, which the prior for a
  multi-task objective on a probe that reads the same trace does not make likely.

FIX ROUND (written after the review of the first version and after the first pilots, which had shown both heads 5 to 18 per mille of Brier worse than the
model's own Platt-calibrated confidence; the margin, the seed count and the sampling model above are unchanged). Two derivations are added, both from the
sampling model of a logistic-type predictor and not from any pilot number:

* head-gap tolerance (guard `head_gap_id_permille <= 4`). A predictor with k effective free parameters fitted on n labelled examples of an event of probability
  p has an expected excess Brier score of about k p (1 - p) / n over the best predictor in its class (its variance: the Fisher information is p (1 - p) per
  parameter). For the model's own two-parameter Platt scaling on the 3000 calibration and validation examples (p about 0.35) that is 0.15 per mille; for an
  unconstrained 90-parameter head on 2000 examples it is about 10 per mille. A head that is "adequately tuned" against the Platt baseline therefore has no
  more than a few tens of effective extra parameters: the tolerance is 4 per mille (about 35 extra effective parameters at n = 2000), and a head further above
  the baseline than that on the in-distribution test split is too weak for the question (INVALID).
* signal floor (guard `signal_permille >= 4`). The per-seed requirement that the best trace predictor beats the constant predictor on the shifted split by at
  least the margin (4 per mille) is the claim that the traces carry at least the information of an effect the experiment could detect; the gross control carries the
  same requirement at the aggregate. Planning prior for the signal: mean 25 per mille, between-seed sd 10 (the information a calibrated confidence alone carries
  about errors of a classifier that errs on a third of a shifted split), so a seed falls below 4 with probability about 0.02 and some seed of 20 with probability
  about 0.3 *if the prior were exact*; the guard is therefore a validity condition that can fire by chance on a weak world, which is why the harness reports the
  signal of every pilot seed and the preregistration states the pilot probability. Not loosened after any pilot.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from stats import bounds_of, t_quantile  # noqa: E402

SIGMA_TRAIN, RHO, SIGMA_EVAL = 3.0, 0.5, 1.0
HEAD_GAP_TOLERANCE_PERMILLE = 4
SIGNAL_FLOOR_PERMILLE = 4


def excess_brier_permille(effective_parameters: float, examples: int, event_probability: float) -> float:
    """Expected excess Brier score (per mille) of a predictor with `effective_parameters` free parameters fitted on `examples` examples of an event with
    the given probability: k p (1 - p) / n."""
    return 1000 * effective_parameters * event_probability * (1 - event_probability) / examples


def signal_below_floor_probability(mean: float = 25.0, sd: float = 10.0, floor: float = SIGNAL_FLOOR_PERMILLE) -> float:
    """P(signal < floor) for a normal signal prior."""
    return 0.5 * (1 + math.erf((floor - mean) / (sd * math.sqrt(2))))
SIGMA_D = math.sqrt(2 * (1 - RHO) * SIGMA_TRAIN**2 + 2 * SIGMA_EVAL**2)
MARGIN = 4
ALTERNATIVE, NULL = 8.0, 0.0
COMPARATORS = 2  # the information-matched probe and the model's own (calibrated) confidence


def outcome_probabilities(means, n, margin=MARGIN, sigma=SIGMA_D, trials=20000, seed=0):
    """Monte-Carlo (COMPLETED, NOT_SHOWN, FALSIFIED) for independent normal paired differences against each comparator
    (`means` per comparator), the decision rule of the aggregator: COMPLETED iff every one-sided 95% lower bound reaches the margin,
    FALSIFIED iff some Bonferroni upper bound (alpha / 2) is below it. Guards and controls are not modelled here."""
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


def main() -> None:
    print(f"excess Brier of a 2-parameter Platt fit on 3000 examples (p 0.35): {excess_brier_permille(2, 3000, 0.35):.2f} per mille; of a 90-parameter head on 2000: "
          f"{excess_brier_permille(90, 2000, 0.35):.1f}; effective extra parameters the tolerance of {HEAD_GAP_TOLERANCE_PERMILLE} allows at n = 2000: "
          f"{HEAD_GAP_TOLERANCE_PERMILLE * 2000 / (1000 * 0.35 * 0.65):.0f}")
    low = signal_below_floor_probability()
    print(f"signal prior N(25, 10): P(seed below the floor of {SIGNAL_FLOOR_PERMILLE}) = {low:.3f}, P(some seed of 20 below) = {1 - (1 - low) ** 20:.2f}")
    print(f"sigma_d = {SIGMA_D:.1f}; margin rule 1.2 sigma_d = {1.2 * SIGMA_D:.1f}; margin = {MARGIN}")
    print(f"{'n':>3s} {'alt: COMPLETED':>15s} {'null: COMPLETED':>16s} {'null: FALSIFIED':>16s} {'margin: NOT_SHOWN':>18s} {'alt, 2x sd':>11s}")
    chosen = None
    for n in (5, 6, 8, 10, 12, 14, 16, 20):
        alt = outcome_probabilities((ALTERNATIVE,) * COMPARATORS, n)
        null = outcome_probabilities((NULL,) * COMPARATORS, n)
        edge = outcome_probabilities((float(MARGIN),) * COMPARATORS, n)
        wide = outcome_probabilities((ALTERNATIVE,) * COMPARATORS, n, sigma=2 * SIGMA_D)
        ok = alt[0] >= 0.9 and null[2] >= 0.8 and null[0] < 0.01
        robust = ok and wide[0] >= 0.8
        chosen = chosen or (n if ok else None)
        note = "  <- smallest n meeting the first three conditions" if chosen == n and ok else ""
        print(f"{n:3d} {alt[0]:15.3f} {null[0]:16.3f} {null[2]:16.3f} {edge[1]:18.3f} {wide[0]:11.3f}{note}{'  (also robust)' if robust else ''}")
    print(f"smallest n meeting the first three conditions: {chosen}; the robustness condition (power >= 0.8 at twice the sd) is met by no n <= 20, so the seed count is the cap, 20")


if __name__ == "__main__":
    main()
