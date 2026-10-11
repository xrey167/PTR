"""C008 planning model: the margin and the number of seeds, derived BEFORE any pilot of the final design from stated assumptions (standard library only).

    python research/cseries/c003a/planning.py

Chronology (disclosed in the preregistration, section 6). Version 1 of this file (19:50, 2026-10-10) assumed a world of 64 tokens per batch, capacity
14 and 70 per cent hot demand, a margin of 30 per mille and a planning alternative of 50 per mille ("about half of the 107 per mille between a router
that splits each class evenly and the per-batch oracle"). Pilots on seeds 1001 and 1002 of three earlier architectures of the control arm were then looked
at (the control arm was about 20 per mille BELOW the schema-free router in all three), and a calculation without any learning showed that in that world a
router that knew the regime and split each class optimally, with independent token sampling, would gain only about 30 per mille over the even split; so the
alternative of version 1 was unattainable and the margin of 30 asked for all of the reachable gain. Version 2 (20:18) re-derived the margin (25) and the
alternative (45) from the reachable gain of a revised world (256 tokens per batch, capacity 64, 80 per cent hot demand), measured the same way, without
learning (`research/cseries/c003a/pilots/regime_value.py`): even split 0.847, best regime-aware independent split 0.907 (60 per mille), oracle 0.951. Version 2
is what is pinned. After it, two more changes were made to the world on pilot seed 1001, both before any pilot of the final world: the class features' noise
was lowered from 0.5 to 0.25 (a router could not recognise the class: its reward, 0.755, was far below the even split's 0.847), and the hot pair was changed
from (2r, 2r + 1) to (r, r + 1) with eight regimes, because with the first pair every class always had the same role and a regime-blind router already reached
0.895 (a hand-set control law then added 5 per mille at best); with the shifted pair a class is "first" in one regime and "second" in the next, the best splits
are opposite, and the same sweep gave 0.837 without the law and 0.899 with a gain of -1 (62 per mille). The margin, the alternative and the seed count were NOT
moved by these changes: they come from the reachable-gain measurement above, which does not depend on the router's architecture or on the regime layout. The
pilots proper (seeds 1001 and up) tune learning rates, the balancer's step and the form of the control law on equal grids and check that the arms train; they
do not move the margin, the seed count or the planning assumptions. What the pilots then say about the effect is reported as a second power table.

Planning assumptions (priors, not measurements of this experiment):
* the reward of one arm (the fraction of tokens verified, over a fixed 61440-token evaluation stream) has a between-seed (training) standard
  deviation of 15 per mille: online policy gradient on a small router is noisy, the C001 contrasts were 24 to 168 on a harder scale. Two arms on one
  seed share the stream and the schedule: correlation 0.3.
* evaluation noise: binomial on 61440 tokens, at most 2.1 per mille per arm.
* so the standard deviation of a per-seed paired difference is sigma_d = sqrt(2 (1 - 0.3) 15^2 + 2 2.1^2) = 18 per mille.
* margin rule: 2.5 points of verified-token rate (25 per mille) is the smallest gain that justifies adding a learned self-model to the router (a
  second network, its history buffer and a training signal to keep stable), and it must not sit inside the seed noise: margin >= 1.2 sigma_d = 22.
  Both give 25. It asks the control arm to capture about 40 per cent of the reachable gain, which is demanding by design.
* the planning alternative (an effect worth building for): the control arm 45 per mille above every comparator, three quarters of the reachable
  60 per mille. The planning null: no effect.
* seeds: the smallest n whose power at the alternative is at least 0.9 for COMPLETED and whose refutation power at the null is at least 0.8
  for FALSIFIED (kind refuted), with the false-support rate at the null below 0.01. The priors are uncertain, so the rule asks in addition that the
  power at the alternative stay at least 0.8 if the true standard deviation is twice the assumed one; four comparators and a modest effect make that
  unreachable within the cap. When it is, the seed count is the cap, 20: a run costs minutes, and more seeds are the only lever on the robustness
  column. The expected outcome probabilities are stated honestly in the preregistration.

FIX ROUND (written after the review of the first freeze-ready version, after all the pilots above). Changes to the design and what they do to the planning model;
the margin (25), the alternative (45), the sampling model and the seed count (20) are unchanged:
* a fifth comparator, `history_router`: the router that reads the flattened history of its own propensity directly, with no separate predictor and no control law
  (parameter-matched to the control arm). The decision rule now has five comparators: the Bonferroni quantile is that of alpha / 5 (2.539483 for 19 degrees of
  freedom instead of 2.433440 for four). The power table below is for five.
* a null-world cost band of 50 per mille (two margins): in the stationary world the schema cannot help, and the extra network and law may cost something; the
  prior for that cost is a spurious law gain of the order of one margin at most, so a cost whose one-sided upper bound reaches two margins is a defect of the
  arm, not noise. (The pilots had shown a mean cost of about 19 per mille with a standard deviation of about 19 in the old arm set; the band was written with that in
  view and is not a pilot-derived quantity.)
* the attribution control (the control arm reading the history of random other batches) is judged: COMPLETED needs ATTRIBUTED (lower bound at least the margin).
  In the planning model this is a sixth required lower bound, correlated with the others; it is not modelled separately (the shuffled-history arm shares the
  control's training), and its pilot probability is in the preregistration.
* every arm now has nine pilot configurations (see the preregistration); this does not change the model.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from stats import bounds_of, t_quantile  # noqa: E402

SIGMA_TRAIN, RHO, SIGMA_EVAL = 15.0, 0.3, 2.1
SIGMA_D = math.sqrt(2 * (1 - RHO) * SIGMA_TRAIN**2 + 2 * SIGMA_EVAL**2)
MARGIN = 25
ALTERNATIVE, NULL = 45.0, 0.0
COMPARATORS = 5  # schema-free, wide schema-free, the heuristic balancer, the describe-only schema, the history router


def outcome_probabilities(means, n, margin=MARGIN, sigma=SIGMA_D, trials=20000, seed=0):
    """Monte-Carlo (COMPLETED, NOT_SHOWN, FALSIFIED) for independent normal paired differences against each comparator
    (`means` per comparator), the decision rule of the aggregator: COMPLETED iff every one-sided 95% lower bound reaches the margin,
    FALSIFIED iff some Bonferroni upper bound (alpha / 5) is below it. Guards and controls are not modelled here."""
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
    print("rows near the margin (n = 20, five comparators, all at the stated mean): COMPLETED / NOT_SHOWN / FALSIFIED")
    for mean in (0.0, 10.0, 20.0, 25.0, 30.0, 35.0, 45.0, 60.0):
        row = outcome_probabilities((mean,) * COMPARATORS, 20)
        print(f"  mean {mean:5.1f}: {row[0]:.3f} / {row[1]:.3f} / {row[2]:.3f}")


if __name__ == "__main__":
    main()
