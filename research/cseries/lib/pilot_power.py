"""The second power table of a stage-A experiment: the paired contrasts of its decision rule measured on PILOT seeds (>= 1001, never a declared seed),
then the Monte-Carlo outcome probabilities of the preregistered rule at the preregistered number of seeds, with the pilot means and covariance as the
generating model (and with twice the covariance). The first power table (planning.py) is the one the margin and the seed count came from; this one only
says what the pilots suggest, and changes neither.

    python research/cseries/lib/pilot_power.py <ID> <pilot.json>...

Each file is the one-line JSON a harness prints for one seed. Standard library only. `--sign` of each experiment: C004 and C008 the arm minus each
comparator; C007 the comparator minus the arm (a Brier score: lower is better)."""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import agg  # noqa: E402
import stats  # noqa: E402

DIRS = {
    "C004": "experiments/consciousness/C004-recurrent-integration",
    "C007": "experiments/consciousness/C007-metacognitive-self-model-cpu",
    "C008": "experiments/consciousness/C008-attention-schema-routing-cpu",
}
REVERSED = {"C007"}


def contrasts(results: dict, seeds: list, paired: dict, reverse: bool) -> dict[str, list[float]]:
    out = {}
    for other in paired["against"]:
        left, right = (other, paired["arm"]) if reverse else (paired["arm"], other)
        out[other] = [float(d * 1000) for d in agg.differences(results, seeds, paired["metric"], left, right)]
    return out


def guard_specs(experiment: str, paired: dict) -> list[tuple[str, str, str, float]]:
    """(name, left, right, threshold) of the sensitivity controls whose lower bound must reach a threshold for the run to be valid."""
    if experiment == "C004":
        return [("gross (recurrent - permuted labels)", paired["arm"], paired["sensitivity_gross"]["permuted"], paired["sensitivity_gross"]["margin_permille"]),
                ("graded (recurrent - corrupted input)", paired["arm"], paired["sensitivity_graded"]["corrupted"], paired["sensitivity_graded"]["min_lower_permille"])]
    if experiment == "C007":
        return [("gross (constant - self-model)", paired["sensitivity_gross"]["floor"], paired["arm"], paired["sensitivity_gross"]["margin_permille"]),
                ("graded (noised - self-model)", paired["sensitivity_graded"]["degraded"], paired["arm"], paired["sensitivity_graded"]["min_lower_permille"])]
    return [("gross (schema-free - random router)", paired["sensitivity_gross"]["arm"], paired["sensitivity_gross"]["floor"], paired["sensitivity_gross"]["margin_permille"]),
            ("graded (schema-free - epsilon routing)", paired["sensitivity_graded"]["arm"], paired["sensitivity_graded"]["degraded"], paired["sensitivity_graded"]["min_lower_permille"])]


def null_specs(experiment: str, paired: dict) -> list[tuple[str, str, str, float]]:
    """(name, arm, against, margin) of the null worlds: neither direction's lower bound may reach the margin, else the run is INVALID."""
    if experiment == "C004":
        return [("null (permuted labels)", paired["null_permuted"]["arm"], paired["null_permuted"]["against"], paired["null_permuted"]["margin_permille"]),
                ("null (last two steps)", paired["null_window"]["arm"], paired["null_window"]["against"], paired["null_window"]["margin_permille"])]
    null = paired["null_world"]
    return [("null world", null["arm"], null["against"] if "against" in null else null["arm"], null["margin_permille"])]


def extra_specs(experiment: str, paired: dict) -> list[tuple[str, str, str, str, str, float]]:
    """(name, left, right, metric, kind, threshold) of the further bounds the decision reads: kind "upper<" (the upper bound of left - right must stay below the
    threshold), "lower>=" (the lower bound must reach it), "lower<" (the lower bound must stay below it)."""
    if experiment == "C007":
        out = [(f"removal ({arm})", paired["removal"]["reference"], arm, paired["metric"], "upper<", float(paired["removal_margin_permille"])) for arm in paired["removal"]["arms"]]
        out.append(("removal against the probe", paired["removal_vs_probe"]["against"], paired["removal_vs_probe"]["arm"], paired["metric"], "upper<", float(paired["removal_margin_permille"])))
        return out
    if experiment == "C004":
        return []
    null = paired["null_world"]
    return [("null world cost (schema-free - control, stationary)", null["against"], null["arm"], null["metric"], "upper<", float(null["cost_band_permille"])),
            ("attribution (control - shuffled history)", paired["arm"], paired["attribution_arm"], paired["metric"], "lower>=", float(paired["margin_permille"]))]


def guard_rates(experiment: str, results: dict, criteria: dict) -> list[str]:
    """The per-seed guards of the criteria evaluated on the pilot results: how many pilot seeds fail each, the per-seed failure probability under a normal fit of the
    guard's scalar (mean and standard deviation over the pilot seeds; 0 or 1 when the scalar is constant), and the probability that at least one of the preregistered
    seeds fails it (which makes the run INVALID)."""
    import math

    lines = []
    n = int(criteria["paired"]["seeds_required"])
    for rule in criteria["guards"]:
        left, op, right = agg.parse_rule(rule)
        values = [float(agg.resolve(r, left)) for r in results.values()]
        failed = sum(1 for r in results.values() if not agg.evaluate(r, rule))
        m = mean(values)
        spread = (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5 if len(values) > 1 else 0.0
        threshold = float(right)
        if spread == 0:
            p = float(failed > 0)
        else:
            z = (threshold - m) / spread
            below = 0.5 * (1 + math.erf(z / math.sqrt(2)))
            p = below if op == ">=" else 1 - below
        lines.append(f"  guard {rule!r}: scalar mean {m:.1f} sd {spread:.1f}, fails on {failed} of {len(results)} pilot seeds; normal-fit per-seed failure probability {p:.4f}, P(some of {n} seeds fails) = {1 - (1 - p) ** n:.3f}")
    return lines


def mean(xs):
    return sum(xs) / len(xs)


def main(argv: list[str]) -> int:
    experiment, files = argv[0], argv[1:]
    criteria = tomllib.loads((ROOT / DIRS[experiment] / "criteria.toml").read_text(encoding="utf-8"))
    paired = criteria["paired"]
    results = {}
    for path in files:
        line = [x for x in Path(path).read_text(encoding="utf-8").splitlines() if x.startswith("{")][-1]
        record = json.loads(line)
        results[int(record["seed"])] = record
    seeds = sorted(results)
    assert all(s >= 1001 for s in seeds), "pilot seeds only"
    table = contrasts(results, seeds, paired, experiment in REVERSED)
    names = list(table)
    k = len(seeds)
    print(f"{experiment}: {k} pilot seeds {seeds}; margin {paired['margin_permille']} per mille; preregistered seeds {paired['seeds_required']}")
    for name in names:
        xs = table[name]
        sd = (sum((x - mean(xs)) ** 2 for x in xs) / (k - 1)) ** 0.5 if k > 1 else float("nan")
        print(f"  contrast vs {name:18s} mean {mean(xs):8.1f}  sd {sd:6.1f}  per-seed {[round(x, 1) for x in xs]}")
    means, cov = stats.covariance([list(row) for row in zip(*[table[n] for n in names])])
    n = int(paired["seeds_required"])
    margin = float(paired["margin_permille"])
    for label, scale in (("pilot covariance", 1.0), ("twice the pilot standard deviation", 4.0)):
        scaled = [[c * scale + (0.0) for j, c in enumerate(row)] for i, row in enumerate(cov)]
        probs = stats.outcome_probabilities(means, scaled, n, margin, float(paired["t_support"]), float(paired["t_refute"]), trials=20000, seed=7)
        print(f"  {label:36s} COMPLETED {probs[0]:.3f}  NOT_SHOWN {probs[1]:.3f}  FALSIFIED {probs[2]:.3f}  (guards and controls assumed to hold)")
    import random

    rng = random.Random(11)
    t = float(paired["t_support"])
    for name, left, right, threshold in guard_specs(experiment, criteria["paired"]):
        xs = [float(d * 1000) for d in agg.differences(results, seeds, paired["metric"], left, right)]
        m = mean(xs)
        sd = (sum((x - m) ** 2 for x in xs) / (k - 1)) ** 0.5
        passes = 0
        for _ in range(20000):
            draw = [rng.gauss(m, sd) for _ in range(n)]
            mm = mean(draw)
            se = (sum((x - mm) ** 2 for x in draw) / (n - 1)) ** 0.5 / n**0.5
            passes += mm - t * se >= threshold
        print(f"  control {name:42s} pilot mean {m:7.1f} sd {sd:6.1f} threshold {threshold}: resolves at n={n} with probability {passes / 20000:.3f}")
    for name, arm, against, margin_null in null_specs(experiment, criteria["paired"]):
        for direction, (left, right) in (("forward", (arm, against)), ("reverse", (against, arm))):
            xs = [float(d * 1000) for d in agg.differences(results, seeds, paired["null_world"]["metric"] if "null_world" in paired else paired["metric"], left, right)]
            m = mean(xs)
            sd = (sum((x - m) ** 2 for x in xs) / (k - 1)) ** 0.5
            clean = 0
            for _ in range(20000):
                draw = [rng.gauss(m, sd) for _ in range(n)]
                mm = mean(draw)
                se = (sum((x - mm) ** 2 for x in draw) / (n - 1)) ** 0.5 / n**0.5
                clean += mm - t * se < margin_null
            print(f"  {name:22s} {direction:7s} pilot mean {m:7.1f} sd {sd:6.1f}: lower bound stays below {margin_null} with probability {clean / 20000:.3f}")
    if experiment == "C004":
        removed = criteria["paired"]["extrapolation_removed"]
        share = removed["share_permille"] / 1000
        recovered = agg.differences(results, seeds, removed["metric"], removed["trained_on_length"], removed["stack"])
        advantage = agg.differences(results, seeds, removed["metric"], removed["arm"], removed["stack"])
        xs = [float((r - share * a) * 1000) for r, a in zip(recovered, advantage)]
        m = mean(xs)
        sd = (sum((x - m) ** 2 for x in xs) / (k - 1)) ** 0.5
        ok = 0
        for _ in range(20000):
            draw = [rng.gauss(m, sd) for _ in range(n)]
            mm = mean(draw)
            ok += mm - t * (sum((x - mm) ** 2 for x in draw) / (n - 1)) ** 0.5 / n**0.5 >= 0
        print(f"  check extrapolation removed (recovered - {removed['share_permille']}/1000 advantage)  pilot mean {m:7.1f} sd {sd:6.1f} (lower>= 0): holds at n={n} with probability {ok / 20000:.3f}")
    for name, left, right, metric, kind, threshold in extra_specs(experiment, criteria["paired"]):
        xs = [float(d * 1000) for d in agg.differences(results, seeds, metric, left, right)]
        m = mean(xs)
        sd = (sum((x - m) ** 2 for x in xs) / (k - 1)) ** 0.5
        ok = 0
        for _ in range(20000):
            draw = [rng.gauss(m, sd) for _ in range(n)]
            mm = mean(draw)
            half = t * (sum((x - mm) ** 2 for x in draw) / (n - 1)) ** 0.5 / n**0.5
            ok += (mm + half < threshold) if kind == "upper<" else (mm - half >= threshold) if kind == "lower>=" else (mm - half < threshold)
        print(f"  check {name:58s} pilot mean {m:7.1f} sd {sd:6.1f} ({kind} {threshold:g}): holds at n={n} with probability {ok / 20000:.3f}")
    for line in guard_rates(experiment, results, criteria):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
