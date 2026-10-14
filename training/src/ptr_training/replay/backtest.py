"""Leave-one-generation-out backtest of an off-policy outcome model over a generation history.

The self-improvement cycle proposes a change of curriculum or adaptation policy and wants to
know, before training anything, how a candidate policy would score. The only data is the
history: per generation, the decisions taken and the per-family outcome. The model here is the
simplest one that history can support, an intercept plus one response coefficient per decision,
fitted from consecutive generations in which exactly one decision moved.

That model has as many free parameters as the history has clean transitions, so in sample it
reproduces its own fitting points and a small in-sample error says nothing. This module does
not let that pass as evidence:

* The pool is append-only and chronological. A generation that repeats a name or does not
  come after the last one is refused, because the response coefficients come from consecutive
  pairs and an out-of-order insert would make a later run explain an earlier one.
* The backtest holds out each generation in turn and predicts it from the others. A generation
  is predicted only if the remaining history identifies every decision it uses; otherwise it
  is skipped with the reason (``not_identified``), never counted.
* ``degenerate`` is set when the model has at least as many parameters as the history has
  generations, which is when the in-sample fit is arithmetic.
* A policy that asks for a level of a decision the history never showed is flagged
  ``extrapolation``. A winner that extrapolates aborts the selection.
* ``predictive`` is true only if at least ``min_within_bound`` generations were predicted out of
  sample with every family within the pinned error bound, and those are at least
  ``min_within_fraction`` of the generations that could be predicted at all. Both numbers are
  pinned with the error bound before the first run; a single lucky generation among many bad
  ones is not predictive. ``pipeline_ran`` says only that the backtest completed; it is a
  different claim and is reported separately. No policy is selected from a history that is not
  ``predictive``.

Dreams are estimates, never evidence: a real training run is the only thing that measures an
outcome.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

MAX_GENERATIONS = 256
MAX_DECISIONS = 64
MAX_FAMILIES = 16
MAX_NAME_CHARS = 128

Level = bool | int | float | str


class HistoryError(ValueError):
    """The history pool refused a generation; nothing was added."""


class ExtrapolationAbort(RuntimeError):
    """The best-scored policy asks for decision levels the history never showed."""


class NotPredictive(RuntimeError):
    """The history did not predict any held-out generation within the error bound."""


@dataclass(frozen=True)
class Outcome:
    """Correct and total cases of one family in one generation."""

    correct: int
    total: int

    def __post_init__(self) -> None:
        for field_name in ("correct", "total"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise HistoryError(f"outcome {field_name} must be an int")
        if self.total < 1:
            raise HistoryError("outcome total must be at least 1: a rate of no cases is undefined")
        if not 0 <= self.correct <= self.total:
            raise HistoryError("outcome correct must lie between 0 and total")

    @property
    def rate(self) -> float:
        return self.correct / self.total


@dataclass(frozen=True)
class Generation:
    name: str
    order: int
    decisions: Mapping[str, Level]
    outcome: Mapping[str, Outcome]
    levels: Mapping[str, float] = field(default_factory=dict)


def levels_of(
    decisions: Mapping[str, Level],
    categories: dict[str, list[str]],
    zero_based: frozenset[str] | set[str] = frozenset(),
    *,
    register: bool = False,
) -> dict[str, float]:
    """Decision values as numbers, one number per decision.

    A string is one categorical decision, not one indicator per value. Its level is the position
    of the value among the values the pool has seen for that decision, in order of first
    appearance. If the decision is present in the pool's first generation (``zero_based``), the
    first value is the reference level 0, so an alternating history has a level-zero generation
    to anchor the intercept on. If the decision appears later, 0 is "not set" and the values
    start at 1, so introducing it is a visible move. Changing ``mode`` from "off" to "on" is one
    decision moving by one level, which the fit can attribute. A decision with more than two
    values is treated as linear in that order; that is a stated limit of the model.

    ``register`` adds a value the pool has not seen to ``categories``; without it, an unseen
    value gets the level one past the last known one, which is outside the observed range and
    is therefore flagged as extrapolation.
    """
    levels: dict[str, float] = {}
    for name, value in decisions.items():
        if isinstance(value, bool):
            levels[name] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            try:
                number = float(value)
            except OverflowError:
                # An int too large for a float is not a level; refuse it as any other bad value.
                raise HistoryError(f"decision {name!r} is not finite") from None
            if not math.isfinite(number):
                raise HistoryError(f"decision {name!r} is not finite")
            levels[name] = number
        elif isinstance(value, str):
            known = categories.setdefault(name, []) if register else categories.get(name, [])
            if value not in known:
                if register:
                    known.append(value)
            base = 0 if name in zero_based else 1
            position = known.index(value) if value in known else len(known)
            levels[name] = float(position + base)
        else:
            raise HistoryError(f"decision {name!r} has an unsupported type")
    return levels


def _check_name(kind: str, name: Any) -> None:
    if not isinstance(name, str) or not name or len(name) > MAX_NAME_CHARS:
        raise HistoryError(f"{kind} name must be a non-empty string of at most {MAX_NAME_CHARS}")


class HistoryPool:
    """Append-only, chronological pool of generations; refuses before it changes anything."""

    def __init__(self) -> None:
        self._generations: list[Generation] = []
        self._categories: dict[str, list[str]] = {}
        self._zero_based: set[str] = set()

    def zero_based(self) -> set[str]:
        """Categorical decisions present in the first generation (their first value is level 0)."""
        return set(self._zero_based)

    def categories(self) -> dict[str, list[str]]:
        """The values seen per categorical decision, in order of first appearance (a copy)."""
        return {name: list(values) for name, values in self._categories.items()}

    @property
    def generations(self) -> tuple[Generation, ...]:
        return tuple(self._generations)

    def families(self) -> tuple[str, ...]:
        return tuple(self._generations[0].outcome) if self._generations else ()

    def add_generation(
        self,
        name: str,
        *,
        order: int,
        decisions: Mapping[str, Level],
        outcome: Mapping[str, Outcome],
    ) -> None:
        """Append one generation.

        Raises ``HistoryError`` for a repeated name, an ``order`` that is not above the last
        one, a bound exceeded, a family set that differs from the pool's, or a malformed
        decision or outcome. The pool is unchanged after a refusal.
        """
        _check_name("generation", name)
        if isinstance(order, bool) or not isinstance(order, int):
            raise HistoryError("order must be an int")
        if len(self._generations) >= MAX_GENERATIONS:
            raise HistoryError(f"the pool holds {MAX_GENERATIONS} generations already")
        if any(existing.name == name for existing in self._generations):
            raise HistoryError(f"generation already in the pool: {name!r}")
        if self._generations and order <= self._generations[-1].order:
            raise HistoryError(
                f"generation {name!r} has order {order}, not after "
                f"{self._generations[-1].order}: the pool is append-only"
            )
        if len(decisions) > MAX_DECISIONS or not outcome or len(outcome) > MAX_FAMILIES:
            raise HistoryError("decision or family count out of bounds")
        for family in outcome:
            _check_name("family", family)
        for decision in decisions:
            _check_name("decision", decision)
        if self._generations and set(outcome) != set(self.families()):
            raise HistoryError("a generation must report the pool's families")
        for value in outcome.values():
            if not isinstance(value, Outcome):
                raise HistoryError("outcome values must be Outcome instances")
        # Encode against a copy, so a refusal leaves the registered categories untouched.
        categories = self.categories()
        zero_based = self.zero_based()
        if not self._generations:
            zero_based = {n for n, v in decisions.items() if isinstance(v, str)}
        levels = levels_of(decisions, categories, zero_based, register=True)
        self._categories = categories
        self._zero_based = zero_based
        self._generations.append(
            Generation(
                name=name,
                order=order,
                # Read-only views over private copies: neither the caller's dicts nor a
                # generation handed out by ``generations`` can change the history afterwards.
                decisions=MappingProxyType(dict(decisions)),
                outcome=MappingProxyType(dict(outcome)),
                levels=MappingProxyType(levels),
            )
        )

    def fingerprint(self) -> str:
        """Identity of the history, covering names, order, decisions and outcomes."""
        material = [
            {
                "name": g.name,
                "order": g.order,
                "decisions": {k: g.decisions[k] for k in sorted(g.decisions)},
                "outcome": {k: [g.outcome[k].correct, g.outcome[k].total] for k in sorted(g.outcome)},
            }
            for g in self._generations
        ]
        canonical = json.dumps(material, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Prediction:
    rate: float | None
    identified: bool
    unidentified: tuple[str, ...] = ()


class ReplaySimulator:
    """Intercept plus one response coefficient per decision, from a pool without ``exclude``."""

    def __init__(self, pool: HistoryPool, *, exclude: str | None = None) -> None:
        self.pool = pool
        self.exclude = exclude
        self.generations = [g for g in pool.generations if g.name != exclude]
        if not self.generations:
            raise HistoryError("nothing left in the pool after exclusion")
        self.families = pool.families()
        self._encode()
        self.responses, self.ambiguous = self._fit()
        self.limits = self._limits()
        self.baseline = self._baseline()

    def _encode(self) -> None:
        """Encode the decisions of this fold from the generations it keeps and nothing else.

        The pool's own encoding saw every generation, the held-out one included: a category that
        appears only there would take a position in the training levels, and the order of
        first appearance could shift every later category. Here the categories, which
        decisions are zero-based (present in the first kept generation) and every kept
        generation's levels are rebuilt in order from the kept generations alone. The held-out
        generation's edges are still skipped when the transitions are read.
        """
        self._categories: dict[str, list[str]] = {}
        self._zero_based: set[str] = set()
        self._levels: dict[str, dict[str, float]] = {}
        for index, generation in enumerate(self.generations):
            if index == 0:
                self._zero_based = {
                    n for n, v in generation.decisions.items() if isinstance(v, str)
                }
            self._levels[generation.name] = levels_of(
                generation.decisions, self._categories, self._zero_based, register=True
            )

    def _fit(self) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
        slopes: dict[str, dict[str, list[float]]] = {}
        ambiguous: list[dict[str, Any]] = []
        # Decisions whose step or slope is not a finite number (a step of 5e-324 overflows the
        # slope): they are not identified, and no NaN or infinity reaches a prediction.
        self.nonfinite: set[str] = set()
        # Transitions are the pool's own neighbours. A held-out generation takes its two edges
        # with it: the pair on either side of it is not a transition of the history.
        everything = self.pool.generations
        for prev, curr in zip(everything, everything[1:]):
            if self.exclude in (prev.name, curr.name):
                continue
            before, after = self._levels[prev.name], self._levels[curr.name]
            names = set(before) | set(after)
            moved = sorted(n for n in names if after.get(n, 0.0) != before.get(n, 0.0))
            if not moved:
                continue
            if len(moved) > 1:
                ambiguous.append({"from": prev.name, "to": curr.name, "decisions": moved})
                continue
            decision = moved[0]
            step = after.get(decision, 0.0) - before.get(decision, 0.0)
            if not math.isfinite(step) or step == 0.0:
                self.nonfinite.add(decision)
                continue
            for family in self.families:
                delta = curr.outcome[family].rate - prev.outcome[family].rate
                slope = delta / step
                if not math.isfinite(slope):
                    self.nonfinite.add(decision)
                    continue
                slopes.setdefault(decision, {}).setdefault(family, []).append(slope)
        responses: dict[str, dict[str, float]] = {}
        for decision, families in slopes.items():
            if decision in self.nonfinite:
                continue
            means = {f: sum(v) / len(v) for f, v in families.items()}
            if all(math.isfinite(m) for m in means.values()):
                responses[decision] = means
            else:
                self.nonfinite.add(decision)
        return responses, ambiguous

    def _limits(self) -> dict[str, tuple[float, float]]:
        # A decision a generation does not name is at level 0 there, so a decision that appears
        # only later still has 0 inside its observed range.
        names: set[str] = set()
        for generation in self.generations:
            names |= set(self._levels[generation.name])
        limits: dict[str, tuple[float, float]] = {}
        for generation in self.generations:
            for name in names:
                level = self._levels[generation.name].get(name, 0.0)
                low, high = limits.get(name, (level, level))
                limits[name] = (min(low, level), max(high, level))
        return limits

    def _baseline(self) -> dict[str, float] | None:
        """The intercept is the generation that took every decision at level zero, if any."""
        for generation in self.generations:
            if all(level == 0.0 for level in self._levels[generation.name].values()):
                return {f: generation.outcome[f].rate for f in self.families}
        return None

    def simulate(self, decisions: Mapping[str, Level]) -> dict[str, Prediction]:
        levels = levels_of(decisions, dict(self._categories), self._zero_based)
        unknown = tuple(
            sorted(n for n, level in levels.items() if level != 0.0 and n not in self.responses)
        )
        if self.baseline is not None and not unknown:
            # A response times a level can still leave the finite numbers.
            for name, level in levels.items():
                for response in self.responses.get(name, {}).values():
                    if not math.isfinite(response * level):
                        unknown = (*unknown, name)
            unknown = tuple(sorted(set(unknown)))
        if self.baseline is None or unknown:
            reason = unknown or ("intercept",)
            return {f: Prediction(None, False, reason) for f in self.families}
        out: dict[str, Prediction] = {}
        for family in self.families:
            rate = self.baseline[family]
            for name, level in levels.items():
                response = self.responses.get(name, {})
                if family in response:
                    rate += response[family] * level
            if not math.isfinite(rate):
                return {f: Prediction(None, False, ("non-finite prediction",)) for f in self.families}
            out[family] = Prediction(min(max(rate, 0.0), 1.0), True)
        return out

    def extrapolation(self, decisions: Mapping[str, Level]) -> list[dict[str, Any]]:
        """Levels asked for that the history never showed."""
        beyond = []
        for name, level in levels_of(decisions, dict(self._categories), self._zero_based).items():
            low, high = self.limits.get(name, (0.0, 0.0))
            if level < low or level > high:
                beyond.append({"decision": name, "level": level, "observed": [low, high]})
        return beyond

    def rank(self, policies: Iterable[Mapping[str, Level]]) -> list[dict[str, Any]]:
        """Score candidate policies; unidentified ones rank last with a ``None`` score."""
        ranked = []
        for decisions in policies:
            prediction = self.simulate(decisions)
            identified = all(p.identified for p in prediction.values())
            score = (
                sum(p.rate for p in prediction.values() if p.rate is not None)
                if identified
                else None
            )
            ranked.append(
                {
                    "decisions": dict(decisions),
                    # The exact score decides the order; the rounded one is for the report.
                    "score": score,
                    "score_reported": None if score is None else round(score, 6),
                    "identified": identified,
                    "extrapolation": self.extrapolation(decisions),
                }
            )
        ranked.sort(key=lambda r: (r["score"] is not None, r["score"] or 0.0), reverse=True)
        return ranked


@dataclass(frozen=True)
class BacktestReport:
    """The outcome of a backtest. ``pipeline_ran`` and ``predictive`` are separate claims."""

    pipeline_ran: bool
    predictive: bool
    generations: int
    parameters: int
    degenerate: bool
    error_bound: float
    min_within_bound: int
    min_within_fraction: float
    predicted: int
    within_bound: int
    max_abs_error: float | None
    in_sample_max_abs_error: float | None
    ambiguous_transitions: tuple[dict[str, Any], ...]
    per_generation: tuple[dict[str, Any], ...]
    pool_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": "leave_one_generation_out",
            "pipeline_ran": self.pipeline_ran,
            "predictive": self.predictive,
            "generations": self.generations,
            "parameters": self.parameters,
            "degenerate": self.degenerate,
            "error_bound": self.error_bound,
            "predictive_rule": {
                "min_within_bound": self.min_within_bound,
                "min_within_fraction": self.min_within_fraction,
            },
            "out_of_sample": {
                "predicted": self.predicted,
                "within_bound": self.within_bound,
                "max_abs_error": self.max_abs_error,
            },
            "in_sample": {
                "max_abs_error": self.in_sample_max_abs_error,
                "note": "not a measure of predictive accuracy while degenerate",
            },
            "ambiguous_transitions": list(self.ambiguous_transitions),
            "per_generation": list(self.per_generation),
            "pool_fingerprint": self.pool_fingerprint,
        }


def backtest(
    pool: HistoryPool,
    *,
    error_bound: float,
    min_within_bound: int,
    min_within_fraction: float,
) -> BacktestReport:
    """Leave each generation out and predict it from the rest.

    ``error_bound`` is the largest absolute rate error, per family, that still counts as a
    prediction. ``min_within_bound`` is the fewest generations that must be predicted within it
    and ``min_within_fraction`` the smallest share of the predictable generations they must
    make up. None of the three has a default: they are pinned by whoever preregisters the
    claim, before the first run, and a pilot number is never a reason to change them.
    """
    if isinstance(error_bound, bool) or not isinstance(error_bound, (int, float)):
        raise ValueError("error_bound must be a number")
    if not math.isfinite(error_bound) or not 0.0 < error_bound <= 1.0:
        raise ValueError("error_bound must be in (0, 1]")
    if (
        isinstance(min_within_bound, bool)
        or not isinstance(min_within_bound, int)
        or min_within_bound < 1
    ):
        raise ValueError("min_within_bound must be an int of at least 1")
    if (
        isinstance(min_within_fraction, bool)
        or not isinstance(min_within_fraction, (int, float))
        or not math.isfinite(min_within_fraction)
        or not 0.0 < min_within_fraction <= 1.0
    ):
        raise ValueError("min_within_fraction must be in (0, 1]")
    if not pool.generations:
        raise HistoryError("the history pool is empty")

    full = ReplaySimulator(pool)
    parameters = 1 + len(full.responses)
    in_sample: list[float] = []
    for generation in full.generations:
        prediction = full.simulate(generation.decisions)
        for family, value in prediction.items():
            if value.rate is not None:
                in_sample.append(abs(value.rate - generation.outcome[family].rate))

    rows: list[dict[str, Any]] = []
    out_errors: list[float] = []
    predicted = within = 0
    for generation in pool.generations:
        try:
            reduced = ReplaySimulator(pool, exclude=generation.name)
        except HistoryError:
            rows.append(
                {"generation": generation.name, "predicted": False, "reason": "pool empty without it"}
            )
            continue
        prediction = reduced.simulate(generation.decisions)
        if not all(p.identified for p in prediction.values()):
            unidentified = next(iter(prediction.values())).unidentified
            rows.append(
                {
                    "generation": generation.name,
                    "predicted": False,
                    "not_identified": True,
                    "reason": "not identified without it: " + ", ".join(unidentified),
                }
            )
            continue
        # The decision is taken on the exact errors; rounding is for the report only.
        exact = {
            f: abs((p.rate or 0.0) - generation.outcome[f].rate) for f, p in prediction.items()
        }
        errors = {f: round(value, 6) for f, value in exact.items()}
        extrapolation = reduced.extrapolation(generation.decisions)
        ok = max(exact.values()) <= error_bound
        predicted += 1
        within += 1 if ok else 0
        out_errors.extend(exact.values())
        rows.append(
            {
                "generation": generation.name,
                "predicted": True,
                "abs_error": errors,
                "within_bound": ok,
                "extrapolation": extrapolation,
            }
        )

    return BacktestReport(
        pipeline_ran=True,
        predictive=(
            within >= min_within_bound and within >= min_within_fraction * predicted
        ),
        generations=len(pool.generations),
        parameters=parameters,
        degenerate=parameters >= len(full.generations),
        error_bound=float(error_bound),
        min_within_bound=min_within_bound,
        min_within_fraction=float(min_within_fraction),
        predicted=predicted,
        within_bound=within,
        max_abs_error=round(max(out_errors), 6) if out_errors else None,
        in_sample_max_abs_error=round(max(in_sample), 6) if in_sample else None,
        ambiguous_transitions=tuple(full.ambiguous),
        per_generation=tuple(rows),
        pool_fingerprint=pool.fingerprint(),
    )


def choose_winner(
    simulator: ReplaySimulator, report: BacktestReport, policies: Iterable[Mapping[str, Level]]
) -> dict[str, Any]:
    """The best identified policy, or an abort.

    Raises ``NotPredictive`` when the backtest of this history did not predict a held-out
    generation within its bound, ``ExtrapolationAbort`` when the best identified policy asks
    for decision levels the history never showed, and ``ValueError`` when no policy is
    identified at all or the report belongs to another history.
    """
    if report.pool_fingerprint != simulator.pool.fingerprint():
        raise ValueError("the backtest report was made on another history")
    if not report.predictive:
        raise NotPredictive("no held-out generation was predicted within the error bound")
    ranked = simulator.rank(policies)
    best = next((r for r in ranked if r["identified"]), None)
    if best is None:
        raise ValueError("no policy is identified by this history")
    if best["extrapolation"]:
        raise ExtrapolationAbort(f"winner goes beyond observed history: {best['extrapolation']}")
    return best
