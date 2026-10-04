"""Verifiable reward mirroring `ptr_verifier::VerificationReport` (pure Python, no torch).

Report shape (crates/ptr-verifier/src/lib.rs): `status` is one of FOUR values, "Pass",
"Fail", "Disputed", "Unknown"; plus `level`, `score` in [0, 1] and `findings`
[{code, message, hard}].

Reward mapping (`rl_reward`), the only number RL ever sees:

* Pass     -> `score`, which must be in (0, 1] (exact-answer tasks: 1.0), but only when
              `report.level` is at or above the configured minimum verification level
              (`moe.rl.min_verification_level`); a Pass below it earns 0.0.
* Fail     -> 0.0
* Disputed -> 0.0
* Unknown  -> 0.0

A non-Pass report can never receive a passing reward, and its own `score` (for example
the shape credit of a wrong-valued integer) is kept in the report for diagnostics but is
never used as reward. A hard finding forces `Fail` (`VerificationReport` demotes a Pass
that carries one, and a Pass whose score is not in (0, 1]).

One answer only: the whole stripped completion must be exactly one answer line. For
arithmetic, further non-empty lines are `Disputed` when they state a different integer and
a hard `trailing-content` `Fail` otherwise (even an identical repeat); the integer must be
canonical decimal. For JSON, any second non-empty line is a hard `trailing-content` `Fail`.

Status production here: `Unknown` = empty completion (nothing to verify); `Disputed` =
arithmetic completion whose lines contain contradicting integers; `Fail` = anything else
that is not the exact answer; `Pass` = exact answer.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

from ptr_training.moe.tasks import KIND_ARITH, KIND_JSON

PASS = "Pass"
FAIL = "Fail"
DISPUTED = "Disputed"
UNKNOWN = "Unknown"
STATUSES = (PASS, FAIL, DISPUTED, UNKNOWN)
# ptr_types::VerificationLevel, weakest -> strongest (crates/ptr-types/src/lib.rs; the same
# order as `level_rank` in crates/ptr-verifier/src/lib.rs).
LEVELS = ("Unverified", "LatentAgreement", "SampleVerified", "FullSemantic", "Deterministic")
# What this module's deterministic verifiers produce, and the default minimum for RL.
LEVEL = "Deterministic"

SHAPE_CREDIT = 0.2
JSON_PARSE_CREDIT = 0.2
JSON_KEYS_CREDIT = 0.3
_INT_RE = re.compile(r"^-?\d+$")


@dataclass(frozen=True)
class Finding:
    code: str
    message: str
    hard: bool


@dataclass(frozen=True)
class VerificationReport:
    status: str
    level: str
    score: float
    findings: list[Finding] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {self.status!r}")
        if self.level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {self.level!r}")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.score}")
        if self.status == PASS and (self.score <= 0.0 or any(f.hard for f in self.findings)):
            # Invariant, not a convenience: a Pass with a hard finding (or zero score) is a Fail.
            object.__setattr__(self, "status", FAIL)

    def to_dict(self) -> dict:
        return asdict(self)


def _report(score: float, findings: list[Finding]) -> VerificationReport:
    score = min(1.0, max(0.0, score))
    status = PASS if score == 1.0 and not any(f.hard for f in findings) else FAIL
    return VerificationReport(status=status, level=LEVEL, score=score, findings=findings)


def level_at_least(level: str, minimum: str) -> bool:
    for name in (level, minimum):
        if name not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {name!r}")
    return LEVELS.index(level) >= LEVELS.index(minimum)


def rl_reward(report: VerificationReport, min_level: str = LEVEL) -> float:
    """RL reward: the score iff status is Pass and level >= `min_level`, else 0.0."""
    if not level_at_least(report.level, min_level):
        return 0.0
    if report.status == PASS and 0.0 < report.score <= 1.0 and not any(
        f.hard for f in report.findings
    ):
        return report.score
    return 0.0


def _lines(completion: str) -> list[str]:
    """Non-empty stripped lines of the completion."""
    return [line.strip() for line in completion.strip().split("\n") if line.strip()]


def _verify_arith(completion: str, spec: dict) -> VerificationReport:
    if not completion.strip():
        return VerificationReport(
            UNKNOWN, LEVEL, 0.0, [Finding("empty-completion", "nothing to verify", False)]
        )
    lines = _lines(completion)
    answers = {int(line) for line in lines if _INT_RE.match(line)}
    if len(answers) > 1:
        return VerificationReport(
            DISPUTED,
            LEVEL,
            SHAPE_CREDIT,
            [Finding("contradicting-answers", f"completion states {sorted(answers)}", False)],
        )
    text = lines[0]
    if len(lines) > 1:  # one answer only: any further content (even a repeat) is a failure
        return _report(
            SHAPE_CREDIT if _INT_RE.match(text) else 0.0,
            [Finding("trailing-content", f"expected a single answer, got {len(lines)} lines", True)],
        )
    if not _INT_RE.match(text):
        return _report(0.0, [Finding("not-integer", f"expected integer, got {text!r}", True)])
    if text != str(int(text)):  # canonical decimal only ("05", "-0" are not the answer)
        return _report(
            SHAPE_CREDIT, [Finding("non-canonical-integer", f"got {text!r}", True)]
        )
    if int(text) == spec["value"]:
        return _report(1.0, [])
    return _report(
        SHAPE_CREDIT,
        [Finding("wrong-value", f"expected {spec['value']}, got {text}", True)],
    )


class _DuplicateKeys(ValueError):
    pass


def _no_duplicate_keys(pairs: list) -> dict:
    keys = [k for k, _ in pairs]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    if dupes:
        raise _DuplicateKeys(f"duplicate JSON keys {dupes}")
    return dict(pairs)


def _verify_json(completion: str, spec: dict) -> VerificationReport:
    if not completion.strip():
        return VerificationReport(
            UNKNOWN, LEVEL, 0.0, [Finding("empty-completion", "nothing to verify", False)]
        )
    lines = _lines(completion)
    text = lines[0]
    expected = spec["values"]
    if len(lines) > 1:  # one answer only: the whole completion must be a single JSON value
        return _report(
            0.0, [Finding("trailing-content", f"expected a single JSON value, got {len(lines)} lines", True)]
        )
    try:
        parsed = json.loads(text, object_pairs_hook=_no_duplicate_keys)
    except _DuplicateKeys as error:
        return _report(0.0, [Finding("duplicate-keys", str(error), True)])
    except (ValueError, TypeError):
        return _report(0.0, [Finding("invalid-json", f"not parseable JSON: {text!r}", True)])
    if not isinstance(parsed, dict):
        return _report(
            JSON_PARSE_CREDIT, [Finding("not-object", "JSON value is not an object", True)]
        )
    if set(parsed) != set(expected):
        return _report(
            JSON_PARSE_CREDIT,
            [Finding("wrong-keys", f"expected keys {sorted(expected)}, got {sorted(parsed)}", True)],
        )
    if all(type(parsed[k]) is int and parsed[k] == v for k, v in expected.items()):
        return _report(1.0, [])
    return _report(
        JSON_PARSE_CREDIT + JSON_KEYS_CREDIT,
        [Finding("wrong-values", f"expected {expected}, got {parsed}", True)],
    )


def verify(kind: str, spec: str | dict, completion: str) -> VerificationReport:
    """Deterministically verify one completion against a task spec."""
    spec_obj = json.loads(spec) if isinstance(spec, str) else spec
    if kind == KIND_ARITH:
        return _verify_arith(completion, spec_obj)
    if kind == KIND_JSON:
        return _verify_json(completion, spec_obj)
    raise ValueError(f"unknown task kind: {kind!r}")


def reward(kind: str, spec: str | dict, completion: str, min_level: str = LEVEL) -> float:
    return rl_reward(verify(kind, spec, completion), min_level)


def reward_batch(
    completions: list, kind: list[str], spec: list[str], *, min_level: str = LEVEL, **_ignored
) -> list[float]:
    """TRL-compatible reward function (extra dataset columns arrive as keyword lists)."""
    out = []
    for completion, k, s in zip(completions, kind, spec, strict=True):
        if not isinstance(completion, str):  # conversational format: [{"role","content"}]
            completion = completion[-1]["content"]
        out.append(reward(k, s, completion, min_level))
    return out


def make_reward_fn(min_level: str):
    """TRL reward function with the verification-level gate bound (TRL needs `__name__`)."""
    if min_level not in LEVELS:
        raise ValueError(f"min_level must be one of {LEVELS}, got {min_level!r}")

    def reward_batch_gated(completions, kind, spec, **ignored):
        return reward_batch(completions, kind, spec, min_level=min_level)

    reward_batch_gated.__name__ = "reward_batch"
    return reward_batch_gated
