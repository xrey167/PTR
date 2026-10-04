"""Verifiable reward mirroring `ptr_verifier::VerificationReport` (pure Python, no torch).

Report shape (crates/ptr-verifier/src/lib.rs): `status` is one of FOUR values, "Pass",
"Fail", "Disputed", "Unknown"; plus `level`, `score` in [0, 1] and `findings`
[{code, message, hard}].

Reward mapping (`rl_reward`), the only number RL ever sees:

* Pass     -> `score`, which must be in (0, 1] (exact-answer tasks: 1.0).
* Fail     -> 0.0
* Disputed -> 0.0
* Unknown  -> 0.0

A non-Pass report can never receive a passing reward, and its own `score` (for example
the shape credit of a wrong-valued integer) is kept in the report for diagnostics but is
never used as reward. A hard finding forces `Fail` (`VerificationReport` demotes a Pass
that carries one, and a Pass whose score is not in (0, 1]).

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


def rl_reward(report: VerificationReport) -> float:
    """RL reward of a report: its score if (and only if) status is Pass, else 0.0."""
    if report.status == PASS and 0.0 < report.score <= 1.0 and not any(
        f.hard for f in report.findings
    ):
        return report.score
    return 0.0


def _first_line(completion: str) -> str:
    return completion.strip().split("\n", 1)[0].strip()


def _verify_arith(completion: str, spec: dict) -> VerificationReport:
    if not completion.strip():
        return VerificationReport(
            UNKNOWN, LEVEL, 0.0, [Finding("empty-completion", "nothing to verify", False)]
        )
    answers = {
        int(line.strip())
        for line in completion.strip().split("\n")
        if _INT_RE.match(line.strip())
    }
    if len(answers) > 1:
        return VerificationReport(
            DISPUTED,
            LEVEL,
            SHAPE_CREDIT,
            [Finding("contradicting-answers", f"completion states {sorted(answers)}", False)],
        )
    text = _first_line(completion)
    if not _INT_RE.match(text):
        return _report(0.0, [Finding("not-integer", f"expected integer, got {text!r}", True)])
    if int(text) == spec["value"]:
        return _report(1.0, [])
    return _report(
        SHAPE_CREDIT,
        [Finding("wrong-value", f"expected {spec['value']}, got {text}", True)],
    )


def _verify_json(completion: str, spec: dict) -> VerificationReport:
    if not completion.strip():
        return VerificationReport(
            UNKNOWN, LEVEL, 0.0, [Finding("empty-completion", "nothing to verify", False)]
        )
    text = _first_line(completion)
    expected = spec["values"]
    try:
        parsed = json.loads(text)
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


def reward(kind: str, spec: str | dict, completion: str) -> float:
    return rl_reward(verify(kind, spec, completion))


def reward_batch(
    completions: list, kind: list[str], spec: list[str], **_ignored
) -> list[float]:
    """TRL-compatible reward function (extra dataset columns arrive as keyword lists)."""
    out = []
    for completion, k, s in zip(completions, kind, spec, strict=True):
        if not isinstance(completion, str):  # conversational format: [{"role","content"}]
            completion = completion[-1]["content"]
        out.append(reward(k, s, completion))
    return out
