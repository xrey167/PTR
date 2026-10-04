"""Deterministic verifiable tasks (pure Python, seeded)."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

KIND_ARITH = "arith"
KIND_JSON = "json"
KINDS = (KIND_ARITH, KIND_JSON)

_JSON_KEYS = ("a", "b", "c")


@dataclass(frozen=True)
class Task:
    kind: str
    prompt: str
    # Canonical expected answer as plain text; the reward re-derives checks from `spec`.
    answer: str
    # JSON-serialisable spec the verifier uses (kept as a string so it survives HF datasets).
    spec: str

    def to_row(self) -> dict:
        return {"prompt": self.prompt, "answer": self.answer, "kind": self.kind, "spec": self.spec}


def _arith(rng: random.Random, max_operand: int) -> Task:
    a = rng.randint(0, max_operand)
    b = rng.randint(0, max_operand)
    op = rng.choice("+-")
    value = a + b if op == "+" else a - b
    return Task(
        kind=KIND_ARITH,
        prompt=f"Q:{a}{op}{b}=\nA:",
        answer=str(value),
        spec=json.dumps({"value": value}, sort_keys=True),
    )


def _json_shape(rng: random.Random, max_operand: int) -> Task:
    n_keys = rng.randint(1, 2)
    keys = list(_JSON_KEYS[:n_keys])
    values = {k: rng.randint(0, max_operand) for k in keys}
    shown = ",".join(f"{k}={v}" for k, v in values.items())
    return Task(
        kind=KIND_JSON,
        prompt=f"J:{shown}\nA:",
        answer=json.dumps(values, separators=(",", ":")),
        spec=json.dumps({"values": values}, sort_keys=True),
    )


def generate_tasks(
    n: int, *, seed: int, kinds: tuple[str, ...] = KINDS, max_operand: int = 9
) -> list[Task]:
    """`n` tasks, cycling through `kinds`, fully determined by `seed`."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if not kinds:
        raise ValueError("kinds must be non-empty")
    unknown = [k for k in kinds if k not in KINDS]
    if unknown:
        raise ValueError(f"unknown task kinds: {unknown}")
    if max_operand < 1:
        raise ValueError("max_operand must be >= 1")
    rng = random.Random(seed)
    out = []
    for i in range(n):
        kind = kinds[i % len(kinds)]
        out.append(_arith(rng, max_operand) if kind == KIND_ARITH else _json_shape(rng, max_operand))
    return out


def sft_text(task: Task, eos: str = "") -> str:
    """Full supervised sequence: prompt followed by the exact answer."""
    return f"{task.prompt}{task.answer}{eos}"
