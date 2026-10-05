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


def _check_args(n: int, kinds: tuple[str, ...], max_operand: int) -> None:
    if n < 0:
        raise ValueError("n must be non-negative")
    if not kinds:
        raise ValueError("kinds must be non-empty")
    unknown = [k for k in kinds if k not in KINDS]
    if unknown:
        raise ValueError(f"unknown task kinds: {unknown}")
    if max_operand < 1:
        raise ValueError("max_operand must be >= 1")


def space_size(kind: str, max_operand: int = 9) -> int:
    """Number of distinct prompts a kind can produce."""
    v = max_operand + 1
    if kind == KIND_ARITH:
        return v * v * 2
    if kind == KIND_JSON:
        return v + v * v  # one key, or two keys
    raise ValueError(f"unknown task kind {kind!r}")


def _one(rng: random.Random, kind: str, max_operand: int) -> Task:
    return _arith(rng, max_operand) if kind == KIND_ARITH else _json_shape(rng, max_operand)


def generate_tasks(
    n: int,
    *,
    seed: int,
    kinds: tuple[str, ...] = KINDS,
    max_operand: int = 9,
    exclude_prompts: frozenset[str] | set[str] = frozenset(),
) -> list[Task]:
    """`n` tasks, cycling through `kinds`, fully determined by `seed`.

    Sampled with replacement. Tasks whose prompt is in `exclude_prompts` are never produced;
    if a kind's whole space is excluded a ValueError is raised.
    """
    _check_args(n, kinds, max_operand)
    if exclude_prompts:
        for kind in kinds:
            if len(exclude_prompts) >= space_size(kind, max_operand) and _exhausted(
                kind, max_operand, exclude_prompts
            ):
                raise ValueError(f"task space for kind {kind!r} is fully excluded")
    rng = random.Random(seed)
    out = []
    for i in range(n):
        kind = kinds[i % len(kinds)]
        for _ in range(100_000):
            task = _one(rng, kind, max_operand)
            if task.prompt not in exclude_prompts:
                break
        else:
            raise ValueError(f"could not sample a non-excluded {kind!r} task")
        out.append(task)
    return out


def _exhausted(kind: str, max_operand: int, exclude: set[str] | frozenset[str]) -> bool:
    return len({t.prompt for t in _enumerate(kind, max_operand)} - set(exclude)) == 0


def _enumerate(kind: str, max_operand: int):
    if kind == KIND_ARITH:
        for a in range(max_operand + 1):
            for b in range(max_operand + 1):
                for op in "+-":
                    value = a + b if op == "+" else a - b
                    yield Task(KIND_ARITH, f"Q:{a}{op}{b}=\nA:", str(value),
                               json.dumps({"value": value}, sort_keys=True))
    else:
        v = range(max_operand + 1)
        combos = [{"a": x} for x in v] + [{"a": x, "b": y} for x in v for y in v]
        for values in combos:
            shown = ",".join(f"{k}={val}" for k, val in values.items())
            yield Task(KIND_JSON, f"J:{shown}\nA:", json.dumps(values, separators=(",", ":")),
                       json.dumps({"values": values}, sort_keys=True))


def longest_prompt_chars(kinds: tuple[str, ...] = KINDS, max_operand: int = 9) -> int:
    """Longest prompt (characters) over the whole task space of `kinds`."""
    return max(len(t.prompt) for k in kinds for t in _enumerate(k, max_operand))


def longest_answer_chars(kinds: tuple[str, ...] = KINDS, max_operand: int = 9) -> int:
    """Longest canonical answer (characters) over the whole task space of `kinds`."""
    return max(len(t.answer) for k in kinds for t in _enumerate(k, max_operand))


def generate_heldout(
    n: int, *, seed: int, kinds: tuple[str, ...] = KINDS, max_operand: int = 9
) -> list[Task]:
    """`n` tasks with pairwise-distinct prompts (sampled without replacement), cycling kinds.

    Raises ValueError, rather than repeating prompts, if the space cannot supply `n`.
    """
    _check_args(n, kinds, max_operand)
    need = {k: sum(1 for i in range(n) if kinds[i % len(kinds)] == k) for k in kinds}
    for kind, count in need.items():
        if count > space_size(kind, max_operand):
            raise ValueError(
                f"requested {count} distinct {kind!r} tasks but the space has only "
                f"{space_size(kind, max_operand)} (max_operand={max_operand})"
            )
    rng = random.Random(seed)
    pools = {k: list(_enumerate(k, max_operand)) for k in kinds}
    for pool in pools.values():
        rng.shuffle(pool)
    return [pools[kinds[i % len(kinds)]].pop() for i in range(n)]


def sft_text(task: Task, eos: str = "") -> str:
    """Full supervised sequence: prompt followed by the exact answer."""
    return f"{task.prompt}{task.answer}{eos}"
