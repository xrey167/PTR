"""Training integrity checks: the run proves it trained what it says it trained.

Four properties are checked and recorded in `metrics.json` under `integrity`:

* `frozen_unchanged`: every parameter with `requires_grad=False` is bit-identical to the
  one the run started with (a LoRA run must not touch its base model).
* `adapter_changed`: training actually moved the trainable parameters. A "trained" adapter
  equal to its initialisation is a silent no-op.
* `steps_exact`: the optimizer took exactly the configured number of steps, not fewer
  (early stop, empty data) and not more.
* `eval_identity_unchanged`: evaluation did not change the model it measured.

The checks fail closed: a violation raises `IntegrityError` and the pipeline writes no
metrics. All comparison logic is plain Python over hex digests; only `tensor_bytes` needs
torch, and every digest function accepts a replacement for it, so the logic is testable
without the `moe` extra.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

SCHEMA = "ptr.training-integrity.v1"
REQUIRED_CHECKS = (
    "frozen_unchanged",
    "adapter_changed",
    "steps_exact",
    "eval_identity_unchanged",
)


class IntegrityError(RuntimeError):
    """A training integrity property does not hold; the run must not produce evidence."""

    def __init__(self, check: str, message: str, *, expected: Any = None, actual: Any = None):
        super().__init__(f"integrity check {check!r} failed: {message}")
        self.check = check
        self.expected = expected
        self.actual = actual


def tensor_bytes(tensor) -> bytes:
    """Raw bytes of a tensor, independent of dtype (bf16 has no numpy equivalent)."""
    import torch

    flat = tensor.detach().to("cpu").contiguous().reshape(-1)
    return flat.view(torch.uint8).numpy().tobytes() if flat.numel() else b""


def canonical_sha256(payload: Any) -> str:
    """SHA-256 of the canonical JSON encoding (sorted keys, no whitespace, ASCII)."""
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _digest(
    named: Iterable[tuple[str, Any]], encode: Callable[[Any], bytes]
) -> tuple[str, int]:
    """(hex digest, tensor count) over name, dtype, shape and bytes, sorted by name."""
    hasher = hashlib.sha256()
    count = 0
    for name, tensor in sorted(named, key=lambda item: item[0]):
        header = f"{name}|{tensor.dtype}|{tuple(tensor.shape)}|".encode("utf-8")
        data = encode(tensor)
        hasher.update(len(header).to_bytes(8, "big") + header)
        hasher.update(len(data).to_bytes(8, "big") + data)
        count += 1
    return hasher.hexdigest(), count


def frozen_parameter_digest(model, encode: Callable[[Any], bytes] = tensor_bytes) -> str:
    """Digest of every parameter with `requires_grad=False`, in name order."""
    return _digest(
        ((n, p) for n, p in model.named_parameters() if not p.requires_grad), encode
    )[0]


def trainable_parameter_digest(model, encode: Callable[[Any], bytes] = tensor_bytes) -> str:
    """Digest of every parameter with `requires_grad=True`, in name order."""
    return _digest(
        ((n, p) for n, p in model.named_parameters() if p.requires_grad), encode
    )[0]


def model_digest(model, encode: Callable[[Any], bytes] = tensor_bytes) -> str:
    """Digest of all parameters, frozen and trainable alike."""
    return _digest(model.named_parameters(), encode)[0]


def parameter_counts(model) -> dict:
    """{"frozen": n, "trainable": m} tensor counts."""
    frozen = sum(1 for _, p in model.named_parameters() if not p.requires_grad)
    total = sum(1 for _ in model.named_parameters())
    return {"frozen": frozen, "trainable": total - frozen}


def assert_frozen_unchanged(before: str, after: str) -> None:
    if before != after:
        raise IntegrityError(
            "frozen_unchanged",
            "a frozen parameter changed during training",
            expected=before,
            actual=after,
        )


def assert_adapter_changed(before: str, after: str) -> None:
    if before == after:
        raise IntegrityError(
            "adapter_changed",
            "trainable parameters are identical before and after training",
            expected="different digest",
            actual=after,
        )


def assert_step_count(trainer_state, expected: int, what: str = "training") -> int:
    """`trainer_state` is an int or any object with `global_step`; returns the actual count."""
    actual = trainer_state if isinstance(trainer_state, int) else trainer_state.global_step
    if isinstance(actual, bool) or actual != expected:
        raise IntegrityError(
            "steps_exact",
            f"{what} took {actual} optimizer steps, configured {expected}",
            expected=expected,
            actual=actual,
        )
    return actual


def assert_model_unchanged_during_eval(before: str, after: str, what: str = "evaluation") -> None:
    if before != after:
        raise IntegrityError(
            "eval_identity_unchanged",
            f"{what} changed the model it was measuring",
            expected=before,
            actual=after,
        )


def directory_sha256(path: Path) -> str:
    """SHA-256 over relative path and bytes of every file below `path`, in path order."""
    hasher = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"no files to hash under {path}")
    for file in files:
        name = file.relative_to(path).as_posix().encode("utf-8")
        data = file.read_bytes()
        hasher.update(len(name).to_bytes(8, "big") + name)
        hasher.update(len(data).to_bytes(8, "big") + data)
    return hasher.hexdigest()


class IntegrityLog:
    """Collects the outcome of every check of one run and renders the `integrity` block.

    Each `record_*` method raises `IntegrityError` on a violation, so a run cannot go on
    past a failed check. `report()` raises as well when a required check never ran: a
    property nobody tested is not a property that holds.
    """

    def __init__(self, encode: Callable[[Any], bytes] = tensor_bytes):
        self._encode = encode
        self._checks: list[dict] = []
        self._frozen_reference: str | None = None
        self._counts: dict = {"frozen": 0, "trainable": 0}

    def start(self, model) -> None:
        """Fix the frozen reference at the point training is about to begin."""
        self._counts = parameter_counts(model)
        self._frozen_reference = frozen_parameter_digest(model, self._encode)

    def trainable_digest(self, model) -> str:
        return trainable_parameter_digest(model, self._encode)

    def _record(self, check: str, stage: str, **detail) -> None:
        self._checks.append({"check": check, "stage": stage, "ok": True, **detail})

    def record_stage(self, stage: str, model, trainable_before: str) -> str:
        """After a training stage: frozen untouched, trainable moved. Returns the new digest."""
        if self._frozen_reference is None:
            raise RuntimeError("IntegrityLog.start() was not called before training")
        frozen_after = frozen_parameter_digest(model, self._encode)
        assert_frozen_unchanged(self._frozen_reference, frozen_after)
        self._record("frozen_unchanged", stage, tensors=self._counts["frozen"])
        trainable_after = trainable_parameter_digest(model, self._encode)
        assert_adapter_changed(trainable_before, trainable_after)
        self._record("adapter_changed", stage, tensors=self._counts["trainable"])
        return trainable_after

    def record_steps(self, stage: str, expected: int, actual: int) -> None:
        assert_step_count(actual, expected, stage)
        self._record("steps_exact", stage, steps=actual)

    def checked_eval(self, stage: str, model, evaluate: Callable[[], Any]):
        """Run `evaluate()` and require the model to be bit-identical before and after."""
        before = model_digest(model, self._encode)
        result = evaluate()
        assert_model_unchanged_during_eval(before, model_digest(model, self._encode), stage)
        self._record("eval_identity_unchanged", stage)
        return result

    def report(self) -> dict:
        ran = {entry["check"] for entry in self._checks}
        missing = [name for name in REQUIRED_CHECKS if name not in ran]
        if missing:
            raise IntegrityError(
                "report", f"required checks never ran: {', '.join(missing)}", expected=list(
                    REQUIRED_CHECKS
                ), actual=sorted(ran),
            )
        return {
            "schema": SCHEMA,
            **{name: True for name in REQUIRED_CHECKS},
            "frozen_tensors": self._counts["frozen"],
            "trainable_tensors": self._counts["trainable"],
            "frozen_digest_sha256": self._frozen_reference,
            "checks": list(self._checks),
        }
