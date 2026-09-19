from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REASONING_OPERATORS = {
    "semantic",
    "deductive",
    "probabilistic",
    "statistical",
    "temporal",
    "causal",
    "search",
    "optimization",
    "simulation",
    "symbolic",
    "external_pod",
}

EPISTEMIC_STATES = {
    "unknown",
    "assumed",
    "hypothesis",
    "observed",
    "inferred",
    "verified",
}

UNCERTAINTY_KINDS = {
    "point",
    "interval",
    "distribution",
}


def _probability(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be in [0,1]")
    return number


def _validate_operator_route(obj: dict[str, Any]) -> None:
    if not isinstance(obj.get("task"), str) or not obj["task"].strip():
        raise ValueError("task must be a non-empty string")

    routes = obj.get("routes")
    if not isinstance(routes, list) or not routes:
        raise ValueError("routes must be a non-empty array")

    total = 0.0
    for index, route in enumerate(routes):
        if not isinstance(route, dict):
            raise ValueError(f"routes[{index}] must be an object")
        operator = route.get("operator")
        if operator not in REASONING_OPERATORS:
            raise ValueError(f"routes[{index}].operator unknown: {operator!r}")
        total += _probability(route.get("target"), f"routes[{index}].target")

    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"route targets must sum to 1.0, got {total}")

    cost_budget = obj.get("cost_budget")
    if cost_budget is not None and (
        not isinstance(cost_budget, (int, float)) or float(cost_budget) < 0.0
    ):
        raise ValueError("cost_budget must be a non-negative number")

    codebook = obj.get("type_codebook_version")
    if codebook is not None and (not isinstance(codebook, str) or not codebook.strip()):
        raise ValueError("type_codebook_version must be a non-empty string")


def _validate_epistemic_calibration(obj: dict[str, Any]) -> None:
    if not isinstance(obj.get("question"), str) or not obj["question"].strip():
        raise ValueError("question must be a non-empty string")

    state = obj.get("epistemic_state")
    if state is not None and state not in EPISTEMIC_STATES:
        raise ValueError(f"unknown epistemic_state: {state!r}")

    uncertainty = obj.get("uncertainty_kind")
    if uncertainty is not None and uncertainty not in UNCERTAINTY_KINDS:
        raise ValueError(f"unknown uncertainty_kind: {uncertainty!r}")

    distribution = obj.get("distribution")
    if not isinstance(distribution, dict) or not distribution:
        raise ValueError("distribution must be a non-empty object")

    total = 0.0
    for label, value in distribution.items():
        if not isinstance(label, str) or not label:
            raise ValueError("distribution labels must be non-empty strings")
        total += _probability(value, f"distribution[{label!r}]")

    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"distribution probabilities must sum to 1.0, got {total}")

    if "outcome" not in obj:
        raise ValueError("outcome is required")

    codebook = obj.get("type_codebook_version")
    if codebook is not None and (not isinstance(codebook, str) or not codebook.strip()):
        raise ValueError("type_codebook_version must be a non-empty string")


def validate_record(path: Path, obj: dict[str, Any]) -> None:
    match path.name:
        case "operator_route.jsonl":
            _validate_operator_route(obj)
        case "epistemic_calibration.jsonl":
            _validate_epistemic_calibration(obj)
        case _:
            # Generic imported/generated datasets retain the base JSON-object
            # contract until their canonical schema receives an executable validator.
            return


def validate(path: Path) -> tuple[int, int]:
    ok = bad = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError("record must be object")
                validate_record(path, obj)
                ok += 1
            except Exception as exc:
                bad += 1
                print(f"{path}:{line_no}: {exc}")
    return ok, bad


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    ok, bad = validate(args.path)
    print(f"ok={ok} bad={bad}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
