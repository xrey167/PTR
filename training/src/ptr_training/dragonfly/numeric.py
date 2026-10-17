"""Reductions that give the same bits on every supported Python.

CPython 3.12 made the builtin `sum()` of floats use compensated (Neumaier) summation, so
`sum(floats)` differs in the last bits between 3.11 and 3.12 and later. The exported weights and
the committed fixture must not depend on the interpreter, so every float reduction in this package
is this explicit left-to-right loop (the order `ptr-router` also uses).
"""

from __future__ import annotations

from typing import Iterable


def ordered_sum(values: Iterable[float]) -> float:
    """Left-to-right float64 sum, starting from 0.0."""
    total = 0.0
    for value in values:
        total += value
    return total
