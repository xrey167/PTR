#!/usr/bin/env python3
"""Independent label derivation for operator-routing v2 TSV records."""

from __future__ import annotations

from dataclasses import dataclass

ROLE_NAMES = ["goal", "constraint", "claim", "evidence", "resource", "capability", "relation", "procedure", "action"]
EPI_NAMES = ["unknown", "assumed", "hypothesis", "observed", "inferred", "verified"]
OP_NAMES = ["semantic", "deductive", "probabilistic", "statistical", "temporal", "causal", "search", "optimization", "simulation", "symbolic", "external_pod"]
REGIME_NAMES = ["tabular", "temporal", "interventional", "textual"]
BUDGETS = [0.4, 0.7, 1.0]
LIVE = 0
G = [0.0, 0.5, 1.0, 1.0, 1.25]
W = [0.30, 0.55, 0.80, 1.30, 1.05, 1.55]
U = [0.35, 0.60, 0.85, 0.0, 0.0, 0.0]
COST = [0.2, 0.3, 0.5, 0.5, 0.5, 0.6, 0.6, 0.8, 0.8, 0.4, 0.9]
REGIME_OP = [3, 4, 5, 0]
AFFINITY_NAMES = {
    "goal": ("search", "optimization"), "constraint": ("optimization", "symbolic"),
    "claim": ("deductive", None), "evidence": (None, "probabilistic"),
    "resource": ("external_pod", "search"), "capability": ("simulation", "external_pod"),
    "relation": ("causal", "deductive"), "procedure": ("symbolic", "simulation"),
    "action": ("external_pod", "temporal"),
}


@dataclass
class Item:
    id: str
    gold: int
    regime: int
    budget: int
    tokens: list[int]
    facts: list[tuple[int, int, float, int, int]]


def parse_tsv_line(line: str) -> Item:
    ident, gold, regime, budget, tokens, encoded_facts = line.rstrip("\n").split("\t")
    facts = []
    for encoded in encoded_facts.split(";"):
        role, epi, confidence, validity, entity = encoded.split(",")
        facts.append((int(role), int(epi), float(confidence), int(validity), int(entity)))
    return Item(
        ident, int(gold), int(regime), int(budget),
        [int(token) for token in tokens.split(",")], facts,
    )


def affinity(role: int, regime: int) -> list[float]:
    primary, secondary = AFFINITY_NAMES[ROLE_NAMES[role]]
    vector = [0.0] * len(OP_NAMES)
    vector[REGIME_OP[regime] if primary is None else OP_NAMES.index(primary)] += 2.0
    vector[REGIME_OP[regime] if secondary is None else OP_NAMES.index(secondary)] += 0.9
    return vector


def derive(item: Item) -> tuple[int, list[float]]:
    values = [0.0] * len(OP_NAMES)
    for role, epi, confidence, validity, _entity in item.facts:
        confidence_bucket = int(5.0 * confidence)
        if validity != LIVE or confidence_bucket == 0:
            continue
        votes = affinity(role, item.regime)
        for operator in range(len(OP_NAMES)):
            vote = W[epi] * votes[operator]
            if operator == 2:
                vote += U[epi]
            values[operator] += G[confidence_bucket] * vote - max(0.0, COST[operator] - BUDGETS[item.budget])
    best = max(values)
    tied = [operator for operator, value in enumerate(values) if value >= best - 1e-9]
    return min(tied, key=lambda operator: (COST[operator], operator)), values
