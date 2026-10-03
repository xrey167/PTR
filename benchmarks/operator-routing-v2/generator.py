#!/usr/bin/env python3
"""Deterministic operator-routing v2 fold generator.

The three confirmatory folds hold one role x regime cell out of train and val
and require that cell as a live, nonzero fact in test_ood.  The historical
Evidence x interventional is retained as a diagnostic fold. Claim x temporal
is a separate development fold used only by the non-claimable pilot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_OUT = ROOT / "datasets/generated/operator_routing_v2"
LOCK_PATH = HERE / "splits.lock.json"
SAMPLE_PATH = HERE / "samples.jsonl"
CODEBOOK_PATH = ROOT / "datasets/generated/codebook.json"

GENERATOR_VERSION = 2
BENCHMARK_SEED = 20261001
CODEBOOK_VERSION = 1
CODEBOOK_FINGERPRINT = "2b6f8175a7a7bb648910e7acb17dcfcff5149834f2fe87355c5d2c4524c113a3"

MASK64 = (1 << 64) - 1
GOLDEN_GAMMA = 0x9E3779B97F4A7C15
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x00000100000001B3
TWO_POW_53 = 9007199254740992.0


def mix64(value: int) -> int:
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return value ^ (value >> 31)


def splitmix64(value: int) -> int:
    return mix64((value + GOLDEN_GAMMA) & MASK64)


def fnv1a64(data: bytes, state: int = FNV_OFFSET) -> int:
    for byte in data:
        state ^= byte
        state = (state * FNV_PRIME) & MASK64
    return state


class Stream:
    __slots__ = ("state",)

    def __init__(self, seed: int) -> None:
        self.state = seed & MASK64

    def next(self) -> int:
        self.state = (self.state + GOLDEN_GAMMA) & MASK64
        return mix64(self.state)

    def u01(self) -> float:
        return (self.next() >> 11) / TWO_POW_53

    def randbelow(self, n: int) -> int:
        return int(self.u01() * n)


def load_codebook() -> dict[str, dict[str, int]]:
    document = json.loads(CODEBOOK_PATH.read_text(encoding="utf-8"))
    digest = hashlib.sha256(bytes.fromhex(document["canonical_bytes_hex"])).hexdigest()
    if digest != document["fingerprint_sha256"] or digest != CODEBOOK_FINGERPRINT:
        raise SystemExit(f"{CODEBOOK_PATH}: unexpected codebook fingerprint {digest}")
    if document["version"] != CODEBOOK_VERSION:
        raise SystemExit(f"{CODEBOOK_PATH}: unexpected codebook version {document['version']}")
    return {
        family["family"]: {member["name"]: member["code"] for member in family["members"]}
        for family in document["families"]
    }


CODEBOOK = load_codebook()
ROLE = CODEBOOK["semantic_role"]
EPI = CODEBOOK["epistemic_state"]
OP = CODEBOOK["reasoning_operator"]
VALIDITY = CODEBOOK["validity"]
ROLE_NAMES = sorted(ROLE, key=ROLE.get)
EPI_NAMES = sorted(EPI, key=EPI.get)
OP_NAMES = sorted(OP, key=OP.get)
VALIDITY_NAMES = sorted(VALIDITY, key=VALIDITY.get)

REGIME_NAMES = ["tabular", "temporal", "interventional", "textual"]
REGIME = {name: index for index, name in enumerate(REGIME_NAMES)}
BUDGET = [0.4, 0.7, 1.0]
LIVE = VALIDITY["live"]
NON_LIVE = [VALIDITY["superseded"], VALIDITY["revoked"], VALIDITY["disputed"]]
PROBABILISTIC = OP["probabilistic"]


@dataclass(frozen=True)
class Fold:
    name: str
    role: int
    regime: int
    kind: str

    @property
    def confirmatory(self) -> bool:
        return self.kind == "confirmatory"


FOLDS = (
    Fold("evidence-temporal", ROLE["evidence"], REGIME["temporal"], "confirmatory"),
    Fold("evidence-tabular", ROLE["evidence"], REGIME["tabular"], "confirmatory"),
    Fold("claim-interventional", ROLE["claim"], REGIME["interventional"], "confirmatory"),
    Fold("evidence-interventional", ROLE["evidence"], REGIME["interventional"], "diagnostic"),
    Fold("claim-temporal-development", ROLE["claim"], REGIME["temporal"], "development"),
)
FOLD_BY_NAME = {fold.name: fold for fold in FOLDS}
CONFIRMATORY_FOLDS = tuple(fold.name for fold in FOLDS if fold.confirmatory)
DIAGNOSTIC_FOLDS = tuple(fold.name for fold in FOLDS if fold.kind == "diagnostic")
DEVELOPMENT_FOLDS = tuple(fold.name for fold in FOLDS if fold.kind == "development")
SPLITS = ("train", "val", "test_iid", "test_ood")
SPLIT_SIZES = {"train": 2048, "val": 256, "test_iid": 512, "test_ood": 512}
PREFIX_N = 64

ROLE_PRIOR = [
    ROLE[name]
    for name in (
        "goal", "constraint", "claim", "claim", "evidence", "evidence",
        "resource", "capability", "relation", "procedure", "action",
    )
]
G = [0.0, 0.5, 1.0, 1.0, 1.25]
_W = {"unknown": 0.30, "assumed": 0.55, "hypothesis": 0.80, "observed": 1.30, "inferred": 1.05, "verified": 1.55}
_U = {"unknown": 0.35, "assumed": 0.60, "hypothesis": 0.85, "observed": 0.0, "inferred": 0.0, "verified": 0.0}
W = [_W[name] for name in EPI_NAMES]
U = [_U[name] for name in EPI_NAMES]
_COST = {
    "semantic": 0.2, "deductive": 0.3, "probabilistic": 0.5, "statistical": 0.5,
    "temporal": 0.5, "causal": 0.6, "search": 0.6, "optimization": 0.8,
    "simulation": 0.8, "symbolic": 0.4, "external_pod": 0.9,
}
COST = [_COST[name] for name in OP_NAMES]
REGIME_OP = [OP["statistical"], OP["temporal"], OP["causal"], OP["semantic"]]
_REGIME = "op(R)"
_AFFINITY_TABLE = {
    "goal": ("search", "optimization"), "constraint": ("optimization", "symbolic"),
    "claim": ("deductive", _REGIME), "evidence": (_REGIME, "probabilistic"),
    "resource": ("external_pod", "search"), "capability": ("simulation", "external_pod"),
    "relation": ("causal", "deductive"), "procedure": ("symbolic", "simulation"),
    "action": ("external_pod", "temporal"),
}
TIE_TOL = 1e-9
SOFTMAX_TEMPERATURE = 0.5
N_SLOTS = 6
VOCAB = 216


def _op_code(name: str, regime: int) -> int:
    return REGIME_OP[regime] if name == _REGIME else OP[name]


AFFINITY: list[list[list[float]]] = []
for _role in range(len(ROLE)):
    per_regime = []
    for _regime in range(len(REGIME_NAMES)):
        vector = [0.0] * len(OP)
        primary, secondary = _AFFINITY_TABLE[ROLE_NAMES[_role]]
        vector[_op_code(primary, _regime)] += 2.0
        vector[_op_code(secondary, _regime)] += 0.9
        per_regime.append(vector)
    AFFINITY.append(per_regime)


def confidence(draw: int) -> float:
    return (draw + 0.5) / 1000.0


def bucket(value: float) -> int:
    return int(5.0 * value)


def example_seed(fold: str, split: str, index: int) -> int:
    key = f"{fold}/{split}".encode("utf-8")
    return splitmix64(BENCHMARK_SEED ^ fnv1a64(key) ^ splitmix64(index))


@dataclass
class Example:
    fold: Fold
    split: str
    index: int
    regime: int
    budget: int
    facts: list[tuple[int, int, float, int, int]]
    tokens: list[int]

    @property
    def id(self) -> str:
        return f"{self.fold.name}-{self.split}-{self.index:05d}"


def contains_target(fold: Fold, regime: int, facts, *, usable_only: bool) -> bool:
    if regime != fold.regime:
        return False
    return any(
        role == fold.role and (not usable_only or (validity == LIVE and bucket(c) > 0))
        for role, _epi, c, validity, _entity in facts
    )


def sample(fold_name: str, split: str, index: int) -> Example:
    """Draw one fold example; rejection enforces the fold's contamination contract."""
    if fold_name not in FOLD_BY_NAME or split not in SPLITS:
        raise ValueError(f"unknown fold/split: {fold_name}/{split}")
    fold = FOLD_BY_NAME[fold_name]
    stream = Stream(example_seed(fold_name, split, index))
    while True:
        regime = fold.regime if split == "test_ood" else stream.randbelow(4)
        budget = stream.randbelow(3)
        facts = []
        for _ in range(N_SLOTS):
            role = ROLE_PRIOR[stream.randbelow(len(ROLE_PRIOR))]
            epi = stream.randbelow(len(EPI))
            c = confidence(stream.randbelow(1000))
            validity = LIVE if stream.u01() < 0.78 else NON_LIVE[stream.randbelow(len(NON_LIVE))]
            entity = stream.randbelow(256)
            facts.append((role, epi, c, validity, entity))
        if not any(fact[3] == LIVE for fact in facts):
            continue
        if split in ("train", "val", "test_iid") and contains_target(
            fold, regime, facts, usable_only=False
        ):
            continue
        if split == "test_ood" and not contains_target(fold, regime, facts, usable_only=True):
            continue
        break

    tokens = []
    for slot, (role, epi, c, validity, _entity) in enumerate(facts):
        base = 1 + 24 * slot
        tokens.extend((base + role, base + 9 + epi, base + 15 + bucket(c), base + 20 + validity))
    tokens.extend((145 + regime, 149 + budget))
    tokens.extend(152 + stream.randbelow(64) for _ in range(10))
    for position in range(len(tokens) - 1, 0, -1):
        swap = stream.randbelow(position + 1)
        tokens[position], tokens[swap] = tokens[swap], tokens[position]
    return Example(fold, split, index, regime, budget, facts, tokens)


def utilities(facts, regime: int, budget: int) -> list[float]:
    values = [0.0] * len(OP)
    for role, epi, c, validity, _entity in facts:
        if validity != LIVE or bucket(c) == 0:
            continue
        gain = G[bucket(c)]
        for operator in range(len(OP)):
            vote = W[epi] * AFFINITY[role][regime][operator]
            if operator == PROBABILISTIC:
                vote += U[epi]
            penalty = max(0.0, COST[operator] - BUDGET[budget])
            values[operator] += gain * vote - penalty
    return values


def decide(values: list[float]) -> int:
    best = max(values)
    tied = [operator for operator, value in enumerate(values) if value >= best - TIE_TOL]
    return min(tied, key=lambda operator: (COST[operator], operator))


def route_targets(values: list[float]) -> list[float]:
    scaled = [value / SOFTMAX_TEMPERATURE for value in values]
    top = max(scaled)
    weights = [math.exp(value - top) for value in scaled]
    total = sum(weights)
    micros = [int(round(weight / total * 1_000_000)) for weight in weights]
    largest = min(range(len(OP)), key=lambda operator: (-micros[operator], operator))
    micros[largest] += 1_000_000 - sum(micros)
    return [value / 1_000_000 for value in micros]


def build(example: Example) -> tuple[dict, str, int]:
    values = utilities(example.facts, example.regime, example.budget)
    label = decide(values)
    ordered = sorted(values, reverse=True)
    target = {"role": ROLE_NAMES[example.fold.role], "regime": REGIME_NAMES[example.fold.regime]}
    record = {
        "id": example.id,
        "fold": example.fold.name,
        "fold_kind": example.fold.kind,
        "target_cell": target,
        "split": example.split,
        "generator_version": GENERATOR_VERSION,
        "task": render_task(example),
        "routes": [
            {"operator": OP_NAMES[index], "target": probability}
            for index, probability in enumerate(route_targets(values))
        ],
        "cost_budget": BUDGET[example.budget],
        "type_codebook_version": CODEBOOK_VERSION,
        "type_codebook_fingerprint": CODEBOOK_FINGERPRINT,
        "regime": {"code": example.regime, "name": REGIME_NAMES[example.regime]},
        "budget": {"code": example.budget, "value": BUDGET[example.budget]},
        "slots": [
            {
                "index": slot, "role": ROLE_NAMES[role], "epistemic": EPI_NAMES[epi],
                "confidence": c, "confidence_bucket": bucket(c),
                "validity": VALIDITY_NAMES[validity], "entity": entity,
                "provenance_bucket": slot,
            }
            for slot, (role, epi, c, validity, entity) in enumerate(example.facts)
        ],
        "raw_tokens": list(example.tokens),
        "label": {"operator": OP_NAMES[label], "code": label},
        "utility": [round(value, 10) + 0.0 for value in values],
        "margin": round(ordered[0] - ordered[1], 10) + 0.0,
    }
    facts = ";".join(
        f"{role},{epi},{c:.4f},{validity},{entity}"
        for role, epi, c, validity, entity in example.facts
    )
    line = (
        f"{example.id}\t{label}\t{example.regime}\t{example.budget}\t"
        f"{','.join(str(token) for token in example.tokens)}\t{facts}\n"
    )
    return record, line, label


def render_task(example: Example) -> str:
    facts = "; ".join(
        f"[{slot}] {ROLE_NAMES[role]} ({EPI_NAMES[epi]}, confidence {c:.4f}, "
        f"{VALIDITY_NAMES[validity]}) about entity-{entity}"
        for slot, (role, epi, c, validity, entity) in enumerate(example.facts)
    )
    return (
        f"Route this {REGIME_NAMES[example.regime]} problem to one reasoning operator "
        f"under a cost budget of {BUDGET[example.budget]:.1f}. Facts: {facts}."
    )


def jsonl_line(record: dict) -> str:
    return json.dumps(record, separators=(",", ":"), ensure_ascii=True) + "\n"


def render_split(fold: str, split: str, n: int | None = None) -> tuple[bytes, bytes, str]:
    count = SPLIT_SIZES[split] if n is None else n
    jsonl, tsv, labels = [], [], []
    for index in range(count):
        record, line, label = build(sample(fold, split, index))
        jsonl.append(jsonl_line(record))
        tsv.append(line)
        labels.append(format(label, "x"))
    return "".join(jsonl).encode(), "".join(tsv).encode(), "".join(labels)


def render_all(verbose: bool = True) -> dict[str, tuple[bytes, bytes, str]]:
    rendered = {}
    for fold in FOLDS:
        for split in SPLITS:
            key = f"{fold.name}/{split}"
            rendered[key] = render_split(fold.name, split)
            if verbose:
                print(f"  {key}: {SPLIT_SIZES[split]} examples", file=sys.stderr)
    return rendered


def hex64(value: int) -> str:
    return f"{value:016x}"


def lock_document(rendered: dict[str, tuple[bytes, bytes, str]]) -> dict:
    state = FNV_OFFSET
    entries = {}
    for fold in FOLDS:
        fold_entries = {}
        fold_state = FNV_OFFSET
        for split in SPLITS:
            key = f"{fold.name}/{split}"
            jsonl, tsv, labels = rendered[key]
            state = fnv1a64(tsv, state)
            fold_state = fnv1a64(tsv, fold_state)
            fold_entries[split] = {
                "n": SPLIT_SIZES[split],
                "jsonl_sha256": hashlib.sha256(jsonl).hexdigest(),
                "tsv_sha256": hashlib.sha256(tsv).hexdigest(),
                "label_fnv1a64": hex64(fnv1a64(labels.encode("ascii"))),
                "prefix_jsonl_sha256": hashlib.sha256(b"".join(jsonl.splitlines(True)[:PREFIX_N])).hexdigest(),
                "prefix_tsv_sha256": hashlib.sha256(b"".join(tsv.splitlines(True)[:PREFIX_N])).hexdigest(),
            }
        entries[fold.name] = {
            "kind": fold.kind,
            "target_role": ROLE_NAMES[fold.role],
            "target_regime": REGIME_NAMES[fold.regime],
            "data_fnv1a64": hex64(fold_state),
            "splits": fold_entries,
        }
    return {
        "generator_version": GENERATOR_VERSION,
        "benchmark_seed": BENCHMARK_SEED,
        "type_codebook_version": CODEBOOK_VERSION,
        "type_codebook_fingerprint": CODEBOOK_FINGERPRINT,
        "prefix_n": PREFIX_N,
        "data_fnv1a64": hex64(state),
        "folds": entries,
    }


def compare(expected, actual, path: str = "") -> list[str]:
    problems = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in expected:
                problems.append(f"{path}{key}: absent from lock")
            elif key not in actual:
                problems.append(f"{path}{key}: absent from regenerated data")
            else:
                problems.extend(compare(expected[key], actual[key], f"{path}{key}."))
    elif expected != actual:
        problems.append(f"{path.rstrip('.')}: lock {expected!r} != regenerated {actual!r}")
    return problems


def check_on_disk(out: Path, lock: dict) -> list[str]:
    problems = []
    for fold in FOLDS:
        for split in SPLITS:
            entry = lock["folds"][fold.name]["splits"][split]
            for kind in ("jsonl", "tsv"):
                path = out / fold.name / f"{split}.{kind}"
                if not path.is_file():
                    problems.append(f"{path}: required versioned dataset file is missing")
                    continue
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != entry[f"{kind}_sha256"]:
                    problems.append(f"{path}: sha256 {digest} != lock {entry[f'{kind}_sha256']}")
    return problems


def write_data(out: Path, rendered: dict[str, tuple[bytes, bytes, str]]) -> None:
    for fold in FOLDS:
        directory = out / fold.name
        directory.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            jsonl, tsv, _labels = rendered[f"{fold.name}/{split}"]
            (directory / f"{split}.jsonl").write_bytes(jsonl)
            (directory / f"{split}.tsv").write_bytes(tsv)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--update-lock", action="store_true")
    parser.add_argument("--write-sample", nargs="?", const=SAMPLE_PATH, type=Path)
    args = parser.parse_args(argv)

    if args.write_sample is not None:
        records = [jsonl_line(build(sample(fold.name, split, 0))[0]) for fold in FOLDS for split in SPLITS]
        args.write_sample.write_text("".join(records), encoding="utf-8")
        if not (args.check or args.update_lock):
            return 0

    rendered = render_all()
    actual = lock_document(rendered)
    if args.check:
        if not args.lock.is_file():
            print(f"error: {args.lock} does not exist", file=sys.stderr)
            return 1
        expected = json.loads(args.lock.read_text(encoding="utf-8"))
        problems = compare(expected, actual) + check_on_disk(args.out, expected)
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        if problems:
            return 1
        print(f"OK: operator-routing-v2 matches {args.lock.name} ({actual['data_fnv1a64']})")
        return 0

    write_data(args.out, rendered)
    if args.update_lock:
        args.lock.write_text(json.dumps(actual, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.lock}")
        return 0
    if args.lock.is_file():
        problems = compare(json.loads(args.lock.read_text(encoding="utf-8")), actual)
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        if problems:
            return 1
    print(f"wrote operator-routing-v2 to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
