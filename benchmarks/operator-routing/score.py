#!/usr/bin/env python3
"""Operator-routing v1: the independent scorer.

This file deliberately shares no code with generator.py and never imports it. It
reads the TSV twin (and, for the agreement check, the JSONL records), re-derives
every label, utility vector, margin and counterfactual tag with its own
implementation of the rule, and scores `PRED <arm> <split> <hex>` lines found in
the stdout of run records written by scripts/run_experiment.py.

    # 100% agreement between this re-derivation and the generator's records
    python3 benchmarks/operator-routing/score.py agree

    # route_accuracy, cost_adjusted_regret, task_success and the subset accuracies
    python3 benchmarks/operator-routing/score.py score experiments/model/M001-*/results/run-*.json

The rule is restated in README.md ("Label rule"). This implementation sums each
fact's vote and the budget penalty separately (z_k = sum of votes - n_usable *
penalty_k), which is the same quantity in a different float order: the 1e-9 tie
tolerance absorbs the rounding difference, and the agreement check proves it on
every record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "datasets/generated/operator_routing_v1"
CODEBOOK_JSON = ROOT / "datasets/generated/codebook.json"

SPLIT_NAMES = (
    "train", "val", "test_iid",
    "ood_compose_epi", "ood_compose_regime", "ood_distractors", "ood_validity", "ood_payload",
)
DECISIVE = ("validity", "regime", "budget", "confidence", "epistemic")
ALL_TAGS = DECISIVE + ("heldout", "transfer")
TOL = 1e-9
SUCCESS_REGRET = 0.25
CLEAR_MARGIN = 0.25


# ---------------------------------------------------------------- codebook (names)


def _families() -> dict[str, dict[str, int]]:
    doc = json.loads(CODEBOOK_JSON.read_text(encoding="utf-8"))
    table = {}
    for fam in doc["families"]:
        table[fam["family"]] = {m["name"]: int(m["code"]) for m in fam["members"]}
    return table


_CB = _families()
ROLES = _CB["semantic_role"]
STATES = _CB["epistemic_state"]
OPERATORS = _CB["reasoning_operator"]
VALIDITIES = _CB["validity"]
OPERATOR_BY_CODE = {code: name for name, code in OPERATORS.items()}
ROLE_BY_CODE = {code: name for name, code in ROLES.items()}
STATE_BY_CODE = {code: name for name, code in STATES.items()}
VALIDITY_BY_CODE = {code: name for name, code in VALIDITIES.items()}

# ------------------------------------------------------------- the rule, restated

REGIMES = ("tabular", "temporal", "interventional", "textual")
BUDGETS = (0.4, 0.7, 1.0)
CONFIDENCE_GAIN = {0: 0.0, 1: 0.5, 2: 1.0, 3: 1.0, 4: 1.25}
STATE_WEIGHT = {
    "unknown": 0.30, "assumed": 0.55, "hypothesis": 0.80,
    "observed": 1.30, "inferred": 1.05, "verified": 1.55,
}
PROBABILISTIC_BONUS = {
    "unknown": 0.35, "assumed": 0.60, "hypothesis": 0.85,
    "observed": 0.0, "inferred": 0.0, "verified": 0.0,
}
PENALTY_SCALE = 1.0
OPERATOR_COST = {
    "semantic": 0.2, "deductive": 0.3, "probabilistic": 0.5, "statistical": 0.5,
    "temporal": 0.5, "causal": 0.6, "search": 0.6, "optimization": 0.8,
    "simulation": 0.8, "symbolic": 0.4, "external_pod": 0.9,
}
REGIME_OPERATOR = {
    "tabular": "statistical", "temporal": "temporal",
    "interventional": "causal", "textual": "semantic",
}
HELD_OUT_PAIRS = {
    ("goal", "verified"), ("constraint", "hypothesis"), ("claim", "observed"),
    ("evidence", "assumed"), ("resource", "inferred"), ("capability", "unknown"),
    ("relation", "verified"), ("procedure", "hypothesis"), ("action", "observed"),
}


def votes_for(role: str, regime: str) -> dict[str, float]:
    """Primary 2.0 and secondary 0.9 affinities of one role under one regime."""
    by_regime = REGIME_OPERATOR[regime]
    primary, secondary = {
        "goal": ("search", "optimization"),
        "constraint": ("optimization", "symbolic"),
        "claim": ("deductive", by_regime),
        "evidence": (by_regime, "probabilistic"),
        "resource": ("external_pod", "search"),
        "capability": ("simulation", "external_pod"),
        "relation": ("causal", "deductive"),
        "procedure": ("symbolic", "simulation"),
        "action": ("external_pod", "temporal"),
    }[role]
    out = {primary: 2.0}
    out[secondary] = out.get(secondary, 0.0) + 0.9
    return out


def bucket_of(confidence_text: str) -> int:
    """floor(5c) from the 4-decimal text, in integer arithmetic: c = d / 10000."""
    whole, _, frac = confidence_text.partition(".")
    tenth_thousandths = int(whole) * 10000 + int((frac + "0000")[:4])
    return (5 * tenth_thousandths) // 10000


class Fact:
    __slots__ = ("role", "state", "conf", "bucket", "validity", "entity")

    def __init__(self, role, state, conf, bucket, validity, entity):
        self.role, self.state, self.conf = role, state, conf
        self.bucket, self.validity, self.entity = bucket, validity, entity


def scores(
    facts: list[Fact],
    regime: str,
    budget: float,
    *,
    everyone_admitted=False,
    flat_gain=False,
    flat_state=False,
    skip=None,
    evidence_regime=None,
) -> list[float]:
    names = [OPERATOR_BY_CODE[c] for c in range(len(OPERATOR_BY_CODE))]
    vote = dict.fromkeys(names, 0.0)
    usable = 0
    for fact in facts:
        if not everyone_admitted and fact.validity != "live":
            continue
        if skip is not None and skip(fact):
            continue
        gain = 1.0 if flat_gain else CONFIDENCE_GAIN[fact.bucket]
        if gain == 0.0:
            continue
        usable += 1
        weight = 1.0 if flat_state else STATE_WEIGHT[fact.state]
        bonus = 0.0 if flat_state else PROBABILISTIC_BONUS[fact.state]
        fact_regime = evidence_regime if (evidence_regime and fact.role == "evidence") else regime
        for op, affinity in votes_for(fact.role, fact_regime).items():
            vote[op] += gain * weight * affinity
        vote["probabilistic"] += gain * bonus
    return [
        vote[name] - usable * PENALTY_SCALE * max(0.0, OPERATOR_COST[name] - budget)
        for name in names
    ]


def winner(z: list[float]) -> int:
    top = max(z)
    candidates = [code for code, value in enumerate(z) if top - value <= TOL]
    candidates.sort(key=lambda code: (OPERATOR_COST[OPERATOR_BY_CODE[code]], code))
    return candidates[0]


class Item:
    """One example as the scorer sees it, with everything it re-derived."""

    __slots__ = ("id", "split", "gold", "regime", "budget", "tokens", "facts",
                 "z", "label", "margin", "tags")


def derive(item: Item) -> None:
    regime, budget = REGIMES[item.regime], BUDGETS[item.budget]
    z = scores(item.facts, regime, budget)
    item.z = z
    item.label = winner(z)
    top_two = sorted(z)[-2:]
    item.margin = top_two[1] - top_two[0]
    y = item.label
    tags = []
    if winner(scores(item.facts, regime, budget, everyone_admitted=True)) != y:
        tags.append("validity")
    if any(winner(scores(item.facts, other, budget)) != y for other in REGIMES if other != regime):
        tags.append("regime")
    if any(winner(scores(item.facts, regime, other)) != y for other in BUDGETS if other != budget):
        tags.append("budget")
    if winner(scores(item.facts, regime, budget, flat_gain=True)) != y:
        tags.append("confidence")
    if winner(scores(item.facts, regime, budget, flat_state=True)) != y:
        tags.append("epistemic")
    if item.split == "ood_compose_epi":
        held = winner(scores(item.facts, regime, budget,
                             skip=lambda f: (f.role, f.state) in HELD_OUT_PAIRS))
        if held != y:
            tags.append("heldout")
    if item.split == "ood_compose_regime":
        alternatives = [winner(scores(item.facts, regime, budget, skip=lambda f: f.role == "evidence"))]
        for other in ("tabular", "temporal", "textual"):
            alternatives.append(winner(scores(item.facts, regime, budget, evidence_regime=other)))
        if y not in alternatives:
            tags.append("transfer")
    item.tags = tags


# ------------------------------------------------------------------------ reading


def parse_tsv_line(line: str, split: str) -> Item:
    fields = line.rstrip("\n").split("\t")
    if len(fields) != 6:
        raise ValueError(f"{split}: expected 6 TSV fields, found {len(fields)}")
    item = Item()
    item.id, item.split = fields[0], split
    item.gold = int(fields[1])
    item.regime, item.budget = int(fields[2]), int(fields[3])
    item.tokens = [int(t) for t in fields[4].split(",")]
    facts = []
    for chunk in fields[5].split(";"):
        role, state, conf, validity, entity = chunk.split(",")
        facts.append(Fact(ROLE_BY_CODE[int(role)], STATE_BY_CODE[int(state)], conf,
                          bucket_of(conf), VALIDITY_BY_CODE[int(validity)], int(entity)))
    item.facts = facts
    return item


def load_split(data_dir: Path, split: str) -> list[Item]:
    items = []
    with (data_dir / f"{split}.tsv").open("r", encoding="utf-8", newline="\n") as handle:
        for line in handle:
            item = parse_tsv_line(line, split)
            derive(item)
            items.append(item)
    return items


def fnv1a64(data: bytes, h: int = 0xCBF29CE484222325) -> int:
    for b in data:
        h = ((h ^ b) * 0x100000001B3) % (1 << 64)
    return h


# ---------------------------------------------------------------------- agreement


def agree(data_dir: Path, splits=SPLIT_NAMES, verbose=True) -> tuple[int, list[str]]:
    """Every record: TSV gold == re-derived label == JSONL label; tags, margin and
    utility equal the re-derivation; the JSONL and TSV carry the same example."""
    problems: list[str] = []
    checked = 0
    for split in splits:
        items = load_split(data_dir, split)
        with (data_dir / f"{split}.jsonl").open("r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle]
        if len(records) != len(items):
            problems.append(f"{split}: {len(records)} JSONL records but {len(items)} TSV lines")
        for item, rec in zip(items, records):
            checked += 1
            where = f"{split}:{item.id}"
            if item.gold != item.label:
                problems.append(f"{where}: TSV label {item.gold} but the rule gives {item.label}")
            if rec["label"]["code"] != item.label or rec["label"]["operator"] != OPERATOR_BY_CODE[item.label]:
                problems.append(f"{where}: JSONL label {rec['label']} but the rule gives {item.label}")
            if rec["tags"] != item.tags:
                problems.append(f"{where}: JSONL tags {rec['tags']} but re-derived {item.tags}")
            if abs(rec["margin"] - item.margin) > 1e-8:
                problems.append(f"{where}: JSONL margin {rec['margin']} but re-derived {item.margin}")
            if any(abs(a - b) > 1e-8 for a, b in zip(rec["utility"], item.z)):
                problems.append(f"{where}: JSONL utility differs from the re-derivation")
            problems.extend(round_trip_problems(where, item, rec))
        if verbose:
            print(f"  {split}: {len(items)} records re-derived", file=sys.stderr)
    return checked, problems


def round_trip_problems(where: str, item: Item, rec: dict) -> list[str]:
    out = []
    if rec["id"] != item.id or rec["split"] != item.split:
        out.append(f"{where}: JSONL id/split {rec['id']}/{rec['split']}")
    if rec["regime"]["code"] != item.regime or rec["regime"]["name"] != REGIMES[item.regime]:
        out.append(f"{where}: JSONL regime {rec['regime']}")
    if rec["budget"]["code"] != item.budget or rec["cost_budget"] != BUDGETS[item.budget]:
        out.append(f"{where}: JSONL budget {rec['budget']}")
    if rec["raw_tokens"] != item.tokens:
        out.append(f"{where}: JSONL raw_tokens differ from the TSV tokens")
    if len(rec["slots"]) != len(item.facts):
        out.append(f"{where}: {len(rec['slots'])} JSONL slots but {len(item.facts)} TSV facts")
    for index, (slot, fact) in enumerate(zip(rec["slots"], item.facts)):
        same = (
            slot["index"] == index and slot["provenance_bucket"] == index
            and slot["role"] == fact.role and slot["epistemic"] == fact.state
            and f"{slot['confidence']:.4f}" == fact.conf
            and slot["confidence_bucket"] == fact.bucket
            and slot["validity"] == fact.validity and slot["entity"] == fact.entity
        )
        if not same:
            out.append(f"{where}: slot {index} differs between JSONL and TSV")
    return out


# ------------------------------------------------------------------------ scoring

PRED_LINE = re.compile(r"^PRED (\S+) (\S+) ([0-9a-fA-F]*)$")


def pred_lines(stdout: str) -> list[tuple[str, str, str]]:
    found = []
    for number, line in enumerate(stdout.splitlines(), 1):
        if not line.startswith("PRED"):
            continue
        match = PRED_LINE.match(line.strip())
        if not match:
            raise ValueError(f"stdout line {number} is a malformed PRED line")
        found.append(match.groups())
    return found


def json_rows(stdout: str) -> list[dict]:
    rows = []
    for line in stdout.splitlines():
        text = line.strip()
        if text.startswith("{"):
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _rate(correct: int, n: int):
    return {"n": n, "correct": correct, "accuracy": (correct / n) if n else None}


def score_predictions(items: list[Item], predictions: list[int]) -> dict:
    if len(items) != len(predictions):
        raise ValueError(f"{len(predictions)} predictions for {len(items)} examples")
    n = len(items)
    correct = 0
    regret_sum = 0.0
    success = 0
    subsets = {tag: [0, 0] for tag in ALL_TAGS + ("clear_margin",)}
    for item, pred in zip(items, predictions):
        if not 0 <= pred < len(OPERATOR_BY_CODE):
            raise ValueError(f"{item.id}: prediction {pred} is not an operator code")
        hit = int(pred == item.label)
        correct += hit
        regret = item.z[item.label] - item.z[pred]
        regret_sum += regret
        success += int(regret <= SUCCESS_REGRET + TOL)
        for tag in item.tags:
            subsets[tag][0] += 1
            subsets[tag][1] += hit
        if item.margin >= CLEAR_MARGIN - TOL:
            subsets["clear_margin"][0] += 1
            subsets["clear_margin"][1] += hit
    split = items[0].split if items else None
    report = {
        "n": n,
        "correct": correct,
        "route_accuracy": correct / n,
        "cost_adjusted_regret": regret_sum / n,
        "task_success": success / n,
        "subsets": {},
    }
    for tag in DECISIVE + ("clear_margin",):
        report["subsets"][tag] = _rate(subsets[tag][1], subsets[tag][0])
    if split == "ood_compose_epi":
        report["subsets"]["heldout"] = _rate(subsets["heldout"][1], subsets["heldout"][0])
    if split == "ood_compose_regime":
        report["subsets"]["transfer"] = _rate(subsets["transfer"][1], subsets["transfer"][0])
    return report


def rust_final_row(rows: list[dict], arm: str, split: str):
    """The binary's final row for (arm, split): any JSON row whose `arm` and
    `split` fields name them and that carries a numeric `correct`."""
    for row in rows:
        if row.get("arm") == arm and row.get("split") == split and isinstance(row.get("correct"), (int, float)) \
                and not isinstance(row.get("correct"), bool):
            return row
    return None


def score_records(paths: list[Path], data_dir: Path) -> tuple[dict, list[str]]:
    cache: dict[str, list[Item]] = {}
    problems: list[str] = []
    results = []
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        stdout = record.get("stdout")
        if not isinstance(stdout, str):
            problems.append(f"{path}: no stdout text")
            continue
        rows = json_rows(stdout)
        for arm, split, hexes in pred_lines(stdout):
            if split not in SPLIT_NAMES:
                problems.append(f"{path}: PRED {arm} names unknown split {split!r}")
                continue
            if split not in cache:
                cache[split] = load_split(data_dir, split)
            items = cache[split]
            try:
                report = score_predictions(items, [int(ch, 16) for ch in hexes])
            except ValueError as error:
                problems.append(f"{path}: PRED {arm} {split}: {error}")
                continue
            entry = {
                "record": str(path),
                "experiment_id": record.get("experiment_id"),
                "seed": record.get("seed"),
                "arm": arm,
                "split": split,
                **report,
            }
            final = rust_final_row(rows, arm, split)
            if final is not None:
                entry["rust"] = {k: final.get(k) for k in ("n", "correct", "accuracy", "nll", "ece15")}
                entry["rust_count_agrees"] = final.get("n") == report["n"] and final.get("correct") == report["correct"]
                if not entry["rust_count_agrees"]:
                    problems.append(
                        f"{path}: {arm}/{split}: Rust counts {final.get('correct')}/{final.get('n')} "
                        f"but score.py counts {report['correct']}/{report['n']} (gate G5)"
                    )
            else:
                entry["rust"] = None
                entry["rust_count_agrees"] = None
            results.append(entry)
    tsv_bytes = b"".join((data_dir / f"{s}.tsv").read_bytes() for s in SPLIT_NAMES
                         if (data_dir / f"{s}.tsv").is_file())
    document = {
        "data_dir": str(data_dir),
        "data_fnv1a64": f"{fnv1a64(tsv_bytes):016x}",
        "results": results,
    }
    return document, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_DIR, help="directory holding <split>.tsv/.jsonl")
    sub = parser.add_subparsers(dest="command", required=True)
    agree_cmd = sub.add_parser("agree", help="check generator/score agreement on every record")
    agree_cmd.add_argument("--splits", nargs="*", default=list(SPLIT_NAMES))
    score_cmd = sub.add_parser("score", help="score PRED lines in run records")
    score_cmd.add_argument("records", nargs="+", type=Path)
    score_cmd.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.command == "agree":
        checked, problems = agree(args.data, args.splits)
        for problem in problems[:50]:
            print(f"error: {problem}", file=sys.stderr)
        if problems:
            print(f"FAIL: {len(problems)} disagreement(s) in {checked} records", file=sys.stderr)
            return 1
        print(f"OK: score.py agrees with the generator on {checked}/{checked} records "
              f"(label, tags, margin, utility, TSV/JSONL round trip)")
        return 0

    document, problems = score_records(args.records, args.data)
    text = json.dumps(document, indent=2) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
