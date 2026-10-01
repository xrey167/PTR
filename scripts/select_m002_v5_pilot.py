"""Select the non-claimable M002-v5 pilot candidate by the frozen rule."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = Path("benchmarks/operator-routing-v2/splits.lock.json")
SEEDS = (7, 13)
FOLDS = ("evidence-interventional", "claim-temporal-development")
ARMS = ("factorized-v2", "factorized-v2-off")
CANDIDATES = tuple(
    (rank, limit, dropout)
    for rank in (8, 16)
    for limit in (1.0, 2.0)
    for dropout in (0.0, 0.1)
)
RANKS = (8, 16)
BIAS_LIMITS = (1, 2)
METADATA_DROPOUTS = (0.0, 0.1)
SOURCE_SHA = re.compile(r"[0-9a-f]{40}")
DIGEST = re.compile(r"[0-9a-f]{16}")


class PilotError(ValueError):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fixed_protocol() -> dict:
    return {
        "phase": "paired-v5",
        "experiment": "M002-v5",
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "folds": list(FOLDS),
        "ranks": list(RANKS),
        "bias_limits": list(BIAS_LIMITS),
        "metadata_dropouts": list(METADATA_DROPOUTS),
        "d_model": 48,
        "steps": 1500,
        "learning_rate": 0.005,
    }


def command_for(candidate: dict, seed: int, digests: dict[str, str]) -> list[str]:
    folds = ",".join(f"{name}={digests[name]}" for name in FOLDS)
    return [
        "cargo", "+1.95.0-x86_64-pc-windows-gnu", "run", "--release",
        "--locked", "--quiet", "--jobs", "1", "--manifest-path",
        "model/burn-a0/Cargo.toml", "--example", "a0_ablation", "--",
        "--phase", "paired-v5", "--experiment", "M002-v5", "--arms",
        "factorized-v2,factorized-v2-off", "--seed", str(seed), "--steps",
        "1500", "--lr", "0.005", "--data",
        "datasets/generated/operator_routing_v2", "--folds", folds,
        "--d-model", "48", "--rank", str(candidate["rank"]),
        "--bias-limit", str(candidate["bias_limit"]), "--metadata-dropout",
        str(candidate["metadata_dropout"]),
    ]


def locked_fold_digests(root: Path = ROOT) -> dict[str, str]:
    path = root / LOCK_PATH
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        result = {name: document["folds"][name]["data_fnv1a64"] for name in FOLDS}
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise PilotError(f"cannot read fixed fold digests from {path}: {error}") from None
    if any(not isinstance(value, str) or not DIGEST.fullmatch(value) for value in result.values()):
        raise PilotError(f"{path}: fixed fold digests are malformed")
    return result


def metadata_candidate(value: object, name: str) -> dict[str, int | float]:
    if not isinstance(value, dict) or set(value) != {"rank", "bias_limit", "metadata_dropout"}:
        raise PilotError(f"{name}: metadata candidate is not exact")
    rank, limit, dropout = value["rank"], value["bias_limit"], value["metadata_dropout"]
    if (
        type(rank) is not int or rank not in RANKS
        or type(limit) is not int or limit not in BIAS_LIMITS
        or type(dropout) is not float or dropout not in METADATA_DROPOUTS
    ):
        raise PilotError(f"{name}: metadata candidate is not declared")
    return {"rank": rank, "bias_limit": limit, "metadata_dropout": dropout}


def metadata_digests(value: object, name: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(FOLDS):
        raise PilotError(f"{name}: fold_digests does not exactly name the fixed folds")
    if any(not isinstance(value[fold], str) or not DIGEST.fullmatch(value[fold]) for fold in FOLDS):
        raise PilotError(f"{name}: fold_digests contains a malformed digest")
    return {fold: value[fold] for fold in FOLDS}


def protocol_identity(text: str, name: str) -> tuple[int, tuple[int, float, float], dict[str, str]]:
    parsed = rows(text, name)
    protocols = [row for row in parsed if row.get("row") == "run-v5"]
    if len(protocols) != 1:
        raise PilotError(f"{name}: expected one run-v5 row")
    protocol = protocols[0]
    try:
        seed = protocol["seed"]
        candidate = (
            protocol["rank"],
            float(protocol["bias_limit"]),
            float(protocol["metadata_dropout"]),
        )
    except (KeyError, TypeError, ValueError):
        raise PilotError(f"{name}: malformed run-v5 identity") from None
    data_rows = [row for row in parsed if row.get("row") == "data-v2"]
    digests: dict[str, str] = {}
    for row in data_rows:
        fold, digest = row.get("fold"), row.get("data_fnv64")
        if fold not in FOLDS or fold in digests or not isinstance(digest, str):
            raise PilotError(f"{name}: malformed or duplicate data-v2 identity")
        digests[fold] = digest
    if tuple(digests) != FOLDS:
        raise PilotError(f"{name}: data-v2 rows do not exactly name the fixed folds in order")
    return seed, candidate, digests


def load_provenanced_outputs(
    paths: list[Path], root: Path = ROOT
) -> tuple[list[tuple[str, str]], dict]:
    seen_paths: set[Path] = set()
    seen_names: set[str] = set()
    outputs: list[tuple[str, str]] = []
    input_hashes: dict[str, str] = {}
    sources: set[str] = set()
    digest_sets: set[tuple[tuple[str, str], ...]] = set()

    for supplied in paths:
        path = supplied.resolve()
        if supplied.suffix != ".stdout":
            raise PilotError(f"{supplied}: input must have the .stdout suffix")
        if path in seen_paths or supplied.name in seen_names:
            raise PilotError(f"{supplied}: duplicate stdout artifact")
        seen_paths.add(path)
        seen_names.add(supplied.name)
        sidecar = supplied.with_suffix(".json")
        if not sidecar.is_file():
            raise PilotError(f"{supplied}: missing sibling metadata {sidecar.name}")
        raw = supplied.read_bytes()
        digest = sha256(raw)
        text = raw.decode("utf-8")
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise PilotError(f"{sidecar}: invalid JSON: {error}") from None
        if not isinstance(metadata, dict):
            raise PilotError(f"{sidecar}: metadata is not an object")
        if type(metadata.get("schema_version")) is not int or metadata["schema_version"] != 1:
            raise PilotError(f"{sidecar}: schema_version must be integer 1")
        source = metadata.get("source_sha")
        if not isinstance(source, str) or not SOURCE_SHA.fullmatch(source):
            raise PilotError(f"{sidecar}: source_sha is not a lowercase 40-hex commit")
        candidate = metadata_candidate(metadata.get("candidate"), str(sidecar))
        seed = metadata.get("seed")
        if type(seed) is not int or seed not in SEEDS:
            raise PilotError(f"{sidecar}: seed is not declared")
        digests = metadata_digests(metadata.get("fold_digests"), str(sidecar))
        row_seed, row_candidate, row_digests = protocol_identity(text, str(supplied))
        expected_row_candidate = (
            candidate["rank"], float(candidate["bias_limit"]), candidate["metadata_dropout"]
        )
        if row_seed != seed or row_candidate != expected_row_candidate or row_digests != digests:
            raise PilotError(f"{supplied}: metadata and stdout row identity disagree")
        if metadata.get("command") != command_for(candidate, seed, digests):
            raise PilotError(f"{sidecar}: command does not exactly match the fixed protocol")
        if type(metadata.get("exit_code")) is not int or metadata["exit_code"] != 0:
            raise PilotError(f"{sidecar}: cell did not exit successfully")
        if metadata.get("stdout_file") != supplied.name:
            raise PilotError(f"{sidecar}: stdout_file does not bind this artifact")
        if metadata.get("stdout_sha256") != digest:
            raise PilotError(f"{supplied}: stdout SHA256 does not match metadata")
        sources.add(source)
        digest_sets.add(tuple(digests.items()))
        outputs.append((supplied.name, text))
        input_hashes[supplied.name] = digest

    if len(sources) != 1:
        raise PilotError("pilot artifacts contain mixed source commits")
    if len(digest_sets) != 1:
        raise PilotError("pilot artifacts contain mixed fold digests")
    digests = dict(next(iter(digest_sets)))
    if digests != locked_fold_digests(root):
        raise PilotError("pilot fold digests do not match splits.lock.json")
    return outputs, {
        "source_sha": next(iter(sources)),
        "fold_digests": digests,
        "fixed_protocol": fixed_protocol(),
        "input_stdout_sha256": dict(sorted(input_hashes.items())),
    }


def rows(text: str, name: str) -> list[dict]:
    result = []
    for number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise PilotError(f"{name}: line {number}: {error}") from None
        if not isinstance(value, dict):
            raise PilotError(f"{name}: line {number} is not an object")
        result.append(value)
    return result


def finite(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise PilotError(f"{where} is not finite numeric evidence")
    return float(value)


def select(outputs: list[tuple[str, str]]) -> dict:
    cells = {}
    for name, text in outputs:
        parsed = rows(text, name)
        protocols = [row for row in parsed if row.get("row") == "run-v5"]
        if len(protocols) != 1:
            raise PilotError(f"{name}: expected one run-v5 row")
        protocol = protocols[0]
        seed = protocol.get("seed")
        candidate = (
            protocol.get("rank"),
            float(protocol.get("bias_limit")),
            float(protocol.get("metadata_dropout")),
        )
        if seed not in SEEDS or candidate not in CANDIDATES:
            raise PilotError(f"{name}: undeclared seed or candidate {seed!r}/{candidate!r}")
        key = (candidate, seed)
        if key in cells:
            raise PilotError(f"duplicate pilot run {candidate}/{seed}")
        finals = {}
        for row in parsed:
            if row.get("row") != "final-v5" or row.get("split") != "test_ood":
                continue
            index = (row.get("fold"), row.get("arm"))
            if index in finals:
                raise PilotError(f"{name}: duplicate final row {index}")
            if index[0] not in FOLDS or index[1] not in ARMS:
                raise PilotError(f"{name}: foreign pilot cell {index}")
            finals[index] = {
                metric: finite(row.get(metric), f"{name}: {index}/{metric}")
                for metric in ("accuracy", "nll", "ece15")
            }
        expected = {(fold, arm) for fold in FOLDS for arm in ARMS}
        if set(finals) != expected:
            raise PilotError(f"{name}: incomplete pilot folds/arms")
        cells[key] = finals
    expected_runs = {(candidate, seed) for candidate in CANDIDATES for seed in SEEDS}
    if set(cells) != expected_runs:
        missing = sorted(expected_runs - set(cells))
        raise PilotError(f"incomplete pilot grid; missing {missing}")

    candidates = []
    for candidate in CANDIDATES:
        by_fold = {}
        all_nll = []
        all_ece = []
        for fold in FOLDS:
            accuracy = []
            for seed in SEEDS:
                final = cells[(candidate, seed)]
                on, off = final[(fold, ARMS[0])], final[(fold, ARMS[1])]
                accuracy.append(on["accuracy"] - off["accuracy"])
                all_nll.append(on["nll"] - off["nll"])
                all_ece.append(on["ece15"] - off["ece15"])
            by_fold[fold] = statistics.fmean(accuracy)
        mean_ece = statistics.fmean(all_ece)
        candidates.append({
            "rank": candidate[0],
            "bias_limit": candidate[1],
            "metadata_dropout": candidate[2],
            "accuracy_delta_by_fold": by_fold,
            "minimum_accuracy_delta": min(by_fold.values()),
            "mean_nll_delta": statistics.fmean(all_nll),
            "mean_ece15_delta": mean_ece,
            "eligible": mean_ece <= 0.02,
        })
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        return {"schema_version": 1, "claimable": False, "decision": "NO-ELIGIBLE-CANDIDATE", "candidates": candidates}
    selected = min(
        eligible,
        key=lambda item: (
            -item["minimum_accuracy_delta"],
            item["mean_nll_delta"],
            item["rank"],
            item["bias_limit"],
            item["metadata_dropout"],
        ),
    )
    return {
        "schema_version": 1,
        "claimable": False,
        "decision": "SELECTED",
        "selection": selected,
        "candidates": candidates,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("outputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        outputs, provenance = load_provenanced_outputs(args.outputs)
        result = select(outputs)
        result.update(provenance)
    except (OSError, UnicodeDecodeError, PilotError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    document = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(document)
    else:
        print(document, end="")
    return 0 if result["decision"] == "SELECTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
