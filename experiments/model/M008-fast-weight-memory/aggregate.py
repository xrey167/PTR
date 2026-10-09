"""Aggregate the M008 seed runs into results/metrics.json and results/run.json.

Inputs are the run records `scripts/run_experiment.py run M008 --seed <s>` writes
(results/run-<timestamp>-seed-<s>.json): the newest record of every declared
seed whose command ran to an exit code. Nothing of the experiment is run again.
Every declared seed must have a record, the records must be of one
configuration and have run the checkout's code, this script included
(`experiment_records.source_revision`), and the preregistration digests of the
manifest, the table, the preregistration text and `criteria.toml` must be the
frozen ones.

The decision is mechanical, and it is `criteria.toml`'s (its header defines the
rule grammar, which `evaluate` below implements):

- a seed is judged by applying every coverage rule and every falsifier rule to
  the one JSON object its harness printed;
- COMPLETED iff every seed holds every coverage rule and every falsifier rule
  and the mutation evidence is complete;
- FALSIFIED if some valid seed fails a rule (a coverage rule that fails makes
  the run vacuous, which counts as falsified);
- INVALID otherwise when a seed cannot be judged: no result line, a harness
  that exited 2 or by a signal, another benchmark, seed, scale or experiment
  than the record's, harness gaps, a negative control that was not separated, a pilot seed, a pool or vector digest that is not the pinned one,
  a rule whose counter is missing, null or of the wrong type, an exit status
  that disagrees with the rules (0 with a failed rule, 1 with none), or a
  harness-integrity failure: a disagreement among the oracle's readings of the
  write list (`oracle_label_mismatches > 0`), a stale version in the live baseline index (`stale_in_live_index > 0`) or
  a recency window over the budget (`window_budget_violations > 0`). These counters are also coverage rules,
  because the grammar and the mutation checker want a named counter and a nonzero
  exit, but a harness defect says nothing about the hypothesis, so such a seed is
  INVALID, never FALSIFIED. INVALID says nothing about the hypothesis.

The harness is shown able to fail by `results/mutations.json`
(scripts/mutation_check.py M008), which must be this experiment's evidence from the
same code, checker and mutation plan with every mutation killed; a missing or partly
surviving record makes the verdict INVALID (the harness was not shown to see the defects it
claims to see), never COMPLETED. A falsification stays a falsification whatever the
mutation evidence: a run the harness produced and judged as failing is a finding, and the
mutation state is reported beside it (`mutations_ok`). criteria.toml names no static
evidence, so otherwise the verdict rests on the seeds alone.

metrics.json and run.json are published as one aggregate
(`experiment_records.publish_aggregate`). `--report PATH` also writes the
Markdown reading of the verdict to PATH (not into the results directory: a file
added there after the seeds ran would make the archived results stale).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shlex
import subprocess
import sys
import tomllib
from decimal import Decimal
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE / "results"
BENCHMARK = "fast-weight-memory"
EXPERIMENT = "M008"
TAIL = 2000

sys.path.insert(0, str(ROOT / "scripts"))
import experiment_records  # noqa: E402

COMPLETED, FALSIFIED, INVALID = "COMPLETED", "FALSIFIED", "INVALID"
OPERATORS = (">=", "<=", "==")
NUMBER = re.compile(r"-?[0-9]+")
SCOPE = (
    "revocable fast-weight working memory against a recency buffer and plain retrieval at an equal token budget: "
    "ptr_fastmem's gated delta-rule memory (FastMemory::write keyed by the projected frozen MiniLM embedding of a cue "
    "text, valued by the identifier code of the fact, beta 1 so an update under the same cue replaces the association; "
    "FastMemory::read_admitted and decode_readout with the min_score and min_margin set on the pilot seeds) is compared "
    "on seeded synthetic sessions of 400 cues written, 200 of them updated later with a near-duplicate fact, and 400 "
    "queries (160 about updated cues, 160 about never-updated cues, 40 about cues nobody wrote and 40 off-topic) asked "
    "as paraphrases that never repeat the cue text; the arms are the memory, a recency buffer of the last 39 writes "
    "(512 tokens / the longest fact of 13), and the pinned rag_reference hybrid retrieval (BM25 + cosine, RRF k=60) over "
    "the live facts only (gated) and over an append-only index (reported); every arm presents at most five facts within "
    "the 512-token budget and abstains at a matched operating point set on pilot seeds (at most 5 percent of the "
    "unanswerable pilot queries answered); the memory must beat both gated baselines in update-aware recall@1 (the lower "
    "bound of a paired bootstrap over the answerable queries must be positive, a fact counts only if it is the current "
    "version of the queried cue) without a higher false recall rate than the recency buffer, and its own contracts must "
    "hold (decode equals the harness's independent decode, every write reads back, an exact-key overwrite leaves no "
    "stale association, the key projection binding refuses foreign queries); texts and vectors are the frozen pool "
    "research/embeddings/pool-m008.jsonl and vectors-m008.f16 (sha256 pinned); the oracle, the three negative controls "
    "(oracle-cheat, random, stale-first) and the abstention thresholds are the harness's"
)
LIMITATIONS = [
    "the claim is about a synthetic templated English pool (5 attributes' worth of one-sentence facts and short questions) embedded once by a small general sentence embedder (all-MiniLM-L6-v2, CPU, vectors pinned as half floats); nothing is claimed about real corpora, long contexts, LLM answer quality or other embedders, and sibling cues of one entity are similar by construction, which is the interference the memory is tested against",
    "the memory is keyed by the embedding of a cue text (paraphrase 0 of the cue), which stands for a cue extractor in front of the memory; the recency and hybrid baselines index the fact texts only and have no cue extractor, so the comparison favours the memory on this point; a result in the memory's favour is a statement about the mechanism given such an extractor, not about an end-to-end system",
    "hybrid_live is told which document an update supersedes (the harness deletes the old version), the same kind of cue knowledge the memory gets from its key; the projection and the codebook are single seeded draws used for all five seeds, so variance across draws is not sampled and the bootstrap is over queries only",
    "the claim is scoped to 400 cues written per session, 8 heads x key_dim 32 x value_dim 64 (pilot shapes up to key_dim 384 and 16 heads did not change the picture), untrained seeded random projections of frozen MiniLM vectors; the capacity table (16 to 400 cues) and the unthresholded recall in every run are descriptive and show where the memory holds its cues and where it does not; the matched-operating-point comparison is one point (the memory abstains on most queries there because crosstalk makes unanswerable readouts as strong as correct ones), and the false-recall rule is weak because an arm that abstains almost always passes it",
    "the memory is a derived, revocable projection with random (untrained) orthogonal key projections and sign identifier codes as values; it is bounded by the delta rule's capacity (heads x key_dim), and a result against it is a statement about this configuration and pool, not about learned projections, which are not implemented",
    "the recency buffer is the last 39 writes answered by cosine on the same frozen vectors and cannot answer for facts outside the window; its window is the budget over the longest fact, so it holds the budget in the worst case; the strata (inside and outside the window) are reported and the old stratum is where the buffer cannot win",
    "the hybrid arms are the digest-pinned rag_reference implementation run through a Python driver; its rank score is ordinal, so their abstention is a cosine between the query and the top hit chosen by the harness, which is not part of the pinned baseline; hybrid_append is reported and not gated",
    "the abstention thresholds and the memory shape were set on pilot seeds 1001..1005 by a rule fixed beforehand and are pinned; the confirmatory seeds apply them unchanged, so the realised false answer rate on the unanswerable queries differs from the 5 percent target by sampling (it is reported, and a transfer failure above 150 permille makes the seed vacuous); the pilot may have been optimistic for one arm and pessimistic for another",
    "the truth is the harness's table (the last write of each cue), read three ways that must agree (a forward fold, a backward scan and the generator's label); the oracle shares no code with the arms, but all three share the generator's session; the controls use no code under test and are scored by the same function as the arms",
    "the memory's contract checks (decode agreement, readback, stale residual, projection binding) are read from the readout and the codes the product publishes; the product's decode_readout is compared with a re-implementation of the same rule in the harness, so a defect both share would not be seen; the checks run on the pinned shape only",
    "one process, one machine profile, no concurrency; wall-clock times (timing_ns, latency_ns_per_query) are recorded and never judged; vectors were produced on CPU with torch 2.7.1 and are pinned, so no claim of bit-reproducibility across machines is made",
]

CONFIG = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))


class Unreadable(Exception):
    """A rule that cannot be read or resolved: the seed is INVALID."""


def parse_rule(rule: str) -> tuple[str, str, str]:
    """`(left, op, right)` of a rule, which is exactly three tokens separated
    by single spaces with a known operator. Raises Unreadable."""
    tokens = rule.split(" ") if isinstance(rule, str) else []
    if len(tokens) != 3 or tokens[1] not in OPERATORS or not all(tokens):
        raise Unreadable(f"not '<left> <op> <right>' with an operator of {OPERATORS}: {rule!r}")
    return tokens[0], tokens[1], tokens[2]


def resolve(result: dict, path: str):
    """The value at the dotted path of object keys from the root of `result`.
    Raises Unreadable when a key is missing or a step is not an object."""
    at = result
    for key in path.split("."):
        if not isinstance(at, dict) or key not in at:
            raise Unreadable(f"{path} does not resolve in the harness result")
        at = at[key]
    return at


def operand(token: str, result: dict, decimal: bool, equality: bool):
    """`("int" | "num" | "text", value)` of a literal or reference token."""
    if NUMBER.fullmatch(token):
        return ("num", Decimal(token)) if "." in token else ("int", int(token))
    value = resolve(result, token)
    if isinstance(value, bool) or value is None:
        raise Unreadable(f"{token} is {value!r}, not a number")
    if isinstance(value, int):
        return "int", value
    if isinstance(value, float) and decimal:
        return "num", Decimal(repr(value))
    if isinstance(value, str) and equality:
        return "text", value
    raise Unreadable(f"{token} is {value!r}, which this rule does not compare")


def evaluate(result: dict, rule: str) -> bool:
    """Whether `rule` holds for the harness result. Raises Unreadable."""
    left, op, right = parse_rule(rule)
    decimal = any(NUMBER.fullmatch(token) and "." in token for token in (left, right))
    a = operand(left, result, decimal, op == "==")
    b = operand(right, result, decimal, op == "==")
    if (a[0] == "text") != (b[0] == "text"):
        raise Unreadable(f"{rule!r} compares a string with a number")
    x, y = a[1], b[1]
    if a[0] == "int" and b[0] == "num":
        x = Decimal(x)
    if a[0] == "num" and b[0] == "int":
        y = Decimal(y)
    return x >= y if op == ">=" else x <= y if op == "<=" else x == y


def judge_seed(result: dict | None, exit_code, criteria: dict, seed: int) -> dict:
    """One seed's judgement: the coverage and falsifier rules that fail,
    the reasons it cannot be judged, and its validity. Pure."""
    invalid: list[str] = []
    coverage_failed: list[str] = []
    falsifiers_failed: list[str] = []
    if not isinstance(result, dict):
        return {"seed": seed, "valid": False, "invalid": ["no harness result"], "coverage_failed": [], "falsifiers_failed": []}
    if result.get("benchmark") != BENCHMARK or result.get("experiment") != EXPERIMENT:
        invalid.append(f"reports benchmark {result.get('benchmark')!r}, experiment {result.get('experiment')!r}")
    if result.get("seed") != seed or isinstance(result.get("seed"), bool):
        invalid.append(f"reports seed {result.get('seed')!r}, not {seed}")
    if result.get("scale") != "pinned":
        invalid.append(f"ran at scale {result.get('scale')!r}, not the pinned one")
    procedure = result.get("procedure")
    if not isinstance(procedure, dict) or procedure.get("seed_is_declared") is not True:
        invalid.append("does not report a declared seed")
    elif procedure.get("shape_is_pinned") is not True:
        invalid.append("did not run the pinned shape")
    # The negative controls run no code under test, so a control that was not separated is the harness's.
    controls = result.get("controls")
    for name in ("oracle_cheat", "random", "stale_first", "positive"):
        control = controls.get(name) if isinstance(controls, dict) else None
        if not isinstance(control, dict) or control.get("required_rule_tripped") is not True:
            invalid.append(f"the negative control {name} did not separate from the arms under test")
    pinned = CONFIG["pinned_thresholds"]
    if isinstance(procedure, dict):
        if procedure.get("seed_is_pilot") is not False:
            invalid.append("ran a pilot seed: pilot seeds are for calibration and count for nothing")
        for key, pin in (("pool_sha256", "pool_sha256"), ("vectors_sha256", "vectors_sha256"),
                         ("embedding_model", "embedding_model"), ("embedding_revision", "embedding_revision")):
            if procedure.get(key) != pinned[pin]:
                invalid.append(f"procedure.{key} is {procedure.get(key)!r}, not the pinned {pinned[pin]!r}")
    if result.get("harness_gaps") != []:
        invalid.append(f"harness gaps: {result.get('harness_gaps')!r}")
    # Harness-integrity guard: a defect of the harness says nothing about the hypothesis. It is a coverage rule too (the
    # mutation checker wants a named counter and exit 1), but the verdict of such a seed is INVALID, never FALSIFIED.
    for counter, reason in (
        ("oracle_label_mismatches", "the oracle's readings of the write list and the generator's labels disagree"),
        ("stale_in_live_index", "the pinned live hybrid index returned a stale version: a defect of the driver or the harness"),
        ("window_budget_violations", "the recency window times the longest fact exceeds the token budget: a defect of the harness"),
    ):
        count = result.get(counter)
        if isinstance(count, int) and not isinstance(count, bool) and count > 0:
            invalid.append(f"harness integrity: {counter} = {count} ({reason})")
    for key, failed in (("coverage", coverage_failed), ("falsifiers", falsifiers_failed)):
        for rule in criteria[key]:
            try:
                if not evaluate(result, rule):
                    failed.append(rule)
            except Unreadable as error:
                invalid.append(f"{key}: {error}")
    if exit_code not in (0, 1) or isinstance(exit_code, bool):
        invalid.append(f"the harness exited with {exit_code!r}")
    elif not invalid:
        failing = bool(coverage_failed or falsifiers_failed)
        if failing != (exit_code == 1):
            invalid.append(f"the harness exited {exit_code} but the rules {'fail' if failing else 'all hold'}")
    return {
        "seed": seed,
        "valid": not invalid,
        "invalid": invalid,
        "coverage_failed": coverage_failed,
        "falsifiers_failed": falsifiers_failed,
    }


def decide(seeds: list[dict], static: list[dict] | None, declared: list[int], mutations: dict | None = None,
           planned: int | None = None) -> dict:
    """The verdict from the seeds' judgements (`judge_seed`, in any order), the
    static-evidence results (None when they were not run) and the mutation evidence
    (`killed` and `total`; None when there is none; `planned` is the number of mutations the plan
    lists, when known). COMPLETED needs complete mutation evidence (`mutations_ok`); a
    falsification is never downgraded by it. Pure."""
    present = sorted(entry["seed"] for entry in seeds)
    missing = sorted(set(declared) - set(present))
    duplicate = sorted({seed for seed in present if present.count(seed) > 1})
    extra = sorted(set(present) - set(declared))
    problems = []
    if missing:
        problems.append(f"no judgement of seeds {missing}")
    if duplicate or extra:
        problems.append(f"judgements of seeds {duplicate + extra} that are duplicated or not declared")
    if static is None:
        problems.append("the static evidence was not run")
    mutations_ok = (
        mutations is not None
        and isinstance(mutations.get("total"), int)
        and mutations.get("killed") == mutations.get("total")
        and mutations["total"] > 0
        and (planned is None or mutations["total"] == planned)
    )
    if mutations is None:
        problems.append("mutation evidence missing or incomplete (no results/mutations.json)")
    elif not mutations_ok:
        problems.append(
            f"mutation evidence missing or incomplete: killed {mutations.get('killed')} of {mutations.get('total')} "
            f"mutations" + (f", the plan lists {planned}" if planned is not None else "")
        )
    unmet = [entry["seed"] for entry in seeds if entry["valid"] and entry["coverage_failed"]]
    tripped = [entry["seed"] for entry in seeds if entry["valid"] and entry["falsifiers_failed"]]
    unjudged = [entry["seed"] for entry in seeds if not entry["valid"]]
    static_failed = [item["command"] for item in static or [] if item.get("exit_code") != 0]
    if unmet or tripped or static_failed:
        verdict = FALSIFIED
    elif problems or unjudged:
        verdict = INVALID
    else:
        verdict = COMPLETED
    return {
        "verdict": verdict,
        "vacuous_seeds": unmet,
        "falsified_seeds": tripped,
        "invalid_seeds": unjudged,
        "static_evidence_failed": static_failed,
        "mutations_ok": mutations_ok,
        "problems": problems,
        "recommended_status": {COMPLETED: "completed", FALSIFIED: "failed"}.get(verdict),
    }


def run_command(command: str) -> tuple[int, str]:
    """Run one static-evidence command in the repository root. A command that
    cannot start is exit code -1."""
    try:
        process = subprocess.run(shlex.split(command), cwd=ROOT, text=True, capture_output=True)
    except OSError as error:
        return -1, str(error)
    return process.returncode, process.stdout + process.stderr


def static_evidence(commands: list[str], runner: Callable[[str], tuple[int, str]] = run_command) -> list[dict]:
    """Run every command and record its exit code and output digest."""
    results = []
    for command in commands:
        code, output = runner(command)
        results.append(
            {
                "command": command,
                "exit_code": code,
                "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
                "output_tail": output[-TAIL:],
            }
        )
    return results


def report(verdict: dict, seeds: list[dict], static: list[dict] | None, mutations: dict | None = None) -> str:
    """The Markdown reading of the verdict."""
    lines = [f"# M008 fast-weight memory: {verdict['verdict']}", ""]
    lines.append(f"Seeds: {', '.join(str(entry['seed']) for entry in sorted(seeds, key=lambda e: e['seed']))}. "
                 "Every seed must hold every coverage and falsifier rule.")
    lines.append("")
    for entry in sorted(seeds, key=lambda e: e["seed"]):
        state = "valid" if entry["valid"] else "INVALID"
        lines.append(f"- seed {entry['seed']}: {state}; {len(entry['coverage_failed'])} coverage rules unmet, "
                     f"{len(entry['falsifiers_failed'])} falsifier rules tripped")
        for key in ("invalid", "coverage_failed", "falsifiers_failed"):
            lines.extend(f"    - {key}: {item}" for item in entry[key])
    lines.append("")
    for item in static or []:
        lines.append(f"- static evidence `{item['command']}`: exit {item['exit_code']}")
    if mutations is not None:
        lines.append(f"- mutation check: {mutations.get('killed')} of {mutations.get('total')} killed")
    lines.append(f"- mutations_ok: {verdict['mutations_ok']}")
    for problem in verdict["problems"]:
        lines.append(f"- problem: {problem}")
    lines.append("")
    lines.append("Limitations: " + "; ".join(LIMITATIONS) + ".")
    return "\n".join(lines) + "\n"


def newest_per_seed(paths: list[Path]) -> dict[int, Path]:
    """The record to aggregate for each seed: the newest by name of those whose run
    finished, the newest of the others only where none finished."""
    chosen: dict[int, tuple[bool, Path]] = {}
    for path in sorted(paths):
        record = json.loads(path.read_text(encoding="utf-8"))
        seed = int(record["seed"])
        launched = experiment_records.finished_run(record)
        if launched or not chosen.get(seed, (False, path))[0]:
            chosen[seed] = (launched, path)
    return {seed: path for seed, (_, path) in chosen.items()}


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def load_frozen() -> tuple[dict, dict, dict, str]:
    """The manifest, the preregistered table, the criteria and the table's digest, once the
    manifest, the table, the preregistration text and the criteria agree. Raises SystemExit."""
    manifest = tomllib.loads((HERE / "experiment.toml").read_text(encoding="utf-8"))
    if "preregistration" not in CONFIG:
        raise SystemExit(
            f"{EXPERIMENT}: refusing to aggregate: config.toml has no [preregistration] table (M008 is not frozen yet)"
        )
    table = CONFIG["preregistration"]
    try:
        digest = experiment_records.preregistration_digest(table)
    except ValueError as error:
        raise SystemExit(f"{EXPERIMENT}: the preregistration cannot be read: {error}")
    if manifest.get("preregistration_sha256") != digest:
        raise SystemExit(
            f"{EXPERIMENT}: refusing to aggregate: experiment.toml names preregistration_sha256 "
            f"{manifest.get('preregistration_sha256')!r}, not {digest}, the digest of config.toml's table"
        )
    for key in ("criteria", "preregistration_text"):
        path = ROOT / table[key]
        actual = experiment_records.preregistered_file_digest(path)
        if table.get(f"{key}_sha256") != actual:
            raise SystemExit(f"{EXPERIMENT}: refusing to aggregate: {table[key]} is not the file frozen ({actual})")
    criteria = tomllib.loads((ROOT / table["criteria"]).read_text(encoding="utf-8"))
    return manifest, table, criteria, digest


def aggregate(report_path: Path | None = None) -> None:
    """Validate the declared seed records, judge them and publish metrics.json and run.json.
    Missing records or failed preregistration or provenance checks raise SystemExit; a
    verdict of any kind is published."""
    manifest, table, criteria, digest = load_frozen()
    per_seed = newest_per_seed(sorted(RESULTS.glob("run-*-seed-*.json")))
    missing = sorted(set(manifest["seeds"]) - set(per_seed))
    if missing:
        raise SystemExit(f"no run record for seeds {missing}")
    chosen = {seed: json.loads(per_seed[seed].read_text(encoding="utf-8")) for seed in manifest["seeds"]}
    named = {per_seed[seed].name: chosen[seed] for seed in manifest["seeds"]}
    results_path = experiment_records.relative_to_root(RESULTS, ROOT)
    experiment_path = experiment_records.relative_to_root(HERE, ROOT)
    try:
        revision = experiment_records.source_revision(
            EXPERIMENT,
            manifest,
            named,
            ROOT,
            HERE,
            paths=experiment_records.listed_record_paths(results_path),
            checkout_paths=experiment_records.listed_staleness_paths(results_path, experiment_path),
            listed_experiment=True,
        )
        seed_records = experiment_records.seed_record_digests(RESULTS)
        mutations = experiment_records.mutation_evidence(EXPERIMENT, RESULTS / "mutations.json", BENCHMARK, ROOT, HERE)
    except experiment_records.ProvenanceError as error:
        raise SystemExit(f"{EXPERIMENT}: refusing to aggregate: {error}")

    judgements, records, metrics_seeds = [], [], []
    metric_names = manifest["metrics"].split(", ")
    for seed in manifest["seeds"]:
        record = chosen[seed]
        result = None
        try:
            result = experiment_records.harness_results({per_seed[seed].name: record}, BENCHMARK)[per_seed[seed].name]
            error = None
        except experiment_records.ProvenanceError as problem:
            error = str(problem)
        judgement = judge_seed(result, record.get("exit_code"), criteria, seed)
        if error:
            judgement["valid"] = False
            judgement["invalid"].append(error)
        judgements.append(judgement)
        records.append(
            {"seed": seed, "record": per_seed[seed].name, "git_sha": record["git_sha"], "exit_code": record.get("exit_code")}
        )
        metrics_seeds.append(
            {
                **judgement,
                "metrics": {name: result.get(name) for name in metric_names} if result else None,
                "summary": result.get("summary") if result else None,
            }
        )
    static = static_evidence(criteria.get("static_evidence", []))
    planned = len(tomllib.loads((HERE / "tests" / "mutations.toml").read_text(encoding="utf-8")).get("mutation", []))
    verdict = decide(judgements, static, manifest["seeds"], mutations, planned)
    text = report(verdict, judgements, static, mutations)
    metrics = {
        "experiment_id": EXPERIMENT,
        "scope": SCOPE,
        "verdict": verdict["verdict"],
        "decision": verdict,
        "seeds": metrics_seeds,
        "static_evidence": static,
        "mutation_checks": mutations,
        "report": text,
    }
    run = {
        "experiment_id": EXPERIMENT,
        "status": manifest["status"],
        "scope": SCOPE,
        "git_sha": revision,
        "aggregated_at_git_sha": git_sha(),
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "preregistration_sha256": digest,
        "preregistration_rules_sha256": manifest.get("preregistration_rules_sha256"),
        "criteria_sha256": table["criteria_sha256"],
        "entrypoint": manifest["entrypoint"],
        "seeds": records,
        "seed_records": seed_records,
        "verdict": verdict["verdict"],
        "recommended_status": verdict["recommended_status"],
        "static_evidence": [{key: item[key] for key in ("command", "exit_code", "output_sha256")} for item in static],
        "mutation_checks": mutations,
        "limitations": LIMITATIONS,
    }
    try:
        experiment_records.publish_aggregate(RESULTS, metrics, run)
    except experiment_records.ProvenanceError as error:
        raise SystemExit(f"{EXPERIMENT}: {error}")
    if report_path is not None:
        report_path.write_text(text, encoding="utf-8")
    print(f"{EXPERIMENT} verdict={verdict['verdict']} recommended_status={verdict['recommended_status']} "
          f"vacuous={verdict['vacuous_seeds']} falsified={verdict['falsified_seeds']} invalid={verdict['invalid_seeds']} "
          f"static_failed={len(verdict['static_evidence_failed'])}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, help="also write the Markdown reading of the verdict here")
    args = parser.parse_args()
    aggregate(report_path=args.report)


if __name__ == "__main__":
    main()
