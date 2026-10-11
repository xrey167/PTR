"""Aggregate the Q003 seed runs into results/metrics.json and results/run.json.

Inputs are the run records `scripts/run_experiment.py run Q003-v3 --seed <s>` writes
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
  than the record's, harness gaps, a run without the full strong arm (the
  baseline is BM25 + bge-m3 dense + RRF + the bge-reranker-v2-m3 cross-encoder;
  a development run without it or without the cross-encoder is no evidence), a
  no-live-join control that did not return the planted rows, a rule whose counter
  is missing, null or of the wrong type, an exit status that disagrees with the
  rules (0 with a failed rule, 1 with none), or a harness-integrity failure: a
  generated workload that is not the pinned one (`workload_digest_mismatches >
  0`). That counter is also a coverage rule, because the grammar and the
  mutation checker want a named counter and a nonzero exit, but a harness defect
  says nothing about the hypothesis, so such a seed is INVALID, never FALSIFIED.
  INVALID says nothing about the hypothesis.

The harness is shown able to fail by `results/mutations.json`
(scripts/mutation_check.py Q003-v3), which must be this experiment's evidence from the
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
BENCHMARK = "postgres-hybrid"
EXPERIMENT = "Q003-v3"
TAIL = 2000

sys.path.insert(0, str(ROOT / "scripts"))
import experiment_records  # noqa: E402

COMPLETED, FALSIFIED, INVALID = "COMPLETED", "FALSIFIED", "INVALID"
OPERATORS = (">=", "<=", "==")
NUMBER = re.compile(r"-?[0-9]+")
CONFIG = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
PINNED = CONFIG["pinned_thresholds"]
QUERIES = PINNED["queries"]
STRONG_QUERIES = PINNED["strong_queries"]
MARGIN = PINNED["recall_margin_permille"]
CAPSULES = PINNED["capsules"]

# The claim as the criteria gate it, with the numbers read from config.toml's [pinned_thresholds]; a unit test holds the exact
# text, so it cannot say more (or less) than criteria.toml does.
SCOPE = (
    "generation-joined retrieval in PostgreSQL: ptr_pg's search (a full-text list and a halfvec HNSW list, each joined to the "
    "lifecycle catalog of its snapshot before its limit, fused by weighted reciprocal rank fusion keyed by capsule and "
    "generation), its projector and upsert_document must never return a generation the harness's own live table does not call "
    "live, must leave no superseded or revoked document behind, must refuse every late write of a non-live generation with "
    f"NotLive and store nothing, must answer a search the same way twice, must return the live document for the live-body probes "
    f"through its lexical mode, and must have fused recall@10 at most {MARGIN} per mille below both that of the reference "
    "rag_reference (BM25 with OR semantics + cosine + RRF over the same MiniLM vectors, rebuilt from the live table at each "
    "checkpoint; a sanity floor) and that of the harness's own exact dense search over the live set (the sensitivity control); "
    f"seeded synthetic live-lifecycle corpora of {CAPSULES} capsules of up to three generations (kept at generation 1, superseded "
    "once or twice, revoked at generation 1, 2 or 3) built through the projector and upsert_document with frozen MiniLM vectors, "
    f"{QUERIES} paraphrased queries per seed asked after the supersessions and after the revocations, stale rows of superseded and "
    "revoked generations planted by raw SQL into the derived table, late writes of non-live generations and probe queries that "
    "exercise the lexical mode; on natural-language queries ptr-pg's lexical list is empty (plainto_tsquery joins every term with "
    "AND), so its fused list is effectively its dense list and the recall comparison does not test the quality of a working "
    "lexical half; strong_retrieval (BM25 + bge-m3 dense + RRF + bge-reranker-v2-m3) answers a seeded subset of "
    f"{STRONG_QUERIES} answerable queries and is reported, never gated; nDCG@10, the recall of the lexical and vector lists alone, "
    "of the harness's exact dense search and of BM25 alone, the paired standard error against the reference, latency and index "
    "build time are descriptive"
)
LIMITATIONS = [
    "the claim is about a synthetic corpus of templated English facts (sixteen attributes, a few templates each, names from fixed lists): nothing is claimed about real text, a real query distribution or a corpus of another size; the vectors are frozen MiniLM and bge-m3 files, so the result says nothing about another embedding model; the five seeds and the pilot seeds draw from the same 2000-cue pool, so they are not independent corpora",
    "the main queries are natural-language questions, which plainto_tsquery joins with AND term by term and no fact sentence satisfies, so ptr-pg's lexical mode returns no hit on them and its fused list is its vector list (main_query_lexical_hits counts it); this is a finding about ptr-pg's lexical half, recorded as a limitation and not tested as a falsifier; the lexical mode and its lifecycle join are exercised by the probe queries (a stale or a live body as the query), which are audited for stale hits and are never part of recall; recall against BM25 + cosine + RRF therefore compares an approximate-nearest-neighbour dense list with a hybrid whose lexical half is a weak OR-semantics BM25 on templated text, where fusion lowers the reference's recall below plain dense retrieval: ptr-pg being above the reference is mostly that artefact, not evidence of the quality of a ptr-pg hybrid; recall10_exact_cosine_permille (the harness's exact dense search) separates the HNSW approximation from the fusion effect",
    "the recall rules are point estimates with one-sided margins and no stated power beyond the following: the rule against the reference is a sanity floor that trips only when ptr-pg's recall falls more than the margin below the reference's (912 for a reference at 942, far below the product, because the BM25-OR hybrid is worse than plain dense retrieval here); the rule against the harness's exact dense search is the sensitivity control and sees a smaller regression of the dense list (the pilot showed no deficit at all); neither tests the lexical half on natural-language queries",
    "stale hits are judged against the harness's own live table, which is built from the workload's events by code that shares nothing with the product; the product's is_admissible is compared with it for every capsule and generation, so a disagreement of the table with the projection is a counted mismatch, but the table and the generator are the same author's and the lifecycle rules (live generation, tombstone) are the ones the product states",
    "stale rows are planted by raw SQL into the derived table, which simulates an index that lags the ledger or a delete that failed; the real failure modes of the projector (a crash between the lifecycle change and the document delete) are the subject of L004, not of this experiment, and the product deletes a superseded or revoked document in the projector's own transaction, so without the plants a stale hit could not occur at all; the check that the projector leaves no residual row (projector_residual_rows) is the only place a missing delete shows, because the search join hides such rows",
    "the baselines are rebuilt from the live table, so they hold no stale rows by construction and a baseline hit is never audited for staleness; the comparison is of recall only and the baselines are not under test; their drivers are Python processes on a JSON-lines bridge",
    f"the strong baseline answers a seeded subset of {STRONG_QUERIES} answerable queries stratified by checkpoint and live generation, is never gated, and its cross-encoder runs in fp32 on shared CPU and is not bit-reproducible across machines or thread counts (rank order of well-separated candidates is stable, near ties may flip); its comparison with ptr-pg is reported with the discordant queries and a 95 percent upper bound on their share, not a standard error, which reads 0 when both arms hit every query; a run that resumed answers from an earlier run says so (procedure.strong_arm.resumed_queries)",
    f"the recall margin of {MARGIN} per mille is three paired standard errors of the difference at {QUERIES} queries and recall about 0.9; the pilot (seeds outside the confirmatory five, same pool) measured a paired standard error of at most 10 per mille after the query count was raised from 400 to {QUERIES}; every run reports its own, descriptive and unjudged, because a noisy estimate must not make a seed vacuous; at zero true deficit a seed is falsified with probability about 0.1 percent per rule and seed (about 0.7 percent over the five seeds if they were independent, and they share one pool, so they are correlated), and a true deficit of 20 per mille would falsify 10 to 16 percent of seeds (SE 8 to 10), about 40 to 58 percent of five-seed experiments if independent",
    "the declared seeds 43, 71, 101, 131 and 173 were never run against PostgreSQL before the freeze (the workload generator was run for their digests, nothing more; every development run, mutation run and pilot used seed 17 or a pilot seed from 1001 on), so the confirmatory evidence is blind on all five; seeds 17 and 29, which Q003 and Q003-v2 ran or reserved, were dropped for that reason, and the numbers Q003-v2's completed seed 17 produced steered no rule, threshold, margin or seed here",
    "the HNSW list is approximate (pgvector iterative scan, ef_search 40): the recall of the vector mode against an exact cosine is part of what is measured; latency and index build time (the product's register_space rebuilding the index over the final corpus after every query) are descriptive on one local PostgreSQL 16 / pgvector 0.8 server, one session at a time, and are never judged",
    "a late write is attempted for four kinds of non-live generation (superseded, revoked, ahead of the live one, a capsule never committed) in a quiescent database; races between the projector and a writer are the subject of the ptr-pg integration tests, not of this experiment",
]

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
        return {
            "seed": seed,
            "valid": False,
            "invalid": ["no harness result"],
            "unevaluable": [],
            "coverage_failed": [],
            "falsifiers_failed": [],
        }
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
    strong = procedure.get("strong_arm") if isinstance(procedure, dict) else None
    if not isinstance(strong, dict) or (strong.get("mode"), strong.get("ran"), strong.get("reranked")) != ("full", True, True):
        invalid.append("the strong arm did not run in full (BM25 + dense + RRF + cross-encoder): a development run is no evidence")
    # Q003-v3 thread pin: the fixed command runs the cross-encoder on one torch thread; any other recorded value (or none) is a
    # validity failure, not a finding about the product.
    threads = strong.get("strong_threads") if isinstance(strong, dict) else None
    if isinstance(threads, bool) or not isinstance(threads, int) or threads != 1:
        invalid.append(f"the strong arm recorded strong_threads {threads!r}, not the pinned 1")
    controls = result.get("controls")
    # The no-live-join control runs the harness's own SQL, not code under test: a control that returned no planted row
    # shows nothing about the product's lifecycle join.
    control = controls.get("no_live_join") if isinstance(controls, dict) else None
    if not isinstance(control, dict) or control.get("required_rule_tripped") is not True:
        invalid.append("the negative control no_live_join did not return the planted stale rows")
    if result.get("harness_gaps") != []:
        invalid.append(f"harness gaps: {result.get('harness_gaps')!r}")
    unsupported = result.get("unsupported")
    if unsupported != {}:
        invalid.append(f"metrics declared unsupported: {unsupported!r}")
    # Harness-integrity guard: a defect of the harness says nothing about the hypothesis. It is a coverage rule too (the
    # mutation checker wants a named counter and exit 1), but the verdict of such a seed is INVALID, never FALSIFIED.
    count = result.get("workload_digest_mismatches")
    if isinstance(count, int) and not isinstance(count, bool) and count > 0:
        invalid.append(f"harness integrity: workload_digest_mismatches = {count} (the generated workload is not the pinned one)")
    # A reference that finds little is no product defect, and it makes only the recall rule against the reference unevaluable:
    # the seed is judged on every other rule (a stale hit still falsifies), and is INVALID only if none trips.
    unevaluable: list[str] = []
    floor = result.get("recall10_reference_permille")
    if isinstance(floor, int) and not isinstance(floor, bool) and floor < PINNED["min_recall10_reference_permille"]:
        unevaluable.append(
            f"recall10_reference_permille = {floor} is below the baseline floor "
            f"{PINNED['min_recall10_reference_permille']}: the recall rule against the reference is not evaluable"
        )
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
    if unevaluable:
        # What the floor makes void: the floor rule itself and the deficit against the reference. The rest stands.
        coverage_failed = [rule for rule in coverage_failed if not rule.startswith("recall10_reference_permille ")]
        falsifiers_failed = [
            rule for rule in falsifiers_failed if not rule.startswith("recall10_pg_deficit_vs_reference_permille ")
        ]
    return {
        "seed": seed,
        "valid": not invalid,
        "invalid": invalid,
        "unevaluable": unevaluable,
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
    unevaluable = [entry["seed"] for entry in seeds if entry["valid"] and entry.get("unevaluable")]
    static_failed = [item["command"] for item in static or [] if item.get("exit_code") != 0]
    if unmet or tripped or static_failed:
        verdict = FALSIFIED
    elif problems or unjudged or unevaluable:
        verdict = INVALID
    else:
        verdict = COMPLETED
    return {
        "verdict": verdict,
        "vacuous_seeds": unmet,
        "falsified_seeds": tripped,
        "invalid_seeds": unjudged,
        "unevaluable_seeds": unevaluable,
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
    lines = [f"# Q003 generation-joined retrieval: {verdict['verdict']}", ""]
    lines.append(
        "Scope: ptr-pg's lifecycle guarantees and the recall of its dense list. On natural-language queries its lexical list is "
        "empty, so this verdict does not say that the PostgreSQL hybrid matches a hybrid (see Limitations)."
    )
    lines.append("")
    lines.append(f"Seeds: {', '.join(str(entry['seed']) for entry in sorted(seeds, key=lambda e: e['seed']))}. "
                 "Every seed must hold every coverage and falsifier rule.")
    lines.append("")
    for entry in sorted(seeds, key=lambda e: e["seed"]):
        state = "valid" if entry["valid"] else "INVALID"
        if entry["valid"] and entry.get("unevaluable"):
            state = "recall rule against the reference not evaluable"
        lines.append(f"- seed {entry['seed']}: {state}; {len(entry['coverage_failed'])} coverage rules unmet, "
                     f"{len(entry['falsifiers_failed'])} falsifier rules tripped")
        for key in ("invalid", "unevaluable", "coverage_failed", "falsifiers_failed"):
            lines.extend(f"    - {key}: {item}" for item in entry.get(key, []))
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
            f"{EXPERIMENT}: refusing to aggregate: config.toml has no [preregistration] table (Q003-v3 is not frozen yet)"
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
