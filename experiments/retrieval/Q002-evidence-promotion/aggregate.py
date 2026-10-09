"""Aggregate the Q002 seed runs into results/metrics.json and results/run.json.

Inputs are the run records `scripts/run_experiment.py run Q002 --seed <s>` writes
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
  than the record's, harness gaps, a baseline control that did not hallucinate,
  a rule whose counter is missing, null or of the wrong type, an exit status
  that disagrees with the rules (0 with a failed rule, 1 with none), or a
  harness-integrity failure: a disagreement between the generator's labels and
  the harness's naive reading of the corpus (`oracle_label_mismatches > 0`) or a
  generated corpus that is not the pinned one (`corpus_selection_digest_mismatches
  > 0`). The two counters are also coverage rules, because the grammar and the
  mutation checker want a named counter and a nonzero exit, but a harness
  defect says nothing about the hypothesis, so such a seed is INVALID, never
  FALSIFIED. INVALID says nothing about the hypothesis.

The harness is shown able to fail by `results/mutations.json`
(scripts/mutation_check.py Q002), which must be this experiment's evidence from the
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
BENCHMARK = "evidence-promotion"
EXPERIMENT = "Q002"
TAIL = 2000

sys.path.insert(0, str(ROOT / "scripts"))
import experiment_records  # noqa: E402

COMPLETED, FALSIFIED, INVALID = "COMPLETED", "FALSIFIED", "INVALID"
OPERATORS = (">=", "<=", "==")
NUMBER = re.compile(r"-?[0-9]+")
SCOPE = (
    "evidence promotion of search hits by exact-source verification: ptr_search's typed promotion path "
    "(ExactSourceResolver, SearchHit::observe_resolved against the lifecycle authority and the source ownership, verify_claim, "
    "SearchHit::promote_verified_with) must promote a retrieved hit to VerifiedKnown only if its citation is the byte-exact "
    "quote at the cited span of an intact raw event that belongs to the capsule generation the hit names, its claim is what that "
    "one statement states (subject, predicate and value byte for byte, not negated) and the lifecycle authority "
    "(PtrRuntime::generation_validity, with a harness overlay for the one state the runtime cannot answer, Disputed) holds the "
    "hit's generation live; the raw event source, the ownership record and the lifecycle closure are trust roots taken from the "
    "generator's table; seeded synthetic worlds of 36 capsules of one to three generations (healthy, revoked, disputed), 156 "
    "answerable, 60 quarantined and 40 absent questions, and a noisy retriever whose hits include nine classes of poison "
    "(fabricated source id, wrong or empty or two-statement or newline-padded span or doctored quote, tampered digest, lookalike "
    "subject or other predicate or subject differing by case, contradicting value or negation or truncated negation, superseded, "
    "revoked-but-live and disputed generations, and a live label on the exact event of another generation or capsule) and hard "
    "honest hits (multibyte value, value containing not, repeated statement, one-line event, spans at the event edges); the "
    "mechanism must promote none of the poison, refuse none of the hard honest hits, accept none of the adversarial stage skips "
    "at the API boundary, promote at least 950 per mille of the honest hits and get at least as many questions right (correct "
    "answer or correct abstention) as direct retrieval-to-answer, the lifecycle filter alone and the boolean API fed true; "
    "direct retrieval (B0) and the boolean API (B2) are the negative controls and the lifecycle filter alone (B1) a contrast"
)
LIMITATIONS = [
    "the claim is about a synthetic structured world: every fact is one statement of the fixed format `subject: predicate is [not ]value` in a raw event the generator wrote; nothing is claimed about free text, paraphrase, statements that span lines, units, dates or a real retriever, all of which the verifier refuses rather than interprets",
    "the honest hits are exact by construction (their citation and claim are the statement itself), so recall_permille measures that the mechanism does not over-refuse well-formed evidence (the hard honest class, with its zero-tolerance falsifier, is where that is tested), not that it recovers hits a real retriever garbles; a recall of 1000 is the expected value",
    "the typed path's guarantee holds given three trust roots, each taken from the generator's table and none checked by the code under test: the RawEventSource (the digest check shows internal consistency, not that the source is the store of record; the trait is open and the P3 double is such an implementation), the SourceOwnership (which capsule generation admitted which raw event; it must come from the admitting record, not from the retriever's index) and the lifecycle closure (a truthful validity answer); what P9 and K11 show is that, given a truthful ownership, a live label cannot launder an event of another generation or capsule, not that an ownership record can be derived from KnowledgeStore (it cannot yet)",
    "the claim a proof is made for is the caller's: a SearchHit carries no assertion of the retriever's own, so nothing ties the claim verified to a claim the retriever made; the boolean observe/promote_verified path reaches VerifiedKnown with no raw bytes at all and is the documented trusted-caller path (baseline B2), not restricted by this experiment",
    "Disputed is not a state of PtrRuntime::generation_validity and the runtime is never wired to answer it: P8 tests the contract that observe_resolved refuses a Disputed answer of the validity closure (the harness overlay, with the mutation observe-accepts-disputed), not that a PtrRuntime-backed system refuses disputed generations; P6 and P7 use the real runtime (a ledger of CapsuleCommitted, CapsuleSuperseded and Revoked events)",
    "a KnowledgeStore cannot hold an event whose bytes disagree with its digest, so P3 cites corrupt replicas served beside the store by a RawEventSource double, recorded as copies of their originals; P3 shows the resolver's defence in depth against a source that violates the store's invariant, not that the store is tamper-evident, and no honest hit reads a replica",
    "a Known is false by the generator's label (eligible: exact source, exact claim, own live generation); the baselines are counted once per question and the mechanism on every hit it promotes; a P2 hit whose claim is true in the world but whose span is wrong still counts as a false Known (false_known_untrue_* reports the subset whose claim is also false); the generator's labels are checked against a naive reading of the corpus that is independent of the code under test, not of the generator (it shares the rendering, the world data and the lifecycle table with it) and that checks labels, not refusal reasons, which are checked only by wrong_refusal_reason against the generator's table",
    "the lifecycle is read once per hit at observation time (a revocation committed after observe_resolved is not seen by promote_verified_with), one process, one machine profile, no concurrency; the boolean path observe/promote_verified is the documented trusted-caller path and is not restricted",
    "the stage-skip attempts are a fixed list of twelve kinds (K6 split into another event and another span of the same event, K11 an honest hit given another capsule's source) on honest and lifecycle-ineligible hits through the public API; a caller that can construct a ResolvedSource or ClaimProof is excluded by privacy of their fields (compile-fail doctests), which this experiment does not test; wall-clock times (timing_ns) are recorded and never judged",
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
    controls = result.get("controls")
    # B0 runs no code under test, so a baseline that did not hallucinate is the harness's; B2 runs the boolean API of the
    # code under test and is judged by the coverage rule false_known_bool_trust >= 1.
    control = controls.get("baseline_direct") if isinstance(controls, dict) else None
    if not isinstance(control, dict) or control.get("required_rule_tripped") is not True:
        invalid.append("the negative control baseline_direct did not produce a false Known")
    control = controls.get("bool_trust") if isinstance(controls, dict) else None
    if not isinstance(control, dict) or not isinstance(control.get("required_rule_tripped"), bool):
        invalid.append("the result does not report the negative control bool_trust")
    if result.get("harness_gaps") != []:
        invalid.append(f"harness gaps: {result.get('harness_gaps')!r}")
    # Harness-integrity guards: a defect of the harness says nothing about the hypothesis. They are coverage rules too (the
    # mutation checker wants a named counter and exit 1), but the verdict of such a seed is INVALID, never FALSIFIED.
    for counter, reason in (
        ("oracle_label_mismatches", "the generator's labels differ from the naive reading of the corpus"),
        ("corpus_selection_digest_mismatches", "the generated corpus is not the pinned one"),
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
    lines = [f"# Q002 evidence promotion: {verdict['verdict']}", ""]
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
            f"{EXPERIMENT}: refusing to aggregate: config.toml has no [preregistration] table (Q002 is not frozen yet)"
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
