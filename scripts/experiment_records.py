"""Bind an experiment's aggregated result to the code its run records exercised.

`scripts/run_experiment.py run` stamps every run record with the experiment
id, the commit it ran at (`git_sha`), the manifest, the Cargo.lock hash, the
parameters, the entrypoint and the command. An aggregator that combines one
record per seed must not publish a verdict for code those records never ran,
so before it combines them it asks `source_revision`, which refuses records
that

- belong to another experiment, or do not name the commit they ran at;
- disagree about the Cargo.lock hash, the parameters or the entrypoint, ran an
  entrypoint the manifest does not declare, or ran under a manifest that
  differs from the current `experiment.toml` in anything but its `status`
  (which changes when the experiment completes);
- ran at commits whose provenance files (`seed_record_paths`: the Rust
  sources, Cargo manifests, lock file, toolchain, Cargo configuration and SQL
  files of `CODE_PATHS`, the scripts that record and judge seed runs, and the
  experiment's own `aggregate.py`) differ from each other, or from the
  checkout the aggregate is written in.

Records archived one commit at a time sit at different commits; they still
agree when no provenance file changed between those commits.

Each record's harness result must then be the run the wrapper selected it for
(`harness_results`): the same benchmark, seed and iteration count. Mutation
evidence (`results/mutations.json`, written by `scripts/mutation_check.py`) is
bound the same way (`mutation_evidence`): it is refused, not omitted, unless
it is this experiment's evidence for this benchmark and its commit has the
checkout's code, mutation checker, aggregator and mutation plan
(`mutation_record_paths`). A refused file is rerun or removed; an aggregate
never carries mutation counts of other code.

`run_experiment.py` and `mutation_check.py` refuse to record from a working
tree with uncommitted or untracked provenance files (`uncommitted_files`), so
a record's `git_sha` is the code it ran. Records made by an older recorder
cannot slip past this: the recorder is itself a provenance file, so a record
aggregates only while the recorder is the one that ran it.

Archived results of a completed experiment must describe HEAD's code, or carry
a `results/STALE.toml` marker saying since when they do not
(`staleness_errors`, run by `scripts/check_research_gates.py`). The marker
names a commit at which every stale file has the provenance files it ran at,
which `run.json` and `mutations.json` share even when they ran at different
commits, and the first commit descending from it that changed one; commits
merged in from another line of history, such as a pull request's base
branch, never count as that change, so one marker holds on the pull
request's head, on the merge CI checks and on the base branch after it.
"""

from __future__ import annotations

import json
import re
import subprocess
import tomllib
from pathlib import Path

# The files that decide what an experiment binary does, as git pathspecs. A
# pathspec without magic matches `*` across `/`, so `*.rs` is every Rust file.
CODE_PATHS = (
    "*.rs",
    "*.sql",
    "Cargo.toml",
    "*/Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    ".cargo/config.toml",
)
# The scripts that decide what a record says and how it is judged.
JUDGE = "scripts/experiment_records.py"
RECORDER = "scripts/run_experiment.py"
MUTATOR = "scripts/mutation_check.py"
STALE_MARKER = "STALE.toml"
COMMIT = re.compile(r"[0-9a-f]{7,64}")
AGREED = (
    ("cargo_lock_sha256", "Cargo.lock"),
    ("parameters", "parameters"),
    ("entrypoint", "entrypoint"),
)


class ProvenanceError(Exception):
    """The run records cannot be published as one result of the current code."""


def relative_to_root(path: Path, root: Path) -> str:
    """`path` as a git pathspec relative to the repository `root`."""
    return path.resolve().relative_to(root.resolve()).as_posix()


def seed_record_paths(experiment_dir: Path, root: Path) -> tuple[str, ...]:
    """The provenance files of a seed record: `CODE_PATHS`, the recorder and
    this module, and the experiment's `aggregate.py`, which turns the records
    into a verdict."""
    experiment = relative_to_root(experiment_dir, root)
    return (*CODE_PATHS, JUDGE, RECORDER, f"{experiment}/aggregate.py")


def mutation_record_paths(experiment_dir: Path, root: Path) -> tuple[str, ...]:
    """The provenance files of mutation evidence: `CODE_PATHS`, the mutation
    checker and this module, the experiment's `aggregate.py` and the mutation
    plan (`tests/mutations.toml`) the evidence was planted from."""
    experiment = relative_to_root(experiment_dir, root)
    return (
        *CODE_PATHS,
        JUDGE,
        MUTATOR,
        f"{experiment}/aggregate.py",
        f"{experiment}/tests/mutations.toml",
    )


def tree_pathspecs(experiment_dir: Path, results_dir: Path, root: Path, paths: tuple[str, ...]) -> list[str]:
    """The pathspecs a run must find committed: the provenance `paths` and the
    whole experiment directory except its results, where records accumulate."""
    return [
        *paths,
        relative_to_root(experiment_dir, root),
        f":(exclude){relative_to_root(results_dir, root)}",
    ]


def without_status(manifest: dict) -> dict:
    """A manifest apart from its `status`, which completing an experiment changes."""
    return {key: value for key, value in manifest.items() if key != "status"}


def agreement_errors(experiment_id: str, manifest: dict, records: dict[str, dict]) -> list[str]:
    """Why `records` (file name -> run record) cannot be seeds of one run of
    `experiment_id`, whose manifest is now `manifest`; empty when they can.
    This compares the records alone; `source_revision` also compares code."""
    errors = []
    for name, record in records.items():
        if record.get("experiment_id") != experiment_id:
            errors.append(f"{name} is a record of {record.get('experiment_id')!r}, not {experiment_id!r}")
        if not COMMIT.fullmatch(str(record.get("git_sha", ""))):
            errors.append(f"{name} names no commit it ran at: {record.get('git_sha')!r}")
    for key, label in AGREED:
        groups: dict[str, list[str]] = {}
        for name, record in records.items():
            groups.setdefault(json.dumps(record.get(key), sort_keys=True), []).append(name)
        if len(groups) > 1:
            errors.append(
                f"the records disagree on {label}: "
                + "; ".join(f"{value} in {', '.join(names)}" for value, names in groups.items())
            )
    current = without_status(manifest)
    for name, record in records.items():
        if without_status(record.get("manifest") or {}) != current:
            errors.append(f"{name} ran under an experiment.toml that differs from the current one")
        entrypoint = record.get("entrypoint")
        if not isinstance(entrypoint, str) or not str(manifest.get(entrypoint, "")).strip():
            errors.append(f"{name} ran {entrypoint!r}, which experiment.toml declares no command for")
    return errors


def git(root: Path, *args: str) -> subprocess.CompletedProcess:
    """Run git in `root`; raises `ProvenanceError` when git cannot start."""
    try:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    except OSError as error:
        raise ProvenanceError(f"cannot run git: {error}") from error


def code_changes(base: str, head: str | None, root: Path, paths: tuple[str, ...] = CODE_PATHS) -> list[str]:
    """The files under the pathspecs `paths` that differ between commit `base`
    and commit `head`, or the working tree of `root` when `head` is None.
    Raises `ProvenanceError` when git cannot compare them (say a commit is
    missing)."""
    revisions = [base] if head is None else [base, head]
    diff = git(root, "diff", "--name-only", "--no-renames", *revisions, "--", *paths)
    if diff.returncode != 0:
        against = "the checkout" if head is None else head
        raise ProvenanceError(f"cannot compare {base} with {against}: {diff.stderr.strip()}")
    return diff.stdout.splitlines()


def uncommitted_files(root: Path, pathspecs: list[str]) -> list[str]:
    """The files under `pathspecs` whose working-tree state HEAD does not
    hold: modified, staged, deleted or untracked (files git ignores excepted).
    A record made from such a tree would name a commit that is not the code
    it ran. Raises `ProvenanceError` when git cannot list the tree."""
    status = git(
        root, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--no-renames", "--", *pathspecs
    )
    if status.returncode != 0:
        raise ProvenanceError(f"cannot list the working tree: {status.stderr.strip()}")
    return [entry[3:] for entry in status.stdout.split("\0") if entry]


def resolve_commit(value: str, root: Path) -> str | None:
    """The full name of the commit `value` names in `root`, or None."""
    if not isinstance(value, str) or not COMMIT.fullmatch(value):
        return None
    parsed = git(root, "rev-parse", "--verify", "--quiet", f"{value}^{{commit}}")
    return parsed.stdout.strip() if parsed.returncode == 0 else None


def is_ancestor(ancestor: str, descendant: str, root: Path) -> bool:
    """Whether commit `ancestor` is `descendant` or on its history."""
    return git(root, "merge-base", "--is-ancestor", ancestor, descendant).returncode == 0


def changes_after(base: str, head: str, root: Path, paths: tuple[str, ...]) -> list[str]:
    """The commits that descend from `base`, are `head` or on its history, and
    changed a file under `paths`, parents before children. Only descendants
    of `base` count (`--ancestry-path`): a commit merged in from another line
    of history, say the base branch of a pull request, whose merge CI checks,
    changed files the results at `base` never ran, but it is not a change
    after them; the merge that brings it to their line is. Raises
    `ProvenanceError` when git cannot list them."""
    listed_commits = git(
        root, "rev-list", "--reverse", "--topo-order", "--ancestry-path", f"{base}..{head}", "--", *paths
    )
    if listed_commits.returncode != 0:
        raise ProvenanceError(f"cannot list the commits after {base}: {listed_commits.stderr.strip()}")
    return listed_commits.stdout.split()


def listed(paths: list[str], limit: int = 5) -> str:
    more = f" and {len(paths) - limit} more" if len(paths) > limit else ""
    return ", ".join(paths[:limit]) + more


def source_revision(
    experiment_id: str, manifest: dict, records: dict[str, dict], root: Path, experiment_dir: Path
) -> str:
    """The commit the results in `records` were produced at: the earliest
    record's `git_sha`, once every record agrees (`agreement_errors`) and
    every record's commit and the checkout in `root` have the same
    provenance files (`seed_record_paths` of `experiment_dir`). Raises
    `ProvenanceError` naming what disagrees."""
    if not records:
        raise ProvenanceError("no run records")
    errors = agreement_errors(experiment_id, manifest, records)
    if errors:
        raise ProvenanceError("; ".join(errors))
    paths = seed_record_paths(experiment_dir, root)
    ordered = sorted(records.items(), key=lambda item: (str(item[1].get("started_at", "")), item[0]))
    revision = ordered[0][1]["git_sha"]
    for name, record in ordered[1:]:
        if record["git_sha"] != revision:
            changed = code_changes(revision, record["git_sha"], root, paths)
            if changed:
                errors.append(f"{name} ran at {record['git_sha']}, whose code differs in {listed(changed)}")
    if errors:
        raise ProvenanceError(f"the records ran different code from {revision}: " + "; ".join(errors))
    changed = code_changes(revision, None, root, paths)
    if changed:
        raise ProvenanceError(
            f"the records ran at {revision}, and the checkout's code has changed since, "
            f"in {listed(changed)}; rerun the seeds or aggregate at that commit"
        )
    return revision


def is_count(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def last_result_line(stdout: str) -> dict:
    """The last line of `stdout` that starts a JSON object, parsed. Raises
    `ProvenanceError` when there is none or it is not a JSON object."""
    for line in reversed(stdout.splitlines()):
        if line.startswith("{"):
            try:
                result = json.loads(line)
            except json.JSONDecodeError as error:
                raise ProvenanceError(f"its result line is not a JSON object: {error}") from error
            if not isinstance(result, dict):
                raise ProvenanceError("its result line is not a JSON object")
            return result
    raise ProvenanceError("it has no result line")


def harness_results(
    records: dict[str, dict], benchmark: str, counts: tuple[str, ...] = ()
) -> dict[str, dict]:
    """The harness result each of `records` (file name -> run record) carries,
    once every result is the run its wrapper was selected for: the wrapper's
    command ran `benchmark`, and the result reports that benchmark, the
    wrapper's seed and, when the wrapper set one, its iteration count. Every
    counter in `counts` (the hard counters an aggregator judges) must be
    reported as a nonnegative count, so a counter the harness stopped
    reporting cannot read as zero failures. Raises `ProvenanceError` naming
    every record that fails."""
    results: dict[str, dict] = {}
    errors = []
    for name, record in records.items():
        try:
            result = last_result_line(str(record.get("stdout") or ""))
        except ProvenanceError as error:
            errors.append(f"{name}: {error}")
            continue
        if benchmark not in (record.get("command") or []):
            errors.append(f"{name} did not run {benchmark!r}: {record.get('command')!r}")
        if result.get("benchmark") != benchmark:
            errors.append(f"{name} reports benchmark {result.get('benchmark')!r}, not {benchmark!r}")
        seed = record.get("seed")
        if not is_count(result.get("seed")) or result.get("seed") != seed:
            errors.append(f"{name} reports seed {result.get('seed')!r}, not {seed!r}")
        iterations = (record.get("parameters") or {}).get("iterations")
        if iterations is not None and str(result.get("iterations")) != str(iterations):
            errors.append(f"{name} reports {result.get('iterations')!r} iterations, not {iterations}")
        uncounted = [key for key in counts if not (is_count(result.get(key)) and result[key] >= 0)]
        if uncounted:
            errors.append(f"{name} reports no count of {listed(uncounted)}")
        results[name] = result
    if errors:
        raise ProvenanceError("; ".join(errors))
    return results


def mutation_evidence(
    experiment_id: str, path: Path, subcommand: str, root: Path, experiment_dir: Path
) -> dict | None:
    """The mutation check summary (`killed`, `total`, `git_sha`) of `path`,
    or None when there is no such file. The evidence must be of
    `experiment_id`, have mutated `subcommand`, count its own outcomes, and
    have run at a commit whose provenance files (`mutation_record_paths`) are
    the checkout's; `source_revision` has already shown the checkout's code
    is the aggregated records'. Raises `ProvenanceError` otherwise."""
    if not path.exists():
        return None
    name = path.name
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProvenanceError(f"{name} cannot be read: {error}") from error
    if not isinstance(evidence, dict):
        raise ProvenanceError(f"{name} is not a JSON object")
    errors = []
    if evidence.get("experiment_id") != experiment_id:
        errors.append(f"{name} is evidence of {evidence.get('experiment_id')!r}, not {experiment_id!r}")
    if evidence.get("subcommand") != subcommand:
        errors.append(f"{name} mutated {evidence.get('subcommand')!r}, not {subcommand!r}")
    killed, total, outcomes = evidence.get("killed"), evidence.get("total"), evidence.get("mutations")
    if not (is_count(killed) and is_count(total) and isinstance(outcomes, list)):
        errors.append(f"{name} has no killed and total counts and outcomes")
    elif not (
        0 < total == len(outcomes)
        and killed == sum(isinstance(item, dict) and item.get("result") == "killed" for item in outcomes)
    ):
        errors.append(f"{name} counts {killed} of {total} killed, but lists {len(outcomes)} outcomes")
    sha = evidence.get("git_sha")
    if not COMMIT.fullmatch(str(sha or "")):
        errors.append(f"{name} names no commit it ran at: {sha!r}")
    if errors:
        raise ProvenanceError("; ".join(errors))
    changed = code_changes(sha, None, root, mutation_record_paths(experiment_dir, root))
    if changed:
        raise ProvenanceError(
            f"{name} ran at {sha}, and the checkout's code has changed since, in {listed(changed)}; "
            f"rerun scripts/mutation_check.py {experiment_id} or remove {name}"
        )
    return {"killed": killed, "total": total, "git_sha": sha}


def clear_stale_marker(results_dir: Path) -> None:
    """Remove `results_dir`'s stale marker: results just aggregated from the
    checkout's code describe it."""
    (results_dir / STALE_MARKER).unlink(missing_ok=True)


# The archived evidence of a completed experiment and its provenance files.
ARCHIVED = (("run.json", seed_record_paths), ("mutations.json", mutation_record_paths))


def staleness_errors(experiment_id: str, experiment_dir: Path, results_dir: Path, root: Path) -> list[str]:
    """Why the archived results of `experiment_id` misdescribe HEAD's code;
    empty when they do not. `results/run.json` and `results/mutations.json`
    are stale when HEAD's provenance files (`seed_record_paths`,
    `mutation_record_paths`) differ from those at their `git_sha`. Stale
    results pass only with a `results/STALE.toml` marker that names them, the
    first commit after them that changed a provenance file (`stale_since`)
    and a `reason` (`marker_errors`); a marker next to current results is an
    error too, so none outlives the rerun that replaces them."""
    results = relative_to_root(results_dir, root)
    errors = []
    stale: dict[str, tuple[str, tuple[str, ...], list[str]]] = {}
    for name, paths_of in ARCHIVED:
        path = results_dir / name
        if not path.exists():
            continue
        try:
            sha = json.loads(path.read_text(encoding="utf-8")).get("git_sha")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError) as error:
            errors.append(f"{experiment_id}: {results}/{name} cannot be read: {error}")
            continue
        if not COMMIT.fullmatch(str(sha or "")):
            errors.append(f"{experiment_id}: {results}/{name} names no commit it ran at: {sha!r}")
            continue
        paths = paths_of(experiment_dir, root)
        try:
            changed = code_changes(sha, "HEAD", root, paths)
        except ProvenanceError as error:
            errors.append(f"{experiment_id}: {results}/{name}: {error}")
            continue
        if changed:
            stale[name] = (sha, paths, changed)
    marker = results_dir / STALE_MARKER
    if not marker.exists():
        for name, (sha, _, changed) in stale.items():
            errors.append(
                f"{experiment_id}: {results}/{name} ran at {sha}, and the code has changed since, in "
                f"{listed(changed)}; rerun the experiment or record since when it is stale in "
                f"{results}/{STALE_MARKER}"
            )
        return errors
    if not stale:
        if not errors:
            errors.append(
                f"{experiment_id}: {results}/{STALE_MARKER} marks results stale that match HEAD's code; "
                "remove it"
            )
        return errors
    return errors + marker_errors(experiment_id, f"{results}/{STALE_MARKER}", marker, stale, root)


def marker_errors(
    experiment_id: str,
    shown: str,
    marker: Path,
    stale: dict[str, tuple[str, tuple[str, ...], list[str]]],
    root: Path,
) -> list[str]:
    """Why the stale marker `marker` (shown as `shown`) does not honestly
    describe the `stale` results; empty when it does.

    `results_git_sha` names the results: a commit on HEAD's history at which
    every stale file's provenance files are those at its own `git_sha`. When
    `run.json` and `mutations.json` ran at one commit that is it; when the
    seeds and the mutation check ran at different commits with the same
    code, either commit (or any other with that code) is. `stale_since` is a
    first change after it: a commit that descends from it, is HEAD or on
    HEAD's history, and changed a provenance file of a stale result, with no
    such change between the two. Commits that do not descend from the
    results (a base branch's, merged in) never count; the merge that brings
    their change to the results' line does."""
    try:
        fields = tomllib.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        return [f"{experiment_id}: {shown} cannot be read: {error}"]
    errors = []
    for key in ("results_git_sha", "stale_since", "reason"):
        if key not in fields:
            errors.append(f"{experiment_id}: {shown} has no {key}")
    if errors:
        return errors
    if not isinstance(fields["reason"], str) or not fields["reason"].strip():
        errors.append(f"{experiment_id}: {shown} gives no reason")
    marked = resolve_commit(fields["results_git_sha"], root)
    since = resolve_commit(fields["stale_since"], root)
    for key, resolved in (("results_git_sha", marked), ("stale_since", since)):
        if resolved is None:
            errors.append(f"{experiment_id}: {shown}: {key} {fields[key]!r} is not a commit")
    if errors:
        return errors
    others = []
    for name, (sha, name_paths, _) in stale.items():
        differing = code_changes(sha, marked, root, name_paths)
        if differing:
            others.append(f"{name} ran at {sha}, whose provenance files differ there in {listed(differing)}")
    if others:
        return [f"{experiment_id}: {shown} names results of {marked}, but " + "; ".join(others)]
    if not is_ancestor(marked, "HEAD", root):
        return [f"{experiment_id}: {shown} names results of {marked}, which is not on HEAD's history"]
    paths: list[str] = []
    for _, name_paths, _ in stale.values():
        paths.extend(path for path in name_paths if path not in paths)
    changes = changes_after(marked, "HEAD", root, tuple(paths))
    # A first change is one no other change after the results precedes. Two
    # lines of history leaving the results each have their own; the marker
    # may name either.
    if since not in changes or changes_after(marked, since, root, tuple(paths)) != [since]:
        first = changes[0] if changes else None
        errors.append(
            f"{experiment_id}: {shown} says stale since {since}, but the first commit after {marked} "
            f"that changed a provenance file is {first}"
        )
    return errors
