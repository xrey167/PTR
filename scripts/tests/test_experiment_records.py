import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
# Imported as the aggregators import it, so patching it here patches theirs.
sys.path.insert(0, str(ROOT / "scripts"))
import experiment_records as mod  # noqa: E402

AGGREGATORS = {
    "L003": ROOT / "experiments/lifecycle/L003-fastmem-revocation",
    "L004": ROOT / "experiments/lifecycle/L004-projection-equivalence",
}
MANIFEST = {"id": "L900", "status": "running", "seeds": [1, 2], "entrypoint": "run <seed>"}


def record(sha: str, seed: int = 1, started_at: str = "20260101T000000Z", **changes) -> dict:
    """A run record as `scripts/run_experiment.py run` writes it."""
    base = {
        "experiment_id": "L900",
        "git_sha": sha,
        "manifest": dict(MANIFEST),
        "cargo_lock_sha256": "c" * 64,
        "parameters": {"iterations": "30"},
        "entrypoint": "entrypoint",
        "seed": seed,
        "started_at": started_at,
        "exit_code": 0,
    }
    base.update(changes)
    return base


def git(root: Path, *args: str) -> str:
    command = [
        "git",
        "-c", "user.name=PTR tests",
        "-c", "user.email=tests@example.invalid",
        "-c", "commit.gpgsign=false",
        "-c", "init.defaultBranch=main",
        *args,
    ]
    return subprocess.run(command, cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def commit(root: Path, files: dict[str, str], message: str) -> str:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-q", "--no-verify", "-m", message)
    return git(root, "rev-parse", "HEAD")


class AgreementTests(unittest.TestCase):
    def test_records_of_one_configuration_agree_whatever_the_status(self):
        # The manifest's status changes when the experiment completes.
        records = {"a.json": record("a" * 40), "b.json": record("b" * 40, seed=2)}
        self.assertEqual(mod.agreement_errors("L900", {**MANIFEST, "status": "completed"}, records), [])

    def test_a_record_of_another_experiment_or_of_no_commit_is_refused(self):
        records = {
            "other.json": record("a" * 40, experiment_id="L901"),
            "unknown.json": record("unknown"),
            "option.json": record("--output=/tmp/x"),
            "none.json": record(None),
        }
        errors = mod.agreement_errors("L900", MANIFEST, records)
        self.assertEqual(len(errors), 4)
        self.assertIn("other.json is a record of 'L901', not 'L900'", errors)
        for name in ("unknown.json", "option.json", "none.json"):
            self.assertTrue(any(error.startswith(f"{name} names no commit") for error in errors), name)

    def test_records_of_different_configurations_are_refused(self):
        cases = {
            "parameters": {"parameters": {"iterations": "3"}},
            "Cargo.lock": {"cargo_lock_sha256": "d" * 64},
            "entrypoint": {"entrypoint": "smoke"},
        }
        for label, change in cases.items():
            with self.subTest(label=label):
                records = {"a.json": record("a" * 40), "b.json": record("a" * 40, seed=2, **change)}
                errors = mod.agreement_errors("L900", MANIFEST, records)
                self.assertEqual(len(errors), 1)
                self.assertTrue(errors[0].startswith(f"the records disagree on {label}: "), errors[0])
                self.assertIn("b.json", errors[0])

    def test_a_record_run_under_another_manifest_is_refused(self):
        records = {
            "a.json": record("a" * 40),
            "b.json": record("a" * 40, seed=2, manifest={**MANIFEST, "seeds": [1, 2, 3]}),
        }
        self.assertEqual(
            mod.agreement_errors("L900", MANIFEST, records),
            ["b.json ran under an experiment.toml that differs from the current one"],
        )
        with self.assertRaisesRegex(mod.ProvenanceError, "b.json ran under"):
            mod.source_revision("L900", MANIFEST, records, ROOT)
        with self.assertRaisesRegex(mod.ProvenanceError, "no run records"):
            mod.source_revision("L900", MANIFEST, {}, ROOT)


class RevisionTests(unittest.TestCase):
    """`source_revision` against a scratch repository."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        git(self.root, "init", "-q")
        self.first = commit(
            self.root,
            {"Cargo.toml": "[workspace]\n", "src/lib.rs": "pub fn f() {}\n", "results/a.json": "{}\n"},
            "code",
        )
        # Archiving a record commits only the record.
        self.archived = commit(self.root, {"results/b.json": "{}\n"}, "archive")

    def tearDown(self):
        self.directory.cleanup()

    def records(self, *shas: str) -> dict[str, dict]:
        return {
            f"run-{index}.json": record(sha, seed=index, started_at=f"20260101T00000{index}Z")
            for index, sha in enumerate(shas)
        }

    def test_records_archived_one_commit_at_a_time_share_their_code(self):
        revision = mod.source_revision("L900", MANIFEST, self.records(self.first, self.archived), self.root)
        self.assertEqual(revision, self.first)
        # The earliest record names the revision, whatever order they come in.
        records = self.records(self.archived, self.first)
        records["run-0.json"]["started_at"] = "20260102T000000Z"
        self.assertEqual(mod.source_revision("L900", MANIFEST, records, self.root), self.first)

    def test_records_that_ran_different_code_are_refused(self):
        changed = commit(self.root, {"src/lib.rs": "pub fn f() { g() }\n"}, "change")
        with self.assertRaisesRegex(mod.ProvenanceError, r"run-1\.json ran at .* differs in src/lib\.rs"):
            mod.source_revision("L900", MANIFEST, self.records(self.first, changed), self.root)

    def test_a_checkout_whose_code_changed_since_the_records_is_refused(self):
        records = self.records(self.first, self.archived)
        for relative in ("src/lib.rs", "Cargo.lock", "migrations/0001.sql", "crates/x/Cargo.toml"):
            with self.subTest(changed=relative):
                (self.root / relative).parent.mkdir(parents=True, exist_ok=True)
                (self.root / relative).write_text("-- changed\n", encoding="utf-8")
                git(self.root, "add", "-A")
                # Uncommitted, and then committed.
                with self.assertRaisesRegex(mod.ProvenanceError, "code has changed since"):
                    mod.source_revision("L900", MANIFEST, records, self.root)
                git(self.root, "commit", "-q", "--no-verify", "-m", relative)
                with self.assertRaisesRegex(mod.ProvenanceError, f"code has changed since, in {relative}"):
                    mod.source_revision("L900", MANIFEST, records, self.root)
                git(self.root, "reset", "-q", "--hard", self.archived)

    def test_a_change_outside_the_code_is_not_a_change_of_code(self):
        (self.root / "README.md").write_text("notes\n", encoding="utf-8")
        commit(self.root, {"results/run.json": "{}\n"}, "aggregate")
        records = self.records(self.first, self.archived)
        self.assertEqual(mod.source_revision("L900", MANIFEST, records, self.root), self.first)

    def test_a_commit_git_cannot_find_is_refused(self):
        with self.assertRaisesRegex(mod.ProvenanceError, "cannot compare"):
            mod.source_revision("L900", MANIFEST, self.records("0" * 40), self.root)


class AggregatorTests(unittest.TestCase):
    """The L003 and L004 aggregators bind their output to the records' commit."""

    RESULT_KEYS = (
        "writes_replayed",
        "revocations_with_removal",
        "full_refold_writes",
        "window_denials",
        "window_reads",
        "append_races_append_first",
        "append_races",
        "checkpoint_races_stored",
        "checkpoint_races",
        "revocation_crashes_committed",
        "revocation_crashes",
        "max_replayed",
        "max_journal_len",
    )

    def aggregate(self, exp_id: str, change=None, code_changes=()):
        """Run `exp_id`'s aggregator over synthetic records, one per declared
        seed at commits "1111111" and "2222222" with the same code, into a
        scratch results directory; `change` edits the last record, and
        `code_changes` are the code files the checkout changed since. Returns
        the run.json it wrote, or raises the SystemExit it refused with."""

        def changes(_base, head, _root):
            return [] if head is not None else list(code_changes)

        here = AGGREGATORS[exp_id]
        spec = importlib.util.spec_from_file_location(f"aggregate_{exp_id}", here / "aggregate.py")
        agg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agg)
        manifest = tomllib.loads((here / "experiment.toml").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            inputs = []
            for index, seed in enumerate(manifest["seeds"]):
                result = {"seed": seed, "iterations": 2, "server": "PostgreSQL 18", "hard_failures": 0}
                result.update({key: 0 for key in self.RESULT_KEYS})
                run = {
                    **record("1111111" if index < 2 else "2222222", seed=seed),
                    "experiment_id": exp_id,
                    "manifest": {**manifest, "status": "running"},
                    "started_at": f"20260101T00000{index}Z",
                    "stdout": "Compiling\n" + json.dumps(result) + "\n",
                }
                if change and index == len(manifest["seeds"]) - 1:
                    run.update(change)
                path = results / f"run-20260101T00000{index}Z-seed-{seed}.json"
                path.write_text(json.dumps(run), encoding="utf-8")
                inputs.append(str(path))
            with (
                mock.patch.object(agg, "RESULTS", results),
                mock.patch.object(sys, "argv", ["aggregate.py", *inputs]),
                mock.patch.object(mod, "code_changes", side_effect=changes),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                try:
                    agg.main()
                except SystemExit:
                    self.assertFalse((results / "run.json").exists())
                    self.assertFalse((results / "metrics.json").exists())
                    raise
            return json.loads((results / "run.json").read_text(encoding="utf-8"))

    def test_the_aggregate_names_the_commit_the_records_ran_at(self):
        for exp_id in AGGREGATORS:
            with self.subTest(experiment=exp_id):
                run = self.aggregate(exp_id)
                self.assertTrue(run["hard_pass"])
                self.assertEqual(run["git_sha"], "1111111")
                self.assertIn("aggregated_at_git_sha", run)
                self.assertEqual(
                    [seed["git_sha"] for seed in run["seeds"]],
                    ["1111111", "1111111", "2222222", "2222222", "2222222"],
                )

    def test_records_of_other_code_or_configuration_are_not_aggregated(self):
        for exp_id in AGGREGATORS:
            with self.subTest(experiment=exp_id, refused="stale code"):
                with self.assertRaisesRegex(SystemExit, "refusing to aggregate: .* changed since"):
                    self.aggregate(exp_id, code_changes=["crates/ptr-pg/src/lib.rs"])
            with self.subTest(experiment=exp_id, refused="parameters"):
                with self.assertRaisesRegex(SystemExit, "disagree on parameters"):
                    self.aggregate(exp_id, change={"parameters": {"iterations": "3"}})
            with self.subTest(experiment=exp_id, refused="experiment"):
                with self.assertRaisesRegex(SystemExit, "is a record of 'L999'"):
                    self.aggregate(exp_id, change={"experiment_id": "L999"})


if __name__ == "__main__":
    unittest.main()
