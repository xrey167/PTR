import contextlib
import errno
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "run_experiment", ROOT / "scripts/run_experiment.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


@contextlib.contextmanager
def disk_full_writing(name: str):
    """Fail every write to a file whose name contains `name` as a full disk
    does: the file is opened, and truncated when its mode says so, and its
    first write raises ENOSPC."""
    real_open = io.open

    def opener(file, mode="r", *args, **kwargs):
        handle = real_open(file, mode, *args, **kwargs)
        if isinstance(file, (str, os.PathLike)) and name in Path(file).name and set(mode) & set("wxa+"):

            def full(*_args, **_kwargs):
                raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

            handle.write = full
        return handle

    with mock.patch("io.open", side_effect=opener), mock.patch("builtins.open", side_effect=opener):
        yield


class ExperimentRunnerTests(unittest.TestCase):
    def test_registry_resolves_known_experiment(self):
        item, root, data = mod.resolve("M001")
        self.assertEqual(data["id"], "M001")
        self.assertTrue(root.exists())
        self.assertEqual(item["status"], data["status"])

    def test_build_command_substitutes_seed_and_parameters(self):
        _, _, data = mod.resolve("L001")
        command = mod.build_command(
            data,
            entrypoint="entrypoint",
            seed=17,
            params={"iterations": "3"},
        )
        self.assertEqual(
            command,
            [
                "cargo",
                "run",
                "--release",
                "--locked",
                "-p",
                "ptr-bench",
                "--",
                "ledger-recovery",
                "3",
                "17",
            ],
        )

    def test_build_command_rejects_a_seed_given_as_a_parameter(self):
        # The record names the seed --seed gave, so no parameter replaces it.
        _, _, data = mod.resolve("L001")
        with self.assertRaisesRegex(ValueError, "the seed is given by --seed, not by a parameter"):
            mod.build_command(data, entrypoint="entrypoint", seed=17, params={"iterations": "3", "seed": "99"})

    def test_build_command_rejects_unlisted_seed(self):
        _, _, data = mod.resolve("L001")
        with self.assertRaisesRegex(ValueError, "not declared"):
            mod.build_command(
                data,
                entrypoint="entrypoint",
                seed=999,
                params={"iterations": "3"},
            )

    def test_build_command_rejects_unresolved_placeholder(self):
        _, _, data = mod.resolve("L001")
        with self.assertRaisesRegex(ValueError, "missing value"):
            mod.build_command(data, entrypoint="entrypoint", seed=17)

    def test_execute_command_captures_process_evidence(self):
        result = mod.execute_command(
            [sys.executable, "-c", "print('runner-ok')"]
        )
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["stdout"].strip(), "runner-ok")
        self.assertEqual(result["stderr"], "")
        self.assertIsNone(result["launch_error"])
        self.assertGreaterEqual(result["duration_ns"], 0)

    def test_a_seed_run_from_a_dirty_source_tree_is_refused_before_it_runs(self):
        # A record names HEAD as the code it ran; an edit reverted before
        # aggregation would otherwise be attributed to the clean revision.
        stderr = io.StringIO()
        with (
            mock.patch.object(
                mod.experiment_records, "uncommitted_files", return_value=["crates/ptr-pg/src/lib.rs"]
            ) as dirty,
            mock.patch.object(mod, "execute_command") as execute,
            mock.patch.object(mod, "write_json_exclusive") as write,
            contextlib.redirect_stderr(stderr),
        ):
            status = mod.run_experiment("L001", entrypoint="entrypoint", seed=17, params={"iterations": "3"})
        self.assertEqual(status, 2)
        execute.assert_not_called()
        write.assert_not_called()
        self.assertIn("refusing to run", stderr.getvalue())
        self.assertIn("crates/ptr-pg/src/lib.rs", stderr.getvalue())
        root, pathspecs = dirty.call_args.args
        self.assertEqual(root, mod.ROOT)
        experiment = "experiments/lifecycle/L001-revocation-crash"
        expected = ("*.rs", "Cargo.lock", "scripts/run_experiment.py", f"{experiment}/aggregate.py", experiment)
        for spec in expected:
            self.assertIn(spec, pathspecs)
        self.assertIn(f":(exclude){experiment}/results", pathspecs)

    def test_a_seed_run_from_a_clean_source_tree_runs_and_is_recorded(self):
        with (
            mock.patch.object(mod.experiment_records, "uncommitted_files", return_value=[]),
            mock.patch.object(
                mod, "execute_command", return_value={"exit_code": 0, "duration_ns": 1}
            ) as execute,
            mock.patch.object(mod, "write_json_exclusive") as write,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            status = mod.run_experiment("L001", entrypoint="entrypoint", seed=17, params={"iterations": "3"})
        self.assertEqual(status, 0)
        execute.assert_called_once()
        self.assertEqual(write.call_args.args[1]["seed"], 17)


def git(root: Path, *args: str) -> str:
    command = [
        "git",
        "-c", "user.name=PTR tests",
        "-c", "user.email=tests@example.invalid",
        "-c", "commit.gpgsign=false",
        "-c", "init.defaultBranch=main",
        # No automatic maintenance: a commit would start it detached, and it
        # can still be writing into .git/objects when the test deletes the
        # repository.
        "-c", "maintenance.auto=false",
        "-c", "gc.auto=0",
        *args,
    ]
    return subprocess.run(command, cwd=root, check=True, capture_output=True, text=True).stdout.strip()


class RunWatchTests(unittest.TestCase):
    """A seed record names the commit its run started from, so its sources must
    stay that commit's from before the run until the record is written: a
    `cargo run` entrypoint compiles whatever the tree holds while it runs."""

    ORIGINAL = "pub fn f() {}\n"

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        files = {
            ".gitignore": "__pycache__/\n",
            "Cargo.lock": "# lock\n",
            "README.md": "readme\n",
            "src/lib.rs": self.ORIGINAL,
            "experiments/registry.toml": '[[experiment]]\nid = "L900"\npath = "x/L900-x"\nstatus = "running"\n',
            "experiments/x/L900-x/experiment.toml": (
                'id = "L900"\nstatus = "running"\nseeds = [17]\nentrypoint = "bench <seed>"\n'
            ),
            "experiments/x/L900-x/results/.gitkeep": "",
            # L900 is not listed, so it runs without a preregistration.
            "experiments/preregistration.toml": "version = 1\n",
        }
        for relative, text in files.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        git(self.root, "init", "-q")
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "--no-verify", "-m", "code")
        self.head = git(self.root, "rev-parse", "HEAD")
        # A source last written long ago, so rewriting it with the same text
        # during a run changes its stamp whatever the clock's resolution.
        os.utime(self.root / "src/lib.rs", ns=(10**18, 10**18))
        self.results = self.root / "experiments/x/L900-x/results"

    def tearDown(self):
        self.directory.cleanup()

    def run_seed(self, during=None):
        """Run seed 17 of L900 in the temporary repository, calling `during`
        while the command "runs"; returns the exit status, the records written
        and what was printed to stderr."""

        def execute(_command):
            if during is not None:
                during()
            return {"exit_code": 0, "stdout": "", "stderr": "", "launch_error": None, "duration_ns": 1}

        stderr = io.StringIO()
        with (
            mock.patch.object(mod, "ROOT", self.root),
            mock.patch.object(mod, "REGISTRY", self.root / "experiments/registry.toml"),
            mock.patch.object(mod, "execute_command", side_effect=execute),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(stderr),
        ):
            status = mod.run_experiment("L900", entrypoint="entrypoint", seed=17)
        records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(self.results.glob("run-*.json"))]
        return status, records, stderr.getvalue()

    def write(self, relative: str, text: str) -> None:
        (self.root / relative).write_text(text, encoding="utf-8")

    def test_a_seed_run_whose_tree_stays_at_head_is_recorded_at_the_commit_it_started_from(self):
        status, records, stderr = self.run_seed()
        self.assertEqual((status, stderr), (0, ""))
        self.assertEqual([record["git_sha"] for record in records], [self.head])

    def test_a_seed_run_whose_source_is_edited_and_restored_while_it_runs_is_not_recorded(self):
        # The build may have compiled the edit; the tree looks clean again
        # when the run ends.
        def edit_and_restore():
            self.write("src/lib.rs", "pub fn f() { panic!() }\n")
            self.write("src/lib.rs", self.ORIGINAL)

        status, records, stderr = self.run_seed(edit_and_restore)
        self.assertEqual((status, records), (2, []))
        self.assertIn("not recording", stderr)
        self.assertIn("src/lib.rs", stderr)
        self.assertEqual(git(self.root, "status", "--porcelain"), "")

    def test_a_seed_run_whose_sources_are_left_edited_or_added_to_is_not_recorded(self):
        for relative in ("src/lib.rs", "src/new.rs", "experiments/x/L900-x/config.toml"):
            with self.subTest(file=relative):
                try:
                    status, records, stderr = self.run_seed(lambda: self.write(relative, "changed\n"))
                    self.assertEqual((status, records), (2, []))
                    self.assertIn(relative, stderr)
                finally:
                    git(self.root, "checkout", "-q", "--", ".")
                    git(self.root, "clean", "-qfd")

    def test_a_seed_run_during_which_head_moves_is_not_recorded(self):
        status, records, stderr = self.run_seed(
            lambda: git(self.root, "commit", "-q", "--no-verify", "--allow-empty", "-m", "moved")
        )
        self.assertEqual((status, records), (2, []))
        self.assertIn(f"HEAD moved from {self.head}", stderr)

    def test_a_record_whose_write_fails_leaves_no_partial_record(self):
        # A partial record is the newest of its seed, and the aggregator reads
        # every record it selects from.
        with disk_full_writing("-seed-17.json"), self.assertRaises(OSError):
            self.run_seed()
        self.assertEqual(sorted(path.name for path in self.results.iterdir()), [".gitkeep"])

    def test_a_record_never_replaces_another(self):
        path = self.results / "run-1-seed-17.json"
        path.write_text('{"first": true}\n', encoding="utf-8")
        with self.assertRaises(FileExistsError):
            mod.write_json_exclusive(path, {"second": True})
        self.assertEqual(sorted(p.name for p in self.results.iterdir()), [".gitkeep", "run-1-seed-17.json"])
        self.assertEqual(path.read_text(encoding="utf-8"), '{"first": true}\n')
        other = self.results / "run-2-seed-17.json"
        mod.write_json_exclusive(other, {"second": True})
        self.assertEqual(json.loads(other.read_text(encoding="utf-8")), {"second": True})

    def preregister(self, status: str, table: str | None = "schema = 1\n", frozen: bool = True) -> None:
        """List L900 as preregistering one integer, give it `status`, the
        `[preregistration]` table `table` (None for none) and, when `frozen`,
        the manifest's digests of that table and of its list entry; commit."""
        listed = 'version = 1\n\n[experiment.L900.required]\nseeds = "int-list"\nschema = "int"\n'
        self.write("experiments/preregistration.toml", listed)
        if table is not None:
            table = "seeds = [17]\n" + table
        manifest = f'id = "L900"\nstatus = "{status}"\nseeds = [17]\nentrypoint = "bench <seed>"\n'
        self.write("experiments/x/L900-x/config.toml", "version = 1\n" + ("" if table is None else "\n[preregistration]\n" + table))
        if table is not None:
            if frozen:
                digest = mod.experiment_records.preregistration_digest(tomllib.loads(table))
                rules = mod.experiment_records.canonical_digest(tomllib.loads(listed)["experiment"]["L900"])
                manifest += f'preregistration_sha256 = "{digest}"\npreregistration_rules_sha256 = "{rules}"\n'
        self.write("experiments/x/L900-x/experiment.toml", manifest)
        self.write(
            "experiments/registry.toml",
            f'[[experiment]]\nid = "L900"\npath = "x/L900-x"\nstatus = "{status}"\n',
        )
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "--no-verify", "-m", f"preregister at {status}")
        self.head = git(self.root, "rev-parse", "HEAD")

    def test_a_listed_experiment_is_not_run_or_prepared_before_its_preregistration_is_frozen(self):
        cases = [
            ("planned", "schema = 1\n", True, "L900 preregisters (experiments/preregistration.toml) and is 'planned'"),
            ("running", None, True, "L900: config.toml has no [preregistration] table"),
            ("running", 'schema = "must-be-pinned-before-prepared"\n', True, "L900: preregistration key schema is a placeholder"),
            ("running", "schema = 1\n", False, "L900: experiment.toml names no preregistration_sha256"),
        ]
        for status, table, frozen, refusal in cases:
            with self.subTest(status=status, table=table, frozen=frozen):
                self.preregister(status, table, frozen)
                ran = []
                code, records, stderr = self.run_seed(lambda: ran.append(True))
                self.assertEqual((code, records, ran), (2, [], []))
                self.assertIn(refusal, stderr)
                prepared = io.StringIO()
                with (
                    mock.patch.object(mod, "ROOT", self.root),
                    mock.patch.object(mod, "REGISTRY", self.root / "experiments/registry.toml"),
                    contextlib.redirect_stderr(prepared),
                ):
                    self.assertEqual(mod.prepare("L900"), 2)
                self.assertIn(refusal, prepared.getvalue())
                self.assertEqual(sorted(path.name for path in self.results.iterdir()), [".gitkeep"])

    def test_a_listed_experiment_runs_once_frozen_and_its_record_is_bound_to_that_preregistration(self):
        self.preregister("running")
        status, records, stderr = self.run_seed()
        self.assertEqual((status, stderr), (0, ""))
        manifest = records[0]["manifest"]
        self.assertEqual(
            manifest["preregistration_sha256"],
            mod.experiment_records.preregistration_digest({"seeds": [17], "schema": 1}),
        )
        self.assertEqual(records[0]["git_sha"], self.head)
        # Committed, the record passes the gate: the commit it names holds
        # the preregistration frozen now and the manifest it names.
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "--no-verify", "-m", "record")
        self.assertEqual(mod.check_research_gates.launch_errors(self.root, "L900"), [])
        # The same experiment runs again under the same freeze.
        status, records, _ = self.run_seed()
        self.assertEqual((status, len(records)), (0, 2))

    def test_what_decides_a_launch_is_held_to_head_before_and_while_the_run(self):
        self.preregister("running")
        listed = (self.root / "experiments/preregistration.toml").read_text(encoding="utf-8")
        # Edited, and not committed: refused before it runs, even where the
        # edit leaves the experiment's entry as it was.
        self.write("experiments/preregistration.toml", listed + "# edited\n")
        ran = []
        status, records, stderr = self.run_seed(lambda: ran.append(True))
        self.assertEqual((status, records, ran), (2, [], []))
        self.assertIn("refusing to run", stderr)
        self.assertIn("experiments/preregistration.toml", stderr)
        # An entry taken out of the list does not open the runner either: an
        # experiment the list has named stays listed.
        self.write("experiments/preregistration.toml", "version = 1\n")
        status, records, stderr = self.run_seed(lambda: ran.append(True))
        self.assertEqual((status, records, ran), (2, [], []))
        self.assertIn("L900 was listed at", stderr)
        self.write("experiments/preregistration.toml", listed)
        # Edited while it runs, and put back: no record.
        def edit_and_restore():
            self.write("experiments/preregistration.toml", "version = 1\n")
            self.write("experiments/preregistration.toml", listed)

        status, records, stderr = self.run_seed(edit_and_restore)
        self.assertEqual((status, records), (2, []))
        self.assertIn("experiments/preregistration.toml", stderr)
        self.assertEqual(
            mod.check_research_gates.launch_inputs(self.root, "L900"),
            ["experiments/preregistration.toml", "experiments/registry.toml", "scripts/check_research_gates.py"],
        )

    def test_a_results_directory_below_the_experiment_and_no_link_holds_the_records(self):
        manifest = (self.root / "experiments/x/L900-x/experiment.toml").read_text(encoding="utf-8")
        # The experiment's own directory would take all of it out of the
        # HEAD check, and one outside it is not the experiment's.
        for results_dir in (".", "../elsewhere"):
            with self.subTest(results_dir=results_dir):
                self.write("experiments/x/L900-x/experiment.toml", manifest + f'results_dir = "{results_dir}"\n')
                git(self.root, "commit", "-q", "--no-verify", "-am", f"results in {results_dir}")
                ran = []
                status, _, stderr = self.run_seed(lambda: ran.append(True))
                self.assertEqual((status, ran), (2, []))
                self.assertIn(f"results_dir {results_dir!r} is not a directory below experiments/x/L900-x", stderr)
        self.write("experiments/x/L900-x/experiment.toml", manifest)
        git(self.root, "commit", "-q", "--no-verify", "-am", "results in results")
        # A link would carry the records, and what the check leaves out,
        # elsewhere.
        shutil.rmtree(self.results)
        try:
            os.symlink(self.root, self.results)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"cannot create a symlink: {error}")
        ran = []
        status, _, stderr = self.run_seed(lambda: ran.append(True))
        self.assertEqual((status, ran), (2, []))
        self.assertIn("results directory experiments/x/L900-x/results is a symlink", stderr)

    def test_prepare_holds_what_decides_a_launch_to_head(self):
        self.preregister("prepared")
        listed = (self.root / "experiments/preregistration.toml").read_text(encoding="utf-8")

        def prepare() -> tuple[int, str]:
            stderr = io.StringIO()
            with (
                mock.patch.object(mod, "ROOT", self.root),
                mock.patch.object(mod, "REGISTRY", self.root / "experiments/registry.toml"),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(stderr),
            ):
                return mod.prepare("L900"), stderr.getvalue()

        # An uncommitted edit to the list, even one that leaves the entry as
        # it was, refuses the prepare record: it would name a HEAD that is not
        # the tree the decision read.
        self.write("experiments/preregistration.toml", listed + "# edited\n")
        status, stderr = prepare()
        self.assertEqual(status, 2)
        self.assertIn("refusing to run", stderr)
        self.assertIn("experiments/preregistration.toml", stderr)
        self.assertEqual(sorted(path.name for path in self.results.iterdir()), [".gitkeep"])
        self.write("experiments/preregistration.toml", listed)
        status, stderr = prepare()
        self.assertEqual((status, stderr), (0, ""))
        [record] = [json.loads(path.read_text(encoding="utf-8")) for path in self.results.glob("run-*.json")]
        self.assertEqual((record["status"], record["git_sha"]), ("prepared", self.head))

    def test_a_record_the_results_hold_is_no_change_of_the_sources(self):
        # Records accumulate in results/, which the run writes into itself.
        status, records, _ = self.run_seed(lambda: self.write("experiments/x/L900-x/results/other.json", "{}\n"))
        self.assertEqual(status, 0)
        self.assertEqual(len(records), 1)


if __name__ == "__main__":
    unittest.main()
