import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "run_m002_v5_pilot", ROOT / "scripts/run_m002_v5_pilot.py"
)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


class M002V5PilotTests(unittest.TestCase):
    def lock_root(self, lock: dict) -> tempfile.TemporaryDirectory:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        path = root / MOD.LOCK_PATH
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(lock), encoding="utf-8")
        return temporary

    def test_command_matrix_is_frozen_and_deterministic(self):
        digests = {
            "evidence-interventional": "1" * 16,
            "claim-temporal-development": "2" * 16,
        }
        cells = list(MOD.plan(digests))
        self.assertEqual(len(cells), 16)
        self.assertEqual(len({name for name, *_ in cells}), 16)
        self.assertEqual([seed for _, _, seed, _ in cells[:4]], [7, 13, 7, 13])
        candidates = [candidate for _, candidate, seed, _ in cells if seed == 7]
        self.assertEqual(
            candidates,
            [
                {"rank": rank, "bias_limit": bias, "metadata_dropout": dropout}
                for rank in (8, 16)
                for bias in (1, 2)
                for dropout in (0.0, 0.1)
            ],
        )
        command = cells[0][3]
        self.assertEqual(command[:7], [
            "cargo", "run", "--release", "--locked", "--quiet", "--jobs", "1",
        ])
        self.assertIn("factorized-v2,factorized-v2-off", command)
        self.assertEqual(command[command.index("--folds") + 1],
            "evidence-interventional=" + "1" * 16 +
            ",claim-temporal-development=" + "2" * 16)
        for option, value in (("--steps", "1500"), ("--lr", "0.005"), ("--d-model", "48")):
            self.assertEqual(command[command.index(option) + 1], value)

    def test_fold_digests_are_read_and_identity_validated(self):
        lock = {"folds": {
            "evidence-interventional": {
                "kind": "diagnostic", "target_role": "evidence",
                "target_regime": "interventional", "data_fnv1a64": "ABCDEF0123456789",
            },
            "claim-temporal-development": {
                "kind": "development", "target_role": "claim",
                "target_regime": "temporal", "data_fnv1a64": "0123456789abcdef",
            },
        }}
        with self.lock_root(lock) as directory:
            self.assertEqual(MOD.fold_digests(Path(directory)), {
                "evidence-interventional": "abcdef0123456789",
                "claim-temporal-development": "0123456789abcdef",
            })
        lock["folds"]["evidence-interventional"]["target_role"] = "claim"
        with self.lock_root(lock) as directory:
            with self.assertRaisesRegex(MOD.PilotError, "identity"):
                MOD.fold_digests(Path(directory))

    def test_dirty_worktree_refuses_real_run(self):
        clean_sha = subprocess.CompletedProcess([], 0, "a" * 40 + "\n", "")
        detached = subprocess.CompletedProcess([], 1, "", "")
        dirty = subprocess.CompletedProcess([], 0, " M other-agent.py\n", "")
        with mock.patch.object(MOD.subprocess, "run", side_effect=[clean_sha, detached, dirty]):
            with self.assertRaisesRegex(MOD.PilotError, "not clean and committed"):
                MOD.require_clean_detached_worktree(Path("repo/output"), Path("repo"))

    def test_execution_environment_is_minimal_and_drops_inherited_compiler_overrides(self):
        with mock.patch.dict(
            MOD.os.environ,
            {
                "PATH": "C:/toolchain",
                "SystemRoot": "C:/Windows",
                "RUSTC": "C:/unbound-rustc.exe",
                "RUSTC_WRAPPER": "C:/wrapper.exe",
                "SECRET": "not-an-input",
            },
            clear=True,
        ):
            environment = MOD.execution_environment(Path("C:/isolated"))
        self.assertEqual(environment["PATH"], "C:/toolchain")
        self.assertEqual(environment["SystemRoot"], "C:/Windows")
        self.assertEqual(environment["CARGO_HOME"], str(Path("C:/isolated") / "cargo-home"))
        self.assertNotIn("RUSTC", environment)
        self.assertNotIn("RUSTC_WRAPPER", environment)
        self.assertNotIn("SECRET", environment)

    def test_resume_accepts_only_complete_exact_success_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            candidate = {"rank": 8, "bias_limit": 1, "metadata_dropout": 0.0}
            digests = {name: str(index) * 16 for index, name in enumerate(MOD.FOLDS, 1)}
            command = MOD.command_for(candidate, 7, digests)
            expected = MOD.expected_identity("a" * 40, command, candidate, 7, digests, {"test": True})
            name = MOD.cell_name(candidate, 7)
            stdout_path, stderr_path, metadata_path = MOD.artifact_paths(output, name)
            stdout_path.write_bytes(b"out\n")
            stderr_path.write_bytes(b"err\n")
            metadata = {
                **expected, "exit_code": 0, "duration_seconds": 1.25,
                "stdout_file": stdout_path.name, "stderr_file": stderr_path.name,
                "stdout_sha256": MOD.sha256(b"out\n"),
                "stderr_sha256": MOD.sha256(b"err\n"),
            }
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            self.assertTrue(MOD.resumable(output, name, expected))
            mismatched = {**expected, "seed": 13}
            with self.assertRaisesRegex(MOD.PilotError, "do not match"):
                MOD.resumable(output, name, mismatched)
            metadata["stdout_sha256"] = "0" * 64
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(MOD.PilotError, "not a complete successful"):
                MOD.resumable(output, name, expected)
            metadata_path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(MOD.PilotError, "not a JSON object"):
                MOD.resumable(output, name, expected)

    def test_run_cell_preserves_raw_streams_and_records_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            candidate = {"rank": 8, "bias_limit": 1, "metadata_dropout": 0.0}
            digests = {name: str(index) * 16 for index, name in enumerate(MOD.FOLDS, 1)}
            command = MOD.command_for(candidate, 7, digests)
            expected = MOD.expected_identity("a" * 40, command, candidate, 7, digests, {"test": True})
            name = MOD.cell_name(candidate, 7)
            completed = subprocess.CompletedProcess(command, 0, b"raw\x00stdout", b"raw\xffstderr")
            with (
                mock.patch.object(MOD.subprocess, "run", return_value=completed),
                mock.patch.object(MOD.time, "monotonic", side_effect=[10.0, 12.5]),
            ):
                self.assertEqual(
                    MOD.run_cell(Path("repo"), output, name, command, expected, {"X": "1"}),
                    0,
                )
            stdout_path, stderr_path, metadata_path = MOD.artifact_paths(output, name)
            self.assertEqual(stdout_path.read_bytes(), b"raw\x00stdout")
            self.assertEqual(stderr_path.read_bytes(), b"raw\xffstderr")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["duration_seconds"], 2.5)
            self.assertEqual(metadata["stdout_sha256"], MOD.sha256(b"raw\x00stdout"))
            self.assertEqual(metadata["stderr_sha256"], MOD.sha256(b"raw\xffstderr"))

    def test_dry_run_does_not_check_git_create_output_or_execute_cargo(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / MOD.LOCK_PATH
            lock.parent.mkdir(parents=True)
            lock.write_text(json.dumps({"folds": {
                "evidence-interventional": {
                    "kind": "diagnostic", "target_role": "evidence",
                    "target_regime": "interventional", "data_fnv1a64": "1" * 16,
                },
                "claim-temporal-development": {
                    "kind": "development", "target_role": "claim",
                    "target_regime": "temporal", "data_fnv1a64": "2" * 16,
                },
            }}), encoding="utf-8")
            output = root / "must-not-exist"
            with mock.patch.object(MOD.subprocess, "run") as ran:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(MOD.execute(output, True, root), 0)
            ran.assert_not_called()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
