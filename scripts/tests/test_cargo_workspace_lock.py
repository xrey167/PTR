import contextlib
import hashlib
import importlib.util
import io
import json
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "run_experiment", ROOT / "scripts/run_experiment.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)




class CargoWorkspaceLockTests(unittest.TestCase):

    def test_archived_a0_jobs_cannot_launch_through_the_exploratory_path(self):
        with patch.object(mod, "resolve") as resolve, patch.object(mod, "execute_command") as execute:
            with contextlib.redirect_stderr(io.StringIO()):
                for experiment in ("M001", "M002", "M003", "M004"):
                    for phase in ("a0_sweep_entrypoint", "a0_ablation_entrypoint", "a0_rerun_entrypoint", "a0_contingency_entrypoint"):
                        self.assertEqual(mod.run_experiment(experiment, entrypoint=phase, seed=17, params={}), 2)
            resolve.assert_not_called()
            execute.assert_not_called()

    def test_cargo_lock_follows_selected_workspace_and_toolchain(self):
        for manifest, suffix in (("Cargo.toml", []),
                                 ("model/burn-a0/Cargo.toml", ["--manifest-path", "model/burn-a0/Cargo.toml"]),
                                 ("model/burn-a0/Cargo.toml", ["--manifest-path=model/burn-a0/Cargo.toml"]),
                                 ("Cargo.toml", ["--manifest-path", "crates/ptr-types/Cargo.toml"])):
            with self.subTest(suffix=suffix), patch.object(mod.subprocess, "run", return_value=Mock(stdout=str(ROOT / manifest))) as query:
                command = ["cargo", "+1.95.0", "run", *suffix, "--", "--manifest-path", "not-cargo.toml"]
                result = mod.cargo_lock_record(command)
                lock = Path(manifest).with_name("Cargo.lock")
                self.assertEqual(result, {"cargo_lock_path": lock.as_posix(), "cargo_lock_sha256": mod.sha(ROOT / lock)})
                argv = query.call_args.args[0]
                self.assertEqual(argv[:5], ["cargo", "+1.95.0", "locate-project", "--workspace", "--message-format"])
                self.assertNotIn("not-cargo.toml", argv)
                if suffix:
                    self.assertEqual(argv[-2:], ["--manifest-path", suffix[-1].split("=", 1)[-1]])

    def test_cargo_lock_missing_or_outside_repo_is_not_root_fallback(self):
        for manifest in (ROOT / "missing/Cargo.toml", Path("/tmp/outside/Cargo.toml")):
            with self.subTest(manifest=manifest), patch.object(mod.subprocess, "run", return_value=Mock(stdout=str(manifest))):
                with self.assertRaisesRegex(ValueError, "cannot bind Cargo workspace lockfile"):
                    mod.cargo_lock_record(["cargo", "run"])

    def test_non_cargo_command_has_no_claimed_cargo_lock(self):
        with patch.object(mod.subprocess, "run") as query:
            self.assertEqual(mod.cargo_lock_record(["python3", "train.py"]),
                             {"cargo_lock_path": None, "cargo_lock_sha256": None})
            query.assert_not_called()
