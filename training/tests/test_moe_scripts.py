import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

TRAINING = Path(__file__).resolve().parents[1]
CONFIGS = TRAINING / "configs"


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_moe_tests", TRAINING / "scripts" / "run_moe_tests.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Result:
    def __init__(self, ok=True, skipped=()):
        self._ok, self.skipped = ok, list(skipped)

    def wasSuccessful(self):
        return self._ok


class RunMoeTestsScriptTests(unittest.TestCase):
    def setUp(self):
        self.runner = _load_runner()
        # Hermetic: never spawn a real interpreter (a present training/.venv would re-exec the
        # whole suite recursively), and never leak the re-exec guard between tests.
        env = mock.patch.dict(self.runner.os.environ)
        env.start()
        self.addCleanup(env.stop)
        self.runner.os.environ.pop("PTR_MOE_TESTS_REEXEC", None)
        venv = mock.patch.object(self.runner, "VENV_PYTHON", Path("/nonexistent/python"))
        venv.start()
        self.addCleanup(venv.stop)
        spawn = mock.patch.object(
            self.runner.subprocess, "run", side_effect=AssertionError("unexpected spawn")
        )
        spawn.start()
        self.addCleanup(spawn.stop)

    def _run(self, argv, result):
        with mock.patch.object(
            self.runner.unittest.TextTestRunner, "run", return_value=result
        ), mock.patch.object(self.runner.unittest.defaultTestLoader, "discover"):
            return self.runner.main(argv)

    def test_default_allows_skips(self):
        self.assertEqual(self._run([], _Result(skipped=[("t", "no torch")])), 0)

    def test_require_backend_fails_on_skips(self):
        from ptr_training.moe import pipeline

        with mock.patch.object(pipeline, "require_extra"):
            self.assertEqual(self._run(["--require-backend"], _Result(skipped=[("t", "x")])), 3)
            self.assertEqual(self._run(["--require-backend"], _Result()), 0)
            self.assertEqual(self._run(["--require-backend"], _Result(ok=False)), 1)

    def test_require_backend_fails_when_extra_missing(self):
        from ptr_training.moe import pipeline

        err = pipeline.MissingExtraError("needs the moe extra")
        with mock.patch.object(pipeline, "require_extra", side_effect=err):
            self.assertEqual(self._run(["--require-backend"], _Result()), 2)

    def test_reexecs_with_training_venv_when_extra_missing(self):
        from ptr_training.moe import pipeline

        err = pipeline.MissingExtraError("needs the moe extra")
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            pipeline, "require_extra", side_effect=err
        ):
            fake = Path(tmp) / "python"
            fake.write_text("")
            with mock.patch.object(self.runner, "VENV_PYTHON", fake), mock.patch.object(
                self.runner.subprocess, "run", return_value=mock.Mock(returncode=7)
            ) as run:
                self.assertEqual(self._run(["--require-backend"], _Result()), 7)
                argv = run.call_args.args[0]
                self.assertEqual(argv[0], str(fake))
                self.assertEqual(argv[-1], "--require-backend")
                self.assertEqual(run.call_args.kwargs["env"]["PTR_MOE_TESTS_REEXEC"], "1")
                # already re-executed once: no loop, plain failure
                run.reset_mock()
                self.runner.os.environ["PTR_MOE_TESTS_REEXEC"] = "1"
                self.assertEqual(self._run(["--require-backend"], _Result()), 2)
                run.assert_not_called()
            # no venv interpreter: fail clearly with the environment-missing code
            self.runner.os.environ.pop("PTR_MOE_TESTS_REEXEC", None)
            self.assertEqual(self._run(["--require-backend"], _Result()), 2)

    def test_candidate_command_uses_flag(self):
        text = (
            TRAINING.parent / "evaluations" / "components" / "training-backend" / "candidates.toml"
        ).read_text(encoding="utf-8")
        self.assertIn("run_moe_tests.py --require-backend", text)


class UnselectedBackendGateTests(unittest.TestCase):
    def _main(self, training_table, execute=True):
        from ptr_training import run

        manifest = {"git_dirty": False, "config": {"training": training_table}}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "m.json"
            argv = ["run", "--config", str(CONFIGS / "stage-moe-smoke.toml"), "--out", str(out)]
            if execute:
                argv.append("--execute")
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                run, "build_manifest", return_value=manifest
            ), mock.patch.object(run, "ROOT", Path(tmp)), mock.patch.object(
                run, "_execute_trl_moe", return_value=0
            ) as execute_mock, mock.patch.object(
                run, "resolve_trl_moe_preflight", return_value={"status": "matched"}
            ):
                try:
                    run.main()
                    code = None
                except SystemExit as exit_:
                    code = exit_.code
            written = out.read_text() if out.exists() else ""
        return code, execute_mock, written

    def test_refused_without_opt_in(self):
        code, execute_mock, written = self._main({"backend": "trl-moe"})
        self.assertIn("not selected", str(code))
        execute_mock.assert_not_called()
        self.assertIn("unselected backend, pipeline evidence only", written)

    def test_refused_when_opt_in_not_true(self):
        code, execute_mock, _ = self._main({"backend": "trl-moe", "allow_unselected_backend": False})
        self.assertIn("not selected", str(code))
        execute_mock.assert_not_called()

    def test_allowed_with_explicit_opt_in(self):
        code, execute_mock, written = self._main(
            {"backend": "trl-moe", "allow_unselected_backend": True}
        )
        self.assertEqual(code, 0)
        execute_mock.assert_called_once()
        self.assertIn("unselected", written)


if __name__ == "__main__":
    unittest.main()
