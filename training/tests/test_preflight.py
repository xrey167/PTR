import json
import tempfile
import unittest
from pathlib import Path

from ptr_training import preflight
from ptr_training.moe.pipeline import stack_available

HAVE_STACK = stack_available(("torch", "trl", "transformers", "datasets", "peft"))
SMOKE = Path(__file__).resolve().parents[1] / "configs" / "stage-moe-smoke.toml"

PROTOCOL = "p" * 64
RUNNER = "r" * 64
DEVICE = "cpu"


def record(**overrides) -> dict:
    fields = {
        "protocol_sha256": PROTOCOL,
        "runner_sha256": RUNNER,
        "device": DEVICE,
        "input_fingerprint_sha256": "i" * 64,
        "probe": {"sft_step_loss": 1.5},
        "created_at": "2026-10-13T00:00:00+00:00",
    }
    fields.update(overrides)
    return preflight.build_record(**fields)


def require(directory: Path, **overrides):
    arguments = {
        "protocol_sha256": PROTOCOL,
        "runner_sha256": RUNNER,
        "device": DEVICE,
        "evidence_class": "pipeline-smoke",
        "allow_missing": False,
        "directory": directory,
    }
    arguments.update(overrides)
    return preflight.require_preflight(**arguments)


class RecordTests(unittest.TestCase):
    def test_a_record_binds_the_three_hashes_and_verifies(self):
        built = record()
        self.assertEqual(preflight.verify_record(built), built)
        for field in ("protocol_sha256", "runner_sha256", "device", "input_fingerprint_sha256"):
            self.assertIn(field, built)

    def test_editing_any_field_breaks_the_record_hash(self):
        for field, value in (("device", "cuda:other"), ("protocol_sha256", "q" * 64),
                             ("probe", {"sft_step_loss": 0.1})):
            tampered = {**record(), field: value}
            with self.assertRaises(preflight.PreflightError, msg=field):
                preflight.verify_record(tampered)

    def test_a_record_without_a_bound_field_is_rejected(self):
        broken = record()
        del broken["runner_sha256"]
        with self.assertRaises(preflight.PreflightError):
            preflight.verify_record(broken)

    def test_a_foreign_schema_is_rejected(self):
        with self.assertRaises(preflight.PreflightError):
            preflight.verify_record({**record(), "schema": "other"})

    def test_device_identity_names_the_gpu(self):
        self.assertEqual(preflight.device_identity("cpu", None), "cpu")
        self.assertEqual(preflight.device_identity("cuda", "NVIDIA RTX 3090"), "cuda:NVIDIA RTX 3090")
        with self.assertRaises(preflight.PreflightError):
            preflight.device_identity("cuda", None)
        with self.assertRaises(preflight.PreflightError):
            preflight.device_identity("tpu", None)


class RunnerHashTests(unittest.TestCase):
    def test_the_hash_is_stable_and_covers_every_source_file(self):
        self.assertEqual(preflight.runner_sha256(), preflight.runner_sha256())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_text("x = 1\n")
            (root / "sub").mkdir()
            (root / "sub" / "b.py").write_text("y = 2\n")
            first = preflight.runner_sha256(root)
            (root / "sub" / "b.py").write_text("y = 3\n")
            self.assertNotEqual(preflight.runner_sha256(root), first)
            (root / "sub" / "b.py").write_text("y = 2\n")
            self.assertEqual(preflight.runner_sha256(root), first)
            (root / "sub" / "new.py").write_text("")
            self.assertNotEqual(preflight.runner_sha256(root), first)

    def test_bytecode_caches_do_not_change_the_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_text("x = 1\n")
            first = preflight.runner_sha256(root)
            (root / "__pycache__").mkdir()
            (root / "__pycache__" / "a.cpython-313.py").write_text("noise")
            self.assertEqual(preflight.runner_sha256(root), first)

    def test_an_empty_package_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(preflight.PreflightError):
            preflight.runner_sha256(Path(tmp))


class RequirePreflightTests(unittest.TestCase):
    def test_a_matching_record_is_returned(self):
        with tempfile.TemporaryDirectory() as tmp:
            written = record()
            preflight.write_record(Path(tmp), written)
            self.assertEqual(require(Path(tmp)), written)

    def test_no_record_refuses_the_run(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(preflight.PreflightError) as c:
            require(Path(tmp))
        self.assertIn("no preflight record matches", str(c.exception))

    def test_a_missing_directory_refuses_the_run(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(preflight.PreflightError):
            require(Path(tmp) / "absent")

    def test_each_bound_hash_must_match(self):
        for field, wrong in (
            ("protocol_sha256", "q" * 64),
            ("runner_sha256", "q" * 64),
            ("device", "cuda:NVIDIA RTX 3090"),
        ):
            with tempfile.TemporaryDirectory() as tmp:
                preflight.write_record(Path(tmp), record(**{field: wrong}))
                with self.assertRaises(preflight.PreflightError, msg=field) as caught:
                    require(Path(tmp))
                self.assertIn(field, str(caught.exception))

    def test_a_tampered_record_is_ignored_and_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = preflight.write_record(Path(tmp), record())
            data = json.loads(path.read_text())
            data["probe"]["sft_step_loss"] = 0.0
            path.write_text(json.dumps(data))
            with self.assertRaises(preflight.PreflightError) as caught:
                require(Path(tmp))
            self.assertIn("modified after it was written", str(caught.exception))

    def test_unreadable_files_do_not_hide_a_good_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "preflight-garbage.json").write_text("{not json")
            written = record()
            preflight.write_record(Path(tmp), written)
            self.assertEqual(require(Path(tmp)), written)

    def test_the_waiver_is_allowed_for_pipeline_smoke_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(require(Path(tmp), allow_missing=True))
            with self.assertRaises(preflight.PreflightError) as caught:
                require(Path(tmp), allow_missing=True, evidence_class="experiment")
            self.assertIn("pipeline-smoke", str(caught.exception))

    def test_a_matching_record_wins_over_the_waiver(self):
        with tempfile.TemporaryDirectory() as tmp:
            written = record()
            preflight.write_record(Path(tmp), written)
            self.assertEqual(require(Path(tmp), allow_missing=True), written)

    def test_records_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            preflight.write_record(Path(tmp), record())
            with self.assertRaises(FileExistsError):
                preflight.write_record(Path(tmp), record())


@unittest.skipUnless(HAVE_STACK, "torch/trl/transformers/datasets/peft not installed (moe extra)")
class PreflightProbeTests(unittest.TestCase):
    def test_probe_takes_one_step_on_the_longest_row_and_binds_the_run(self):
        from ptr_training.moe import config as moe_config

        cfg = moe_config.load(SMOKE)
        probe, fingerprint = preflight.run_probe(cfg)
        self.assertEqual(len(fingerprint), 64)
        self.assertEqual(probe["rollout_new_tokens"], cfg.rl.max_completion_length)
        self.assertGreater(probe["longest_sft_tokens"], 0)
        self.assertEqual(preflight.current_device_identity(cfg.model.device), "cpu")

    def test_the_cli_writes_a_record_the_run_gate_accepts(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = preflight.main(["--config", str(SMOKE), "--out-dir", tmp])
            self.assertEqual(code, 0)
            found = require(
                Path(tmp),
                protocol_sha256=preflight.file_sha256(SMOKE),
                runner_sha256=preflight.runner_sha256(),
            )
        self.assertEqual(found["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
