import copy
import importlib.util
import subprocess
import tempfile
import unittest
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_PATH = "experiments/model/M002-v9-factorized-typed-attention"
CANDIDATE = ROOT / CANDIDATE_PATH
# The commit that froze M002-v9 and ran its first seed. model/burn-a0 at this
# commit has the Git-tree digest the study pins; the live tree no longer does
# (the A0 lockfile and, for the successor study, the runner sources changed).
FREEZE_COMMIT = "3ab2070a198ab5e8f782022a43c2f74016149496"
TREE_ERROR = "M002-v9: implementation tree differs from its pinned Git digest"
MANIFEST = {
    "id": "M002-v9",
    "status": "prepared",
    "entrypoint": (
        "cargo run -- --phase paired-v5 --experiment M002-v9 "
        "--arms factorized-v2-v9,factorized-v2-off-v9 "
        "--folds <folds> --rank <rank> --bias-limit <bias_limit> "
        "--metadata-dropout <metadata_dropout>"
    ),
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def v9_labels(stdout):
    return (
        stdout.replace("factorized-v2-off", "__CONTROL__")
        .replace("factorized-v2", "factorized-v2-v9")
        .replace("__CONTROL__", "factorized-v2-off-v9")
    )


class M002V9CandidateTests(unittest.TestCase):
    """The v9 binding is evaluated against the repository as the freeze commit
    held it (a clone of this one, checked out detached at `FREEZE_COMMIT`), so
    the checks stay meaningful although the live `model/burn-a0` moved on and
    the study is superseded. The gate code under test is the live one."""

    @classmethod
    def setUpClass(cls):
        held = subprocess.run(
            ["git", "-C", str(ROOT), "cat-file", "-e", f"{FREEZE_COMMIT}^{{commit}}"],
            capture_output=True,
        )
        if held.returncode != 0:
            raise unittest.SkipTest(f"the M002-v9 freeze commit {FREEZE_COMMIT[:12]} is not in this clone's history")
        scratch = tempfile.TemporaryDirectory(prefix="m002-v9-freeze-")
        cls.addClassCleanup(scratch.cleanup)
        cls.frozen_root = Path(scratch.name) / "repo"
        for command in (
            ["git", "clone", "--quiet", "--shared", "--no-checkout", str(ROOT), str(cls.frozen_root)],
            ["git", "-C", str(cls.frozen_root), "checkout", "--quiet", "--detach", FREEZE_COMMIT],
        ):
            subprocess.run(command, check=True, capture_output=True)
        cls.frozen_candidate = cls.frozen_root / CANDIDATE_PATH
        cls.gates = load_module("m002_v9_frozen_gate", ROOT / "scripts/check_research_gates.py")

    def frozen_config(self):
        return tomllib.loads((self.frozen_candidate / "config.toml").read_text(encoding="utf-8"))["preregistration"]

    def test_superseded_candidate_is_no_longer_held_to_the_freeze_gate(self):
        gates = self.gates
        real = tomllib.loads((CANDIDATE / "experiment.toml").read_text(encoding="utf-8"))
        self.assertEqual(real["status"], "superseded")
        self.assertEqual(gates.m002_v9_successor_freeze_errors(ROOT, CANDIDATE, real), [])
        # Held to the gate as a prepared study, the repository as it stood at the
        # freeze satisfies the whole binding (fold identity, pilot archive,
        # contract, pinned files, runner pair, tree digest) ...
        frozen = {**MANIFEST, "entrypoint": real["entrypoint"]}
        self.assertEqual(gates.m002_v9_successor_freeze_errors(self.frozen_root, self.frozen_candidate, frozen), [])
        # ... while the live repository no longer does: the pinned runner sources
        # and the Git-tree digest of model/burn-a0 changed after the supersession.
        # That is why the study stays superseded rather than being re-pinned.
        live = gates.m002_v9_successor_freeze_errors(ROOT, CANDIDATE, frozen)
        self.assertIn(TREE_ERROR, live)
        self.assertTrue(all(error.startswith("M002-v9:") for error in live))

    def test_runner_tree_digest_and_cross_study_pair_are_hard_requirements(self):
        gates = self.gates
        config = self.frozen_config()
        root = self.frozen_root
        pair = "factorized-v2-v9,factorized-v2-off-v9"
        self.assertEqual(gates.m002_v9_fold_binding_errors(root, config), [])
        self.assertEqual(gates.m002_versioned_runner_binding_errors(root, config, MANIFEST, "M002-v9", pair), [])
        invalid = dict(config)
        invalid["arm_pair"] = "factorized-v2-v8,factorized-v2-off-v8"
        self.assertTrue(gates.m002_versioned_runner_binding_errors(root, invalid, MANIFEST, "M002-v9", pair))
        invalid = dict(config)
        invalid["implementation_tree_git_digest"] = "0" * 64
        self.assertEqual(
            gates.m002_versioned_runner_binding_errors(root, invalid, MANIFEST, "M002-v9", pair), [TREE_ERROR]
        )
        invalid = dict(config)
        invalid["runner_arm_table_sha256"] = "0" * 64
        self.assertTrue(gates.m002_versioned_runner_binding_errors(root, invalid, MANIFEST, "M002-v9", pair))

    def test_arm_mode_is_checked_inside_each_arms_own_block(self):
        gates = self.gates
        arm_table = (self.frozen_root / "model/burn-a0/examples/a0_ablation/arms.rs").read_text(encoding="utf-8")
        treatment, control = "factorized-v2-v9", "factorized-v2-off-v9"
        self.assertEqual(gates.m002_arm_block_binding_errors(arm_table, "M002-v9", treatment, control), [])
        control_block = arm_table[arm_table.index(f'name: "{control}"'):]
        control_block = control_block[: control_block.index("\n    },")]
        # The control arm edited to run the treatment mode: both mode names
        # still occur elsewhere in the file, so only a per-block check sees it.
        swapped = arm_table.replace(
            control_block,
            control_block.replace("TypedAttentionMode::Off", "TypedAttentionMode::FactorizedV2"),
        )
        self.assertNotEqual(swapped, arm_table)
        errors = gates.m002_arm_block_binding_errors(swapped, "M002-v9", treatment, control)
        self.assertEqual(len(errors), 1)
        self.assertIn(control, errors[0])
        # The treatment arm edited to the off mode is rejected the same way.
        treatment_block = arm_table[arm_table.index(f'name: "{treatment}"'):]
        treatment_block = treatment_block[: treatment_block.index("\n    },")]
        swapped = arm_table.replace(
            treatment_block,
            treatment_block.replace("TypedAttentionMode::FactorizedV2", "TypedAttentionMode::Off"),
        )
        errors = gates.m002_arm_block_binding_errors(swapped, "M002-v9", treatment, control)
        self.assertEqual(len(errors), 1)
        self.assertIn(treatment, errors[0])
        # An arm that is missing altogether is rejected.
        self.assertEqual(len(gates.m002_arm_block_binding_errors("", "M002-v9", treatment, control)), 2)

    def test_entrypoint_must_pass_every_frozen_architecture_flag(self):
        gates = self.gates
        config = self.frozen_config()
        manifest = tomllib.loads((self.frozen_candidate / "experiment.toml").read_text(encoding="utf-8"))
        root = self.frozen_root
        pair = "factorized-v2-v9,factorized-v2-off-v9"
        self.assertEqual(gates.m002_versioned_runner_binding_errors(root, config, manifest, "M002-v9", pair), [])
        for flag, placeholder in gates.M002_ENTRYPOINT_FLAGS:
            with self.subTest(flag=flag):
                self.assertIn(f"{flag} {placeholder}", manifest["entrypoint"])
                dropped = {**manifest, "entrypoint": manifest["entrypoint"].replace(f"{flag} {placeholder}", "")}
                errors = gates.m002_versioned_runner_binding_errors(root, config, dropped, "M002-v9", pair)
                self.assertEqual(errors, [f"M002-v9: entrypoint must pass {flag} {placeholder}"])

    def test_adapter_restores_v9_identity_and_rejects_unknown_labels(self):
        v5_tests = load_module("m002_v5_fixture", ROOT / "scripts/tests/test_aggregate_m002_v5.py")
        adapter = load_module("aggregate_m002_v9", ROOT / "scripts/aggregate_m002_v9.py")
        core = adapter.load_core()
        records = []
        for name, record in v5_tests.records():
            clone = copy.deepcopy(record)
            clone["experiment_id"] = "M002-v9"
            clone["manifest"]["id"] = "M002-v9"
            clone["stdout"] = v9_labels(clone["stdout"])
            records.append((name, clone))
        decision = adapter.decide(records, protocol=core.fixture_protocol())
        self.assertEqual(decision["experiment_id"], "M002-v9")
        malformed = copy.deepcopy(records)
        malformed[0][1]["stdout"] = malformed[0][1]["stdout"].replace(
            "factorized-v2-v9", "factorized-v2-v9-unknown", 1
        )
        with self.assertRaisesRegex(adapter.EvidenceError, "unrecognized v9 arm label"):
            adapter.decide(malformed, protocol=core.fixture_protocol())


if __name__ == "__main__":
    unittest.main()
