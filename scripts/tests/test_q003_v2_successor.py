"""Q003-v2 is Q003's successor with every value of the experiment unchanged: only the
identifiers, the declared `command_environment` and the successor note differ."""

import hashlib
import re
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import check_research_gates as gate  # noqa: E402

OLD = ROOT / "experiments/retrieval/Q003-postgres-hybrid"
NEW = ROOT / "experiments/retrieval/Q003-v2-postgres-hybrid"
PLACEHOLDER = "must-be-pinned-before-prepared"
HEX64 = re.compile(r"[0-9a-f]{64}")
# The statuses of a listed experiment from its freeze on; the test holds before the freeze (planned,
# placeholders) and after it (pinned digests), and does not depend on which.
FROZEN = ("prepared", "running", "completed", "failed")
STATUSES = ("planned", *FROZEN)


def load(directory: Path, name: str) -> dict:
    return tomllib.loads((directory / name).read_text(encoding="utf-8"))


def read(directory: Path, name: str) -> str:
    return (directory / name).read_text(encoding="utf-8")


class SuccessorTests(unittest.TestCase):
    def test_criteria_differ_only_in_the_experiment_id(self):
        old, new = load(OLD, "criteria.toml"), load(NEW, "criteria.toml")
        self.assertEqual((old.pop("experiment"), new.pop("experiment")), ("Q003", "Q003-v2"))
        self.assertEqual(old, new)
        self.assertEqual(read(OLD, "criteria.toml").replace('experiment = "Q003"', 'experiment = "Q003-v2"'), read(NEW, "criteria.toml"))

    def test_thresholds_workloads_and_pins_are_identical(self):
        old, new = load(OLD, "config.toml"), load(NEW, "config.toml")
        self.assertEqual(old["pinned_thresholds"], new["pinned_thresholds"])
        pinned_old, pinned_new = dict(old["preregistration"]), dict(new["preregistration"])
        # What names the experiment, and the two digests the freeze pins.
        for key in ("preregistration_text", "criteria"):
            self.assertEqual(
                pinned_new.pop(key), pinned_old.pop(key).replace("Q003-postgres-hybrid", "Q003-v2-postgres-hybrid")
            )
        # Before the freeze the two digests are placeholders; after it they are the files' SHA-256.
        frozen = load(NEW, "experiment.toml")["status"] in FROZEN
        for key, name in (("preregistration_text_sha256", "PREREGISTRATION.md"), ("criteria_sha256", "criteria.toml")):
            value = pinned_new.pop(key)
            if frozen:
                self.assertEqual(value, hashlib.sha256((NEW / name).read_bytes()).hexdigest(), key)
            else:
                self.assertEqual(value, PLACEHOLDER, key)
            self.assertTrue(value == PLACEHOLDER or HEX64.fullmatch(value), key)
        pinned_old.pop("preregistration_text_sha256")
        pinned_old.pop("criteria_sha256")
        self.assertEqual(pinned_old, pinned_new)
        for key in old:
            if key not in ("name", "path", "preregistration", "pinned_thresholds"):
                self.assertEqual(old[key], new[key], key)
        self.assertEqual(new["name"], "Q003-v2-postgres-hybrid")
        self.assertEqual(new["path"], "experiments/retrieval/Q003-v2-postgres-hybrid")

    def test_the_manifest_differs_only_in_the_id_status_digests_and_declared_environment(self):
        old, new = load(OLD, "experiment.toml"), load(NEW, "experiment.toml")
        self.assertEqual((old["id"], new["id"]), ("Q003", "Q003-v2"))
        self.assertIn(new["status"], STATUSES)
        # The freeze pins the two manifest digests, and nothing before it does.
        for key in ("preregistration_sha256", "preregistration_rules_sha256"):
            if new["status"] in FROZEN:
                self.assertRegex(new[key], HEX64.pattern, key)
            else:
                self.assertNotIn(key, new)
        self.assertEqual(
            new["command_environment"], ["PTR_PG_EXPERIMENT_DSN", "PTR_Q003_STRONG_PYTHON", "PTR_Q003_MODELS_ROOT"]
        )
        self.assertEqual(gate.command_environment_errors("Q003-v2", new), [])
        self.assertEqual(gate.preflight_errors("Q003-v2", new), [])
        self.assertEqual(new["preflight"], new["entrypoint"].replace(" <seed>", " --preflight"))
        self.assertNotIn("command_environment", old)
        for key in ("id", "status", "command_environment", "preflight", "preregistration_sha256", "preregistration_rules_sha256"):
            old.pop(key, None)
            new.pop(key, None)
        self.assertEqual(old, new)

    def test_the_mutation_plan_the_driver_and_the_workload_pinner_are_identical(self):
        self.assertEqual(read(OLD, "tests/mutations.toml"), read(NEW, "tests/mutations.toml"))
        self.assertEqual(
            read(OLD, "bm25_driver.py").replace("Q003-postgres-hybrid", "Q003-v2-postgres-hybrid"), read(NEW, "bm25_driver.py")
        )
        self.assertEqual(
            read(OLD, "tests/pin_workload.py").replace("Q003-postgres-hybrid", "Q003-v2-postgres-hybrid"),
            read(NEW, "tests/pin_workload.py"),
        )

    def test_the_aggregator_differs_only_in_the_experiment_id(self):
        def normal(text: str) -> str:
            return re.sub(r"Q003(-v2)?\b", "Q003", text)

        self.assertEqual(normal(read(OLD, "aggregate.py")), normal(read(NEW, "aggregate.py")))
        self.assertIn('EXPERIMENT = "Q003-v2"', read(NEW, "aggregate.py"))

    def test_the_preregistration_is_q003s_text_plus_a_successor_section(self):
        old, new = read(OLD, "PREREGISTRATION.md"), read(NEW, "PREREGISTRATION.md")
        self.assertTrue(new.startswith(old.rstrip("\n") + "\n"))
        section = new[len(old.rstrip("\n")) + 1 :]
        self.assertTrue(section.lstrip().startswith("## 11. Successor of Q003"))
        for fact in ("seed 17", "no product observation", "command_environment", "never its value", "before it reserves a seed"):
            self.assertIn(fact, section)

    def test_q003_is_superseded_with_its_results_in_place(self):
        self.assertEqual(load(OLD, "experiment.toml")["status"], "superseded")
        self.assertTrue((OLD / "SUPERSEDED.md").is_file())
        registry = {entry["id"]: entry for entry in load(ROOT / "experiments", "registry.toml")["experiment"]}
        self.assertEqual(registry["Q003"]["status"], "superseded")
        self.assertEqual(registry["Q003-v2"]["path"], "retrieval/Q003-v2-postgres-hybrid")
        self.assertEqual(registry["Q003-v2"]["status"], load(NEW, "experiment.toml")["status"])
        self.assertIn(registry["Q003-v2"]["status"], STATUSES)
        entries = load(ROOT / "experiments", "preregistration.toml")["experiment"]
        self.assertEqual(entries["Q003"], entries["Q003-v2"])
        self.assertTrue((NEW / "results" / ".gitkeep").is_file())


if __name__ == "__main__":
    unittest.main()
