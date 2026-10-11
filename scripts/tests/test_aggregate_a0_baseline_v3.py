"""Tests of the A0 baseline v3-m003 decider (scripts/aggregate_a0_baseline_v3.py).

The v3 decider is the v2 decider (scripts/aggregate_a0_baseline_v2.py, imported
unchanged and pinned) with only its study name, default criteria path, criteria loader
and reported script digest rebound. Every statistic, gate, rule and refusal is the
v2/v1 decider's and is tested by scripts/tests/test_aggregate_a0_baseline_v2.py and
test_aggregate_a0_baseline.py; here: the binding to the v3 criteria (equal to the v2
values for M003 except ids), the pins, that the v2 module itself is untouched, and the
INCONCLUSIVE-INTERRUPTED outcome through the v3 entry points.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import tomllib
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mod = load("aggregate_a0_baseline_v3_under_test", SCRIPTS / "aggregate_a0_baseline_v3.py")
pristine_v2 = load("aggregate_a0_baseline_v2_pristine", SCRIPTS / "aggregate_a0_baseline_v2.py")
V3_CRITERIA = ROOT / "research/falsification/A0-baseline-v3-m003/criteria.toml"
V2_CRITERIA = ROOT / "research/falsification/A0-baseline-v2/criteria.toml"
CRITERIA = mod.load_criteria()
STUDY = "M003-v4"


def v1_scenario_module() -> types.ModuleType:
    """The synthetic-row builder of the v1 tests (Scenario), pointed at the v3 decider and M003-v4."""
    path = SCRIPTS / "tests/test_aggregate_a0_baseline.py"
    source = path.read_text(encoding="utf-8")
    for old, new in (("M003-v2", "M003-v4"), ("A0-baseline-v1", "A0-baseline-v3-m003"),
                     ("aggregate_a0_baseline.py", "aggregate_a0_baseline_v3.py")):
        source = source.replace(old, new)
    module = types.ModuleType("v1_cases_for_v3")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


gates = load("check_research_gates_under_v3_test", SCRIPTS / "check_research_gates.py")


def reservation(seed: int) -> dict:
    """What the runner leaves when it is killed: no stdout, no exit code."""
    return {
        "experiment_id": STUDY, "entrypoint": "entrypoint", "manifest": {"id": STUDY}, "seed": seed,
        "status": "started", "git_sha": "a" * 40, "started_at": f"20261010T1{seed % 100:02d}000.000000Z",
    }


def completed(seed: int) -> dict:
    record = reservation(seed)
    record.update(status="completed", exit_code=0, stdout="not rows\n")
    return record


class V3Criteria(unittest.TestCase):
    def test_the_values_are_the_v2_values_for_m003_except_ids_name_and_the_additions(self):
        v2 = V2_CRITERIA.read_text(encoding="utf-8").replace("M003-v3", "M003-v4")
        old, new = tomllib.loads(v2), tomllib.loads(V3_CRITERIA.read_text(encoding="utf-8"))
        self.assertEqual((old.pop("name"), new.pop("name")), ("A0-baseline-v2", "A0-baseline-v3-m003"))
        # M001-v6 and M002-v11 are decided in v2 and not repeated.
        self.assertEqual([s["id"] for s in old["study"]], ["M001-v6", "M002-v11", "M003-v4"])
        old["study"] = [s for s in old["study"] if s["id"] == STUDY]
        self.assertEqual([s["id"] for s in new["study"]], [STUDY])
        for key in ("parent", "parent_sha256"):
            self.assertIn(key, new["decision"])
            new["decision"] = {k: v for k, v in new["decision"].items() if k != key}
        self.assertEqual(new.pop("sweep_reuse"), {
            "study": STUDY, "experiment": "M003-v3-sweep",
            "lr_file": "research/falsification/A0-baseline-v2/lr_M003-v3.tsv",
        })
        self.assertEqual(new, old)

    def test_the_learning_rate_file_is_a_byte_copy_of_the_v2_selection(self):
        self.assertEqual(
            (ROOT / "research/falsification/A0-baseline-v3-m003/lr_M003-v4.tsv").read_bytes(),
            (ROOT / "research/falsification/A0-baseline-v2/lr_M003-v3.tsv").read_bytes(),
        )

    def test_the_v2_decider_is_pinned_unchanged(self):
        pin = CRITERIA["raw"]["decision"]
        self.assertEqual(pin["parent"], "scripts/aggregate_a0_baseline_v2.py")
        self.assertEqual(pin["parent_sha256"], hashlib.sha256((SCRIPTS / "aggregate_a0_baseline_v2.py").read_bytes()).hexdigest())

    def test_each_decider_accepts_only_its_own_criteria(self):
        self.assertEqual(list(CRITERIA["studies"]), [STUDY])
        with self.assertRaisesRegex(mod.EvidenceError, "name is not A0-baseline-v3-m003"):
            mod.load_criteria(V2_CRITERIA)
        # The v2 file loaded on its own is not changed by importing the v3 layer.
        self.assertEqual(pristine_v2.NAME, "A0-baseline-v2")
        with self.assertRaisesRegex(pristine_v2.EvidenceError, "name is not A0-baseline-v2"):
            pristine_v2.load_criteria(V3_CRITERIA)
        self.assertEqual(len(pristine_v2.load_criteria()["studies"]), 3)

    def test_a_tampered_pin_or_reuse_table_is_refused(self):
        text = V3_CRITERIA.read_text(encoding="utf-8")
        cases = {
            "parent pin": text.replace(CRITERIA["raw"]["decision"]["parent_sha256"], "0" * 64),
            "parent": text.replace('parent = "scripts/aggregate_a0_baseline_v2.py"', 'parent = "scripts/aggregate_a0_baseline.py"'),
            "base pin": text.replace(CRITERIA["raw"]["decision"]["base_sha256"], "0" * 64),
            "label": text.replace("INCONCLUSIVE-INTERRUPTED", "INCONCLUSIVE"),
            "reuse table": text.replace("[sweep_reuse]", "[sweep_reuse_x]"),
            "reuse study": text.replace('study = "M003-v4"', 'study = "M003-v9"', 1),
            "name": text.replace('name = "A0-baseline-v3-m003"', 'name = "A0-baseline-v2"'),
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, changed in cases.items():
                self.assertNotEqual(changed, text, name)
                path = Path(directory) / f"{name.replace(' ', '_')}.toml"
                path.write_text(changed, encoding="utf-8")
                with self.subTest(name=name), self.assertRaises(mod.EvidenceError):
                    mod.load_criteria(path)


class V3Outcome(unittest.TestCase):
    def sources(self, interrupted: int = 71):
        return [(f"s{seed}.json", reservation(seed) if seed == interrupted else completed(seed)) for seed in (17, 29, 43, 71)]

    def test_an_interrupted_seed_is_the_named_outcome_of_this_study(self):
        result = mod.decide(self.sources(), STUDY)
        self.assertEqual(result["study"], "A0-baseline-v3-m003")
        self.assertEqual(result["experiment_id"], STUDY)
        self.assertEqual(result["headline"], {"contrast": "M003-v4-primary", "verdict": "INCONCLUSIVE", "label": "INCONCLUSIVE-INTERRUPTED"})
        self.assertEqual(result["interrupted"], {"seeds_interrupted": [71], "seeds_completed": [17, 29, 43], "seeds_not_run": [101]})
        self.assertEqual([c["id"] for c in result["contrasts"]], ["M003-v4-primary", "M003-v4-nonlinearity", "M003-v4-bundle"])
        self.assertFalse(any(c["quality_claim_allowed"] for c in result["contrasts"]))
        # The v2 study ids are not studies of this criteria.
        with self.assertRaisesRegex(mod.EvidenceError, "unknown study"):
            mod.decide(self.sources(), "M003-v3")
        with self.assertRaisesRegex(mod.EvidenceError, "unknown study"):
            mod.decide(self.sources(), "M001-v6")
        with self.assertRaisesRegex(mod.EvidenceError, "deferred"):
            mod.decide(self.sources(), "M004-v2")

    def test_the_outputs_carry_the_v3_script_digest_and_study_name(self):
        digest = hashlib.sha256((SCRIPTS / "aggregate_a0_baseline_v3.py").read_bytes()).hexdigest()
        self.assertEqual(mod.script_sha256(), digest)
        self.assertNotEqual(digest, hashlib.sha256((SCRIPTS / "aggregate_a0_baseline_v2.py").read_bytes()).hexdigest())
        result = mod.decide(self.sources(), STUDY)
        with tempfile.TemporaryDirectory() as directory:
            ci, decision = Path(directory) / "paired-ci-95.json", Path(directory) / "DECISION.toml"
            mod.write_outputs(result, ci, decision)
            parsed = tomllib.loads(decision.read_text(encoding="utf-8"))
            self.assertEqual((parsed["study"], parsed["experiment_id"], parsed["go"]), ("A0-baseline-v3-m003", STUDY, False))
            self.assertEqual(parsed["decision_script_sha256"], digest)
            self.assertEqual(parsed["criteria_sha256"], CRITERIA["sha256"])
            self.assertEqual(parsed["label"], "INCONCLUSIVE-INTERRUPTED")
            self.assertEqual(json.loads(ci.read_text(encoding="utf-8"))["status"], "not-computed-interrupted")
            with self.assertRaisesRegex(mod.EvidenceError, "interrupted study publishes no aggregate"):
                mod.publish_results(result, Path(directory))

    def test_the_cli_uses_the_v3_criteria_by_default_and_exits_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for name, record in self.sources():
                path = root / name
                path.write_text(json.dumps(record), encoding="utf-8")
                paths.append(str(path))
            args = ["--study", STUDY, "--criteria-sha256", CRITERIA["sha256"], "--paired-ci", str(root / "ci.json"),
                    "--decision", str(root / "D.toml")]
            with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(mod.main([*args, *paths]), 0, err.getvalue())
            summary = json.loads(out.getvalue())
            self.assertEqual(summary["headline"]["label"], "INCONCLUSIVE-INTERRUPTED")
            self.assertEqual(summary["interrupted"]["seeds_interrupted"], [71])
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
                bad = ["--study", STUDY, "--criteria-sha256", "0" * 64, "--paired-ci", str(root / "c2.json"), "--decision", str(root / "D2.toml")]
                self.assertEqual(mod.main([*bad, *paths]), 2)
            self.assertIn("--criteria-sha256", err.getvalue())

    def test_the_v2_entry_points_still_run_v2_criteria(self):
        # Rebinding happened on a private copy: the v2 decider still decides v2's studies.
        result = pristine_v2.decide([("s.json", {**reservation(71), "experiment_id": "M003-v3", "manifest": {"id": "M003-v3"}})], "M003-v3")
        self.assertEqual(result["study"], "A0-baseline-v2")
        self.assertEqual(result["provenance"]["decision_script_sha256"],
                         hashlib.sha256((SCRIPTS / "aggregate_a0_baseline_v2.py").read_bytes()).hexdigest())


class V3Publish(unittest.TestCase):
    """The aggregate of a complete study carries what the research gate binds a listed
    experiment's aggregate to: both preregistration digests and seed_records."""

    @classmethod
    def setUpClass(cls):
        cls.cases = v1_scenario_module()

    def build(self, started_seed=None):
        root = Path(self.enterContext(tempfile.TemporaryDirectory(prefix="a0-v3-publish-")))
        experiment = root / "model/M003-v4-latent-recurrence"
        results = experiment / "results"
        results.mkdir(parents=True)
        real = ROOT / "experiments/model/M003-v4-latent-recurrence"
        table = tomllib.loads((real / "config.toml").read_text(encoding="utf-8"))["preregistration"]
        entry = tomllib.loads((ROOT / "experiments/preregistration.toml").read_text(encoding="utf-8"))["experiment"][STUDY]
        self.digest = gates.experiment_records.preregistration_digest(table)
        self.rules = gates.experiment_records.canonical_digest(entry)
        (experiment / "experiment.toml").write_text(
            f'id = "{STUDY}"\npreregistration_sha256 = "{self.digest}"\npreregistration_rules_sha256 = "{self.rules}"\n',
            encoding="utf-8")
        scenario = self.cases.Scenario(STUDY)
        sources = []
        for index, seed in enumerate(self.cases.SEEDS):
            record = scenario.record(seed, started_at=f"20261010T1{index}0000.000000Z",
                                     finished_at="20261010T120000.000000Z", stderr="")
            if seed == started_seed:
                record = {k: v for k, v in record.items() if k not in ("stdout", "stderr", "exit_code", "finished_at")}
                record["status"] = "started"
            name = f"run-20261010T1{index}0000.000000Z-seed-{seed}.json"
            (results / name).write_text(json.dumps(record), encoding="utf-8")
            sources.append((name, record))
        return results, sources

    def test_a_complete_study_publishes_an_aggregate_the_gate_accepts(self):
        results, sources = self.build()
        result = mod.decide(sources, STUDY)
        self.assertNotIn("interrupted", result)
        mod.publish_results(result, results)
        run = json.loads((results / "run.json").read_text(encoding="utf-8"))
        # The gate's own functions, on the written aggregate.
        self.assertEqual(run["preregistration_sha256"], self.digest)
        self.assertEqual(run["preregistration_rules_sha256"], self.rules)
        digests = gates.experiment_records.seed_record_digests(results)
        self.assertEqual(sorted(digests), ["101", "17", "29", "43", "71"])
        self.assertEqual(run["seed_records"], digests)
        for seed in ("17", "29", "43", "71", "101"):  # the gate's seed loop: every preregistered seed bound
            self.assertEqual(run["seed_records"].get(seed), digests[seed])
        self.assertEqual(set(run["seed_records"]) - set(digests), set())
        self.assertEqual(gates.experiment_records.aggregate_problems(run, results), [])
        self.assertEqual(run["metrics_sha256"], hashlib.sha256((results / "metrics.json").read_bytes()).hexdigest())
        self.assertEqual(run["decision_script_sha256"], mod.script_sha256())
        # A record rewritten after the aggregate no longer matches what it binds.
        victim = next(results.glob("run-*seed-17.json"))
        victim.write_text(victim.read_text(encoding="utf-8") + " ", encoding="utf-8")
        self.assertNotEqual(gates.experiment_records.seed_record_digests(results)["17"], run["seed_records"]["17"])

    def test_the_cli_publishes_the_same_aggregate(self):
        results, sources = self.build()
        out = Path(self.enterContext(tempfile.TemporaryDirectory()))
        args = ["--study", STUDY, "--criteria-sha256", CRITERIA["sha256"], "--paired-ci", str(out / "ci.json"),
                "--decision", str(out / "D.toml"), "--publish-results", str(results), *[str(results / n) for n, _ in sources]]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(mod.main(args), 0, err.getvalue())
        run = json.loads((results / "run.json").read_text(encoding="utf-8"))
        self.assertEqual((run["preregistration_sha256"], run["preregistration_rules_sha256"]), (self.digest, self.rules))
        self.assertEqual(len(run["seed_records"]), 5)

    def test_one_started_record_is_interrupted_and_publishes_nothing(self):
        results, sources = self.build(started_seed=71)
        result = mod.decide(sources, STUDY)
        self.assertEqual(result["headline"]["label"], "INCONCLUSIVE-INTERRUPTED")
        with self.assertRaisesRegex(mod.EvidenceError, "interrupted study publishes no aggregate"):
            mod.publish_results(result, results)
        out = Path(self.enterContext(tempfile.TemporaryDirectory()))
        args = ["--study", STUDY, "--criteria-sha256", CRITERIA["sha256"], "--paired-ci", str(out / "ci.json"),
                "--decision", str(out / "D.toml"), "--publish-results", str(results), *[str(results / n) for n, _ in sources]]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            mod.main(args)
        self.assertFalse((results / "run.json").exists())
        self.assertFalse((results / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
