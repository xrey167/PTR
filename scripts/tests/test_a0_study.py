"""The A0 ablation study v2: its identities, its registration of listed
experiments, the driver and aggregator parameters it selects, and the rule that
keeps v1 refused while v2 can launch."""

import contextlib
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "scripts"))
import a0_study  # noqa: E402
import check_research_gates  # noqa: E402
import experiment_records  # noqa: E402


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver = load("a0_driver_v2_test", "scripts/run_a0_ablation_study.py")
aggregator = load("a0_aggregator_v2_test", "scripts/aggregate_a0_ablation.py")
run_experiment = load("run_experiment_v2_test", "scripts/run_experiment.py")

V2 = a0_study.V2
PLAN = {
    "steps": 1500, "d_model": 48, "sweep": True, "sweep_steps": 1500,
    "arms": {"M001": ["full", "no-semantic-slots", "no-semantic-slots-masked"],
             "M002": ["no-typed-attention", "raw-blind"],
             "M003": ["latent-0", "latent-1", "latent-linear"], "M004": ["frozen-router"]},
}


class Identities(unittest.TestCase):
    def test_v1_is_unchanged_and_v2_is_hardened(self):
        self.assertEqual((a0_study.V1.key, a0_study.V1.prereg_tag, a0_study.V1.hardened),
                         ("v1", "a0-ablation-prereg-v1", False))
        self.assertEqual(a0_study.V1.prereg_commit, "45b2d5491e40171dd242d0bac6cf7d78d9c70746")
        self.assertEqual((V2.key, V2.prereg_tag, V2.hardened, V2.prereg_commit),
                         ("v2", "a0-ablations-v2-prereg", True, None))
        self.assertEqual(V2.directory, "research/falsification/A0-ablations-v2")

    def test_ids_round_trip_through_classify(self):
        self.assertEqual(V2.classify(V2.sweep_id("M003", 0.0125)), ("sweep", "M003"))
        self.assertEqual(V2.learning_rate("M003-a0v2-sweep-lr0p0125"), 0.0125)
        self.assertEqual(V2.classify(V2.eval_id("M002")), ("eval", "M002"))
        self.assertEqual(V2.classify(V2.rerun_id()), ("rerun", "M001"))
        self.assertEqual(V2.classify(V2.contingency_id("M001", "no-semantic-slots")), ("contingency", "M001"))
        for foreign in ("M001", "M001-v2", "M001-v2-semantic-slots", "M005-a0v2-eval", "M001-a0v2-other"):
            self.assertIsNone(V2.classify(foreign), foreign)
        self.assertIsNone(a0_study.V1.classify("M001-a0v2-eval"))

    def test_the_v1_study_files_and_v2_files_are_disjoint(self):
        self.assertNotEqual(a0_study.V1.path, V2.path)
        self.assertNotEqual(a0_study.V1.config, V2.config)


class Frozen(unittest.TestCase):
    def test_v2_criteria_are_v1s_but_for_the_study_name_and_tag(self):
        v1 = (ROOT / a0_study.V1.directory / "criteria.toml").read_text(encoding="utf-8")
        v2 = (ROOT / V2.directory / "criteria.toml").read_text(encoding="utf-8")
        normalised = (v2.replace("A0-ablations-v2", "A0-ablations-v1").replace("study v2", "study v1")
                      .replace("a0-ablations-v2-prereg", "a0-ablation-prereg-v1"))
        self.assertEqual(normalised, v1)

    def test_v2_configuration_is_v1s_but_for_the_study_name(self):
        v1 = tomllib.loads((ROOT / a0_study.V1.config).read_text(encoding="utf-8"))
        v2 = tomllib.loads((ROOT / V2.config).read_text(encoding="utf-8"))
        self.assertEqual(v2.pop("study"), "A0-ablations-v2")
        self.assertEqual(v1.pop("study"), "A0-ablations-v1")
        self.assertEqual(v1, v2)


def scaffold(root: Path) -> None:
    """A minimal tree: the registries, a hardware profile, an lr file."""
    (root / "experiments/model").mkdir(parents=True)
    (root / "experiments/registry.toml").write_text('version = 1\n', encoding="utf-8")
    (root / "experiments/preregistration.toml").write_text('version = 1\n', encoding="utf-8")
    (root / V2.directory).mkdir(parents=True)
    (root / V2.directory / "lr_selection.tsv").write_text("arm\tlr\tval_accuracy\tflag\n", encoding="utf-8")


def make(kind, family, **extra):
    base = dict(arms=PLAN["arms"][family], seeds=[17], steps=PLAN["steps"], d_model=48,
                data="datasets/generated/operator_routing_v1", data_fnv64="0ad71688f09b0d0d")
    return a0_study.registration(V2, kind, family, **{**base, **extra})


class Registration(unittest.TestCase):
    def test_a_registration_is_a_frozen_listed_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scaffold(root)
            reg = make("eval", "M002", seeds=[17, 29, 43, 71, 101], lr_file=f"{V2.directory}/lr_selection.tsv")
            a0_study.write_experiment(root, V2, reg, "hardware/a0-cpu-4core.toml")
            target = root / "experiments" / reg.path
            manifest = tomllib.loads((target / "experiment.toml").read_text(encoding="utf-8"))
            config = tomllib.loads((target / "config.toml").read_text(encoding="utf-8"))
            entry = tomllib.loads((root / "experiments/preregistration.toml").read_text(encoding="utf-8"))["experiment"][reg.exp_id]
            table = config["preregistration"]
            self.assertEqual((manifest["id"], manifest["status"], manifest["seeds"]),
                             ("M002-a0v2-eval", "prepared", [17, 29, 43, 71, 101]))
            self.assertEqual(manifest["preregistration_sha256"], experiment_records.preregistration_digest(table))
            self.assertEqual(manifest["preregistration_rules_sha256"], experiment_records.canonical_digest(entry))
            lr = (root / V2.directory / "lr_selection.tsv").read_bytes()
            self.assertEqual(table["lr_file_sha256"], hashlib.sha256(lr).hexdigest())
            self.assertEqual(set(entry["required"]) - {"lr_file"}, set(table) - {"lr_file", "lr_file_sha256"})
            self.assertIn(f'id = "{reg.exp_id}"', (root / "experiments/registry.toml").read_text(encoding="utf-8"))
            self.assertTrue((target / "results/.gitkeep").exists())
            self.assertEqual(table["arms"], "no-typed-attention,raw-blind")

    def test_the_command_is_one_the_hardened_runner_can_build(self):
        for kind, extra in (("sweep", {"lr": 0.005}), ("eval", {"lr_file": f"{V2.directory}/lr_selection.tsv"})):
            reg = make(kind, "M001", **extra)
            manifest = {"entrypoint": reg.entrypoint, "seeds": reg.seeds}
            table = dict(reg.table)
            self.assertEqual(check_research_gates.command_errors(reg.exp_id, manifest, table), [], kind)
            placeholders = {name for token in reg.entrypoint.split() for name in experiment_records.PLACEHOLDER.findall(token)}
            self.assertEqual(placeholders - {"seed"}, set(reg.table) - {"seeds"})
            self.assertNotIn("--target-dir", reg.entrypoint)
            self.assertNotIn("--set", reg.entrypoint)

    def test_sweep_and_evaluation_differ_only_where_the_phase_does(self):
        sweep = make("sweep", "M004", lr=0.002)
        evaluation = make("eval", "M004", lr_file="x")
        self.assertIn("--phase sweep", sweep.entrypoint)
        self.assertIn("--lr <lr>", sweep.entrypoint)
        self.assertIn("--phase eval", evaluation.entrypoint)
        self.assertIn("--lr-file <lr_file>", evaluation.entrypoint)
        self.assertEqual(sweep.table["lr"], "0.002")

    def test_a_second_registration_of_the_same_experiment_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scaffold(root)
            reg = make("sweep", "M001", lr=0.002)
            a0_study.write_experiment(root, V2, reg, "hardware/a0-cpu-4core.toml")
            with self.assertRaises(ValueError):
                a0_study.write_experiment(root, V2, reg, "hardware/a0-cpu-4core.toml")


class DriverSelectsTheStudy(unittest.TestCase):
    def setUp(self):
        self.addCleanup(driver.use_study, "v1")

    def test_v1_is_the_default(self):
        self.assertEqual(driver.STUDY.key, "v1")
        self.assertEqual(driver.PREREG_TAG, "a0-ablation-prereg-v1")
        self.assertEqual(driver.STUDY_DIR, ROOT / "research/falsification/A0-ablations-v1")

    def test_v2_constants(self):
        driver.use_study("v2")
        self.assertEqual(driver.PREREG_TAG, "a0-ablations-v2-prereg")
        self.assertEqual(driver.STUDY_DIR, ROOT / "research/falsification/A0-ablations-v2")
        self.assertEqual(driver.CONFIG, ROOT / "model/configs/a0_ablation_study_v2.toml")
        self.assertEqual(driver.config()["study"], "A0-ablations-v2")

    def test_the_stage_plans(self):
        driver.use_study("v2")
        sweep = driver.v2_registrations("sweep", PLAN)
        self.assertEqual(len(sweep), 12)
        self.assertEqual({r.table["lr"] for r in sweep}, {"0.002", "0.005", "0.0125"})
        self.assertTrue(all(r.seeds == [17] for r in sweep))
        with patch.object(driver, "STUDY_DIR", ROOT / driver.STUDY.directory):
            evaluation = driver.v2_registrations("eval", PLAN)
        self.assertEqual([r.exp_id for r in evaluation if r.kind == "eval"],
                         ["M001-a0v2-eval", "M003-a0v2-eval", "M002-a0v2-eval", "M004-a0v2-eval"])
        rerun = [r for r in evaluation if r.kind == "rerun"]
        self.assertEqual((len(rerun), rerun[0].seeds, rerun[0].table["arms"]), (1, [17], "full"))
        for reg in evaluation:
            self.assertEqual(reg.table["lr_file"], "research/falsification/A0-ablations-v2/lr_selection.tsv")

    def test_a_skipped_sweep_is_refused(self):
        driver.use_study("v2")
        with self.assertRaises(SystemExit):
            driver.v2_registrations("sweep", {**PLAN, "sweep": False})

    def test_contingency_registers_the_paired_full_comparator_outside_m001(self):
        driver.use_study("v2")
        masked = driver.v2_registrations("contingency", PLAN, arm="no-semantic-slots")
        self.assertEqual([(r.exp_id, r.table["arms"]) for r in masked],
                         [("M001-a0v2-cont-no-semantic-slots", "full,no-semantic-slots")])
        other = driver.v2_registrations("contingency", PLAN, arm="raw-blind")
        self.assertEqual([(r.exp_id, r.table["arms"]) for r in other],
                         [("M002-a0v2-cont-raw-blind", "raw-blind"), ("M001-a0v2-cont-full", "full")])
        self.assertTrue(all(r.table["steps"] == 4000 and len(r.seeds) == 5 for r in other))

    def test_phases_that_exist_only_for_v2_refuse_v1(self):
        with self.assertRaises(SystemExit):
            driver.main(["g6"])

    def test_a_failed_run_spends_its_seed_and_stops_the_study(self):
        driver.use_study("v2")
        calls = []
        done = {"count": 0}

        def ran(exp_id, seed):
            return done["count"] > 0

        def run(command, **kwargs):
            done["count"] += 1
            calls.append(command)
            return type("R", (), {"returncode": 1, "stderr": "boom", "stdout": ""})()
        with patch.object(driver, "seed_ran", ran), patch.object(driver, "worktree_clean", lambda: True), \
                patch.object(driver, "ignored_files", lambda: []), patch.object(driver, "run", run), \
                patch.object(driver, "commit_record") as commit:
            with self.assertRaises(SystemExit) as stop:
                driver.serial([("M001-a0v2-eval", 17), ("M001-a0v2-eval", 29)], commit=True)
        self.assertIn("spent", str(stop.exception))
        self.assertEqual(len(calls), 1)
        commit.assert_called_once()

    def test_a_dirty_tree_or_an_ignored_file_blocks_a_launch(self):
        driver.use_study("v2")
        with patch.object(driver, "seed_ran", lambda *a: False), patch.object(driver, "worktree_clean", lambda: False):
            with self.assertRaises(SystemExit):
                driver.serial([("M001-a0v2-eval", 17)], commit=False)
        with patch.object(driver, "seed_ran", lambda *a: False), patch.object(driver, "worktree_clean", lambda: True), \
                patch.object(driver, "ignored_files", lambda: ["model/burn-a0/target-a0-study/"]):
            with self.assertRaises(SystemExit):
                driver.serial([("M001-a0v2-eval", 17)], commit=False)

    def test_g6_checks_for_v2_build_outside_the_repository(self):
        driver.use_study("v2")
        commands = []
        with tempfile.TemporaryDirectory() as logs, \
                patch.object(driver, "run", lambda command, **k: commands.append((command, k.get("env"))) or
                             type("R", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()), \
                patch.object(driver, "must", lambda command, **k: commands.append((command, k.get("env")))), \
                patch.object(driver, "worktree_clean", lambda: True):
            outcome = driver.correctness(Path(logs))
        self.assertTrue(all(v == "pass" for v in outcome.values()), outcome)
        build = commands[0][0]
        self.assertNotIn(driver.TARGET_DIR, build)
        self.assertTrue(Path(build[build.index("--target-dir") + 1]).is_absolute())
        self.assertEqual(commands[0][1]["CARGO_INCREMENTAL"], "0")
        names = [c[0] for c in commands[1:]]
        self.assertTrue(names[1][0].endswith("/release/examples/a0_ablation"))
        self.assertIn("scripts/tests/test_a0_study.py", names[3])


class AggregatorSelectsTheStudy(unittest.TestCase):
    def setUp(self):
        self.addCleanup(aggregator.use_study, "v1")

    def test_v1_constants_are_unchanged(self):
        self.assertEqual(aggregator.PREREG_TAG, "a0-ablation-prereg-v1")
        self.assertEqual(aggregator.PREREG_COMMIT, "45b2d5491e40171dd242d0bac6cf7d78d9c70746")

    def test_v2_is_identified_by_its_tag_alone(self):
        aggregator.use_study("v2")
        self.assertEqual((aggregator.PREREG_TAG, aggregator.PREREG_COMMIT), ("a0-ablations-v2-prereg", None))
        self.assertEqual(aggregator.CRITERIA, ROOT / "research/falsification/A0-ablations-v2/criteria.toml")
        with patch.object(aggregator, "git", side_effect=__import__("subprocess").CalledProcessError(1, "git")):
            with self.assertRaises(ValueError) as error:
                aggregator.preregistration_ref()
        self.assertIn("a0-ablations-v2-prereg", str(error.exception))

    def test_g4_allows_only_the_selection_records_and_later_stage_registration(self):
        aggregator.use_study("v2")
        d = V2.directory
        allowed = [f"{d}/lr_selection.tsv", f"{d}/lr_selection.json",
                   "experiments/model/M001-a0v2-sweep-lr0p002/results/run-1.json",
                   "experiments/model/M001-a0v2-eval/experiment.toml",
                   "experiments/model/M001-a0v2-rerun/config.toml",
                   "experiments/model/M002-a0v2-cont-raw-blind/README.md",
                   "experiments/registry.toml", "experiments/preregistration.toml"]
        self.assertEqual(aggregator.v2_freeze_violations(allowed), [])
        stray = ["scripts/run_experiment.py", "experiments/model/M001-a0v2-sweep-lr0p002/experiment.toml",
                 f"{d}/criteria.toml", "model/burn-a0/examples/a0_ablation/train.rs",
                 "experiments/model/M001-semantic-slots/results/run-2.json"]
        self.assertEqual(aggregator.v2_freeze_violations(allowed + stray), stray)

    def test_code_changes_ignore_records_and_logs_only(self):
        aggregator.use_study("v2")
        names = ["experiments/model/M001-a0v2-eval/results/run-1.json", f"{V2.directory}/logs/g6.json",
                 "scripts/aggregate_a0_ablation.py"]
        with patch.object(aggregator, "git", return_value="\n".join(names)):
            self.assertEqual(aggregator.code_changes("a", "b"), ["scripts/aggregate_a0_ablation.py"])

    def test_records_must_be_clean_with_an_empty_diff_but_may_name_different_commits(self):
        clean = {"git_dirty": False, "schema_version": 2, "git_tracked_diff_sha256": hashlib.sha256(b"").hexdigest()}
        records = {"a": {**clean, "git_sha": "1"}, "b": {**clean, "git_sha": "2"}}
        self.assertTrue(aggregator.v2_provenance(records)["pass"])
        records["b"]["git_dirty"] = True
        self.assertEqual(aggregator.v2_provenance(records)["dirty"], ["b"])
        self.assertFalse(aggregator.v2_provenance({})["pass"])

    def test_the_evaluation_g6_binding_accepts_a_scratch_binary_only_for_v2(self):
        aggregator.use_study("v2")
        fnv = "0ad71688f09b0d0d"
        tails = a0_study.python_check_tails(V2)
        checks = {
            "cargo_test": {"argv": ["/usr/bin/cargo", "+1.95.0", "test", "--locked", "--manifest-path", "model/burn-a0/Cargo.toml"]},
            "self_test": {"argv": ["/tmp/a0-g6-x/bin/release/examples/a0_ablation", "--phase", "self-test", "--data",
                                   "datasets/generated/operator_routing_v1", "--data-fnv64", fnv]},
            **{name: {"argv": ["/usr/bin/python3", *tail]} for name, tail in tails.items()},
        }
        with tempfile.TemporaryDirectory() as directory:
            logs = Path(directory)
            for name, check in checks.items():
                (logs / f"g6-{name}.log").write_text("ok")
                check["exit_code"] = 0
                check["log_sha256"] = hashlib.sha256(b"ok").hexdigest()
            evidence = {"schema_version": 1, "git_sha": "c" * 40, "git_sha_after": "c" * 40,
                        "git_dirty": False, "git_dirty_after": False, "checks": checks}
            self.assertTrue(aggregator.correctness_evidence(evidence, "c" * 40, logs, fnv)["pass"])
            aggregator.use_study("v1")
            result = aggregator.correctness_evidence(evidence, "c" * 40, logs, fnv)
            self.assertFalse(result["pass"])

    def test_the_cross_check_compares_printed_final_rows_with_the_scored_table(self):
        row = {"row": "final", "arm": "full", "split": "test_iid", "accuracy": 0.5}
        record = {"seed": 17, "stdout": json.dumps(row) + "\nPRED full test_iid 00\n"}
        chosen = {"eval": {"M001": {17: record}}}
        table = {"full": {17: {"test_iid": {"accuracy": 0.5}}}}
        self.assertEqual(aggregator.v2_stock_cross_check(chosen, {"eval": table}, [17]), [])
        table["full"][17]["test_iid"]["accuracy"] = 0.6
        self.assertEqual(aggregator.v2_stock_cross_check(chosen, {"eval": table}, [17]), ["eval/M001/full/17/test_iid"])
        self.assertEqual(aggregator.v2_stock_cross_check(chosen, {"eval": None}, [17]), ["eval/M001/full/17/test_iid"])


class RunnerLaunchRule(unittest.TestCase):
    """The v1 refusal stays; a v2 experiment is not caught by it."""

    class Reached(Exception):
        pass

    def launch(self, exp_id, entrypoint):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), patch.object(run_experiment, "resolve", side_effect=self.Reached):
            try:
                code = run_experiment.run_experiment(exp_id, entrypoint=entrypoint, seed=17, params={})
            except self.Reached:
                return "reached", err.getvalue()
        return code, err.getvalue()

    def test_v1_entrypoints_stay_refused(self):
        for exp_id in ("M001", "M002", "M003", "M004"):
            for entrypoint in ("a0_ablation_entrypoint", "a0_sweep_entrypoint", "a0_rerun_entrypoint"):
                code, message = self.launch(exp_id, entrypoint)
                self.assertEqual(code, 2, (exp_id, entrypoint))
                self.assertIn("A0 v1 is archival", message)

    def test_v2_experiments_are_not_refused_by_that_rule(self):
        for exp_id in ("M001-a0v2-eval", "M004-a0v2-sweep-lr0p005", "M001-a0v2-rerun", "M002-a0v2-cont-raw-blind"):
            code, message = self.launch(exp_id, "entrypoint")
            self.assertEqual(code, "reached", (exp_id, message))

    def test_v2_experiments_are_registered_listed_and_prepared_once_frozen(self):
        registered = tomllib.loads((ROOT / "experiments/registry.toml").read_text(encoding="utf-8"))["experiment"]
        ids = {e["id"] for e in registered}
        listed = tomllib.loads((ROOT / "experiments/preregistration.toml").read_text(encoding="utf-8"))["experiment"]
        for exp_id in sorted(i for i in ids if V2.classify(i)):
            self.assertIn(exp_id, listed)
            self.assertEqual(next(e for e in registered if e["id"] == exp_id)["status"], "prepared")


if __name__ == "__main__":
    unittest.main()
