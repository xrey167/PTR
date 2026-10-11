"""Tests of the A0 baseline-v2 decider (scripts/aggregate_a0_baseline_v2.py).

Two parts.

1. Every test of the v1 decider (scripts/tests/test_aggregate_a0_baseline.py) is
   re-run against v2: the v1 test source is loaded with the study ids, paths and
   module name of v1 replaced by those of v2, so the v2 decider is held to every
   statistic, gate, decision rule and refusal of v1 (and a test added to v1 is
   re-run here without being copied).
2. The two preregistered changes, exercised on temporary git repositories: the
   record-identity rule (records at different commits of one line of history whose
   diff touches only results/ files of the study) and the INCONCLUSIVE-INTERRUPTED
   outcome.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import tomllib
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V1_TESTS = ROOT / "scripts/tests/test_aggregate_a0_baseline.py"
SUBSTITUTIONS = (
    ("M001-v5", "M001-v6"),
    ("M002-v10", "M002-v11"),
    ("M003-v2", "M003-v3"),
    ("A0-baseline-v1", "A0-baseline-v2"),
    ("aggregate_a0_baseline.py", "aggregate_a0_baseline_v2.py"),
    ('"aggregate_a0_baseline"', '"aggregate_a0_baseline_v2"'),
)


def v1_cases_for_v2() -> types.ModuleType:
    source = V1_TESTS.read_text(encoding="utf-8")
    for old, new in SUBSTITUTIONS:
        source = source.replace(old, new)
    module = types.ModuleType(f"{__name__}.v1_cases")
    module.__file__ = str(V1_TESTS)
    sys.modules[module.__name__] = module
    exec(compile(source, str(V1_TESTS), "exec"), module.__dict__)
    return module


V1_CASES = v1_cases_for_v2()
mod = V1_CASES.mod
Scenario = V1_CASES.Scenario
supports_m002 = V1_CASES.supports_m002
CRITERIA = V1_CASES.CRITERIA
SEEDS = V1_CASES.SEEDS
contrast = V1_CASES.contrast
# The v1 test classes, run against v2 (unittest finds them as members of this module).
for _name, _value in list(vars(V1_CASES).items()):
    if isinstance(_value, type) and issubclass(_value, unittest.TestCase):
        globals()[_name] = _value

STUDY = "M002-v11"
DIRECTORY = "experiments/model/M002-v11-typed-attention/results"
OTHER_STUDY_DIRECTORY = "experiments/model/M001-v6-semantic-slots/results"


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false", *args],
        cwd=repo, check=True, capture_output=True, text=True,
    )
    return done.stdout.strip()


class Repo:
    """A temporary git repository with one code file and the study's results
    directories, the way the run loop leaves them."""

    def __init__(self, test: unittest.TestCase):
        self.path = Path(test.enterContext(tempfile.TemporaryDirectory(prefix="a0-v2-repo-"))) / "repo"
        self.path.mkdir()
        git(self.path, "init", "-q", "-b", "main")
        self.write("scripts/tool.py", "print('tool')\n")
        self.write(f"{DIRECTORY}/.gitkeep", "")
        self.write("experiments/model/M002-v11-typed-attention/config.toml", "a = 1\n")
        self.commit("freeze")

    def write(self, relative: str, text: str) -> None:
        target = self.path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    def commit(self, message: str) -> str:
        git(self.path, "add", "-A")
        git(self.path, "commit", "-q", "-m", message, "--allow-empty")
        return self.head()

    def head(self) -> str:
        return git(self.path, "rev-parse", "HEAD")


def stamp(index: int) -> str:
    return f"20261009T0{index}0000.000000Z"


def run_loop(repo: Repo, scenario: Scenario, between=None, directory: str = DIRECTORY) -> list[tuple[str, dict]]:
    """The run protocol: each seed runs at HEAD, its record is committed before the next run.
    `between(index)` may change the repository before run `index` starts."""
    sources = []
    for index, seed in enumerate(SEEDS):
        if between is not None:
            between(index)
        record = scenario.record(seed, git_sha=repo.head(), started_at=stamp(index))
        name = f"run-{stamp(index)}-seed-{seed}.json"
        repo.write(f"{directory}/{name}", json.dumps(record))
        repo.commit(f"record seed {seed}")
        sources.append((name, record))
    return sources


class RecordIdentity(unittest.TestCase):
    def setUp(self):
        self.repo = Repo(self)
        self.scenario = supports_m002()

    def decide(self, sources, repo=None):
        return mod.decide(sources, STUDY, CRITERIA, repo=self.repo.path if repo is None else repo)

    def refusal(self, sources, text, repo=None):
        with self.assertRaises(mod.EvidenceError) as caught:
            self.decide(sources, repo)
        self.assertIn("disagree in git_sha", str(caught.exception))
        self.assertIn(text, str(caught.exception))

    def test_the_v1_decider_refuses_the_records_of_a_correct_run_loop(self):
        # The defect that replaced study v1: five records, five commits.
        sources = run_loop(self.repo, self.scenario)
        self.assertEqual(len({record["git_sha"] for _, record in sources}), 5)
        with self.assertRaisesRegex(mod.EvidenceError, "run records disagree in git_sha"):
            mod.V1.decide(sources, STUDY, mod.V1.load_criteria(mod.CRITERIA_PATH))

    def test_per_seed_commits_with_only_results_added_are_accepted(self):
        sources = run_loop(self.repo, self.scenario)
        result = self.decide(sources)
        self.assertEqual(result["headline"], {"contrast": "M002-v11-primary", "verdict": "SUPPORTS", "label": "SUPPORTS"})
        evidence = result["provenance"]
        self.assertEqual([c["git_sha"] for c in evidence["record_commits"]], [r["git_sha"] for _, r in sources])
        self.assertEqual(evidence["git_sha"], sources[0][1]["git_sha"])
        self.assertEqual(evidence["record_identity"], "single-commit-or-linear-history-results-only")
        self.assertEqual(evidence["decision_script_sha256"], hashlib.sha256((ROOT / "scripts/aggregate_a0_baseline_v2.py").read_bytes()).hexdigest())
        self.assertEqual(evidence["decision_base_sha256"], hashlib.sha256((ROOT / "scripts/aggregate_a0_baseline.py").read_bytes()).hexdigest())

    def test_the_order_of_the_sources_does_not_matter_only_started_at_does(self):
        sources = run_loop(self.repo, self.scenario)
        self.assertEqual(
            json.dumps(self.decide(list(reversed(sources)))["contrasts"], sort_keys=True),
            json.dumps(self.decide(sources)["contrasts"], sort_keys=True),
        )

    def test_results_of_the_studys_other_experiments_between_records_are_accepted(self):
        def between(index):
            if index == 2:
                self.repo.write(f"{OTHER_STUDY_DIRECTORY}/run-x.json", "{}")
                self.repo.write("experiments/model/M002-v11-sweep/results/run-y.json", "{}")
                self.repo.write(f"{DIRECTORY}/STALE.toml", "reason = 'x'\n")
                self.repo.commit("other results of the study")

        self.assertEqual(self.decide(run_loop(self.repo, self.scenario, between))["headline"]["verdict"], "SUPPORTS")

    def test_a_code_change_between_records_is_refused(self):
        def between(index):
            if index == 3:
                self.repo.write("scripts/tool.py", "print('changed')\n")
                self.repo.commit("code change")

        self.refusal(run_loop(self.repo, self.scenario, between), "scripts/tool.py")

    def test_anything_but_results_files_of_the_study_is_refused(self):
        cases = {
            "experiment config": ("experiments/model/M002-v11-typed-attention/config.toml", "a = 2\n"),
            "another study's results": ("experiments/model/M002-v10-typed-attention/results/x.json", "{}"),
            "a file below results": (f"{DIRECTORY}/sub/x.json", "{}"),
            "criteria": ("research/falsification/A0-baseline-v2/criteria.toml", "x\n"),
            "an experiment prefix only": ("experiments/model/M002-v11/results/x.json", "{}"),
        }
        for name, (path, text) in cases.items():
            with self.subTest(name=name):
                repo = Repo(self)

                def between(index, repo=repo, path=path, text=text):
                    if index == 2:
                        repo.write(path, text)
                        repo.commit("change")

                sources = run_loop(repo, self.scenario, between)
                self.refusal(sources, path, repo=repo.path)

    def test_a_non_linear_history_is_refused(self):
        git(self.repo.path, "checkout", "-q", "-b", "side")
        self.repo.write(f"{DIRECTORY}/side.json", "{}")
        side = self.repo.commit("side")
        git(self.repo.path, "checkout", "-q", "main")
        self.repo.write(f"{DIRECTORY}/main.json", "{}")
        main = self.repo.commit("main")
        git(self.repo.path, "merge", "-q", "--no-edit", "side")
        merged = self.repo.head()
        self.assertNotIn(merged, (side, main))
        commits = [main, side, main, side, merged]
        sources = [
            (f"s{seed}.json", self.scenario.record(seed, git_sha=sha, started_at=stamp(i)))
            for i, (seed, sha) in enumerate(zip(SEEDS, commits))
        ]
        self.refusal(sources, "not an ancestor")
        self.refusal(sources, "not on one line of history")

    def test_run_order_against_history_is_refused(self):
        sources = run_loop(self.repo, self.scenario)
        backwards = [(name, {**record, "started_at": stamp(4 - i)}) for i, (name, record) in enumerate(sources)]
        self.refusal(backwards, "not an ancestor")

    def test_commits_outside_head_history_or_unknown_are_refused(self):
        sources = run_loop(self.repo, self.scenario)
        git(self.repo.path, "checkout", "-q", "-b", "abandoned", sources[1][1]["git_sha"])
        self.repo.write(f"{DIRECTORY}/abandoned.json", "{}")
        abandoned = self.repo.commit("abandoned")
        git(self.repo.path, "checkout", "-q", "main")
        for sha, text in ((abandoned, "not in HEAD's history"), ("1" * 40, "does not hold")):
            with self.subTest(sha=sha[:8]):
                changed = [*sources[:4], (sources[4][0], {**sources[4][1], "git_sha": sha})]
                self.refusal(changed, text)

    def test_different_commits_need_a_repository_a_run_order_and_real_commits(self):
        sources = run_loop(self.repo, self.scenario)
        with self.assertRaisesRegex(mod.EvidenceError, "without a repository"):
            mod.decide(sources, STUDY, CRITERIA, repo=None)
        untimed = [(name, {k: v for k, v in record.items() if k != "started_at"}) for name, record in sources]
        self.refusal(untimed, "no started_at")
        twice = [(name, {**record, "started_at": stamp(0)}) for name, record in sources]
        self.refusal(twice, "share a started_at")
        malformed = [*sources[:4], (sources[4][0], {**sources[4][1], "git_sha": "abc"})]
        self.refusal(malformed, "names no commit")

    def test_every_other_identity_field_must_still_be_identical(self):
        fields = {
            "manifest_sha256": "0" * 64, "parameters": {"d_model": "49"}, "rustc": "other", "toolchain": {"x": 2},
            "environment": {"E": "2"}, "executable": {"sha256": "1" * 64}, "host": {"system": "Other"},
            "cargo_lock_sha256": "2" * 64, "uv_lock_sha256": "3" * 64,
        }
        for field, value in fields.items():
            with self.subTest(field=field):
                repo = Repo(self)
                sources = run_loop(repo, self.scenario)
                sources[2] = (sources[2][0], {**sources[2][1], field: value})
                with self.assertRaisesRegex(mod.EvidenceError, f"run records disagree in {field}"):
                    mod.decide(sources, STUDY, CRITERIA, repo=repo.path)
        for digest in ("preregistration_sha256", "preregistration_rules_sha256"):
            with self.subTest(digest=digest):
                repo = Repo(self)
                sources = run_loop(repo, self.scenario)
                manifest = {**sources[1][1]["manifest"], digest: "7" * 64}
                sources[1] = (sources[1][0], {**sources[1][1], "manifest": manifest})
                with self.assertRaisesRegex(mod.EvidenceError, f"run records disagree in manifest.{digest}"):
                    mod.decide(sources, STUDY, CRITERIA, repo=repo.path)

    def test_identical_commits_need_no_repository(self):
        decided = mod.decide(self.scenario.sources(), STUDY, CRITERIA, repo=None)
        self.assertEqual(decided["headline"]["verdict"], "SUPPORTS")
        self.assertEqual(decided["provenance"]["git_sha"], "a" * 40)

    def test_the_cli_takes_the_repository_and_publishes_the_aggregate_at_the_earliest_commit(self):
        sources = run_loop(self.repo, self.scenario)
        results = self.repo.path / DIRECTORY
        paths = [str(results / name) for name, _ in sources]
        decision, ci = self.repo.path / "D.toml", self.repo.path / "ci.json"
        args = ["--study", STUDY, "--repo", str(self.repo.path), "--paired-ci", str(ci), "--decision", str(decision)]
        with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(mod.main([*args, "--publish-results", str(results), *paths]), 0)
        self.assertEqual(json.loads(out.getvalue())["headline"]["verdict"], "SUPPORTS")
        parsed = tomllib.loads(decision.read_text(encoding="utf-8"))
        self.assertEqual(parsed["record_identity"]["commits"], [record["git_sha"] for _, record in sources])
        self.assertEqual(parsed["git_sha"], sources[0][1]["git_sha"])
        self.assertEqual(parsed["decision_script_sha256"], hashlib.sha256((ROOT / "scripts/aggregate_a0_baseline_v2.py").read_bytes()).hexdigest())
        run = json.loads((results / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(run["git_sha"], sources[0][1]["git_sha"])
        # A wrong repository makes the commits unknown: a refusal, exit code 2, nothing written.
        other = Repo(self)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(
                mod.main(["--study", STUDY, "--repo", str(other.path), "--paired-ci", str(self.repo.path / "c2.json"),
                          "--decision", str(self.repo.path / "D2.toml"), *paths]), 2)
        self.assertIn("does not hold", err.getvalue())
        self.assertFalse((self.repo.path / "D2.toml").exists())


class Interruption(unittest.TestCase):
    def setUp(self):
        self.scenario = supports_m002()

    def reservation(self, seed: int = 101, **overrides) -> dict:
        """What the runner leaves when it is killed: no stdout, no exit code, no finish."""
        record = {k: v for k, v in self.scenario.record(seed).items() if k not in ("stdout", "exit_code")}
        record.update(status="started", **overrides)
        return record

    def sources_with(self, seed: int = 101) -> list[tuple[str, dict]]:
        sources = self.scenario.sources()
        return [(name, self.reservation(seed) if record["seed"] == seed else record) for name, record in sources]

    def test_an_interrupted_seed_is_a_named_outcome_not_a_refusal(self):
        result = mod.decide(self.sources_with(), STUDY, CRITERIA)
        self.assertEqual(result["headline"], {"contrast": "M002-v11-primary", "verdict": "INCONCLUSIVE", "label": "INCONCLUSIVE-INTERRUPTED"})
        self.assertEqual(result["outcome"], "INCONCLUSIVE-INTERRUPTED")
        self.assertEqual(result["interrupted"], {"seeds_interrupted": [101], "seeds_completed": [17, 29, 43, 71], "seeds_not_run": []})
        self.assertEqual([(c["id"], c["label"], c["quality_claim_allowed"]) for c in result["contrasts"]],
                         [("M002-v11-primary", "INCONCLUSIVE-INTERRUPTED", False), ("M002-v11-bundle", "INCONCLUSIVE-INTERRUPTED", False)])
        for item in result["contrasts"]:
            self.assertEqual((item["endpoints"], item["gates"], item["failing_gates"]), ([], {}, []))
        self.assertEqual(result["arms"], {})

    def test_no_statistic_is_computed_the_completed_seeds_rows_are_not_read(self):
        sources = [(name, {**record, "stdout": "not rows at all"} if record["status"] == "completed" else record)
                   for name, record in self.sources_with()]
        self.assertEqual(mod.decide(sources, STUDY, CRITERIA)["headline"]["label"], "INCONCLUSIVE-INTERRUPTED")

    def test_which_seed_and_how_many_records_do_not_matter(self):
        for seed in (17, 43):
            with self.subTest(seed=seed):
                self.assertEqual(mod.decide(self.sources_with(seed), STUDY, CRITERIA)["interrupted"]["seeds_interrupted"], [seed])
        # The loop stopped at the interruption: only records up to it exist.
        sources = self.sources_with(43)[:3]
        result = mod.decide(sources, STUDY, CRITERIA)
        self.assertEqual(result["interrupted"], {"seeds_interrupted": [43], "seeds_completed": [17, 29], "seeds_not_run": [71, 101]})
        only = mod.decide([("only.json", self.reservation(17))], STUDY, CRITERIA)
        self.assertEqual(only["interrupted"]["seeds_not_run"], [29, 43, 71, 101])
        both = [("a.json", self.reservation(29)), ("b.json", self.reservation(101))]
        self.assertEqual(mod.decide(both, STUDY, CRITERIA)["interrupted"]["seeds_interrupted"], [29, 101])

    def test_every_study_has_the_outcome_and_the_label_forbids_claims(self):
        for study_id in CRITERIA["studies"]:
            with self.subTest(study=study_id):
                scenario = Scenario(study_id)
                record = {k: v for k, v in scenario.record(101).items() if k not in ("stdout", "exit_code")}
                record["status"] = "started"
                result = mod.decide([("r.json", record)], study_id, CRITERIA)
                self.assertEqual(result["headline"]["label"], "INCONCLUSIVE-INTERRUPTED")
                self.assertEqual(len(result["contrasts"]), len(CRITERIA["studies"][study_id]["contrast"]))
                self.assertFalse(any(item["quality_claim_allowed"] for item in result["contrasts"]))

    def test_what_is_not_an_interruption_stays_a_refusal(self):
        started_with_output = {**self.reservation(), "stdout": "x\n"}
        started_with_code = {**self.reservation(), "exit_code": 0}
        failed = {**self.scenario.record(29), "status": "failed", "exit_code": 1}
        killed_by_signal = {**self.reservation(), "status": "failed", "exit_code": -9}
        for name, record in (("output", started_with_output), ("exit code", started_with_code), ("failed", failed), ("signal", killed_by_signal)):
            with self.subTest(name=name), self.assertRaises(mod.EvidenceError):
                mod.decide([*self.scenario.sources()[:4], ("x.json", record)], STUDY, CRITERIA)
        # An interrupted study still admits only this study's hardened records.
        for override, text in (
            ({"experiment_id": "M002-v10"}, "experiment_id"), ({"entrypoint": "other"}, "frozen entrypoint"),
            ({"manifest": {"id": "M002-v10"}}, "frozen entrypoint"), ({"seed": 7}, "not confirmatory"),
        ):
            with self.subTest(override=override), self.assertRaisesRegex(mod.EvidenceError, text):
                mod.decide([("x.json", self.reservation(**override))], STUDY, CRITERIA)
        with self.assertRaisesRegex(mod.EvidenceError, "duplicate"):
            mod.decide([("a.json", self.reservation(101)), ("b.json", self.reservation(101))], STUDY, CRITERIA)
        with self.assertRaisesRegex(mod.EvidenceError, "did not complete"):
            mod.decide([("a.json", self.reservation(101)), ("b.json", failed)], STUDY, CRITERIA)
        with self.assertRaisesRegex(mod.EvidenceError, "hardened run records only"):
            mod.decide([("a.json", self.reservation(101)), ("rows.jsonl", {"rows": []})], STUDY, CRITERIA, allow_rows=True)
        with self.assertRaisesRegex(mod.EvidenceError, "between 1 and 5"):
            mod.decide([*self.sources_with(), ("sixth.json", self.reservation(7))], STUDY, CRITERIA)
        with self.assertRaisesRegex(mod.EvidenceError, "deferred"):
            mod.decide(self.sources_with(), "M004-v2", CRITERIA)

    def test_outputs_record_the_outcome_and_publish_nothing(self):
        result = mod.decide(self.sources_with(), STUDY, CRITERIA)
        with tempfile.TemporaryDirectory() as directory:
            ci, decision = Path(directory) / "results/paired-ci-95.json", Path(directory) / "DECISION.toml"
            digest = mod.write_outputs(result, ci, decision)
            parsed = tomllib.loads(decision.read_text(encoding="utf-8"))
            self.assertEqual((parsed["verdict"], parsed["label"], parsed["go"]), ("INCONCLUSIVE", "INCONCLUSIVE-INTERRUPTED", False))
            self.assertEqual(parsed["paired_ci_sha256"], digest)
            self.assertEqual(parsed["reporting"]["required_label"], "INCONCLUSIVE-INTERRUPTED")
            self.assertFalse(parsed["reporting"]["allow_quality_claim"])
            self.assertFalse(parsed["reporting"]["rerun_allowed"])
            self.assertEqual(parsed["interrupted"]["seeds_interrupted"], [101])
            self.assertEqual(len(parsed["source_record"]), 5)
            self.assertEqual([c["label"] for c in parsed["contrast"]], ["INCONCLUSIVE-INTERRUPTED"] * 2)
            self.assertIn("record_identity", parsed)
            document = json.loads(ci.read_text(encoding="utf-8"))
            self.assertEqual(document["status"], "not-computed-interrupted")
            self.assertEqual([c["endpoints"] for c in document["contrasts"]], [[], []])
            with self.assertRaises(mod.EvidenceError):
                mod.write_outputs(result, ci, decision)
            with self.assertRaisesRegex(mod.EvidenceError, "interrupted study publishes no aggregate"):
                mod.publish_results(result, Path(directory))
            self.assertFalse((Path(directory) / "run.json").exists())

    def test_the_cli_reads_a_reservation_record_and_exits_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for name, record in self.sources_with():
                path = root / name
                path.write_text(json.dumps(record), encoding="utf-8")
                paths.append(str(path))
            args = ["--study", STUDY, "--paired-ci", str(root / "ci.json"), "--decision", str(root / "D.toml")]
            with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(mod.main([*args, "--publish-results", str(root), *paths]), 2)
                self.assertIn("nothing was written", err.getvalue())
                self.assertFalse((root / "D.toml").exists())
                self.assertEqual(mod.main([*args, *paths]), 0)
            summary = json.loads(out.getvalue())
            self.assertEqual(summary["headline"]["label"], "INCONCLUSIVE-INTERRUPTED")
            self.assertEqual(summary["interrupted"]["seeds_interrupted"], [101])
            self.assertTrue((root / "D.toml").exists())


class V2Criteria(unittest.TestCase):
    def test_the_criteria_are_the_v1_criteria_except_ids_name_and_the_v2_rules(self):
        v1 = (ROOT / "research/falsification/A0-baseline-v1/criteria.toml").read_text(encoding="utf-8")
        for old, new in SUBSTITUTIONS[:3]:
            v1 = v1.replace(old, new)
        old, new = tomllib.loads(v1), CRITERIA["raw"]
        for table, key in (("decision", "base"), ("decision", "base_sha256"), ("decision_rules", "interrupted_label")):
            self.assertIn(key, new[table])
            new = {**new, table: {k: v for k, v in new[table].items() if k != key}}
        new = {k: v for k, v in new.items() if k != "record_identity"}
        self.assertEqual((old.pop("name"), new.pop("name")), ("A0-baseline-v1", "A0-baseline-v2"))
        self.assertEqual(new, old)

    def test_the_v1_decider_is_pinned_unchanged(self):
        self.assertEqual(
            CRITERIA["raw"]["decision"]["base_sha256"],
            hashlib.sha256((ROOT / "scripts/aggregate_a0_baseline.py").read_bytes()).hexdigest(),
        )

    def test_a_tampered_v2_rule_is_refused(self):
        text = (ROOT / "research/falsification/A0-baseline-v2/criteria.toml").read_text(encoding="utf-8")
        cases = {
            "base pin": text.replace(CRITERIA["raw"]["decision"]["base_sha256"], "0" * 64),
            "label": text.replace("INCONCLUSIVE-INTERRUPTED", "INCONCLUSIVE"),
            "mode": text.replace("single-commit-or-linear-history-results-only", "any"),
            "fields": text.replace('"uv_lock_sha256", ', ""),
            "name": text.replace('name = "A0-baseline-v2"', 'name = "A0-baseline-v1"'),
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, changed in cases.items():
                self.assertNotEqual(changed, text, name)
                path = Path(directory) / f"{name.replace(' ', '_')}.toml"
                path.write_text(changed, encoding="utf-8")
                with self.assertRaises(mod.EvidenceError, msg=name):
                    mod.load_criteria(path)


if __name__ == "__main__":
    unittest.main()
