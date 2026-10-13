import importlib.util
import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = load("check_component_protocols", "scripts/check_component_protocols.py")
runner = load("run_component_eval", "scripts/run_component_eval.py")

CRITERIA = """[criteria]
correctness = 0.3
failure_semantics = 0.15
latency = 0.12
throughput = 0.1
memory = 0.08
portability = 0.07
operability = 0.06
integration_cost = 0.05
license_fit = 0.04
research_flexibility = 0.03
"""

PROTOCOL = """version = 1
component = "alloc"
protocol_id = "alloc-v1"
hardware_profile = "hardware/h.toml"
bench_workspace = "evaluations/bench/b"
lockfile = "evaluations/bench/b/Cargo.lock"
[repetitions]
warmup = 1
measured = 5
[workload]
description = "w"
seeds = [17, 29]
samples = 10000
[[gate]]
id = "digest"
criterion = "correctness"
description = "d"
[[gate]]
id = "no-growth"
criterion = "failure_semantics"
description = "d"
[[metric]]
id = "p99_ns"
criterion = "latency"
unit = "ns"
direction = "lower"
[[metric]]
id = "ops"
criterion = "throughput"
unit = "ops/s"
direction = "higher"
[scoring]
weights_from = "config.toml"
excluded = ["research_flexibility"]
method = "ratio-to-best"
[scoring.proxies.license_fit]
source = "cargo-deny"
[scoring.proxies.integration_cost]
parts = ["adapter_loc", "transitive_crates", "build_seconds"]
[scoring.proxies.portability]
targets = ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"]
[[candidate]]
id = "system"
kind = "incumbent"
version = "std"
[[candidate]]
id = "mimalloc"
kind = "crate"
version = "0.1.48"
lock_package = "mimalloc"
[[candidate]]
id = "jemalloc"
kind = "crate"
version = "0.6.0"
lock_package = "tikv-jemallocator"
"""

LOCK = """version = 4
[[package]]
name = "mimalloc"
version = "0.1.48"
[[package]]
name = "tikv-jemallocator"
version = "0.6.0"
"""


def candidates_toml(statuses: dict, notes: dict | None = None, evidence: dict | None = None) -> str:
    notes = notes or {}
    evidence = evidence or {}
    out = 'component = "alloc"\nstatus = "open"\ndecision = "unlocked"\n' + CRITERIA
    for cid in ("system", "mimalloc", "jemalloc"):
        out += f'\n[[candidate]]\nid = "{cid}"\nsource = "external"\nstatus = "{statuses[cid]}"\n'
        out += f"evidence = {json.dumps(evidence.get(cid, []))}\n"
        if cid in notes:
            out += f'note = "{notes[cid]}"\n'
    return out


def record(cid, protocol_hash, gates=None, ran=True, measured=5, grade=True, **over):
    bench = {
        "candidate_version": "x",
        "gates": gates if gates is not None else {"digest": True, "no-growth": True},
        "benchmark_ran": ran,
        "repetitions": {"warmup": 1, "measured": measured},
        "decision_grade": grade,
        "decision_note": "",
        "metrics": {},
    }
    base = {field: None for field in mod.RECORD_FIELDS}
    base.update({"schema_version": 1, "candidate": {"id": cid}, "protocol_sha256": protocol_hash,
                 "bench": bench})
    base.update(over)
    return base


class Tree:
    """A throwaway repository root with one component `alloc`."""

    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "hardware").mkdir()
        (self.root / "hardware/h.toml").write_text('id = "h"\ndecision_grade = true\n')
        (self.root / "evaluations/bench/b").mkdir(parents=True)
        (self.root / "evaluations/bench/b/Cargo.toml").write_text("[workspace]\n")
        (self.root / "evaluations/bench/b/Cargo.lock").write_text(LOCK)
        self.component = self.root / "evaluations/components/alloc"
        (self.component / "evidence").mkdir(parents=True)
        self.write_protocol(PROTOCOL)

    def write_protocol(self, text):
        (self.component / "protocol.toml").write_text(text)

    @property
    def hash(self):
        return mod.sha256_file(self.component / "protocol.toml")

    def candidates(self, statuses, notes=None, evidence=None):
        (self.component / "candidates.toml").write_text(candidates_toml(statuses, notes, evidence))

    def evidence(self, name, rec):
        (self.component / "evidence" / name).write_text(json.dumps(rec))
        return f"evidence/{name}"

    def errors(self):
        return mod.check_component(self.root, "alloc", use_git=False)

    def close(self):
        self.tmp.cleanup()


class ProtocolSchema(unittest.TestCase):
    def setUp(self):
        self.tree = Tree()
        self.addCleanup(self.tree.close)

    def validate(self, text=PROTOCOL):
        import tomllib

        return mod.validate_protocol(tomllib.loads(text), "alloc", self.tree.root)

    def test_a_good_protocol_has_no_errors(self):
        self.assertEqual(self.validate(), [])

    def test_the_shipped_template_validates_against_the_kit_lock(self):
        import tomllib

        template = tomllib.loads((ROOT / "evaluations/protocol.template.toml").read_text())
        self.assertEqual(mod.validate_protocol(template, "allocator", ROOT), [])

    def test_each_rule_has_a_failing_example(self):
        cases = {
            "version = 1": ("version = 2", "version must be 1"),
            'component = "alloc"': ('component = "other"', "component must be"),
            "warmup = 1": ("warmup = 0", "repetitions.warmup"),
            "measured = 5": ("measured = 4", "repetitions.measured"),
            "samples = 10000": ("samples = 9999", "workload.samples >= 10000"),
            'id = "no-growth"\ncriterion = "failure_semantics"': (
                'id = "digest"\ncriterion = "failure_semantics"', "duplicate gate id"),
            'criterion = "latency"': ('criterion = "research_flexibility"', "owner-scored"),
            'direction = "lower"': ('direction = "down"', "direction must be"),
            'excluded = ["research_flexibility"]': ("excluded = []", "scoring.excluded"),
            'method = "ratio-to-best"': ('method = "mean"', "scoring.method"),
            'source = "cargo-deny"': ('source = "none"', "license_fit.source"),
            'parts = ["adapter_loc", "transitive_crates", "build_seconds"]': (
                'parts = ["adapter_loc"]', "integration_cost.parts"),
            'targets = ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"]': (
                'targets = ["x86_64-unknown-linux-gnu"]', "portability.targets"),
            'version = "0.1.48"': ('version = "0.1.99"', "is not in"),
            'hardware_profile = "hardware/h.toml"': (
                'hardware_profile = "hardware/missing.toml"', "does not exist"),
            'weights_from = "config.toml"': ('weights_from = "x"', "weights_from"),
        }
        for old, (new, expect) in cases.items():
            with self.subTest(rule=expect):
                self.assertIn(old, PROTOCOL)
                errors = self.validate(PROTOCOL.replace(old, new, 1))
                self.assertTrue(any(expect in e for e in errors), (expect, errors))

    def test_a_protocol_compares_at_least_two_candidates_with_gates_and_metrics(self):
        head = PROTOCOL[: PROTOCOL.index("[[gate]]")]
        errors = self.validate(head)
        for expect in ("[[gate]]", "[[metric]]", "at least two candidates"):
            self.assertTrue(any(expect in e for e in errors), (expect, errors))

    def test_binary_candidates_pin_sha256_and_https_url(self):
        text = PROTOCOL.replace(
            'id = "system"\nkind = "incumbent"\nversion = "std"',
            'id = "system"\nkind = "binary"\nversion = "1"\nurl = "http://x"\nsha256 = "abc"',
        )
        errors = self.validate(text)
        self.assertTrue(any("64-hex sha256" in e for e in errors))
        self.assertTrue(any("https url" in e for e in errors))
        ok = PROTOCOL.replace(
            'id = "system"\nkind = "incumbent"\nversion = "std"',
            f'id = "system"\nkind = "binary"\nversion = "1"\nurl = "https://x"\nsha256 = "{"a" * 64}"',
        )
        self.assertEqual(self.validate(ok), [])


class Scoring(unittest.TestCase):
    WEIGHTS = {
        "correctness": 0.3, "failure_semantics": 0.15, "latency": 0.12, "throughput": 0.1,
        "memory": 0.08, "portability": 0.07, "operability": 0.06, "integration_cost": 0.05,
        "license_fit": 0.04, "research_flexibility": 0.03,
    }

    def protocol(self):
        import tomllib

        return tomllib.loads(PROTOCOL)

    def test_hand_computed_scores(self):
        measured = {
            "a": {"metrics": {"p99_ns": 100.0, "ops": 50.0},
                  "proxies": {"license_fit": True,
                              "integration_cost": {"adapter_loc": 100, "transitive_crates": 10, "build_seconds": 20},
                              "portability": {"x": True, "y": True}}},
            "b": {"metrics": {"p99_ns": 200.0, "ops": 100.0},
                  "proxies": {"license_fit": True,
                              "integration_cost": {"adapter_loc": 200, "transitive_crates": 10, "build_seconds": 10},
                              "portability": {"x": True, "y": False}}},
        }
        scores = mod.automated_score(self.WEIGHTS, self.protocol(), measured)
        a, b = scores["a"]["criteria"], scores["b"]["criteria"]
        self.assertAlmostEqual(a["latency"], 1.0)
        self.assertAlmostEqual(b["latency"], 0.5)  # 100/200 (lower is better)
        self.assertAlmostEqual(a["throughput"], 0.5)  # 50/100 (higher is better)
        self.assertAlmostEqual(b["throughput"], 1.0)
        self.assertAlmostEqual(a["integration_cost"], (1.0 + 1.0 + 0.5) / 3)
        self.assertAlmostEqual(b["integration_cost"], (0.5 + 1.0 + 1.0) / 3)
        self.assertAlmostEqual(a["portability"], 1.0)
        self.assertAlmostEqual(b["portability"], 0.5)
        # a: all weights of the scored criteria except operability, memory (no basis)
        scored = ["correctness", "failure_semantics", "latency", "throughput", "portability",
                  "integration_cost", "license_fit"]
        total = sum(self.WEIGHTS[c] for c in scored)
        expected_a = sum(self.WEIGHTS[c] * a[c] for c in scored) / total
        self.assertAlmostEqual(scores["a"]["score"], expected_a)
        self.assertEqual(scores["a"]["unscored"], ["memory", "operability", "research_flexibility"])
        self.assertGreater(scores["a"]["score"], scores["b"]["score"])

    def test_research_flexibility_never_enters_the_score(self):
        measured = {"a": {"metrics": {"p99_ns": 1.0}, "proxies": {}},
                    "b": {"metrics": {"p99_ns": 2.0}, "proxies": {}}}
        base = mod.automated_score(self.WEIGHTS, self.protocol(), measured)
        heavy = dict(self.WEIGHTS, research_flexibility=100.0)
        self.assertEqual(base, mod.automated_score(heavy, self.protocol(), measured))
        self.assertIn("research_flexibility", base["a"]["unscored"])
        self.assertNotIn("research_flexibility", base["a"]["criteria"])

    def test_a_dirty_cargo_deny_verdict_scores_zero_for_license_fit(self):
        measured = {"a": {"proxies": {"license_fit": False}}, "b": {"proxies": {"license_fit": True}}}
        scores = mod.automated_score(self.WEIGHTS, self.protocol(), measured)
        self.assertEqual(scores["a"]["criteria"]["license_fit"], 0.0)
        self.assertEqual(scores["b"]["criteria"]["license_fit"], 1.0)

    def test_equal_candidates_tie_and_zero_values_do_not_divide_by_zero(self):
        measured = {"a": {"metrics": {"p99_ns": 0.0, "ops": 0.0}}, "b": {"metrics": {"p99_ns": 0.0, "ops": 0.0}}}
        scores = mod.automated_score(self.WEIGHTS, self.protocol(), measured)
        self.assertEqual(scores["a"]["score"], scores["b"]["score"])

    def test_worse_candidate_scores_lower(self):
        measured = {"a": {"metrics": {"p99_ns": 10.0, "ops": 100.0}},
                    "b": {"metrics": {"p99_ns": 40.0, "ops": 25.0}}}
        scores = mod.automated_score(self.WEIGHTS, self.protocol(), measured)
        self.assertGreater(scores["a"]["score"], scores["b"]["score"])


class StatusRules(unittest.TestCase):
    def setUp(self):
        self.tree = Tree()
        self.addCleanup(self.tree.close)
        self.h = self.tree.hash

    def run_with(self, statuses, records, notes=None):
        for stale in (self.tree.component / "evidence").glob("*.json"):
            stale.unlink()
        evidence = {}
        for cid, recs in records.items():
            evidence[cid] = [self.tree.evidence(f"20261009T10000{i}.000000Z-{cid}-run.json", r)
                             for i, r in enumerate(recs)]
        self.tree.candidates(statuses, notes, evidence)
        return self.tree.errors()

    def grep(self, errors, text):
        return [e for e in errors if text in e]

    def test_no_evidence_no_protocol_errors(self):
        self.assertEqual(self.run_with({"system": "evaluating", "mimalloc": "to-evaluate", "jemalloc": "to-evaluate"}, {}), [])

    def test_a_component_without_a_protocol_is_fine_until_it_has_benchmark_evidence(self):
        (self.tree.component / "protocol.toml").unlink()
        statuses = {"system": "evaluating", "mimalloc": "to-evaluate", "jemalloc": "to-evaluate"}
        self.assertEqual(self.run_with(statuses, {}), [])
        errors = self.run_with(statuses, {"system": [record("system", "0" * 64)]})
        self.assertTrue(self.grep(errors, "no protocol.toml"), errors)
        errors = self.run_with({**statuses, "system": "preferred"}, {})
        self.assertTrue(self.grep(errors, "preferred needs a protocol.toml"), errors)

    def test_legacy_runner_records_without_bench_are_ignored(self):
        legacy = {f: None for f in mod.RECORD_FIELDS} | {"candidate": {"id": "system"}}
        errors = self.run_with({"system": "evaluating", "mimalloc": "to-evaluate", "jemalloc": "to-evaluate"},
                               {"system": [legacy]})
        self.assertEqual(errors, [])

    def test_evidence_must_name_the_current_protocol_hash(self):
        statuses = {"system": "evaluating", "mimalloc": "to-evaluate", "jemalloc": "to-evaluate"}
        errors = self.run_with(statuses, {"system": [record("system", "f" * 64)]},
                               notes={"system": "lone survivor"})
        self.assertTrue(self.grep(errors, "names protocol"), errors)
        errors = self.run_with(statuses, {"system": [record("system", self.h)]},
                               notes={"system": "lone survivor"})
        self.assertEqual(errors, [])
        self.tree.write_protocol(PROTOCOL + "\n# edited after evidence\n")
        errors = self.tree.errors()
        self.assertTrue(self.grep(errors, "names protocol"), errors)

    def test_missing_runner_fields_are_reported(self):
        rec = record("system", self.h)
        del rec["stdout"]
        del rec["duration_ns"]
        errors = self.run_with({"system": "evaluating", "mimalloc": "to-evaluate", "jemalloc": "to-evaluate"},
                               {"system": [rec]}, notes={"system": "lone survivor"})
        self.assertTrue(self.grep(errors, "lacks fields"), errors)

    def test_a_record_for_another_candidate_is_rejected(self):
        errors = self.run_with({"system": "evaluating", "mimalloc": "to-evaluate", "jemalloc": "to-evaluate"},
                               {"system": [record("mimalloc", self.h)]}, notes={"system": "lone survivor"})
        self.assertTrue(self.grep(errors, "different candidate"), errors)

    def test_a_failed_gate_forces_rejected(self):
        failed = record("mimalloc", self.h, gates={"digest": False, "no-growth": True})
        statuses = {"system": "evaluating", "mimalloc": "evaluating", "jemalloc": "to-evaluate"}
        errors = self.run_with(statuses, {"mimalloc": [failed]}, notes={})
        self.assertTrue(self.grep(errors, "a gate failed, so the status must be rejected"), errors)
        statuses["mimalloc"] = "rejected"
        self.assertEqual(self.run_with(statuses, {"mimalloc": [failed]}), [])

    def test_incomplete_gate_sets_and_short_repetitions_are_not_passes(self):
        statuses = {"system": "evaluating", "mimalloc": "to-evaluate", "jemalloc": "to-evaluate"}
        errors = self.run_with(statuses, {"system": [record("system", self.h, gates={"digest": True})]})
        self.assertTrue(self.grep(errors, "differ from the protocol"), errors)
        errors = self.run_with(statuses, {"system": [record("system", self.h, measured=3)]})
        self.assertTrue(self.grep(errors, "below the protocol"), errors)
        errors = self.run_with(statuses, {"system": [record("system", self.h, ran=False)]})
        self.assertTrue(self.grep(errors, "did not run"), errors)

    def test_preferred_needs_two_candidates_that_passed_and_ran(self):
        both = {"system": [record("system", self.h)], "mimalloc": [record("mimalloc", self.h)]}
        statuses = {"system": "evaluating", "mimalloc": "preferred", "jemalloc": "to-evaluate"}
        self.assertEqual(self.run_with(statuses, both), [])
        lone = {"mimalloc": [record("mimalloc", self.h)]}
        errors = self.run_with(statuses, lone)
        self.assertTrue(self.grep(errors, "needs >= 2 candidates"), errors)

    def test_preferred_with_a_rejected_rival_is_still_a_lone_survivor(self):
        failed = record("system", self.h, gates={"digest": True, "no-growth": False})
        recs = {"system": [failed], "mimalloc": [record("mimalloc", self.h)]}
        statuses = {"system": "rejected", "mimalloc": "preferred", "jemalloc": "to-evaluate"}
        errors = self.run_with(statuses, recs)
        self.assertTrue(self.grep(errors, "needs >= 2 candidates"), errors)

    def test_a_lone_survivor_stays_evaluating_with_a_note(self):
        failed = record("system", self.h, gates={"digest": False, "no-growth": True})
        recs = {"system": [failed], "mimalloc": [record("mimalloc", self.h)]}
        statuses = {"system": "rejected", "mimalloc": "evaluating", "jemalloc": "to-evaluate"}
        errors = self.run_with(statuses, recs)
        self.assertTrue(self.grep(errors, "lone survivor"), errors)
        self.assertEqual(self.run_with(statuses, recs, notes={"mimalloc": "lone survivor; jemalloc not yet run"}), [])

    def test_not_decision_grade_evidence_cannot_make_a_candidate_preferred(self):
        recs = {"system": [record("system", self.h, grade=False)],
                "mimalloc": [record("mimalloc", self.h, grade=False)]}
        statuses = {"system": "evaluating", "mimalloc": "preferred", "jemalloc": "to-evaluate"}
        errors = self.run_with(statuses, recs)
        self.assertTrue(self.grep(errors, "not decision-grade"), errors)
        self.assertTrue(self.grep(errors, "needs >= 2"), errors)

    def test_a_record_cannot_claim_decision_grade_on_a_profile_that_is_not(self):
        (self.tree.root / "hardware/h.toml").write_text('id = "h"\ndecision_grade = false\n')
        recs = {"system": [record("system", self.h)], "mimalloc": [record("mimalloc", self.h)]}
        statuses = {"system": "evaluating", "mimalloc": "preferred", "jemalloc": "to-evaluate"}
        errors = self.run_with(statuses, recs)
        self.assertTrue(self.grep(errors, "claims decision_grade"), errors)
        self.assertTrue(self.grep(errors, "needs >= 2"), errors)
        honest = {c: [record(c, self.h, grade=False)] for c in ("system", "mimalloc")}
        errors = self.run_with(statuses, honest)
        self.assertFalse(self.grep(errors, "claims decision_grade"), errors)
        self.assertTrue(self.grep(errors, "not decision-grade"), errors)

    def test_preferred_needs_its_own_passing_record(self):
        recs = {"system": [record("system", self.h)], "jemalloc": [record("jemalloc", self.h)]}
        statuses = {"system": "evaluating", "mimalloc": "preferred", "jemalloc": "evaluating"}
        errors = self.run_with(statuses, recs)
        self.assertTrue(self.grep(errors, "mimalloc: preferred but its latest benchmark record"), errors)

    def test_rejected_without_gate_evidence_needs_a_note(self):
        statuses = {"system": "evaluating", "mimalloc": "rejected", "jemalloc": "to-evaluate"}
        errors = self.run_with(statuses, {})
        self.assertTrue(self.grep(errors, "needs a `note`"), errors)
        self.assertEqual(self.run_with(statuses, {}, notes={"mimalloc": "blocked: RUSTSEC-0000-0000"}), [])

    def test_the_latest_record_by_name_decides(self):
        old = record("mimalloc", self.h, gates={"digest": False, "no-growth": True})
        new = record("mimalloc", self.h)
        statuses = {"system": "evaluating", "mimalloc": "evaluating", "jemalloc": "to-evaluate"}
        errors = self.run_with(statuses, {"mimalloc": [old, new]}, notes={"mimalloc": "lone survivor"})
        self.assertEqual(errors, [])

    def test_protocol_candidates_must_exist_in_candidates_toml(self):
        self.tree.candidates({"system": "evaluating", "mimalloc": "to-evaluate", "jemalloc": "to-evaluate"})
        text = (self.tree.component / "candidates.toml").read_text().replace('"jemalloc"', '"other"')
        (self.tree.component / "candidates.toml").write_text(text)
        errors = self.tree.errors()
        self.assertTrue(self.grep(errors, "protocol candidate jemalloc"), errors)

    def test_a_missing_listed_evidence_file_is_an_error(self):
        statuses = {"system": "evaluating", "mimalloc": "to-evaluate", "jemalloc": "to-evaluate"}
        self.tree.candidates(statuses, evidence={"system": ["evidence/nope.json"]})
        self.assertTrue(self.grep(self.tree.errors(), "is missing"))

    def test_a_protocol_that_does_not_parse_is_reported(self):
        self.tree.write_protocol("version = [")
        self.tree.candidates({"system": "evaluating", "mimalloc": "to-evaluate", "jemalloc": "to-evaluate"})
        self.assertTrue(self.grep(self.tree.errors(), "does not parse"))


class GitOrder(unittest.TestCase):
    def git(self, root, *args):
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=root,
                       check=True, capture_output=True)

    def test_protocol_must_be_committed_before_its_evidence(self):
        tree = Tree()
        self.addCleanup(tree.close)
        root = tree.root
        self.git(root, "init", "-q")
        statuses = {"system": "evaluating", "mimalloc": "to-evaluate", "jemalloc": "to-evaluate"}
        tree.candidates(statuses, notes={"system": "lone survivor"},
                        evidence={"system": ["evidence/20261009T100000.000000Z-system-run.json"]})
        evidence = tree.component / "evidence/20261009T100000.000000Z-system-run.json"
        tree.evidence(evidence.name, record("system", tree.hash))
        protocol = tree.component / "protocol.toml"
        # Same commit: refused.
        self.git(root, "add", "-A")
        self.git(root, "commit", "-qm", "both")
        self.assertIn("same commit", mod.protocol_precedes(root, protocol, evidence))
        errors = mod.check_component(root, "alloc", use_git=True)
        self.assertTrue([e for e in errors if "same commit" in e], errors)

    def test_protocol_first_then_evidence_passes(self):
        tree = Tree()
        self.addCleanup(tree.close)
        root = tree.root
        self.git(root, "init", "-q")
        name = "20261009T100000.000000Z-system-run.json"
        statuses = {"system": "evaluating", "mimalloc": "to-evaluate", "jemalloc": "to-evaluate"}
        tree.candidates(statuses, notes={"system": "lone survivor"}, evidence={"system": [f"evidence/{name}"]})
        self.git(root, "add", "-A")
        self.git(root, "commit", "-qm", "protocol")
        evidence = tree.component / "evidence" / name
        tree.evidence(name, record("system", tree.hash))
        # Untracked evidence cannot predate the protocol.
        self.assertIsNone(mod.protocol_precedes(root, tree.component / "protocol.toml", evidence))
        self.git(root, "add", "-A")
        self.git(root, "commit", "-qm", "evidence")
        self.assertIsNone(mod.protocol_precedes(root, tree.component / "protocol.toml", evidence))
        self.assertEqual(mod.check_component(root, "alloc", use_git=True), [])

    def test_committed_evidence_with_an_untracked_protocol_is_refused(self):
        tree = Tree()
        self.addCleanup(tree.close)
        root = tree.root
        self.git(root, "init", "-q")
        name = "20261009T100000.000000Z-system-run.json"
        tree.evidence(name, record("system", tree.hash))
        evidence = tree.component / "evidence" / name
        self.git(root, "add", str(evidence.relative_to(root)))
        self.git(root, "commit", "-qm", "evidence only")
        self.assertIn("protocol.toml is not", mod.protocol_precedes(root, tree.component / "protocol.toml", evidence))


class TheRecordFormatMatchesTheRunnerAndTheKit(unittest.TestCase):
    def test_record_fields_equal_what_run_component_eval_py_writes(self):
        path = ROOT / "evaluations/components/ledger/candidates.toml"
        data = runner.load(path)
        match = data["candidate"][0]
        base = list(runner.base_record("ledger", match, data, path))
        run_keys = list(runner.execute_command(["true"]))
        expected = base + ["status", "started_at", "command", "finished_at"] + run_keys
        self.assertEqual(list(mod.RECORD_FIELDS), expected)

    def test_the_kit_writes_the_same_list(self):
        source = (ROOT / "evaluations/bench/kit/src/evidence.rs").read_text()
        block = re.search(r"RECORD_FIELDS: \[&str; \d+\] = \[(.*?)\];", source, re.S).group(1)
        self.assertEqual(re.findall(r'"([a-z_0-9]+)"', block), list(mod.RECORD_FIELDS))
        ext = re.search(r"EXTENSION_FIELDS: \[&str; \d+\] = \[(.*?)\];", source, re.S).group(1)
        self.assertEqual(re.findall(r'"([a-z_0-9]+)"', ext), list(mod.EXTENSION_FIELDS))

    def test_criteria_match_the_component_configs(self):
        data = runner.load(ROOT / "evaluations/components/allocator/candidates.toml")
        self.assertEqual(list(mod.CRITERIA), list(data["criteria"]))


class TheRepositoryItself(unittest.TestCase):
    def test_the_checked_in_components_pass(self):
        self.assertEqual(mod.check(ROOT, use_git=False), [])


if __name__ == "__main__":
    unittest.main()
