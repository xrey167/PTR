"""Q003-v3 is Q003-v2's successor: the hypothesis, metrics, baselines, margins, thresholds and criteria
are unchanged value for value; it changes the harness (typed stops, timeouts, a watchdog), the five
declared seeds (the two that Q003-v2 ran or reserved are dropped) and the identifiers."""

import ast
import hashlib
import difflib
import re
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import check_research_gates as gate  # noqa: E402

OLD = ROOT / "experiments/retrieval/Q003-v2-postgres-hybrid"
NEW = ROOT / "experiments/retrieval/Q003-v3-postgres-hybrid"
PLACEHOLDER = "must-be-pinned-before-prepared"
HEX64 = re.compile(r"[0-9a-f]{64}")
FROZEN = ("prepared", "running", "completed", "failed")
STATUSES = ("planned", *FROZEN)
OLD_SEEDS = [17, 29, 43, 71, 101]
NEW_SEEDS = [43, 71, 101, 131, 173]
# The seeds Q003-v2 ran (17, completed) or reserved (29, stopped by owner decision).
SPENT = (17, 29)


def load(directory: Path, name: str) -> dict:
    return tomllib.loads((directory / name).read_text(encoding="utf-8"))


def read(directory: Path, name: str) -> str:
    return (directory / name).read_text(encoding="utf-8")


def without_seed_keys(table: dict) -> dict:
    return {key: value for key, value in table.items() if not key.startswith("workload_digest_seed_") and key != "seeds"}


def code_of(text: str) -> str:
    """A Python module's source without its docstring."""
    tree = ast.parse(text)
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
        tree.body.pop(0)
    return ast.unparse(tree)


class SuccessorTests(unittest.TestCase):
    def test_the_seeds_drop_the_two_spent_ones_and_keep_the_unrun_ones(self):
        manifest, config = load(NEW, "experiment.toml"), load(NEW, "config.toml")
        self.assertEqual(manifest["seeds"], NEW_SEEDS)
        self.assertEqual(config["preregistration"]["seeds"], NEW_SEEDS)
        self.assertEqual(config["pinned_thresholds"]["seeds"], NEW_SEEDS)
        self.assertFalse(set(NEW_SEEDS) & set(SPENT))
        # 43, 71 and 101 were generated and never run in Q003-v2: their digests are the same.
        old = load(OLD, "config.toml")["pinned_thresholds"]
        for seed in (43, 71, 101):
            self.assertEqual(config["pinned_thresholds"][f"workload_digest_seed_{seed}"], old[f"workload_digest_seed_{seed}"])
        for seed in NEW_SEEDS:
            digest = config["pinned_thresholds"][f"workload_digest_seed_{seed}"]
            self.assertRegex(digest, HEX64.pattern)
            self.assertNotEqual(digest, "0" * 64)
            self.assertEqual(config["preregistration"][f"workload_digest_seed_{seed}"], digest)
        for seed in SPENT:
            self.assertNotIn(f"workload_digest_seed_{seed}", config["pinned_thresholds"])
        # The declared seeds are none of the pilot seeds (1001 and up).
        self.assertTrue(all(seed < 1001 for seed in NEW_SEEDS))

    def test_criteria_differ_only_in_the_experiment_id(self):
        old, new = load(OLD, "criteria.toml"), load(NEW, "criteria.toml")
        self.assertEqual((old.pop("experiment"), new.pop("experiment")), ("Q003-v2", "Q003-v3"))
        self.assertEqual(old, new)
        self.assertEqual(read(OLD, "criteria.toml").replace('experiment = "Q003-v2"', 'experiment = "Q003-v3"'), read(NEW, "criteria.toml"))

    def test_thresholds_margins_and_pins_are_identical_but_for_the_seeds_and_the_limits(self):
        old, new = load(OLD, "config.toml"), load(NEW, "config.toml")
        self.assertEqual(without_seed_keys(old["pinned_thresholds"]), without_seed_keys(new["pinned_thresholds"]))
        self.assertEqual(new["pinned_thresholds"]["recall_margin_permille"], 30)
        self.assertEqual(new["pinned_thresholds"]["recall_exact_margin_permille"], 30)
        pinned_old, pinned_new = dict(old["preregistration"]), dict(new["preregistration"])
        for key in ("preregistration_text", "criteria"):
            self.assertEqual(
                pinned_new.pop(key), pinned_old.pop(key).replace("Q003-v2-postgres-hybrid", "Q003-v3-postgres-hybrid")
            )
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
        self.assertEqual(without_seed_keys(pinned_old), without_seed_keys(pinned_new))
        for key in old:
            if key not in ("name", "path", "preregistration", "pinned_thresholds"):
                self.assertEqual(old[key], new[key], key)
        self.assertEqual(new["name"], "Q003-v3-postgres-hybrid")
        self.assertEqual(new["path"], "experiments/retrieval/Q003-v3-postgres-hybrid")
        # The additions: the harness's own limits, which no rule reads, and the development seed's pin (the mutation
        # checker's seed 17, no longer declared, keeps Q003-v2's workload digest so that the generator mutation is seen).
        self.assertEqual(
            new["development_seed"], {"seed": 17, "workload_digest": old["pinned_thresholds"]["workload_digest_seed_17"]}
        )
        self.assertEqual(new["harness_limits"], {"request_timeout_seconds": 7200, "run_deadline_seconds": 86400, "heartbeat_seconds": 600})
        self.assertEqual(set(new) - set(old), {"harness_limits", "development_seed"})

    def test_the_manifest_differs_only_in_the_id_status_seeds_and_digests(self):
        old, new = load(OLD, "experiment.toml"), load(NEW, "experiment.toml")
        self.assertEqual((old["id"], new["id"]), ("Q003-v2", "Q003-v3"))
        self.assertIn(new["status"], STATUSES)
        for key in ("preregistration_sha256", "preregistration_rules_sha256"):
            if new["status"] in FROZEN:
                self.assertRegex(new[key], HEX64.pattern, key)
            else:
                self.assertNotIn(key, new)
        self.assertEqual(new["command_environment"], old["command_environment"])
        self.assertEqual(gate.command_environment_errors("Q003-v3", new), [])
        self.assertEqual(gate.preflight_errors("Q003-v3", new), [])
        # The one procedure change of round 2: the strong arm's driver is pinned to ONE torch thread (the thread count
        # changes only fp32 summation order); the entrypoint and the preflight carry the flag, nothing else differs.
        self.assertEqual(new["entrypoint"], old["entrypoint"] + " --strong-threads 1")
        self.assertEqual(new["preflight"], old["preflight"] + " --strong-threads 1")
        self.assertEqual(new["preflight"], new["entrypoint"].replace(" <seed>", " --preflight"))
        self.assertEqual(old["seeds"], OLD_SEEDS)
        for key in ("id", "status", "seeds", "entrypoint", "preflight", "preregistration_sha256", "preregistration_rules_sha256"):
            old.pop(key, None)
            new.pop(key, None)
        # Hypothesis, metrics, baseline, falsification, entrypoint, environment, preflight: word for word.
        self.assertEqual(old, new)

    def test_the_mutation_plan_keeps_v2s_and_adds_the_harness_stops(self):
        old, new = read(OLD, "tests/mutations.toml"), read(NEW, "tests/mutations.toml")
        self.assertTrue(new.startswith(old.rstrip("\n") + "\n"))
        plan_old, plan_new = load(OLD, "tests/mutations.toml"), load(NEW, "tests/mutations.toml")
        self.assertEqual(plan_new["mutation"][: len(plan_old["mutation"])], plan_old["mutation"])
        added = plan_new["mutation"][len(plan_old["mutation"]) :]
        self.assertEqual(
            [(m["name"], m.get("expect_stop") or m["expect_invalid"]) for m in added],
            [
                ("driver-reply-is-not-flushed", "request-timeout"),
                ("driver-stalls-on-a-search", "request-timeout"),
                ("driver-crashes-on-a-search", "driver-exited"),
                ("driver-exits-nonzero-after-quit", "driver-exited"),
                ("the-run-outlasts-its-deadline", "run-deadline"),
                ("driver-sleeps-while-a-large-request-is-written", "request-timeout"),
                ("driver-refuses-an-upsert", "driver-refused"),
                ("driver-prints-a-line-that-is-not-a-reply", "bad-reply"),
                ("strong-arm-records-the-default-thread-count", "the strong arm recorded strong_threads 4,"),
            ],
        )
        for key in ("package", "features", "subcommand", "cases", "seed", "hard_counters"):
            self.assertEqual(plan_old[key], plan_new[key], key)

    def test_the_driver_keeps_its_code_and_the_pinner_its_logic(self):
        self.assertEqual(code_of(read(OLD, "bm25_driver.py")), code_of(read(NEW, "bm25_driver.py")))
        # The pinner is the old one with the seeds replaced and the development seed (17) pinned beside them.
        old_pinner = read(OLD, "tests/pin_workload.py").replace("Q003-v2-postgres-hybrid", "Q003-v3-postgres-hybrid").replace(
            "SEEDS = (17, 29, 43, 71, 101)", "SEEDS = (43, 71, 101, 131, 173)"
        )
        new_pinner = read(NEW, "tests/pin_workload.py")
        self.assertNotEqual(old_pinner, new_pinner)
        self.assertIn("DEVELOPMENT_SEED = 17", new_pinner)
        self.assertIn("for seed in (*SEEDS, DEVELOPMENT_SEED):", new_pinner)
        for function in ("pin_params", "default_binary", "main"):
            self.assertEqual(
                old_pinner[old_pinner.index(f"def {function}("):].split("\n\n\n")[0],
                new_pinner[new_pinner.index(f"def {function}("):].split("\n\n\n")[0],
                function,
            )

    def test_the_aggregator_differs_only_in_the_experiment_id_and_the_seed_disclosure(self):
        # The two seed-disclosure sentences are one physical line each and are the only lines dropped; the thread-pin rule is
        # the one block of lines between its two anchors. Anything else (a second copy of the HNSW sentence, say) differs.
        def normal(text: str) -> str:
            lines = [line for line in text.splitlines() if not line.lstrip().startswith(('"seed 17 was run', '"the declared seeds 43'))]
            text = "\n".join(lines)
            start = text.find("    # Q003-v3 thread pin:")
            if start >= 0:
                end = text.index('    controls = result.get("controls")', start)
                text = text[:start] + text[end:]
            return re.sub(r"Q003(-v[23])?\b", "Q003", text)

        self.assertEqual(normal(read(OLD, "aggregate.py")), normal(read(NEW, "aggregate.py")))
        self.assertEqual(read(NEW, "aggregate.py").count("the HNSW list is approximate"), 1)
        self.assertEqual(read(NEW, "aggregate.py").count("# Q003-v3 thread pin:"), 1)
        for line in read(NEW, "aggregate.py").splitlines():
            if line.lstrip().startswith('"the declared seeds 43'):
                self.assertTrue(line.rstrip().endswith('",'), line[-60:])
        self.assertIn('EXPERIMENT = "Q003-v3"', read(NEW, "aggregate.py"))

    def test_the_preregistration_is_v2s_text_plus_a_successor_section(self):
        old, new = read(OLD, "PREREGISTRATION.md"), read(NEW, "PREREGISTRATION.md")
        # Sections 1 to 11 are v2's text but for the three places that state the command's thread flag (round 2: the
        # strong arm is pinned to one torch thread); everything after them is the new section 12.
        old_lines, new_lines = old.rstrip("\n").split("\n"), new.split("\n")
        changed = [
            "\n".join(old_lines[i1:i2])
            for tag, i1, i2, _j1, _j2 in difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False).get_opcodes()
            if tag != "equal" and i2 > i1
        ]
        self.assertEqual(len(changed), 3, changed)
        for block, anchor in zip(changed, ("postgres-hybrid <seed>`", "`--strong-threads N`", "Under the runner the command is fixed")):
            self.assertIn(anchor, block)
        start = new.index("## 12. Successor of Q003-v2")
        self.assertEqual(difflib.SequenceMatcher(None, old_lines, new[:start].rstrip("\n").split("\n"), autojunk=False).ratio() > 0.97, True)
        section = new[start:]
        self.assertTrue(section.lstrip().startswith("## 12. Successor of Q003-v2"))
        for fact in (
            "INVALID",
            "seed 17",
            "seed 29",
            "unchanged",
            "no loosening",
            "request-timeout",
            "run-deadline",
            "driver-exited",
            "Freeze procedure",
            "freeze_sign.py",
            "q003v2-hang-diagnosis.md",
            "Chronology",
        ):
            self.assertIn(fact, section)
        self.assertTrue((NEW / "q003v2-hang-diagnosis.md").is_file())

    def test_q003_v2_is_superseded_with_its_records_in_place(self):
        manifest = load(OLD, "experiment.toml")
        self.assertEqual(manifest["status"], "superseded")
        self.assertTrue((OLD / "SUPERSEDED.md").is_file())
        text = read(OLD, "SUPERSEDED.md")
        for fact in ("seed 17", "seed 29", "Q003-v3", "INVALID", "not evidence"):
            self.assertIn(fact, text.lower() if fact.startswith("seed") else text)
        registry = {entry["id"]: entry for entry in load(ROOT / "experiments", "registry.toml")["experiment"]}
        self.assertEqual(registry["Q003-v2"]["status"], "superseded")
        self.assertEqual(registry["Q003-v3"]["path"], "retrieval/Q003-v3-postgres-hybrid")
        self.assertEqual(registry["Q003-v3"]["status"], load(NEW, "experiment.toml")["status"])
        self.assertIn(registry["Q003-v3"]["status"], STATUSES)
        # The predecessor's prepared records are not patched: its results stay as they were.
        results = sorted(path.name for path in (OLD / "results").iterdir())
        self.assertEqual(len(results), 4, results)
        self.assertTrue(any(name.endswith("seed-17.json") for name in results))
        self.assertTrue(any(name.endswith("seed-29.json") for name in results))
        entries = load(ROOT / "experiments", "preregistration.toml")["experiment"]
        self.assertEqual(entries["Q003-v2"]["baseline"], entries["Q003-v3"]["baseline"])
        required_old, required_new = dict(entries["Q003-v2"]["required"]), dict(entries["Q003-v3"]["required"])
        self.assertEqual(without_seed_keys(required_old), without_seed_keys(required_new))
        self.assertEqual(sorted(k for k in required_new if k.startswith("workload_digest_seed_")), sorted(f"workload_digest_seed_{s}" for s in NEW_SEEDS))
        self.assertTrue((NEW / "results" / ".gitkeep").is_file())


if __name__ == "__main__":
    unittest.main()
