"""M011 corpora, scorers and tuning sampler (pure Python; offline, no torch)."""

import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
from unittest import mock
from pathlib import Path

from ptr_training.moe import corpora, m011, scorers


class PartitionTests(unittest.TestCase):
    def setUp(self):
        self.parts = corpora.fixture_partitions()

    def test_gsm8k_mod10_partition(self):
        p = self.parts["gsm8k"]
        self.assertEqual({e.index % 10 for e in p.validation}, {9})
        self.assertTrue(all(e.index % 10 != 9 for e in p.train))
        self.assertEqual(len(p.train) + len(p.validation), 30)  # fixture train split
        self.assertEqual(len(p.evaluation), 6)  # the test split, untouched

    def test_single_split_corpora_mod3(self):
        for key in ("humaneval", "bfcl_simple"):
            p = self.parts[key]
            self.assertEqual({e.index % 3 for e in p.train}, {0})
            self.assertEqual({e.index % 3 for e in p.validation}, {1})
            self.assertEqual({e.index % 3 for e in p.evaluation}, {2})
            self.assertEqual(len(p.train) + len(p.validation) + len(p.evaluation), 12)

    def test_partitions_are_disjoint_and_in_index_order(self):
        for key, p in self.parts.items():
            ids = [e.item_id for e in p.train + p.validation]
            self.assertEqual(len(ids), len(set(ids)), key)
            for split in (p.train, p.validation, p.evaluation):
                self.assertEqual([e.index for e in split], sorted(e.index for e in split))
            self.assertEqual(corpora.corpus_report(key, p)["validation_overlap_count"], 0)

    def test_domain_labels(self):
        self.assertEqual(
            {k: {e.domain for e in p.train} for k, p in self.parts.items()},
            {"gsm8k": {"math"}, "humaneval": {"code"}, "bfcl_simple": {"tool"}},
        )
        by = corpora.by_domain(self.parts, "train")
        self.assertEqual(set(by), {"math", "code", "tool"})

    def test_digests_deterministic_and_sensitive(self):
        a = corpora.corpus_report("gsm8k", corpora.fixture_partitions()["gsm8k"])
        b = corpora.corpus_report("gsm8k", corpora.fixture_partitions()["gsm8k"])
        self.assertEqual(a, b)
        digests = a["selection_digest"]
        self.assertEqual(len(set(digests.values())), 3)
        self.assertTrue(all(d.startswith("sha256:") and len(d) == 71 for d in digests.values()))
        # dropping one item changes the digest
        p = self.parts["gsm8k"]
        self.assertNotEqual(corpora.selection_digest(p.train), corpora.selection_digest(p.train[1:]))
        # reordering changes it too (selection is "in index order")
        self.assertNotEqual(
            corpora.selection_digest(p.train), corpora.selection_digest(p.train[::-1])
        )

    def test_missing_ground_truth_is_an_error(self):
        rows = [{"id": "simple_0", "question": [[{"role": "user", "content": "x"}]], "function": []}]
        with self.assertRaises(ValueError):
            corpora.partition_bfcl(rows, [])

    def test_pins_match_the_m011_config(self):
        self.assertEqual(
            {k: p.revision for k, p in corpora.PINNED_CORPORA.items()},
            {
                "gsm8k": "740312add88f781978c0658806c59bc2815b9866",
                "humaneval": "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
                "bfcl_simple": "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda",
            },
        )

    def test_loaders_are_lazy(self):
        # importing the module must not import the download libraries
        import subprocess
        import sys

        code = (
            "import sys; import ptr_training.moe.corpora, ptr_training.moe.m011;"
            "assert 'datasets' not in sys.modules and 'huggingface_hub' not in sys.modules;"
            "assert 'torch' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", code], check=True, env={**os.environ})


class ScorerTests(unittest.TestCase):
    def test_gsm8k(self):
        ref = scorers.gsm8k_reference("a + b = <<1+2=3>>3\n#### 1,234")
        self.assertEqual(ref, "1234")
        self.assertTrue(scorers.score_gsm8k("so #### 1234", ref))
        self.assertTrue(scorers.score_gsm8k("the total is $1,234.00 dollars", ref))
        self.assertFalse(scorers.score_gsm8k("the total is 1235", ref))
        self.assertFalse(scorers.score_gsm8k("no digits here", ref))
        self.assertTrue(scorers.score_gsm8k("x\n#### -5", "-5"))
        with self.assertRaises(ValueError):
            scorers.gsm8k_reference("no marker")

    def test_bfcl_ast(self):
        gt = [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]
        self.assertEqual(scorers.bfcl_target(gt), "calculate_triangle_area(base=10, height=5)")
        ok = scorers.score_bfcl
        self.assertTrue(ok("calculate_triangle_area(base=10, height=5)", gt))
        self.assertTrue(ok("[calculate_triangle_area(base=10.0, height=5, unit='Units')]", gt))
        self.assertFalse(ok("calculate_triangle_area(base=10)", gt))  # required missing
        self.assertFalse(ok("calculate_triangle_area(base=11, height=5)", gt))  # wrong value
        self.assertFalse(ok("calculate_triangle_area(base=10, height=5, color='red')", gt))
        self.assertFalse(ok("other(base=10, height=5)", gt))
        self.assertFalse(ok("not a call (", gt))
        dotted = [{"math.factorial": {"number": [5]}}]
        self.assertTrue(ok("math.factorial(number=5)", dotted))

    def test_fixture_targets_pass_their_own_scorers(self):
        parts = corpora.fixture_partitions()
        for ex in parts["gsm8k"].train:
            self.assertTrue(scorers.score("math", ex.answer, ex.reference))
        for ex in parts["bfcl_simple"].train:
            self.assertTrue(scorers.score("tool", ex.answer, ex.reference))

    def setUp(self):
        # Hosts without bubblewrap (CI) exercise the scoring logic through the recorded opt-in.
        if not scorers._bwrap_usable():
            patcher = mock.patch.dict(os.environ, {scorers.UNSANDBOXED_ENV: "1"})
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_humaneval_subprocess(self):
        ex = corpora.fixture_partitions()["humaneval"].train[1]
        r = ex.reference
        run = scorers.run_humaneval
        self.assertTrue(run(r["prompt"], ex.answer, r["test"], r["entry_point"]))
        # stop sequences cut trailing junk
        self.assertTrue(
            run(r["prompt"], ex.answer + "\nprint(undefined_name)", r["test"], r["entry_point"])
        )
        self.assertFalse(run(r["prompt"], "    return 0\n", r["test"], r["entry_point"]))
        self.assertFalse(run(r["prompt"], "    return (\n", r["test"], r["entry_point"]))

    def test_humaneval_timeout_and_no_network(self):
        r = corpora.fixture_partitions()["humaneval"].train[0].reference
        loop = "    while True:\n        pass\n"
        self.assertFalse(scorers.run_humaneval(r["prompt"], loop, r["test"], r["entry_point"], timeout=1.0))
        net = (
            "    import socket\n    socket.create_connection(('example.com', 80), 1)\n    return x\n"
        )
        self.assertFalse(scorers.run_humaneval(r["prompt"], net, r["test"], r["entry_point"]))

    def test_gsm8k_reads_only_well_formed_numbers(self):
        self.assertTrue(scorers.score_gsm8k("#### 1,234", "1234"))
        self.assertTrue(scorers.score_gsm8k("The answer is 12.", "12"))
        for malformed in ("#### 1,2", "#### 1,,2", "#### 12,34", "#### 1.2.3"):
            self.assertIsNone(scorers.gsm8k_extract(malformed), malformed)
            self.assertFalse(scorers.score_gsm8k(malformed, "12"), malformed)

    def test_bfcl_rejects_a_repeated_keyword(self):
        self.assertIsNone(scorers.parse_call("math.double(number=999, number=2)"))
        self.assertEqual(scorers.parse_call("f(a=1, b=2)"), ("f", {"a": 1, "b": 2}))

    def test_bfcl_mixed_key_dicts_score_false_without_raising(self):
        gt = [{"f": {"x": [{"1": "a", "b": 2}]}}]
        self.assertFalse(scorers.score_bfcl("f(x={1: 'a', 'b': 2, '1': 'a'})", gt))
        self.assertTrue(scorers.score_bfcl("f(x={1: 'a', 'b': 2})", gt))

    def test_bfcl_compares_large_integers_exactly(self):
        gt = [{"f": {"n": [9007199254740992]}}]
        self.assertTrue(scorers.score_bfcl("f(n=9007199254740992)", gt))
        self.assertFalse(scorers.score_bfcl("f(n=9007199254740993)", gt))
        self.assertTrue(scorers.score_bfcl("f(n=5.0)", [{"f": {"n": [5]}}]))

    def test_bfcl_keeps_booleans_apart_from_zero_and_one(self):
        self.assertFalse(scorers.score_bfcl("f(x=True)", [{"f": {"x": [1]}}]))
        self.assertFalse(scorers.score_bfcl("f(x=0)", [{"f": {"x": [False]}}]))
        self.assertTrue(scorers.score_bfcl("f(x=True)", [{"f": {"x": [True]}}]))
        self.assertTrue(scorers.score_bfcl("f(x=1)", [{"f": {"x": [1]}}]))

    def test_selection_digest_binds_the_scorer_reference(self):
        import dataclasses

        p = corpora.fixture_partitions()["humaneval"]
        changed = [dataclasses.replace(p.train[0], reference={**p.train[0].reference, "test": "x"})]
        self.assertNotEqual(corpora.selection_digest(p.train[:1]), corpora.selection_digest(changed))

    def test_validation_overlap_counts_evaluation_ids_and_identical_prompts(self):
        import dataclasses

        p = corpora.fixture_partitions()["gsm8k"]
        leaked_id = corpora.Partitions(
            p.train, [dataclasses.replace(p.validation[0], item_id=p.evaluation[0].item_id)], p.evaluation
        )
        self.assertEqual(corpora.corpus_report("gsm8k", leaked_id)["validation_overlap_count"], 1)
        leaked_text = corpora.Partitions(
            p.train, [dataclasses.replace(p.validation[0], prompt=p.evaluation[0].prompt)], p.evaluation
        )
        self.assertEqual(corpora.corpus_report("gsm8k", leaked_text)["validation_overlap_count"], 1)

    def test_humaneval_refuses_without_a_sandbox_or_an_opt_in(self):
        r = corpora.fixture_partitions()["humaneval"].train[0].reference
        with mock.patch.object(scorers, "_bwrap_usable", return_value=False), mock.patch.dict(
            os.environ, {scorers.UNSANDBOXED_ENV: ""}
        ):
            self.assertEqual(scorers.sandbox_info()["filesystem"], "refused")
            with self.assertRaises(scorers.SandboxUnavailable):
                scorers.run_humaneval(r["prompt"], "    return x\n", r["test"], r["entry_point"])

    def test_sandbox_command_is_read_only_hides_homes_and_unshares(self):
        cmd = scorers.sandbox_command("/usr/bin/bwrap", "/tmp/work", ["python3", "-I", "c.py"])
        self.assertEqual(cmd[:4], ["/usr/bin/bwrap", "--ro-bind", "/usr", "/usr"])
        # The host root is never bound wholesale, and nothing under /etc, /run or /var is (a
        # home directory appears only as the interpreter's own prefix, if a venv lives there).
        binds = [cmd[i + 1] for i, arg in enumerate(cmd) if arg in ("--ro-bind", "--bind")]
        self.assertNotIn("/", binds)
        for secret in ("/etc", "/run", "/var"):
            self.assertFalse(any(b == secret or b.startswith(secret + "/") for b in binds), secret)
        self.assertIn("--unshare-all", cmd)
        self.assertIn("--clearenv", cmd)
        self.assertIn("--die-with-parent", cmd)
        self.assertEqual(cmd[cmd.index("--bind") + 1 : cmd.index("--bind") + 3], ["/tmp/work"] * 2)
        hidden = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "--tmpfs"]
        self.assertIn("/tmp", hidden)
        self.assertEqual(cmd[-4:], ["--", "python3", "-I", "c.py"])

    @unittest.skipUnless(shutil.which("bwrap") and scorers._bwrap_usable(), "bubblewrap not usable")
    def test_the_sandbox_sees_no_host_secrets_but_runs_python(self):
        with tempfile.TemporaryDirectory() as work:
            # A venv under a home directory makes its own prefix path appear, read-only; nothing
            # else of the host may: no /etc, /run, /var or /root, and not this test file.
            hidden = ("/etc", "/run", "/var", "/root", os.path.abspath(__file__))
            probe = f"import os; print(sorted(p for p in {hidden!r} if os.path.exists(p)))"
            cmd = scorers.sandbox_command(shutil.which("bwrap"), work, [sys.executable, "-I", "-c", probe])
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True).stdout
        self.assertEqual(out.strip(), "[]")


class TuningTests(unittest.TestCase):
    def test_trial_sampler_deterministic_and_in_range(self):
        a = m011.sample_trials()
        self.assertEqual(a, m011.sample_trials())
        self.assertEqual(len(a), 8)
        for t in a:
            self.assertTrue(1e-5 <= t["learning_rate"] <= 1e-3)
            self.assertIn(t["batch_size"], (16, 32, 64))
        self.assertEqual([t["index"] for t in a], list(range(8)))

    def test_budget_is_a_prefix_and_digest_tracks_the_set(self):
        full, two = m011.sample_trials(), m011.sample_trials(budget=2)
        self.assertEqual(two, full[:2])
        self.assertEqual(m011.trial_set_digest(two), m011.trial_set_digest(m011.sample_trials(budget=2)))
        self.assertNotEqual(m011.trial_set_digest(two), m011.trial_set_digest(full))

    def test_tb1_tuning_steps_and_deviations(self):
        self.assertEqual(m011.PINS["seeds"], [17, 29, 43, 71, 101])
        self.assertEqual(m011.tuning_steps_for(2000), 500)
        self.assertEqual(m011.tuning_steps_for(7), 1)
        self.assertEqual(m011.tuning_steps_for(1), 1)  # at least one step
        self.assertEqual(m011.PINS["tuning_seed"], 1001)
        self.assertNotIn(m011.PINS["tuning_seed"], m011.PINS["seeds"])
        self.assertEqual(m011.tuning_deviations(seed=1001, steps=500, trials=8), [])
        self.assertEqual(
            [d["parameter"] for d in m011.tuning_deviations(seed=29, steps=499, trials=8)],
            ["tuning_seed", "tuning_steps"],
        )

    def test_select_best_ties_to_lowest_index(self):
        rs = [
            {"index": 0, "macro_accuracy": 0.2},
            {"index": 1, "macro_accuracy": 0.5},
            {"index": 2, "macro_accuracy": 0.5},
        ]
        self.assertEqual(m011.select_best(rs)["index"], 1)


class ConfigParityTests(unittest.TestCase):
    """Cross-check the embedded pins against the M011 config when it is available.

    Set PTR_M011_CONFIG to `experiments/model/M011-routed-peft/config.toml` (it lives in the
    experiment PR, not in this tree); the test is skipped otherwise.
    """

    CONFIG = os.environ.get("PTR_M011_CONFIG")

    @unittest.skipUnless(CONFIG and Path(CONFIG).is_file(), "PTR_M011_CONFIG not set")
    def test_pins_equal_config(self):
        cfg = tomllib.loads(Path(self.CONFIG).read_text())["pinned_thresholds"]
        self.assertEqual(m011.PINS["training_steps_per_arm"], cfg["training_steps_per_arm"])
        self.assertEqual(m011.PINS["tuning_search_spec"], cfg["tuning_search_spec"])
        self.assertEqual(m011.PINS["model_checkpoint_revision"], cfg["model_checkpoint_revision"])
        self.assertEqual(m011.PINS["tokenizer_revision"], cfg["tokenizer_revision"])
        self.assertEqual(m011.PINS["moe_topology"]["router_aux_loss_coef"], cfg["moe_topology"]["router_aux_loss_coef"])
        for key, pin in corpora.PINNED_CORPORA.items():
            short = {"gsm8k": "math", "humaneval": "code", "bfcl_simple": "tool"}[key]
            self.assertEqual(pin.revision, cfg[f"corpus_{short}_revision"])
            self.assertEqual(pin.repo, cfg[f"corpus_{short}"])
        self.assertEqual(corpora.PINNED_CORPORA["bfcl_simple"].file, cfg["corpus_tool_file"])


if __name__ == "__main__":
    unittest.main()
