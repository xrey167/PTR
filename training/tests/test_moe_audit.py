"""Regression tests for the correctness audit of `ptr_training.moe` (pure Python, no torch).

Each test pins one defect found in the audit; the defect is named in the test docstring.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ptr_training.moe import corpora, m011, scorers


def _gsm_rows(n: int, offset: int = 0) -> list[dict]:
    return [
        {"question": f"Q{offset + i}: what is {i} + 1?", "answer": f"{i} + 1 = {i + 1}\n#### {i + 1}"}
        for i in range(n)
    ]


class Gsm8kItemIdTests(unittest.TestCase):
    """GSM8K train and test rows both used `gsm8k:<index>` as item id, so on the real corpus
    (7473 train / 1319 test) every validation item with index < 1319 collided with a test id
    and `validation_overlap_count` reported 131 overlaps that do not exist: a spurious
    invalidity of every M011 run."""

    def test_validation_does_not_collide_with_test_ids(self):
        parts = corpora.partition_gsm8k(_gsm_rows(40), _gsm_rows(40, offset=1000))
        report = corpora.corpus_report("gsm8k", parts)
        self.assertEqual(report["validation_overlap_count"], 0)
        ids = [ex.item_id for split in (parts.train, parts.validation, parts.evaluation) for ex in split]
        self.assertEqual(len(ids), len(set(ids)), "item ids must be unique across splits")

    def test_real_overlap_is_still_counted(self):
        train = _gsm_rows(40)
        test = _gsm_rows(40, offset=1000)
        test[3] = dict(train[9])  # validation row 9 reappears verbatim in test
        report = corpora.corpus_report("gsm8k", corpora.partition_gsm8k(train, test))
        self.assertEqual(report["validation_overlap_count"], 1)


class HumanEvalExitTests(unittest.TestCase):
    """A completion that calls `sys.exit(0)` or `os._exit(0)` ended the candidate process with
    exit code 0 before `check()` could fail, so it scored as a pass. The reference harness
    counts a `SystemExit` as a failure."""

    def setUp(self):
        if not scorers._bwrap_usable():
            patcher = mock.patch.dict(os.environ, {scorers.UNSANDBOXED_ENV: "1"})
            patcher.start()
            self.addCleanup(patcher.stop)
        self.ref = corpora.fixture_partitions()["humaneval"].train[1].reference

    def run_candidate(self, body: str) -> bool:
        r = self.ref
        return scorers.run_humaneval(r["prompt"], body, r["test"], r["entry_point"])

    def test_early_exit_is_not_a_pass(self):
        for body in (
            "    import sys\n    sys.exit(0)\n",
            "    import os\n    os._exit(0)\n",
            "    raise SystemExit(0)\n",
        ):
            with self.subTest(body=body):
                self.assertFalse(self.run_candidate(body))

    def test_a_correct_candidate_still_passes(self):
        ex = corpora.fixture_partitions()["humaneval"].train[1]
        self.assertTrue(self.run_candidate(ex.answer))


class Gsm8kFirstAnswerTests(unittest.TestCase):
    """The extractor read the number after the LAST `####`. A model that answers and then keeps
    generating an invented follow-up question with its own `#### n` was scored on the invented
    answer. The committed answer is the first `####` (the lm-eval strict-match rule)."""

    def test_first_marker_wins(self):
        gen = " 2 + 3 = 5\n#### 5\n\nQuestion: Ann has 7 pens.\nAnswer: 7\n#### 7"
        self.assertTrue(scorers.score_gsm8k(gen, "5"))
        self.assertFalse(scorers.score_gsm8k(gen, "7"))

    def test_single_marker_unchanged(self):
        self.assertTrue(scorers.score_gsm8k("so #### 1,234", "1234"))
        self.assertTrue(scorers.score_gsm8k("x\n#### -5", "-5"))


class TuningSeedTests(unittest.TestCase):
    """`--tune --seed 29` tuned on a confirmatory seed and only recorded a deviation. AGENTS.md
    and the M011 config (tuning_seed = 1001, "never one of the confirmatory seeds") forbid it,
    and a final run accepted such a tuning.json."""

    def test_confirmatory_and_low_seeds_are_refused_for_tuning(self):
        for seed in (*m011.PINS["seeds"], 0, 1000):
            with self.subTest(seed=seed), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(SystemExit):
                    m011.main(["--arm", "plain_lora", "--tune", "--seed", str(seed), "--out", tmp])

    def test_pilot_seeds_are_accepted_by_the_check(self):
        for seed in (1001, 1002, 5000):
            m011.check_tuning_seed(seed)

    def test_a_tuning_file_from_a_confirmatory_seed_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tuning.json"
            path.write_text(json.dumps({
                "arm": "mixlora", "tuning_seed": 17,
                "selected": {"learning_rate": 1e-3, "batch_size": 16},
            }))
            with self.assertRaises(SystemExit):
                m011._load_tuning(path, "mixlora")


class TopologyPinTests(unittest.TestCase):
    """run.json recorded `pins.moe_topology` (a copy of the pin) and no observed topology, so
    `model_pin_mismatches` compared the pin with itself and could never be nonzero."""

    def test_mismatches_compare_observed_with_pinned(self):
        pinned = m011.PINS["moe_topology"]
        same = {"layers": 16, "num_experts": 64, "top_k": 8, "norm_topk_prob": False,
                "router_aux_loss_coef": 0.01}
        self.assertEqual(m011.topology_mismatches(same, pinned), [])
        other = {**same, "top_k": 2}
        self.assertEqual(
            m011.topology_mismatches(other, pinned),
            [{"field": "top_k", "pinned": 8, "observed": 2}],
        )


class TrainingPartitionCounterTests(unittest.TestCase):
    """`training_outside_training_partition_count` was the constant 0, not a measurement."""

    def test_counter_counts_drawn_items_outside_train(self):
        parts = corpora.fixture_partitions()
        drawn = [parts["gsm8k"].train[0].item_id, parts["gsm8k"].evaluation[0].item_id,
                 parts["humaneval"].train[0].item_id, parts["bfcl_simple"].validation[0].item_id]
        counts = m011.outside_training_counts(drawn, parts)
        self.assertEqual(counts, {"gsm8k": 1, "humaneval": 0, "bfcl_simple": 1})


class PtrCudaDeviceTests(unittest.TestCase):
    """OD-3: a frozen GPU run is pinned through the declared PTR_CUDA_DEVICE; the M011
    entrypoint read neither it nor anything else, so such a run landed on card 0."""

    def test_sets_cuda_visible_devices_from_ptr_cuda_device(self):
        from ptr_training.moe import provenance

        env = {"PTR_CUDA_DEVICE": "1"}
        self.assertEqual(provenance.apply_ptr_cuda_device(env), "1")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")
        same = {"PTR_CUDA_DEVICE": "1", "CUDA_VISIBLE_DEVICES": "1"}
        self.assertEqual(provenance.apply_ptr_cuda_device(same), "1")
        self.assertIsNone(provenance.apply_ptr_cuda_device({}))
        for bad in ({"PTR_CUDA_DEVICE": "x"}, {"PTR_CUDA_DEVICE": "0,1"},
                    {"PTR_CUDA_DEVICE": "1", "CUDA_VISIBLE_DEVICES": "0"}):
            with self.subTest(env=bad), self.assertRaises(provenance.DeviceSelectionError):
                provenance.apply_ptr_cuda_device(dict(bad))

    def test_m011_refuses_a_conflicting_pair_before_any_work(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ, {"PTR_CUDA_DEVICE": "1", "CUDA_VISIBLE_DEVICES": "0"}
        ):
            with self.assertRaises(SystemExit):
                m011.main(["--arm", "mixlora", "--seed", "3", "--out", tmp])


if __name__ == "__main__":
    unittest.main()
