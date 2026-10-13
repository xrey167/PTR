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


class HumanEvalForgedPassTests(unittest.TestCase):
    """Codex P1: the pass marker path and token lived in the namespace the completion ran in,
    so a completion could write the marker itself and `os._exit(0)`. The completion now runs
    in a separate sandboxed process that holds no secret and never sees the tests; the trusted
    harness runs `check()` against a proxy and alone reports the outcome."""

    ATTACKS = {
        "writes the old marker": (
            "    import os\n"
            "    m = globals().get('_ptr_marker')\n"
            "    if m:\n"
            "        with m[2](m[0], 'w') as fh:\n"
            "            fh.write(m[1])\n"
            "    os._exit(0)\n"
        ),
        "dumps globals then os._exit": (
            "    import os, sys\n"
            "    for k, v in list(globals().items()):\n"
            "        print(k, repr(v), flush=True)\n"
            "        if isinstance(v, (tuple, list)):\n"
            "            for item in v:\n"
            "                if isinstance(item, str) and os.path.isabs(item):\n"
            "                    try:\n"
            "                        open(item, 'w').write(repr(v))\n"
            "                    except OSError:\n"
            "                        pass\n"
            "    sys.stdout.write('PASS\\n'); sys.stdout.flush()\n"
            "    os._exit(0)\n"
        ),
        "walks gc and frames then os._exit": (
            "    import gc, inspect, os, sys, re\n"
            "    found = set()\n"
            "    for obj in gc.get_objects():\n"
            "        if isinstance(obj, dict):\n"
            "            found.update(v for v in obj.values() if isinstance(v, str))\n"
            "    f = inspect.currentframe()\n"
            "    while f is not None:\n"
            "        found.update(v for v in f.f_locals.values() if isinstance(v, str))\n"
            "        found.update(v for v in f.f_globals.values() if isinstance(v, str))\n"
            "        f = f.f_back\n"
            "    for tok in found:\n"
            "        if re.fullmatch(r'[0-9a-f]{32}', tok):\n"
            "            for out in (sys.stdout, sys.__stdout__):\n"
            "                out.write(tok + '\\n'); out.flush()\n"
            "            for name in os.listdir('.'):\n"
            "                try:\n"
            "                    open(name, 'w').write(tok)\n"
            "                except OSError:\n"
            "                    pass\n"
            "            os.write(1, tok.encode())\n"
            "    os._exit(0)\n"
        ),
        "answers the harness protocol itself": (
            "    import os, sys, json\n"
            "    for fd in (1, 2, 3, 4, 5):\n"
            "        try:\n"
            "            os.write(fd, b'{\"ok\": true}\\nPASS\\n')\n"
            "        except OSError:\n"
            "            pass\n"
            "    os._exit(0)\n"
        ),
    }

    def setUp(self):
        if not scorers._bwrap_usable():
            patcher = mock.patch.dict(os.environ, {scorers.UNSANDBOXED_ENV: "1"})
            patcher.start()
            self.addCleanup(patcher.stop)
        self.ex = corpora.fixture_partitions()["humaneval"].train[1]

    def run_candidate(self, body: str) -> bool:
        r = self.ex.reference
        return scorers.run_humaneval(r["prompt"], body, r["test"], r["entry_point"])

    def test_forged_passes_are_failures(self):
        for name, body in self.ATTACKS.items():
            with self.subTest(attack=name):
                self.assertFalse(self.run_candidate(body))

    def test_correct_and_printing_candidates_pass_wrong_ones_fail(self):
        self.assertTrue(self.run_candidate(self.ex.answer))
        noisy = "    print('debug', x)\n" + self.ex.answer
        self.assertTrue(self.run_candidate(noisy))
        self.assertFalse(self.run_candidate("    return 0\n"))
        self.assertFalse(self.run_candidate("    raise ValueError('no')\n"))

    def test_prompt_helpers_and_rich_return_values(self):
        prompt = (
            "from typing import List, Tuple\n\n"
            "def helper(s: str) -> str:\n    return s[::-1]\n\n\n"
            "def solve(xs: List[int]) -> Tuple[List[float], dict, set, bool, None]:\n"
            '    """Return several value kinds."""\n'
        )
        test = (
            "def check(candidate):\n"
            "    assert candidate([1, 2]) == ([0.5, float('inf')], {'a': (1, 2)}, {3}, True, None)\n"
            "    assert candidate([3]) is not None\n"
            "    assert helper('ab') == 'ba'\n"
        )
        good = (
            "    return ([0.5, float('inf')], {'a': (1, 2)}, {3}, True, None) if len(xs) == 2 "
            "else ([], {}, set(), False, 1)\n"
        )
        self.assertTrue(scorers.run_humaneval(prompt, good, test, "solve"))
        self.assertFalse(scorers.run_humaneval(prompt, "    return None\n", test, "solve"))


class ModelPinTests(unittest.TestCase):
    """Codex P1: an unverifiable checkpoint commit (None) counted as no mismatch and the
    tokenizer commit was never checked. Unknown now counts as a mismatch, and both the model and
    the tokenizer commit are compared with their pins."""

    PIN = m011.PINS["model_checkpoint_revision"]

    def test_unknown_and_wrong_commits_count(self):
        ok = {"model": self.PIN, "tokenizer": self.PIN}
        self.assertEqual(m011.commit_pin_report(ok)["mismatches"], 0)
        unknown = m011.commit_pin_report({"model": None, "tokenizer": self.PIN})
        self.assertEqual((unknown["mismatches"], unknown["unverified"]), (1, 1))
        wrong_tok = m011.commit_pin_report({"model": self.PIN, "tokenizer": "0" * 40})
        self.assertEqual((wrong_tok["mismatches"], wrong_tok["unverified"]), (1, 0))

    def test_observed_commit_from_hub_or_cache(self):
        import types

        fake = types.SimpleNamespace(
            HfApi=lambda: types.SimpleNamespace(
                model_info=lambda repo, revision: types.SimpleNamespace(sha=self.PIN)
            ),
            try_to_load_from_cache=lambda *a, **k: None,
        )
        with mock.patch.dict("sys.modules", {"huggingface_hub": fake}):
            got = m011.observed_commit("org/repo", self.PIN, "config.json")
        self.assertEqual(got, {"commit": self.PIN, "source": "hub"})

        def offline():
            raise OSError("offline")

        snap = f"/cache/models--org--repo/snapshots/{self.PIN}/config.json"
        fake = types.SimpleNamespace(
            HfApi=lambda: types.SimpleNamespace(model_info=lambda *a, **k: offline()),
            try_to_load_from_cache=lambda *a, **k: snap,
        )
        with mock.patch.dict("sys.modules", {"huggingface_hub": fake}):
            got = m011.observed_commit("org/repo", self.PIN, "config.json")
        self.assertEqual(got["commit"], self.PIN)
        self.assertEqual(got["source"], "cache")
        fake.try_to_load_from_cache = lambda *a, **k: None
        with mock.patch.dict("sys.modules", {"huggingface_hub": fake}):
            got = m011.observed_commit("org/repo", self.PIN, "config.json")
        self.assertIsNone(got["commit"])
        self.assertIn("offline", got["reason"])


if __name__ == "__main__":
    unittest.main()
