import json
import unittest

from ptr_training.moe import tasks


class MoeTaskTests(unittest.TestCase):
    def test_deterministic_for_seed(self):
        self.assertEqual(tasks.generate_tasks(20, seed=3), tasks.generate_tasks(20, seed=3))
        self.assertNotEqual(tasks.generate_tasks(20, seed=3), tasks.generate_tasks(20, seed=4))

    def test_kinds_cycle(self):
        out = tasks.generate_tasks(6, seed=1)
        self.assertEqual([t.kind for t in out], ["arith", "json"] * 3)

    def test_arithmetic_answer_is_exact(self):
        for t in tasks.generate_tasks(60, seed=5, kinds=("arith",), max_operand=20):
            expr = t.prompt.removeprefix("Q:").split("=")[0]
            self.assertEqual(str(eval(expr, {"__builtins__": {}})), t.answer)  # noqa: S307
            self.assertEqual(json.loads(t.spec)["value"], int(t.answer))

    def test_json_answer_matches_prompt(self):
        for t in tasks.generate_tasks(30, seed=9, kinds=("json",)):
            shown = dict(p.split("=") for p in t.prompt.removeprefix("J:").split("\n")[0].split(","))
            parsed = json.loads(t.answer)
            self.assertEqual({k: int(v) for k, v in shown.items()}, parsed)

    def test_sft_text_and_row(self):
        t = tasks.generate_tasks(1, seed=0)[0]
        self.assertEqual(tasks.sft_text(t, "<eos>"), t.prompt + t.answer + "<eos>")
        self.assertEqual(set(t.to_row()), {"prompt", "answer", "kind", "spec"})

    def test_enumerated_space_matches_sampler(self):
        for kind in tasks.KINDS:
            space = {t.prompt for t in tasks._enumerate(kind, 9)}
            self.assertEqual(len(space), tasks.space_size(kind, 9))
            sampled = {t.prompt for t in tasks.generate_tasks(400, seed=2, kinds=(kind,))}
            self.assertLessEqual(sampled, space)

    def test_heldout_is_distinct_and_deterministic(self):
        a = tasks.generate_heldout(32, seed=19)
        self.assertEqual(len({t.prompt for t in a}), 32)
        self.assertEqual(a, tasks.generate_heldout(32, seed=19))
        self.assertEqual([t.kind for t in a[:4]], ["arith", "json", "arith", "json"])

    def test_heldout_raises_when_space_too_small(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            tasks.generate_heldout(2000, seed=1)

    def test_exclusion(self):
        held = frozenset(t.prompt for t in tasks.generate_heldout(100, seed=3))
        out = tasks.generate_tasks(300, seed=4, exclude_prompts=held)
        self.assertFalse(held & {t.prompt for t in out})
        every_arith = frozenset(t.prompt for t in tasks._enumerate("arith", 9))
        with self.assertRaisesRegex(ValueError, "fully excluded"):
            tasks.generate_tasks(4, seed=1, exclude_prompts=every_arith)

    def test_validation(self):
        with self.assertRaises(ValueError):
            tasks.generate_tasks(-1, seed=0)
        with self.assertRaises(ValueError):
            tasks.generate_tasks(1, seed=0, kinds=("bogus",))
        with self.assertRaises(ValueError):
            tasks.generate_tasks(1, seed=0, kinds=())


if __name__ == "__main__":
    unittest.main()
