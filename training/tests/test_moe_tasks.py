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

    def test_validation(self):
        with self.assertRaises(ValueError):
            tasks.generate_tasks(-1, seed=0)
        with self.assertRaises(ValueError):
            tasks.generate_tasks(1, seed=0, kinds=("bogus",))
        with self.assertRaises(ValueError):
            tasks.generate_tasks(1, seed=0, kinds=())


if __name__ == "__main__":
    unittest.main()
