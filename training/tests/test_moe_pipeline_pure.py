import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from ptr_training.moe import config as moe_config
from ptr_training.moe import pipeline

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


class MoeTaskSplitTests(unittest.TestCase):
    def test_eval_disjoint_from_train_for_both_configs(self):
        for name in ("stage-moe-smoke.toml", "stage-moe-olmoe.toml"):
            with self.subTest(config=name):
                cfg = moe_config.load(CONFIGS / name)
                sft, rl, ev, stats = pipeline.build_task_splits(cfg)
                train = {t.prompt for t in sft} | {t.prompt for t in rl}
                self.assertFalse({t.prompt for t in ev} & train)
                self.assertEqual(stats["eval_overlap"], 0)
                self.assertEqual(stats["eval_unique_prompts"], stats["eval_tasks"])
                self.assertEqual(stats["sft_tasks"], cfg.sft.examples)
                self.assertEqual(len(ev), pipeline.EVAL_TASKS)

    def test_deterministic(self):
        cfg = moe_config.load(CONFIGS / "stage-moe-smoke.toml")
        self.assertEqual(pipeline.build_task_splits(cfg), pipeline.build_task_splits(cfg))


class MoeRunDirTests(unittest.TestCase):
    def test_unique_dir_never_reused_and_metrics_not_overwritten(self):
        now = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            first = pipeline.unique_run_dir(Path(tmp), "my run", "abcd1234", now)
            self.assertEqual(first.name, "my-run-20260102T030405Z-abcd1234")
            with self.assertRaises(FileExistsError):
                pipeline.unique_run_dir(Path(tmp), "my run", "abcd1234", now)
            later = pipeline.unique_run_dir(Path(tmp), "my run", "abcd1234", now.replace(second=6))
            self.assertNotEqual(first, later)
            path = pipeline.write_metrics({"a": 1}, first)
            with self.assertRaises(FileExistsError):
                pipeline.write_metrics({"a": 2}, first)
            self.assertEqual(json.loads(path.read_text()), {"a": 1})


if __name__ == "__main__":
    unittest.main()
