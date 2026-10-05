import json
import tempfile
import unittest
from unittest import mock
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

    def _rl_rows(self, name, **rl):
        import dataclasses

        cfg = moe_config.load(CONFIGS / name)
        cfg = dataclasses.replace(cfg, rl=dataclasses.replace(cfg.rl, **rl))
        return len(pipeline.build_task_splits(cfg)[1])

    def test_rl_pool_sized_from_optimizer_steps(self):
        # per optimizer step TRL consumes prompts_per_step NEW prompts; the pool covers whole
        # generation batches: ceil(steps / mini_steps) * mini_steps * prompts_per_step
        self.assertEqual(self._rl_rows("stage-moe-smoke.toml", steps=400, mini_steps=4,
                                       prompts_per_step=2), 400 * 2)
        # partial final generation batch: 10 steps, mini_steps 4 -> 3 batches x 4 x 2 = 24 rows
        # (floor of 64 hides it, so use a larger prompts_per_step)
        self.assertEqual(self._rl_rows("stage-moe-smoke.toml", steps=10, mini_steps=4,
                                       prompts_per_step=8), 3 * 4 * 8)
        self.assertEqual(self._rl_rows("stage-moe-smoke.toml", steps=13, mini_steps=4,
                                       prompts_per_step=8), 4 * 4 * 8)
        self.assertEqual(self._rl_rows("stage-moe-smoke.toml", steps=1, mini_steps=1,
                                       prompts_per_step=1), 64)  # floor

    def test_rl_pool_for_stage_configs(self):
        smoke = moe_config.load(CONFIGS / "stage-moe-smoke.toml").rl  # 16 steps, mini 2, 2/step
        self.assertEqual(self._rl_rows("stage-moe-smoke.toml"),
                         max(64, -(-smoke.steps // smoke.mini_steps) * smoke.mini_steps
                             * smoke.prompts_per_step))
        olmoe = moe_config.load(CONFIGS / "stage-moe-olmoe.toml").rl  # 50 steps, mini 4, 2/step
        rows = self._rl_rows("stage-moe-olmoe.toml")
        self.assertGreaterEqual(rows, olmoe.steps * olmoe.prompts_per_step)  # >= 100
        self.assertEqual(rows, max(64, 13 * 4 * 2))  # ceil(50/4)=13 batches of 4 steps x 2

    def test_no_rl_pool_when_no_algorithms(self):
        import dataclasses

        cfg = moe_config.load(CONFIGS / "stage-moe-smoke.toml")
        cfg = dataclasses.replace(cfg, rl=dataclasses.replace(cfg.rl, algorithms=()))
        sft, rl_tasks, ev, stats = pipeline.build_task_splits(cfg)
        self.assertEqual(rl_tasks, [])
        self.assertEqual((stats["rl_pool_size"], stats["rl_pool_built"]), (0, False))
        self.assertEqual(stats["eval_overlap"], 0)
        full = pipeline.build_task_splits(moe_config.load(CONFIGS / "stage-moe-smoke.toml"))[3]
        self.assertTrue(full["rl_pool_built"])

    def test_deterministic(self):
        cfg = moe_config.load(CONFIGS / "stage-moe-smoke.toml")
        self.assertEqual(pipeline.build_task_splits(cfg), pipeline.build_task_splits(cfg))


class MoeEntryPointGateTests(unittest.TestCase):
    def _unselected_config(self, tmp: Path) -> Path:
        text = (CONFIGS / "stage-moe-smoke.toml").read_text(encoding="utf-8")
        text = text.replace("allow_unselected_backend = true\n", "")
        path = tmp / "stage.toml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_run_config_refuses_unselected_backend_before_creating_anything(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            config = self._unselected_config(Path(tmp))
            out = Path(tmp) / "runs"
            with mock.patch.object(
                pipeline, "execute", side_effect=AssertionError("executed")
            ) as execute:
                with self.assertRaises(moe_config.BackendNotSelected) as ctx:
                    pipeline.run_config(config, out)
            execute.assert_not_called()
            self.assertEqual(str(ctx.exception), moe_config.UNSELECTED_BACKEND_MESSAGE)
            self.assertFalse(out.exists())  # no run directory was created

    def test_execute_enforces_the_same_gate_first(self):
        import dataclasses
        from unittest import mock

        cfg = dataclasses.replace(
            moe_config.load(CONFIGS / "stage-moe-smoke.toml"), allow_unselected_backend=False
        )
        with mock.patch.object(pipeline, "require_extra", side_effect=AssertionError("order")):
            with self.assertRaises(moe_config.BackendNotSelected):
                pipeline.execute(cfg, Path("unused"))

    def test_cli_and_run_config_share_the_message(self):
        from ptr_training import run

        self.assertTrue(hasattr(run, "main"))
        text = Path(run.__file__).read_text(encoding="utf-8")
        self.assertIn("require_backend_opt_in", text)
        self.assertNotIn("is not selected (component", text)  # message lives in one place


class MoeRunDirTests(unittest.TestCase):
    def test_immediate_calls_give_distinct_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            dirs = {pipeline.unique_run_dir(Path(tmp), "r", "h") for _ in range(20)}
            self.assertEqual(len(dirs), 20)

    def test_unique_dir_never_reused_and_metrics_not_overwritten(self):
        now = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            first = pipeline.unique_run_dir(Path(tmp), "my run", "abcd1234", now)
            self.assertRegex(first.name, r"^my-run-20260102T030405000000Z-abcd1234-[0-9a-f]{6}$")
            with mock.patch.object(pipeline.secrets, "token_hex", return_value="aaaaaa"):
                pipeline.unique_run_dir(Path(tmp), "my run", "abcd1234", now)
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
