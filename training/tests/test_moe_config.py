import copy
import re
import tomllib
import unittest
from pathlib import Path

from ptr_training.moe import config as moe_config
from ptr_training.moe.config import ConfigError

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _raw(name="stage-moe-smoke.toml") -> dict:
    return tomllib.loads((CONFIGS / name).read_text(encoding="utf-8"))


class MoeConfigTests(unittest.TestCase):
    def test_smoke_config_parses(self):
        cfg = moe_config.load(CONFIGS / "stage-moe-smoke.toml")
        self.assertEqual(cfg.model.kind, "tiny-random")
        self.assertEqual((cfg.model.layers, cfg.model.experts, cfg.model.top_k), (2, 8, 2))
        self.assertEqual(cfg.rl.algorithms, ("gspo", "sapo"))
        self.assertGreaterEqual(cfg.rl.mini_steps, 2)
        self.assertFalse(cfg.lora.enabled)

    def test_olmoe_config_parses(self):
        cfg = moe_config.load(CONFIGS / "stage-moe-olmoe.toml")
        self.assertEqual(cfg.model.kind, "pretrained")
        self.assertEqual(cfg.model.hf_id, "allenai/OLMoE-1B-7B-0125")
        self.assertRegex(cfg.model.revision, r"^[0-9a-f]{40}$")
        self.assertEqual(cfg.model.precision, "bf16")
        self.assertTrue(cfg.lora.enabled)
        self.assertEqual(cfg.lora.rank, 16)
        self.assertIn("q_proj", cfg.lora.target_modules)
        self.assertIn("mlp.experts.gate_up_proj", cfg.lora.target_parameters)

    def _mutate(self, fn):
        raw = copy.deepcopy(_raw())
        fn(raw)
        return raw

    def test_wrong_backend_rejected(self):
        raw = self._mutate(lambda r: r["training"].update(backend="dry-run"))
        with self.assertRaisesRegex(ConfigError, "trl-moe"):
            moe_config.parse(raw)

    def test_unknown_model_kind(self):
        raw = self._mutate(lambda r: r["model"].update(kind="mystery"))
        with self.assertRaisesRegex(ConfigError, "model.kind"):
            moe_config.parse(raw)

    def test_pretrained_requires_hf_id(self):
        def mutate(r):
            r["model"]["kind"] = "pretrained"

        with self.assertRaisesRegex(ConfigError, "hf_id"):
            moe_config.parse(self._mutate(mutate))

    def test_pretrained_requires_pinned_revision(self):
        def mutate(r):
            r["model"].update(kind="pretrained", hf_id="org/model")
            for key in ("layers", "hidden", "intermediate", "heads", "experts", "top_k", "max_positions"):
                r["model"].pop(key)

        with self.assertRaisesRegex(ConfigError, r"model\.revision is required"):
            moe_config.parse(self._mutate(mutate))
        for bad in ("main", "master", "HEAD", "v1.0", "abc123", "A" * 40, "0123456789abcdef" * 3):
            raw = self._mutate(mutate)
            raw["model"]["revision"] = bad
            with self.assertRaisesRegex(ConfigError, "40-hex"):
                moe_config.parse(raw)
        raw = self._mutate(mutate)
        raw["model"]["revision"] = "0123456789abcdef0123456789abcdef01234567"
        with self.assertRaisesRegex(ConfigError, "allow_full_finetune"):
            moe_config.parse(raw)  # pretrained without LoRA = full fine-tune
        raw["moe"]["allow_full_finetune"] = True
        self.assertEqual(moe_config.parse(raw).model.revision, "0123456789abcdef0123456789abcdef01234567")

    def test_pretrained_lora_does_not_need_full_finetune_flag(self):
        cfg = moe_config.load(CONFIGS / "stage-moe-olmoe.toml")
        self.assertTrue(cfg.lora.enabled)
        self.assertFalse(cfg.allow_full_finetune)

    def test_revision_rejected_for_tiny_random(self):
        raw = self._mutate(lambda r: r["model"].update(revision="abc"))
        with self.assertRaisesRegex(ConfigError, "only valid"):
            moe_config.parse(raw)

    def test_rl_steps_must_be_positive(self):
        for bad in (0, -1):
            raw = self._mutate(lambda r: r["moe"]["rl"].update(steps=bad))
            with self.assertRaisesRegex(ConfigError, r"moe\.rl\.steps"):
                moe_config.parse(raw)

    def test_empty_algorithms_disables_rl(self):
        cfg = moe_config.parse(self._mutate(lambda r: r["moe"]["rl"].update(algorithms=[])))
        self.assertEqual(cfg.rl.algorithms, ())

    def test_typo_in_owned_tables_rejected(self):
        cases = {
            "moe.rl": lambda r: r["moe"]["rl"].update(num_generation=4),
            "moe.sft": lambda r: r["moe"]["sft"].update(stepz=1),
            "moe.lora": lambda r: r["moe"]["lora"].update(rnk=4),
            "moe.output": lambda r: r["moe"]["output"].update(dirr="x"),
            "model": lambda r: r["model"].update(layerz=2),
            "moe": lambda r: r["moe"].update(extra={}),
        }
        for table, fn in cases.items():
            with self.subTest(table=table):
                with self.assertRaisesRegex(ConfigError, "unknown key.*" + re.escape(f"[{table}]")):
                    moe_config.parse(self._mutate(fn))

    def test_typo_message_names_key(self):
        raw = self._mutate(lambda r: r["moe"]["rl"].update(gspo_epsilom=0.1))
        with self.assertRaisesRegex(ConfigError, r"gspo_epsilom.*moe\.rl"):
            moe_config.parse(raw)

    def test_shared_manifest_tables_not_rejected(self):
        raw = self._mutate(lambda r: r["data"].update(extra_data_key=1))
        raw["codebook"]["extra"] = "x"
        raw["evaluation"]["another"] = "x"
        moe_config.parse(raw)

    def test_ignored_training_keys_rejected(self):
        for key in ("learning_rate", "max_steps", "batch_size", "gradient_accumulation",
                    "checkpoint_every", "anything"):
            raw = self._mutate(lambda r: r["training"].update({key: 1}))
            with self.assertRaisesRegex(ConfigError, rf"\[training\] keys .*{key}"):
                moe_config.parse(raw)

    def test_evaluation_every_and_suites_must_be_inert(self):
        raw = self._mutate(lambda r: r["evaluation"].update(every=10))
        with self.assertRaisesRegex(ConfigError, "evaluation"):
            moe_config.parse(raw)
        raw = self._mutate(lambda r: r["evaluation"].update(suites=["x"]))
        with self.assertRaisesRegex(ConfigError, "evaluation"):
            moe_config.parse(raw)

    def test_sapo_temperature_neg_must_exceed_pos(self):
        for pos, neg in ((1.0, 1.0), (1.2, 1.0)):
            raw = self._mutate(
                lambda r: r["moe"]["rl"].update(sapo_temperature_pos=pos, sapo_temperature_neg=neg)
            )
            with self.assertRaisesRegex(ConfigError, "sapo_temperature_neg"):
                moe_config.parse(raw)
        raw = self._mutate(
            lambda r: r["moe"]["rl"].update(sapo_temperature_pos=1.0, sapo_temperature_neg=1.01)
        )
        moe_config.parse(raw)

    def test_resource_maxima_and_allow_large(self):
        cases = (("sft", "steps", 100_001), ("sft", "examples", 100_001), ("sft", "batch", 1025),
                 ("rl", "steps", 10_001), ("rl", "num_generations", 65),
                 ("rl", "prompts_per_step", 65), ("rl", "max_completion_length", 513),
                 ("output", "probe_tokens", 4097))
        for table, key, value in cases:
            with self.subTest(key=f"{table}.{key}"):
                raw = self._mutate(lambda r: r["moe"][table].update({key: value}))
                raw["model"]["max_positions"] = 4096  # keep the position bound out of the way
                with self.assertRaisesRegex(ConfigError, rf"moe\.{table}\.{key}.*allow_large"):
                    moe_config.parse(raw)
                raw["moe"]["allow_large"] = True
                self.assertTrue(moe_config.parse(raw).allow_large)

    def test_top_k_exceeding_experts(self):
        raw = self._mutate(lambda r: r["model"].update(top_k=9))
        with self.assertRaisesRegex(ConfigError, "top_k"):
            moe_config.parse(raw)

    def test_unknown_algorithm(self):
        raw = self._mutate(lambda r: r["moe"]["rl"].update(algorithms=["gspo", "ppo"]))
        with self.assertRaisesRegex(ConfigError, "unknown entries"):
            moe_config.parse(raw)

    def test_empty_and_duplicate_algorithms(self):
        with self.assertRaisesRegex(ConfigError, "list of strings"):
            moe_config.parse(self._mutate(lambda r: r["moe"]["rl"].update(algorithms="gspo")))
        with self.assertRaisesRegex(ConfigError, "duplicates"):
            moe_config.parse(
                self._mutate(lambda r: r["moe"]["rl"].update(algorithms=["sapo", "sapo"]))
            )

    def test_num_generations_minimum(self):
        raw = self._mutate(lambda r: r["moe"]["rl"].update(num_generations=1))
        with self.assertRaisesRegex(ConfigError, "num_generations"):
            moe_config.parse(raw)

    def test_type_errors_are_clear(self):
        raw = self._mutate(lambda r: r["moe"]["sft"].update(steps="many"))
        with self.assertRaisesRegex(ConfigError, r"moe\.sft\.steps must be int"):
            moe_config.parse(raw)

    def test_missing_key(self):
        def mutate(r):
            del r["moe"]["sft"]["lr"]

        with self.assertRaisesRegex(ConfigError, r"moe\.sft\.lr"):
            moe_config.parse(self._mutate(mutate))

    def test_lora_enabled_needs_targets(self):
        raw = self._mutate(lambda r: r["moe"].update(lora={"enabled": True}))
        with self.assertRaisesRegex(ConfigError, "no target_modules"):
            moe_config.parse(raw)

    def test_gspo_epsilon_defaults_and_values(self):
        cfg = moe_config.load(CONFIGS / "stage-moe-smoke.toml")
        self.assertEqual((cfg.rl.gspo_epsilon, cfg.rl.gspo_epsilon_high), (3e-4, 4e-4))
        raw = self._mutate(lambda r: [r["moe"]["rl"].pop(k) for k in ("gspo_epsilon", "gspo_epsilon_high")])
        cfg = moe_config.parse(raw)
        self.assertEqual((cfg.rl.gspo_epsilon, cfg.rl.gspo_epsilon_high), (3e-4, 4e-4))

    def test_gspo_epsilon_validation_names_key(self):
        raw = self._mutate(lambda r: r["moe"]["rl"].update(gspo_epsilon=0.0))
        with self.assertRaisesRegex(ConfigError, "gspo_epsilon"):
            moe_config.parse(raw)
        raw = self._mutate(lambda r: r["moe"]["rl"].update(gspo_epsilon=0.1, gspo_epsilon_high=0.01))
        with self.assertRaisesRegex(ConfigError, "gspo_epsilon_high"):
            moe_config.parse(raw)
        raw = self._mutate(lambda r: r["moe"]["rl"].update(gspo_epsilon_high="x"))
        with self.assertRaisesRegex(ConfigError, "gspo_epsilon_high"):
            moe_config.parse(raw)

    def test_smoke_logs_kl_and_olmoe_mini_steps(self):
        self.assertGreater(moe_config.load(CONFIGS / "stage-moe-smoke.toml").rl.beta, 0)
        self.assertEqual(moe_config.load(CONFIGS / "stage-moe-olmoe.toml").rl.mini_steps, 4)

    def test_router_aux_loss_coef_non_negative(self):
        raw = self._mutate(lambda r: r["model"].update(router_aux_loss_coef=-0.1))
        with self.assertRaisesRegex(ConfigError, "router_aux_loss_coef"):
            moe_config.parse(raw)
        raw = self._mutate(lambda r: r["model"].update(router_aux_loss_coef=0))
        self.assertEqual(moe_config.parse(raw).model.router_aux_loss_coef, 0.0)

    def test_training_precision_must_match_model_precision(self):
        raw = self._mutate(lambda r: r["training"].update(precision="bf16"))
        with self.assertRaisesRegex(ConfigError, "precision"):
            moe_config.parse(raw)

    def test_seed_range(self):
        for table, bad in (("run", -1), ("run", 2**32), ("model", -5), ("model", 2**40)):
            with self.subTest(table=table, seed=bad):
                raw = self._mutate(lambda r: r[table].update(seed=bad))
                with self.assertRaisesRegex(ConfigError, rf"{table}\.seed"):
                    moe_config.parse(raw)
        for ok in (0, 2**32 - 1):
            raw = self._mutate(lambda r: r["run"].update(seed=ok))
            self.assertEqual(moe_config.parse(raw).seed, ok)

    def test_tiny_max_positions_must_cover_longest_sequence(self):
        # sft.max_length 26 and longest prompt 12 + max_completion_length 14 = 26
        raw = self._mutate(lambda r: r["model"].update(max_positions=25))
        with self.assertRaisesRegex(ConfigError, "max_positions"):
            moe_config.parse(raw)
        raw = self._mutate(lambda r: r["model"].update(max_positions=26))
        moe_config.parse(raw)

    def test_max_positions_review_example_rejected_even_without_rl(self):
        def mutate(r):
            r["model"]["max_positions"] = 15
            r["moe"]["sft"]["max_length"] = 26
            r["moe"]["rl"].update(max_prompt_length=1, max_completion_length=14, algorithms=[])

        with self.assertRaisesRegex(ConfigError, "max_positions"):
            moe_config.parse(self._mutate(mutate))

    def test_max_prompt_length_must_cover_task_prompts_when_rl_runs(self):
        raw = self._mutate(lambda r: r["moe"]["rl"].update(max_prompt_length=11))
        with self.assertRaisesRegex(ConfigError, "max_prompt_length"):
            moe_config.parse(raw)
        raw = self._mutate(lambda r: r["moe"]["rl"].update(max_prompt_length=11, algorithms=[]))
        moe_config.parse(raw)

    def test_gates_subspace_rank(self):
        self.assertEqual(moe_config.load(CONFIGS / "stage-moe-smoke.toml").gates.subspace_rank, 4)
        for bad in (0, 64, 100):  # smoke: min(hidden, intermediate) = 64
            raw = self._mutate(lambda r: r["moe"].setdefault("gates", {}).update(subspace_rank=bad))
            with self.assertRaisesRegex(ConfigError, r"moe\.gates\.subspace_rank"):
                moe_config.parse(raw)
        raw = self._mutate(lambda r: r["moe"].setdefault("gates", {}).update(subspace_rnk=2))
        with self.assertRaisesRegex(ConfigError, r"\[moe\.gates\]"):
            moe_config.parse(raw)
        raw = self._mutate(lambda r: r["moe"].pop("gates"))
        self.assertEqual(moe_config.parse(raw).gates.subspace_rank, 4)

    def test_allow_unselected_backend_flag(self):
        self.assertTrue(moe_config.load(CONFIGS / "stage-moe-smoke.toml").allow_unselected_backend)
        self.assertFalse(moe_config.load(CONFIGS / "stage-moe-olmoe.toml").allow_unselected_backend)
        raw = self._mutate(lambda r: r["training"].update(allow_unselected_backend="yes"))
        with self.assertRaisesRegex(ConfigError, "allow_unselected_backend"):
            moe_config.parse(raw)

    def test_head_dim_must_be_even(self):
        raw = self._mutate(lambda r: r["model"].update(hidden=60, heads=4))  # head dim 15
        with self.assertRaisesRegex(ConfigError, "head dim"):
            moe_config.parse(raw)

    def test_lora_dropout_range(self):
        for bad in (-0.1, 1.0, 2.0):
            with self.subTest(dropout=bad):
                raw = self._mutate(lambda r: r["moe"].update(lora={"enabled": False, "dropout": bad}))
                with self.assertRaisesRegex(ConfigError, r"moe\.lora\.dropout"):
                    moe_config.parse(raw)
        raw = self._mutate(lambda r: r["moe"].update(lora={"enabled": False, "dropout": 0.5}))
        self.assertEqual(moe_config.parse(raw).lora.dropout, 0.5)

    def test_nonfinite_floats_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            for table, key in (("sft", "lr"), ("rl", "lr"), ("rl", "temperature"),
                               ("rl", "beta"), ("rl", "epsilon"), ("rl", "gspo_epsilon_high")):
                with self.subTest(key=key, value=bad):
                    raw = self._mutate(lambda r: r["moe"][table].update({key: bad}))
                    with self.assertRaisesRegex(ConfigError, rf"moe\.{table}\.{key}"):
                        moe_config.parse(raw)
        raw = self._mutate(lambda r: r["model"].update(router_aux_loss_coef=float("nan")))
        with self.assertRaisesRegex(ConfigError, "router_aux_loss_coef"):
            moe_config.parse(raw)
        raw = self._mutate(lambda r: r["moe"].update(lora={"enabled": False, "dropout": float("inf")}))
        with self.assertRaisesRegex(ConfigError, "dropout"):
            moe_config.parse(raw)

    def test_lora_targets_must_be_string_lists(self):
        for key in ("target_modules", "target_parameters"):
            for bad in ("q_proj", 3, ["q_proj", 1], [""]):
                with self.subTest(key=key, bad=bad):
                    raw = self._mutate(
                        lambda r: r["moe"].update(lora={"enabled": True, "target_modules": ["q"], key: bad})
                    )
                    with self.assertRaisesRegex(ConfigError, rf"moe\.lora\.{key}"):
                        moe_config.parse(raw)

    def test_sft_max_length_must_leave_a_label(self):
        from ptr_training.moe.tasks import longest_prompt_chars

        from ptr_training.moe.tasks import longest_answer_chars

        need = longest_prompt_chars() + longest_answer_chars() + 1  # prompt + answer + EOS
        self.assertEqual(need, 26)
        raw = self._mutate(lambda r: r["moe"]["sft"].update(max_length=need - 1))
        with self.assertRaisesRegex(ConfigError, r"moe\.sft\.max_length"):
            moe_config.parse(raw)
        raw = self._mutate(lambda r: r["moe"]["sft"].update(max_length=need))
        moe_config.parse(raw)

    def test_max_completion_length_must_fit_longest_answer(self):
        from ptr_training.moe.tasks import longest_answer_chars

        need = longest_answer_chars() + 1
        self.assertEqual(need, 14)  # {"a":9,"b":9} + EOS
        raw = self._mutate(lambda r: r["moe"]["rl"].update(max_completion_length=need - 1))
        with self.assertRaisesRegex(ConfigError, r"moe\.rl\.max_completion_length"):
            moe_config.parse(raw)
        raw = self._mutate(lambda r: r["moe"]["rl"].update(max_completion_length=need))
        moe_config.parse(raw)

    def test_pretrained_rejects_tiny_only_keys(self):
        for key, value in (("layers", 2), ("hidden", 64), ("intermediate", 64), ("heads", 4),
                           ("experts", 8), ("top_k", 2), ("max_positions", 64),
                           ("router_aux_loss_coef", 0.01)):
            with self.subTest(key=key):
                raw = _raw("stage-moe-olmoe.toml")
                raw["model"][key] = value
                with self.assertRaisesRegex(ConfigError, key):
                    moe_config.parse(raw)

    def test_min_verification_level_default_and_validation(self):
        from ptr_training.moe import reward

        cfg = moe_config.load(CONFIGS / "stage-moe-smoke.toml")
        self.assertEqual(cfg.rl.min_verification_level, reward.LEVEL)
        raw = self._mutate(lambda r: r["moe"]["rl"].update(min_verification_level="SampleVerified"))
        self.assertEqual(moe_config.parse(raw).rl.min_verification_level, "SampleVerified")
        raw = self._mutate(lambda r: r["moe"]["rl"].update(min_verification_level="Strong"))
        with self.assertRaisesRegex(ConfigError, "moe.rl.min_verification_level"):
            moe_config.parse(raw)

    def test_epsilon_high_below_epsilon(self):
        raw = self._mutate(lambda r: r["moe"]["rl"].update(epsilon=0.3, epsilon_high=0.2))
        with self.assertRaisesRegex(ConfigError, "epsilon_high"):
            moe_config.parse(raw)


class MoeManifestCompatTests(unittest.TestCase):
    def test_manifest_builds_for_both_stage_configs(self):
        from ptr_training.run import ROOT, build_manifest

        for name in ("stage-moe-smoke.toml", "stage-moe-olmoe.toml"):
            manifest = build_manifest(ROOT / "training" / "configs" / name)
            self.assertEqual(manifest["config"]["training"]["backend"], "trl-moe")
            self.assertEqual(len(manifest["uv_lock_sha256"]), 64)


class MoeDispatchTests(unittest.TestCase):
    def test_missing_extra_gives_clear_message(self):
        from unittest import mock

        from ptr_training import run
        from ptr_training.moe import pipeline

        with mock.patch.object(
            pipeline, "require_extra", side_effect=pipeline.MissingExtraError("needs the moe extra")
        ):
            self.assertEqual(run._execute_trl_moe({}, Path("x.toml")), "needs the moe extra")

    def test_require_extra_names_the_extra_when_stack_absent(self):
        from unittest import mock

        from ptr_training.moe import pipeline

        from importlib import metadata

        # find_spec would still succeed for `datasets` (repo-level namespace dir); the check
        # must go through distribution metadata.
        with mock.patch.object(
            pipeline.metadata, "version", side_effect=metadata.PackageNotFoundError("x")
        ):
            with self.assertRaisesRegex(pipeline.MissingExtraError, "moe"):
                pipeline.require_extra()

    def test_moe_minimums_match_pyproject(self):
        import re
        import tomllib

        from ptr_training.moe import pipeline

        pyproject = tomllib.loads(
            (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
        )
        declared = {}
        for spec in pyproject["project"]["optional-dependencies"]["moe"]:
            name, minimum, upper = re.fullmatch(
                r"([A-Za-z0-9_.-]+)>=([\w.]+),<([\w.]+)", spec
            ).groups()
            declared[name] = (minimum, upper)
        self.assertEqual(
            {k: (v, pipeline.MOE_UPPER_BOUNDS[k]) for k, v in pipeline.MOE_MINIMUMS.items()},
            declared,
        )

    def test_require_extra_rejects_too_old_and_accepts_minimum(self):
        from unittest import mock

        from ptr_training.moe import pipeline

        def installed(overrides):
            return lambda dist: overrides.get(dist, pipeline.MOE_MINIMUMS[dist])

        with mock.patch.object(pipeline.metadata, "version", side_effect=installed({})):
            pipeline.require_extra()  # exactly the minimums
        newer = installed({"trl": "1.14.9", "torch": "2.14.1+cpu"})
        with mock.patch.object(pipeline.metadata, "version", side_effect=newer):
            pipeline.require_extra()
        old = installed({"peft": "0.21.1", "trl": "0.9"})
        with mock.patch.object(pipeline.metadata, "version", side_effect=old):
            with self.assertRaises(pipeline.MissingExtraError) as ctx:
                pipeline.require_extra()
        message = str(ctx.exception)
        self.assertIn("peft 0.21.1 < 0.21.2", message)
        self.assertIn("trl 0.9 < 1.14.1", message)
        self.assertNotIn("torch", message)
        new = installed({"trl": "1.15.0", "torch": "3.0.0"})
        with mock.patch.object(pipeline.metadata, "version", side_effect=new):
            with self.assertRaisesRegex(pipeline.MissingExtraError, "too new.*trl 1.15.0 >= 1.15"):
                pipeline.require_extra()

    def test_version_comparator_without_packaging(self):
        import builtins
        from unittest import mock

        from ptr_training.moe import pipeline

        real_import = builtins.__import__

        def no_packaging(name, *a, **k):
            if name.startswith("packaging"):
                raise ImportError(name)
            return real_import(name, *a, **k)

        with mock.patch.object(builtins, "__import__", no_packaging):
            self.assertTrue(pipeline.version_at_least("2.14.1+cpu", "2.14.1"))
            self.assertTrue(pipeline.version_at_least("2.14", "2.14.0"))
            self.assertTrue(pipeline.version_at_least("10.0.0", "9.9"))
            self.assertFalse(pipeline.version_at_least("5.17.9", "5.18.0"))

    def test_require_extra_reports_only_missing_distribution(self):
        from importlib import metadata
        from unittest import mock

        from ptr_training.moe import pipeline

        def version(dist):
            if dist == "datasets":
                raise metadata.PackageNotFoundError(dist)
            return pipeline.MOE_MINIMUMS[dist]

        with mock.patch.object(pipeline.metadata, "version", side_effect=version):
            with self.assertRaisesRegex(pipeline.MissingExtraError, r"missing: datasets\)"):
                pipeline.require_extra()


if __name__ == "__main__":
    unittest.main()
