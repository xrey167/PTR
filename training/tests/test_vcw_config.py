import copy
import tomllib
import unittest
from pathlib import Path

from ptr_training import run as run_module
from ptr_training.moe import config as moe_config
from ptr_training.moe.config import ConfigError
from ptr_training.vcw import config as vcw_config
from ptr_training.vcw import plan

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "training" / "configs"
PROFILES = {"rtx4090": "rtx4090-reference", "rtx5090": "rtx5090-reference"}
STAGES = ("perft-r-sft", "rlvr-router", "repair-expert", "self-model-probe")


def _raw(stage="rlvr-router", profile="rtx4090") -> dict:
    return tomllib.loads((CONFIGS / f"vcw-{stage}-{profile}.toml").read_text(encoding="utf-8"))


def _parse(raw):
    return vcw_config.parse(raw, ROOT)


class VcwConfigFilesTests(unittest.TestCase):
    def test_all_eight_configs_exist_and_load(self):
        for stage in STAGES:
            for profile in PROFILES:
                with self.subTest(stage=stage, profile=profile):
                    cfg = vcw_config.load(CONFIGS / f"vcw-{stage}-{profile}.toml", ROOT)
                    self.assertEqual(cfg.stage, stage)
                    self.assertEqual(cfg.profile_id, PROFILES[profile])
                    self.assertTrue(cfg.estimate_hours[0] > cfg.estimate_hours[2] > 0)
                    self.assertLessEqual(cfg.estimate_vram_gib, 32 - 1.5)

    def test_4090_configs_leave_headroom_on_24_gib(self):
        for stage in STAGES:
            cfg = vcw_config.load(CONFIGS / f"vcw-{stage}-rtx4090.toml", ROOT)
            self.assertLessEqual(cfg.estimate_vram_gib, 22.5, stage)

    def test_configs_match_the_plan_protocol(self):
        for profile, pid in PROFILES.items():
            p = plan.PROTOCOL[pid]
            sft = vcw_config.load(CONFIGS / f"vcw-perft-r-sft-{profile}.toml", ROOT).moe
            self.assertEqual((sft.sft.steps, sft.sft.batch, sft.sft.gradient_accumulation_steps, sft.sft.max_length),
                             (p["sft"]["steps"], p["sft"]["micro_batch"], p["sft"]["accumulation"], p["sft"]["seq_tokens"]))
            rl = vcw_config.load(CONFIGS / f"vcw-rlvr-router-{profile}.toml", ROOT).moe.rl
            self.assertEqual((rl.steps, rl.prompts_per_step, rl.num_generations, rl.mini_steps, rl.max_prompt_length, rl.max_completion_length),
                             (p["rl"]["steps"], p["rl"]["prompts"], p["rl"]["generations"], p["rl"]["mini_steps"], p["rl"]["prompt_tokens"], p["rl"]["completion_tokens"]))
            rep = vcw_config.load(CONFIGS / f"vcw-repair-expert-{profile}.toml", ROOT).moe.sft
            self.assertEqual((rep.steps, rep.batch, rep.gradient_accumulation_steps, rep.max_length),
                             (p["repair"]["steps"], p["repair"]["micro_batch"], p["repair"]["accumulation"], p["repair"]["seq_tokens"]))
            probe = _raw("self-model-probe", profile)["vcw"]["probe"]
            self.assertEqual((probe["micro_batch"], probe["seq_tokens"]), (p["probe"]["micro_batch"], p["probe"]["seq_tokens"]))

    def test_pins_agree_with_the_m010_m011_m012_manifests(self):
        for exp in ("M010-moe-rl-stability", "M011-routed-peft", "M012-lora-expert-balancing"):
            pinned = tomllib.loads((ROOT / "experiments" / "model" / exp / "config.toml").read_text(encoding="utf-8"))["pinned_thresholds"]
            for stage in STAGES:
                pins = _raw(stage)["vcw"]["pins"]
                self.assertEqual(pins["model_revision"], pinned["model_checkpoint_revision"], (exp, stage))
                self.assertEqual(pins["tokenizer_revision"], pinned["tokenizer_revision"], (exp, stage))
        m11 = tomllib.loads((ROOT / "experiments/model/M011-routed-peft/config.toml").read_text(encoding="utf-8"))["pinned_thresholds"]
        revs = {d["repo"]: d["revision"] for d in _raw("perft-r-sft")["vcw"]["pins"]["datasets"]}
        self.assertEqual(revs["openai/gsm8k"], m11["corpus_math_revision"])
        self.assertEqual(revs["openai/openai_humaneval"], m11["corpus_code_revision"])
        self.assertEqual(revs["gorilla-llm/Berkeley-Function-Calling-Leaderboard"], m11["corpus_tool_revision"])

    def test_perft_adapter_matches_the_m011_emergent_arm(self):
        m11 = tomllib.loads((ROOT / "experiments/model/M011-routed-peft/config.toml").read_text(encoding="utf-8"))["pinned_thresholds"]
        arm = m11["arm_lora"]["perft_r_emergent"]
        a = _raw("perft-r-sft")["vcw"]["adapter"]
        self.assertEqual((a["num_experts"], a["router_top_k"], a["rank"], a["alpha"], a["trainable_params"]),
                         (arm["num_experts"], arm["router_top_k"], arm["rank"], arm["alpha"], arm["trainable_params"]))

    def test_vcw_config_is_not_a_trl_moe_run(self):
        for stage in STAGES:
            raw = _raw(stage)
            self.assertEqual(raw["training"]["backend"], f"vcw-{stage}")
            with self.assertRaises(ConfigError):
                moe_config.parse(raw)

    def test_run_manifest_builds_and_names_the_vcw_backend(self):
        manifest = run_module.build_manifest(CONFIGS / "vcw-rlvr-router-rtx4090.toml")
        self.assertEqual(manifest["config"]["training"]["backend"], "vcw-rlvr-router")
        self.assertNotEqual(manifest["config"]["training"]["backend"], "trl-moe")


class VcwConfigRefusalTests(unittest.TestCase):
    def _refuses(self, fn, pattern, stage="rlvr-router", profile="rtx4090"):
        raw = copy.deepcopy(_raw(stage, profile))
        fn(raw)
        with self.assertRaisesRegex(ConfigError, pattern):
            _parse(raw)

    def test_self_report_field_is_refused_by_the_guard(self):
        self._refuses(lambda r: r["vcw"]["data"]["fields"].update(introspection="self-report"), "self-report guard")
        self._refuses(lambda r: r["vcw"]["data"]["fields"].update(note="self-report"), "marked as a self-report")
        self._refuses(lambda r: r["vcw"]["data"].update(targets=["completion"]), "targets must come from")

    def test_self_report_field_refused_in_every_stage(self):
        for stage in STAGES:
            with self.subTest(stage=stage):
                self._refuses(lambda r: r["vcw"]["data"]["fields"].update(verbal_confidence="verifier"), "self-report guard", stage)

    def test_welfare_block_is_enforced(self):
        self._refuses(lambda r: r["vcw"]["welfare"].update(self_report_training_signals=1), "exactly 0")
        self._refuses(lambda r: r["vcw"]["welfare"].update(self_report_guard=False), "self_report_guard")
        self._refuses(lambda r: r["vcw"]["welfare"].update(integrated_system=True), "owner decision")
        self._refuses(lambda r: r["vcw"]["welfare"].update(stop_rules=[]), "stop_rules")

    def test_probe_has_no_write_path(self):
        self._refuses(lambda r: r["vcw"]["probe"].update(write_paths=["verifier"]), "write_paths", "self-model-probe")
        self._refuses(lambda r: r["vcw"]["probe"].update(arms=["self_model"]), "must include", "self-model-probe")

    def test_unpinned_or_mutable_revisions_refused(self):
        for bad in ("main", "abc", "A" * 40):
            self._refuses(lambda r: r["vcw"]["pins"].update(model_revision=bad), "40-hex|differ")
        self._refuses(lambda r: r["vcw"]["pins"]["datasets"][0].update(revision="main"), "40-hex")

    def test_final_seeds_cannot_be_pilot_seeds(self):
        self._refuses(lambda r: r["vcw"]["pins"].update(pilot_seeds=[17]), "pilot_seeds")
        self._refuses(lambda r: r["vcw"]["pins"].update(seeds=[17, 29]), "seeds must be")

    def test_status_and_backend(self):
        self._refuses(lambda r: r["vcw"].update(status="measured"), "plan-not-run")
        self._refuses(lambda r: r["training"].update(backend="trl-moe"), "backend")
        self._refuses(lambda r: r["training"].update(allow_unselected_backend=True), "may only hold")

    def test_unknown_keys_rejected(self):
        self._refuses(lambda r: r["vcw"].update(extra={}), "unknown key")
        self._refuses(lambda r: r["vcw"]["estimate"].update(extra=1), "unknown key")

    def test_reward_contract_is_enforced(self):
        self._refuses(lambda r: r["vcw"]["verifier"].update(learned_score_in_reward=True), "invariant 11")
        self._refuses(lambda r: r["vcw"]["verifier"].update(report_score_for_non_pass=True), "invariant 11")
        self._refuses(lambda r: r["vcw"]["verifier"].update(non_pass_reward=0.5), "non_pass_reward")
        self._refuses(lambda r: r["vcw"]["verifier"].update(gate="combined-level"), "weakest-level")
        self._refuses(lambda r: r["vcw"]["verifier"]["min_level"].update(code="LatentAgreement"), "weakest|Latent")
        self._refuses(lambda r: r["moe"]["rl"].update(min_verification_level="Deterministic"), "must equal the weakest")

    def test_rl_arms_are_gspo_and_sapo(self):
        self._refuses(lambda r: r["moe"]["rl"].update(algorithms=["gspo"]), "gspo")

    def test_routing_metrics_required(self):
        self._refuses(lambda r: r["vcw"]["routing"]["metrics"].remove("crash_step"), "missing")

    def test_estimate_is_recomputed_and_must_fit(self):
        self._refuses(lambda r: r["vcw"]["estimate"].update(vram_gib=10.0), "recomputes")
        self._refuses(lambda r: r["vcw"]["estimate"].update(wall_hours_mid=0.01), "recomputes")
        self._refuses(lambda r: r["moe"]["rl"].update(num_generations=16, mini_steps=2), "headroom|leaves less")
        self._refuses(lambda r: r["moe"]["sft"].update(gradient_checkpointing=False), "headroom|leaves less|recomputes", "perft-r-sft")

    def test_adapter_arithmetic(self):
        self._refuses(lambda r: r["vcw"]["adapter"].update(trainable_params=1), "PERFT-R arithmetic")
        self._refuses(lambda r: r["vcw"]["adapter"].update(base_router_frozen=False), "frozen")

    def test_repair_experts_match_total_rank_and_route_by_code(self):
        self._refuses(lambda r: r["vcw"]["repair"].update(generic_rank=8), "total rank", "repair-expert")
        self._refuses(lambda r: r["vcw"]["repair"].update(routing="learned"), "finding-code", "repair-expert")
        self._refuses(lambda r: r["vcw"]["repair"]["arms"].remove("shuffled_map"), "shuffled_map", "repair-expert")

    def test_profile_must_match(self):
        self._refuses(lambda r: r["vcw"]["estimate"].update(profile="rtx5090-reference"), "profile")
        self._refuses(lambda r: r["run"].update(hardware_profile="hardware/nope.toml"), "does not exist")

    def test_only_the_pretrained_model(self):
        def tiny(r):
            r["model"].update(kind="tiny-random")
        self._refuses(tiny, "pretrained")


if __name__ == "__main__":
    unittest.main()


class VcwAuditTests(unittest.TestCase):
    def test_audit_is_zero_on_the_repository_configs(self):
        paths = sorted(CONFIGS.glob("vcw-*.toml"))
        report = vcw_config.audit(paths)
        self.assertEqual(report["configs"], 8)
        self.assertEqual(report["self_report_training_signals"], 0, report["refused"])

    def test_audit_counts_a_tampered_config(self):
        import tempfile

        text = (CONFIGS / "vcw-rlvr-router-rtx4090.toml").read_text(encoding="utf-8")
        text = text.replace('[vcw.data.fields]\n', '[vcw.data.fields]\nself_report = "verifier"\n')
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "vcw-bad.toml"
            bad.write_text(text, encoding="utf-8")
            report = vcw_config.audit([bad])
        self.assertEqual(report["self_report_training_signals"], 1)
        self.assertIn("self-report", list(report["refused"].values())[0])
