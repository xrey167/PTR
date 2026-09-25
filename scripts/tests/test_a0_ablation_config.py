"""The A0 ablation study's arm table exists in three places, which must agree.

model/burn-a0/examples/a0_ablation/arms.rs is what runs;
model/configs/a0_ablation_study.toml is what the preregistration freezes; and
model/configs/ablations.toml maps the M00x hypotheses onto the arms. The two
TOML files are compared on every run. The binary's `--phase list-arms` output is
compared when the study binary is built (the study's gate G6 builds it first);
CI's Python job does not build it, so that one comparison is skipped there.
"""

import json
import subprocess
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "model/configs/a0_ablation_study.toml"
ABLATIONS = ROOT / "model/configs/ablations.toml"
BINARY = ROOT / "model/burn-a0/target-a0-study/release/examples/a0_ablation"

SWITCHES = ["typed_attention", "typed_query", "latent_nonlinearity", "frozen_router"]


def study_arms() -> dict[str, dict]:
    """Load study arm definitions indexed by their configured names."""
    return {arm["name"]: arm for arm in tomllib.loads(STUDY.read_text(encoding="utf-8"))["arm"]}


class TheTomlFilesAgree(unittest.TestCase):
    def test_every_arm_but_the_reference_realises_exactly_one_declared_ablation(self):
        """Require each non-reference arm to map to exactly one applicable ablation."""
        arms = study_arms()
        mapped: dict[str, str] = {}
        for ablation in tomllib.loads(ABLATIONS.read_text(encoding="utf-8"))["ablation"]:
            self.assertIn(ablation["a0_status"], {"run", "negative-control", "not-applicable"})
            if ablation["a0_status"] == "not-applicable":
                self.assertEqual(ablation["a0_arms"], [], ablation["id"])
            for arm in ablation["a0_arms"]:
                self.assertNotIn(arm, mapped, f"{arm} is under two ablations")
                mapped[arm] = ablation["id"]
        self.assertEqual(set(mapped), set(arms) - {"full"})

    def test_the_reference_arm_is_the_unswitched_model_at_two_latent_steps(self):
        """Pin the full arm's default switches, typed batch, and two latent steps."""
        full = study_arms()["full"]
        self.assertEqual(
            {key: full[key] for key in SWITCHES + ["latent_steps", "batch"]},
            {
                "typed_attention": True,
                "typed_query": True,
                "latent_nonlinearity": True,
                "frozen_router": False,
                "latent_steps": 2,
                "batch": "typed",
            },
        )


@unittest.skipUnless(BINARY.exists(), "the study binary is not built (cargo --release into target-a0-study)")
class TheBinaryAgrees(unittest.TestCase):
    def test_list_arms_equals_the_study_config(self):
        """Compare the built study binary's arm listing with the preregistered TOML definitions."""
        listed = json.loads(
            subprocess.run(
                [str(BINARY), "--phase", "list-arms"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        on = {True: "on", False: "off"}
        expected = [
            {
                "arm": arm["name"],
                "experiment": arm["experiment"],
                "tier": str(arm["tier"]),
                **{key: on[arm[key]] for key in SWITCHES},
                "latent_steps": str(arm["latent_steps"]),
                "batch": arm["batch"],
            }
            for arm in tomllib.loads(STUDY.read_text(encoding="utf-8"))["arm"]
        ]
        normalise = lambda arms: sorted((json.dumps(arm, sort_keys=True) for arm in arms))
        self.assertEqual(normalise(listed), normalise(expected))


if __name__ == "__main__":
    unittest.main()
