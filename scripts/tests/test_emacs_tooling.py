import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class EmacsToolingTests(unittest.TestCase):
    def test_ptr_emacs_commands_preserve_repository_contracts(self):
        text = (ROOT / "tooling/emacs/ptr-rust.el").read_text(encoding="utf-8")

        required = [
            "check --workspace --all-targets --locked",
            "test --workspace --locked",
            "clippy --workspace --all-targets -- -D warnings",
            "fmt --all -- --check",
            "python scripts/check_repo.py",
            "python scripts/check_architecture_catalog.py",
            "python scripts/check_rust_conventions.py",
            "python scripts/run_experiment.py validate",
            "python scripts/run_component_eval.py validate",
        ]
        for command in required:
            self.assertIn(command, text)

    def test_emacs_candidates_are_registered_and_unlocked(self):
        registry = tomllib.loads(
            (ROOT / "evaluations/registry.toml").read_text(encoding="utf-8")
        )
        ids = {item["id"] for item in registry.get("component", [])}
        self.assertIn("rust-editor-emacs", ids)

        candidates = tomllib.loads(
            (
                ROOT
                / "evaluations/components/rust-editor-emacs/candidates.toml"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(candidates["decision"], "unlocked")
        candidate_ids = {item["id"] for item in candidates.get("candidate", [])}
        self.assertEqual(
            candidate_ids,
            {
                "rust-mode",
                "rust-mode+cargo-mode",
                "rust-mode+rustic",
                "rustic+cargo-mode",
            },
        )


if __name__ == "__main__":
    unittest.main()
