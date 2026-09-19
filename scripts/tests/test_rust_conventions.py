import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "check_rust_conventions", ROOT / "scripts/check_rust_conventions.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class RustConventionTests(unittest.TestCase):
    def test_check_functions_require_result_and_common_contains_no_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crate = root / "crates" / "ptr-demo"
            src = crate / "src"
            common = crate / "tests" / "common"
            src.mkdir(parents=True)
            common.mkdir(parents=True)
            (crate / "Cargo.toml").write_text("[package]\nname='ptr-demo'\n", encoding="utf-8")
            (src / "lib.rs").write_text(
                "pub fn check_name(value: &str) -> Result<(), String> { "
                "if value.is_empty() { Err(\"empty\".into()) } else { Ok(()) } }\n",
                encoding="utf-8",
            )
            (common / "mod.rs").write_text("pub fn fixture() -> u8 { 1 }\n", encoding="utf-8")

            original_root = mod.ROOT
            mod.ROOT = root
            try:
                self.assertEqual(mod.check(), [])
            finally:
                mod.ROOT = original_root

    def test_invalid_check_signature_and_common_test_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crate = root / "crates" / "ptr-demo"
            src = crate / "src"
            common = crate / "tests" / "common"
            src.mkdir(parents=True)
            common.mkdir(parents=True)
            (crate / "Cargo.toml").write_text("[package]\nname='ptr-demo'\n", encoding="utf-8")
            (src / "lib.rs").write_text(
                "pub fn validate_name(value: &str) -> bool { !value.is_empty() }\n",
                encoding="utf-8",
            )
            (common / "mod.rs").write_text(
                "#[test]\nfn helper_is_not_a_helper() {}\n",
                encoding="utf-8",
            )

            original_root = mod.ROOT
            mod.ROOT = root
            try:
                errors = mod.check()
            finally:
                mod.ROOT = original_root

            self.assertTrue(any("validate_name must return Result" in error for error in errors))
            self.assertTrue(any("must not contain test annotations" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
