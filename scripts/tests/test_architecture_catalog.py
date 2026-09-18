import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "check_architecture_catalog", ROOT / "scripts/check_architecture_catalog.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class ArchitectureCatalogTests(unittest.TestCase):
    def test_catalog_is_consistent(self):
        self.assertEqual(mod.check(), [])

    def test_all_workspace_crates_have_contract_entries(self):
        crates = {
            path.name
            for path in (ROOT / "crates").glob("ptr-*")
            if (path / "Cargo.toml").exists()
        }
        entries = {
            item["id"]
            for item in mod.load("component-contracts.toml").get("component", [])
        }
        self.assertEqual(entries, crates)

    def test_all_workspace_crates_have_rust_api_layout(self):
        crates = {
            path.name
            for path in (ROOT / "crates").glob("ptr-*")
            if (path / "Cargo.toml").exists()
        }
        entries = {
            item["id"]
            for item in mod.load("rust-api-layout.toml").get("crate", [])
        }
        self.assertEqual(entries, crates)

    def test_backend_slots_keep_candidates_replaceable(self):
        slots = mod.load("backend-slots.toml").get("slot", [])
        self.assertGreaterEqual(len(slots), 20)
        for slot in slots:
            self.assertIn("contract", slot)
            self.assertIn("open_questions", slot)


if __name__ == "__main__":
    unittest.main()
