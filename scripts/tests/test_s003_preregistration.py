"""The S003 preregistration table, as Python and the Rust harness read it."""
import hashlib
import sys
import tomllib
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiment_records  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments" / "semdb" / "S003-certified-branches" / "config.toml"
REGISTRY = ROOT / "experiments" / "preregistration.toml"
# The SHA-256 of the canonical text without `low_cells`, which the pilot fills.
# `s003::params::tests::the_canonical_text_of_the_frozen_table_matches_the_python_digest`
# in bins/ptr-bench holds the same digest: a change to the table changes both.
DIGEST_WITHOUT_LOW_CELLS = "470e3cd81360ab0ee7dcee090b9637a41cf2563fbbf2d88d7a6dae9bc52993a0"


class S003PreregistrationTests(unittest.TestCase):
    def table(self) -> dict:
        return tomllib.loads(CONFIG.read_text(encoding="utf-8"))["preregistration"]

    def test_the_table_has_the_digest_the_rust_harness_compiles(self):
        table = self.table()
        table.pop("low_cells")
        text = experiment_records.preregistration_canonical(table)
        self.assertEqual(hashlib.sha256(text.encode("utf-8")).hexdigest(), DIGEST_WITHOUT_LOW_CELLS)

    def test_the_table_holds_exactly_the_keys_the_registry_requires(self):
        required = tomllib.loads(REGISTRY.read_text(encoding="utf-8"))["experiment"]["S003"]["required"]
        self.assertEqual(sorted(self.table()), sorted(required))


if __name__ == "__main__":
    unittest.main()
