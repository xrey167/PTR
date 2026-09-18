import tomllib
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

class StrongRagConfigTests(unittest.TestCase):
    def test_blocked_config_cannot_be_mistaken_for_ready_baseline(self):
        data=tomllib.loads((ROOT/"config.toml").read_text(encoding="utf-8"))
        self.assertTrue(data["status"].startswith("blocked-"))
        self.assertEqual(data["dense"]["revision"], "")
        self.assertEqual(data["reranker"]["revision"], "")
        self.assertEqual(data["answer"]["revision"], "")

if __name__=="__main__":
    unittest.main()
