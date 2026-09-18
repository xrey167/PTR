import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

class SiteTests(unittest.TestCase):
    def test_index_links_to_versioned_architecture_and_status(self):
        html=(ROOT/"index.html").read_text(encoding="utf-8")
        self.assertIn("docs/TECHNICAL_ARCHITECTURE.md",html)
        self.assertIn("docs/components/STATUS.md",html)
        self.assertIn("github.com/xrey167/PTR",html)

if __name__=="__main__":
    unittest.main()
