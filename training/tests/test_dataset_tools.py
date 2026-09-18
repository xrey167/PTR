import tempfile
import unittest
from pathlib import Path

from ptr_training.validate_dataset import validate

class DatasetToolsTests(unittest.TestCase):
    def test_validate_jsonl_counts_bad_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            path.write_text('{"ok": 1}\nnot-json\n', encoding="utf-8")
            ok, bad = validate(path)
            self.assertEqual((ok, bad), (1, 1))

if __name__ == "__main__":
    unittest.main()
