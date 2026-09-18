import importlib.util
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("rag_runner",HERE/"runner.py")
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

class BM25Tests(unittest.TestCase):
    def test_update_delete_and_search(self):
        idx=mod.BM25(); idx.upsert("a","red apple"); idx.upsert("b","blue ocean")
        self.assertEqual(idx.search("apple",1)[0][0],"a")
        idx.delete("a")
        self.assertFalse(idx.search("apple"))
