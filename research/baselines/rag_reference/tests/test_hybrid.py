import importlib.util
import sys
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from hybrid import HybridRAG

class HybridRAGTests(unittest.TestCase):
    def test_hybrid_update_delete(self):
        rag=HybridRAG()
        rag.upsert("a","red apple",[1.0,0.0])
        rag.upsert("b","blue ocean",[0.0,1.0])
        self.assertEqual(rag.search("apple",[1.0,0.0],1)[0][0],"a")
        rag.delete("a")
        self.assertNotIn("a",dict(rag.search("apple",[1.0,0.0],10)))
