import importlib.util
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("generate_sbom",ROOT/"scripts/generate_sbom.py")
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class SbomTests(unittest.TestCase):
    def test_cyclonedx_contains_components_and_dependencies(self):
        metadata={
            "packages":[
                {"id":"a 0.1.0 (path+file:///a)","name":"a","version":"0.1.0","license":"MIT","description":None,"repository":None},
                {"id":"b 1.2.3 (registry+https://example.invalid)","name":"b","version":"1.2.3","license":"Apache-2.0","description":None,"repository":None},
            ],
            "workspace_members":["a 0.1.0 (path+file:///a)"],
            "resolve":{"nodes":[
                {"id":"a 0.1.0 (path+file:///a)","dependencies":["b 1.2.3 (registry+https://example.invalid)"]},
                {"id":"b 1.2.3 (registry+https://example.invalid)","dependencies":[]},
            ]},
        }
        bom=mod.build(metadata)
        self.assertEqual(bom["bomFormat"],"CycloneDX")
        self.assertEqual(bom["specVersion"],"1.5")
        self.assertEqual(len(bom["components"]),2)
        a_ref=mod.purl("a","0.1.0")
        b_ref=mod.purl("b","1.2.3")
        edge=next(item for item in bom["dependencies"] if item["ref"]==a_ref)
        self.assertEqual(edge["dependsOn"],[b_ref])

if __name__=="__main__":
    unittest.main()
