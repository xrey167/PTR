import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "report_rust_api", ROOT / "scripts/report_rust_api.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


class RustApiReportTests(unittest.TestCase):
    def test_visibility_parser(self):
        self.assertEqual(mod.visibility(None), "private")
        self.assertEqual(mod.visibility("pub "), "pub")
        self.assertEqual(mod.visibility("pub(crate) "), "pub(crate)")

    def test_scans_functions_types_modules_and_pub_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            crate = Path(tmp) / "ptr-demo"
            src = crate / "src"
            src.mkdir(parents=True)
            (src / "lib.rs").write_text(
                "mod internal;\n"
                "pub mod public_mod;\n"
                "pub use internal::Thing;\n"
                "#[derive(Debug)]\n"
                "pub struct Request<T> { pub value: T }\n"
                "pub(crate) enum State { Ready, Closed }\n"
                "pub fn run<T>(request: Request<T>) -> Option<T> { Some(request.value) }\n"
                "fn helper() {}\n",
                encoding="utf-8",
            )
            items = mod.scan_crate(crate)
            found = {(item.visibility, item.kind, item.name) for item in items}
            self.assertIn(("private", "mod", "internal"), found)
            self.assertIn(("pub", "mod", "public_mod"), found)
            self.assertIn(("pub", "use", "internal::Thing"), found)
            self.assertIn(("pub", "struct", "Request"), found)
            self.assertIn(("pub(crate)", "enum", "State"), found)
            self.assertIn(("pub", "fn", "run"), found)
            self.assertIn(("private", "fn", "helper"), found)

    def test_module_name_uses_file_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src"
            nested = src / "adapters"
            nested.mkdir(parents=True)
            file = nested / "iroh.rs"
            file.write_text("pub struct Transport;\n", encoding="utf-8")
            self.assertEqual(mod.module_name(src, file), "adapters::iroh")


if __name__ == "__main__":
    unittest.main()
