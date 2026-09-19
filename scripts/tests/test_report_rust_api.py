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

    def test_scans_functions_types_modules_pub_use_and_implementors(self):
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
                "macro_rules! internal_macro { () => { 1 }; }\n"
                "#[macro_export]\n"
                "macro_rules! exported_macro { () => { 2 }; }\n"
                "pub trait Handler<T> { fn handle(&self, value: T); }\n"
                "impl<T> Handler<T> for Request<T> { fn handle(&self, _value: T) {} }\n"
                "pub(crate) enum State { Ready, Closed }\n"
                "pub fn run<'request, T>(request: &'request Request<T>) -> Option<&'request T> "
                "where T: Send { Some(&request.value) }\n"
                "fn helper() {}\n",
                encoding="utf-8",
            )
            items = mod.scan_crate(crate)
            self.assertEqual({item.file for item in items}, {"ptr-demo/src/lib.rs"})
            found = {(item.visibility, item.kind, item.name) for item in items}
            self.assertIn(("private", "mod", "internal"), found)
            self.assertIn(("pub", "mod", "public_mod"), found)
            self.assertIn(("pub", "use", "internal::Thing"), found)
            self.assertIn(("pub", "struct", "Request"), found)
            self.assertIn(("private", "macro", "internal_macro"), found)
            self.assertIn(("pub", "macro", "exported_macro"), found)
            self.assertIn(("pub", "trait", "Handler"), found)
            self.assertIn(("implementation", "impl", "Handler<T> for Request<T>"), found)
            self.assertIn(("pub(crate)", "enum", "State"), found)
            self.assertIn(("pub", "fn", "run"), found)
            self.assertIn(("private", "fn", "helper"), found)

            run = next(item for item in items if item.kind == "fn" and item.name == "run")
            self.assertIn("'request", run.signature)
            self.assertIn("where T: Send", run.signature)

    def test_repository_paths_stay_repository_relative(self):
        src = ROOT / "crates" / "ptr-types" / "src"
        self.assertEqual(
            mod.source_path("ptr-types", src, src / "lib.rs"),
            "crates/ptr-types/src/lib.rs",
        )

    def test_external_nested_paths_are_stable_across_temporary_roots(self):
        reported = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmp:
                src = Path(tmp) / "ptr-demo" / "src"
                file = src / "adapters" / "transport.rs"
                file.parent.mkdir(parents=True)
                file.write_text("pub struct Transport;\n", encoding="utf-8")
                items = mod.scan_crate(src.parent)
                self.assertEqual(items[0].module, "adapters::transport")
                reported.append(items[0].file)
        self.assertEqual(reported, ["ptr-demo/src/adapters/transport.rs"] * 2)

    def test_paths_normalize_parent_segments(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "ptr-demo" / "src"
            self.assertEqual(
                mod.source_path("ptr-demo", src, src / "nested" / ".." / "lib.rs"),
                "ptr-demo/src/lib.rs",
            )

    def test_impl_name_strips_where_clause(self):
        signature = "impl<T> Handler<T> for Thing<T> where T: Send {"
        self.assertEqual(mod.impl_name(signature), "Handler<T> for Thing<T>")

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
