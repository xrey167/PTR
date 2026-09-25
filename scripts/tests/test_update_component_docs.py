"""The generated test counts must see async tests and ignore quoted attributes.

The generator counted the substring `#[test]`, so every `#[tokio::test]` was
invisible: ptr-server's README said 1 test where it had 4, and ptr-cluster's
integration tests did not show up at all.
"""

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "update_component_docs", ROOT / "scripts/update_component_docs.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class TestCounting(unittest.TestCase):
    def test_sync_and_async_attributes_count_and_quoted_ones_do_not(self):
        """Count real sync and async test attributes while excluding quoted text and other macros."""
        directory = tempfile.TemporaryDirectory()
        crate = Path(directory.name)
        with directory:
            (crate / "src").mkdir()
            (crate / "tests").mkdir()
            (crate / "src/lib.rs").write_text(
                "pub fn f() {}\n"
                "#[cfg(test)]\n"
                "mod tests {\n"
                "    #[test]\n"
                "    fn a() {}\n"
                "    // A `#[test]` quoted in a comment is not a test.\n"
                '    const S: &str = "#[test]";\n'
                "}\n",
                encoding="utf-8",
            )
            (crate / "tests/it.rs").write_text(
                "#[tokio::test]\n"
                "async fn b() {}\n"
                '#[tokio::test(flavor = "multi_thread", worker_threads = 4)]\n'
                "async fn c() {}\n"
                "#[test] // trailing comment\n"
                "fn d() {}\n"
                "#[test_case(1)]\n"
                "fn not_counted(_: u8) {}\n"
                "/// Doc text naming #[test] and #[test] is not a test either.\n"
                "fn helper() {}\n",
                encoding="utf-8",
            )
            metrics = mod.code_metrics(crate)
        # Four attribute lines. The old substring count saw six here (two sync
        # attributes plus four quotations of the attribute) and none of the async
        # ones, so it cannot pass this by accident.
        self.assertEqual(metrics["tests"], 4)
        self.assertEqual(metrics["files"], 1)
        self.assertEqual(metrics["test_files"], 1)

    def test_async_crates_are_no_longer_undercounted(self):
        """Require the real server crate's metrics to include its async tests."""
        server = mod.code_metrics(ROOT / "crates/ptr-server")["tests"]
        self.assertGreaterEqual(server, 4)


if __name__ == "__main__":
    unittest.main()
