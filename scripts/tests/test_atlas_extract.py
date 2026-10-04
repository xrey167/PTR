"""The Atlas extractor must record the modules the code imports, and only those.

Four things went wrong before and each has a test here: a nested path recorded its
first segment instead of the module it names, a Rustdoc link in a comment counted as
an import, a grouped `use` tree recorded nothing, and `self`/`super` were resolved
from the wrong module. Each test also has a control that must still be recorded, so a
change that simply recorded nothing would fail.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_argv = sys.argv
sys.argv = ["extract_atlas"]  # the module reads its arguments when it is imported
try:
    spec = importlib.util.spec_from_file_location("extract_atlas", ROOT / "docs/atlas/extract_atlas.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
finally:
    sys.argv = _argv

# A package with the module tree  (root), a, a::b, c, experiments, experiments::json,
# experiments::s003::params.
MODULES = {
    (): "(root)",
    ("a",): "a",
    ("a", "b"): "b",
    ("c",): "c",
    ("experiments",): "experiments",
    ("experiments", "json"): "json",
    ("experiments", "s003", "params"): "params",
}


class ResolveUse(unittest.TestCase):
    def test_a_nested_path_names_the_module_not_its_first_segment(self):
        self.assertEqual(mod.resolve_use("crate::experiments::json::json_string", MODULES, ("c",)), "json")
        self.assertEqual(mod.resolve_use("crate::experiments::s003::params::Params", MODULES, ("c",)), "params")

    def test_the_longest_known_prefix_wins_and_a_plain_item_names_no_module(self):
        self.assertEqual(mod.resolve_use("crate::a::b::Item", MODULES, ("c",)), "b")
        self.assertEqual(mod.resolve_use("crate::a::Item", MODULES, ("c",)), "a")
        self.assertIsNone(mod.resolve_use("crate::RootItem", MODULES, ("c",)))

    def test_self_and_super_start_from_the_current_module(self):
        params = ("experiments", "s003", "params")
        # self:: starts at the current module
        self.assertEqual(mod.resolve_use("self::b::Item", MODULES, ("a",)), "b")
        # super:: starts at the parent: the parent of a::b is a, and a has no child c
        self.assertIsNone(mod.resolve_use("super::c::Item", MODULES, ("a", "b")))
        # the parent of experiments::s003::params is experiments::s003, which has a
        # child params but no child json; json lives one level higher
        self.assertEqual(mod.resolve_use("super::params::P", MODULES, params), "params")
        self.assertIsNone(mod.resolve_use("super::json::f", MODULES, params))
        self.assertEqual(mod.resolve_use("super::super::json::f", MODULES, params), "json")


class CodeOnly(unittest.TestCase):
    def test_comments_and_strings_are_removed_and_code_is_kept(self):
        source = (
            "/// see [x](crate::ghost_doc)\n"
            "// crate::ghost_line\n"
            "/* crate::ghost_block /* crate::ghost_nested */ */\n"
            'fn f<\'a>(x: &\'a str) { let _ = "crate::ghost_string"; let _ = r#"crate::ghost_raw"#; '
            "let _ = '\\''; crate::real::go(); }\n"
        )
        code = mod.code_only(source)
        for ghost in ("ghost_doc", "ghost_line", "ghost_block", "ghost_nested", "ghost_string", "ghost_raw"):
            self.assertNotIn(ghost, code)
        self.assertIn("crate::real::go()", code)
        self.assertIn("<'a>", code)  # a lifetime is not a char literal


class ExpandUse(unittest.TestCase):
    def test_groups_nest_and_rename_and_glob(self):
        self.assertEqual(
            mod.expand_use("crate::{fields, macros::impl_trace_value_from, TraceError}"),
            ["crate::fields", "crate::macros::impl_trace_value_from", "crate::TraceError"],
        )
        self.assertEqual(
            mod.expand_use("crate::a::{self, b::{c, d as e}, f::*}"),
            ["crate::a", "crate::a::b::c", "crate::a::b::d", "crate::a::f"],
        )

    def test_a_plain_path_is_returned_unchanged(self):
        self.assertEqual(mod.expand_use("super::params"), ["super::params"])


class AnalyseSource(unittest.TestCase):
    def analyse(self, text):
        return mod.analyse_source("x.rs", text, MODULES, ("c",))["uses"]

    def test_a_doc_link_is_not_an_import_but_the_same_path_in_code_is(self):
        self.assertEqual(self.analyse("/// [a](crate::a)\nfn f() {}\n"), [])
        self.assertEqual(self.analyse("fn f() { crate::a::go(); }\n"), ["a"])

    def test_a_grouped_use_records_every_module_it_names(self):
        self.assertEqual(self.analyse("use crate::{a, c::X, a::b::Y};\n"), ["a", "b", "c"])

    def test_a_relative_re_export_is_not_recorded(self):
        # Documented limit: only crate::, self:: and super:: paths are imports here.
        self.assertEqual(self.analyse("pub use a::{X, Y};\n"), [])


if __name__ == "__main__":
    unittest.main()
