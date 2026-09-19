from __future__ import annotations
import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'check_component_metadata.py'
spec = importlib.util.spec_from_file_location('metadata_freshness', SCRIPT)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class MetadataFreshnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git('init', '-q')
        self.git('config', 'user.email', 'test@example.invalid')
        self.git('config', 'user.name', 'PTR regression test')
        crate = self.root / 'crates/ptr-example'
        (crate / 'src').mkdir(parents=True)
        (crate / 'src/lib.rs').write_text('pub fn value() -> u8 { 1 }\n')
        (crate / 'component.toml').write_text('version = 1\n')
        self.commit('base')
        self.base = self.git('rev-parse', 'HEAD').strip()

    def git(self, *args):
        return subprocess.check_output(['git', *args], cwd=self.root, text=True, stderr=subprocess.DEVNULL)

    def commit(self, message):
        self.git('add', '.')
        self.git('commit', '-qm', message)

    def run_check(self, base):
        with patch.object(checker, 'ROOT', self.root), patch.object(sys, 'argv', ['check', '--base', base]), contextlib.redirect_stdout(io.StringIO()):
            return checker.main()

    def test_missing_base_is_a_failure_not_a_skipped_success(self):
        self.assertEqual(self.run_check('not-an-existing-ref'), 1)

    def test_unavailable_git_is_a_failure(self):
        with patch.object(checker, 'git', side_effect=FileNotFoundError('git')):
            self.assertEqual(self.run_check(self.base), 1)

    def test_earlier_source_change_cannot_hide_behind_a_docs_commit(self):
        (self.root / 'crates/ptr-example/src/lib.rs').write_text('pub fn value() -> u8 { 2 }\n')
        self.commit('source without metadata')
        (self.root / 'README.md').write_text('unrelated follow-up\n')
        self.commit('later docs')
        self.assertEqual(self.run_check(self.base), 1)

    def test_actual_metadata_update_passes_full_range(self):
        (self.root / 'crates/ptr-example/src/lib.rs').write_text('pub fn value() -> u8 { 2 }\n')
        (self.root / 'crates/ptr-example/component.toml').write_text('version = 2\n')
        self.commit('source and metadata')
        self.assertEqual(self.run_check(self.base), 0)


if __name__ == '__main__':
    unittest.main()
