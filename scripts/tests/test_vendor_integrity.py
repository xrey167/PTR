"""Reject unreviewed changes to downstream packages and their legal notices."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    'vendor_integrity', Path(__file__).resolve().parents[1] / 'check_vendor_integrity.py'
)
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class VendorIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.package = self.root / 'vendor/example'
        self.package.mkdir(parents=True)
        files = {'Cargo.toml': '[package]\nname="example"\nversion="1.0.0"\n',
                 'lib.rs': 'pub fn answer() -> u32 { 42 }\n',
                 'LICENSE': 'original license\n', 'PATENTS': 'original patent notice\n'}
        for name, content in files.items():
            (self.package / name).write_text(content, encoding='utf-8')
        origins = [{'name': 'example', 'version': '1.0.0', 'sha256': 'a' * 64,
                    'upstream_files': {p.name: CHECKER.digest(p.read_bytes())
                                       for p in self.package.iterdir()}}]
        (self.root / 'vendor/ORIGINS.json').write_text(json.dumps(origins), encoding='utf-8')
        self.baseline = CHECKER.inventory(self.root)
        (self.root / 'vendor/CURRENT.json').write_text(json.dumps(self.baseline), encoding='utf-8')

    def test_exact_inventory_passes_without_writes(self):
        before = (self.root / 'vendor/CURRENT.json').read_bytes()
        CHECKER.check(self.root)
        self.assertEqual(before, (self.root / 'vendor/CURRENT.json').read_bytes())

    def test_modified_source_fails(self):
        (self.package / 'lib.rs').write_text('pub fn answer() -> u32 { 0 }\n')
        with self.assertRaises(ValueError):
            CHECKER.check(self.root)

    def test_added_or_missing_files_fail(self):
        (self.package / 'extra.rs').write_text('// not reviewed\n')
        with self.assertRaises(ValueError):
            CHECKER.check(self.root)
        (self.package / 'extra.rs').unlink()
        (self.package / 'lib.rs').unlink()
        with self.assertRaises(ValueError):
            CHECKER.check(self.root)

    def test_license_and_patent_notices_cannot_be_rebaselined(self):
        for name in ('LICENSE', 'PATENTS'):
            with self.subTest(name=name):
                path = self.package / name
                content = path.read_bytes()
                path.write_text('changed notice\n')
                with self.assertRaisesRegex(ValueError, 'legal notice'):
                    CHECKER.inventory(self.root)
                path.write_bytes(content)

    def test_package_identity_cannot_be_changed(self):
        path = self.package / 'Cargo.toml'
        path.write_text(path.read_text().replace('1.0.0', '9.9.9'))
        with self.assertRaisesRegex(ValueError, 'identity'):
            CHECKER.inventory(self.root)

    def test_unrecorded_directory_fails(self):
        (self.root / 'vendor/unrecorded').mkdir()
        with self.assertRaisesRegex(ValueError, 'unrecorded'):
            CHECKER.inventory(self.root)

    def test_unsafe_origin_path_fails(self):
        path = self.root / 'vendor/ORIGINS.json'
        origins = json.loads(path.read_text())
        origins[0]['directory'] = 'vendor/../example'
        path.write_text(json.dumps(origins))
        with self.assertRaisesRegex(ValueError, 'unsafe'):
            CHECKER.inventory(self.root)

    def test_duplicate_origin_fails(self):
        path = self.root / 'vendor/ORIGINS.json'
        origins = json.loads(path.read_text())
        path.write_text(json.dumps(origins + origins))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            CHECKER.inventory(self.root)


if __name__ == '__main__':
    unittest.main()
