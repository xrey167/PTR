import importlib.util
import tempfile
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('security_workspaces', ROOT / 'scripts/security_workspaces.py')
security = importlib.util.module_from_spec(spec)
spec.loader.exec_module(security)

class SecurityWorkspaceTests(unittest.TestCase):
    def test_all_owned_workspaces_are_scanned_and_new_workspace_is_not_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in security.EXPECTED | {'new-component/Cargo.toml', 'vendor/upstream/Cargo.toml', 'model/target/temporary/Cargo.toml'}:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('[workspace]\n')
            found = {str(x) for x in security.workspaces(root)}
            self.assertEqual(found, security.EXPECTED | {'new-component/Cargo.toml'})

    def test_missing_workspace_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'required workspaces disappeared'):
                security.workspaces(Path(tmp))

    def test_security_policy_does_not_suppress_advisories_or_accept_unknown_sources(self):
        policy = tomllib.loads((ROOT / 'deny.toml').read_text())
        self.assertEqual(policy['advisories']['ignore'], [])
        self.assertTrue(policy['graph']['all-features'])
        self.assertEqual(policy['sources']['unknown-registry'], 'deny')
        self.assertEqual(policy['sources']['unknown-git'], 'deny')
        for denied in ['GPL-3.0', 'AGPL-3.0', 'MPL-2.0']:
            self.assertNotIn(denied, policy['licenses']['allow'])
        self.assertEqual({(e['name'], e['version']) for e in policy['licenses']['exceptions']},
                         {('colored', '=3.1.1')})
        self.assertTrue(all(e['allow'] == ['MPL-2.0'] for e in policy['licenses']['exceptions']))

if __name__ == '__main__':
    unittest.main()
