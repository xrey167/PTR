"""Keep root-level cargo-deny options ahead of its check subcommand."""
import importlib.util
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "security_cli", Path(__file__).resolve().parents[1] / "security_workspaces.py"
)
SECURITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SECURITY)


class ScannerCommandTests(unittest.TestCase):
    def test_deny_uses_explicit_root_policy_before_check(self):
        root = Path("repo")
        manifest = root / "model/burn-a0/Cargo.toml"
        self.assertEqual(SECURITY.scanner_command("deny", manifest, root), [
            "cargo", "+stable", "deny", "--format", "json", "--manifest-path",
            str(manifest), "--config", str(root / "deny.toml"), "check",
        ])

    def test_audit_reads_the_selected_workspaces_lockfile(self):
        root = Path("repo")
        manifest = root / "fuzz/Cargo.toml"
        self.assertEqual(SECURITY.scanner_command("audit", manifest, root), [
            "cargo", "+stable", "audit", "--file", str(root / "fuzz/Cargo.lock"), "--json",
        ])

    def test_unknown_scanner_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported security scanner"):
            SECURITY.scanner_command("skip", Path("Cargo.toml"), Path("."))


if __name__ == "__main__":
    unittest.main()
