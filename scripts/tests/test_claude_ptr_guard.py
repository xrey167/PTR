import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "Cargo.toml").is_file() and (p / "CLAUDE.md").is_file())
SCRIPT = ROOT / ".claude/plugins/ptr-dev/hooks/ptr_guard.py"
spec = importlib.util.spec_from_file_location("ptr_guard", SCRIPT)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def decision(out):
    if out is None:
        return None
    return out["hookSpecificOutput"].get("permissionDecision")


class GuardUnit(unittest.TestCase):
    def pre(self, path, tool="Write", **ti):
        os.environ["CLAUDE_PROJECT_DIR"] = str(ROOT)
        ti.setdefault("content", "x")
        return guard.run("pre", {"tool_name": tool, "cwd": str(ROOT), "tool_input": {"file_path": path, **ti}})

    def post(self, path):
        os.environ["CLAUDE_PROJECT_DIR"] = str(ROOT)
        return guard.run("post", {"cwd": str(ROOT), "tool_input": {"file_path": path}})

    def frozen_dir(self):
        d = guard.frozen_experiment_dirs(ROOT)
        self.assertTrue(d, "expected at least one frozen experiment")
        return d[0][1]

    def test_allow_crate_edit(self):
        self.assertIsNone(self.pre("crates/ptr-ledger/src/lib.rs"))

    def test_deny_vendor(self):
        out = self.pre("vendor/foo/src/lib.rs")
        self.assertEqual(decision(out), "deny")
        self.assertIn("VENDOR_PATCH_POLICY", out["hookSpecificOutput"]["permissionDecisionReason"])

    def test_deny_training_runs(self):
        self.assertEqual(decision(self.pre(str(ROOT / "training/runs/a/metrics.json"))), "deny")

    def test_deny_frozen_experiment(self):
        d = self.frozen_dir()
        out = self.pre(f"{d}/config.toml")
        self.assertEqual(decision(out), "deny")
        self.assertIn("new_experiment.py", out["hookSpecificOutput"]["permissionDecisionReason"])
        self.assertEqual(decision(self.pre("experiments/model/M001-v2-semantic-slots/config.toml")), "deny")

    def test_allow_planned_experiment(self):
        self.assertIsNone(self.pre("experiments/model/M003-latent-recurrence/config.toml"))

    def test_cargo_env(self):
        self.assertEqual(decision(self.pre(".cargo/config.toml", content="[env]\nA=\"1\"\n")), "deny")
        self.assertEqual(decision(self.pre(".cargo/config.toml", tool="Edit", new_string="  [env]\n")), "deny")
        self.assertEqual(decision(self.pre(".cargo/config.toml", tool="MultiEdit",
                                           edits=[{"old_string": "a", "new_string": "[env]"}])), "deny")
        self.assertIsNone(self.pre(".cargo/config.toml", content="[alias]\nx = \"y\"\n"))

    def test_post_reminder(self):
        msg = self.post("crates/ptr-ledger/src/lib.rs")["hookSpecificOutput"]["additionalContext"]
        self.assertIn("crates/ptr-ledger/component.toml", msg)
        self.assertIn("make docs", msg)
        self.assertIn("component.toml", self.post("crates/ptr-ledger/Cargo.toml")["hookSpecificOutput"]["additionalContext"])

    def test_post_no_reminder_component_toml(self):
        self.assertIsNone(self.post("crates/ptr-ledger/component.toml"))

    def test_post_experiments(self):
        msg = self.post("experiments/model/M003-latent-recurrence/config.toml")["hookSpecificOutput"]["additionalContext"]
        self.assertIn("check_research_gates.py", msg)

    def test_windows_path(self):
        os.environ["CLAUDE_PROJECT_DIR"] = "C:\\Users\\me\\PTR"
        try:
            out = guard.run("pre", {"tool_input": {"file_path": "C:\\Users\\me\\PTR\\vendor\\x"}, "cwd": "C:\\Users\\me\\PTR"})
        finally:
            os.environ["CLAUDE_PROJECT_DIR"] = str(ROOT)
        # repo root cannot be found on this host for a fake Windows dir; check the path logic directly
        self.assertIsNone(out)
        rel = guard.relative_to_repo("C:\\Users\\me\\PTR\\vendor\\x", Path("C:/Users/me/PTR"))
        self.assertEqual(rel, "vendor/x")
        self.assertEqual(guard.relative_to_repo("c:/users/ME/ptr/vendor/x", Path("C:\\Users\\me\\PTR")), "vendor/x")
        self.assertEqual(guard.relative_to_repo("/c/Users/me/PTR/vendor/x", Path("C:/Users/me/PTR")), "vendor/x")


class GuardEndToEnd(unittest.TestCase):
    def run_hook(self, mode, stdin, env_dir=str(ROOT)):
        env = {**os.environ, "CLAUDE_PROJECT_DIR": env_dir}
        return subprocess.run([sys.executable, str(SCRIPT), mode], input=stdin, capture_output=True,
                              text=True, env=env, timeout=30)

    def test_deny_vendor_e2e(self):
        p = self.run_hook("pre", json.dumps({"hook_event_name": "PreToolUse", "cwd": str(ROOT), "tool_name": "Write",
                                            "tool_input": {"file_path": "vendor/x.rs", "content": "x"}}))
        self.assertEqual(p.returncode, 0)
        self.assertEqual(json.loads(p.stdout)["hookSpecificOutput"]["permissionDecision"], "deny")

    def test_windows_vendor_e2e_with_project_dir(self):
        p = self.run_hook("pre", json.dumps({"cwd": str(ROOT), "tool_name": "Edit", "tool_input": {
            "file_path": str(ROOT).replace("/", "\\") + "\\vendor\\x"}}))
        self.assertEqual(p.returncode, 0)
        if os.name == "nt":
            self.assertIn("deny", p.stdout)

    def test_post_e2e(self):
        p = self.run_hook("post", json.dumps({"cwd": str(ROOT), "tool_name": "Edit",
                                             "tool_input": {"file_path": "crates/ptr-ledger/src/lib.rs"}}))
        self.assertEqual(p.returncode, 0)
        self.assertIn("component.toml", json.loads(p.stdout)["hookSpecificOutput"]["additionalContext"])

    def test_malformed_stdin(self):
        for bad in ("", "not json", "[1,2]", '{"tool_input": 5}'):
            p = self.run_hook("pre", bad)
            self.assertEqual(p.returncode, 0)
            self.assertEqual(p.stdout, "")

    def test_outside_repo(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.run_hook("pre", json.dumps({"cwd": str(ROOT), "tool_name": "Write",
                                                "tool_input": {"file_path": str(Path(d) / "vendor" / "x"), "content": "x"}}))
        self.assertEqual((p.returncode, p.stdout), (0, ""))


if __name__ == "__main__":
    unittest.main()
