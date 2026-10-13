"""`scripts/setup_desktop.sh --dry-run` must list every step and change nothing.

The desktop setup cannot be run for real in CI (no GPU, no Ubuntu desktop, several GB of
downloads), so these tests pin what can be checked: the script parses, the dry run prints
the seven steps in order with the commands the runbook documents, it never invokes sudo
or the driver install, and it leaves a fresh HOME empty.
"""

import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "setup_desktop.sh"

EXPECTED_STEPS = [
    "check the OS is Ubuntu",
    "check the NVIDIA driver",
    "install rustup and the toolchain pinned",
    "install uv",
    "create or sync training/.venv",
    "build ptr_py into training/.venv",
    "run the GPU doctor",
]


def dry_run(*extra):
    with tempfile.TemporaryDirectory() as home:
        env = {"HOME": home, "PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C"}
        proc = subprocess.run(
            ["bash", str(SCRIPT), "--dry-run", *extra],
            capture_output=True, text=True, env=env, timeout=60, check=False,
        )
        return proc, sorted(p.name for p in Path(home).iterdir())


class SetupDesktopDryRun(unittest.TestCase):
    def test_syntax(self):
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    @unittest.skipUnless(shutil.which("shellcheck"), "shellcheck not installed")
    def test_shellcheck(self):
        subprocess.run(["shellcheck", "-x", str(SCRIPT)], check=True)

    def test_step_list_in_order(self):
        proc, _ = dry_run("--yes")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        steps = re.findall(r"^STEP (\d)/7: (.*)$", proc.stdout, re.M)
        self.assertEqual([n for n, _ in steps], list("1234567"))
        for (_, title), expected in zip(steps, EXPECTED_STEPS):
            self.assertIn(expected, title)

    def test_documented_commands_are_printed(self):
        out, _ = dry_run()[0].stdout, None
        for needle in (
            "uv sync --project training --python 3.11 --extra moe --locked",
            "maturin develop --uv --locked -m bindings/ptr-py/Cargo.toml",
            "python3 scripts/gpu_doctor.py --require-gpu",
            "nvidia-driver-580-open",
        ):
            self.assertIn(needle, out)

    def test_pinned_toolchain_comes_from_rust_toolchain_toml(self):
        channel = re.search(r'channel\s*=\s*"([^"]+)"', (ROOT / "rust-toolchain.toml").read_text()).group(1)
        out = dry_run()[0].stdout
        self.assertIn(f"pinned channel: {channel}", out)

    def test_disk_floor_is_checked_in_step_one_before_any_install(self):
        out = dry_run("--yes")[0].stdout
        floor = out.index("free disk at")
        self.assertLess(floor, out.index("STEP 2/7"))
        proc = subprocess.run(
            ["bash", "-c", 'sed -n "/^MIN_FREE_GB=/,/^fi$/p" "$0"', str(SCRIPT)],
            capture_output=True, text=True, check=True,
        )
        self.assertIn("exit 1", proc.stdout)

    def test_changes_nothing_and_never_runs_sudo(self):
        proc, home_entries = dry_run("--yes")
        self.assertEqual(home_entries, [])
        executed = [l for l in proc.stdout.splitlines() if l.startswith("  + ")]
        self.assertTrue(executed)
        self.assertFalse([l for l in executed if "sudo" in l or "apt" in l], executed)
        # The driver commands are only ever printed as notes, never prefixed with run's "+".
        code = [l for l in SCRIPT.read_text().splitlines() if not l.lstrip().startswith(("#", "note"))]
        self.assertFalse([l for l in code if re.search(r"\bsudo\b", l)])

    def test_unknown_argument_is_rejected(self):
        proc = subprocess.run(["bash", str(SCRIPT), "--bogus"], capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 2)


if __name__ == "__main__":
    unittest.main()
