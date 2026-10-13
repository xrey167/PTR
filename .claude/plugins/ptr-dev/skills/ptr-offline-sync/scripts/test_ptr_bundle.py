"""Tests for ptr_bundle.py using throwaway git repositories."""

import argparse
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ptr_bundle  # noqa: E402

ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com",
    "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull,
}


def run(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, env=ENV, check=True,
                          capture_output=True, text=True).stdout.strip()


def make_repo(path, name="a.txt", content="hello\n", branch="main"):
    Path(path).mkdir(parents=True, exist_ok=True)
    run(path, "init", "-q", "-b", branch)
    commit(path, name, content)


def commit(path, name, content):
    (Path(path) / name).write_bytes(content if isinstance(content, bytes) else content.encode())
    run(path, "add", name)
    run(path, "commit", "-q", "-m", f"add {name}")
    return run(path, "rev-parse", "HEAD")


def cli(cwd, *args):
    old = os.getcwd()
    os.chdir(cwd)
    try:
        return ptr_bundle.main(list(args))
    finally:
        os.chdir(old)


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        os.environ.update({k: v for k, v in ENV.items() if k.startswith("GIT_")})

    def test_create_split_join_apply_roundtrip(self):
        src = self.root / "src"
        make_repo(src)
        run(src, "checkout", "-q", "-b", "feature")
        # incompressible payload so the bundle exceeds 1 MiB and gets split
        commit(src, "blob.bin", os.urandom(3 * 1024 * 1024))
        head = run(src, "rev-parse", "HEAD")
        out = self.root / "out"
        self.assertEqual(cli(src, "create", "--out", str(out), "--split-mib", "1"), 0)
        parts = sorted(p.name for p in out.glob("repo.bundle.part*"))
        self.assertGreaterEqual(len(parts), 3)
        self.assertFalse((out / "repo.bundle").exists())
        self.assertTrue((out / "SHA256SUMS").is_file())

        dst = self.root / "dst"
        make_repo(dst, name="other.txt", content="mine\n")
        before = run(dst, "rev-parse", "HEAD")
        self.assertEqual(cli(dst, "join", str(out)), 0)
        bundle = out / "repo.bundle"
        self.assertTrue(bundle.is_file())
        self.assertEqual(cli(dst, "apply", str(bundle)), 0)
        self.assertEqual(run(dst, "rev-parse", "import/feature"), head)
        self.assertEqual(run(dst, "rev-parse", "import/main"), run(src, "rev-parse", "main"))
        # current branch untouched
        self.assertEqual(run(dst, "rev-parse", "HEAD"), before)
        self.assertEqual(run(dst, "branch", "--show-current"), "main")
        self.assertEqual((dst / "other.txt").read_text(), "mine\n")

    def test_small_repo_is_single_file(self):
        src = self.root / "src"
        make_repo(src)
        out = self.root / "out"
        self.assertEqual(cli(src, "create", "--out", str(out)), 0)
        self.assertTrue((out / "repo.bundle").is_file())
        self.assertEqual(list(out.glob("*.part*")), [])

    def test_join_detects_corruption(self):
        src = self.root / "src"
        make_repo(src)
        commit(src, "blob.bin", os.urandom(2 * 1024 * 1024))
        out = self.root / "out"
        cli(src, "create", "--out", str(out), "--split-mib", "1")
        part = out / "repo.bundle.part001"
        data = bytearray(part.read_bytes())
        data[10] ^= 0xFF
        part.write_bytes(bytes(data))
        self.assertEqual(cli(src, "join", str(out)), 1)
        self.assertFalse((out / "repo.bundle").exists())

    def test_join_rejects_paths_outside_the_directory(self):
        out = self.root / "out"
        out.mkdir()
        secret = self.root / "secret.txt"
        secret.write_text("not part of any bundle\n")
        digest = ptr_bundle.sha256_file(secret)
        for name in ("../secret.txt", str(secret), "repo.bundle.part001/../../secret.txt"):
            (out / "SHA256SUMS").write_text(f"{'0' * 64}  {name}\n")
            with self.assertRaises(ptr_bundle.BundleError) as caught:
                ptr_bundle.cmd_join(argparse.Namespace(dir=str(out)))
            self.assertIn("unexpected entries", str(caught.exception))
            self.assertNotIn(digest, str(caught.exception))

    def test_since_creates_incremental_bundle(self):
        src = self.root / "src"
        make_repo(src)
        base = run(src, "rev-parse", "HEAD")
        second = commit(src, "b.txt", "two\n")
        out = self.root / "out"
        self.assertEqual(cli(src, "create", "--out", str(out), "--since", base), 0)
        listing = run(src, "bundle", "list-heads", str(out / "repo.bundle"))
        self.assertIn(second, listing)
        # applying onto a clone that already has `base` works
        dst = self.root / "dst"
        subprocess.run(["git", "clone", "-q", "--branch", "main", str(src), str(dst)],
                       env=ENV, check=True, capture_output=True)
        run(dst, "reset", "-q", "--hard", base)
        self.assertEqual(cli(dst, "join", str(out)), 0)
        self.assertEqual(cli(dst, "apply", str(out / "repo.bundle"), "--branch-prefix", "in/"), 0)
        self.assertEqual(run(dst, "rev-parse", "in/main"), second)
        self.assertEqual(run(dst, "rev-parse", "HEAD"), base)

    def test_create_refuses_non_empty_out(self):
        src = self.root / "src"
        make_repo(src)
        out = self.root / "out"
        out.mkdir()
        (out / "x").write_text("x")
        self.assertEqual(cli(src, "create", "--out", str(out)), 1)

    def test_apply_rejects_unsafe_prefix(self):
        src = self.root / "src"
        make_repo(src)
        out = self.root / "out"
        cli(src, "create", "--out", str(out))
        self.assertEqual(cli(src, "apply", str(out / "repo.bundle"), "--branch-prefix", "../x"), 1)

    def test_split_file_sizes(self):
        f = self.root / "f.bin"
        f.write_bytes(b"x" * 2500)
        out = self.root / "p"
        out.mkdir()
        parts = ptr_bundle.split_file(f, out, 1000)
        self.assertEqual([p.stat().st_size for p in parts], [1000, 1000, 500])


if __name__ == "__main__":
    unittest.main()
