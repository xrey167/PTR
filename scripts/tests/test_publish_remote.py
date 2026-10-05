"""Tests for scripts/publish_remote.py using throw-away local repositories."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'publish_remote.py'

ENV = {
    **os.environ,
    'GIT_CONFIG_GLOBAL': os.devnull,
    'GIT_CONFIG_NOSYSTEM': '1',
    'GIT_TERMINAL_PROMPT': '0',
    'GIT_AUTHOR_NAME': 't', 'GIT_AUTHOR_EMAIL': 't@example.invalid',
    'GIT_COMMITTER_NAME': 't', 'GIT_COMMITTER_EMAIL': 't@example.invalid',
}


def git(cwd, *args):
    proc = subprocess.run(['git', *args], cwd=cwd, env=ENV, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise AssertionError(f'git {args} failed: {proc.stderr}')
    return proc.stdout.strip()


def commit(cwd, name):
    Path(cwd, name).write_text(name)
    git(cwd, 'add', name)
    git(cwd, 'commit', '-q', '-m', name)


class PublishRemoteTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.origin = root / 'origin.git'
        self.target = root / 'target.git'
        self.src = root / 'src'
        self.other = root / 'other'
        for bare in (self.origin, self.target):
            git(root, 'init', '-q', '--bare', '-b', 'main', str(bare))
        # Seed origin with main and two claude branches from a helper clone.
        git(root, 'clone', '-q', str(self.origin), str(self.other))
        git(self.other, 'checkout', '-q', '-b', 'main')
        commit(self.other, 'a')
        git(self.other, 'push', '-q', 'origin', 'main')
        for name in ('claude/one', 'claude/two', 'feature/x'):
            git(self.other, 'checkout', '-q', '-b', name, 'main')
            commit(self.other, name.replace('/', '_'))
            git(self.other, 'push', '-q', 'origin', name)
        # The working repo: local main plus remote-tracking refs only.
        git(root, 'clone', '-q', str(self.origin), str(self.src))
        git(self.src, 'checkout', '-q', 'main')
        git(self.src, 'tag', 'v1')

    def run_script(self, *args, url=True, cwd=None):
        cmd = [sys.executable, str(SCRIPT), *args]
        if url:
            cmd += ['--remote-url', str(self.target)]
        return subprocess.run(cmd, cwd=cwd or self.src, env=ENV, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, input='')

    def target_branches(self):
        out = git(self.target, 'for-each-ref', '--format=%(refname:short)', 'refs/heads')
        return set(out.split())

    def test_first_push_includes_origin_only_branches(self):
        proc = self.run_script('--yes')
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(self.target_branches(), {'main', 'claude/one', 'claude/two'})
        self.assertIn('new', proc.stdout)

    def test_second_run_is_noop(self):
        self.assertEqual(self.run_script('--yes').returncode, 0)
        before = git(self.target, 'for-each-ref')
        proc = self.run_script('--yes', url=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(git(self.target, 'for-each-ref'), before)
        self.assertEqual(proc.stdout.count('up-to-date'), 3)

    def test_dry_run_changes_nothing(self):
        proc = self.run_script('--dry-run')
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn('refs/remotes/origin/claude/one:refs/heads/claude/one', proc.stdout)
        self.assertNotIn('+refs', proc.stdout)
        self.assertEqual(self.target_branches(), set())
        self.assertEqual(git(self.src, 'remote'), 'origin')

    def test_credential_url_refused(self):
        for url in ('https://user:tok@gitlab.com/g/r.git', 'https://tok@gitlab.com/g/r.git'):
            proc = self.run_script('--yes', '--remote-url', url, url=False)
            self.assertEqual(proc.returncode, 1)
            self.assertIn('credentials', proc.stderr)
            self.assertNotIn('tok', proc.stdout)
        self.assertEqual(git(self.src, 'remote'), 'origin')

    def test_existing_remote_with_other_url_refused_without_set_url(self):
        git(self.src, 'remote', 'add', 'gitlab', str(self.origin))
        proc = self.run_script('--yes')
        self.assertEqual(proc.returncode, 1)
        self.assertIn('--set-url', proc.stderr)
        self.assertEqual(self.target_branches(), set())
        proc = self.run_script('--yes', '--set-url')
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(git(self.src, 'config', 'remote.gitlab.url'), str(self.target))

    def test_diverged_branch_rejected_and_not_forced(self):
        git(self.other, 'checkout', '-q', 'main')
        git(self.other, 'checkout', '-q', '--orphan', 'alt')
        commit(self.other, 'diverged')
        git(self.other, 'push', '-q', str(self.target), 'alt:main')
        diverged = git(self.target, 'rev-parse', 'main')
        proc = self.run_script('--yes')
        self.assertEqual(proc.returncode, 3, proc.stderr)
        self.assertEqual(git(self.target, 'rev-parse', 'main'), diverged)
        self.assertIn('rejected', proc.stdout)
        self.assertIn('main', proc.stderr)

    def test_branches_filter(self):
        proc = self.run_script('--yes', '--branches', 'claude/one,feature/*')
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(self.target_branches(), {'claude/one', 'feature/x'})

    def test_tags_only_with_flag(self):
        self.assertEqual(self.run_script('--yes').returncode, 0)
        self.assertEqual(git(self.target, 'tag'), '')
        proc = self.run_script('--yes', '--tags', url=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(git(self.target, 'tag'), 'v1')

    def test_set_upstream_main(self):
        proc = self.run_script('--yes', '--set-upstream-main')
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(git(self.src, 'config', 'branch.main.remote'), 'gitlab')

    def test_declined_confirmation_aborts(self):
        proc = self.run_script()  # stdin is empty: treated as "no"
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(self.target_branches(), set())


if __name__ == '__main__':
    unittest.main()
