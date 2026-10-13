#!/usr/bin/env python3
"""Publish local and origin-tracking branches to a new git remote in ONE push.

Meant to be run by the maintainer on their own machine (Windows PowerShell or
Ubuntu): `python3 scripts/publish_remote.py ...` or `py scripts/publish_remote.py ...`.
Standard library only, Python 3.11+.

Design rules:
  * Credentials stay in the user's git credential manager. The script never reads,
    prints or stores tokens and refuses URLs that embed credentials such as
    `https://user:token@host/...`.
  * Everything goes out in a single `git push --porcelain` invocation (low
    automation volume), never a loop of single-ref pushes.
  * Never forces. Refspecs are `<src>:refs/heads/<name>` without `+`.
  * Branch patterns (`--branches`, default `main,claude/*`) are globbed over local
    `refs/heads/` AND remote-tracking `refs/remotes/origin/`. When a branch exists
    in both, the local branch is the source.
  * A second run is a no-op: every ref is reported as up-to-date.

Exit codes:
  0  success, nothing rejected (including an all-up-to-date run and --dry-run)
  1  precondition or usage error (not a git repo, credential URL, remote URL
     mismatch without --set-url, no matching refs, confirmation declined)
  2  command-line syntax error (argparse)
  3  the remote rejected one or more refs (for example non-fast-forward)
  4  `git push` failed without per-ref rejections (authentication, network, ...)
"""
from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_REJECTED = 3
EXIT_PUSH_FAILED = 4

PORCELAIN_LINE = re.compile(r'^([ +\-*!=])\t([^\t]*)\t(.*)$')
STATUS_BY_FLAG = {
    '*': 'new',
    ' ': 'updated',
    '+': 'updated',
    '-': 'deleted',
    '=': 'up-to-date',
    '!': 'rejected',
}


class PublishError(Exception):
    """A precondition failed; carries the process exit code."""

    def __init__(self, message: str, code: int = EXIT_USAGE) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Ref:
    source: str  # full source ref, e.g. refs/remotes/origin/claude/x
    dest: str  # full destination ref, e.g. refs/heads/claude/x

    @property
    def refspec(self) -> str:
        return f'{self.source}:{self.dest}'


def git(repo: str, *args: str, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    cmd = ['git', '-C', repo, *args]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE if capture else None,
                              stderr=subprocess.PIPE if capture else None,
                              text=True, encoding='utf-8')
    except FileNotFoundError:
        raise PublishError('git was not found on PATH') from None
    if check and proc.returncode != 0:
        detail = (proc.stderr or '').strip()
        raise PublishError(f'`git {" ".join(args)}` failed: {detail}')
    return proc


def check_url(url: str) -> None:
    """Refuse URLs that embed credentials."""
    if '://' not in url:
        return  # scp-like `git@host:path` or a local path: no embedded password possible
    try:
        parts = urlsplit(url)
        password = parts.password
    except ValueError as exc:
        raise PublishError(f'unparseable remote URL: {exc}') from None
    if password is not None or (parts.username and parts.scheme in ('http', 'https')):
        raise PublishError(
            'remote URL embeds credentials. Use a plain URL (https://host/group/repo.git) '
            'and let your git credential manager prompt for the token.')


def repo_toplevel() -> str:
    proc = subprocess.run(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, encoding='utf-8')
    if proc.returncode != 0:
        raise PublishError('not inside a git repository')
    return proc.stdout.strip()


def collect_refs(repo: str, patterns: list[str], tracking_remote: str = 'origin') -> list[Ref]:
    out = git(repo, 'for-each-ref', '--format=%(refname)', 'refs/heads',
              f'refs/remotes/{tracking_remote}').stdout.split()
    local, tracked = {}, {}
    prefix = f'refs/remotes/{tracking_remote}/'
    for ref in out:
        if ref.startswith('refs/heads/'):
            local[ref[len('refs/heads/'):]] = ref
        elif ref.startswith(prefix):
            name = ref[len(prefix):]
            if name != 'HEAD':
                tracked[name] = ref
    chosen = {**tracked, **local}  # local wins
    refs = [Ref(src, f'refs/heads/{name}') for name, src in sorted(chosen.items())
            if any(fnmatch.fnmatchcase(name, p) for p in patterns)]
    return refs


def ensure_remote(repo: str, name: str, url: str | None, set_url: bool, dry_run: bool) -> str:
    """Make sure the remote exists with the right URL; return the effective URL."""
    proc = git(repo, 'config', '--get', f'remote.{name}.url', check=False)
    existing = proc.stdout.strip() if proc.returncode == 0 else None
    if existing is not None:
        check_url(existing)
    if existing is None:
        if not url:
            raise PublishError(f'remote {name!r} does not exist; pass --remote-url')
        if dry_run:
            print(f'[dry-run] would run: git remote add {name} {url}')
        else:
            git(repo, 'remote', 'add', name, url)
            print(f'Added remote {name!r} -> {url}')
        return url
    if url is None or url == existing:
        return existing
    if not set_url:
        raise PublishError(
            f'remote {name!r} already points to {existing!r}, not {url!r}. '
            'Pass --set-url to change it or choose another --name.')
    if dry_run:
        print(f'[dry-run] would run: git remote set-url {name} {url}')
    else:
        git(repo, 'remote', 'set-url', name, url)
        print(f'Updated remote {name!r} -> {url}')
    return url


def check_push_urls(repo: str, name: str, url: str) -> None:
    """Refuse unless `git push` would publish to exactly `url`.

    `remote.<name>.pushurl` replaces the URL for pushing and several push URLs are all used, so
    checking and showing only `remote.<name>.url` could confirm one server while the push goes
    to another or to several. Every effective push URL is credential-checked first.

    The push URLs are derived from the configuration as it will be once `ensure_remote` has
    run, so a dry run of `--set-url` checks the URL the real run would push to: configured
    `pushurl` entries if there are any (set-url leaves them alone), else every configured `url`
    entry when there are several (git pushes to all of them), else `url` itself."""
    def config_all(key: str) -> list[str]:
        proc = git(repo, 'config', '--get-all', f'remote.{name}.{key}', check=False)
        return [line.strip() for line in proc.stdout.splitlines() if line.strip()]

    urls = config_all('url')
    push_urls = config_all('pushurl') or (urls if len(urls) > 1 else [url])
    for push_url in push_urls:
        check_url(push_url)
    if push_urls != [url]:
        raise PublishError(
            f'remote {name!r} pushes to {push_urls!r}, not only to {url!r}; remove '
            f'remote.{name}.pushurl (git config --unset-all remote.{name}.pushurl) or choose '
            'another --name')


def parse_porcelain(stdout: str) -> list[tuple[str, str, str]]:
    """Return (status, destination ref, detail) per ref line of `git push --porcelain`."""
    results = []
    for line in stdout.splitlines():
        match = PORCELAIN_LINE.match(line)
        if not match:
            continue
        flag, refs, summary = match.groups()
        dest = refs.split(':', 1)[-1]
        results.append((STATUS_BY_FLAG[flag], dest, summary))
    return results


def short(ref: str) -> str:
    for prefix in ('refs/heads/', 'refs/tags/'):
        if ref.startswith(prefix):
            return ('tag ' if prefix.endswith('tags/') else '') + ref[len(prefix):]
    return ref


def print_table(rows: list[tuple[str, str, str]]) -> None:
    width = max([len('ref')] + [len(short(dest)) for _, dest, _ in rows])
    print(f'{"ref".ljust(width)}  status')
    for status, dest, detail in rows:
        extra = f'  {detail}' if status == 'rejected' else ''
        print(f'{short(dest).ljust(width)}  {status}{extra}')


def confirm(count: int, url: str) -> bool:
    try:
        answer = input(f'Push {count} ref(s) to {url}? [y/N] ')
    except EOFError:
        return False
    return answer.strip().lower() in ('y', 'yes')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Publish branches (and optionally tags) to a new git remote in one push. '
                    'Never forces; never handles credentials.')
    parser.add_argument('--remote-url', help='remote URL (required unless the remote exists)')
    parser.add_argument('--name', default='gitlab', help='remote name (default: gitlab)')
    parser.add_argument('--branches', default='main,claude/*',
                        help='comma-separated globs over local and origin/* branches '
                             '(default: main,claude/*)')
    parser.add_argument('--tags', action='store_true', help='also push tags')
    parser.add_argument('--dry-run', action='store_true',
                        help='print the git command and refspecs, change nothing')
    parser.add_argument('--yes', action='store_true', help='skip the confirmation prompt')
    parser.add_argument('--set-url', action='store_true',
                        help='allow changing the URL of an existing remote')
    parser.add_argument('--set-upstream-main', action='store_true',
                        help='after a successful push make main track <name>/main')
    return parser


def run(args: argparse.Namespace) -> int:
    patterns = [p.strip() for p in args.branches.split(',') if p.strip()]
    if not patterns:
        raise PublishError('--branches is empty')
    if args.remote_url:
        check_url(args.remote_url)
    repo = repo_toplevel()
    if not all((Path(repo) / f).exists() for f in ('Cargo.toml', 'CLAUDE.md')):
        print(f'warning: {repo} does not look like the PTR repository root', file=sys.stderr)
    if git(repo, 'status', '--porcelain', '--untracked-files=no').stdout.strip():
        print('warning: uncommitted tracked changes (ignored; only committed refs are pushed)',
              file=sys.stderr)

    refs = collect_refs(repo, patterns)
    if not refs and not args.tags:
        raise PublishError(f'no branches match {patterns} in refs/heads or refs/remotes/origin')
    url = ensure_remote(repo, args.name, args.remote_url, args.set_url, args.dry_run)
    check_push_urls(repo, args.name, url)
    cmd = ['git', 'push', '--porcelain', args.name, *[r.refspec for r in refs]]
    if args.tags:
        cmd.append('--tags')

    print(f'Remote {args.name!r}: {url}')
    print(f'{len(refs)} branch ref(s)' + (' plus tags' if args.tags else ''))
    print('Command: ' + ' '.join(cmd))
    for ref in refs:
        print(f'  {ref.refspec}')
    if args.dry_run:
        print('[dry-run] nothing was changed.')
        return EXIT_OK
    if not args.yes and not confirm(len(refs), url):
        raise PublishError('aborted by user')

    proc = subprocess.run(cmd[:1] + ['-C', repo] + cmd[1:], stdout=subprocess.PIPE, text=True,
                          encoding='utf-8')  # stderr stays attached: prompts and progress
    rows = parse_porcelain(proc.stdout)
    if rows:
        print_table(rows)
    rejected = [r for r in rows if r[0] == 'rejected']
    if rejected:
        print('\nRejected refs (not forced):', file=sys.stderr)
        for _, dest, detail in rejected:
            print(f'  {short(dest)}  {detail}', file=sys.stderr)
        return EXIT_REJECTED
    if proc.returncode != 0:
        print(f'git push failed with exit code {proc.returncode}', file=sys.stderr)
        return EXIT_PUSH_FAILED

    if args.set_upstream_main:
        has_main = git(repo, 'rev-parse', '--verify', '--quiet', 'refs/heads/main',
                       check=False).returncode == 0
        pushed = any(dest == 'refs/heads/main' for _, dest, _ in rows)
        if has_main and pushed:
            git(repo, 'config', 'branch.main.remote', args.name)
            git(repo, 'config', 'branch.main.merge', 'refs/heads/main')
            print(f'main now tracks {args.name}/main')
        else:
            print('note: --set-upstream-main skipped (no local main or main not pushed)')
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except PublishError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return exc.code


if __name__ == '__main__':
    sys.exit(main())
