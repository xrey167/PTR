---
name: ptr-gitlab
description: Work with the PTR GitLab remote while GitHub is suspended - check remotes, publish with scripts/publish_remote.py (dry run first), open and follow draft merge requests with glab, read pipeline status and job logs, and troubleshoot auth or rejected pushes. Use when asked to "push to GitLab", "publish the repo", "open an MR", "check the pipeline", "why did CI fail on GitLab", or "set up the remote".
---

# GitLab remote

Background, setup and runner registration are in `docs/HOSTING.md`. This skill is how to act in a session.

## Hard rules

- Never handle tokens: do not ask for, print, paste, store or put them into a URL, file or command. Authentication is done by the user (credential manager, `glab auth login`). If a token appears in output or a chat, tell the user to revoke it.
- Never force-push (no `--force`, `--force-with-lease`, `+refs`).
- Low automation: batch pushes, no loops of comments or pipeline re-runs, at most a few pipelines per hour, never mass-create issues or MRs. A failed pipeline is read and fixed locally, not re-run to see what happens.
- Push or create an MR only when the user asked for it in this session.

## Check first

```bash
git remote -v
git status --short
git branch --show-current
```

If no GitLab remote exists, the user must create the private project and credentials first (`docs/HOSTING.md`, section 3).

## Publish

`scripts/publish_remote.py` makes one `git push`, refuses URLs with embedded credentials and exits 3 on rejected refs. Windows: `py` instead of `python3`.

1. Dry run and show the user the output:

```bash
python3 scripts/publish_remote.py --remote-url https://gitlab.com/<user>/ptr.git --dry-run
```

2. Only when the user asked to publish, run the same command without `--dry-run`. Options: `--name`, `--branches main,claude/*`, `--tags`, `--yes`, `--set-url`, `--set-upstream-main`; check `python3 scripts/publish_remote.py --help`.

## Merge requests

Use `glab` (see `glab mr --help` and `glab ci --help` for flags):

```bash
glab mr create --draft --fill        # draft MR from the current branch
glab mr view                         # read the MR
glab mr list
glab ci status                       # pipeline of the current branch
glab ci view                         # jobs of the pipeline
glab ci trace <job-id>               # log of one job
```

- Merge only after a green pipeline, with a merge commit, and only when the user says so.
- Backend jobs are manual in MRs: start one only for a change that touches that backend, once.
- To reproduce a failing job, run its `make ci-<job>` target locally (see `make ci-local`).

## Troubleshooting

- Auth failure (401/403, "authentication failed"): run `glab auth status` and tell the user to log in again with `glab auth login`; on Windows the Git Credential Manager entry may be stale or the token expired or missing the `write_repository` and `api` scopes. Do not ask for the token.
- Rejected, non-fast-forward: `git fetch`, inspect `git log --oneline HEAD..<remote>/main`, merge (or rebase local-only commits with the user's consent), re-run the dry run, push again. Never force.
- Protected branch rejects the push: push a feature branch and open an MR instead.
- `publish_remote.py` exit 3: refs were rejected; read the script output, fix as above.
- Pipeline waits for a runner: `gpu-smoke` needs the desktop runner tagged `ptr-gpu` to be online (`docs/HOSTING.md`, section 4).
- No GitLab at all: fall back to `ptr-dev:ptr-offline-sync` (bundles).
