# Claude Code tooling for PTR

The repository ships two Claude Code plugins in a repo-local marketplace (`.claude-plugin/marketplace.json`, name `ptr`). They encode the repository's own workflows and rules so a local Claude Code session follows them without being told each time. The working method for every agent (roles, rules, freeze and merge discipline, report formats) is `AGENTS.md`; the agents below implement its roles.

| Plugin | Path | What it adds |
|---|---|---|
| `ptr-dev` | `.claude/plugins/ptr-dev/` | 8 skills, 6 agents (5 read-only, 1 coding channel), 2 guard hooks |
| `ptr-mods` | `.claude/plugins/ptr-mods/` | a `/runs` pane for GPU training runs and a status line |

## Install (once per machine)

Run from the repository root. The `./` matters: a bare `.` is rejected.

```bash
claude plugin marketplace add ./
claude plugin install ptr-dev@ptr
claude plugin install ptr-mods@ptr      # optional, needs a Claude Code build with hooks modules
claude plugin list                      # both show "Read from: <repo>/.claude/plugins/..."
```

The same commands work in PowerShell on Windows. Because the marketplace is a folder and the plugin sources are relative paths, Claude Code reads the plugins straight from the working copy: after `git pull` or editing a skill, run `/reload-plugins` in the session; no reinstall and no version bump. To try a plugin without installing it: `claude --plugin-dir .claude/plugins/ptr-dev`.

`claude plugin details ptr-dev@ptr` shows the component inventory and the always-on token cost (about 1.6k tokens per session for the skill and agent descriptions).

## ptr-dev

### Skills

Invoke by asking for the task, or explicitly as `/ptr-dev:<name>`.

| Skill | Use it to |
|---|---|
| `ptr-verify` | derive the checks a change needs from the diff (per-crate cargo test/clippy/fmt, component.toml and `make docs-check`, feature-gated `make ci-<job>` targets, Python and experiment validators) and report each with real output |
| `ptr-crate-change` | change or add a crate following `docs/RUST_API_STYLE.md`, `docs/TESTING.md`, docs-as-code and the backend-feature rules |
| `ptr-new-experiment` | create an experiment with `scripts/new_experiment.py`, with hypothesis, metrics, baseline and falsification; successors instead of patches for prepared records |
| `ptr-results` | write up a run or experiment with the right evidence class, seeds, baselines and intervals, without over-claiming |
| `ptr-gitlab` | work with the GitLab remote (GitHub is suspended): publish with `scripts/publish_remote.py` (dry run first), draft MRs and pipeline status with `glab`, low-automation rules; background in `docs/HOSTING.md` |
| `ptr-offline-sync` | move the repository between machines with git bundles (create, split, join, verify, apply), bash and PowerShell |
| `ptr-status` | produce a short Done / In progress / Blocked / Next report |
| `experiment-run` | run and monitor a GPU smoke run or experiment on the Ubuntu desktop with two RTX 3090 (`hardware/dual-rtx3090.toml`) (`docs/DESKTOP_SETUP.md`) |

`ptr-offline-sync` ships a helper, `.claude/plugins/ptr-dev/skills/ptr-offline-sync/scripts/ptr_bundle.py` (Python 3.11, stdlib only):

```bash
python3 .claude/plugins/ptr-dev/skills/ptr-offline-sync/scripts/ptr_bundle.py create --out ../ptr-export
python3 .claude/plugins/ptr-dev/skills/ptr-offline-sync/scripts/ptr_bundle.py join ../ptr-export
python3 .claude/plugins/ptr-dev/skills/ptr-offline-sync/scripts/ptr_bundle.py apply ../ptr-export/<bundle>
```

### Agents

| Agent | Use it to |
|---|---|
| `ptr-invariant-reviewer` | review a diff against the 17 invariants in `docs/INVARIANTS.md`, `docs/COMPONENT_CONTRACTS.md` and the CLAUDE.md conventions; findings ranked P0 to P2 with file:line |
| `ptr-claims-auditor` | check docs, STATUS files and write-ups for claims the evidence does not support |
| `ptr-experiment-designer` | draft a falsifiable experiment design and the exact `new_experiment.py` command |
| `ptr-rereviewer` | read-only adversarial re-review of a channel round: diff against the base, read-only and Python checks re-run, findings P0 to P2 with file:line, verdict |
| `ptr-fact-checker` | read-only check of numbers, verdicts, hashes, tags, seed lists, mutation counts and file:line citations in changed docs against `results/`, `STALE.toml`, `registry.toml` and git tags; table of mismatches |
| `ptr-channel` | the one agent that writes: a coding channel in a single worktree per `AGENTS.md`; never commits, pushes, merges, tags or freezes; one hand-back |

The first five agents are read-only. `ptr-channel` is the exception and is bound by `AGENTS.md`; the head (the session that commits) is never delegated.

### Hooks

`hooks/ptr_guard.py` runs on every Edit, Write, MultiEdit and NotebookEdit. It needs `python3` or `python` on PATH and uses the standard library only.

| When | Rule |
|---|---|
| before an edit | deny `vendor/**` (see `docs/VENDOR_PATCH_POLICY.md`) |
| before an edit | deny `training/runs/**` (run artifacts) |
| before an edit | deny the directory of any experiment frozen by `experiments/preregistration.toml` (status prepared, running, completed, failed or superseded); create a successor instead |
| before an edit | deny adding an `[env]` table to `.cargo/config.toml` |
| after an edit | under `crates/<c>/src/**` or `crates/<c>/Cargo.toml`: remind to update `crates/<c>/component.toml` and run `make docs` |
| after an edit | under `experiments/**`: remind to run the experiment validators |

Malformed input or any internal error allows the edit and prints one line to stderr; the hook never blocks by accident. Tests: `python3 -m unittest scripts/tests/test_claude_ptr_guard.py` (also part of `python3 -m unittest discover -s scripts/tests`, so CI runs them).

### Allowlist note

`.claude/settings.json` pre-approves only commands that run no repository code: read-only `git` (status, diff, log, show, merge-base, worktree list, tag -l), `nvidia-smi`, `claude plugin validate` and read-only GitLab commands (`glab mr view`, `glab mr list`, `glab ci status`, `glab ci view`, `glab ci list`). Commands that execute the checkout's Makefile, scripts, tests or build code (`make check/test/fmt/...`, `cargo check/test`, `python3 scripts/...`, `claude plugin test`, `python3 scripts/publish_remote.py --dry-run`) are in `.claude/settings.local.example.json`: copy it to `.claude/settings.local.json` (git-ignored) on a machine and checkout you trust, never for a contributor's branch you are reviewing. Real publishing, MR creation and merging always ask.

## ptr-mods

A hooks-module plugin (live UI). Details and limits are in `.claude/plugins/ptr-mods/README.md`.

- `/runs` opens the **PTR Runs** pane: newest log in `training/runs/logs/`, whether its PID is alive, and one GPU line from `nvidia-smi`; refreshes every 30 seconds and on `r`.
- Status line: `ptr <branch> · <n> changed`, plus `· docs?` when crate sources changed without their `component.toml`.

Checks: `claude plugin validate .claude/plugins/ptr-mods` and `claude plugin test .claude/plugins/ptr-mods`. It was built and tested against Claude Code 2.1.289; an older local build may not load hooks modules, which does not affect `ptr-dev`.

## Changing the tooling

- Validate after every change: `claude plugin validate .claude/plugins/ptr-dev`, `claude plugin validate .claude/plugins/ptr-mods`, `claude plugin validate ./` (marketplace).
- Skills name only commands that exist in the Makefile or `scripts/`; check new ones before adding them.
- Review agents stay read-only (`tools: Read, Grep, Glob, Bash` and a body that forbids writes). Only `ptr-channel` may write, and it may not commit.
