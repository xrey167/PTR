# ptr-mods

Claude Code mod (function hooks module with live UI) for the PTR repository.
Early-access API; requires a Claude Code version with hooks modules (tested
with 2.1.289).

## What it does

- `/runs` opens the pane **PTR Runs**:
  - newest `training/runs/logs/*.log` (name, mtime, last 15 lines);
  - whether the PID in the sibling `<name>.pid` is alive (`kill -0` through
    `$.process`; "unknown" where `kill` is unavailable, e.g. Windows);
  - a GPU line from `nvidia-smi` (`no NVIDIA GPU / nvidia-smi not found` on failure);
  - refreshes every 30 s while the pane is open, plus a Refresh button (`r`);
  - empty state when there is no log directory: start one with `make smoke-gpu`,
    see `docs/DESKTOP_SETUP.md`.
- Status line: `ptr <branch> · <n> changed`, plus ` · docs?` when a
  `crates/<c>/src/**` or `crates/<c>/Cargo.toml` change has no matching
  `crates/<c>/component.toml` change (the docs-as-code rule in CLAUDE.md).
  Updated on session start and after each Edit/Write/MultiEdit/Bash call
  (after the call has run; it never blocks or alters it). Cleared outside a git repo.

All process and file calls are wrapped; failures are shown as text and never
thrown out of a hook.

## Load

```bash
claude --plugin-dir .claude/plugins/ptr-mods
```

or add this repo's marketplace (if one lists `./.claude/plugins/ptr-mods`) and
install `ptr-mods`. Saving a file reloads the module in a watched session.

## Develop

```bash
claude plugin validate .claude/plugins/ptr-mods
claude plugin test .claude/plugins/ptr-mods
```

Pure logic (status text, porcelain parsing, GPU csv, log picking) lives in
`hooks/lib.ts` and is covered by `tests/*.test.ts`. Type-check with a tsconfig
kept outside the folder (see the header of the engine's `claude-code.d.ts`).
