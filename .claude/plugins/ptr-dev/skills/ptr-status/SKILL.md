---
name: ptr-status
description: Produce a short PTR project status report (Done / In progress / Blocked / Next) from git log, git status, branches, STATUS.md, docs/components/STATUS.md and experiments/registry.toml status counts. Use when asked for "status", "where are we", "what's next", "progress report", or at the start of a session.
---

# Project status report

Read-only: do not edit files. Facts come from the repository; mark anything you could not check.

## Gather

```bash
git log --oneline -15
git status --short --branch
git branch -a -vv
git log --oneline --since="14 days ago" | wc -l
sed -n 1,60p STATUS.md                      # hand-maintained evidence notes; can lag (crate count is out of date)
sed -n 1,12p docs/components/STATUS.md      # generated: component count, maturity distribution, test markers
python3 -c "import tomllib,collections;print(collections.Counter(e['status'] for e in tomllib.load(open('experiments/registry.toml','rb'))['experiment']))"
python3 scripts/check_research_gates.py     # optional: are gates green
```

Planned work and open items live in docs, not an issue tracker (GitHub may be unreachable): `docs/PRIORITIES.md` (priority ledger), `docs/ROADMAP.md`, `docs/OPEN_ITEMS_PLAN_20260921.md` and `docs/RECOMMENDATIONS_20260924.md` (dated records; read their status notes, do not treat bodies as current), and the "does not close" sections of `docs/architecture/*.md`. List the few items that matter, with the file they came from.

## Output format

```
## Done
- <recent merged/committed work, with short sha>; components/tests from docs/components/STATUS.md
## In progress
- <current branch, uncommitted changes, running/prepared experiments>
## Blocked
- <missing hardware, unresolved preregistration pins (must-be-pinned-...), failing checks, GitHub unavailable>
## Next
- <3 to 5 concrete next steps, each tied to a doc or experiment ID>
```

Keep it under one screen. Distinguish implemented-and-tested from scaffold; a README is not evidence. Label any result with its evidence class (pipeline evidence / mechanism check / research result, see `ptr-results`). Note that experiment counts come from `experiments/registry.toml` on this branch only; other branches may hold more (`git branch -a`).
