---
name: ptr-offline-sync
description: Move the PTR repository between machines with git bundles when GitHub is unavailable, using the cross-platform helper scripts/ptr_bundle.py (create split bundle parts with SHA256 sums, join and verify them, apply into import/ branches without touching the current branch). Use when asked to "sync without GitHub", "transfer the repo", "make a bundle", "import a bundle", "offline sync", or "copy my branches to the other machine".
---

# Offline sync with git bundles

The helper needs only Python 3.11+ and `git`. It lives at `.claude/plugins/ptr-dev/skills/ptr-offline-sync/scripts/ptr_bundle.py` (below `HELPER`). `--help` works on every subcommand.

Safety: `apply` only writes `refs/heads/<prefix>*` (default `import/`), never the current branch or working tree. Merge or cherry-pick explicitly afterwards. Do not push, and do not put secrets in bundles (a bundle contains full history).

## Source machine

bash:
```bash
HELPER=.claude/plugins/ptr-dev/skills/ptr-offline-sync/scripts/ptr_bundle.py
python3 "$HELPER" create --out ../ptr-bundle                       # all refs
python3 "$HELPER" create --out ../ptr-bundle2 --since origin/main  # only commits not on origin/main
python3 "$HELPER" create --out ../ptr-bundle --split-mib 10        # smaller parts
```

PowerShell (`py` or `python`; no `cat`, the script does the concatenation):
```powershell
$Helper = ".claude\plugins\ptr-dev\skills\ptr-offline-sync\scripts\ptr_bundle.py"
py $Helper create --out ..\ptr-bundle
py $Helper create --out ..\ptr-bundle2 --since origin/main
```

`--out` must be new or empty. Output: `repo.bundle` (or `repo.bundle.part000`, `.part001`, ... when larger than `--split-mib`, default 25) plus `SHA256SUMS`. Copy the whole directory (USB, scp, chat attachments in parts).

## Target machine (inside a clone or an `git init`ed repo)

bash:
```bash
python3 "$HELPER" join ../ptr-bundle          # checks sha256 of every part, concatenates, runs git bundle verify
python3 "$HELPER" apply ../ptr-bundle/repo.bundle                       # branches land as import/<name>
python3 "$HELPER" apply ../ptr-bundle/repo.bundle --branch-prefix from-laptop/
git log --oneline import/main -5
git merge --ff-only import/main               # when ready, explicit and reviewable
```

PowerShell:
```powershell
py $Helper join ..\ptr-bundle
py $Helper apply ..\ptr-bundle\repo.bundle
git log --oneline import/main -5
```

If the helper script is not yet on the target machine, copy that single file first (it has no dependencies); plain `git bundle verify repo.bundle` and `git fetch repo.bundle 'refs/heads/*:refs/heads/import/*'` do the same by hand.

## Notes

- An incremental bundle (`--since REF`) has prerequisites: the target must already contain `REF`'s history, otherwise `join`/`apply` fail at `git bundle verify` with a message naming the missing commits. Use a full bundle then.
- `join` fails on any sha256 mismatch and leaves no joined file; re-copy the bad part.
- Only branches are imported; tags are not fetched by `apply`.
- After import run the `ptr-verify` skill before merging.

## Tests

```bash
python3 -m unittest discover -s .claude/plugins/ptr-dev/skills/ptr-offline-sync/scripts -p test_ptr_bundle.py -v
```
