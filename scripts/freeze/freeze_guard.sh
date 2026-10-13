# Source after `cd <worktree>`. Restores tracked files if the script exits between guard_on and
# guard_off (that is: after signing started and before the freeze commit exists).
#
#   . scripts/freeze/freeze_guard.sh
#   guard_on     # right before freeze_sign.py
#   ...          # sign, docs write, validate, gates, post tests, freeze commit + tag
#   guard_off    # right after the freeze commit
#
# Tracked files and the index are restored (git reset --hard); untracked files are never touched. The restore
# happens only when HEAD is still the commit recorded by guard_on, so it can never undo a commit.
GUARD_HEAD=""
guard_restore() {
  local rc=$?
  if [ -n "$GUARD_HEAD" ] && [ "$(git rev-parse HEAD)" = "$GUARD_HEAD" ] \
     && [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    # reset --hard restores index AND working tree of tracked files (a plain `git checkout -- .` restores
    # from the index, which keeps files staged by a successful `git add` before a failed commit).
    git reset -q --hard "$GUARD_HEAD"
    if [ -z "$(git status --porcelain --untracked-files=no)" ]; then
      echo "GUARD: exit $rc before the freeze commit, tracked files and index restored to $GUARD_HEAD (untracked files were not touched)"
    else
      echo "GUARD: restore FAILED after exit $rc; tracked files still differ from $GUARD_HEAD, inspect with git status" >&2
    fi
  fi
}
guard_on() { GUARD_HEAD=$(git rev-parse HEAD); trap guard_restore EXIT; }
guard_off() { GUARD_HEAD=""; }
