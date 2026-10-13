# Shared shell helpers of the generated freeze scripts. Source after `cd <worktree>`.

# waitdisk [GB]: block until at least GB (default $FREEZE_DISK_FLOOR_GB, else 6) GiB are free on
# $FREEZE_DISK_PATH (default the current directory). A floor of 0 disables the wait.
waitdisk() {
  local floor=${1:-${FREEZE_DISK_FLOOR_GB:-6}} path=${FREEZE_DISK_PATH:-.} free
  [ "$floor" -gt 0 ] 2>/dev/null || return 0
  while :; do
    free=$(df -Pk "$path" | awk 'NR==2{print int($4/1048576)}')
    [ "${free:-0}" -ge "$floor" ] && return 0
    echo "waitdisk: ${free:-0} GiB free, need $floor; sleeping ${FREEZE_DISK_POLL:-120}s"
    sleep "${FREEZE_DISK_POLL:-120}"
  done
}

# clean: remove bytecode caches (they would show up as untracked files).
clean() { find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true; }

# only_paths REGEX: fail if anything outside the given path prefixes (extended regex, anchored at
# the start) is modified or untracked.
only_paths() {
  local bad
  bad=$(git status --porcelain | awk '{print $NF}' | grep -vE "^($1)" || true)
  [ -z "$bad" ] || { echo "UNEXPECTED CHANGES:"; echo "$bad" | head; return 1; }
}

# add_existing PATH...: git add -- only for explicit paths that exist (never `git add -A`).
add_existing() {
  local p keep=()
  for p in "$@"; do [ -e "$p" ] && keep+=("$p"); done
  [ "${#keep[@]}" -gt 0 ] && git add -- "${keep[@]}"
  return 0
}
