#!/usr/bin/env python3
"""Generate the freeze script of one preregistered experiment from a TOML spec and/or CLI flags.

    python3 scripts/freeze/make_freeze_script.py --spec freeze-E008.toml --output /tmp/freeze_E008.sh
    python3 scripts/freeze/make_freeze_script.py --id E008 --dir experiments/system/E008-... \
        --tag e008-prereg --pilot-ref experiments/system/E008-.../pilots/pilot-1002.json --pilot-seed 1002 \
        --pilot-command 'cargo run --release --locked -p ptr-bench -- e008 {seed}' --timing-key elapsed_ms

Spec keys (TOML; CLI flags of the same name with dashes override; list keys are repeatable flags):
  id, dir, tag               required. The tag is never derived: name it (q003v3-prereg for Q003-v3).
  pilot_ref, pilot_seed, pilot_command, timing_keys
                             optional pilot rerun on the frozen commit; {seed} is replaced in the
                             command, which prints the pilot JSON on stdout. All three of ref, seed,
                             command or none.
  pre_sign_checks            extra shell commands run after the standard pre-sign checks.
  unittest_modules           the experiment's own tests, e.g. scripts.tests.test_e008_aggregate.
  crate_tests                crates for `cargo test -p <crate> --locked`.
  cargo_package, cargo_features
                             package whose features get their own clippy and test run.
  sign                       placeholder = int table / --sign must-be-signed-x=INT for freeze_sign.py.
  env_file                   shell file sourced quietly (never printed) for DSNs and paths.
  gpu                        optional GPU index: exports CUDA_VISIBLE_DEVICES and tags every log line.
  disk_floor_gb              waitdisk floor, default 6 (0 disables).
  work_paths                 path prefixes the work commit may contain (default: experiments crates
                             bins docs scripts/tests research STATUS.md).
  work_message               work commit subject.
  seed_timeout               seconds per seed, default 100000.

Runtime variables of the generated script: FREEZE_TRAILER (commit message trailer, e.g. the
attribution lines; empty by default), FREEZE_LOGS (log directory), FREEZE_DISK_FLOOR_GB,
FREEZE_DISK_PATH, FREEZE_RESUME_FROM (or --resume-from N), FREEZE_SKIP_SEEDS, FREEZE_CLEAN_TARGET=1
(remove ./target after the heavy steps). The script never uses `git add -A`.
"""

import argparse
import re
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

LIST_KEYS = ("timing_keys", "pre_sign_checks", "unittest_modules", "crate_tests", "cargo_features",
             "work_paths", "sign_items")
SCALAR_KEYS = ("id", "dir", "tag", "pilot_ref", "pilot_seed", "pilot_command", "cargo_package",
               "env_file", "gpu", "disk_floor_gb", "work_message", "seed_timeout")
STR_KEYS = ("id", "dir", "tag", "pilot_ref", "pilot_command", "cargo_package", "env_file", "work_message")
INT_KEYS = ("pilot_seed", "gpu", "disk_floor_gb", "seed_timeout")
SPEC_KEYS = tuple(sorted({*STR_KEYS, *INT_KEYS, *(k for k in LIST_KEYS if k != "sign_items"), "sign"}))
DEFAULT_WORK_PATHS = ["experiments", "crates", "bins", "docs", "scripts/tests", "research", "STATUS.md"]
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
PATH_RE = re.compile(r"^[A-Za-z0-9._/-]+$")
MODULE_RE = re.compile(r"^[A-Za-z0-9_.]+$")


class SpecError(Exception):
    pass


def is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def check_types(spec):
    """Unknown keys and wrong types are errors, never silently ignored (a typo like `crate_test` would
    otherwise drop a check)."""
    unknown = sorted(set(spec) - {*SPEC_KEYS, "sign_items"})
    if unknown:
        raise SpecError(f"unknown spec keys {unknown}; allowed: {list(SPEC_KEYS)}")
    for key, value in spec.items():
        if key in STR_KEYS and not isinstance(value, str):
            raise SpecError(f"`{key}` must be a string")
        if key in INT_KEYS and not is_int(value):
            raise SpecError(f"`{key}` must be an integer")
        if key in LIST_KEYS and not (isinstance(value, list) and all(isinstance(i, str) for i in value)):
            raise SpecError(f"`{key}` must be a list of strings")


def load_spec(args):
    spec = {}
    if args.spec:
        spec = tomllib.loads(Path(args.spec).read_text(encoding="utf-8"))
        unknown = sorted(set(spec) - set(SPEC_KEYS))
        if unknown:
            raise SpecError(f"unknown spec keys {unknown}; allowed: {list(SPEC_KEYS)}")
        sign = spec.pop("sign", {})
        if not isinstance(sign, dict) or not all(is_int(v) for v in sign.values()):
            raise SpecError("`sign` must be a table placeholder = int")
        spec["sign_items"] = [f"{key}={value}" for key, value in sign.items()]
    for key in SCALAR_KEYS:
        value = getattr(args, key, None)
        if value is not None:
            spec[key] = value
    for key in LIST_KEYS:
        value = getattr(args, key, None)
        if value:
            spec[key] = value
    return spec


def validate(spec):
    check_types(spec)
    for key in ("id", "dir", "tag"):
        if not spec.get(key):
            raise SpecError(f"`{key}` is required (the tag is never derived)")
    if not ID_RE.match(spec["id"]):
        raise SpecError(f"bad id {spec['id']!r}")
    if not TAG_RE.match(spec["tag"]):
        raise SpecError(f"bad tag {spec['tag']!r}")
    for key in ("dir", "pilot_ref"):
        value = spec.get(key)
        if value and (not PATH_RE.match(value) or value.startswith("/") or ".." in value.split("/")):
            raise SpecError(f"`{key}` must be a repo-relative path, got {value!r}")
    for prefix in spec.get("work_paths", []):
        if not PATH_RE.match(prefix) or prefix.startswith("/") or ".." in prefix.split("/"):
            raise SpecError(f"bad work path {prefix!r}")
    pilot = [spec.get("pilot_ref"), spec.get("pilot_seed"), spec.get("pilot_command")]
    if any(pilot) and not all(item not in (None, "") for item in pilot):
        raise SpecError("pilot_ref, pilot_seed and pilot_command go together")
    if spec.get("pilot_seed") is not None and not isinstance(spec["pilot_seed"], int):
        raise SpecError("pilot_seed must be an integer")
    if spec.get("pilot_command") and "{seed}" not in spec["pilot_command"]:
        raise SpecError("pilot_command must contain {seed}")
    if spec.get("gpu") is not None and (not isinstance(spec["gpu"], int) or spec["gpu"] < 0):
        raise SpecError("gpu must be a non-negative integer")
    for module in spec.get("unittest_modules", []):
        if not MODULE_RE.match(module):
            raise SpecError(f"bad unittest module {module!r}")
    for crate in spec.get("crate_tests", []):
        if not re.match(r"^[A-Za-z0-9_-]+$", crate):
            raise SpecError(f"bad crate {crate!r}")
    for feature in spec.get("cargo_features", []):
        if not re.match(r"^[A-Za-z0-9_,/-]+$", feature):
            raise SpecError(f"bad feature {feature!r}")
    if spec.get("cargo_features") and not spec.get("cargo_package"):
        raise SpecError("cargo_features needs cargo_package")
    for item in spec.get("sign_items", []):
        if not re.fullmatch(r"must-be-[A-Za-z0-9_.-]+=-?[0-9]+", item):
            raise SpecError(f"bad sign item {item!r} (must-be-...=<int>)")
    for command in spec.get("pre_sign_checks", []):
        done = subprocess.run(["bash", "-n", "-c", command], capture_output=True, text=True)
        if done.returncode != 0:
            raise SpecError(f"pre-sign check does not parse as shell: {command!r}: {done.stderr.strip()}")
    if spec.get("pilot_command"):
        done = subprocess.run(["bash", "-n", "-c", spec["pilot_command"].replace("{seed}", "0")],
                              capture_output=True, text=True)
        if done.returncode != 0:
            raise SpecError(f"pilot_command does not parse as shell: {done.stderr.strip()}")


def q(value):
    return shlex.quote(str(value))


def generate(spec):
    validate(spec)
    ident, d, tag = spec["id"], spec["dir"], spec["tag"]
    slug = re.sub(r"[^A-Za-z0-9]+", "-", ident).strip("-").lower()
    gpu = spec.get("gpu")
    floor = spec.get("disk_floor_gb", 6)
    work = spec.get("work_paths") or DEFAULT_WORK_PATHS
    work_re = "|".join(re.escape(p) + "(/|$)" for p in work)
    message = spec.get("work_message") or f"{ident}: pre-freeze work (harness, preregistration, mutation plan, pilots)"
    unittests = " ".join(["scripts.tests.test_research_gates.EnrolledExperimentTests", *spec.get("unittest_modules", [])])
    pilot = spec.get("pilot_ref") and spec.get("pilot_command")
    L = []
    w = L.append

    w("#!/bin/bash")
    w(f"# Freeze script of {ident}, generated by scripts/freeze/make_freeze_script.py. Procedure and reasons per step:")
    w("# docs/FREEZE_PROCEDURE.md. Only the head runs it, in the worktree that holds the experiment.")
    w("# Order: dry-run -> work commit -> pre-sign checks on the committed tree -> clean tree -> guard_on -> sign")
    w("# -> docs write, validate, gates, post tests -> freeze commit + tag -> guard_off -> WHOLE mutation_check")
    w("# (CARGO_TARGET_DIR unset) -> evidence commit -> pilot rerun (timing keys stripped) -> preflight")
    w("# -> prepare -> seeds -> aggregate.")
    w("# Resume after a stop: bash THIS --resume-from N (6 mutation, 7 pilot, 8 prepare, 9 seeds, 10 aggregate);")
    w("# the freeze commit/tag must then be HEAD (N=6) or an ancestor of HEAD (N>6) on a clean tree.")
    w("set -u -o pipefail")
    w('START=${FREEZE_RESUME_FROM:-1}')
    w('if [ "${1:-}" = "--resume-from" ]; then START=${2:-}; fi')
    w('case "$START" in [1-9]|10) ;; *) echo "FREEZE-STOP: resume step must be an integer 1..10, got: $START" >&2; exit 4;; esac')
    w('W=${FREEZE_WORKTREE:-$(pwd)}')
    w('cd "$W" || exit 4')
    w('TOOLS=${FREEZE_TOOLS:-$W/scripts/freeze}')
    w('export PYTHONDONTWRITEBYTECODE=1')
    w("unset PYTHONUNBUFFERED")
    if spec.get("env_file"):
        w(f"ENV_FILE={q(spec['env_file'])}")
        w('[ -f "$ENV_FILE" ] || { echo "env file missing: $ENV_FILE"; exit 4; }')
        w('set -a; . "$ENV_FILE" >/dev/null 2>&1; set +a   # values are never printed')
    if gpu is not None:
        w(f"GPU={gpu}")
        w("export CUDA_VISIBLE_DEVICES=$GPU")
    w(f"ID={q(ident)}")
    w(f"D={q(d)}")
    w(f"TAG={q(tag)}")
    w(f"LOGS=${{FREEZE_LOGS:-${{TMPDIR:-/tmp}}/freeze-{slug}-logs}}")
    w('mkdir -p "$LOGS"')
    w(f"export FREEZE_DISK_FLOOR_GB=${{FREEZE_DISK_FLOOR_GB:-{int(floor)}}}")
    w('T=""; [ -n "${FREEZE_TRAILER:-}" ] && T=$(printf "\\n\\n%s" "$FREEZE_TRAILER")')
    w('say() { echo "[$ID${GPU:+ gpu=$GPU}] $*"; }')
    w('stop() { say "FREEZE-STOP step $1: $2"; exit "$3"; }')
    w('. "$TOOLS/freeze_lib.sh"')
    w('. "$TOOLS/freeze_guard.sh"')
    w('say "start at step $START (log dir $LOGS)"')
    if pilot:
        w(f"REF={q(spec['pilot_ref'])}")
    w("")
    w('if [ "$START" -gt 1 ]; then   # resume: the freeze commit must stand')
    w('  git rev-parse -q --verify "refs/tags/$TAG" >/dev/null || stop 0 "tag $TAG does not exist" 4')
    w('  [ -z "$(git status --porcelain)" ] || stop 0 "tree not clean" 4')
    w('  if [ "$START" -eq 6 ]; then')
    w('    [ "$(git rev-parse HEAD)" = "$(git rev-list -n1 "$TAG")" ] || stop 0 "HEAD is not the tagged freeze commit" 4')
    w("  else")
    w('    git merge-base --is-ancestor "$(git rev-list -n1 "$TAG")" HEAD || stop 0 "$TAG is not an ancestor of HEAD" 4')
    w("  fi")
    w("fi")
    w("")
    w('if [ "$START" -le 5 ]; then')
    w('say "STEP 1: dry-run before any commit or file change"')
    sign_args = " ".join(f"--sign {q(item)}" for item in spec.get("sign_items", []))
    sign_args = (" " + sign_args) if sign_args else ""
    w(f'python3 "$TOOLS/freeze_sign.py" --id "$ID" --dir "$D"{sign_args} --root "$W" --dry-run || stop 1 "dry-run failed" 4')
    w('git rev-parse -q --verify "refs/tags/$TAG" >/dev/null && stop 1 "tag $TAG already exists (use --resume-from)" 4')
    if pilot:
        w('[ -f "$REF" ] || stop 1 "pilot reference $REF missing" 4')
    w("")
    w('say "STEP 2: work commit (explicit path prefixes only)"')
    w("clean")
    w('if [ -n "$(git status --porcelain)" ]; then')
    w(f'  only_paths {q(work_re)} || stop 2 "changes outside the work paths" 4')
    w(f"  add_existing {' '.join(q(p) for p in work)}")
    w(f'  git commit -q -m {q(message)}"$T" || stop 2 "work commit failed" 4')
    w("fi")
    w('[ -z "$(git status --porcelain)" ] || { git status --short | head; stop 2 "tree not clean after the work commit" 4; }')
    w("")
    w('say "STEP 3: pre-sign checks on the committed tree"')
    w('python3 scripts/check_research_gates.py > "$LOGS/pre-gates.log" 2>&1 || { grep ERROR "$LOGS/pre-gates.log" | head; stop 3 "gates failed before signing" 5; }')
    w('python3 scripts/check_repo.py > "$LOGS/pre-repo.log" 2>&1 || stop 3 "check_repo failed" 4')
    w('python3 scripts/update_component_docs.py --check > "$LOGS/pre-docs.log" 2>&1 || stop 3 "component docs stale" 4')
    w('python3 scripts/mutation_check.py "$ID" --check > "$LOGS/pre-mutplan.log" 2>&1 || stop 3 "mutation plan check failed" 4')
    w('python3 scripts/run_experiment.py validate > "$LOGS/pre-validate.log" 2>&1 || stop 3 "validate failed" 4')
    w(f'python3 -m unittest {unittests} > "$LOGS/pre-unittest.log" 2>&1 || {{ tail -20 "$LOGS/pre-unittest.log"; stop 3 "enrolled-experiment or aggregate tests failed" 4; }}')
    w("waitdisk")
    w('cargo fmt --all -- --check > "$LOGS/pre-fmt.log" 2>&1 || stop 3 "fmt failed" 4')
    w('[ ! -d model/burn-a0 ] || (cd model/burn-a0 && cargo fmt --all -- --check) > "$LOGS/pre-fmt-a0.log" 2>&1 || stop 3 "fmt burn-a0 failed" 4')
    w('cargo clippy --workspace --all-targets --locked -- -D warnings > "$LOGS/pre-clippy.log" 2>&1 || { tail -20 "$LOGS/pre-clippy.log"; stop 3 "clippy failed" 4; }')
    features = ",".join(spec.get("cargo_features", []))
    pkg = spec.get("cargo_package")
    if features:
        w(f'cargo clippy -p {q(pkg)} --features {q(features)} --all-targets --locked -- -D warnings > "$LOGS/pre-clippy-features.log" 2>&1 || stop 3 "clippy with features failed" 4')
    if spec.get("crate_tests"):
        crates = " ".join(f"-p {q(c)}" for c in spec["crate_tests"])
        w(f'cargo test {crates} --locked > "$LOGS/pre-crates.log" 2>&1 || {{ grep -E "FAILED|panicked" "$LOGS/pre-crates.log" | head; stop 3 "crate tests failed" 4; }}')
    if pkg:
        feat = f" --features {q(features)}" if features else ""
        w(f'cargo test -p {q(pkg)}{feat} --locked > "$LOGS/pre-package-tests.log" 2>&1 || {{ grep -E "FAILED|panicked" "$LOGS/pre-package-tests.log" | head; stop 3 "package tests failed" 4; }}')
    for index, command in enumerate(spec.get("pre_sign_checks", []), 1):
        w(f'bash -c {q(command)} > "$LOGS/pre-check-{index}.log" 2>&1 || {{ tail -20 "$LOGS/pre-check-{index}.log"; stop 3 "pre-sign check {index} failed" 4; }}')
    w('[ "${FREEZE_CLEAN_TARGET:-0}" = 1 ] && rm -rf target')
    w("clean")
    w('[ -z "$(git status --porcelain)" ] || { git status --short | head; stop 3 "tree not clean before signing" 4; }')
    w("")
    w('say "STEP 4: sign under guard (pins criteria and text, status prepared)"')
    w("guard_on")
    w(f'python3 "$TOOLS/freeze_sign.py" --id "$ID" --dir "$D"{sign_args} --root "$W" || stop 4 "sign failed" 4')
    w('python3 scripts/update_component_docs.py --write | tail -1; [ "${PIPESTATUS[0]}" -eq 0 ] || stop 4 "docs write failed" 5')
    w('python3 scripts/run_experiment.py validate | tail -1; [ "${PIPESTATUS[0]}" -eq 0 ] || stop 4 "validate failed after signing" 5')
    w('python3 scripts/check_research_gates.py > "$LOGS/freeze-gates.log" 2>&1 || { grep ERROR "$LOGS/freeze-gates.log" | head; stop 4 "gates failed after signing" 5; }')
    w(f'python3 -m unittest {unittests} > "$LOGS/post-unittest.log" 2>&1 || {{ tail -20 "$LOGS/post-unittest.log"; stop 4 "post-sign tests failed" 5; }}')
    w("clean")
    w('only_paths "experiments/|crates/[^/]+/README.md|docs/components/" || stop 4 "signing changed unexpected files" 5')
    w("")
    w('say "STEP 5: freeze commit + tag"')
    w('add_existing experiments crates docs')
    w('git commit -q -m "$ID: freeze preregistration (pins signed by the head, status prepared)$T" || stop 5 "freeze commit failed" 5')
    w('FZ=$(git rev-parse HEAD); say "freeze $FZ"; git tag "$TAG" "$FZ" || stop 5 "tag failed" 5')
    w("guard_off")
    w("fi")
    w("")
    w('if [ "$START" -le 6 ]; then')
    w('say "STEP 6: WHOLE mutation_check on the committed tree (CARGO_TARGET_DIR unset: mutation_check builds into ./target)"')
    w("waitdisk")
    w("unset CARGO_TARGET_DIR")
    w('rc=0; python3 scripts/mutation_check.py "$ID" > "$LOGS/mutations.log" 2>&1 || rc=$?')
    w('say "mutations EXIT $rc"; tail -3 "$LOGS/mutations.log"')
    w('[ $rc -eq 0 ] || stop 6 "mutation check did not pass" 6')
    w('only_paths "$D/results/" || stop 6 "mutation run left another change" 6')
    w("clean")
    w('git add -- "$D/results" && git commit -q -m "research: $ID mutation evidence (committed tree)$T" || stop 6 "evidence commit failed" 6')
    w("fi")
    w("")
    if pilot:
        timing = " ".join(f"--timing-key {q(k)}" for k in spec.get("timing_keys", []))
        command = spec["pilot_command"].replace("{seed}", str(spec["pilot_seed"]))
        w('if [ "$START" -le 7 ]; then')
        w(f'say "STEP 7: pilot {spec["pilot_seed"]} rerun on the frozen commit, compared recursively with timing keys stripped"')
        w("waitdisk")
        w("unset CARGO_TARGET_DIR")
        w(f'rc=0; timeout "${{FREEZE_PILOT_TIMEOUT:-10800}}" bash -c {q(command)} > "$LOGS/pilot-rerun.json" 2> "$LOGS/pilot-rerun.err" || rc=$?')
        w('say "pilot rerun EXIT $rc"; [ $rc -eq 0 ] || stop 7 "pilot rerun failed" 7')
        w(f'python3 "$TOOLS/compare_pilot.py" "$REF" "$LOGS/pilot-rerun.json" {timing} || stop 7 "pilot outcome mismatch" 7')
        w('[ "${FREEZE_CLEAN_TARGET:-0}" = 1 ] && rm -rf target')
        w("clean")
        w("fi")
        w("")
    w('if [ "$START" -le 8 ]; then')
    w('say "STEP 8: preflight, prepare"')
    w('python3 scripts/run_experiment.py preflight "$ID" | tail -1; [ "${PIPESTATUS[0]}" -eq 0 ] || stop 8 "preflight failed" 7')
    w('python3 scripts/run_experiment.py prepare "$ID" | tail -1; [ "${PIPESTATUS[0]}" -eq 0 ] || stop 8 "prepare failed" 7')
    w("clean")
    w('only_paths "$D/" || stop 8 "prepare changed files outside the experiment" 7')
    w('git add -- "$D" && git commit -q -m "research: $ID prepare record$T" || stop 8 "prepare commit failed" 7')
    w("fi")
    w("")
    w('if [ "$START" -le 9 ]; then')
    w('say "STEP 9: seeds from config.toml (an exit other than 0 or 1 stops; 1 is a recorded product finding)"')
    w('SEEDS=$(python3 "$TOOLS/config_seeds.py" "$D/config.toml") || stop 9 "no seeds in config.toml" 7')
    w('say "seeds: $SEEDS"')
    w('for s in $SEEDS; do')
    w('  case " ${FREEZE_SKIP_SEEDS:-} " in *" $s "*) say "seed $s skipped"; continue;; esac')
    w("  waitdisk; clean")
    w(f'  rc=0; timeout {int(spec.get("seed_timeout", 100000))} python3 scripts/run_experiment.py run "$ID" --seed "$s" > "$LOGS/run-$s.log" 2>&1 || rc=$?')
    w('  say "seed $s EXIT $rc"')
    w("  clean")
    w('  only_paths "$D/" || stop 9 "seed $s left changes outside the experiment" 3')
    w('  git add -- "$D" && git commit -q -m "research: record $ID seed $s$T" || stop 9 "seed $s commit failed" 3')
    w('  case $rc in 0|1) ;; *) stop 9 "seed $s exit $rc" 3;; esac')
    w("done")
    w('say "RUNSDONE"')
    w("fi")
    w("")
    w('say "STEP 10: aggregate"')
    w('python3 "$D/aggregate.py" --report "$LOGS/report.md" 2>&1 | tail -5; [ "${PIPESTATUS[0]}" -eq 0 ] || stop 10 "aggregate failed" 7')
    w('say "FREEZE-DONE; next: finish_experiment (docs/FREEZE_PROCEDURE.md)"')
    return "\n".join(L) + "\n"


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--spec")
    p.add_argument("--output")
    for key in SCALAR_KEYS:
        flag = "--" + key.replace("_", "-")
        p.add_argument(flag, dest=key, type=int if key in ("pilot_seed", "gpu", "disk_floor_gb", "seed_timeout") else str)
    p.add_argument("--timing-key", dest="timing_keys", action="append")
    p.add_argument("--pre-sign-check", dest="pre_sign_checks", action="append")
    p.add_argument("--unittest-module", dest="unittest_modules", action="append")
    p.add_argument("--crate-test", dest="crate_tests", action="append")
    p.add_argument("--cargo-feature", dest="cargo_features", action="append")
    p.add_argument("--work-path", dest="work_paths", action="append")
    p.add_argument("--sign", dest="sign_items", action="append")
    return p


def main(argv):
    args = build_parser().parse_args(argv)
    try:
        script = generate(load_spec(args))
    except SpecError as error:
        print(f"MAKE-FREEZE-SCRIPT ERROR: {error}", file=sys.stderr)
        return 4
    done = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
    if done.returncode != 0:
        print(f"MAKE-FREEZE-SCRIPT ERROR: generated script fails bash -n: {done.stderr}", file=sys.stderr)
        return 4
    if args.output:
        out = Path(args.output)
        out.write_text(script, encoding="utf-8")
        out.chmod(0o755)
        print(f"wrote {out}")
    else:
        sys.stdout.write(script)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
