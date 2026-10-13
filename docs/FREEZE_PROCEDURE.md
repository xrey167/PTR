# Freeze procedure for preregistered experiments

How a prepared experiment is frozen, run and finished with the same guarantees on any machine. The tools live in
`scripts/freeze/`; `scripts/tests/test_freeze_tools.py` tests them (`make freeze-tools-test`). Background:
`experiments/README.md`, `experiments/preregistration.toml`, `scripts/experiment_records.py`.

## Validation

On 2026-10-12 the tooling replayed the real E007 freeze in a scratch clone (from the pre-freeze commit 03f71143, tag
`e007-replay`, never pushed): the generated script's freeze commit has the same parent and no diff against the real freeze
commit 3c1076b9; the whole mutation check killed 59 of 59 and the pilot 1001 rerun was equal on all 137 outcome entries.
A second replay from the mutation-evidence commit (aa6e3b2e, `--resume-from 7`) ran the pilot, preflight, prepare,
the five seeds from `config.toml` and the aggregate (COMPLETED), then `finish_experiment.py status` and `stale`. It found
one defect, now fixed with a regression test: step 8 left `./target` unless `FREEZE_CLEAN_TARGET=1`, and the runner refuses
a checkout holding git-ignored files. When the tools are used from outside the checkout, pass `--root <checkout>` to
`finish_experiment.py` before its subcommand. Unit tests: `make freeze-tools-test`.

## Roles

Only the head freezes. A channel (a coding agent or a person working on the experiment) builds the harness,
`PREREGISTRATION.md`, `criteria.toml`, the mutation plan and the pilots, leaves the pins as `must-be-pinned-...` and
the owner decisions as `must-be-signed-...`, and never signs, tags or runs confirmatory seeds. The head signs the
decisions, pins the digests, tags the freeze commit and runs the seeds. Nothing here changes that.

## Tools

| Tool | Purpose |
| --- | --- |
| `freeze_sign.py` | Pins `criteria_sha256` and `preregistration_text_sha256`, signs `--sign must-be-signed-<name>=<int>`, sets `planned` to `prepared` in `experiment.toml` and the registry and adds the two preregistration digests. `--dry-run` writes nothing. All-or-nothing. |
| `freeze_guard.sh` | `guard_on` / `guard_off`: if the script exits between signing and the freeze commit, tracked files and the index are restored with `git reset --hard <guard head>` (untracked files are not touched; the guard checks afterwards and prints `GUARD: restore FAILED` if anything tracked still differs). |
| `freeze_lib.sh` | `waitdisk`, `clean`, `only_paths`, `add_existing` (explicit paths, never `git add -A`). |
| `digest_parents.py` | After a merge, compares every experiment's rule digest with both parents. |
| `make_freeze_script.py` | Generates the freeze script of one experiment from a spec. |
| `compare_pilot.py`, `config_seeds.py` | Pilot comparison with timing keys stripped; the seeds of `config.toml`. |
| `finish_experiment.py` | After the aggregate: status edits, STATUS.md line, `results/STALE.toml`. Commit-free. |

## Steps and the reason for each

Run everything in a worktree on the branch that carries the experiment, from its root. Environment: `PYTHONDONTWRITEBYTECODE=1`.
Secrets (DSNs, tokens) go into a file the generated script sources quietly (`env_file`); the values are never printed or committed.

0. **After merging branches into the freeze branch**, run `python3 scripts/freeze/digest_parents.py` (a merge in progress:
   HEAD and MERGE_HEAD; a committed merge: HEAD^1 and HEAD^2; or name two parents). `MISMATCH` or `DROPPED` are errors
   (exit 1): a new `[experiment.X]` block must not split another experiment's rule lines in
   `experiments/preregistration.toml`, because the manifest pins the digest of those lines. `DIFFERS-BETWEEN-PARENTS`
   is information (the merge took one side). Reason: wave 13 had to check this by hand after every merge.
1. **Write the spec** (TOML, see `make_freeze_script.py --help`), generate and syntax-check the script:
   ```bash
   python3 scripts/freeze/make_freeze_script.py --spec freeze-e008.toml --output /tmp/freeze_E008.sh   # runs bash -n itself
   ```
   The tag is explicit and never derived (`q003v3-prereg` for Q003-v3, not `q003-v3-prereg`; a derived name was wrong once and
   the STALE marker needed a manual fix). Pre-sign checks and the experiment's own tests go into the spec. Reason: the
   scripts of wave 13 were hand-copied per experiment and drifted.
2. **Dry-run (script step 1).** `freeze_sign.py ... --dry-run` is the first action, before any commit or file change; it checks the
   id is listed in `experiments/preregistration.toml`, placeholders on the *parsed* TOML values, the digests and the status.
   Reason: an inline text replacement once tripped on the word `must-be` in a comment (the F007 false alarm) and a hidden
   placeholder is worse; comments never count now.
3. **Work commit (step 2).** The channel's uncommitted work becomes one commit with explicit path prefixes (`only_paths`), anything
   else stops the script. Reason: a freeze commit must contain only the signing; everything the experiment tests must already be
   committed so the mutation check and the pilot run on exactly the frozen tree.
4. **Pre-sign checks on the committed tree (step 3).** In this order: `check_research_gates.py` (a gate error of another experiment, or an
   experiment missing from the gate list, stops before any file changes), `check_repo.py`, `update_component_docs.py --check`,
   `mutation_check.py <ID> --check` (plan anchors only), `run_experiment.py validate`, `EnrolledExperimentTests` plus the
   experiment's aggregate tests, `cargo fmt`, `cargo clippy --workspace --all-targets --locked -- -D warnings`, crate and feature
   tests, then the spec's extra checks. Then the tree must be clean. The `make ci-*` recipes hard-code `cargo +stable`; when the local
   stable is older than the MSRV (1.99.0) run the recipe with the pinned toolchain
   (`make -n ci-<job> | sed 's/cargo +stable/cargo +1.99.0/g' | bash -e`).
5. **Sign under guard (step 4).** `guard_on`; `freeze_sign.py`; `update_component_docs.py --write`; `validate`; gates; the tests again;
   only `experiments/`, crate READMEs and `docs/components/` may have changed. Any exit here (also a failed `git commit` after a successful `git add`) restores tracked files and index to the pre-sign commit.
6. **Freeze commit and tag (step 5).** `git add -- experiments crates docs`, one commit, `git tag <explicit tag>`, `guard_off`.
   After this the experiment is `prepared` and frozen; a code change needs a successor experiment, never a patch.
7. **Whole mutation check (step 6).** `mutation_check.py <ID>` on the committed tree, with `CARGO_TARGET_DIR` unset: the script runs
   `ROOT/target/release/<package>` and ignores the variable (the first Q003-v3 run stopped here). Every mutation must be killed;
   afterwards only `<dir>/results/` may be changed; the evidence is committed. No edit to a mutation-bound file may follow.
8. **Pilot rerun (step 7, optional).** The pilot seed (a seed outside the declared ones, >= 1001) is rerun on the frozen commit and compared
   recursively with the committed reference, the spec's timing keys stripped at every depth (`compare_pilot.py`). A difference stops.
9. **Preflight and prepare (step 8)**, **seeds (step 9)** from `config.toml`, each seed once; an exit other than 0 or 1 stops
   (1 is a recorded product finding, 2 a harness stop). Each seed is committed on its own. **Aggregate (step 10).**
10. **Finish** (below), after the aggregate, as three commits.

GPU index on the two-card desktop (RTX 3090 x2): a spec with `gpu = 0` or `gpu = 1` makes the generated script export
`CUDA_VISIBLE_DEVICES=<index>` and prefix every log line with `[ID gpu=N]`. Without `gpu` nothing is exported (CPU experiments). The export
reaches the script's own cargo, pilot and mutation steps, but not the seed runs: they go through `scripts/run_experiment.py run`, whose command
environment allows only `PATH`, `HOME`, `TMPDIR`, `TEMP`, `TMP`, `LANG`, `LC_ALL`, `LC_CTYPE`, `CARGO_HOME`, `RUSTUP_HOME` and the declared `PTR_*`
variables, so `CUDA_VISIBLE_DEVICES` itself does not reach a seed run. GPU pinning of frozen seed runs (OD-3, decided 2026-10-13, no runner change): a GPU experiment's manifest declares `PTR_CUDA_DEVICE` in `command_environment` before its freeze (the runner passes declared `PTR_*` variables through, `scripts/run_experiment.py`), its entrypoint wrapper sets `CUDA_VISIBLE_DEVICES=$PTR_CUDA_DEVICE` and prints the index into its result JSON (the run record names the declared variable, never its value), and the freeze script's `gpu` field exports both variables. An experiment frozen without that declaration runs on card 0, one at a time. Use a separate worktree,
`FREEZE_LOGS` and `CARGO_TARGET_DIR` per script. Never run two scripts in one worktree.

## Runtime variables of the generated script

`FREEZE_TRAILER` (the attribution lines of the commit message, set by the caller; empty by default), `FREEZE_LOGS`, `FREEZE_DISK_FLOOR_GB`
(waitdisk floor, default from the spec, 6), `FREEZE_DISK_PATH`, `FREEZE_PILOT_TIMEOUT`, `FREEZE_SKIP_SEEDS`, `FREEZE_CLEAN_TARGET=1`
(remove `./target` after heavy steps; step 8 always removes it, because the runner refuses a checkout holding git-ignored files), `FREEZE_RESUME_FROM`.

## Resuming after a stop

A stop prints `FREEZE-STOP step N: reason`. Before the freeze commit the guard has reset tracked files and index (look for the `GUARD:` line; `GUARD: restore FAILED` means inspect `git status` by hand); fix the cause and rerun the
script from the start (the work commit is idempotent). After the freeze commit and tag, never sign again: check that `HEAD` is the tagged
commit (resume at step 6) or that the tag is an ancestor of `HEAD` (later steps) and that the tree is clean, then

```bash
git status --porcelain            # must be empty
test "$(git rev-parse HEAD)" = "$(git rev-list -n1 <tag>)" && echo "at the freeze commit"
bash /tmp/freeze_E008.sh --resume-from 6    # 6 mutation, 7 pilot, 8 prepare, 9 seeds, 10 aggregate
```

The script performs the same checks itself and refuses otherwise. For step 9 use `FREEZE_SKIP_SEEDS="17 29"` for seeds already recorded.
If the code under test must change after the tag, stop: that is a new experiment.

## Finish (after the aggregate)

`finish_experiment.py` only edits files; the caller commits with explicit paths (never `git add -A`). Three commits: the aggregate, the status and
STATUS.md line, the stale marker (it needs the shas of the first two). Reason for the marker: the documentation commit counts as a change of the
provenance files, so the results are stale since it, and `check_research_gates.py` requires the marker to say so; write it with the real shas
and the explicit tag, not by hand.

```bash
set -eu
ID=E008; D=experiments/system/E008-split-pod-extensions; TAG=e008-prereg
T=$(printf '\n\n%s' "${FREEZE_TRAILER:-}")
git add -- "$D"                                    && git commit -q -m "research: $ID aggregate$T"
RES=$(git rev-parse HEAD~1)                        # the last seed record, before the aggregate commit
python3 scripts/freeze/finish_experiment.py status --id $ID --dir $D --status-line-file /tmp/status-line.md \
        --anchor '- **Q002 evidence-promotion completed'
python3 scripts/update_component_docs.py --write | tail -1; [ "${PIPESTATUS[0]}" -eq 0 ]
python3 scripts/run_experiment.py validate | tail -1; [ "${PIPESTATUS[0]}" -eq 0 ]
git add -- "$D/experiment.toml" experiments/registry.toml STATUS.md crates docs && git commit -q -m "research: $ID status; record the result in STATUS$T"
DOC=$(git rev-parse HEAD)
python3 scripts/freeze/finish_experiment.py stale --id $ID --dir $D --results-sha "$RES" --stale-since "$DOC" --tag $TAG
git add -- "$D/results/STALE.toml"                 && git commit -q -m "research: mark $ID results stale since the documentation commit$T"
```

`status` takes the final status from `results/run.json` (`recommended_status`: `completed` or `failed`); NOT_SHOWN and INVALID recommend none and
the script refuses, the experiment stays `prepared`. The anchor must be the start of exactly one STATUS.md line (zero or several matches are refused); the new line is inserted right after that line. The three file writes are staged and renamed together, with rollback.
