# Deviations of the A0 baseline study v1

`criteria.toml` and `PREREGISTRATION.md` are digest-pinned by the prepared
experiments of the study and are not edited after the freeze. Whatever happens
to the study after it, and was not foreseen by that text, is recorded here, in
order, append-only: what happened, what was and was not observed, what was
done, and what the final `RESULTS.md` must say. This file changes no criterion,
threshold, protocol value, seed or rule.

## D1: M002-v10-sweep interrupted, successor M002-v10-sweep-r2

**What happened.** On 2026-10-08 the single declared run of `M002-v10-sweep`
(seed 17) started at 20:01:35 UTC from the clean commit 8ce63560 (`research:
record M001-v5-sweep seed 17`) and was killed by a container restart before it
produced any output. Its record
`experiments/model/M002-v10-sweep/results/run-20261008T200135.382237Z-seed-17.json`
has status `started`, no `stdout` field and no exit code. The record is
committed exactly as the runner wrote it.

**What was observed: nothing.** No row, learning rate, accuracy, NLL, timing
line or log of study M002-v10's sweep was seen by anyone. The interruption
therefore carries no information about any learning rate or arm, and the choice
to repeat the sweep cannot depend on an outcome. (The 5-step plumbing rehearsal
before the freeze, recorded in the registration work, ran on throw-away rows in
a scratch copy; none of it is committed or feeds an lr file.)

**Why a successor.** The runner runs a listed experiment's seed once and counts
a record of status `started` as a run (`scripts/run_experiment.py`,
`seed_runs`). Seed 17 is spent, the sweep declares only that seed, so
`M002-v10-sweep` can never complete. `PREREGISTRATION.md` (No reruns) says a
failed study is not rerun and not patched and that a successor with a new id is
created; this is that, applied at the level of the sweep experiment, which is a
preparatory step and not a hypothesis test.

**What was done.**

1. The interrupted record was committed as it is (evidence; never edited).
2. `M002-v10-sweep` was marked `superseded` (manifest status, registry,
   `SUPERSEDED.md`); its configuration, digests and listing in
   `experiments/preregistration.toml` stay as they were frozen.
3. `M002-v10-sweep-r2` was registered (`scripts/a0_baseline_study.py render
   M002-v10-sweep-r2`), listed with the same required keys plus
   `supersedes = "M002-v10-sweep"`, and freeze-ready at `planned`. Every
   protocol value (study `m002-v10`, arms `factorized-v2,factorized-v2-off,full,plain`,
   grid 0.0025, 0.005, 0.0125, 0.025, seed 17, 1500 steps, folds, the selection
   rule), the phase `baseline-v1-sweep`, the command and every pinned file
   digest are derived from the same `criteria.toml` and are identical to those
   of the superseded sweep. The pinned `model/burn-a0` tree digest at the freeze
   must equal the one of the superseded sweep (f19a68f8...); a different one
   would be a second deviation to record here before any run.
4. The confirmatory `M002-v10` (still planned) names `M002-v10-sweep-r2` as its
   `sweep_experiment`. The learning-rate file keeps its name
   `research/falsification/A0-baseline-v1/lr_M002-v10.tsv` (pinned by the
   gate and the decider). Only the record of the study's current sweep can feed
   it: `a0_baseline_study.py extract-sweep` refuses the superseded sweep's
   records.
5. `PREREGISTRATION.md` says "the three per-study sweep experiments"; for
   M002-v10 read `M002-v10-sweep-r2` there. The text itself stays frozen.

**What the final report must say.** `RESULTS.md` states that the M002-v10 lr
file comes from `M002-v10-sweep-r2` after an interrupted first attempt, with
the facts above.

**Residual risk.** The successor runs at a later commit than the first attempt
(the tooling for it was added in between); `model/burn-a0`, the criteria, the
decider and the dataset are unchanged, which the freeze digests and the
`implementation_tree_git_digest` equality check.

## D2: M002-v10 seed 101 interrupted (container restart)

**What happened.** On 2026-10-09 the confirmatory `M002-v10` seed 101 started at
02:56:53 UTC from the clean commit c87d4a1f (`research: record M002-v10 seed 71`)
and was killed by a container restart before any output. Its record
`experiments/model/M002-v10-typed-attention/results/run-20261009T025653.762171Z-seed-101.json`
has status `started`, no `stdout` and no exit code, and is committed exactly as
the runner wrote it (commit 0d3e6060).

**What was observed: nothing** of this seed. The seed is spent
(`scripts/run_experiment.py`, `seed_runs`), so `M002-v10` can never have five
completed records.

**What was done.** Nothing in v1: D3 supersedes the study.

## D3: study v1 replaced by study v2 (decider identity defect; no outcome observed)

**What happened.** On 2026-10-09 the owner decided that the confirmatory v1
studies M001-v5, M002-v10 and M003-v2 are replaced by the complete successor
study `A0-baseline-v2` (M001-v6, M002-v11, M003-v3; sweeps repeated, all
confirmatory runs repeated). Reason, procedural: the v1 decider
(`scripts/aggregate_a0_baseline.py`, through the pinned core
`scripts/aggregate_m002_v5.py`, `IDENTITY_FIELDS`, checked at about lines 455-482
of the decider) requires all five run records to name the same `git_sha`, but
the run protocol commits each record before the next run, so every record names
a different commit and the decider refuses ("run records disagree in git_sha")
before computing anything. In addition D1 and D2 (two container restarts)
interrupted a sweep and a confirmatory seed; v1 had a procedure for the sweep
only, and its decider would have refused an interrupted confirmatory study.

**What was observed: no v1 confirmatory outcome.** The decider never ran on the
confirmatory records, nobody opened their stdout, and only record metadata
(status, `git_sha`, seed) was read to find the defect. The validation-only v1
sweeps exist and are history; v2 does not use their learning-rate files and runs
fresh sweeps.

**What was done.**

1. `M001-v5`, `M002-v10` and `M003-v2` were marked `superseded` (manifest status,
   registry, `SUPERSEDED.md` in each directory with the facts and the successor).
   Their records stay unmodified and unread. This file is not digest-pinned by
   any experiment, so the note is made here (and in the `SUPERSEDED.md` files).
2. The sweeps `M001-v5-sweep`, `M002-v10-sweep-r2` and `M003-v2-sweep` stay as
   they are (their records, lr files and statuses are unchanged).
3. `research/falsification/A0-baseline-v2/` holds the successor with the
   protocol, arms, contrasts, gates, thresholds, seeds, lr grid and selection
   rule of this study unchanged; its changes are the decider
   (`scripts/aggregate_a0_baseline_v2.py`), a stated record-identity rule, an
   interruption outcome (INCONCLUSIVE-INTERRUPTED) and the ids. M004-v3 is not
   created; M004 stays deferred via `M004-v2`.

**What the final report must say.** v1 produced no result; the verdicts are
those of v2, and a v1 confirmatory record must not be quoted.

## Procedure: an interrupted run of a listed experiment

Applies when a listed run left a record with status `started`, no `stdout`, no
exit code, because the process or its host was killed (container restart, power
loss, OOM-kill of the whole container), and nobody saw any output of it.

1. Do not delete, edit, move or rerun it. The runner keeps the seed spent
   (`seed_runs`), also from the copy in git's own directory.
2. Establish and write down "no outcome observed": the record has no `stdout`
   and no exit code; no other channel (terminal, log, partial row file) showed
   rows. If anything was seen, say exactly what; the successor is then
   outcome-informed and no protocol value may change in it either.
3. Commit the record alone: `research: record interrupted <id> seed <n> (killed
   before any output)`.
4. Supersede: set the manifest `status` and the registry status of the
   interrupted experiment to `superseded` (nothing else of it changes), add
   `SUPERSEDED.md` stating the facts. Its `preregistration.toml` listing stays.
5. Successor, by kind of experiment:
   - **Sweep** (`<study>-sweep`): add the next id (`<study>-sweep-r2`, then
     `-r3`, ...) to `SWEEP_SUCCESSORS` in `scripts/a0_baseline_study.py`, run
     `render <successor id>` (refuses unless the predecessor is superseded),
     set nothing else by hand; the generator carries the confirmatory
     experiment's `sweep_experiment` to the current sweep. The gate
     (`a0_baseline_v1_freeze_errors`) binds the successor to `criteria.toml`
     like any sweep, requires `supersedes` to name the predecessor and the
     predecessor to be superseded.
   - **Confirmatory** (`M001-v5`, `M002-v10`, `M003-v2`, one seed of five):
     the same steps 1 to 4 apply, but a successor is NOT an `-r<n>` id of the
     same study: the decider (`scripts/aggregate_a0_baseline.py`,
     `admit_record`) admits only records whose `experiment_id` and manifest id
     equal the criteria's study id, and `criteria.toml` and the decider are
     digest-pinned by every frozen experiment. A successor is therefore a new
     study (new ids, a new directory with its own criteria copied unchanged in
     every value, preregistration and freeze) registered before any further
     run. Seeds of the original that did complete are observed and are not
     pooled with the successor (the decider refuses mixed builds anyway): the
     successor runs all five seeds, and its text discloses which original seeds
     were seen.
6. Record the case in this file as D<n>: facts, "what was observed", successor
   id, and what `RESULTS.md` must say. Freeze the successor with
   `python3 scripts/a0_baseline_study.py freeze <successor id> --write`, run
   `scripts/run_experiment.py validate` and `scripts/check_research_gates.py`,
   commit, and only then run it.

A run that finished with output and failed (exit code not 0, or a refusal) is
not an interruption: it has an outcome, is a failed run, and its successor
needs a stated reason and, where the cause was a defect, a fix recorded as such.
