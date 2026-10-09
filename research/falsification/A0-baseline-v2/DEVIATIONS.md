# Deviations of the A0 baseline study v2

`criteria.toml` and `PREREGISTRATION.md` are digest-pinned by the prepared
experiments of the study and are not edited after the freeze. Whatever happens
to the study after it, and was not foreseen by that text, is recorded here, in
order, append-only: what happened, what was and was not observed, what was
done, and what the final `RESULTS.md` must say. This file changes no criterion,
threshold, protocol value, seed or rule.

This file is not part of any digest pin (check: `scripts/a0_baseline_study.py`
pins `criteria.toml`, `PREREGISTRATION.md`, the deciders, the core, the lock and
the runner sources, not this file), so entries are added when the events happen.

Log: (empty; no run of this study has started.)

The story of why this study exists (the v1 decider's record-identity defect and
the two interruptions of v1) is in `DESIGN.md`, "Why v2 replaced v1", and in
`../A0-baseline-v1/DEVIATIONS.md`; it is not a deviation of v2.

## Procedure: an interrupted run of a listed experiment

Applies when a listed run left a record with status `started`, no `stdout`, no
exit code, because the process or its host was killed (container restart, power
loss, OOM-kill of the whole container), and nobody saw any output of it.

1. Stop. Do not delete, edit, move or rerun it; do not run later seeds of the
   same experiment. The runner keeps the seed spent (`seed_runs`), also from the
   copy in git's own directory.
2. Establish and write down "no outcome observed": the record has no `stdout`
   and no exit code; no other channel (terminal, log, partial row file) showed
   rows. If anything was seen, say exactly what.
3. Commit the record alone: `research: record interrupted <id> seed <n> (killed
   before any output)`.
4. By kind of experiment:
   - **Sweep** (`<study>-sweep`): the confirmatory freeze must not happen yet.
     Supersede the sweep (manifest `status` and registry status to `superseded`,
     `SUPERSEDED.md` stating the facts; its `preregistration.toml` listing stays),
     add the next id (`<study>-sweep-r2`, then `-r3`, ...) to
     `SWEEP_SUCCESSORS_V2` in `scripts/a0_baseline_study.py`, run
     `python3 scripts/a0_baseline_study.py render <successor id>` (refuses unless
     the predecessor is superseded), freeze it, and run it. The generator carries
     the confirmatory experiment's `sweep_experiment` to the current sweep; the
     gate (`a0_baseline_freeze_errors`) binds the successor to `criteria.toml` and
     requires `supersedes` to name the predecessor, which must be superseded.
     `extract-sweep` refuses a record of a superseded sweep.
   - **Confirmatory** (`M001-v6`, `M002-v11`, `M003-v3`, one seed of five): the
     study's outcome is **INCONCLUSIVE-INTERRUPTED** (interruption rule, `DESIGN.md`).
     Run the v2 decider on the records that exist (`--paired-ci`, `--decision`;
     no `--publish-results`) so that `DECISION.toml` records the outcome. The
     study is not rerun, not patched and its remaining seeds are not run. The
     experiment stays `prepared` until the owner registers a successor study
     (new ids, a new directory with its own criteria copied unchanged in every
     value, preregistration and freeze), then it is marked `superseded` with
     `SUPERSEDED.md`. Seeds that did complete are observed by nobody (the
     decider does not read them) and are not pooled with a successor.
5. Record the case in this file as D<n>: facts, "what was observed", the
   outcome or successor id, and what `RESULTS.md` must say.

A run that finished with output and failed (exit code not 0, or a refusal) is
not an interruption: it has an outcome, is a failed run, and its successor
needs a stated reason and, where the cause was a defect, a fix recorded as such.
