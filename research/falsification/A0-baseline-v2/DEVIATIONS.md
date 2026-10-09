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

Log: D1, D2.

### D1: M003-v3 seed 71 interrupted by a container restart (2026-10-09)

Facts, from the record metadata only (`experiments/model/M003-v3-latent-recurrence/results/run-20261009T201837.952950Z-seed-71.json`, committed alone as `542fd559`): `status` is `started`, there is no `exit_code`, no `finished_at`, and no `stdout` or `stderr` key. The container hosting the run was restarted while the seed was running. M003-v3 seeds 17, 29 and 43 had completed with exit 0 and are committed (`50a429fd`, `8fedda14`, `c47fce9e`). Seed 101 was never started: the runner stops a study after a seed that did not complete.

What was observed: no output of seed 71. The record holds none and no terminal, log or partial row file showed rows. The record's contents beyond that metadata were not read, and the rows of seeds 17, 29 and 43 were not read for this entry.

Outcome by the preregistered interruption rule (`DESIGN.md`, "Interruption rule"): M003-v3 is **INCONCLUSIVE-INTERRUPTED**. No statistic is computed and the completed seeds are not read by the decider. The seed is spent, M003-v3 is not rerun, patched or completed, and seed 101 is not run. The experiment stays `prepared` until the owner registers a successor study. M001-v6 and M002-v11 are independent experiments that completed all five seeds with exit 0 and are decided separately.

What `RESULTS.md` says: M003-v3 has no result, no verdict on latent recurrence and no quality claim; the outcome is INCONCLUSIVE-INTERRUPTED.

### D2: the published aggregates are bound to the preregistration and the seed records (2026-10-09)

Facts: the pinned v2 decider (`scripts/aggregate_a0_baseline_v2.py --publish-results`) publishes `run.json` in the v1 envelope: it names the criteria, decider and core digests and each seed record's canonical SHA-256 under `source_records`, but not the three fields `scripts/check_research_gates.py` requires on the aggregate of a listed experiment (`preregistration_sha256`, `preregistration_rules_sha256`, and `seed_records`, the SHA-256 of each seed's run record file). A first publish of M001-v6 and M002-v11 in that envelope failed the gate. It was a local, unshared commit and was discarded before anything was delivered or merged (owner decision 2026-10-09).

What was done: the decider was run as preregistered and its outputs (`DECISION.toml`, `paired-ci-95.json`, `metrics.json`, `run.json`) were kept byte for byte, except that `run.json` was republished through `experiment_records.publish_aggregate` with exactly those three fields added: the two digests copied from the experiment's `experiment.toml` and `seed_records` from `experiment_records.seed_record_digests` over the committed seed records. `metrics_sha256` is recomputed over the unchanged `metrics.json`. No criterion, threshold, metric, verdict, label or seed changed, the decider and its pins were not edited, and no seed record was read for this beyond its file digest. M003-v3 publishes no aggregate (D1) and is untouched.

What `RESULTS.md` says: nothing changes; the verdicts are the decider's.

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
