# M002-v10-sweep supersession record

M002-v10-sweep is `superseded`. Its only declared seed (17) was started on
2026-10-08 at 20:01:35 UTC from the clean commit 8ce63560 and the run was killed
by a container restart before any output. Its record
`results/run-20261008T200135.382237Z-seed-17.json` has status `started`, no
`stdout` and no exit code, and is kept exactly as it was written. No sweep
result of study M002-v10 (no row, no learning rate, no timing) was observed by
anyone, so the interruption carries no outcome.

The runner runs a listed experiment's seed once and counts a `started` record as
a run, so the seed is spent and this one-seed sweep can never complete. The
manifest and configuration are not amended (only the status moves). The
successor is **M002-v10-sweep-r2**: the identical sweep (protocol, grid, arms,
seed 17, selection rule, command) under a new id. Only its record feeds
`research/falsification/A0-baseline-v1/lr_M002-v10.tsv`;
`scripts/a0_baseline_study.py extract-sweep` refuses a record of this
experiment. Full account and the generic procedure:
`research/falsification/A0-baseline-v1/DEVIATIONS.md`.

This status is not evidence and says nothing about typed attention.
