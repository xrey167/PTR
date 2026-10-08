# M002-v10-sweep-r2

One learning-rate sweep (validation only, seed 17) of study `M002-v10` of the A0 baseline study v1. It is a listed experiment: its command, values and seeds are frozen in `config.toml` at the freeze commit, and each seed runs once through the hardened runner. Design and frozen text: `research/falsification/A0-baseline-v1/`. Its records are A0-internal, synthetic-data evidence only (ADR-0021).

Status `planned` until the freeze: placeholders (`must-be-pinned-before-prepared`) in `config.toml` are the values only known then.

## Successor of M002-v10-sweep

`M002-v10-sweep` is superseded: its only seed (17) was spent by a run that a container restart killed before any output, so its record has status `started`, no output and no exit code, and no sweep result of the study was observed by anyone. This experiment is the identical sweep (protocol, grid, arms, seed 17, selection rule, command) under a new id, the only one whose record may feed `research/falsification/A0-baseline-v1/lr_M002-v10.tsv`. The interrupted record is kept as it was written. Account and procedure: `research/falsification/A0-baseline-v1/DEVIATIONS.md`.
