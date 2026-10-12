# Q003 supersession record

Q003 is `superseded` after one seed run, seed 17, which ended with exit status 2 before any product call. No other seed
ran, no `run.json`, `metrics.json` or `mutations.json` exists, and this status is not evidence about `ptr-pg`'s retrieval: it
neither confirms nor falsifies the hypothesis.

What happened: the preregistration was frozen at `07c44206` (tag `q003-prereg`), prepared at `e098f784`, and seed 17 was run
by `scripts/run_experiment.py`; its record is
`results/run-20261010T005730.376632Z-seed-17.json` (status `failed`, `exit_code` 2, committed at `a992b0f4`). The harness built
(2 m 16 s) and stopped at its first check with "postgres-hybrid: the strong arm needs an interpreter with torch and
transformers: pass --strong-python or set PTR_Q003_STRONG_PYTHON". Its stdout is empty: no metric was printed and no
PostgreSQL call was made.

Cause: the runner passes a listed experiment's command only an allowlisted environment (`COMMAND_ENVIRONMENT`: `PATH`, `HOME`,
`TMPDIR`, `TEMP`, `TMP`, `LANG`, `LC_ALL`, `LC_CTYPE`, `CARGO_HOME`, `RUSTUP_HOME`, plus `FIXED_ENVIRONMENT`), so
`PTR_PG_EXPERIMENT_DSN`, `PTR_Q003_STRONG_PYTHON` and `PTR_Q003_MODELS_ROOT`, which the harness needs and the preregistration
names, never reached it. The freeze did not foresee that the runner would not forward them.

The seed is spent (a listed experiment runs each seed once), and the frozen manifest cannot declare the variables after the
fact, so Q003 cannot be repaired in place. Owner decision of 2026-10-10: a successor and a runner extension.

What stays as it was: the Q003 manifest (but for its status), configuration, preregistration text, criteria, harness
history and both records (the prepare record and the seed 17 record) are not amended.

Successor: **Q003-v2** (`experiments/retrieval/Q003-v2-postgres-hybrid`). The harness, criteria, thresholds, workloads, mutation
plan and aggregator are the same in every value (the experiment id aside); the manifest additionally declares
`command_environment = ["PTR_PG_EXPERIMENT_DSN", "PTR_Q003_STRONG_PYTHON", "PTR_Q003_MODELS_ROOT"]`, which the runner now passes
through by name, recording only each name and whether it was set (never a value), and it refuses before a seed is reserved when a
declared variable is unset. Seeds 17, 29, 43, 71 and 101 are kept for Q003-v2 because Q003's seed 17 observed nothing.

This status is not evidence.
