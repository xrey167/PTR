# M001-v5 supersession record

M001-v5 is `superseded`. Its five seeds (17, 29, 43, 71, 101) ran to exit code 0, one record each, committed one by one (2703213a, d2bcef73, 383865cb, 2d9425e5, 473a6b25 are the commits the records name).

The reason is **procedural and no outcome was observed**. The decider of study v1
(`scripts/aggregate_a0_baseline.py`, through the pinned core
`scripts/aggregate_m002_v5.py`) requires the five run records of a study to name
the same `git_sha`, but the run protocol (`scripts/run_experiment.py`) commits
each seed's record before the next run, so the records of a correctly executed
study name 5 different commits and the decider refuses ("run records disagree
in git_sha") before it computes anything. The decider was never run on this
experiment's records; nobody opened their stdout; only metadata fields (status,
`git_sha`, seed) were read to diagnose the defect. The records, if any, stay in
`results/` unread and unmodified as history and must not be read or compared with
the successor's results before the successor's decision. The manifest and the
configuration are not amended: only the status moves (this experiment's
digest-pinned files, the criteria and the decider stay as frozen).

Successor: **M001-v6** of the A0 baseline study v2
(`research/falsification/A0-baseline-v2/`), a complete successor with the same
protocol, arms, contrasts, gates, thresholds, seeds, learning-rate grid and
selection rule, a decider (`scripts/aggregate_a0_baseline_v2.py`) that accepts
the per-seed commits of a correct run loop and names an interrupted seed
INCONCLUSIVE-INTERRUPTED, fresh sweeps and all confirmatory runs repeated. The
learning-rate file of this experiment
(`research/falsification/A0-baseline-v1/lr_M001-v5.tsv`) is not used by the
successor. Account: `research/falsification/A0-baseline-v2/DESIGN.md` ("Why v2
replaced v1") and `research/falsification/A0-baseline-v1/DEVIATIONS.md` (D3).

This status is not evidence and says nothing about semantic slots.
