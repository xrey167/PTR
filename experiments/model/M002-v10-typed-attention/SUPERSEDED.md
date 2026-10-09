# M002-v10 supersession record

M002-v10 is `superseded`. Seeds 17, 29, 43 and 71 ran to exit code 0, one record each, committed one by one. Seed 101 was started on 2026-10-09 at 02:56:53 UTC from the clean commit c87d4a1f and was killed by a container restart before any output: its record has status `started`, no `stdout` and no exit code, and is committed exactly as the runner wrote it (commit 0d3e6060). The seed is spent, so this experiment could never complete; under the v1 decider the study could only be refused.

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

Successor: **M002-v11** of the A0 baseline study v2
(`research/falsification/A0-baseline-v2/`), a complete successor with the same
protocol, arms, contrasts, gates, thresholds, seeds, learning-rate grid and
selection rule, a decider (`scripts/aggregate_a0_baseline_v2.py`) that accepts
the per-seed commits of a correct run loop and names an interrupted seed
INCONCLUSIVE-INTERRUPTED, fresh sweeps and all confirmatory runs repeated. The
learning-rate file of this experiment
(`research/falsification/A0-baseline-v1/lr_M002-v10.tsv`) is not used by the
successor. Account: `research/falsification/A0-baseline-v2/DESIGN.md` ("Why v2
replaced v1") and `research/falsification/A0-baseline-v1/DEVIATIONS.md` (D3).

This status is not evidence and says nothing about typed attention.
