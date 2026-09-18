# L001-revocation-crash

## Hypothesis
Can revoked generations ever resurrect across injected crashes?

## Primary metrics
false accept count; recovery consistency

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.

## Reference-scope evidence

The first executed slice is recorded in [results/run.json](results/run.json) and [results/metrics.json](results/metrics.json). It covers five seeds × 100 partial-tail crash/reopen probes against the reference FileLedger.

Observed in this **limited scope**:

- 500 cases
- 0 stale-generation false accepts
- 0 recovery-consistency failures
- 0 crash-tail truncation failures

L001 remains `running`, not completed: real process-kill/fail-rs schedules, production durable storage, Raft leader/partition cases and broader crash points are still required.
