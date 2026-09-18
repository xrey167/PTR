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

## fail-rs instrumentation

`ptr-ledger` now exposes feature-gated failpoints at:
- `ledger.before_record_write`
- `ledger.after_length_before_payload`
- `ledger.after_payload_before_sync`
- `ledger.after_sync_before_memory`

CI exercises the partial-record panic path separately. This strengthens L001's fault-injection harness, but L001 remains open until process-abort and production-backend crash schedules are archived as experiment evidence.
