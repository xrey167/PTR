# L001-revocation-crash

## Superseded by L001-v2

L001 is `superseded`. Its probes check one subject and one stale action (`stale_action_accepted`)
and compare only the number of recovered events, against the v1 journal format, with no
independent oracle, no preregistration, no criteria and no mutation evidence. The experiment
that replaces it is [L001-v2-revocation-crash](../L001-v2-revocation-crash/README.md): the same
claim over every enumerated crash, power-loss and leadership schedule, with an independent oracle
for every (subject, generation) of a seeded history, baselines, controls and a mutation plan.

Nothing below is changed or retracted: the numbers are still what they were, evidence about the
earlier v1 journal format and the scoped probes named here, and `results/` is untouched. They
support no claim about the current journal format, and L001-v2 does not reuse them.

## Hypothesis
Can revoked generations ever resurrect across injected crashes?

## Primary metrics
false accept count; recovery consistency

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.

## Historical evidence

The numbers below (`results/run.json`, `results/process_crash_run.json` and their metrics) were recorded on 2026-09-18, before the PTRLOG02 journal format landed (`069d4b0`, 2026-09-19). They are evidence about the earlier v1 journal format, not measurements of the current implementation. Re-running L001 against PTRLOG02 is still to do, and until then no claim about the current format follows from them.

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
## Real child-process abort evidence

[process_crash_run.json](results/process_crash_run.json) and [process_crash_metrics.json](results/process_crash_metrics.json) record a second scoped slice:

- 5 seeds × 50 child-process aborts = 250 cases
- 0 stale-generation false accepts
- 0 recovery-consistency failures
- 0 incomplete-tail truncation failures
- 0 child processes that unexpectedly exited successfully

Each child durably committed a generation and its revocation, began writing a subsequent incomplete record, flushed the partial tail to the OS and then called `abort()`. The parent process reopened the ledger, verified recovery/truncation, replayed lifecycle state and checked that generation 1 remained unusable.

This is stronger than the original in-process partial-tail probe, but it still does not model a power loss during the revocation fsync itself, raft-engine/raft-rs leadership changes, or network partitions. L001 therefore remains `running`.
