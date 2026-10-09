# R001-mailbox-backpressure

## Hypothesis
Do bounded typed mailboxes give predictable overload behavior?

## Primary metrics
tail latency; drops/refusals; memory

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.

## Draft status and scope (2026-10-09)
R001 is still `planned`: the harness, the criteria, the aggregator and the mutation plan are built, but nothing is frozen and
no run is evidence. The draft text is [DRAFT_PREREGISTRATION.md](DRAFT_PREREGISTRATION.md) (it becomes the preregistration
at the freeze, together with a `[preregistration]` table in `config.toml` and an entry in
`experiments/preregistration.toml`); the decision rules are [criteria.toml](criteria.toml), applied by
[aggregate.py](aggregate.py); the defects the harness must detect are [tests/mutations.toml](tests/mutations.toml).

Binding owner decisions (2026-10-09): the unit under test is `ptr_exec::Mailbox<T>` and its cloneable `MailboxSender<T>`
only; the threaded arm is in scope; latency is gated in virtual ticks (an admitted message waits at most capacity x S_max),
while wall-clock p99 and the resident set are descriptive and never gated; the baselines are an unbounded `std::sync::mpsc`
channel and a `VecDeque` reference queue used as a differential oracle; the `RequestAdmission` arm is deferred.

The harness is `bins/ptr-bench/src/experiments/r001/` (`ptr-bench mailbox-backpressure <seed>`, one JSON object per seed,
exit 0 / 1 / 2 as for R007). It drives the real mailbox through a deterministic discrete-event simulation (capacities 1, 8
and 128, payload bodies of 64 and 4096 bytes, phases underload, overload x2, overload x8, burst, recovery, and a scheduled
close), checks an ownership ledger and a reference model at every event, runs an unbounded channel through the same oracle as
a negative control, and runs four producer threads against one consumer. Stated limits: one single-server model with
uniform service times, no isolate supervision, no async mailbox, wall-clock numbers are descriptive.
