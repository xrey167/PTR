# S002-snapshot-cancellation

## Hypothesis
Can stale reasoning be cancelled safely on revision changes? Cooperative, revision-granular cancellation of a steppable run before the effect boundary never weakens the hard boundary (no stale action executes or is recorded as attempted, in either arm) and a cancelled run takes no step on a replaced snapshot. Capability exercised in a harness only: no shipped host (ptr-server, ptrd) steps runs, the blocking entry points force cancellation off, and total compute (including the restarts cancellation causes) is reported, not claimed to fall.

## Primary metrics
stale action rate; wasted compute; cancellation latency

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.

## Draft status and scope (2026-10-09)
S002 is still `planned`: the product change (the steppable run and its cancellation in `crates/ptr-runtime`), the harness, the criteria,
the aggregator and the mutation plan are built, but nothing is frozen and no run is evidence. The freeze waits for the invariant
review the owner decision of 2026-10-09 requires. The draft text is [DRAFT_PREREGISTRATION.md](DRAFT_PREREGISTRATION.md) (it
becomes the preregistration at the freeze, together with a `[preregistration]` table in `config.toml` and an entry in
`experiments/preregistration.toml`); the decision rules are [criteria.toml](criteria.toml), applied by
[aggregate.py](aggregate.py); the defects the harness must detect are [tests/mutations.toml](tests/mutations.toml).

Binding owner decisions (2026-10-09): the run loop of `ptr-runtime` is a steppable run (`begin_resumable_run`,
`step_resumable_run`; the blocking entry points are wrappers with cancellation off, behaviour unchanged); the cancellation is coarse,
revision-granular and cooperative, and only before the effect boundary; the boundary stays the only safety barrier; the gated claim is
that cancellation never weakens the boundary and cuts wasted work; coverage requires the baseline (cancellation off) to show at
least one stale attempt reaching the boundary.

The harness is `bins/ptr-bench/src/experiments/s002/` (`ptr-bench snapshot-cancellation <seed>`, one JSON object per seed, exit 0 / 1 /
2 as for S001 and R001). A deterministic cooperative scheduler interleaves a scripted run with a writer (a host write that moves the
revision, one that does not, a generation superseded or revoked) at five positions of the run's life, in arm A0 (cancellation off,
the baseline) and arm A1 (cancellation on), on 1500 cases per seed; an independent revision mirror and an independent scan of the
committed ledger say what must happen. **No shipped host steps runs** (`ptr-server`, `ptrd`): the blocking wrappers force the cancellation off, so `semantic.cancel_stale_revisions` has no effect on any production path today. The claim is "a cancelled run takes no step on a replaced snapshot", not "cancellation cuts wasted work"; the headline numbers are `total_units_a0`, `total_units_a1` and `net_units_a1_minus_a0` (descriptive, ungated). Stale actions are judged against the committed ledger, not the harness' mirror. Stated limits: a scripted backend and one echoing Pod, one run at a time per runtime (S002 makes no multi-run claim: runs that
share a runtime cancel each other and can livelock, which a runtime test shows and this harness does not), a total-compute saving is not claimed.
