# S001-invalidation

## Hypothesis
Dependency-aware invalidation avoids unnecessary recomputation while preserving exact semantics: `SemanticHost::prepare_delta` marks the transitive reverse closure of a change, over the old and the new dependency edges, as affected and evicts the affected derived values, so that arm A (evaluate the harness-supplied derivation only for the derived keys marked affected, write the results back) leaves the host in the same committed state, key by key, as arm G (global recomputation of every derived key after every delta) and as an independent oracle, while on the local-change class (deltas whose oracle structural closure holds at least one derived key and at most 5% of the derived keys) evaluating at most 250 per mille as many derived keys as arm G. Graphs: seeded DAGs of 256, 1024 and 4096 keys of five topology classes (layered fan-in, deep chains, diamonds, hub-and-spoke, lossy min/parity derivations); update streams: leaf edits, hub edits, rewiring, key removal, no-op writes, mixed batches, cycle-creating attempts and MissingDependency probes. The recomputation is a pure derivation supplied by the harness (`ptr-semdb` has no query engine); the hub class is an ungated control; unnecessary invalidations (a recomputed key whose value did not change) are descriptive, because SemDB has no early cutoff; wall-clock times are descriptive.

## Primary metrics
The 33 integer metrics of `experiment.toml` (the nine hard counters, the coverage counters, precision and recall in per mille, the evaluation counts and the locality ratios); the wall-clock times are in `timing_ns` and are never judged.

## Baseline
global recomputation of every derived key after every delta

## Falsification
See `experiment.toml` and [criteria.toml](criteria.toml): nine falsifier rules (`missed_invalidations`, `closure_mismatches`, `semantic_divergences`, `noop_revision_bumps`, `revision_mismatches`, `unexpected_refusals`, cycle refusals, MissingDependency refusals, and `locality_ratio_permille_local <= 250`) and thirteen coverage rules, every seed must pass.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.

## Status of this directory
S001 is still `planned`: the harness (`bins/ptr-bench/src/experiments/s001/`, subcommand `invalidation <seed>`), the draft preregistration ([DRAFT_PREREGISTRATION.md](DRAFT_PREREGISTRATION.md)), `criteria.toml`, `aggregate.py` and the mutation plan ([tests/mutations.toml](tests/mutations.toml)) are in place, but no `[preregistration]` table is in `config.toml`, the experiment is not listed in `experiments/preregistration.toml`, and no seed has been run for the record. `config.toml`'s `[pinned_thresholds]` holds the choices the harness compiles in (a drift test compares them). The owner decisions of 2026-10-09 are in `config.toml` and in section 3 of the draft.

Stated limits: the recomputation is a harness-supplied derivation (the crate has no query engine); `prepare_delta` clones the whole state and `affected_by` rebuilds the reverse map per call, so only the recomputation, not the invalidation step, has locality; synthetic seeded DAGs; the locality threshold (250 per mille) only catches a host that over-invalidates by more than a factor of five on small changes, the exactness of the affected set being `closure_mismatches`.
