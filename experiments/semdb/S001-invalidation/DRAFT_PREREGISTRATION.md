# S001 invalidation: draft preregistration

Pre-freeze text of the preregistration of S001; at the freeze it becomes the `preregistration_text` file of `config.toml`'s
`[preregistration]` table and its SHA-256 is frozen with the table, as R007's was. S001 is `planned`: nothing here is a result
and no run is evidence. The decisions are made by `criteria.toml` and applied by `aggregate.py`; this file gives their
reasons, the definitions, the modelling choices, what the harness cannot produce and the limits of the claim. Every criterion
is the text of `experiment.toml` (hypothesis, falsification, metrics) and the pins of `config.toml`'s `[pinned_thresholds]`;
the harness is `bins/ptr-bench/src/experiments/s001/` (subcommand `invalidation <seed>`).

Entrypoint (`experiment.toml`'s `entrypoint`, the runner's command, set at the freeze):
`cargo run --release --locked -p ptr-bench -- invalidation <seed>`

Scope: `SemanticHost::prepare_delta` (crates/ptr-semdb/src/lib.rs) marks the transitive reverse closure of a change as
affected, over the old and the new dependency edges, and evicts the affected derived values that the delta does not upsert
again; `apply_prepared` publishes the result. The claim is that, driven through its public API on seeded graphs and update
streams, this affected set is exactly the structural closure an independent oracle computes, that recomputing only that set
leaves the host in the same committed state as recomputing everything, and that on changes whose blast radius is small the
recomputation is a small fraction of the global one. `ptr-semdb` has no query engine: what "recompute" means is a pure
derivation supplied by the harness (`derive.rs`: a salted hash chain, the minimum, the parity of the sum), evaluated by the
harness for the keys SemDB marks affected. The code under test is therefore the invalidation and the transaction rules
(revision, cycles, MissingDependency), not any recomputation engine.

## 1. Decision rule

Five seeds (17, 29, 43, 71, 101), one JSON object each, `criteria.toml` applied mechanically (a "< N" prerequisite is written
`>= N`, a "> N" violation `<= N`, an "a != b" violation `a == b`). COMPLETED iff every seed meets every coverage rule and no
falsifier trips; FALSIFIED if a coverage rule (the run is vacuous) or a falsifier fails on any seed; INVALID if a record is
missing, does not describe the frozen run, or a counter is missing or mistyped, or the exit status disagrees with the rules
(INVALID says nothing about the hypothesis). The harness exits 0 when every rule holds, 1 when one fails, 2 when it cannot
run (section 7). A defect of the code under test is never exit 2: it is a falsifier. No rule changes after a run.

Coverage (the run is vacuous if unmet): deltas >= 1000; topology_classes >= 5; max_closure_depth >= 4; rewires,
removals, noop_writes, cycle_attempts, missing_dependency_probes, hub_deltas, lossy_class_deltas, value_changed_total,
evals_global all >= 1; local_class_deltas >= 100 (without a local-change class the locality rule would be vacuous).

Falsifiers: missed_invalidations > 0; closure_mismatches > 0; semantic_divergences > 0; noop_revision_bumps > 0;
revision_mismatches > 0; unexpected_refusals > 0; cycle_attempts != cycle_refusals; missing_dependency_probes !=
missing_dependency_refusals; locality_ratio_permille_local > 250.

## 2. Graphs and update streams

Three sizes (256, 1024, 4096 keys) times five topology classes, 15 graphs per seed, built from the seed
(`generator.rs`); keys are named in rank order and every edge runs from a lower to a higher index, so a graph is acyclic by
construction and the oracle asserts it.

| class | shape | derivations |
|---|---|---|
| layered-fan-in | a quarter of the keys are ground; eight layers; each key takes two or three inputs from a window of four keys of the layer below | hash chain |
| deep-chains | chains of 16 keys (depth 15); 20% of the chain keys also list a key of the neighbouring chain | hash chain |
| diamonds | groups of one ground key and four diamonds (top, left, right, bottom; the bottom is the next top), depth 12 | hash chain |
| hub-and-spoke | one hub ground key listed by every derived key, an eighth of the keys are spoke ground keys, 25% of the derived keys also list one of the previous four derived keys | hash chain |
| lossy | the layered shape; even layers take the minimum, odd layers the parity of the sum; ground values in 0..16 | min, parity |

The update stream of a graph has 150, 150 and 100 operations for the three sizes, drawn per mille: leaf edit 440, hub
edit 80, rewire 110, key removal 50, no-op write 80, mixed batch 120, cycle-creating attempt 70, MissingDependency probe 50
(`config.toml`). A leaf edit sets a ground key to another value. A hub edit sets the designated hub (hub-and-spoke) or the
ground key with the largest cone. A rewire replaces the complete input set of a derived key by one to three keys of lower
index. A removal removes a key together with every derived key that would be left without inputs (at most four keys) and
replaces the input sets of the remaining dependents. A no-op write sets a ground key to the value it holds, restates a
derived key's input set, removes an absent key, or is empty. A mixed batch is two to four leaf edits and, half of the time,
a rewire. A cycle attempt adds to a derived key's inputs one of its own descendants (or itself). A MissingDependency probe
upserts a derived key together with an input set naming a key that does not exist. The stream reads the oracle's model of
the graph, never the host; every operation except the last two is valid by construction, and the oracle must say so (else
exit 2).

## 3. Owner decisions of 2026-10-09 (binding; verbatim)

- S001: locality falsifier `locality_ratio_permille_local <= 250` (local-change class, blast radius <= 5% of derived keys); hub class ungated control; wall time descriptive only; unnecessary invalidations descriptive only (no early cutoff in the design).

Also from the same decision list: no product change is planned for S001; a real defect found in `ptr-semdb` is reported with
a reproducer, not fixed here.

## 4. Arms and oracle

For each graph the same stream goes to a real `SemanticHost` and to the oracle. The graph is first loaded into both in
chunks of 1024 keys (below `MAX_DELTA_ITEMS`).

- **Arm A (the mechanism, two phases).** Phase 1: `prepare_delta(delta)`; the harness reads `affected()` and the prepared
  view, then `apply_prepared`. It evaluates the derivation only for the derived keys of the affected set, in dependency order
  over the host's own input sets, reading the values of unaffected inputs from the host and the fresh values of affected ones
  from this pass. Phase 2: the recomputed values go back as explicit upserts in one delta, so `prepare_delta` checks every
  upsert against its inputs (`MissingDependency`) and the revision advances again. A delta the host refuses although the
  oracle accepts is `unexpected_refusals`; the oracle then does not commit it either, so the two stay in step.
- **Arm G (global recomputation, the baseline).** After every accepted delta, every derived key is evaluated bottom-up
  from the ground values. G reads the graph from the oracle's ground truth, not from SemDB, so a defect in SemDB cannot hide
  in the baseline; G is held equal to the oracle's independent fresh values after every delta (disagreement is exit 2), and
  the host's committed state is compared with it key by key.
- **Oracle (`oracle.rs`).** An edge list over interned key ids, reverse adjacency lists and a breadth-first walk for the
  structural closure (over the old edges and the new edges), peeling by in-degree for cycles, one bottom-up pass for the fresh
  value of every key with its own copy of the three derivation functions, and its own model of ground values and revision. It
  shares no code and no data structure with `ptr_semdb::DependencyGraph` (it does not call it). Its unit tests cover the
  closure, the old-edge case, no-ops, cycles, conflicts, missing inputs and commits.
- **Stale survivors.** An independent signal for missed invalidations: after phase 1, any derived value still present in the
  prepared view whose fresh oracle value differs.

## 5. Metrics (all JSON integers)

| metric | definition |
|---|---|
| deltas | accepted-by-construction deltas issued (cycle attempts and probes are separate) |
| affected_total / structural_closure_total / value_changed_total | sums over deltas of the host's affected set, the oracle's closure, and the derived keys whose fresh value changed |
| missed_invalidations (hard) | per delta, derived keys whose fresh value changed that the affected set did not contain, together with stale survivors |
| closure_mismatches (hard) | per delta, keys on which the affected set and the oracle closure differ (symmetric difference) |
| semantic_divergences (hard) | after both phases, keys on which the host's state and arm G's differ (missing, extra, or another value) |
| recall_permille / precision_structural_permille | share of value-changed keys in the affected set / share of the affected set in the closure |
| precision_value_permille, unnecessary_invalidations | descriptive: recomputed derived keys whose value changed / that did not (no early cutoff in SemDB) |
| evals_global / evals_incremental | derivation evaluations of arm G (every derived key after every accepted delta) / of arm A |
| locality_ratio_permille_local (gated) | 1000 x evals_incremental / evals_global over the local-change class |
| locality_ratio_permille_hub | the same over the hub class (descriptive control) |
| noop_revision_bumps (hard) | deltas the oracle says change nothing after which the host's revision advanced |
| revision_mismatches (hard) | after phase 1 and after phase 2, the host's revision differs from the oracle's (one bump if the delta changes the state; one more iff a derived key was evicted) |
| unexpected_refusals (hard) | the host refused a delta the oracle accepts (phase 1 or phase 2) |
| cycle_attempts, cycle_refusals, cycle_unrefused | attempts the oracle confirms close a cycle / refused as `CyclicDependency` / the difference |
| missing_dependency_probes, _refusals, _unrefused | the same for upserts whose input holds nothing; the refusal must name the oracle's pair |
| locality_violation | 1 iff the local-class ratio exceeds 250 (the rule restated as a count for the mutation checker) |
| max_closure_depth, rewires, removals, noop_writes, hub_deltas, lossy_class_deltas, local_class_deltas, topology_classes | coverage counters |

**Local-change class** (the set the locality rule is judged on): accepted deltas whose oracle structural closure holds at
least one derived key and at most 5% of the derived keys of the graph after the delta (closure_derived x 1000 <= 50 x
derived_keys). The class is defined by the oracle, so the mechanism cannot choose its own denominator; no-ops, cycle attempts
and probes are not in it. **Hub class**: edits of the designated hub of the hub-and-spoke graphs, whose closure is the whole
derived set; the ratio is expected near 1000 (a rewire may drop the hub as an input of some key).

Timings (`timing_ns`: elapsed, recompute time of both arms, phase-one and phase-two time per graph size) are never read by a
rule. `hard_failures` is the sum of the nine hard counters named in `tests/mutations.toml`.

## 6. Negative controls and mutation plan

Two controls run on one 1024-key diamonds graph with 80 operations, through the same code with a doctored affected set:
**drop-one** removes from the set arm A uses, once, a derived key whose value really changed; it must trip
`missed_invalidations <= 0`. **Everything** declares every key affected; it must trip `locality_ratio_permille_local <= 250`
(it measures 1000). If either does not trip, the harness cannot see the defect it plants and the run is exit 2.

`tests/mutations.toml` (run with `python3 scripts/mutation_check.py S001`; `--check` verifies the anchors) plants nine
defects in `ptr-semdb` (direct dependents only; affected values kept; a rewired key not counted as changed; cycles accepted;
revision bumped on a no-op; a value change marking every derived key affected; missing input accepted; an over-eager input check; an equal upsert counted as
a change) and three in the harness (oracle walk truncated; oracle expecting a bump on a no-op; arm A skipping the lossy
derivations). Every hard counter is expected by at least one. Skipping the closure over the old graph in `prepare_delta` is
not listed: it is an equivalent mutant (every edge a delta removes ends at a key whose dependency set changed, and that key
is a root of the closure over the new graph).

## 7. Exit 2 (the harness cannot run), and nothing else

A graph setup the host refuses or whose values differ from the oracle's; a generated operation the oracle does not call what
it was drawn to be (an invalid delta, a cycle attempt that is not cyclic); the oracle's and the host's revision drifting
apart outside a counted mismatch; a derived key without a fresh oracle value; an edge that does not run to a higher index;
arm G disagreeing with the oracle's fresh values; a negative control that does not trip; an argument that does not parse.
Anything the code under test does wrong (a wrong affected set, a wrong value, a wrong revision, a missed or spurious
refusal) is a counter and a falsifier.

## 8. Weak metrics and what is not measured

- Arm A evaluates exactly the set it is given, so with a correct host `affected == closure` and the locality ratio is at
  most the class bound (50 per mille); the 250 threshold only catches a host that over-invalidates by more than a factor of
  five on small changes. Exactness of the set is `closure_mismatches`, not the ratio.
- Precision against the value-changed set is descriptive: SemDB has no early cutoff (`affected_by` is structural), so
  lossy derivations produce unnecessary invalidations by design.
- The hub class is a control, not a claim.
- `unexpected_refusals` also counts phase-2 refusals.
- Wall-clock time is recorded and never judged.

## 9. Stated limits (what this experiment does NOT show)

- `prepare_delta` clones the whole state and `affected_by` rebuilds the reverse map per call (O(edges)), and `acyclic`
  rebuilds its own: the invalidation step is not O(affected). Only the recomputation has locality; wall-clock time of
  `prepare_delta` grows with the graph, whatever the blast radius.
- The recomputation is a harness-supplied derivation. Nothing is claimed about a query engine, about the cost of a real
  derivation, or about derivations that read keys they did not declare (the declared input sets are the dependency graph).
- Synthetic seeded DAGs of up to 4096 keys with small integer values; real semantic state has other shapes and sizes.
  Deltas larger than `MAX_DELTA_ITEMS` (16384) are not exercised (setup is chunked).
- One machine profile, one process, no concurrency; `StalePreparation` and the journal encoding are not exercised.
- Revisions are checked against an oracle model of "bumps iff the state changes"; the absolute revision numbers are not
  independently specified anywhere else.
