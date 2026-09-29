# S003-certified-branches

## Hypothesis
Do dependency-certified agent branches avoid lost updates and phantoms while merging more concurrent work than serial execution?

## Primary metrics
lost updates; undetected phantoms; certified merge rate; conflict rate; merge latency

## Baseline
serial execution; last-writer-wins branches; key-level optimistic locking without range digests

## Falsification
Any lost update or undetected phantom is a hard failure; no throughput gain over serial execution at conflict rates below 10% rejects the default.

## Design
Mechanism and threat model: [35 — Agentic substrate](../../../docs/architecture/35-agentic-substrate.md). Runtime: [ADR-0020](../../../research/decisions/ADR-0020-only-the-runtime-merges-an-agent-branch.md).

`experiments/preregistration.toml` lists every parameter S003 must preregister in the `[preregistration]` table of `config.toml` before it leaves `planned`; `scripts/check_research_gates.py` fails CI until each is pinned and `experiment.toml` names the table's digest. Every number below is a key of that table; the harness compiles it in (`bins/ptr-bench/src/experiments/s003/params.rs`), a test requires the compiled values to equal the file's, and each result echoes the table's canonical text for the aggregator to compare.

### What is measured

The harness is `ptr-bench certified-branches <cases_per_seed> <seed>`
([`bins/ptr-bench/src/experiments/s003/`](../../../bins/ptr-bench/src/experiments/s003/mod.rs)). It runs in memory in every build, with no PostgreSQL: the claims concern certification and the verified commit path, which read no PostgreSQL (ADR-0016). Each case draws one workload and runs it under five arms, at 2, 4, 8 and 16 agents where an arm has agents:

| Arm | How a branch commits |
|---|---|
| `serial` | One agent; `merge_branch` under the auto policy; the background actors wait until the agent is between tasks, so nothing can conflict. The throughput baseline. |
| `certified` | Every agent merges through `PtrRuntime::merge_branch` under the auto policy (`s003-auto/v1`, which auto-proposes whatever the verifiers admit). The primary arm. |
| `certified-review` | The same under `s003-review/v1` (threshold 0.5, calibration slice 0.1). What the policy does not auto-propose waits for one FIFO reviewer (20 to 60 ticks) and merges under that reviewer's approval of the plan's digest; an approval whose plan has since changed is void and the task runs again. Secondary; descriptive throughput. |
| `lww` | The branch's overlay, its operations folded over the values of its base, is written as a host write, whatever the target holds now. A baseline whose anomalies are evidence. |
| `occ` | Key-level optimistic concurrency: every point-read key must hold the value it had at the base, commutative operations are rebased, and the result is a host write. No range, input-set or lifecycle check. A baseline whose anomalies are evidence. |

Every arm sees the same genesis, tasks, attempt durations and background schedule (all drawn from streams named by the seed, the case and a label, so no arm can consume another's draws). A task is opened again after a refusal, up to eight attempts; one the verifiers reject is dropped in every arm.

**Workload.** Per group `g` of the contention level, eight items `item:g:0..7`, a spare `item:g:s`, extras `item:g:x<task>` under the scanned prefix, a derived `total:g` whose input set alternates between the eight items and the seven items plus the spare, and `audit:g`; sixteen counters, eight sets and four policy capsules. Nine programs (read-modify-write, write skew, capped insert, remove, audit over a scan, group total with its derived key, counter add, set operation, guarded decrement) each run once as a real `ptr_branch::Branch` over the snapshot the agent opened, and once as the reference model's view; the two must log the same reads, scans, staged operations and lifecycle answers. Background actors write requests through ingress, overwrite items as a host, supersede or revoke and recommit policy generations, and **rewire** a total: prime two keys to equal values, then swap the total's input set and upsert its unchanged sum, which changes no value any branch read and only the input-set digest detects. The contention ladder is six group counts (8 to 256); a case runs the level `case mod 6`.

**Time.** Integer ticks, one tick being 10 ms of agent time. An attempt lasts one tick per call its program makes plus a think time; merges go through one lane, one per tick. Wall time is measured for the merge histogram and never used for scheduling.

**Oracle.** `oracle.rs` and `model.rs` share no code with certification and never call `certify`, `ValueDigest`, `RangeDigest`, `InputsDigest`, `counter_value` or `set_value`. The model re-implements `ptr-semdb`'s delta rule (removals, upserts, dependency entries, eviction of affected keys over the old and new graph, missing inputs) and the lifecycle authority, and follows what the runtime commits, read back from its journal; a differential test checks the rule against the whole-copy implementation it replaced on random deltas. Before every merge the oracle names the hazards of the branch's footprint against the model (lost update, stale read, phantom, stale scan value, stale input set, stale reliance) and predicts the exact outcome: `LifecycleChanged`, `Conflict` with the keys (a scanned prefix as `<prefix>*`, including the set-member rule that refuses a set operation undoing a concurrent one), the refusal of an operation or delta, a hold for a negative counter, no change, or a merge with the rebased keys and the delta. Nine hand-written canaries judge the oracle itself in every case.

**What is checked after every event.** The runtime's revision, every value, every input set and every generation's validity equal the model's. For a committed merge: the record names the branch, author, seal, plan (recomputed from the record), verifiers and authority; the rebased keys are the predicted ones; the merged state equals the state of running the program again on the model as it was; no increment was lost; no counter is negative. A refusal or hold appends nothing.

**Probes** (P1 to P26, every case, on fresh runtimes): a branch merged again after a replay, a compacted restore and a durable reopen; a branch opened over another state; a generation revoked after a preview; an approval of a plan that changed, by an unlisted reviewer, and an escalation; a negative counter, a weaker level and a hard finding beside a pass held by verification, and a counter jump refused in a host write; ingress keys written by a branch or a host; an unread overwrite; a derived key with an input unread; semantic records through `commit`, a host write with no permission, a second grant, a merge with no grant; seven forged histories (a record with no origin after one with an origin, also above a compaction floor; a merge with the wrong plan digest, with a dependency entry, of a branch merged before; a request writing two keys; a host write to an ingress key; a Pod output with another source), each well formed but for its one defect; and one deterministic case of each hazard class so certification is tried on all of them in every case.

**Counters.** 28 hard counters (each must be 0), the coverage counters (each must be above 0 in every seed; each hazard class needs at least 30 trials per seed), the baselines' anomalies (evidence, never failures), and descriptive counters: refusals where merging would have equalled serial execution (`unnecessary_refusals`, the trigger for predicate digests), refusals of increments written as values (`put_increment_conflicts`, the trigger for typed merge operators), review voids, wasted ticks and abandoned tasks. The result also lists, per case and run, the makespan and the counts of attempts, merges, conflicts, holds and escalations.

### Preregistered rules

1. **Safety (exact).** Any hard counter above 0 in any seed rejects the hypothesis and fails the run. The generalisation is reported as the rule-of-three bound 3/n per hazard class.
2. **Throughput, supported.** For every N, the 2.5th percentile of a paired case bootstrap (2000 resamples, seed 20260926) of `gain_N` = Σ serial makespan / Σ certified makespan over the low cells is above 1.0.
3. **Throughput, rejected.** For some N the 97.5th percentile is at most 1.0.
4. **Inconclusive, recorded as failed.** A coverage counter is 0 or a hazard class has fewer than 30 trials in some seed; the merge time histogram's p99 bucket exceeds 10 ms; the manipulation check fails; some N has no low cell; some mutation is not killed; or neither rule 2 nor rule 3 holds.
5. **Efficiency (secondary).** `gain_N / N` ≥ 0.5 at the point estimate for every N. Reported as met or missed; it does not decide the status.
6. Reported with every result: hazard trials and 3/n per class, the LWW and OCC anomaly rates, the unnecessary-refusal share and the review-void share.

A **cell** is (contention level, N). It is **low** if its pooled conflict rate over the pilot seeds (1, 2, 3; never confirmatory) is below 100‰, where the conflict rate is certification `Conflict` and `LifecycleChanged` refusals over merge attempts of the certified arm. The pilot must leave at least 6 of the 24 cells on each side and at least one low cell for every N; otherwise the preregistered fallback ladder replaces the ladder and the pilot runs once more. The list is written into `low_cells` before the freeze. The manipulation check requires each N's pooled conflict rate over its low cells to be below 100‰ in the confirmatory seeds.

### Limitations

Simulated agents are pure programs; the ledger is in memory and the runtime a single writer; think time dominates merge time by design (so the throughput half is expected to hold whenever waste is modest, and its informative content is the efficiency and the waste); PostgreSQL paths (stored branches, seal tags, projections) are covered by `ptr-pg` tests, not here; the hardware profile is unspecified until measured; and the review arm's single reviewer and the digest's covering the revision make most approvals void, which is reported, not tuned.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
