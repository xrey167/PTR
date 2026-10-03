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

**Workload.** Per group `g` of the contention level, eight items `item:g:0..7`, a spare `item:g:s`, extras `item:g:x<task>` under the scanned prefix, a derived `total:g` whose input set alternates between the eight items and the seven items plus the spare, and `audit:g`; sixteen counters, eight sets and four policy capsules. Nine programs (read-modify-write, write skew, capped insert, remove, audit over a scan, group total with its derived key, counter add, set operation, guarded decrement) each run once as a real `ptr_branch::Branch` over the snapshot the agent opened, and once as the reference model's view; the two must log the same reads, scans, staged operations and lifecycle answers. Background actors write requests through ingress, overwrite items as a host, supersede or revoke and recommit policy generations, and **rewire** a total: prime two keys to equal values, then swap the total's input set and upsert its unchanged sum, which changes no value any branch read and only the input-set digest detects. The contention ladder is six group counts, 4, 8, 16, 32, 64 and 256; a case runs the level `case mod 6`. It is the preregistered fallback ladder (4, 8, 16, 64, 256, 1024), which replaced 8 to 256 after the first pilot left five high cells, with its largest level, 1024 groups, replaced by 32: at about 12,000 keys the reference host's `prepare_delta`, which clones the whole state, took more than the 10 ms one merge is modelled to take for 11.6% of merges, so the time model would have failed by construction. `pilot/README.md` records both changes.

**Time.** Integer ticks, one tick being 10 ms of agent time. An attempt lasts one tick per call its program makes on the case's genesis state plus a think time, so the same attempt lasts the same in every arm whatever state that arm's branch opens on; merges go through one lane, one per tick. Wall time is measured for the merge histogram and never used for scheduling.

**Oracle.** `oracle.rs` and `model.rs` share no code with certification and never call `certify`, `ValueDigest`, `RangeDigest`, `InputsDigest`, `counter_value` or `set_value`. The model re-implements `ptr-semdb`'s delta rule (removals, upserts, dependency entries, eviction of affected keys over the old and new graph, missing inputs) and the lifecycle authority, and follows what the runtime commits, read back from its journal; a differential test checks the rule against the whole-copy implementation it replaced on random deltas. Before every merge the oracle names the hazards of the branch's footprint against the model (lost update, stale read, phantom, stale scan value, stale input set, stale reliance) and predicts the exact outcome: `LifecycleChanged`, `Conflict` with the keys (a scanned prefix as `<prefix>*`, including the set-member rule that refuses a set operation undoing a concurrent one), the refusal of an operation or delta, a hold for a negative counter, no change, or a merge with the rebased keys and the delta. Nine hand-written canaries judge the oracle itself in every case.

**What is checked after every event.** The runtime's revision, every value, every input set and every generation's validity equal the model's. For a committed merge: the record names the branch, author, seal, plan (recomputed from the record), verifiers and authority; the rebased keys are the predicted ones; the merged state equals the state of running the program again on the model as it was; no increment was lost; no counter is negative. A refusal or hold appends nothing.

**Probes** (P1 to P26, every case, on fresh runtimes): a branch merged again after a replay, a compacted restore and a durable reopen; a branch opened over another state; a generation revoked after a preview; an approval of a plan that changed, by an unlisted reviewer, and an escalation; a negative counter, a weaker level and a hard finding beside a pass held by verification, and a counter jump refused in a host write; ingress keys written by a branch or a host; an unread overwrite; a derived key with an input unread; semantic records through `commit`, a host write with no permission, a second grant, a merge with no grant; seven forged histories (a record with no origin after one with an origin, also above a compaction floor; a merge with the wrong plan digest, with a dependency entry, of a branch merged before; a request writing two keys; a host write to an ingress key; a Pod output with another source), each well formed but for its one defect; and, in P22 to P26, a deterministic case of a write skew, a phantom insert, a total whose input set was swapped without a value moving, a counter that is negative once rebased and a set operation that undoes a concurrent one, so certification is tried on each in every case. What these fixed cases run into is coverage, counted apart (`probe_hazard_*`): they repeat one construction in every case, so they are no trials. The workload alone samples the other classes (scan values, lifecycle, rebase).

**Counters.** 28 hard counters (each must be 0), the coverage counters (each must be above 0 in every seed; the paths of the merge itself, clean and rebased merges, conflicts, lifecycle refusals, escalations, reviewed merges and no-change merges, are counted over the workload's own merges only, the probes' counts of them being reported apart as `probe_*` and gating nothing, and the one path the workload does not reach, a merge held by verification, is gated by the probes' count, `probe_verification_holds`, its workload count `verification_holds` being descriptive; each of the seven hazard classes the workload samples, write, read, scan keys, scan values, input set, lifecycle and rebase, needs at least 30 trials per seed, counted only over the workload's own merges), the baselines' anomalies (never failures of the hypothesis, but coverage: each must be above 0 in every seed, or the run shows nothing about what the oracle sees in them: last-writer-wins' lost updates and lost increments; key-level OCC's phantoms, stale scan values, stale input sets and stale reliance), and descriptive counters: OCC's lost updates (`occ_lost_updates`, expected to stay 0, because OCC checks every point read and a program only writes what it read; a count would be a blind overwrite that OCC committed, reported as it is), the two classes that are refusals by a rule the workload does not have to reach, a negative counter held by verification and a set operation that undoes a concurrent one, whose evidence is the fixed probes P25 and P26 and the mutation plan (their trials in the workload and the probes' repetitions are reported and gate nothing, and have no bound), refusals where merging would have equalled serial execution (`unnecessary_refusals`, the trigger for predicate digests), refusals of increments written as values (`put_increment_conflicts`, the trigger for typed merge operators), conflicts whose key set the oracle predicted wider than the runtime reported (`conflict_key_set_differs`), review voids, wasted ticks (the agent's own ticks on attempts that did not commit, never a reviewer's wait) and abandoned tasks, each counted over the arms' runs and apart from the probes' worlds. The result also lists, per case and run, the makespan, whether the run reached its end (`complete`) and the counts of attempts, merges, conflicts, holds and escalations; the throughput of a cell is read only from runs that did.

### Preregistered rules

1. **Safety (exact).** Any hard counter above 0 in any seed rejects the hypothesis and fails the run. Only after a hard pass is the generalisation reported as `bound_if_independent`, the rule-of-three figure 3/n per sampled hazard class over the workload's trials; after any hard failure or nonzero process exit it is unavailable (`null`). The trials of one case's arms replay the same tasks in different interleavings and are not independent, so the figure is what the bound would be if they were, not a bound: each entry says so in its `independence` field, and the real bound is larger.
2. **Throughput, supported.** For every N, the 2.5th percentile of a paired case bootstrap (2000 resamples, seed 20260926) of `gain_N` = Σ serial makespan / Σ certified makespan over the low cells is above 1.0.
3. **Throughput, rejected.** For some N the 97.5th percentile is at most 1.0.
4. **Inconclusive, recorded as failed.** A coverage counter is 0 (the probes P25 and P26 of the two rule classes among them) or a sampled hazard class has fewer than 30 workload trials in some seed; a run stopped on an error or took no time; the merge time histogram's p99 bucket exceeds 10 ms; the manipulation check fails; some N has no low cell; some mutation is not killed; or neither rule 2 nor rule 3 holds.
5. **Efficiency (secondary).** `gain_N / N` ≥ 0.5 at the point estimate for every N. Reported as met or missed; it does not decide the status.
6. Reported with every result: hazard trials and 3/n per sampled class (unavailable without a hard pass), the two rule classes' workload trials and probe repetitions apart, the LWW and OCC anomaly rates, the unnecessary-refusal share and the review-void share.

A **cell** is (contention level, N). It is **low** if its pooled conflict rate over the pilot seeds (1, 2, 3; never confirmatory) is below 100‰, where the conflict rate is certification `Conflict` and `LifecycleChanged` refusals over merge attempts of the certified arm. The pilot must leave at least 6 of the 24 cells on each side and at least one low cell for every N; otherwise the preregistered fallback ladder replaces the ladder and the pilot runs once more. The list is written into `low_cells` before the freeze. The manipulation check requires each N's pooled conflict rate over its low cells to be below 100‰ in the confirmatory seeds.

### Evidence validation

Before calculating a verdict or classifying pilot cells, the aggregator requires
every case index, its configured level and group count, and exactly one serial
run plus the four other arms for every configured agent count. The serial
makespan must agree with its run. Pilot outputs additionally require all hard
counters present and zero and every run complete with positive ticks; rejected
pilot input prints no classification. Confirmatory runs that failed remain
negative evidence and can still produce a failed aggregate.

The harness checks each requested lifecycle event and ingress request against
its appended journal records before the model follows them. Its deterministic
run digest includes semantic revision, live and revoked generations, and run
completion as well as values, dependencies, statistics and counters.

### Limitations

Simulated agents are pure programs; the ledger is in memory and the runtime a single writer; think time dominates merge time by design (so the throughput half is expected to hold whenever waste is modest, and its informative content is the efficiency and the waste); the reference `SemanticHost` clones its state at every merge (`prepare_delta`; persistent data structures are an independent performance task), so a merge costs time in proportion to the state, and the ladder ends at 256 groups (about 3,000 keys) to keep the merge inside the modelled tick: merge time beyond that size is not measured; PostgreSQL paths (stored branches, seal tags, projections) are covered by `ptr-pg` tests, not here; the hardware profile is unspecified until measured; and the review arm's single reviewer and the digest's covering the revision make most approvals void, which is reported, not tuned.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
