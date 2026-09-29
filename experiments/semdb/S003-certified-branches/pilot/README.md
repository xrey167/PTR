# S003 pilot

The pilot fixes `low_cells` in the preregistration table before the freeze. It uses seeds 1, 2 and 3, the
`pilot_seeds` of the table, and never a confirmatory seed. Its outputs are not evidence for any claim: they
only classify the cells (contention level, N) by their pooled conflict rate.

A cell is **low** if the pooled conflict rate of the certified arm over the three pilot seeds, certification
`Conflict` and `LifecycleChanged` refusals over merge attempts, is below 100‰. The pilot must leave at least
6 of the 24 cells on each side and at least one low cell for every N. If it does not, the preregistered
fallback ladder (`groups_ladder_fallback`) replaces `groups_ladder` and the pilot runs once more.

Classification: `python3 experiments/semdb/S003-certified-branches/aggregate.py --pilot [outputs…]`.

## Round 1: ladder 8, 16, 32, 64, 128, 256

- Commit: `c58a9f9` (clean tree, release build).
- Command, for s in 1, 2, 3, run three at a time: `./target/release/ptr-bench certified-branches 48 <s> > pilot/ladder-1/pilot-seed-<s>.json`.
- Outputs: [`ladder-1/pilot-seed-1.json`](ladder-1/pilot-seed-1.json), [`-2`](ladder-1/pilot-seed-2.json), [`-3`](ladder-1/pilot-seed-3.json).
  Each exited 0 with `hard_failures` 0 over 48 cases (417 s, 422 s and 410 s of wall time with three running side by side).

Pooled conflict rate of the certified arm (bold: at or above 100‰):

| level | groups | N=2 | N=4 | N=8 | N=16 |
|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 66‰ | **110‰** | **188‰** | **307‰** |
| 1 | 16 | 34‰ | 58‰ | 98‰ | **171‰** |
| 2 | 32 | 21‰ | 34‰ | 59‰ | **108‰** |
| 3 | 64 | 14‰ | 18‰ | 34‰ | 59‰ |
| 4 | 128 | 9‰ | 14‰ | 24‰ | 43‰ |
| 5 | 256 | 9‰ | 7‰ | 15‰ | 27‰ |

Result: 5 cells are at or above the threshold and 19 below. The rule needs at least 6 on each side, so the
pilot **failed its minimum** and the preregistered fallback ladder `[4, 8, 16, 64, 256, 1024]` replaced the
ladder in `config.toml` and in `params.rs` (the compiled digest of the table changed with it). Nothing else
changed in behaviour: the arms, probes and oracle are the code of round 1, plus five lint fixes (elided lifetimes, a
borrowed map key, a fold) that change no result. Round 1's outputs stay as
they are, as the record of why the ladder changed.

## Round 2: the fallback ladder 4, 8, 16, 64, 256, 1024

- Commit: `3fa8f05` (clean tree, release build, the commit that replaced the ladder).
- Command, for s in 1, 2, 3, run three at a time: `./target/release/ptr-bench certified-branches 48 <s> > pilot/pilot-seed-<s>.json`.
- Outputs: [`round-2/pilot-seed-1.json`](round-2/pilot-seed-1.json), [`-2`](round-2/pilot-seed-2.json), [`-3`](round-2/pilot-seed-3.json).
  Each exited 0 with `hard_failures` 0 over 48 cases (1542 s, 1594 s and 1531 s of wall time with three running side by side).

Pooled conflict rate of the certified arm (bold: at or above 100‰):

| level | groups | N=2 | N=4 | N=8 | N=16 |
|---:|---:|---:|---:|---:|---:|
| 0 | 4 | **123‰** | **199‰** | **324‰** | **482‰** |
| 1 | 8 | 64‰ | **109‰** | **186‰** | **302‰** |
| 2 | 16 | 35‰ | 51‰ | 97‰ | **179‰** |
| 3 | 64 | 14‰ | 18‰ | 34‰ | 59‰ |
| 4 | 256 | 6‰ | 11‰ | 21‰ | 32‰ |
| 5 | 1024 | 7‰ | 6‰ | 11‰ | 18‰ |

Result: 8 cells at or above the threshold and 16 below, at least 6 on each side, and low cells for every N
(5 for N=2, 4 for N=4, 4 for N=8, 3 for N=16): round 2 met the minimum.

## Why round 2 was not the last

Review of the harness at this state (Codex, CodeRabbit and an independent read-only review of six dimensions
with a skeptic per finding) found defects that change what a pilot measures, so its `low_cells` cannot be
pinned:

- **The time model failed at 1024 groups.** 11.6% of `merge_branch` calls of round 2 (41,784 of 359,018)
  took more than 10 ms, the p99 bucket of the pooled histogram was `merge_wall_gt_10ms`, and the
  preregistered check (`merge_wall_budget_us_p99` = 10000) would have failed every confirmatory run. The
  cause is the reference host, not the harness: `SemanticHost::prepare_delta` clones the whole state, and a
  case at 1024 groups holds about 12,000 keys. Round 1, whose largest level was 256 groups, had 0.02% of
  its merges above 10 ms. The fallback's largest level, 1024 groups, is replaced by 32, so the ladder is
  4, 8, 16, 32, 64, 256 (`groups_ladder_fallback` keeps the value the table preregistered).
- **Attempt durations were not paired across arms** (Codex, P1): a task's duration came from the calls its
  program made on the arm's own state. It is now the calls the program makes on the case's genesis state,
  the same in every arm.
- **Hazard trials counted more than certification tried** (Codex, P1, and the review): the fixed probes'
  repetitions counted as trials of every class (the two rule classes, a negative counter held by
  verification and a set operation that undoes a concurrent one, had 48 trials in each seed only from those
  repetitions; the workload made none), and a hazard counted as a trial where certification stopped before
  the check that looks for it. Trials are now the workload's own merges, counted only where the check ran;
  the probes' repetitions are counted apart, and the two rule classes rest on the probes P25 and P26 and
  the mutation plan.
- **Verification holds were counted twice** in a certified run's statistics (Codex, P2), **key-level OCC's
  stale-scan commits and lost updates were not counted among its anomalies** (Codex, P2), a run that stopped
  on an error still contributed a makespan to the throughput (Codex, P1), and the pilot classification
  accepted outputs of another preregistration (Codex, P1).

These change durations, counts and the ladder, and with them conflict rates, so the cells are classified
again on the final harness. The rule is unchanged (at least 6 cells on each side, a low cell for every N).
The fallback ladder is spent: a round 3 that fails the minimum ends the pilot, and what to do then is a
decision to record, not to make in advance.

## Round 3: the final harness

- Commit: `50e50eb` (clean tree, release build; the round-2 outputs were moved to [`round-2/`](round-2/) after it).
- Command, for s in 1, 2, 3, run three at a time: `./target/release/ptr-bench certified-branches 48 <s> > pilot/pilot-seed-<s>.json`.
- Outputs: [`pilot-seed-1.json`](pilot-seed-1.json), [`-2`](pilot-seed-2.json), [`-3`](pilot-seed-3.json).
  Each exited 0 with `hard_failures` 0 over 48 cases, every run complete (303 s, 314 s and 300 s of wall time with three running side by side).
- Merge time: 22 of 360,175 `merge_branch` calls (0.006%) took more than 10 ms, so the p99 bucket of the pooled histogram is within the budget.
- What the workload made of the two rule classes: no negative counter held by verification, and 4, 1 and 8 set operations undoing a concurrent one in the three seeds; their evidence stays the probes P25 and P26 and the mutation plan.

Pooled conflict rate of the certified arm (bold: at or above 100‰):

| level | groups | N=2 | N=4 | N=8 | N=16 |
|---:|---:|---:|---:|---:|---:|
| 0 | 4 | **121‰** | **199‰** | **324‰** | **479‰** |
| 1 | 8 | 64‰ | **108‰** | **184‰** | **300‰** |
| 2 | 16 | 33‰ | 52‰ | 97‰ | **179‰** |
| 3 | 32 | 23‰ | 34‰ | 57‰ | **105‰** |
| 4 | 64 | 14‰ | 21‰ | 38‰ | 66‰ |
| 5 | 256 | 9‰ | 7‰ | 15‰ | 27‰ |

Result: 9 cells at or above the threshold and 15 below, at least 6 on each side, and low cells for every
N (5 for N=2, 4 for N=4, 4 for N=8, 2 for N=16). The pilot **met its minimum** on the final harness. `L3N16` is at
105‰ and `L2N8` at 97‰, close to the threshold either way; the list is what the rule gives. The low cells, written into
`low_cells` of `config.toml` at the freeze:

```toml
low_cells = ["L1N2", "L2N2", "L2N4", "L2N8", "L3N2", "L3N4", "L3N8", "L4N2", "L4N4", "L4N8", "L4N16", "L5N2", "L5N4", "L5N8", "L5N16"]
```

Round 2's outputs stay in `round-2/`, round 1's in `ladder-1/`, as the record of why the harness and the ladder changed; the
aggregator refuses both as pilot inputs, since they ran under other preregistered tables and other code.

## Source-bound pilot evidence (2026-09-29)

Historical raw JSON outputs are preserved for audit, but are no longer accepted by `--pilot`: they do not record the producing source revision, and the harness has changed since they ran. Do not add provenance retroactively. S003 remains planned and must run new pilots before freezing `low_cells`.

From a committed clean source tree, record each declared pilot seed with:

```sh
python3 experiments/semdb/S003-certified-branches/aggregate.py --record-pilot 1 --output experiments/semdb/S003-certified-branches/pilot/source-bound/pilot-seed-1.json
```

Repeat for seeds 2 and 3, then pass the three new paths to `--pilot`. The recorder builds in a fresh temporary target directory, refuses failed runs and source changes, and creates rather than overwrites evidence. Classification requires the producing revision in the checkout's history and unchanged Rust, build and recording/aggregation sources. The echoed table still binds configuration, allowing only `low_cells` to change for the freeze.

This revision also checks unexpected input sets on every runtime key and counts lost LWW operations when a successful write makes no state change. The descriptive rate is now `lww_anomalies_per_settled_attempt`, with committed and no-change attempts in its denominator. The old pilot cannot establish coverage or rates for these repaired checks.
