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
- Outputs: [`pilot-seed-1.json`](pilot-seed-1.json), [`-2`](pilot-seed-2.json), [`-3`](pilot-seed-3.json).
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

Result: 8 cells at or above the threshold and 16 below, at least 6 on each side, and low
cells for every N (5 for N=2, 4 for N=4, 4 for N=8, 3 for N=16). The pilot **met its minimum**. The rule
allows the fallback once, so there is no third round. The low cells, written into `low_cells` of `config.toml` at the freeze:

```toml
low_cells = ["L1N2", "L2N2", "L2N4", "L2N8", "L3N2", "L3N4", "L3N8", "L3N16", "L4N2", "L4N4", "L4N8", "L4N16", "L5N2", "L5N4", "L5N8", "L5N16"]
```
