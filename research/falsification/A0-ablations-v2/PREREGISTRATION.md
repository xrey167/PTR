# A0 mechanism ablation study v2: preregistration

**Frozen at the git tag `a0-ablations-v2-prereg`.** Nothing in this file, in `criteria.toml`, in `DESIGN.md`, in `budget.json`, `references.json`, `calibration.json`, the benchmark's `splits.lock.json`, or in the code the study runs may change after that tag, except as `DESIGN.md` (D4, D5) states: the learning-rate selection, run records and the registration of the later-stage experiments. `scripts/aggregate_a0_ablation.py --study v2` refuses to run if a frozen input differs from the tagged file. No evaluation process of this study (declared seeds 17, 29, 43, 71, 101) has been run; only the sweep (runner seed 17, validation split) follows this tag.

## What this is

The successor of A0-ablations-v1, preregistered by [`DESIGN.md`](DESIGN.md), which lists every difference from v1 with its reason. It is an A0-internal ablation on the synthetic benchmark operator-routing v1; its processes are seeds of listed experiments run through the hardened runner. It is **not** the M001-M004 baseline comparison (those stay `planned`; owner decision O2). Where this file and `DESIGN.md` differ, this file and `criteria.toml` govern.

## What it cannot show

Anything about language-model quality or reasoning over real text; M001's claim against a matched plain backbone; M002's hard-violation and constraint-retention claims; M003's claim about saving reasoning tokens; M004's comparison with LLM-only reasoning; anything about a verifier head, action head, `UncertaintyKind`, learned slot encodings, runtime integration, scaling or latency. A clean positive result would show only that a mechanism helps this A0 on operator-routing v1. The same list as v1's `PREREGISTRATION.md` applies.

## Frozen configuration

- **Model:** `PtrA0Config::new(216, 32).with_provenance_buckets(8)`, one block, codebook V1, `SlotEncoding` V1, the full arm at `latent_steps = 2`. The width is 32: calibration reached its target at the base rung (below), so neither revision-ladder rung applies.
- **Training:** as v1: Adam defaults, batch 128 without replacement, 1500 steps (S*), 100-step warm-up then cosine decay to 0.1x, no early stopping, validation every S*/10 steps; seeds [17, 29, 43, 71, 101] (`init_seed = seed`, `order_seed = seed ^ 0x0BA7C4`).
- **Learning rate:** swept after this freeze, on the corrected code, on grid [0.002, 0.005, 0.0125] at runner seed 17 with salted seeds, 1500 steps, validation only; each arm takes the smallest rate within 0.005 of its best; a NaN loss makes a rate ineligible; an edge pick is flagged, not re-swept.
- **Arms that run** (no budget-ladder rung applies):
  - M001: full, no-semantic-slots, no-semantic-slots-masked
  - M002: no-typed-attention, raw-blind, blind-query-k0, blind-query-k0-no-typed-attention
  - M003: latent-0, latent-1, latent-linear, latent-4
  - M004: frozen-router
- **Rules:** `criteria.toml` (v1's thresholds, contrasts, composites, learnability bars, contingency, leak alarm), gates G0-G6 as `DESIGN.md` adapts them.

## Calibration (outcome-blind: full arm, seed 0, train and val only)

Target: final val accuracy >= 0.88; candidates in order, rung base then R1 then R2, each at lr 0.005.

| Rung | d_model | Steps | Final val accuracy | ms/step | Val curve (10 checkpoints) |
|---|---:|---:|---:|---:|---|
| base | 32 | 1500 | 0.8820 | 10.323 | 0.661, 0.804, 0.852, 0.861, 0.866, 0.863, 0.869, 0.877, 0.872, 0.882 |

Chosen: rung base, d_model 32, S* = 1500.

## Budget rule

From the calibration ms/step (10.323): ladder steps applied []; projected 10.5 minutes on 3 workers (limit 20, hard cap 45); sweep on at 1500 steps. Execution is serial with a cold build per run (`DESIGN.md` D11), which the budget rule does not model.

## Frozen data

Operator-routing v1, generator version 1, benchmark seed 20260925, codebook V1. Data FNV-1a-64 over the eight TSV files: `0ad71688f09b0d0d`; split digests in `benchmarks/operator-routing/splits.lock.json`. The G0 data bands and the reference lines (bound additive, hand-weighted, raw-blind exact ceiling, train-majority) are in `references.json`; all bands passed at this freeze.

## Registered experiments

At this freeze: `M00x-a0v2-sweep-lr<lr>` for each family x grid rate (12 experiments, one seed each). Registered only at the second freeze, before any evaluation seed runs: `M00x-a0v2-eval` (5 seeds each), `M001-a0v2-rerun` (seed 17, the G2 determinism rerun) and, if the learnability rule requires it, `M00x-a0v2-cont-<arm>` (contingency, 4000 steps). Each freezes its command, values and seeds in its `config.toml` and runs each seed once.
