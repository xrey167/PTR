# Falsified in A0: M002-necessity

**Verdict:** FALSIFIES. The rank-1 typed bias adds nothing measurable beyond QK over metadata-bearing slots in A0.

**Evidence:** endpoint `composite:B`, five declared seeds, paired on seed. Per-seed delta (comparator - ablated): -0.0078, -0.0187, +0.0052, +0.0052, +0.0148; mean -0.0003; 95% t-interval [-0.0165, +0.0160], whose upper bound is below the preregistered minimum effect. All gates passed. Details: `RESULTS.md`, `results.json`.

**Scope:** A0 (one block, d_model 32, 1500 steps) on the synthetic operator-routing v1. It does not show the mechanism is useless elsewhere, only that it bought nothing measurable here.

**Consequence, per research/falsification/README.md:** the component is a candidate for removal or redesign in A0; the target is not redefined.
