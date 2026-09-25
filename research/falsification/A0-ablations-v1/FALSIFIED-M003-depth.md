# Falsified in A0: M003-depth

**Verdict:** FALSIFIES. Applying the tied refinement a second time gives no measurable benefit at equal parameters.

**Evidence:** endpoint `composite:C`, five declared seeds, paired on seed. Per-seed delta (comparator - ablated): -0.0204, +0.0136, +0.0150, +0.0019, -0.0053; mean +0.0009; 95% t-interval [-0.0172, +0.0191], whose upper bound is below the preregistered minimum effect. All gates passed. Details: `RESULTS.md`, `results.json`.

**Scope:** A0 (one block, d_model 48, 1500 steps) on the synthetic operator-routing v1. It does not show the mechanism is useless elsewhere, only that it bought nothing measurable here.

**Consequence, per research/falsification/README.md:** the component is a candidate for removal or redesign in A0; the target is not redefined.
