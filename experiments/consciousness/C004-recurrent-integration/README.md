# C004-recurrent-integration

## Hypothesis
[CPU] Indicator RPT-1 (algorithmic recurrence) as a functional property; RPT-2 is not targeted (doc 38): does algorithmic recurrence with a persistent latent state across steps (the latent recurrence of docs/architecture/01-ptr-core.md, model/burn-a0 latent_steps) integrate multi-step evidence better than a depth-matched feed-forward stack with the same parameters and compute, on tasks whose answer depends on a state carried across steps? Extends M003 (which asks only about token savings) with an integration test, and the workspace of C001 as a recurrent variant.

## Primary metrics
task accuracy on held-out step counts longer than trained; accuracy against number of recurrent steps; parameter and FLOP counts per arm; state-carryover probe (does the answer change when an early step is perturbed); paired differences over seeds

## Baseline
depth-matched feed-forward stack with unshared layers, parameter-matched and FLOP-matched; recurrent model with the carried state reset every step

## Falsification
Falsified if the recurrent arm does not beat BOTH the depth-matched feed-forward stack and the reset-state ablation on length extrapolation by a pinned margin with a one-sided 95% paired bound over at least 5 seeds at matched parameters and FLOPs; invalid if parameter or FLOP counts differ beyond the pinned tolerance.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
