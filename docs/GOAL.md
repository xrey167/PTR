# PTR goal and working direction

PTR's goal is a **trainable cognitive architecture and executable Rust runtime**
that jointly use raw language, typed semantic knowledge, uncertainty and
specialist reasoning operators, with independently enforced lifecycle and
action constraints. Its value must be demonstrated against matched models and
strong retrieval baselines at comparable compute and data budgets.

This is the working goal for continuing development. It interprets the existing
[architecture](TECHNICAL_ARCHITECTURE.md), [priority ledger](PRIORITIES.md) and
[Definition of Done](DEFINITION_OF_DONE.md); it does not replace their contracts
or weaken a preregistered experiment's criteria.

## What the goal requires

| Capability | Evidence required |
|---|---|
| Learned reasoning over raw and typed representations | Matched plain-model baselines, mechanism ablations, IID/OOD evaluation and held-out semantic compositions. |
| Appropriate operator and Pod selection | Known-contract/unseen-name generalization, successful tasks and measured compute/latency costs. |
| Useful, updateable knowledge | Versioned inputs, dependency invalidation, live-generation admission and update/revocation tests across derived memory and retrieval paths. |
| Explicit uncertainty | Calibration and abstention/dispute behavior; confidence must not substitute for factual validity or permission. |
| Reliable actions | Verified, authorized commits and effects; lifecycle and capability checks remain independent of learned predictions. |
| Measurable advantage | Reproducible comparisons against strong RAG and matched models, including negative results and a maintained prior-art audit. |

Semantic revocation and excluding a derived memory are not proof that learned
weights have forgotten training data. Neural deletion claims need their own
implemented mechanism and evaluation.

## Current distance to the goal

The repository has substantial runtime, authority, semantic-state and persistence
prototypes. Burn A0 is a small trainable tensor experiment, not a pretrained
language model: it has typed/raw attention, latent refinement and routing, but
no full language-model backbone or demonstrated end-to-end superiority.

S003 tests certified concurrent branch merges with simulated agents. Passing it
would support that subsystem's bounded claim, not learned reasoning or language
quality. PR #32 contains useful A0 ablation work, but its review and provenance
issues must be settled before its reported outcomes guide architecture choices.

## Next scientific milestone

Produce a trustworthy, reproducible account of **which A0 mechanisms help,
under what task and budget conditions, and which can be removed**. Then add the
matched plain-model and strong-RAG comparisons required by the Definition of
Done. Do not scale the model or expand infrastructure to substitute for that
evidence.

The immediate order is:

1. Finish the bounded S003 integrity work and preserve its preregistration.
2. Recover A0 study correctness: bind scoring to the actual data and frozen
   inputs, reject contradictory output, account for every required process,
   and bind test evidence to the evaluated source. Resolve integration with
   current main without losing its newer evidence checks.
3. Recompute the study's gates and report, rerunning invalid evidence rather
   than preserving unsupported verdicts. Keep internal ablations distinct
   from comparisons against an external/model baseline.
4. Implement and evaluate the next cognitive mechanism or simplification
   justified by those results, with matched compute and held-out tests.
5. Integrate a demonstrated learned path into the existing runtime and measure
   the complete task loop, including lifecycle changes and verification costs.

The detailed PR and experiment sequencing is in
[PR_PLAN_20260929.md](PR_PLAN_20260929.md). Independent model work can proceed
on its own branch while S003 is validated; code must not be mixed into a frozen
experiment's source tree midway through its confirmatory seeds.

## How continuing work is judged

Each increment must close a concrete correctness or scientific-evidence gap,
deliver an executable result with relevant tests, and state what remains
unproven. A green CI run, a new crate, a documented architecture or a synthetic
benchmark win alone does not complete PTR's goal.
