# ADR-0019 — Adapter Lineage and Weak Supervision Are Working Records Behind Gates

## Status
Accepted as experiments (`ptr-lineage`, `ptr-labeling`). Gated on R004 and F002.

## Decision
An adapter is a sealed, content-addressed checkpoint bound to one exact base model and domain, with a data manifest. It serves only after a forgetting gate over held-out and public suites; interference is measured as principal-angle overlap of update subspaces per layer against its chance level, and a lineage that grows too deep or too entangled is consolidated (TIES merge of the full updates) instead of extended. Replay samples are scheduled by a forgetting model on the training clock and drawn stratified without replacement; held-out samples cannot enter the pool. Labels come from a Dawid-Skene model over labeling-function votes in which verifier-backed functions only veto classes and are never outvoted; an item nothing decides is `Unknown`, and an item every class is ruled out for is `Disputed`. Gold labels record whether they were sampled uniformly or actively; only uniform gold estimates population accuracy and calibration, and active gold reports accuracy on the sampled items only.

## Consequences
Until adapter promotion is a committed ledger event admitted through checkpoint binding, no registry row decides which adapter serves. Revoking a training input names every adapter, descendant and consolidation that depends on it. A predicted label can never be read back as gold.
