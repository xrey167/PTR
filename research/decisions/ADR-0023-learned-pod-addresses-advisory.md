# ADR-0023 — Learned Pod Addresses Are Advisory, Activation-Gated and Revoked Per Pod

## Status
Accepted for the prototype with work package 1 of the research integration plan (`ptr-types`, `ptr-pods`, `ptr-ledger`, `ptr-runtime`, `ptr-router`, `training/`). The routing benefit is a claim of experiment R015, which has a harness and no frozen result; this ADR fixes what a learned address may do, not that it helps.

## Decision
A pod has a value-free identity derived from its hard semantics (`PodIdentityDescriptor`), and a head that moves only through a committed compare-and-swap record (`LedgerEvent::PodHeadAdvanced`: expected head equal, generation strictly rising, new generation not revoked and not below the live lifecycle generation). A head answers only while its generation is the one the lifecycle authority holds live for the pod.

A learned pod address (a vector and a bias per pod over a frozen encoder) is advisory. It ranks only the candidates the registry offers; the registry and the trusted routing boundary keep capability, input type and project scope as hard constraints, and a learned score never overrides a deterministic verifier contradiction. The router may abstain, and abstention is its own error variant (`RouteError::Abstained`), counted apart from failures. A representation is loaded only when activated: every trusted alias scored at least 0.8 and at least 90 % of the negative examples were rejected, a minimum the loader re-checks against the measured values. A representation is neural state bound to its pod's exact generation; revoking or superseding that generation invalidates that pod's representation and no other, because no trainable state is shared between pods.

Symlink-LoRA, a model that emits a pod link, is not implemented. It is a candidate in `evaluations/components/pod-routing` and a design note with a kill criterion in R002.

Decided with it:
- The cluster key excludes aliases (research definition); the identity includes them.
- The head is scoped by project, `(ProjectId, PodIdentity)`, because two projects with equal semantics derive one identity; it is a projection (`pod-head:<project>/<identity>`) rather than a new field of the compacted snapshot.
- The alias-by-alias gate is a training-coverage check, not a proof for unseen spellings.
- The synthetic experiment uses a deterministic stand-in encoder; the sentence encoder is an optional extra.

## Consequences
A pod can be retrained without renaming it, and the identity-to-head step is replayable and auditable. A wrong or stale learned score can cost an abstention or a wrong pod among eligible candidates, never an ineligible one. Because abstention is distinct, a caller (work package 2) can count it, fall back, or fail the request. Learned routing needs a host-provided query embedding: PTR has no Rust encoder, so `ptrd` does not use it yet. A pod generation must be a committed lifecycle target (`pod:<project>/<namespace>/<pod>`) before its representation binds. If R015 shows no gain over the cosine baseline on five seeds, representations stay deactivated and the default router stays the only router; the loader and the head resolver remain useful on their own. Ledger tag 38 is unreadable by older binaries, so readers upgrade first. The activation gate is self-reported: the gate numbers are written by the exporter and re-checked only against the loader's minima, the manifest digest is unsigned (an integrity check against corruption, not authentication of the exporter), the gate is measured on the training data (a coverage check, not a held-out score), and `activated` is advisory (the store recomputes it from the numbers and never trusts the flag; ADR-0023).
