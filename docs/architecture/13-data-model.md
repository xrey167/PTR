# Data Model

## Core identifiers

`RequestId`, `ProjectId`, `ArtifactId`, `CapsuleId`, `PodId`, `CandidateId`, `Revision`, `Generation`, `CommitIndex`.

## Epistemic domain

Values distinguish Unknown, hypothesis/distribution, observed/inferred and verified/known states. Provenance is attached to evidence-strengthening transitions.

## Lifecycle

Semantic objects are generation-addressed. Supersede/revoke creates a lifecycle event; it does not silently mutate history.

## Evidence

Raw evidence is content-addressed where possible. Normalized/parsed views point back to the original artifact and offsets/ranges where available.
