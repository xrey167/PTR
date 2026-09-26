# PTR Component Contracts

## Contract rule

Crates exchange **domain types**, not provider-native objects. Provider structs are confined to adapters.

## Shared type-kernel rule

`ptr-types` is the shared cognitive/semantic kernel, not merely an identifier crate. Cross-layer concepts such as semantic role, epistemic state, uncertainty shape and reasoning-operator identity belong there when model/core/router/training/runtime must interpret them identically.

Component aggregates remain with their owning crate. For example, `SemanticSlot` remains in `ptr-core`, while the slot's `SemanticRole`, `EpistemicState` and `UncertaintyKind` come from `ptr-types`.

The axes must not be collapsed into one enum: semantic role, epistemic state, uncertainty representation, lifecycle validity, verification and authority are distinct concepts.

## Core contracts

| Contract | Producer | Consumer | Critical fields |
|---|---|---|---|
| RawInput | ingress/server | ingress/SemDB | source, bytes/text, artifact ref, trust |
| TypedProposal | ingress/core | SemDB/verifier | intent, entities, constraints, uncertainty |
| SemanticSnapshot | SemDB | core/router | Revision, live generations, dependency view |
| ModelEvent | model backend/core | runtime | event type, request, revision, payload |
| ExecutionRecipe | router | exec | operators, budgets, dependencies, fallback policy |
| PodRequest | router/exec | pods | capability, typed input, revision/generation, effect |
| Observation | pods/search/env | verifier/SemDB | provenance, source generation, payload |
| VerificationReport | verifier | SemDB/security/feedback | status, level, findings, confidence |
| ActionIR | core | security/exec | capability, effect, target, revision/generation |
| LedgerEvent | security/verifier | ledger | subject, generation, event type, evidence refs |
| CommittedEvent | ledger | state/events | commit index, term/epoch where applicable |
| SemanticCapsule | memory | search/core | generation, claims, provenance, validity |
| EvidenceCandidate | search | verifier | source ref, index generation, score, method |
| FlowSignature | core/runtime | observe/verifier | selected operators and transitions |
| CommittedEvent + LogAnchor | ledger | pg projector | commit index, anchor digest recomputed from the stored anchor |
| SealedBranch | branch | pg branch store/certifier | base revision, value/range digests, touched keys' base values and input-set digests, relied generations, typed ops; private, built only through `SealedBranch::from_parts`, which checks every sealing invariant |
| MergePlan | branch | runtime (verified delta) | certified revision, one SemanticDelta, plan digest |
| TriageOutcome | branch | pg/analytics | decision, eligibility, calibration slice, score, propensity, policy version |
| PolicyRecord | branch | pg policy store, F003 | version, threshold rule and levels, threshold, calibration rate, calibration branches (held out: every other adjudication) |
| WriteRequest/SourceRef | fastmem | pg journal | source key, generation, input digest, raw key/value cells, gates |
| MetricSpec/MetricRow | analytics | pg (compiled SQL) | metric, grouping, window of days, numerator, denominator |

## Compatibility

Every cross-process/network contract needs:
- wire schema version;
- semantic contract version;
- explicit unknown-field/unknown-enum behavior;
- downgrade/upgrade policy;
- idempotency key where effects may be retried.

## Anti-contracts

The following must not cross boundaries as authoritative state:
- raw search score without backend/method metadata;
- arbitrary JSON map standing in for a domain type;
- hidden model state used as durable memory;
- provider tool name used as semantic capability identity;
- unverified source excerpt used as Known<T>.



## Open contract catalog

The table above names the stable cross-component concepts already visible in the architecture. The broader design surface is tracked in the machine-readable [open architecture catalog](../research/catalogs/README.md):

- `type-families.toml` records type families that exist, remain provisional, or are still open.
- `backend-slots.toml` records replaceable backend categories and unresolved selection questions.
- `component-contracts.toml` records required ports and open boundary decisions per crate.

These catalogs are intentionally not an implementation commitment. They exist so future type/backend additions are evaluated against an explicit slot instead of being introduced ad hoc.
