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
| MergePlan | branch | runtime (`preview_merge`, `merge_branch`) | certified revision, one SemanticDelta, rebased keys, plan digest; a read-only what-if: only `merge_branch` commits a sealed branch, certifying it again against the runtime's own state (ADR-0020) |
| SemanticGrant | host | runtime | 1–16 verifiers judging a `SemanticChange`, required level (the weakest report must meet it), host writes on or off, merge policy (its calibration seed never recorded, returned or printed), listed reviewers; installed once, never journaled |
| SemanticOrigin | runtime | ledger/state/pg/replay | why a semantic record was allowed: `Legacy` (tag 8), `Request`, `PodOutput`, `Host { principal, verification }` or `Merge(MergeRecord)` (tag 12): branch, author, seal, plan and dependency digests, rebased keys, verification attestation, authority; replay checks it (R1–R4) |
| MergeReceipt | runtime | host, pg (`record_triage`, `record_outcome`) | branch, seal and plan digests, certification kind, rebased keys, commit (index always set), verdict, recorded authority, triage and its policy version to log |
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

### `LedgerEvent` (the log record and the raft proposal payload)

Stated here because the same codec is both the on-disk record and what members of
a replicated group exchange (D5 in issue #23), and it had none of the above.

- **Schema version:** `ptr_ledger::LEDGER_CODEC_VERSION` (now 2). A record carries
  no version byte of its own; its tag says which version introduced it
  (`event_codec_version`), so every record written before versioning keeps its
  exact bytes (`a_base_record_keeps_its_bytes`). Version 2 adds `PeerAdmitted`,
  `PeerWithdrawn` and `CodecCeilingRaised`.
- **Unknown tag:** refused by name, `PTR_LEDGER_UNKNOWN_TAG`, with the version this
  build reads — never skipped, because skipping would make two members' histories
  mean different things at the same index
  (`an_unknown_tag_is_refused_by_name_and_version_never_skipped`). The decoder and
  the version table cannot drift apart (`every_tag_the_table_admits_decodes`).
- **Upgrade:** a replicated group carries a version's records only after a
  committed `CodecCeilingRaised` for it, which a leader proposes only when the
  host reports **every** voter at that version or above; a voter that reads less
  cannot join, and none can join while a raise is in flight
  (`31-cluster-integrity.md`).
- **Downgrade:** not supported across a version whose records a log holds. The
  older build refuses such a log by name rather than misreading it; a log holding
  none of them is unaffected, byte for byte.

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
