# Step 01 — Cognitive contract, first component

Status: first `ptr-types` increment; not a completed model architecture.
Base reviewed: PR #9 at `2620e6f6`; main at `617c6655` includes training-provenance PR #10.
This increment changes neither the training runner nor the model implementation.

## Scope and ownership

PTR's types describe model-relevant meanings, not just Rust identifiers.
This step builds on PR #9's separated SemanticRole, EpistemicState,
UncertaintyKind and ReasoningOperator instead of inventing a parallel taxonomy.
Only the shared cognitive contract is extended here. Tensor layouts stay with
ptr-core; snapshots with ptr-semdb; authorization with ptr-security.

## Semantic questions must remain distinct

| Axis | Question | Current representation / limit |
|---|---|---|
| Semantic role | Is this a goal, claim, constraint, evidence or other role? | SemanticRole |
| Value type | Does the payload denote a time point, duration, relation, etc.? | TypeId; value schemas remain open |
| Epistemic standing | Is this unknown, assumed, hypothetical, observed or inferred? | EpistemicState; Verified is reported standing, not authority |
| Uncertainty shape | Point, interval or distribution? | UncertaintyKind; numerical value structures remain open |
| Confidence target | Which precise question does the probability answer? | ConfidenceTarget + ConfidenceEstimate |
| Evidence | Which source supports or contradicts the claim? | ProvenanceRef; richer spans/chains remain open |
| Lifecycle | Which generation is live, superseded, disputed or revoked? | Generation + Validity |
| Verification | What checks actually ran and what was their result? | Separate verifier-owned reports |
| Authority | Is the state committed and is an effect permitted? | Runtime/security/ledger, never inferred from confidence |

A constraint's mandatory strength is not its classification confidence. A high
role-classification probability is not a probability that the associated claim
is true. An observed value can still have interval uncertainty. A confident
statement can be revoked or contradicted by another source.

## Implemented first increment

`ConfidenceTarget` distinguishes an explicit semantic-role alternative, value-type
alternative, proposition truth, and reasoning-operator suitability. The owning
record must identify the proposition/context and preserve evidence plus lifecycle
coordinates. These estimates must not be persisted as context-free observations.

`ConfidenceEstimate::new(target, probability)` requires an explicit target and an
already bounded Probability. Missing confidence is `None`, not an implicit 0.5.
The existing legacy Probability default is unchanged and must not be used as an
unknown-value encoding.

`probability_for(&expected)` returns a typed `ConfidenceTargetMismatch` when a
consumer asks a different question, including a different alternative of the
same type family. The error retains expected/actual targets and a stable code.
`probability()` remains an unqualified inspection accessor; it does not establish
that two owners, sources or calibration regimes are comparable.

ConfidenceEstimate intentionally has no Default, no PartialOrd/Ord and no
implicit conversion into VerificationLevel or Effect. These API restrictions
prevent specific accidental conversions, not arbitrary application mistakes.

## Reference cases and executable checks

| Reference case | Required distinction | Test in crates/ptr-types/tests/cognitive_contract.rs |
|---|---|---|
| Hard constraint | Mandatory semantics are not weakened by a confidence update | hard_constraint_role_confidence_is_not_requirement_strength |
| Uncertain claim | Time-point value type, Claim role and Interval shape stay independent | uncertain_claim_keeps_value_type_separate_from_semantic_role |
| Contradictory sources | Separate evidence/estimates; no implicit promotion by averaging | contradictory_sources_keep_their_own_evidence_and_estimates |
| Replaced/revoked information | Full confidence does not relabel an older generation as live | revoked_generation_is_not_relabelled_by_full_confidence |

Additional tests cover all 162 current role/state/shape combinations, missing
confidence versus 0.5, nonfinite/out-of-range probabilities, target mismatches,
and distinct value-type alternatives. Unit tests cover constructor/equality
semantics. Doctests cover normal use and rejected implicit conversions/orderings.

The local Annotation in integration tests is a fixture, NOT a production schema.
These are representation/API tests. They do not implement a conflict resolver,
constraint enforcement, generation invalidation or effect authorization, and
must not be reported as runtime safety or model-quality evidence.

## Inputs versus predictions

Raw source bytes and externally granted capabilities remain external inputs.
Semantic role/value-type assignments may be supplied as oracle labels in an
explicitly labelled mechanism experiment, or predicted from available evidence.
Those conditions must not be mixed in evaluation results. Predicted type identity
and confidence remain proposals until the relevant runtime checks succeed.
Whether full distributions over type assignments improve the model remains an
experiment question; this increment does not hard-code an early argmax policy.

## Compatibility and deliberately deferred work

This is additive: existing Probability fields in SemanticSlot, ModelEvent and
legacy Epistemic<T> are not silently reinterpreted. Their target-aware migration
requires owner-by-owner decisions in the following steps. Burn A0 remains unchanged.

No enum discriminant is a training/checkpoint code. No codebook version is claimed.
No Goal<T>/Claim<T> layout, calibrated distribution implementation, new backend,
LLM training result or stronger-than-RAG claim is introduced here.

## Sequential gates

1. Review this contract; validate ptr-types tests, doctests and synchronized docs.
2. Define only the needed value structures and versioned cognitive codebook.
3. Connect one typed source/snapshot/model/observation path, including real payload revisions.
4. Correct and isolate A0 attention/mask/gradient mechanisms.
5. Run real dataset-backed training against an information/compute-matched baseline.
6. Integrate the trained path into the bounded runtime loop.

Do not open the next implementation step merely because this document exists.
Code changes update component.toml, the generated crate README/status, and the
relevant architecture contract in the same commit. Record check results against
exact commits; distinguish failures already present on the base from regressions.

## Verification commands

```sh
cargo +stable fmt --all -- --check
cargo +stable test -p ptr-types --locked
cargo +1.85.0 test -p ptr-types --locked
cargo +stable clippy -p ptr-types --all-targets --locked -- -D warnings
python scripts/check_component_metadata.py --base HEAD^
python scripts/update_component_docs.py --check
```

A successful compile-fail doctest means the forbidden example failed to compile.
CI logs, not the existence of commands or test markers, establish execution.
Full workspace/PR health remains a separate gate before merging to main.
