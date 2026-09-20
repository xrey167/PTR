# P0.6 — Versioned cognitive codebook and enforced validity masks

Baseline: `4ad14d961639b7e4b0db65cef283f06738ee17f5`.
Owner: `ptr-types::{codebook, validity_mask}`.

Issue #15's *Neural lifecycle/codebook* gate asks for two things: "independently
enforced validity masks and a shared versioned cognitive codebook". Both are
implemented here, in the type kernel, where the taxonomy they describe already
lives. What remains is applying them inside Burn A0's tensors, which is stated at
the end.

This also unblocks a second gate: *Checkpoint and neural-state admission* requires
binding checkpoints to a "codebook version", which until now did not exist.

## Why a discriminant cannot be the code

A model that embeds `SemanticRole` needs an integer per role, and the obvious way
to get one is `role as u16`. That is a trap. Rust enum discriminants follow
declaration order, so inserting a variant or sorting the list renumbers every role
after it. Weights trained against the old numbering keep loading, keep producing
plausible outputs, and now mean something else. Nothing fails — the model is simply
wrong about what it is looking at, and no test that does not assert exact numbers
would notice.

So the assignment is explicit data. A code is a member's position in a versioned
table, and `a_code_is_never_a_discriminant` asserts every exact numeric value, so
reordering a kernel enum breaks a build instead of a checkpoint.

```text
semantic_role       0 goal · 1 constraint · 2 claim · 3 evidence · 4 resource
                    5 capability · 6 relation · 7 procedure · 8 action
epistemic_state     0 unknown · 1 assumed · 2 hypothesis · 3 observed
                    4 inferred · 5 verified
uncertainty_kind    0 point · 1 interval · 2 distribution
reasoning_operator  0 semantic · 1 deductive · 2 probabilistic · 3 statistical
                    4 temporal · 5 causal · 6 search · 7 optimization
                    8 simulation · 9 symbolic · 10 external_pod
validity            0 live · 1 superseded · 2 revoked · 3 disputed
```

`UncertaintyKind` is a family of its own, which closes one of the alignment gaps
Burn A0's README lists: distribution semantics no longer have to be folded into
slot type or epistemic state.

### Properties the tables have to keep

- **Dense.** Codes are `0..cardinality` with no gaps or duplicates, because they
  index embedding tables directly. `cardinality_of` is how a model sizes those
  tables, from the contract rather than from a constant copied next to it.
- **Append-only per version.** V1's tables are never reordered and entries are
  never removed; a change goes into a new version so artifacts recorded against V1
  keep meaning what they meant.
- **Version-bound.** `TypeCode` cannot be built from an arbitrary integer. It only
  exists as the output of `code_of` or the input to `member_of`, both of which name
  a version, because a code without its version identifies nothing.
- **No defaults.** An unknown version, an unassigned code and a member with no code
  in this version are three distinct refusals. A defaulted code is exactly how a
  stale or foreign identity gets admitted unnoticed.

### Binding artifacts to the assignment

`canonical_bytes` serializes the whole assignment — version, then each family's
name, cardinality and member names in code order. Hash it to bind a checkpoint,
dataset or run; any reordering, addition or removal changes the bytes. A test
decodes them with an independent decoder and asserts the exact expected structure,
in the same spirit as the repository's existing hashlib golden vectors.

Hashing lives in the caller because `ptr-types` deliberately has no dependencies.
This is a **commitment, not authentication**: it detects a mismatch and cannot
detect an attacker who recomputes the hash. Recording the version in datasets,
checkpoints and run manifests is the consumers' obligation and is not yet done.

## Why a mask, not a hint

Burn A0 currently feeds the lifecycle in as a *learned validity-id hint*: validity
becomes an embedding and the network is trained to take it into account. That is the
wrong shape for this fact. A hint is something the model weighs against everything
else, so a sufficiently confident pattern can outvote it — and the fact being
outvoted is "this generation was revoked". Global invariant 3 does not admit a
probability.

`ValidityMask` is computed from committed lifecycle state and applied by
construction. Attention to an excluded slot is not unlikely; it is unrepresentable.

- **Only `Live` admits.** `Superseded` and `Revoked` are what the invariant is
  about. `Disputed` is excluded too, deliberately: a disputed value is one the
  verifier fabric has contradicted, and reasoning from it while the contradiction
  stands would put a learned score above a deterministic finding, which global
  invariant 11 forbids. An exhaustive match means a new `Validity` variant cannot
  compile until its admission is decided.
- **Narrowing only.** `narrow` and `exclude` can remove admission; nothing widens
  it. `is_narrowing_of` lets a pipeline check that a step did not reopen a slot.
  Narrowing with an all-admitting mask cannot bring an excluded slot back, which is
  asserted.
- **`attention_bias` uses negative infinity**, not a large negative constant. A
  finite penalty is a strong hint and a strong hint can be overcome; `-inf`
  contributes exactly zero weight after a softmax, which the test checks via
  `exp()`.
- **An out-of-range read is never an admission.** `admits` returns `bool` rather
  than `Result` on purpose: a caller iterating slots should not be able to turn a
  bounds mistake into an admission, and `false` is the safe reading.

The separation is the architectural claim: an excluded slot keeps its perfectly
well-formed metadata codes. Admission is not encoded in the embedding, so no amount
of training can recover it.

## Executed evidence

11 unit tests in `ptr-types::{codebook, validity_mask}` plus 5 integration tests in
`crates/ptr-types/tests/codebook.rs`:

- Every exact numeric code for all 33 members of all five families.
- Exhaustive coverage per family, so adding a kernel variant fails to compile until
  the "new version or append" question is answered.
- Codes dense `0..cardinality`, member names distinct, cardinalities exact.
- Unknown version, unassigned code and unassigned member each refused with their own
  stable code; no path defaults.
- `canonical_bytes` decoded by an independent decoder against the exact expected
  assignment, with every byte accounted for; committed order equals code order.
- Only `Live` admits, every `Validity` decided exhaustively.
- A revoked slot keeps its codes and still cannot participate; its bias is `-inf`
  and `exp()` of it is exactly zero.
- Masks narrow and never widen, including against an all-admitting mask.
- Length mismatch and out-of-range refused without mutating the mask.

## What this does not close

- **Burn A0 still embeds a validity id.** Replacing that path with
  `ValidityMask::attention_bias` inside its attention, and mapping its research-local
  `slot_type` onto `SemanticRole` codes, is the remaining half of this gate. Burn A0
  sits outside the production workspace on a different toolchain and has its own CI
  job, so it is a separate change.
- **Nothing records the version yet.** Datasets, checkpoints and run manifests have
  to carry `CodebookVersion` and a fingerprint of `canonical_bytes`; that wiring,
  and the admission decision built on it, belong to the *Checkpoint and neural-state
  admission* gate, which this unblocks rather than closes.
- **No semantic payloads reach the model.** "Connect actual semantic payloads to the
  model" remains open; this supplies the identity layer such a connection needs.
- Nothing here is evidence about model quality, reasoning improvement or the
  research Definition of Done. A codebook makes a claim checkable; it does not make
  it true.
