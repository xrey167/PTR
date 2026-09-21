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

## The model half — enforcement inside Burn A0

Added at `0c1462b`'s successor; A0 sits on its own toolchain with its own CI job, so
it is a separate change on the same contract.

### Validity left the feature path entirely

A0 used to carry a learned `validity_embedding`, summed into `typed_metadata` and
from there into the attention scores through `cross_bias`. That made lifecycle
validity a **hint**: one term among several, which a confident pattern could
outvote — and the fact being outvoted was "this generation was revoked".

The embedding is gone. `validity_ids` and `validity_count` are gone with it, and
the count is worth recording as its own small lesson: it defaulted to **8** while
`V1_VALIDITIES` has exactly **4** members, so half that table indexed nothing the
contract names.

Validity now enters as `PtrSlotMetadata::admission`, an additive bias built by
`admission_bias` from a `ValidityMask` the kernel computed. A0 depends on
`ptr-types` for it — a crate with no dependencies of its own, so the shared
codebook costs that workspace nothing, and "shared" is the point of the gate.

### Attention was not the only way in

Masking the raw-to-slot attention is necessary and was not sufficient. The router
averages over slots:

```
router_logits = router(slots).mean_dim(1)
```

so an excluded slot reached the output through the mean no matter how it was
attended to. The router now sums with the admission as a weight and divides by the
admitted count. Two paths, both closed, and the second one only shows up if you
look for it rather than stopping at the softmax.

### A row that admits nothing

`ValidityMask::admitting_none` is representable, and a softmax over nothing but
negative infinity is NaN — which would not merely lose that row, it would poison
the whole batch. Such a row gets a finite bias so the softmax stays defined and its
context is dropped afterwards instead. With no admissible typed state there is
nothing to attend to, which is an answer rather than a crash.

This was found by asserting it, not by reasoning about it: the test failed before
the fix existed.

### How it is tested, and why not on attention weights

The tests never inspect attention weights. They assert the property instead: **an
excluded slot's contents cannot change anything the model produces.** The slot's
value is set to `1.0`, `-50.0` and `1000.0` and the output must be bit-identical,
across several initialisations — because the claim is about the construction, not
about one lucky set of weights.

The control needed care. `raw` is a poor observable: scores scale with slot values,
so a large change saturates the raw-side softmax onto whichever slot wins, and when
that is not the slot being varied, `raw` stays constant for a reason that has
nothing to do with admission. That is exactly how a test can pass while proving
nothing. The control therefore uses the router, which responds to every admitted
slot.

`Revoked`, `Superseded` and `Disputed` are each checked, with `Live` in the same
position as the control. `Disputed` is the deliberate one: excluded rather than
down-weighted, because letting the model reason from a value the verifier fabric
has contradicted would put a learned score above a deterministic finding.

The excluded slot's own row in `slots` is still computed and returned. Nothing the
model produces depends on it, and `PtrA0Output::admission` is returned alongside so
a consumer pooling `slots` cannot lose the mask.

## The codebook as data, for everything outside Rust

`validate_dataset.py` carried its own copies of the operator, epistemic and
uncertainty member names as Python sets. That is precisely the duplication this
type exists to prevent: adding a `ReasoningOperator` variant in Rust left the
Python set silently disagreeing, and nothing failed until a dataset meant
something different than it said.

`datasets/generated/codebook.json` is now the single readable form — version,
every family with its members in code order, the canonical bytes as hex, and a
SHA-256 fingerprint of those bytes. It is produced by
`crates/ptr-types/examples/codebook.rs` (an example rather than a binary, so the
ordinary `--all-targets` run builds and lints it) with the fingerprint computed by
`scripts/generate_codebook.py`, because this crate still has no hasher and no
dependencies.

Three checks hold the chain together, and each catches something the others
cannot:

| Check | Catches |
|---|---|
| `crates/ptr-types/tests/codebook_artifact.rs` | the kernel's tables and the artifact disagreeing |
| `scripts/check_codebook.py` | a hand-edited fingerprint, a cardinality that contradicts its member list, a non-dense code sequence, a member absent from the canonical bytes |
| `validate_dataset.py` | a dataset that names no codebook, an unknown version, a moved assignment |

The drift test was verified by breaking the artifact rather than by reasoning
about it: changing one byte of the hex fails the canonical-bytes assertion, and
removing a member fails the family assertion. Different faults, different
failures.

`type_codebook_version` and `type_codebook_fingerprint` are now **required** in a
dataset record, with **no defaulting path**. Treating "absent" as "current" is how
a stale integer assignment gets adopted without anyone deciding to. The two
failures give distinct reasons, deliberately: the version can be right while the
table behind it has moved, which is exactly the silent remapping the fingerprint
exists to catch.

Codes are checked for being dense `0..n` because they index an embedding table
directly. A gap would leave a row nothing can reach, and a duplicate would make
two members share one.

## What this does not close

- **Checkpoints and run manifests do not record it yet.** Datasets do. A
  checkpoint carrying a `StateBinding`, which is what `27-neural-state-admission.md`
  would need to decide about a real artifact rather than opaque test bytes, is
  still open.
- **`slot_type` is still research-local.** Mapping A0's `slot_type_count` onto
  `SemanticRole` codes from this codebook is not done; slot identity therefore
  remains outside the versioned contract.
- **A consumer can still misuse `slots`.** The admission travels with the output,
  but nothing forces a caller to apply it. Zeroing the excluded rows would hide a
  real zero vector, so the mask is supplied rather than baked in.
- **The artifact is trusted as data, not as authentication.** A fingerprint detects
  a mismatch; it cannot detect someone who recomputes it. That was true of
  `canonical_bytes` from the start and is unchanged by writing it to a file.
- **No semantic payloads reach the model.** "Connect actual semantic payloads to the
  model" remains open; this supplies the identity layer such a connection needs.
- Nothing here is evidence about model quality, reasoning improvement or the
  research Definition of Done. A codebook makes a claim checkable; it does not make
  it true.
