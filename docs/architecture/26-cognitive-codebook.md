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

## The model's code spaces are the codebook's

A0 used to take the width of every typed table as an argument, and the numbers its
callers chose disagreed with the kernel:

| Table | Caller's width | Kernel's cardinality |
|---|---|---|
| slot type | 8 | 9 semantic roles |
| epistemic state | 8 (default) or 4 | 6 epistemic states |
| router output | 3, 4 or 5 | 11 reasoning operators |

Both directions of that mismatch are silent. A table longer than its family trains
rows that denote nothing. A table shorter than its family cannot represent the
members past its end — and since the ids were bare integers, an index past the end
was equally unremarkable.

`PtrA0Config` now derives all three from a `Codebook`, which is the only way to
obtain them. `provenance_bucket_count` is **the one recorded exception**: provenance
bucketing is a research-local hashing of sources with no kernel taxonomy behind it,
so it has no members to assign codes to, and a family with no members is not a
family. Naming it an exception is the point — an unexamined free parameter next to
three derived ones is how the next mismatch gets in.

Naming it, though, was all that was done at first, and a name is prose. The width
lived as the literal `64` in one Rust file, where nothing outside Rust could see it
and no check compared it to anything — so a dataset builder bucketing into 128 and
a model built at 64 would have agreed about nothing and complained about nothing.
The failure is the one the codebook exists to prevent, one level down: ids bucketed
into 64 index a 128-row table perfectly well, and the model reads a provenance it
was never given.

So the exception is recorded where the taxonomy is recorded. `ptr_types::EXCEPTIONS`
holds its name, width and reason; the generated artifact carries an `exceptions`
section beside `families`, so everything outside Rust reads one number; and A0's
default resolves from that record through a `const fn`, which means removing the
record fails the **build** rather than falling back to a plausible 64.

**An exception stays outside `canonical_bytes`.** The fingerprint commits to an
assignment of codes and an exception assigns none, so folding one in would move the
fingerprint of a version whose codes had not moved — invalidating every dataset,
checkpoint and run manifest bound to it for a change that renamed nothing. Adding
this section left the fingerprint at
`2b6f8175a7a7bb648910e7acb17dcfcff5149834f2fe87355c5d2c4524c113a3`, unchanged, and
a test asserts the exclusion rather than leaving it to the next person to notice.

**It is a recorded default, not a constraint.** `with_provenance_buckets` still
takes any width, because an experiment may legitimately bucket differently. What
stops two widths meeting silently is the embedding's own shape: a checkpoint
written at one width is refused by a model built at another. Worth being exact
about *which* check that is — it is burn's record validation, which reports a shape
mismatch and not a field name. The identity header does not consult the width at
all, because the width is not a family and the header has no slot for one. Both
facts are asserted in `tests/checkpoint.rs` rather than assumed, since "the header
checks it" is what a reader would otherwise suppose.

### Codes are minted, not typed

```rust
let roles: [&[SemanticRole]; 1] = [&[SemanticRole::Goal, SemanticRole::Claim]];
let slot_types = CodeGrid::new(&Codebook::V1, &roles, &device)?;
```

`CodeGrid<T>` is the only thing `forward` accepts, and the only way to build one is
through `Codebook::code_of`. So an index past the end of a table, or a code from
another family that happens to be in range, is unrepresentable rather than
unlikely — an embedding lookup would have answered both without complaint.

The family is a **type parameter**, so passing epistemic codes where semantic roles
belong does not compile; a `compile_fail` doctest asserts that, paired with the
otherwise identical doctest that does compile, because a `compile_fail` that fails
for an unrelated reason passes vacuously. The codebook version travels in the grid
and is compared in `forward`. With one frozen version that comparison cannot be
reached from outside — it exists so that adding a second version fails loudly
rather than reading v2 codes against v1 tables.

## What a stored checkpoint has to carry

Burn records parameters and nothing else: its constant fields are recorded as
empty. A bare record therefore cannot say which assignment its codes belong to,
and weights whose codes have been renumbered keep loading, keep producing
plausible outputs, and are now reading a different taxonomy than the one they
learned. Nothing fails.

`ptr_types::CheckpointHeader` is the identity prefix that closes it: magic, an
explicit format version, the model's stable name, the codebook version, **the
complete assignment verbatim**, and the table widths the weights actually have.

The assignment is stored verbatim rather than as a digest for two reasons. This
crate has no dependencies and therefore no hasher, and a byte comparison is exact
where a digest is only probably exact. A reader that wants a fingerprint — as
`StateBinding` does — hashes `Codebook::canonical_bytes` itself.

The widths in the header are read from the weights, not from the config that built
them, so the header cannot claim a width the tensors do not have.

Every refusal is distinct and none of them falls back:

| Refusal | What it catches |
|---|---|
| `PTR_CKPT_NOT_A_CHECKPOINT` | foreign or empty bytes |
| `PTR_CKPT_UNKNOWN_FORMAT` | a layout this build does not know, rather than parsing it anyway |
| `PTR_CKPT_TRUNCATED` / `PTR_CKPT_PAYLOAD_LENGTH` | an artifact that ends inside a field or inside its payload |
| `PTR_CKPT_TRAILING_BYTES` | writer and reader disagreeing about the layout |
| `PTR_CKPT_UNKNOWN_FAMILY` | a family name this build does not define |
| `PTR_CKPT_DUPLICATE_FAMILY` | one family with two widths, so neither is its width |
| `PTR_CKPT_UNKNOWN_CODEBOOK_VERSION` | codes from a version with no tables here |
| `PTR_CKPT_CODEBOOK_MOVED` | **the same version with a different assignment** |
| `PTR_CKPT_TABLE_SIZE` | weights built for eight roles where the kernel defines nine |
| `PTR_CKPT_MISSING_TABLE` | a family the reader requires and the artifact never recorded |

`CODEBOOK_MOVED` is the one the version number cannot catch, and it is why the
assignment is carried at all.

A family name is encoded as its stable name and decoded through an explicit table
in both directions — never a position in `CodeFamily::ALL`, which a reordering
would silently follow. That is the same rule as codes not being discriminants,
applied one level up.

Verification names the families the **reader** requires. A model that embeds fewer
families than the kernel defines is not thereby wrong, so the list cannot live in
the kernel: `ptr-burn-a0` names its own as `EMBEDDED_FAMILIES`. Naming none still
checks the assignment, which is the part that decides what every code means.

## A real checkpoint reaching the runtime's admission

`PtrRuntime::bind_checkpoint` reads the header, refuses an artifact whose
assignment this build cannot reproduce, refuses an artifact and a declaration that
disagree about the version — one of the two is wrong about what every code means
and there is no way to tell which from here — and then binds the payload through
the same `bind_state` path every other neural state uses. One rule set, not a
producer's copy and a consumer's copy that can drift.

The payload stays opaque. Nothing in `ptr-runtime` can read burn's parameter
format, and nothing needs to: what a checkpoint must not do is load under an
assignment it was not trained under.

`crates/ptr-runtime/tests/fixtures/ptr-a0-v1.ckpt` is a **real** A0 checkpoint —
header from the shared kernel, weights from burn — regenerated by
`model/burn-a0/examples/write_checkpoint.rs`. The two workspaces do not build
together, so without a committed artifact nothing would catch them disagreeing
about the format. The tests bind it, seal it, reopen it against its anchor, and
check that the weights survive unchanged; then they move the generation it was
bound under and edit the semantic input it read, and it is refused both times.

## A run manifest records what it is interpretable against

`training/configs/run-default.toml` now carries a `[codebook]` section with both
`version` and `fingerprint`, and `build_manifest` refuses a run whose config names
neither, only one, an unknown version, or a moved assignment — four distinct
reasons, no defaulting path. The artifact's own file digest goes into the run's
provenance, so it reaches `input_fingerprint_sha256` rather than only being
reported next to it. The manifest's `schema_version` moves to 4.

One loader serves the whole training stack: `ptr_training.codebook` reads the
artifact, verifies that its recorded fingerprint really is the digest of its own
canonical bytes — an artifact that disagrees with itself identifies nothing — and
`validate_dataset.py` now uses it instead of keeping a second copy of the same
check.

## What this does not close

- **A trainer that writes a checkpoint is still not the thing that binds it.**
  The artifact carries its assignment; the committed facts are attached when a
  runtime binds it, and there is no training loop here that does so as part of
  saving. The seal is the recorded form, and producing one is a separate step.
- **The codebook version comparison in `forward` is entered by a test, which is
  weaker than a second version existing.** It used to be unreachable: one frozen
  version means `Codebook::at` refuses every other, and `CodeGrid`'s version is
  private, so nothing outside the crate could build a foreign grid. A guard nobody
  has ever entered is a guard whose behaviour is a claim, so a `cfg(test)` module
  inside the crate relabels a well-formed grid as version 2 and drives both
  branches — which is the whole hazard in one operation: valid codes, right shape,
  only the assignment they were minted under changed.

  Two details are what make those tests evidence. Each names its guard's **full**
  message, because both guards say "another codebook version" and a shorter
  expectation would let a test aimed at the slot guard pass when the epistemic one
  fired; and there is a control, because `forward` also panics on a batch mismatch
  and three dimension checks, so a bare `should_panic` passes on any of five
  unrelated faults. Removing either guard fails exactly the test aimed at it.

  What is still true: this is a test-only constructor, not a second version. The
  guard has never been exercised by a real foreign artifact, and it will not be
  until there is one.
- **The header is not a manifest.** It records identity, not architecture: a
  checkpoint loaded into a differently shaped model is refused by burn's own
  record validation, which reports shapes rather than saying which model it is.
- **`provenance_bucket_count` remains outside the codebook**, by decision and not
  by omission — but it is no longer unchecked. Its width is a recorded exception
  in `ptr_types::EXCEPTIONS`, published in the artifact's `exceptions` section, and
  resolved into A0's default by a `const fn` so a missing record fails the build.
  What is still true is narrower and worth keeping: **the width is checked by
  shape, not by name.** A checkpoint at another width is refused by burn's record
  validation reporting a tensor mismatch, and the identity header never looks at
  it. Naming that refusal would mean carrying a non-family width in the header,
  which is a format change to `CheckpointHeader` and a decision about what a
  checkpoint's identity is for.
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
