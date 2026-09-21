# Open items after PR #21 — what is left, and the plan to close it

*Recorded 2026-09-21. Baseline: `a635980`, the head of PR #21. Committed on
`claude/open-items-after-21`, which is stacked on `claude/repo-overview-126nh9`
because every item below refers to code that exists only there.*

*Revised after an adversarial review of the first draft. What the review changed
is recorded in "What this document got wrong" at the end, rather than quietly
fixed — the same rule #20 applied to its own four wrong findings.*

## Why this document exists

PR #21 closed every requirement box in #20, with implementation, positive **and**
negative tests, exact-commit evidence and a contract per gate. That is not the
same sentence as "nothing is open", and reading it as though it were is how a
closed issue becomes a false claim about a repository.

Three kinds of open work survive #21, and none of them is a box anywhere:

1. **Two of #15's own requirement sentences were only partly satisfied.** A gate
   whose checkboxes are all ticked can still leave a clause of the sentence that
   generated it unaddressed, because the boxes were written after the sentence and
   are narrower than it. Both are named below — Gate 3's and Gate 2's.
2. **Every contract carries a "What this does not close" section.** Six of them
   carry **41 bullets** between them (8 + 6 + 5 + 6 + 8 + 8, for `26`, `28`, `29`,
   `30`, `31`, `32`). Those sections were written to be honest about the boundary,
   not to be forgotten at it.
3. **Two housekeeping items in #20 are unchecked on purpose**, each with a stated
   reason for being left rather than a reason for being done.

And a fourth kind, which this pass found by accident and which nothing would have
found by reading: **three contract bullets are themselves stale**, one of them
asserting that a thing is open which #21 closed. That is item A6.

## How this list was derived

Not from the checkboxes. The checkboxes are all ticked, which is exactly why they
are the wrong instrument here.

- #15's four gate sentences were read clause by clause against the tree, because
  a gate's boxes are a decomposition of its sentence and a decomposition can drop
  a clause.
- The six contracts' open sections were read in full and their bullets counted.
- Every claim was re-checked in the tree. The first draft of this document said
  that too, and was out by ten bullets and wrong in three of four lockfile
  citations in its own flagship finding. Those are corrected here and the
  correction is recorded at the end.

---

## The two dropped clauses

### Gate 3 — "connect actual semantic payloads to the model"

#15's Gate 3 sentence is: *"Replace learned validity-ID hints with independently
enforced validity masks and a shared versioned cognitive codebook; **connect
actual semantic payloads to the model**."* The first two clauses closed in #21.
The third has **no checkbox in #20 at all** — it appears only as a bullet under
what Gate 3 does not close (`26-cognitive-codebook.md`). It is the largest single
piece of open work in the repository. Tier C, item C1.

### Gate 2 — "durable leadership/fencing … local file locks do not fence distributed writers"

#20 ticks this to `6485d5a` and `228e792`. `31-cluster-integrity.md` says
otherwise, in its own open section:

> **Local file locks fence nothing across nodes.** … Durable leadership fencing is
> a protocol property and **it is not implemented**.

And the same document says the opposite seventeen lines earlier, in its body:

> a deposed leader on another host is excluded by the protocol, not by the
> filesystem.

Both sentences cannot be true. The tested property is narrower than either: a
deposed leader appends but cannot commit, and the entry only it held is
overwritten — which is raft's safety argument, and holds while every member runs
this protocol correctly. It is not a *fence*, which is what excludes a writer that
does not. Tier C, item C2, and the contradiction is repaired in A6.

---

## A finding this planning pass turned up

**The retirement checker reports two moves that cannot be made, and classifies
against the wrong property.**

`15e4598` reports `macerator 0.3.4 → 0.4.0` and `netlink-packet-core 0.8.2 →
0.9.0` as retirement candidates. `scripts/check_vendor_retirement.py:169-173`
derives that from `release_key(observed) > release_key(version)` alone — newer
version published, therefore candidate.

Two things are wrong with it, and they are different things.

**First, the dependents forbid both moves.**

- `macerator` is required as `version = "0.3.4"` — i.e. `^0.3.4`, which admits
  nothing in `0.4` — by `burn-ndarray 0.22.0-pre.3`
  (`vendor/burn-ndarray-0.22.0-pre.3/Cargo.toml:140`), `burn-vision 0.22.0-pre.3`
  (`vendor/burn-vision-0.22.0-pre.3/Cargo.toml:186`) and by a third dependent
  that **cannot be seen from this repository at all**: `burn-flex 0.22.0-pre.3`
  declares `macerator = "0.3.4"` as an *optional* dependency, currently
  inactive, so `model/burn-a0/Cargo.lock:593-610` lists burn-flex with no
  macerator among its dependencies. The requirement exists only in burn-flex's
  own manifest, which is in the registry cache and not in the tree — which is
  the same wall the fix below runs into, arriving from the other side.
- `netlink-packet-core` is pinned at `0.8.2` by `netdev 0.45.1`
  (`Cargo.lock:2443`), `netlink-packet-route 0.31.0` (`:2502`), `netlink-proto
  0.12.2` (`:2528`) and `netwatch 0.19.3` (`:2575`), at `^0.8.x`. Meanwhile
  `0.9.0` is **already in the tree** (`Cargo.lock:2487-2489`) for other dependents
  (`:2470`, `:2514`) — the clearest possible demonstration that a version's
  presence upstream is not what decides whether a patch can go.

**Second — and this is the part the first draft of this document also got wrong —
adoptability is not the recorded retirement condition.** Both packages are class
`paste-alias`, whose `retire_when` (`vendor/RETIREMENT.json:5-12`) is *"An upstream
release of this package depends on a maintained macro crate rather than `paste`,
so the alias has nothing left to redirect."* That is a question about the upstream
release's **dependencies**, not about its version number.

And by that condition, `netlink-packet-core 0.9.0` has **already retired**:
`Cargo.lock:2487-2489` shows it with no `dependencies` block at all, so it is
paste-free. What blocks the patch going is only that four dependents still pin
`^0.8.x`. So the honest report is neither "candidate" nor "impossible" but
*"upstream satisfies `retire_when`; blocked by four dependents at `^0.8.x`;
retires when they move"* — which is an actionable, aimed at the dependents.

**Third, bumping the vendored version is not the same operation as retiring the
patch.** Both patches use the renamed-key form (`Cargo.toml:59`,
`model/burn-a0/Cargo.toml:33`), which patches one specific version. Raising
`vendor/macerator-0.3.4` to 0.4.0 would leave `burn-ndarray` and `burn-vision` on
**unpatched upstream 0.3.4** — which still uses `paste` — while cargo prints that
the patch was unused. A report that conflates the two invites exactly that.

So the tool produces a **false actionable**: the same defect class Gate 4 was
opened to fix — a claim that cannot notice when it stops being true — one level
up, in the checker rather than in the prose. This is item A3.

---

## Tier A — work this PR will do

Six items. Each is bounded, needs no decision that is not already recorded, and
closes to #20's standard as restated at the end of this document.

### A1 — `bins/ptrctl` and `bins/ptrd` are in no workspace

**State.** `Cargo.toml:3` declares `members = ["crates/ptr-*", "bins/ptr-*"]`. The
second glob needs a hyphen after `ptr`, which `ptrctl` and `ptrd` do not have;
`exclude` (`Cargo.toml:4`) does not name them either. `cargo metadata --no-deps`
lists exactly 28 packages and neither is among them. `cargo check` inside
`bins/ptrctl` fails outright — *"current package believes it's in a workspace when
it's not"* — because both manifests use `version.workspace = true`. So
`bins/ptrctl/src/main.rs` (89 lines), `bins/ptrd/src/main.rs` (54 lines) and
`bins/ptrctl/tests/doctor.rs` (20 lines) have never been built, tested, linted or
formatted by anything. The doctor test has never run.

**Why it was left.** #20 records the reason: adding them as members subjects
never-linted code to `-D warnings` in the same commit. That is a reason to do it
carefully, not a reason to leave two binaries outside every gate the repository
has.

**What has to be done.**
1. Add both to `members` explicitly rather than widening the glob, so the next
   binary does not silently inherit a decision nobody made.
2. Bring them under the existing gates and fix what that surfaces **in the same
   commit**, without adding an allow or an exclusion. `rustfmt --check` on
   `bins/ptrctl/src/main.rs` already fails on line 10, so `cargo fmt --all --
   --check` goes red the moment the member is added; that is expected work, not a
   surprise.
3. **Parameterise `scripts/check_repo.py` by root and give it a fixture test.** It
   is an 88-line script with a module-level `ROOT` (`:5`), no `check(root)` entry
   point, and — alone among the checkers — no `scripts/tests/test_check_repo.py`.
   Without that refactor the negative test below cannot be written at all, which
   is why it is part of this item rather than an afterthought.
4. Then add the assertion: every directory under `bins/` containing a `Cargo.toml`
   is either a workspace member or explicitly excluded.

**The per-crate registration question is settled, not hedged.** `check_repo.py`,
`check_rust_conventions.py:30` and `check_component_metadata.py:30-40` all iterate
`crates/` only, so a binary needs no `component.toml`, README block or diagram.

**How it is proven.** The real deliverable is not the two entries; it is that the
glob cannot swallow a third binary.

- Positive: `cargo metadata` lists both; `doctor.rs` runs and passes; clippy is
  clean at `-D warnings` on both.
- Negative: a fixture tree with a binary matching neither glob nor exclude makes
  the check fail, naming the directory.

**One consequence to record rather than discover.** `doctor.rs` asserts on
repository *layout* — `bins/ptrctl/src/main.rs:25-42` requires `Cargo.toml`,
`Cargo.lock`, `config/default.toml`, `training/uv.lock`, three registries and
`docs/components/STATUS.md`. All eight exist, so it passes; but running it makes a
layout assertion into a build-gating test. That is stated in the contract, not
left to be found when someone moves a file.

**Contract.** `docs/DEVELOPMENT_ENVIRONMENT.md` has no sentence about workspace
membership today, so this is an addition rather than a changed sentence — said
plainly, because "the contract sentence that changes" is one of the four closing
criteria and this item does not meet it in the usual way.

### A2 — A0 is linted against an MSRV it does not build with

**State.** `clippy.toml:1` says `msrv = "1.85.0"`. `model/burn-a0/Cargo.toml:5`
says `rust-version = "1.95"`. `model/burn-a0` is excluded from the root workspace
(`Cargo.toml:4`) and is its own workspace, but clippy walks ancestors, so the root
`clippy.toml` governs it and wins over the manifest. Every lint whose suggestion
needs anything newer than 1.85 is suppressed in a crate that requires 1.95.

**Not silent — announced and unheard.** Clippy prints *"the MSRV in 'clippy.toml'
and 'Cargo.toml' differ; using '1.85.0' from 'clippy.toml'"* on every
`.github/workflows/burn-a0.yml` run. It is not a named lint, so `-D warnings`
cannot promote it and the exit code stays 0. The cheap fix is genuinely
unavailable, which is what justifies a script — but the justification is "CI
prints it and nothing enforces it", not "nothing reports it".

**What has to be done.** Give `model/burn-a0/` its own `clippy.toml` declaring
`msrv = "1.95"`, carrying the other two thresholds forward so the style gates do
not change at the same time.

**How it is proven.** A drift this small recurs, so the check is the deliverable: a
script asserting that the `clippy.toml` governing each workspace declares that
workspace's `rust-version`. Two details decide whether it works on day one:

- it must compare **semantically**, because the root is `rust-version = "1.85"`
  against `msrv = "1.85.0"` and clippy treats those as equal — a string comparison
  fails on the root workspace immediately;
- it must resolve `rust-version.workspace = true`, which every crate and both
  binaries use.

- Positive: after the change both workspaces agree and the check passes.
- Negative: a fixture pair that disagrees fails, naming both files and both
  versions; and a fixture pair of `1.85` against `1.85.0` **passes**, which is the
  control that the check is comparing versions rather than strings.

**Contract.** `model/burn-a0/README.md` and `docs/DEVELOPMENT_ENVIRONMENT.md`.

### A3 — the retirement report classifies against the wrong property, and cannot see the blockers

**State.** As set out above.

**A property being traded, stated before it is traded.**
`scripts/check_vendor_retirement.py:19-23` makes offline operation the checker's
defining property: "this checker stays offline … so CI never depends on the
network and a report is reproducible from the repository alone." Dependents'
version *requirements* are not in the tree: `Cargo.lock` records resolved versions,
not requirements, and none of the four netlink dependents nor `burn-flex` is
vendored. They are recoverable only from the registry cache, which `cargo fetch`
populates.

So there is a choice, and this plan takes the first branch:

1. **Keep the checker offline.** Record each dependent's requirement in
   `vendor/RETIREMENT.json` as an explicit observation beside `upstream_observed`,
   refreshed by `scripts/refresh_vendor_upstream.py` (which is already the online,
   maintainer-run half). The checker then re-derives what it *can* from the tree —
   that every named dependent exists in the lockfile at a version the record names
   — and reports the rest as a dated observation, exactly as `upstream_observed`
   already is.
2. Give the checker `cargo metadata --offline --all-features`, and lose
   "reproducible from the repository alone".

Branch 1 keeps the property and makes the staleness *visible* rather than absent,
which is the same trade `upstream_observed` already makes. It is chosen because
reversing a defining property of a checker to fix a report is the wrong size of
change for this item.

**What has to be done.**
1. Record, per package, the dependents that constrain it and the requirement each
   declares, with an observation date.
2. Classify against the recorded `retire_when`, not against the version number:
   `upstream_satisfies_retire_when` (for `paste-alias`, whether the upstream
   release is paste-free) is one field, and `blocked_by` is another. A package can
   be both satisfied and blocked — `netlink-packet-core` is.
3. Report the three states distinctly: retirable, blocked-by-dependents (naming
   them), and upstream-not-yet-satisfying.
4. Say explicitly, in the report and in `VENDOR_PATCH_POLICY.md`, that raising a
   vendored version is not retiring a patch, and that a renamed-key patch entry
   silently goes unused if its version stops satisfying the requirement.
5. Define the optional-and-inactive dependent case, since `burn-flex` is one: a
   dependent constrains a bump whether or not its dependency is currently
   selected.

**How it is proven.**
- Fixtures: a package whose dependents all admit the newer release; one blocked by
  a single dependent, named with its requirement; one whose upstream still fails
  `retire_when`; one with an optional-inactive dependent; one with no dependents.
- Against the real tree, as the control that this describes *this* repository:
  `netlink-packet-core` reports **upstream satisfies `retire_when`, blocked by
  four dependents at `^0.8.x`**, and `macerator` reports blocked by three
  including `burn-flex`.
- Negative: a recorded dependent that is not in the lockfile at all, or at a
  version the record does not name, fails the checker — which is the part that
  *can* be re-derived offline, and the honest limit of what can.

**Contract.** `docs/architecture/30-third-party-notices.md` and
`docs/VENDOR_PATCH_POLICY.md`.

**What this does not do.** It retires nothing. And it does **not** conclude that
the number of available retirements is zero — the first draft of this document
said that, and it was wrong: `netlink-packet-core 0.9.0` already satisfies its
retirement condition, and the work it points at is moving four dependents.

### A4 — `provenance_bucket_count` is checked against nothing

**State.** `model/burn-a0/src/lib.rs:152` declares it, `:174` defaults it to 64,
`:201` lets any caller override it via `with_provenance_buckets`, `:219` sizes an
embedding from `self.provenance_bucket_count`. `26-cognitive-codebook.md` records
it as an exception "by decision and not by omission. Its width is unchecked
against anything." Both halves are true; the second is the problem. Every other
cardinality in A0 is the codebook's.

**What has to be done.** Record the exception where the things it is an exception
to are recorded — an `exceptions` section in the codebook artifact naming the
field, its width and why it is not a code space — and check the model against it.

**The check must bind the constructed width, not the default literal.** A check
comparing the artifact to `64` passes for a model actually built at 128 via
`with_provenance_buckets`. The property claimed is "its width is checked against
something", and only a check on the value the model is constructed with is that
property.

**Two constraints found before writing it.**
- Adding an `exceptions` key does **not** move the fingerprint:
  `scripts/check_codebook.py:79-84` digests `canonical_bytes_hex` only, and
  `crates/ptr-types/tests/codebook_artifact.rs:62-64` compares only that field. So
  existing datasets, checkpoints and manifests are not invalidated.
- The artifact's own note says it is generated by `scripts/generate_codebook.py`
  from `ptr-types` and must not be edited by hand. The constant lives in a
  *different, excluded* workspace built by a separate CI job on a separate
  toolchain, so the generator has to learn about it across that boundary. That is
  part of this item, not a detail.

**How it is proven.**
- Positive: a model constructed at the recorded width passes, and the artifact
  states the reason for the divergence.
- Negative, from both sides: a model constructed at another width fails the check
  naming the field; changing the artifact instead fails the same check from the
  other direction. One-sided checks are how the M001 and README staleness in #20
  survived.
- Control: the fingerprint is unchanged by the addition, asserted rather than
  assumed.

**Contract.** `docs/architecture/26-cognitive-codebook.md`.

### A5 — the codebook version guard in `forward` has never been entered

**State.** `model/burn-a0/src/lib.rs:427-431` guards slot codes and `:432-436`
guards epistemic codes. The contract records that the comparison "is unreachable
today. One frozen version means no test can construct a foreign grid from outside
the crate." `Codebook::at` (`crates/ptr-types/src/codebook.rs:198`) refuses an
unknown version, and `CodeGrid.codebook` is set privately, so no external test can
build one. A guard nobody has ever entered is a guard whose behaviour is a claim.

**Both guards are `assert_eq!` — a panic, not a `Result`.** The first draft said
to assert "on the error the guard produces". There is no error. The test is
`#[should_panic]`, and that changes what makes it valid:

- `expected = "another codebook version"` matches **both** guards, so a test aimed
  at the slot-type guard passes when the epistemic guard fires. Each test must
  name its guard's full message.
- `forward` also panics at `:418` (batch mismatch) and `:437-440` (dimension
  checks), so a `should_panic` without an exact message passes on any of them.

**What has to be done.** A `cfg(test)`-only constructor for a grid at a declared
version — there is already a `#[cfg(test)] mod tests` at `:544` — rather than
putting a second version into the public API, which would be inventing a fact to
test a guard.

**How it is proven.**
- Positive control: a grid at the current version passes through `forward`, so the
  test is responsive to something other than the guard.
- Negative, twice: a foreign slot-code version panics with the slot guard's exact
  message; a foreign epistemic version panics with the epistemic guard's exact
  message. Two tests, because one cannot tell the two guards apart.

**Contract.** `docs/architecture/26-cognitive-codebook.md`, and honestly: the
bullet becomes "entered by a test-only constructor", which is weaker than a tested
path and should not be written as though it were one.

### A6 — three contract bullets are stale, one of them false

**State.** Found while reading the open sections for this plan. This is the Gate 4
defect class — a claim that cannot notice when it stops being true — sitting in
the documents that record the gaps.

1. **`28-durable-execution-audit.md:250` asserts something #21 closed.** It says
   "**No network admission and no project-scoped Pods.** Sessions are still
   process-local handles installed by a trusted host, and `PodRegistry::resolve`
   still matches on capability and input type alone." But
   `crates/ptr-pods/src/lib.rs:59-74` resolves on `(project, capability,
   input_type)` and documents it, and `29` describes the design at length. The
   bullet is false in the tree.
2. **`29-peer-admission-and-pod-scope.md:137-139` says the permission/lifecycle
   race "has no test that drives the race from a second session."** #20 ticks
   exactly that test to `ea08e0b` and `crates/ptr-runtime/component.toml` lists it
   under `checks`. One of the two is wrong; the tree decides which.
3. **`31-cluster-integrity.md` contradicts itself** about whether a deposed leader
   on another host is excluded — `:274-275` says the protocol excludes it,
   `:291-294` says durable leadership fencing is not implemented. The narrower
   true statement replaces both, and the open item survives as C2.

Beyond the contracts, `docs/PRIORITIES.md` is stale after #21: `:10` still says
"network authentication, durable idempotency and scoped Pod access remain", `:26`
"multi-node/session hardening remains", and `:42-45` still points at #15 for
"unimplemented network, cluster, compaction … requirements".

**What has to be done.** Correct each against the tree — not by deleting the
bullet, but by replacing it with what is true, since in two of the three cases an
open item survives in narrower form.

**How it is proven.** A prose correction proven by prose is worth nothing, so the
deliverable is the check that would have caught it. Two of the three are
mechanical: a contract bullet asserting that a symbol "matches on X alone" can be
tied to that symbol. The realistic scope here is narrower and is stated as such —
each corrected bullet gets a test in the crate it describes, asserting the
property the bullet now claims, so the bullet and the test fail together.

- Positive: `crates/ptr-pods` asserts resolution is project-scoped (this test
  exists; the bullet is what was wrong).
- Negative: the second-session race test named in `component.toml` is located and
  run, and either the contract bullet or the metadata entry is corrected to match
  what it actually does.

**Contract.** The three files above, plus `docs/PRIORITIES.md`.

---

## Tier B — needs the owner's decision before any code

**B1 — may `ptr-runtime` depend on `ptr-net`?** Carried from #20 unchanged. With
that edge a peer identity becomes unforgeable **at the type level** rather than
un-forged in the one composition that currently gets it right. The cost of not
taking it is written into `32-execution-wire.md`. Not blocking anything.

**B2 — should a receipt be signed?** `32-execution-wire.md`: a receipt is hearsay
outside the connection that carried it. Signing needs a key, a statement of what
it attests to, and a rotation story.

**B2b — should the codebook artifact be authenticated?** Separate from B2 and the
same shape. `26-cognitive-codebook.md`: "A fingerprint detects a mismatch; it
cannot detect someone who recomputes it."

**B3 — where does the accept loop live?** Both `ptr-cluster` and `ptr-execwire`
provide pieces and neither runs a loop, for the stated reason that "when do we give
up" is policy. Service component or embedding host is a deployment decision.

**B4 — journaling the admission policy: which of two designs?** *This was Tier A in
the first draft and does not belong there.* `29-peer-admission-and-pod-scope.md`
records that the policy is in memory, so a withdrawal does not survive a restart
and a replayed history does not say who was admitted when. Closing it is not
mechanical, for three reasons found in the tree:

- **The policy holds behaviour, not data.** `execution.rs:188-192` —
  `PeerAdmission { principal, ttl, grants: Box<dyn Fn() -> Vec<ExecutionGrant> }`,
  where `ExecutionGrant` (`:242-247`) owns a `Box<dyn Verifier>` and a `Dispatcher`
  (`:254-259`) owning a `Box<dyn ActionExecutor>`. "Rebuild the policy from
  committed history" cannot mean what it says. Either grants become a serializable
  description — a breaking change, and a direct contradiction of the reason
  recorded at `execution.rs:180-182` — or history governs **membership** while the
  host still supplies **authority**, in which case the contract bullet is only half
  closed and `install_admission_policy` (`:677-679`), which replaces the whole
  table wholesale, becomes a second source of truth with no stated precedence.
- **`29-peer-admission-and-pod-scope.md:44` is headed "Authority is re-derived at
  use, never remembered."** "Rebuild from history" is the opposite verb. Whichever
  design is taken, that heading has to be reconciled rather than stepped around.
- **New `LedgerEvent` variants are additive on disk and not across nodes.**
  `crates/ptr-ledger/src/lib.rs` encodes with explicit numeric tags (0–11) and
  refuses an unknown tag at `:390-397`, so new tags leave every existing record
  byte-identical and replay and the digest chain are unaffected — verified. But the
  same codec is the raft proposal payload (`raft_node.rs:208`, `:302`, driven over
  `ALPN_RAFT` by `ptr-cluster/src/member.rs:249`), so a member on an older build
  hard-errors on an unknown tag. `docs/COMPONENT_CONTRACTS.md:36-41` requires a
  wire schema version, an explicit unknown-enum behaviour and a downgrade policy of
  every cross-process contract; `LedgerEvent` has none. Adding admission events
  means supplying them.

Two smaller undecided points ride along: `AdmissionPolicy::admit` refuses a
duplicate (`:216-218`), so a replayed admit→withdraw→admit needs a defined meaning;
and the scope, once decided, pulls in `crates/ptr-runtime/component.toml`,
`crates/ptr-ledger/component.toml`, both regenerated READMEs and
`docs/components/STATUS.md`.

**The one test worth keeping from the first draft's version of this item** is its
primary negative: a peer **withdrawn** before the restart is still refused after
it. A recovery replaying only admissions passes a naive positive test and fails
that one. Its two other proposed checks were inert and are dropped: "a policy never
journaled recovers empty" is true of every conceivable implementation because
`ExecutionState::default()` (`:496-509`) always builds an empty policy at `:507`, and "the
refusal is identical to one never admitted" is true by construction because
`admit_peer` has exactly one refusal path (`:688-690`).

**B5 — should reconciliation be authenticated?** `28-durable-execution-audit.md`:
`reconcile_effect` is a privileged host API and who may call it is "the embedding
host's question". `crates/ptr-runtime/component.toml` lists it under `missing`.
Deciding it is a policy choice, not an implementation.

---

## Tier C — separate gates, each needing its own issue

**C1 — connect actual semantic payloads to the model.** The dropped clause of
#15's Gate 3 sentence, above. The largest open piece in the repository.

**C2 — durable cross-node leadership fencing.** The dropped clause of #15's Gate 2
sentence, above. What exists is raft's safety argument among correct members; a
fence excludes a writer that is not one.

**C3 — Pod access across a network boundary.** No protocol is defined over
`ALPN_PODWIRE`, `ALPN_MODEL`, `ALPN_BLOB` or `ALPN_EVENTS`
(`crates/ptr-net/src/lib.rs:4-7`); the only ALPNs any *component* speaks are
`ALPN_RAFT` (`ptr-cluster`) and `ALPN_EXEC` (`ptr-execwire`).

Stated exactly, because an earlier draft of this line said they "carry no
traffic" and that is not true of one of them. `crates/ptr-net/tests/iroh.rs`
binds `ALPN_PODWIRE` and sends `b"ping"` over it, as an arbitrary label for a
transport test proving the connection authenticates its peer. Bytes have
travelled over that ALPN; **Pod access** has not, and a label borrowed by a
smoke test is not a protocol. All six are also referenced in
`crates/ptr-net/tests/smoke.rs` by the distinctness test.

**C4 — a settlement channel for detached work.** Detached grants are refused over
the wire because there is no channel for the adapter's later report. A second
protocol.

**C5 — membership negotiation over the wire.** Each member records a
configuration and recovers it; adding or removing a voter is not implemented.
Worth being exact about where the line falls, because the tree looks like it
might be: `crates/ptr-ledger/src/raft_storage.rs:54-64` encodes and decodes
`EntryConfChange` and `EntryConfChangeV2` entry types, which it must, since it
stores whatever raft hands it. Nothing anywhere **proposes** one — a search of
`ptr-cluster` and `raft_node.rs` for `ConfChange` returns nothing. So the
storage could durably record a membership change that no code can initiate.

**C6 — independent snapshot-anchor retention.** A snapshot's anchor is only as
trustworthy as the leader that sent it; #20's own acceptance paragraph names this
as open.

**C7 — raft's log floor and the ledger's retention floor**, which are independent.
Related: the at-most-once retention obligation in `28-durable-execution-audit.md`
— a compaction floor rising past a settled attempt discards the memory of its key,
and a retry after that executes again.

**C8 — no discovery and no address authority** for the execution wire.

**C9 — a trainer that writes a checkpoint still does not bind it.**

**Not a gate, and therefore not here: session reuse.** Every exchange in both
compositions opens a connection. Both contracts describe it in the same words —
"correct and wasteful". It carries no correctness property, needs no new protocol
and no decision. It is a task in two files, and the first draft listed it as a gate,
which padded the deferral list. It is recorded in the contracts and left there.

**Also recorded and not scheduled**, because each states its own reason for being
unclosable here: Pod project membership is declared rather than proven; licence
expressions are the packages' own unscanned claims; 117 packages ship no licence
text; denied preparations are not audited, which is deliberate.

---

## Order of work

A1 and A2 first: cheapest, no runtime behaviour, and each ends in a checker that
stops its defect recurring — so the rest is done under gates that are correct.

A6 next, because a plan that reads contracts should not leave three of them
asserting things that are not true, and because two of its corrections narrow
items that C2 and others depend on being stated correctly.

A3 next: it changes what the repository *claims* rather than what it does, and the
current claim is wrong in a way a maintainer could act on.

A4 and A5 last, together: both are A0, both cross the excluded-workspace boundary,
and both convert a prose statement into a checked one.

## The standard for closing anything above

#20's standard, plus the two obligations this repository actually enforces that
the first draft of this document left out:

- implementation, plus a positive **and** a negative test bound to it;
- a control on a responsive observable, so a passing test is evidence of the
  property and not of the test being inert;
- the contract sentence that changes, changed in the same commit — or, where there
  is none, that said plainly rather than glossed;
- **updated component metadata**: `scripts/check_component_metadata.py:30-40` fails
  CI unless `crates/<x>/component.toml` changes in the same diff as
  `crates/<x>/src/**`;
- **regenerated component docs**: `scripts/update_component_docs.py` derives
  metrics from source lines, so any source edit moves `crates/*/README.md` and
  `docs/components/STATUS.md`. It runs *after* `cargo fmt`, never before;
- exact-commit evidence.

A green CI run is not completion. A demonstrated happy path is not completion.

## What this document got wrong

The first draft was reviewed adversarially before any of it was implemented. It
was wrong in five ways, recorded here rather than quietly fixed:

1. **It put the admission-policy journaling in Tier A as decision-free.** The
   policy holds closures over trait objects, so "rebuild from committed history"
   cannot mean what it says, and the choice between the two designs is the owner's.
   Moved to B4.
2. **Three of four lockfile citations in its own flagship finding were wrong** —
   one of them citing a line that shows a dependent taking `0.9.0` as evidence for
   `0.8.2`. Corrected above. The draft stated "every claim was re-checked in the
   tree" in the paragraph immediately before them.
3. **It counted 31 open bullets where there are 41.**
4. **It promised two partly-satisfied requirement sentences and delivered one.**
   Gate 2's durable-fencing clause is the second, and `31-cluster-integrity.md`
   contradicts itself about it.
5. **Its A3 fix classified the wrong property** — semver adoptability rather than
   the recorded `retire_when` — concluded that zero retirements are available when
   `netlink-packet-core 0.9.0` already satisfies its condition, conflated raising a
   vendored version with retiring a patch, missed the `burn-flex` dependent, and
   assumed dependents' requirements are in the tree when they are not.

Four proposed tests were also inert or mismatched and are repaired above: A4 bound
a default rather than the constructed width, A5 asserted on an error where the
guard panics, and two of the admission-policy checks could not fail.

## What this plan deliberately does not schedule

Legal clearance for distribution; any claim about model quality, latency,
throughput or behaviour under load; Byzantine fault tolerance; obtaining licence
texts for the 117 packages that ship none; and the six Tier B decisions, which are
recorded rather than taken.
