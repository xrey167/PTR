# Open items after PR #21 — what is left, and the plan to close it

*Recorded 2026-09-21. Baseline: `claude/repo-overview-126nh9` at `a635980`, the
head of PR #21. This document is stacked on that branch because every item below
refers to code that exists only there.*

## Why this document exists

PR #21 closed every requirement box in #20, with implementation, positive **and**
negative tests, exact-commit evidence and a contract per gate. That is not the
same sentence as "nothing is open", and reading it as though it were is how a
closed issue becomes a false claim about a repository.

Three kinds of open work survive #21, and none of them is a box anywhere:

1. **Two of #15's own requirement sentences were only partly satisfied.** A gate
   whose checkboxes are all ticked can still leave a clause of the sentence that
   generated it unaddressed, because the boxes were written after the sentence and
   are narrower than it.
2. **Every contract carries a "What this does not close" section.** Those sections
   were written to be honest about the boundary, not to be forgotten at it. Six
   contracts carry thirty-one such bullets between them.
3. **Two housekeeping items in #20 are unchecked on purpose**, each with a stated
   reason for being left rather than a reason for being done.

## How this list was derived

Not from the checkboxes. The checkboxes are all ticked, which is exactly why they
are the wrong instrument here.

- #15's four gate sentences were read clause by clause against the tree, because
  a gate's boxes are a decomposition of its sentence and a decomposition can drop
  a clause.
- The six contracts' own open sections were read in full
  (`26`, `28`, `29`, `30`, `31`, `32`).
- **Every claim below was re-checked in the tree rather than trusted from the
  prose**, which is how the finding in the next section turned up. The prose was
  right about the facts it stated and wrong about what followed from them.

## A finding this planning pass turned up

**The retirement checker reports two candidates that cannot be taken.**

`15e4598` compares each pinned vendored version against the newest published
upstream release and reports two moves as available: `macerator 0.3.4 → 0.4.0`
and `netlink-packet-core 0.8.2 → 0.9.0`. `30-third-party-notices.md` records that
"retirement is not performed — the two candidates are reported, not taken", which
reads as a matter of effort not yet spent.

It is not. Neither move is available at all:

- `macerator` is required as `version = "0.3.4"` by `burn-ndarray 0.22.0-pre.3`
  (`vendor/burn-ndarray-0.22.0-pre.3/Cargo.toml:140`) and by
  `burn-vision 0.22.0-pre.3` (`vendor/burn-vision-0.22.0-pre.3/Cargo.toml:186`).
  A caret requirement on `0.3.4` admits nothing in `0.4`. Upstream `0.4.0` is
  unreachable until burn itself moves.
- `netlink-packet-core` is required as `version = "0.8.0"` by
  `netlink-packet-route 0.31.0`, and is pulled at `0.8.2` by `netdev 0.45.1`,
  `netlink-proto 0.12.2` and `netwatch 0.19.3` (`Cargo.lock:2443,2470,2502`).
  `0.9.0` is in fact **already in the tree** for a different dependent
  (`Cargo.lock:2488`), which is the clearest possible demonstration that its
  presence upstream is not what decides whether the patch can go.

So the tool produces a **false actionable**: a maintainer who acts on the report
spends the work of a retirement and discovers at the end that the resolver was
never going to accept it. That is the same defect class as the one Gate 4 was
opened to fix — a claim that cannot notice when it stops being true — one level
up, in the checker rather than in the prose.

This is item **A3** below, and it is the reason the plan does not simply say
"perform the two retirements".

---

## Tier A — work this PR will do

Six items. Each is bounded, needs no decision that is not already recorded, and
is closed to the same standard as #20: implementation, a positive **and** a
negative test bound to that implementation, and the contract sentence it changes.

### A1 — `bins/ptrctl` and `bins/ptrd` are in no workspace

**State.** `Cargo.toml:3` declares `members = ["crates/ptr-*", "bins/ptr-*"]`.
The second glob needs a hyphen after `ptr`, which `ptrctl` and `ptrd` do not
have; `exclude` (`Cargo.toml:4`) does not name them either. `cargo metadata`
lists 28 packages and neither is among them. So `bins/ptrctl/src/main.rs`
(89 lines), `bins/ptrd/src/main.rs` (54 lines) and `bins/ptrctl/tests/doctor.rs`
(20 lines) are not built, not tested, not linted and not formatted by anything
in this repository. The doctor test has never run.

**Why it was left.** #20 records the reason: adding them as members subjects
never-linted code to `-D warnings` in the same commit. That is a reason to do it
carefully, not a reason to leave two binaries outside every gate the repository
has.

**What has to be done.**
1. Add both to `members` explicitly rather than widening the glob, so the next
   binary does not silently inherit a decision nobody made.
2. Bring them under the existing gates — `cargo fmt`, `clippy -D warnings`,
   `cargo test` — and fix whatever that surfaces **in the same commit**, without
   adding an allow or an exclusion.
3. Register them where every other member is registered, if the repository
   checkers require it of a binary.

**How it is proven.** The real fix is not the two entries; it is that the glob
cannot swallow a third binary. `scripts/check_repo.py` gains an assertion that
every directory under `bins/` containing a `Cargo.toml` is either a workspace
member or explicitly excluded, so a name that matches no glob fails the build
instead of disappearing.

- Positive: `cargo metadata` lists both; `doctor.rs` runs and passes; clippy is
  clean at `-D warnings` on both.
- Negative: a fixture binary whose name matches neither glob nor exclude makes
  the new check fail, naming the directory. Without this the item closes itself
  and stays closed only until the next binary is added.

**Contract.** `docs/DEVELOPMENT_ENVIRONMENT.md`.

### A2 — A0 is linted against an MSRV it does not build with

**State.** `clippy.toml:1` says `msrv = "1.85.0"`. `model/burn-a0/Cargo.toml:5`
says `rust-version = "1.95"`. `model/burn-a0` is excluded from the root workspace
(`Cargo.toml:4`) and is its own workspace, but it takes the root `clippy.toml`
when linted from the tree. So clippy suppresses every lint whose suggestion needs
anything newer than 1.85 in a crate that requires 1.95 — silently, since a
suppressed lint reports nothing.

**What has to be done.** Give `model/burn-a0/` its own `clippy.toml` declaring
`msrv = "1.95"`, carrying the other two thresholds forward so the style gates do
not change at the same time.

**How it is proven.** A drift this small recurs, so the check is the deliverable:
a script that reads every workspace manifest's `rust-version` and asserts that
the `clippy.toml` governing that workspace declares the same MSRV.

- Positive: after the change, both workspaces agree and the check passes.
- Negative: a fixture pair that disagrees fails the check, naming both files and
  both versions.

**Contract.** `docs/DEVELOPMENT_ENVIRONMENT.md`.

### A3 — the retirement report cannot tell "published" from "adoptable"

**State.** As set out above. `scripts/check_vendor_retirement.py` reports a
candidate from `upstream_observed` alone; `scripts/refresh_vendor_upstream.py`
records that observation from the sparse index. Neither consults what the
dependents of the patched package will accept, and both of the two reported
candidates are refused by their dependents' own requirements.

**What has to be done.** Teach the report the difference between *a newer
upstream release exists* and *a newer upstream release this tree could adopt*.

1. For each patched package, resolve the set of version requirements its
   dependents declare, from the lockfile and the manifests already in the tree.
   No network: these are all facts the repository already holds.
2. Classify a newer release as `adoptable` when at least one dependent's
   requirement admits it and every other dependent either admits it or is itself
   patched here, and as `blocked` otherwise.
3. Record the blocker in `vendor/RETIREMENT.json` as a **checked field** naming
   the dependent and the requirement — not as prose, for the reason Gate 4
   exists.

**How it is proven.**

- Fixtures: a package whose dependents admit the newer release (reported
  `adoptable`); one whose dependents do not (reported `blocked`, with the
  blocking dependent and requirement named); one with no dependents at all; and
  one where the dependents disagree with each other.
- Against the real tree: `macerator` and `netlink-packet-core` both report
  `blocked`, naming `burn-ndarray`/`burn-vision` and
  `netlink-packet-route`/`netdev`/`netlink-proto`/`netwatch` respectively. This
  is the control that the change describes *this* repository and not only its
  fixtures.
- Negative: a recorded `blocked_by` that no longer matches the tree fails the
  checker, the same way every other field in `RETIREMENT.json` is re-derived
  rather than trusted.

**Contract.** `docs/architecture/30-third-party-notices.md` — the bullet
"retirement is not performed; the two candidates are reported, not taken" is
replaced by what is actually true, which is stronger and less flattering: the
candidates were not takeable and the tool could not say so.
`docs/VENDOR_PATCH_POLICY.md` gains the adoptability rule.

**What this does not do.** It does not retire anything. After it, the correct
number of available retirements is reported, and that number is zero.

### A4 — `provenance_bucket_count` is checked against nothing

**State.** `model/burn-a0/src/lib.rs:152` declares it, `:174` defaults it to 64,
`:219` sizes an embedding from it. `26-cognitive-codebook.md` records that it
"remains outside the codebook, by decision and not by omission. Its width is
unchecked against anything." Both halves are true; the second is the problem.
Every other cardinality in A0 — slot types, epistemic states, the router's width
— is the codebook's, and this one is a number in a Rust default that nothing
outside Rust can see and no check compares to anything.

**What has to be done.** Record the exception where the things it is an exception
to are recorded: an explicit `exceptions` section in the codebook artifact naming
`provenance_bucket_count`, its width, and why it is not a code space. Then have
`scripts/check_codebook.py` require that the A0 default matches it.

A recorded exception that is checked is a decision. An unrecorded one is
indistinguishable from an oversight, which is precisely what the contract says
this is not.

**How it is proven.**
- Positive: default and artifact agree, the check passes, and the artifact states
  the reason for the divergence.
- Negative, from both sides: change the Rust default to 65 and the check fails
  naming the field; change the artifact instead and the same check fails from the
  other direction. One-sided checks are how the M001 and README staleness in #20
  survived.

**Contract.** `docs/architecture/26-cognitive-codebook.md`.

### A5 — the codebook version comparison in `forward` is unreachable

**State.** `model/burn-a0/src/lib.rs:236` writes the version into the grid,
`:265` carries it, `:275` reconstructs the codebook from it. The contract records
that the comparison "is unreachable today. One frozen version means no test can
construct a foreign grid from outside the crate. It is a guard for the second
version, not a tested path."

A guard nobody has ever entered is a guard whose behaviour is a claim. The repo's
own rule — a test that has never failed is not evidence that it can — applies to
a branch that has never been taken.

**What has to be done.** Make the mismatch branch reachable from a test without
putting a second version into the public API, which would be inventing a fact to
test a guard: a `cfg(test)`-only constructor for a grid at a declared version.

**How it is proven.**
- Positive control: a grid at the current version passes, so the test is
  responsive to something other than the guard.
- Negative: a grid at a foreign version is refused, asserted on the **error the
  guard produces**, not on the absence of a result — otherwise the test passes
  for any refusal at all.

**Contract.** `docs/architecture/26-cognitive-codebook.md`: that bullet moves
from "unreachable" to "entered by test", and stays listed, because being tested
is not the same as a second version existing.

### A6 — the admission policy does not survive a restart

**State.** `crates/ptr-runtime/src/execution.rs:184` defines `AdmissionPolicy`,
`:493` holds it as a gateway field, `:507` builds an empty one, `:677` lets a
host install one. Nothing journals it.
`29-peer-admission-and-pod-scope.md` records the consequence exactly: "The policy
is in memory. It is not journaled, so it does not survive a restart and a
replayed history does not describe who was admitted when. The audit records the
principal an effect ran under, which is a different question."

This is the largest Tier A item and the only one that touches the runtime. It is
here rather than in Tier C because it needs no decision: Gate 1a already
established what a journaled, replayed, restart-surviving record looks like in
this codebase, and this is the same shape applied to a second fact.

**What has to be done.** Journal admission changes as ledger events — admitted,
scope changed, withdrawn — and rebuild the policy from committed history on open,
the way the execution fence is derived rather than remembered.

**How it is proven.** The property that matters is not "a policy comes back". It
is that a **withdrawal** comes back.

- Positive: a policy installed, journaled and recovered across a restart admits
  exactly the peer and scope it admitted before.
- Negative, and the real test: a peer withdrawn **before** the restart is still
  refused after it. A recovery that replayed only admissions would pass the
  positive test and fail this one.
- Negative: a policy never journaled recovers empty, rather than recovering
  whatever the process last held in memory.
- Control: the refusal after recovery is identical to the refusal of a peer that
  was never admitted, so recovery is not an existence oracle — the same property
  `c9ae32e` established for cross-project Pods.

**Contract.** `docs/architecture/29-peer-admission-and-pod-scope.md` and
`docs/architecture/28-durable-execution-audit.md`.

---

## Tier B — needs the owner's decision before any code

Three. Each would change a property rather than add one, and none is mine to
settle. They are listed so that "open" does not quietly become "forgotten".

**B1 — may `ptr-runtime` depend on `ptr-net`?** Carried from #20 unchanged. With
that edge, a peer identity becomes unforgeable **at the type level** rather than
un-forged in the one composition that currently gets it right. The cost of not
taking it is written into `32-execution-wire.md`. Not blocking anything.

**B2 — should a receipt be signed?** `32-execution-wire.md`: a receipt is
hearsay outside the connection that carried it. Signing needs a key, a statement
of what the signature attests to, and a rotation story. Deliberately not invented
here, because a receipt that looks like evidence and is not is worse than one
that admits it is not.

**B3 — where does the accept loop live?** Both `ptr-cluster` and `ptr-execwire`
provide pieces and neither runs a loop, for the stated reason that a loop in the
wrong place makes scheduling implicit and "when do we give up" is policy. Whether
that policy belongs in a service component or in the embedding host is a
deployment decision.

---

## Tier C — separate gates, each needing its own issue

Not in this PR. Listed with what makes each one a gate rather than a task, so the
next issue can be written from here instead of rediscovered.

**C1 — connect actual semantic payloads to the model.** #15's Gate 3 sentence was
*"Replace learned validity-ID hints with independently enforced validity masks and
a shared versioned cognitive codebook; **connect actual semantic payloads to the
model**."* The first two clauses closed in #21. The third did not, and it has no
box in #20 — it appears only as a bullet under what Gate 3 does not close. This is
the clearest case of a requirement clause surviving a fully-ticked gate, and it is
the largest single piece of open work in the repository.

**C2 — Pod access across a network boundary.** `ALPN_PODWIRE`, `ALPN_MODEL`,
`ALPN_BLOB` and `ALPN_EVENTS` are declared in `crates/ptr-net/src/lib.rs:4-7` and
carry nothing; the workspace's only uses of any ALPN are `ALPN_RAFT` and
`ALPN_EXEC`. Execution crosses a network boundary; Pod access does not.

**C3 — a settlement channel for detached work.** Detached grants are refused over
the wire because there is no channel for the adapter's later report. A second
protocol, not an extension of this one.

**C4 — membership negotiation.** Members record a configuration and recover it;
adding or removing a voter over the wire is not implemented.

**C5 — raft's log floor and the ledger's retention floor.** Independent today.
Related: the at-most-once retention obligation in `28-durable-execution-audit.md`
— a compaction floor that rises past a settled attempt discards the memory of its
key, and a retry after that executes again.

**C6 — session reuse.** Every exchange in both compositions opens a connection.
Correct and wasteful, recorded as missing in both contracts rather than assumed.

---

## Order of work

A1 and A2 first: they are the cheapest, they touch no runtime behaviour, and both
end in a checker that stops the defect recurring — so the rest of the work is done
under gates that are already correct.

A3 next, because it changes what the repository *claims* rather than what it does,
and the current claim is wrong in a way a maintainer could act on.

A4 and A5 together: both are A0, both are one-file changes plus a test, and both
convert a prose statement into a checked one.

A6 last, alone, because it is the only item that touches the runtime and the only
one whose negative test is more important than its positive one.

## The standard for closing anything above

Unchanged from #15 and #20, restated because it is what makes the difference
between this list and a wish:

- implementation, plus a positive **and** a negative test bound to it;
- a control on a responsive observable, so a passing test is evidence of the
  property and not of the test being inert;
- the contract sentence that changes, changed in the same commit;
- exact-commit evidence.

A green CI run is not completion. A demonstrated happy path is not completion.

## What this plan deliberately does not schedule

Legal clearance for distribution; any claim about model quality, latency,
throughput or behaviour under load; Byzantine fault tolerance; obtaining licence
texts for the 117 packages that ship none; and the three Tier B decisions, which
are recorded rather than taken.
