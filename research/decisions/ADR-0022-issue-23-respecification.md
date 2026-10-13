# ADR-0022: Issue #23 re-specified against the integrated tree

- Status: accepted (owner delegation, 2026-10-13)
- Supersedes the draft pull requests #101–#121 of the issue #23 stack (written on 2026-10-04 against
  `83d61d32`), which no longer apply to the tree that PR #132 integrated
- Related: `docs/OPEN_ITEMS_PLAN_20260921.md` (Tier B and Tier C), `docs/RECOMMENDATIONS_20260924.md`
  (rows O4 and O6), ADR-0021 (the number the draft close-out PR #120 also used)

## Context

Issue #23 carries two kinds of items:

- **Owner decisions D1–D6.** These were B1–B5 in the open-items plan, plus B2b.
- **Separate gates.** These are C3–C9, G1 (connect semantic payloads to the model) and G2 (durable cross-node fencing).

A stack of 13 draft pull requests implemented most of them on 2026-10-04. While they waited, waves 12–14 landed through PR #132, and that changed the ground under the stack:

- **Ledger event tags 27–31 are taken.** The tags are `ExpertStructuralCommitted`, `ExpertFenceCommitted`, `KvTaintCommitted`, `InputInvalidated` and `ExpertAuthorityChanged` (`crates/ptr-ledger/src/lib.rs`, encoder around lines 1041–1150). The stack used 27–32 for `PeerAdmitted`, `PeerWithdrawn`, a codec ceiling, `DetachedSettlerBound`, `PodScopeAdmitted` and `PodScopeWithdrawn`.
- **Snapshot and lifecycle magics `PTRCS004`–`PTRCS009` and `PTRLC004`–`PTRLC007` are taken** (`crates/ptr-runtime/src/compacted.rs`). The stack defined `PTRCS005` and `PTRLC003` as its membership layout.
- **The raft state magic `PTRRST03` is taken** by the raft storage rework (`crates/ptr-ledger/src/raft_storage.rs`). The stack used it for "adds the codec ceiling".
- **Split-Pod activation chunks are already signed with ed25519** (`ptr-podwire/src/chunk.rs`, E007). The stack's receipt-signing design (D2) predates that and does not reuse it.
- **The fast-memory lock-order fix of #113 landed separately** as f62814a0 (PR #129).

The drafts cannot be rebased mechanically. The read-only audit found 4 to 38 conflicting files per branch. This ADR records, for every item, what happens now. Items that are not built now get a new specification against the current tree.

## Decisions

| Item | Decision | What happens |
|---|---|---|
| G2 durable cross-node fencing | **Closed.** The fencing token is checked before each write; it is not atomic with the write, as `docs/architecture/31-cluster-integrity.md` states. `raft_fence.rs` runs in CI (`ci.yml` ledger-raft-rs, `Makefile` `ci-ledger-raft-rs`). | Docs corrected in this change (`docs/PROJECT_MAP.md`). |
| C3 Pod access across a network | **Closed** by `ptr-podwire` (`ALPN_PODWIRE`). | None. |
| C5 membership changes | **Closed** for one voter at a time (`propose_membership`). | None. |
| C7 raft log floor vs ledger retention floor | **Build now.** Raft never compacts or installs a snapshot above the floor the authoritative ledger still has to retain; violations are typed errors. | Re-implemented on the current `raft_storage.rs` in this change (from #110, commit 0a84514c, redone rather than rebased). |
| Unused ALPNs | **Build now.** `ALPN_MODEL`, `ALPN_BLOB` and `ALPN_EVENTS` name protocols that do not exist. | Removed in this change (from #110, commit 7345cf4c), after checking every use. |
| Combined `failpoints` + `raft-engine-backend` build | **Build now.** The combination still failed with twelve E0433 errors. | Fixed in this change (from #103); CI checks the combination. |
| D3 (B2b) authenticate the codebook artifact | **Decided: no signature now.** The codebook is a repository artifact checked by `scripts/check_codebook.py` and CI. A signature adds a key and a rotation story without a deployment that receives codebooks from elsewhere. | Revisit when a codebook is distributed outside the repository. |
| C8 discovery | **Decided: no discovery.** Peers are configured through `PeerBook`, the address-authority half of C8. | None. |
| C6 independent snapshot-anchor retention | **Accepted boundary.** `take_installed_snapshot_matching` exists; the host supplies the independent anchor. | Stays a deployment obligation, stated in `docs/architecture/31-cluster-integrity.md`. |
| D1 peer identity unforgeable at the type level | **Decided: yes, deferred until the first multi-host deployment.** No binary deploys the network crates; `IrohTransport` binds only loopback. | New spec S1 below. |
| D2 signed receipts and Pod answers | **Decided: yes, after D1, reusing one signing abstraction.** | New spec S2 below. |
| D4 where the accept loop lives | **Decided: in the embedding host (`ptrd`), driven by a typed serve policy that `ptr-net` provides.** | New spec S3 below; deferred with D1. |
| D5 journaling the admission policy | **Decided: history governs membership and the host still supplies authority.** This is the second design of B4. | New spec S4 below; deferred with D1. |
| D6 authenticated reconciliation | **Decided: a capability that never crosses the wire.** | New spec S5 below; may be built before D1 because it is local. |
| C4 settlement channel for detached work | **Deferred, re-specified.** It depends on D6 and on a free ledger tag. | New spec S6 below. |
| Pod scope journaling (#121) | **Deferred, re-specified.** It depends on D5. | New spec S7 below. |
| G1 / C9 connect A0 to the runtime (O6) | **Decided: not now.** O6 made this conditional on a study showing A0's mechanisms are worth connecting. The A0 studies so far are INCONCLUSIVE/NO-GO (M001-v4, M002-v4; `check_research_gates.py`). | New spec S8 below; revisit when an A0 study passes. |
| D-stack close-out docs (#120) | **Dropped.** They record the stack as merged and reuse ADR-0021. | This ADR replaces them. |

## New specifications

Every spec below binds to the integrated tree.

- **New ledger event tags start at 32.** Each new tag needs an explicit unknown-tag behaviour. Because the ledger codec is also the raft proposal payload, it also needs a codec version and a downgrade policy, as `docs/COMPONENT_CONTRACTS.md` requires for every cross-process contract.
- **New snapshot and lifecycle layouts take the next free magics,** `PTRCS010` and `PTRLC008` and onward.
- **A new raft state layout takes `PTRRST04`.**

**S1: authenticated peer identity (D1).**
- `ptr-net` exports `AuthenticatedPeer` with no public constructor. Only the transport's completed handshake produces one, plus a `test-support` feature for tests.
- `PtrRuntime::admit_peer` takes `&AuthenticatedPeer` instead of `&NodeId`.
- `ptr-runtime` gains the `ptr-net` dependency that B1 asked about.
- Acceptance:
  - a compile-fail doctest shows an `AuthenticatedPeer` cannot be built from a `NodeId`;
  - every existing admission test passes through a handshake;
  - E007 and R011 harness behaviour is unchanged (their frozen mutations still apply).

**S2: signed receipts and Pod answers (D2).**
- One `SigningDomain` abstraction in `ptr-net` covers both new signatures and the existing split-Pod chunk signature, each with a distinct domain string.
- Execution receipts (`ptr-execwire`) and Pod answers (`ptr-podwire`) carry an ed25519 signature by the endpoint key over a canonical encoding.
- Acceptance:
  - a receipt replayed under another domain or by another key is refused;
  - the existing chunk signatures verify unchanged.
- Key rotation: one key per node incarnation, rotation by re-admission (S4).

**S3: accept loop (D4).**
- `ptr-net` provides `ServePolicy`: concurrency bound, idle timeout and give-up rule.
- `ptrd` runs one accept loop per configured ALPN under a `[wire]` config section, off by default.
- Acceptance:
  - with `[wire]` absent, `ptrd` binds no wire;
  - a bounded test shows the loop gives up by policy, not by panic;
  - `ptr-config` refuses unknown keys in `[wire]`.

**S4: journaled membership (D5).**
- Two ledger events, `PeerAdmitted` (tag 32) and `PeerWithdrawn` (tag 33), carry principal, peer key, TTL and a grant-set name. They never carry grants, so authority stays host-supplied.
- `docs/architecture/29-peer-admission-and-pod-scope.md`'s heading "Authority is re-derived at use" stays true: history says who is a member, and the host's grant table says what a member may do.
- A replayed admit→withdraw→admit means admitted.
- Acceptance: a peer withdrawn before a restart is still refused after it. This is the primary negative test from B4.

**S5: reconciliation authority (D6).**
- `reconcile_effect` takes a `ReconciliationAuthority` capability that only the host can construct, through `take_reconciliation_authority` (once per runtime instance).
- It has no wire encoding.
- Acceptance:
  - a compile-fail doctest shows no wire type can carry it;
  - existing reconcile tests pass with the capability.

**S6: settlement for detached work (C4).** After S5.
- A detached grant binds its settler with `DetachedSettlerBound` (tag 34).
- Settlement arrives over the execution wire from the bound peer only, and only through `settle_detached_from`, which requires the S5 capability on the host side.

**S7: Pod scope (#121).** After S4.
- `PodScopeAdmitted` and `PodScopeWithdrawn` take tags 35 and 36.
- A history-derived `PodAccessPolicy` follows the same membership-versus-authority split as S4.

**S8: A0 bridge (G1, C9).** Only after an A0 study passes its gates.
- A small adapter crate inside the `model/burn-a0` workspace implements `ptr-model-api`'s `InferenceBackend`, as O6 recommended.
- The root workspace gains no Burn dependency.
- The checkpoint binding of C9 (`seal_checkpoint`) comes with it.

## Consequences

- The 13 draft pull requests close with a pointer to this ADR. The branches stay on GitHub as history.
- Items S1–S8 are tracked here and in `docs/PROJECT_MAP.md` §15 instead of in the drafts.
- Building S1–S4 and S6–S7 is tied to the first multi-host deployment. Before that, no binary exercises them. S5 is local and may be built first.
