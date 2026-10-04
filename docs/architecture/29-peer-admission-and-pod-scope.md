# P0.9 — Peer admission and project-scoped Pods

Baseline: `d062e1f70a2408da972af1e39e12afe8a11cbec2`.
Owner: `ptr-runtime::execution`, `ptr-pods`.

The second half of issue #20's *Execution/network trust boundaries* gate. The
first half — durable execution audit, fencing, reconciliation and at-most-once
effects — is `28-durable-execution-audit.md`.

Two things are closed here: a peer that a transport authenticated is mapped to a
session by **host policy** rather than by anything in the request, and a Pod is
resolvable only inside the project it belongs to.

## Admitting an identity is not establishing one

This layer does not authenticate. It cannot: authentication happens where the
bytes arrive, and by the time a request reaches the runtime the only honest
question left is *what is this identity allowed to do*.

So `admit_peer` takes an `AuthenticatedPeer` and nothing else. There is no
principal parameter and no grant parameter, because those are exactly the two
things a caller must not be able to choose. The principal comes from the peer's
**membership**, which is committed history, and the grants from the host's **grant
set** of the name that membership records:

```
admit_member(peer, principal, grant_set)   →  PeerAdmitted, committed
admit_peer(authenticated)  →  members[peer]  →  (principal, grant_set name)
                           →  grant_sets[name]  →  (grants, ttl)  →  session
```

The identity itself is the transport's: `AuthenticatedPeer` has no public
constructor and only `ptr-net`'s transport builds one, from the QUIC-authenticated
`Connection::remote_id` (D1 in issue #23; `32-execution-wire.md`). A value read out
of a payload does not compile where admission needs one.

What is enforced after that point: a peer that is not a member is not a peer with
fewer rights; it is `UnknownPeer` and gets no session at all. A peer cannot be
admitted twice: a second admission is refused (`DuplicatePeerEntry`) rather than
replacing the first, because a silent replacement is how a later, weaker entry
widens an earlier one without anyone deciding to — changing a peer's principal or
grant set is a withdrawal and an admission, two records a history can show. A grant
set cannot be defined twice for the same reason (`DuplicateGrantSet`).

Grants are produced by a closure rather than stored. A grant owns its verifier
and its executor, so each session needs its own; the closure also makes it
explicit that the host decides what a peer receives on *every* admission.

## Membership is remembered; authority is re-derived at use (D5 in issue #23)

This heading used to read *authority is re-derived at use, never remembered*, and
"rebuild the policy from committed history" was the opposite verb. The two are
reconciled by splitting what the old table held into the half that is a fact and
the half that is behaviour:

- **Membership is history.** `PeerAdmitted { peer, principal, grant_set }` and
  `PeerWithdrawn { peer }` are ledger records, committed before they take effect
  and replayed in order on every open, so a peer withdrawn before a restart is
  still refused after it
  (`a_peer_withdrawn_before_a_restart_is_still_refused_after_it`, with the peer
  that was not withdrawn executing as the control) and admit → withdraw → admit
  replays to the last admission
  (`admit_withdraw_admit_replays_to_the_last_admission`). One rule refuses a bad
  record live and in replay — a field that is not a bounded identifier
  (`MAX_MEMBER_FIELD_BYTES`), a second admission, a withdrawal of a non-member — so
  a history cannot hold an admission the runtime would not have written
  (`a_history_that_admits_a_peer_twice_is_refused_on_replay`).
- **Authority is not.** What a grant set permits is a verifier and an executor,
  which cannot be replayed out of a log. The host installs `GrantSets` at every
  start; until it does, a member is still a member and reaches nothing
  (`UnknownGrantSet`,
  `after_a_restart_a_member_reaches_nothing_until_the_host_supplies_its_grant_set`).
  `install_grant_sets` replaces the whole table and governs authority only; it
  cannot admit or withdraw anybody, so there is no second source of truth about
  membership and no precedence to state.

A session carries the peer **and the commit index of the admission** it was issued
under, and every use re-checks that this same admission still stands and that its
grant set is still defined. A withdrawal ends the session at once — not when a TTL
happens to run out — and a later re-admission is a different record that does not
revive it (`a_re_admission_issues_new_sessions_and_revives_none`).

Admission is not the only thing re-derived at use. A session also holds the
*grants* admission returned, and re-checking only "is the admission still standing"
left a replacement that narrowed a grant set running the live session under the
wider one until its TTL expired. So `install_grant_sets` re-derives every peer
session's grants from the new table.

Two choices inside that are worth stating. A session whose grant set the new table
does not define is left in place rather than dropped, so `session()` can still
refuse it by name with `UnknownGrantSet` instead of a bare `SessionClosed`. And the
expiry is deliberately not refreshed: a new table may narrow what a session can do,
but it must not extend how long it lasts.

A membership record is a commit, and **every commit retires every permit prepared
before it**: a permit is authorized against the world as it was. So a withdrawal of
one peer makes another peer's in-flight permit stale too; that peer re-prepares and
proceeds
(`withdrawing_one_peer_in_the_window_refuses_its_permit_and_leaves_the_other_working`).

Membership crosses compaction: it is projected (`peer:<id>`, one key per peer,
`ptr_state::member_entry` or `withdrawn`) into the lifecycle section, which is
`PTRLC003` inside a `PTRCS005` snapshot. No earlier layout could carry it, so a
`PTRCS004` or `PTRCS003` snapshot holding such a key is refused by version
(`membership_keys_are_refused_by_version_before_ptrlc003_and_held_to_their_projection`),
and a restored runtime refuses a peer withdrawn before the floor
(`membership_survives_compaction_withdrawals_included`).

The records are codec version 2 (`docs/COMPONENT_CONTRACTS.md`): a replicated group
carries them only after every voter is reported to read them
(`31-cluster-integrity.md`).

**Pod scope follows the same design, journaled whole.** The project scope a Pod
peer reaches (`PodScope`, below and in `33-pod-wire.md`) was a second policy with a
second durability story: execution membership survived a restart and Pod access
did not. It is now history too — `PodScopeAdmitted { peer, project, entries }` and
`PodScopeWithdrawn { peer }`, ledger tags 31 and 32 at codec version 2, committed by
`PtrRuntime::admit_pod_scope` and `withdraw_pod_scope`, refused live and in replay by
one rule, replayed in order on every open and projected as `pod-scope:<id>` into
the same `PTRLC003` lifecycle section. It differs from membership in exactly the way
its content does: a scope is data rather than behaviour, so the record carries all
of it and there is no host-supplied half to resolve by name at use. A Pod scope
withdrawn before a restart stays withdrawn after it
(`a_pod_scope_withdrawn_before_a_restart_is_still_withdrawn_after_it`), and the two
families are withdrawn independently — withdrawing a Pod scope leaves the same
peer's execution membership standing, which that test checks as part of its control.

This is the same rule `27-neural-state-admission.md` applies to cached state, and
it is here for the same reason: a session that was admissible when it was issued
says nothing about whether its peer is admissible now, and revocation is
precisely the event that arrives afterwards. Withdrawing a peer therefore takes
effect at once — not when a TTL happens to run out, and not only for the next
admission. A permit issued a moment earlier is refused with `PeerNotAdmitted`.

Replacing the whole policy behaves the same way by construction, with no
separate bookkeeping: sessions are not a second record of who is admitted, so
there is nothing to keep in step.

A session the host registered directly with `register_execution_session` carries
no peer and is unaffected. That is deliberate: the policy governs peers, and the
trusted host is not one.

## A Pod belongs to one project

`PodManifest` now names a project, and `PodRegistry` is keyed by project **and**
id.

The project is part of the key rather than a filter applied afterwards, which
buys two things. Two projects may each register an `echo` Pod without one
silently replacing the other — under an id-only key the second registration would
have shadowed the first, and one project's requests would have landed in the
other project's Pod. And no lookup path can forget the check, because there is no
lookup that takes an id alone.

`run_model_with_pods` and `run_resumable_with_pods` take the project the request
belongs to and resolve within it.

**A Pod in another project is unavailable in exactly the same words as a Pod that
does not exist.** `PodUnavailable` names the capability and input type and
nothing else, and a test asserts the two refusals are equal. A distinguishable
refusal would answer, from outside a project, whether a given Pod exists inside
it — which is the kind of oracle that makes an isolation boundary porous without
ever breaking it.

## Executed evidence

8 integration tests in `crates/ptr-runtime/tests/peer_admission.rs`, 3 in
`crates/ptr-pods/tests/registry.rs`, 1 in `crates/ptr-runtime/tests/pod_loop.rs`.

Positive:

- An admitted peer executes its granted action, and the principal in the
  resulting audit record is the policy's.
- Each of two projects resolves its own `echo` Pod, and re-registering within one
  project still replaces as before.
- The same Pod request inside the Pod's own project succeeds, so the cross-project
  refusal below is a boundary and not a broken fixture.

Negative, one per route back in:

- A peer the policy never bound is refused by `admit_peer` and by
  `withdraw_peer`.
- An admitted peer reaches nothing outside its grant set: a different operation
  and a different project are both `ScopeDenied`.
- Withdrawing a peer refuses a permit issued before the withdrawal, refuses new
  preparations, and refuses re-admission — all without the TTL moving.
- Replacing the policy with an empty one kills a live session; re-admitting the
  peer does not revive the old session.
- A host-registered session keeps working under an empty policy.
- A second entry for one peer is refused; a malformed peer id or principal is
  refused.
- A Pod registered for one project is invisible to another through `resolve` and
  through `get`, and the runtime-level refusal is **equal** to the refusal for a
  Pod that does not exist.

## What this does not close

- **Nothing here authenticates a peer, and nothing here can be handed one that
  was not authenticated.** This document's title still says *admission* rather
  than authentication, because the proof is the transport's. But `admit_peer`
  takes an `AuthenticatedPeer`, constructible only by `ptr-net`'s transport from
  the key a QUIC connection verified, so no host can pass an unproven identity:
  it does not compile (D1 in issue #23, decided yes; `32-execution-wire.md`). The
  policy table itself is still keyed by `NodeId`, which is right — the host names
  the peers it is willing to admit; what it cannot do is claim one arrived.
- **The wire protocol is no longer missing, and it is three protocols rather than
  five.** `ALPN_EXEC` carries execution requests with a defined request format, a
  receipt bound to the bytes that arrived and a bounded replay window
  (`32-execution-wire.md`); `ALPN_RAFT` carries raft traffic
  (`31-cluster-integrity.md`); and `ALPN_PODWIRE` now carries Pod access
  (`33-pod-wire.md`), which is where the project scope below reaches across a
  network boundary — a requester cannot name a project, because the format has
  nowhere to put one, so the project comes from the same kind of host policy this
  document describes for principals. `ALPN_MODEL`, `ALPN_BLOB` and `ALPN_EVENTS`
  still carry nothing.
- **Pod project membership is declared, not proven.** A Pod says which project it
  serves in its own manifest. Whoever registers it decides that, exactly as
  whoever installs a grant decides its scope.
- **Permission and lifecycle races** between preparation and execution are driven
  by six tests in `crates/ptr-runtime/tests/execution_races.rs` (`ea08e0b`),
  including from a second session —
  `one_session_s_successful_effect_refuses_another_s_in_flight_permit` and
  `a_host_session_and_an_admitted_peer_do_not_share_a_fate`. This bullet previously
  said there was no such test, which stopped being true in the same pull request
  that wrote it.

  The narrower thing that *is* worth stating: within one process a second session
  can only act through the host, because a session holds no authority to mutate the
  runtime, and both dispatch paths take `&mut self` — so the window those tests
  drive is preparation-to-consumption, not an interleaving. A genuinely concurrent
  race needs two runtimes, which is `32-execution-wire.md`.
- **Grant sets are in memory, by design.** Membership is journaled (D5); what a
  grant set permits is behaviour the host supplies at every start, and a start that
  omits a set refuses its members by name rather than granting by default. A
  history therefore says who was admitted when and under which grant set name,
  and not what that name permitted at the time — which is the half that cannot be
  replayed.
