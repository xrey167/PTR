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

So `admit_peer` takes a `NodeId` and nothing else. There is no principal
parameter and no grant parameter, because those are exactly the two things a
caller must not be able to choose. Both come from `AdmissionPolicy`, a table the
trusted host installs:

```
admit_peer(peer)  →  policy[peer]  →  (principal, grants, ttl)  →  session
```

The boundary is stated rather than disguised: the host passes what its transport
proved — for the iroh backend, the QUIC-authenticated `Connection::remote_id` —
and never a value read out of a payload. Nothing here can check that the host did
so, and a type that pretended to would be decoration.

What *is* enforced is everything after that point. A peer with no entry is not a
peer with fewer rights; it is `UnknownPeer` and gets no session at all. A peer
cannot hold two entries: a second `admit` for the same `NodeId` is refused rather
than replacing the first, because a silent replacement is how a later, weaker
entry widens an earlier one without anyone deciding to.

Grants are produced by a closure rather than stored. A grant owns its verifier
and its executor, so each session needs its own; the closure also makes it
explicit that the host decides what a peer receives on *every* admission.

## Authority is re-derived at use, never remembered

A session admitted from the policy carries the peer it was admitted for, and
every use re-checks that the policy still admits it.

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

- **Nothing here authenticates a peer.** The `NodeId` is taken on the host's
  word. Wiring `ptr-net`'s authenticated `Connection::remote_id` into this API is
  a host's job today; the runtime does not depend on the transport, and making
  the identity unforgeable at the type level would require that it does.
- **No wire protocol.** `ALPN_PODWIRE` and the other ALPNs still carry no
  traffic, and no request format, framing or replay window is defined. A forged
  *receipt* on the wire is therefore still untested, because there is no wire.
- **Pod project membership is declared, not proven.** A Pod says which project it
  serves in its own manifest. Whoever registers it decides that, exactly as
  whoever installs a grant decides its scope.
- **Permission and lifecycle races** between preparation and execution remain
  listed in #20; the existing epoch invalidation covers the local case and has no
  test that drives the race from a second session.
- **The policy is in memory.** It is not journaled, so it does not survive a
  restart and a replayed history does not describe who was admitted when. The
  audit records the principal an effect ran under, which is a different question.
