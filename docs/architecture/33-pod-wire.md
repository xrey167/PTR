# 33 — The Pod wire: the project is policy, not payload

Status: **implemented for the request/answer exchange.** It closes **C3** of issue
#23 — *Pod access across a network boundary* — and nothing beyond it. The last
section says what it does not establish.

Owner: `crates/ptr-podwire`, composing `ptr-pods`, `ptr-verifier` and `ptr-net`.

`ALPN_PODWIRE` has been declared since `ptr-net` declared it. Until now the only
bytes that ever crossed it were `b"ping"` in `crates/ptr-net/tests/iroh.rs` — a
label borrowed to prove a connection authenticates its peer, which is not a
protocol and never claimed to be. This document is the protocol.

## The sentence this is built around

`PodManifest` says, in its own doc comment, why a Pod belongs to one project:

> A Pod that served every project would make the project boundary advisory, since
> resolution is the only thing standing between a request and a Pod's data.

Read that as a statement about a wire and it decides the format. If a request
carried the project, a requester would choose which project's Pods it reaches — and
resolution, the only thing standing in the way, would be doing what the requester
asked. So:

**A request names no project.** The project is looked up from the peer the
connection authenticated, in the same place and for the same reason that the
execution wire looks up a principal.

That is the one thing about this format that is not inherited from
`32-execution-wire.md`. Two more are: a request names no sender (who is asking is
the connection's answer) and no session (a session is a capability the host holds,
not a string a caller presents). The third of those has a history, below.

## The test that is the claim

`two_peers_send_the_same_bytes_and_reach_two_different_projects_pods` sends **one
`PodRequest` value** from two different requesters over two real connections to one
host. The encoded frames are identical — asserted, not assumed — because there is
nothing in the format that could differ. One reaches `alpha`'s summarizer, the
other reaches `beta`'s, and each Pod runs exactly once.

Mutating the host to resolve under a fixed peer rather than the connection's kills
that test and only that test.

## Why this is not `ALPN_EXEC` with a different payload

The execution wire asks for an **effect**. It commits an attempt before the effect,
holds a fence while the outcome is unknown, keys retries on an at-most-once key
recorded in the ledger, and has an outcome that means *it may have applied and I do
not know*.

Pod access asks a Pure or Read Pod a question. Nothing applies, so nothing can
half-apply. Every one of those mechanisms is therefore absent here — and absent by
derivation rather than by omission, each named in the place it would have gone:

| Execution wire | Pod wire | Because |
|---|---|---|
| Four outcomes | Two | Nothing applies, so a refusal always means nothing happened |
| At-most-once key | none | There is no effect for a retry to duplicate |
| Bounded replay window | none | A window stops an *effect* applying twice; a replay here costs the host the work and can do nothing else |
| An attempt committed before dispatch | none | There is nothing to fence, and see below |
| A session per request, revoked after | none | Nothing here issues execution authority |

An effectful Pod inside a requester's own scope is refused with
`RequiresActionBoundary`, which names the other protocol. It is the one refusal that
exists to send a requester somewhere else.

## The host has nothing to write to

A `PodHost` holds a registry, an access policy and a verifier. It holds **no runtime
and no ledger**. "A Pod invoked over this wire writes nothing to the host's history"
is therefore a fact about the type rather than a promise about the code: there is
nothing there to write to.

That is deliberate and it is the reason the in-process path and this one differ.
`run_model_with_pods` promotes a Pod's output into the semantic state of the model
run that asked for it. No such run exists behind a request that arrived over a wire,
and inventing one would let a peer write into a host's semantic store through a
door marked Read.

## What a requester may learn

`RefusalCode::Unavailable` is **one code for three situations**:

1. the capability and payload type are outside this peer's scope;
2. no Pod in this peer's project serves them;
3. a Pod serving them belongs to **another** project.

Telling them apart would answer, from outside a project, whether a given Pod exists
inside it. That is the same oracle `PodUnavailable` already closes in one process,
stated there as *a Pod in another project is unavailable in exactly the same words
as a Pod that does not exist*, and the test here asserts the two refusals are
**equal** rather than merely similar. Folding case 1 in as well keeps the scope from
becoming a directory of what the host serves.

The remaining codes are each a fact about the requester itself or about its own
granted scope, and a requester that could not tell them apart could act on none of
them: how it framed its bytes, which key it addressed, whether its own key is
admitted here at all, whether the Pod it may reach wants the other protocol, whether
it composed for the wrong protocol version, whether the Pod ran and failed as
against ran and was not vouched for, and whether the answer exists and will not fit.

That last one is where the bound lives, and it lives there rather than in a
pre-check on one field. An answer this side cannot encode would otherwise leave the
requester holding an open connection until its deadline — not a wrong reply but *no*
reply, because the host could not phrase what it wanted to say. A pre-check on the
output's bytes would have missed the output's *type*, which is bounded too, and every
miss is one of those hangs. So `serve_once` enforces the bound by trying: if the
answer will not encode, the refusal does. `AnswerTooLarge` is its own code rather
than `Unavailable`, which would be false — something did answer — and rather than
the execution wire's `AppliedWithoutResponse`, which would be worse: there the effect
happened and the bytes are gone, here nothing happened at all.

The order of the checks carries that: admission and scope are decided **before the
registry is touched**, so an unadmitted or out-of-scope requester cannot probe for
Pods at all.

## A scope is an exact pair

`PodScope` holds a project and a set of exact `(capability, payload type)` pairs —
never a prefix, never a wildcard, for the reason `ActionScope` gives in the runtime.
The pair is the pair `PodRegistry::resolve` keys on, so no scope can be expressed
that resolution would not honour, and a scope holding only half of it cannot be
written.

A peer holds one entry. A second `admit` for the same peer is refused rather than
replacing the first, which is `AdmissionPolicy`'s rule kept deliberately so the two
policies cannot disagree about what admitting a peer twice means. Withdrawal takes
effect on the very next request, because the scope is read from the table at every
request rather than remembered.

## `protocol_version` now means something

`PodManifest::protocol_version` has been declared since the type existed and nothing
anywhere read it. In one process that is defensible: the caller and the Pod are
built together. Across a boundary they are not, so a payload composed for one
version of a Pod must not be quietly handed to another. A request carries the
version it was composed for and a mismatch is refused.

Which version the Pod *does* declare is not reported. Finding that out is discovery,
and this wire has none.

## A schema that named its caller

`proto/podwire.proto` declared `PodCall` with a **required `session_id`** — a
caller-supplied identity string, in a request message, in exactly the position this
format refuses to have one. Nothing in the workspace ever sent or received a
`PodCall`; it was a declared shape waiting for somebody to implement against it, and
implementing against it would have put on the wire precisely the thing
`ExecutionSession` exists to make impossible ("No serialized identity string can
construct this handle").

The field is removed and its number and name are reserved. The reservation is held
by a `compile_fail` doctest rather than by the comment beside it, because a comment
cannot notice when somebody reuses the number: restoring `session_id = 1` to the
`.proto` makes that doctest compile, which fails it.

## Executed evidence

`crates/ptr-podwire`: 14 frame unit tests, 10 in `tests/access.rs`, 9 in
`tests/wire.rs` over real authenticated connections. One `compile_fail` doctest in
`crates/ptr-protocol`.

Positive, and each the control for a refusal beside it:

- An admitted peer reaches its project's Pod over a real connection, the output
  comes back typed, and the peer the host saw is the connection's.
- Two peers send identical bytes and reach two different projects' Pods.
- The Pod a cross-project requester could not reach is reachable from inside its own
  project — so the refusal is a boundary and not a broken fixture.
- The same Pod declared `Read` runs where the `Mutation` one was refused.
- The same request composed for the version the Pod declares is answered.
- The same output, verified by a verifier that passes, is returned.
- The same request addressed correctly is answered.
- An honest endpoint's answer is not refused, so the forged-author refusal is about
  the name and not about that endpoint.

Negative, one per route in:

- A peer with no entry is refused and **no Pod runs**.
- A capability outside a peer's scope, a Pod in another project, and a capability
  nobody serves are refused with one code, and the cross-project and does-not-exist
  refusals are asserted **equal**.
- A scope matches the exact pair: right capability with the wrong type, and right
  type with the wrong capability, are both refused.
- An effectful Pod is sent to the action boundary rather than run.
- A payload composed for another protocol version is refused.
- A Pod that fails and an output the host will not vouch for are different refusals.
- Withdrawing a peer takes effect on its very next request over the wire, and only
  the admitted request reached the Pod.
- A second entry for one peer is refused, and the first entry is still the one in
  force.
- A request addressed to another host is refused and no Pod runs.
- A frame this build cannot parse is refused before any Pod, is still answered, and
  the answer is bound to the bytes that arrived with a request id of 0 rather than
  an invented one.
- An answer naming another endpoint, an answer to another request id, and an answer
  bound to other bytes are each refused by the requester.
- A Pod whose output will not fit in a frame is reported as `AnswerTooLarge` rather
  than leaving the requester waiting, the answer is still bound to the request, the
  host records the framing refusal behind it, and a Pod whose output does fit is the
  control.

Frame codec: round trip of every outcome and every refusal code; a request never
read as an answer and an answer never as a request, including against the execution
wire's magic; every prefix refused; no single bit change yielding the original
request; trailing bytes; an unknown layout version; an invented length refused by
the bound; an oversize frame refused before parsing and refused on the way out in
both directions; an unknown outcome or refusal code refused rather than guessed;
zero deliberately not a code; invalid UTF-8 refused rather than replaced; one
distinct diagnostic code per refusal; and this wire's request digest asserted
different from the execution wire's digest of the same bytes.

Every check in the access decision was mutation-checked: dropping the scope check,
adding a resolution fallback that forgets the project, dropping the Pure/Read check,
the protocol-version check, the output verification, or reporting an unadmitted peer
as unavailable each fail exactly their own tests. So were the four endpoint checks
and the peer the decision runs under.

## What this does not close

- **Nothing here authenticates a peer.** The key is the one `ptr-net` reports the
  QUIC connection authenticated, and this crate takes the host's word that it passed
  that rather than something else. The same boundary `29-peer-admission-and-pod-scope.md`
  states, and the same open decision behind it (D1 in issue #23): a `ptr-runtime`
  dependency on `ptr-net` would make the identity unforgeable at the type level.
- **An answer is not signed.** Inside one exchange the connection vouches for its
  author; once the bytes are stored or forwarded, nothing does. This crate offers no
  way to verify a stored answer rather than producing an artifact that looks like
  evidence and is not — D2's question, with the same answer for now.
- **Pod project membership is still declared, not proven.** A Pod says which project
  it serves in its own manifest, and whoever registers it decides that.
- **The access policy is in memory.** It is not journaled, so it does not survive a
  restart, and nothing anywhere records who was admitted to which project when. This
  is D5's question for `AdmissionPolicy` and it is the same question here — with one
  difference worth stating: a `PodScope` holds only data, so unlike an
  `ExecutionGrant` it *could* be rebuilt from history. That makes it a smaller
  problem, not a closed one, and taking it separately would leave two policies with
  two durability stories.
- **No discovery.** Which address belongs to an id nobody told you is a mechanism
  choice with its own trust question, and it is the owner's.

  **Address authority is no longer missing.** This bullet said *no discovery and no
  address authority*, and that a wrong address gets a refusal only because the
  request names the endpoint it is for — which stopped being true one commit later.
  `request` takes a `ptr_net::PeerAddress`, which only a `PeerBook` the deployment
  installed can produce, and a wrong address fails on the authenticated key before
  the host is asked anything. See `34-address-authority.md`.
- **No accept loop, and no backpressure.** `serve_once` answers one connection. A
  peer that asks faster than the host can answer is the deployment's problem, and
  where the loop belongs is D4.
- **Session reuse.** Every request opens a connection, which is correct and wasteful
  — the same sentence `31-cluster-integrity.md` and `32-execution-wire.md` already
  carry.
- **Nothing about latency, throughput or behaviour under load.** The tests establish
  protocol and isolation properties on localhost.
- **The other three ALPNs still carry nothing.** `ALPN_MODEL`, `ALPN_BLOB` and
  `ALPN_EVENTS` remain declared and unspoken. This document closes one of the four
  C3 named, and says so rather than letting "Pod access" stand in for all of them.
