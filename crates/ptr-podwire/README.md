# ptr-podwire — Pod access over authenticated transport

> **Role:** Carries one Pod invocation and its answer between authenticated PTR endpoints on `ALPN_PODWIRE`, composing `ptr-pods`' registry with `ptr-net`'s transport.
> **Maturity:** prototype; which Pod answers which request is proven in `tests/access.rs` where no socket is involved, and `tests/wire.rs` proves that a real connection carries it without either side believing a claim the connection did not make.

<!-- PTR:STATUS:BEGIN -->
## Current implementation status

> **Generated section.** Source of truth: [`component.toml`](component.toml) plus code-derived metrics from `src/`. Run `python3 scripts/update_component_docs.py --write` after editing implementation metadata. Do not hand-edit inside this block.

**Maturity:** `prototype`  
**Last reviewed:** 2026-09-22  
**Code footprint:** 4 Rust source files · 1486 nonblank source lines · 2 integration-test files · 33 test markers (`#[test]`, `#[tokio::test]`)

### Implemented now

- PodHost and PodClient carry one Pod invocation and its answer over ALPN_PODWIRE, composing ptr-pods' registry with ptr-net's transport so neither has to know about the other
- A request frame carries no project field: the project a request resolves in comes from the host's access policy keyed by the authenticated peer, because resolution inside a project is the only thing standing between a request and a Pod's data and a requester that named its project would be choosing which project's Pods it reaches
- A request frame has no field naming its sender and none naming a session, so who is asking is the authenticated connection's answer and a session stays a capability the host holds rather than a string a caller presents
- A request names the endpoint it is for, so a request composed for one host is refused when relayed to another rather than being indistinguishable from one meant for it
- An answer names its own author and is bound to a digest of the exact bytes that arrived; a requester checks the author against the peer the connection authenticated before reading anything else, then the request id, then the digest
- One refusal code covers out of scope, not registered and registered in another project, so a cross-project Pod is refused in exactly the same words as one that does not exist and a scope cannot be read as a directory of what the host serves
- Admission and scope are decided before the registry is touched, so an unadmitted or out-of-scope requester cannot probe for Pods at all
- PodScope matches an exact capability and payload type pair, never a prefix or a wildcard, and it is the same pair PodRegistry::resolve keys on so no scope can be written that resolution would not honour
- One peer holds one entry: a second admit is refused rather than replacing the first, and a withdrawal takes effect on the peer's very next request because the scope is read from the table at every request
- Only Pure and Read Pods are reachable; an effectful Pod is refused with a code that names the action boundary, because this protocol commits no attempt and can report no uncertainty
- PodManifest::protocol_version is checked against the version the requester composed its payload for — the first place anything reads that field, and the only place where the caller and the Pod are not built together
- A Pod's output is verified by the host's verifier before it leaves, and a Pod that failed is a different refusal from an output the host will not vouch for
- The host holds no runtime and no ledger, so a Pod invoked over this wire cannot write to the host's history: a property of the type rather than a promise about the code
- Two outcomes rather than the execution wire's four, and no replay window and no at-most-once key, each absent by derivation rather than omission because nothing reachable here has an effect that could half-apply
- A request/answer frame format with distinct magics per direction, its own digest domain, an explicit layout version, bounded length-prefixed fields, an explicit two-way code table for every enumeration, and a distinct diagnostic code per refusal
- Every request is answered, including one this build cannot parse — its answer is bound to the bytes that arrived and reports no request id rather than inventing one
- An answer this side cannot encode becomes a refusal rather than no reply at all: the frame bound is enforced by trying to encode rather than by pre-checking one field, because a pre-check on the output bytes would miss the output type and every miss leaves a requester waiting on an open connection
- No background loop, timer or retry policy: the caller drives serve_once and request
- The whole composition is feature-gated, so iroh stays out of every other workspace build exactly as ptr-net keeps its own backend out
- A requester dials only what the deployment wrote down: request takes a ptr-net PeerAddress, which has no public constructor, so a bare address does not satisfy the signature and an address handed over by a peer or read out of a payload cannot be dialled

### Missing for the target architecture

- Authentication: the peer key is the one ptr-net reports the connection authenticated, and this crate takes the host's word that it passed that rather than something else
- Signed answers, and therefore any third-party evidence: inside one exchange the connection vouches for an answer's author and outside it nothing does
- A journaled access policy: the scope table is in memory, so it does not survive a restart and nothing records who was admitted to which project when
- Discovery: which address belongs to an id nobody told you is a mechanism choice with its own trust question, recorded as the owner's rather than invented
- An accept loop a deployment would run, including backpressure toward a peer that asks faster than the host can answer
- Session reuse: every request opens a connection, which is correct and wasteful
- Streaming, cancellation and partial answers: one request, one bounded answer
- Any claim about latency, throughput or behaviour under load; the tests establish protocol and isolation properties on localhost

### Next milestones

- Decide whether the access policy and the execution admission policy share one durability story before journaling either
- Decide where the accept loop belongs before adding one, since a loop in the wrong place makes the scheduling implicit again

### Linked experiments

- None recorded.

### Technology evaluations

- None recorded.

### Decision records

- [ADR-0006-typed-isolate-runtime.md](../../research/decisions/ADR-0006-typed-isolate-runtime.md)
- [ADR-0011-semantic-pod-contracts.md](../../research/decisions/ADR-0011-semantic-pod-contracts.md)

### Current automated checks

- two peers send byte-identical frames over two real connections to one host and reach two different projects' Pods, each Pod running exactly once
- an admitted peer reaches its project's Pod over a real ALPN_PODWIRE connection and the peer the host served is the connection's, not a value from the payload
- a Pod in another project is refused with a refusal asserted equal to the one for a capability nobody serves, with the same Pod reachable from inside its own project as the control
- a capability outside a peer's scope is refused with that same code, and a scope matches the exact pair: right capability with the wrong type and right type with the wrong capability are both refused
- a peer the policy never bound is refused and no Pod runs; withdrawing a peer takes effect on its very next request over the wire
- a second entry for one peer is refused and the first entry is still the one in force
- an effectful Pod is refused with the code that names the action boundary, with the same Pod declared Read as the control
- a payload composed for another protocol version is refused, with the version the Pod declares as the control
- a Pod that fails and an output the host will not vouch for are different refusals, each with a passing control
- a request addressed to another host is refused and no Pod runs, with the same request addressed correctly as the control
- a frame this build cannot parse is refused before any Pod, is still answered, and the answer is bound to the bytes that arrived
- an answer naming another endpoint, an answer to another request id and an answer bound to other bytes are each refused, with an honest endpoint answering as the control
- a Pod whose output will not fit in a frame yields a refusal bound to the request rather than no reply, with a Pod whose output does fit as the control
- compile-fail doctest: a bare endpoint address is not an argument to request
- frame codec unit tests: round trip of every outcome and refusal code, a request never read as an answer and never as the execution wire's frame, every prefix refused, no single bit change yielding the original request, trailing bytes, an unknown layout version, an invented length refused by the bound, an oversize frame refused before parsing and on the way out, an unknown outcome or refusal code refused rather than guessed, zero deliberately not a code, invalid UTF-8 refused rather than replaced, one distinct code per refusal, and this wire's digest asserted different from the execution wire's digest of the same bytes

<!-- PTR:STATUS:END -->

## Why this crate exists

`ptr-pods` holds Pod resolution and must not know about transport. `ptr-net` holds
transport and must not know what a Pod is. Something has to know both, and this is
that something — the same shape as `ptr-cluster` and `ptr-execwire`, for the same
reason: a composition crate adds no dependency edge between existing components.

## The sentence it is built around

**The project is policy, not payload.** `PodManifest` says why a Pod belongs to one
project: *a Pod that served every project would make the project boundary advisory,
since resolution is the only thing standing between a request and a Pod's data.*
Read that as a statement about a wire and it decides the format. A requester that
named its project would be choosing which project's Pods it reaches, and resolution
— the only thing in the way — would be doing what the requester asked.

So a request frame has **no project field**. It also has no field naming its sender
and none naming a session, for the reasons `ptr-execwire` gives. The project and the
peer both come from the authenticated connection and the host's policy.

The test that is the claim: two requesters send **byte-identical frames** to one
host over two connections and reach two different projects' Pods.

## Why this is not the execution wire with a different payload

The execution wire asks for an effect. This one asks a Pure or Read Pod a question.
Nothing applies, so nothing can half-apply — and every mechanism that exists for
that is therefore absent here: no committed attempt, no fence, no at-most-once key,
no replay window, and two outcomes instead of four. Each absence is written down in
the place it would have gone, with the reason. An effectful Pod is refused with a
code that names the other protocol.

## What the host cannot do to itself

A `PodHost` holds a registry, an access policy and a verifier. It holds no runtime
and no ledger, so *a Pod invoked over this wire writes nothing to the host's
history* is a fact about the type rather than a promise about the code.

## What a requester may learn

One refusal code covers out of scope, not registered, and registered in another
project — so a cross-project Pod is refused in exactly the same words as one that
does not exist, and a scope cannot be read as a directory. Admission and scope are
decided before the registry is touched, so an unadmitted requester cannot probe for
Pods at all.

## What it is not

It is not a daemon. No background loop, no timer, no retry policy: a caller drives
`serve_once` and `request`.

See [`docs/architecture/33-pod-wire.md`](../../docs/architecture/33-pod-wire.md).
