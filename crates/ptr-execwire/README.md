# ptr-execwire — Execution over authenticated transport

> **Role:** Carries one execution request and its receipt between authenticated PTR endpoints on `ALPN_EXEC`, composing `ptr-runtime`'s execution authority with `ptr-net`'s transport.
> **Maturity:** prototype; what the runtime permits is proven in `ptr-runtime` where no socket is involved, and this crate proves that a real connection carries it without either side believing a claim the connection did not make.

<!-- PTR:STATUS:BEGIN -->
## Current implementation status

> **Generated section.** Source of truth: [`component.toml`](component.toml) plus code-derived metrics from `src/`. Run `python3 scripts/update_component_docs.py --write` after editing implementation metadata. Do not hand-edit inside this block.

**Maturity:** `prototype`  
**Last reviewed:** 2026-09-22  
**Code footprint:** 3 Rust source files · 1447 nonblank source lines · 1 integration-test files · 27 test markers (`#[test]`, `#[tokio::test]`)

### Implemented now

- ExecutionHost and ExecutionClient carry one execution request and its receipt over ALPN_EXEC, composing ptr-runtime's execution authority with ptr-net's transport so neither of them has to know about the other
- A request frame has no field naming its sender, so who is asking is the authenticated connection's answer and the principal an action is audited under is the admission policy's — the shape of the format rather than a check performed on it
- A request names the endpoint it is for, so a request that was legitimate at one runtime is refused when relayed to another rather than being indistinguishable from one composed for it
- A receipt names its own author and is bound to a digest of the exact bytes that arrived; a requester checks the author against the peer the connection authenticated before reading anything else, then the request id, then the digest
- Four outcomes rather than two: applied, applied without a response, uncertain, refused — because a wire that reported an unknown outcome as a refusal would invite a retry of an effect that had already applied and throw the runtime's fence away at the last step
- The mapping from runtime refusals onto wire outcomes is an exhaustive match, so a refusal a later build adds breaks the build rather than being reported as nothing having happened
- Every refusal of authority crosses the wire as one code, so a requester cannot tell a peer it never admitted from an action outside its grant; the full reason stays on the host for an operator
- A bounded per-peer replay window refuses a frame sent twice before the runtime is asked anything, spent only after the peer is admitted so an admitted peer's request is treated the same way whatever the runtime then decides
- The window and the at-most-once key are separate and neither stands in for the other: retrying an intent means a new request id with the same key, which is the only combination that says this one again rather than do it once more
- One session per request, revoked when the request is answered: caching it would keep the grants admission returned, so a policy later replaced with narrower grants would not narrow anything until the cache expired
- Every request is answered, including a request this build cannot parse — its receipt is bound to the bytes that arrived, so its sender can still check it against what it sent
- A request/receipt frame format with distinct magics per direction, an explicit layout version, bounded length-prefixed fields, explicit two-way code tables for every enumeration, and a distinct diagnostic code per refusal
- No background loop, timer or retry policy: the caller drives serve_once and request, which keeps scheduling decisions where they can be made deliberately and keeps the tests free of sleeps
- The whole composition is feature-gated, so iroh stays out of every other workspace build exactly as ptr-net keeps its own backend out
- A requester dials only what the deployment wrote down: request takes a ptr-net PeerAddress, which has no public constructor, so a bare address does not satisfy the signature and an address handed over by a peer or read out of a payload cannot be dialled
- No second check that a request names the peer being dialled, deliberately: it would buy no property since the host checks the name it was sent, and it would leave the host's own check reachable only from a raw transport

### Missing for the target architecture

- Signed receipts, and therefore any third-party evidence: inside one exchange the connection vouches for a receipt's author and outside it nothing does, so this crate offers no way to verify one rather than producing an artifact that looks like evidence and is not
- A peer identity that is unforgeable at the type level, constructible only from an authenticated connection the way VerifiedDispatch is constructible only inside the runtime; that needs a ptr-runtime dependency on ptr-net and is recorded as an open decision
- Detached work over the wire: a detached grant is audited by an attempt that stays unsettled until the adapter reports back, and there is no channel here for that report, so it is refused before the attempt is committed
- Reconciliation: an uncertain outcome is reported and then belongs to the host's operator, because a remote peer must not be able to declare what happened to an effect
- An accept loop a deployment would run, including backpressure toward a peer that asks faster than the runtime can answer
- Session reuse: every request opens a connection, which is correct and wasteful
- Discovery: which address belongs to an id nobody told you is a mechanism choice with its own trust question, recorded as the owner's rather than invented
- Any claim about latency, throughput or behaviour under load; the tests establish protocol and identity properties on localhost

### Next milestones

- Decide whether a receipt should be signed, and if so what it attests to, before anything treats one as evidence
- Decide where the accept loop belongs before adding one, since a loop in the wrong place makes the scheduling implicit again

### Linked experiments

- None recorded.

### Technology evaluations

- None recorded.

### Decision records

- [ADR-0001-rust-runtime.md](../../research/decisions/ADR-0001-rust-runtime.md)

### Current automated checks

- an admitted peer's action is executed over a real ALPN_EXEC connection and the audited principal is the policy's, not a name written into the payload the peer controls
- a receipt naming another runtime is refused, with the same endpoint answering honestly as the control so the refusal is about the name and not about that endpoint
- a receipt answering another request id, and one bound to other bytes, are each refused
- a request addressed to another runtime is refused and writes no ledger record, with the same request addressed correctly as the control
- a peer that was never admitted is refused identically to an admitted peer asking outside its grant, on the first request and on a replay
- the same frame sent twice is refused by the window and the effect applies once
- a retry under a new request id with the same at-most-once key is answered from the record rather than executed again
- a damaged frame arriving on a real connection is refused before it reaches the runtime, writes no record, and is still answered with a receipt bound to the bytes that arrived
- a detached grant is refused over the wire, before the adapter is reached, and leaves the runtime unfenced
- an adapter that cannot say whether it applied yields an uncertain outcome rather than a refusal, the fence stands, and the next request is refused rather than reported as a second uncertainty
- withdrawing a peer takes effect on its very next request over the wire
- two runtimes audit the same peer's requests independently: an at-most-once key spent at one says nothing at the other
- compile-fail doctest: a bare endpoint address is not an argument to request, verified against a positive control so it fails on the type rather than on a path
- frame codec unit tests: round trip of every outcome, a request never read as a receipt and vice versa, every prefix refused, no single bit change yielding the original request, trailing bytes, an unknown layout version, an invented length refused by the bound, an oversize frame refused before parsing, an unknown effect/outcome/refusal code refused rather than guessed, invalid UTF-8 refused rather than replaced, and one distinct code per refusal

<!-- PTR:STATUS:END -->

## Why this crate exists

`ptr-runtime` holds execution authority and must not know about transport. `ptr-net`
holds transport and must not know what an action is. Something has to know both, and
this is that something — so that neither of them does.

## The sentence it is built around

**Authenticating a peer is not authorizing an action.** A connection proves which key
is at the other end. It does not say what that key may ask for, who it counts as
inside this system, or whether anybody granted the action it is asking for. Those are
three questions, and a protocol that conflates any two of them has a hole in it.

So a request frame has **no field naming its sender**: there is nothing to forge
because there is nothing to write. The principal an action is audited under comes from
the host's admission policy, re-derived on every request. What a request *does* name
is the endpoint it is for, so one captured at one runtime cannot be replayed at
another and pass as meant for it.

## Two fields against repetition, and why both

A `request_id` is a nonce in a bounded per-peer window: it refuses a frame sent twice,
cheaply, and it survives neither a restart nor the window's own bound. An
**at-most-once key** is the runtime's, recorded in the ledger, and it is what makes a
retry apply once. So retrying an intent means a *new* `request_id` with the *same*
key — the only combination that says "this one again" rather than "do it once more".

## What a receipt is not

A receipt names its author and is bound to a digest of the bytes that arrived, and a
requester checks the author against the peer the connection authenticated before
reading anything else. Inside that exchange the connection vouches for it. Once the
bytes are stored or forwarded, nothing does — anyone can write them. There is
deliberately no way to verify a receipt out of band, because the alternative is an
artifact that looks like evidence and is not.

## What it is not

It is not a daemon. No background loop, no timer, no retry policy: a caller drives
`serve_once` and `request`. That keeps "when do we give up" and "how often do we
accept" somewhere they can be decided deliberately, and it keeps the tests free of
sleeps.

See [`docs/architecture/32-execution-wire.md`](../../docs/architecture/32-execution-wire.md).
