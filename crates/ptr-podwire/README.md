# ptr-podwire — Pod access over authenticated transport

> **Role:** Carries one Pod invocation and its answer between authenticated PTR endpoints on `ALPN_PODWIRE`, composing `ptr-pods`' registry with `ptr-net`'s transport.
> **Maturity:** prototype; which Pod answers which request is proven in `tests/access.rs` where no socket is involved, and `tests/wire.rs` proves that a real connection carries it without either side believing a claim the connection did not make.

<!-- PTR:STATUS:BEGIN -->
## Current implementation status

> **Generated section.** Source of truth: [`component.toml`](component.toml) plus code-derived metrics from `src/`. Run `python3 scripts/update_component_docs.py --write` after editing implementation metadata. Do not hand-edit inside this block.

**Maturity:** `prototype`  
**Last reviewed:** 2026-09-22  
**Code footprint:** 4 Rust source files · 3385 nonblank source lines · 3 integration-test files · 46 test markers (`#[test]`, `#[tokio::test]`)

### Implemented now

- PodHost and PodClient carry one Pod invocation and its answer over ALPN_PODWIRE, composing ptr-pods' registry with ptr-net's transport so neither has to know about the other
- A request frame carries no project field: the project a request resolves in comes from the host's access policy keyed by the authenticated peer, because resolution inside a project is the only thing standing between a request and a Pod's data and a requester that named its project would be choosing which project's Pods it reaches
- A V1 request frame has no field naming its sender and none naming a session (the V2 frame below carries an identity digest and session id as binding fields, but the sender is still the connection's authenticated peer), so who is asking is the authenticated connection's answer and a session stays a capability the host holds rather than a string a caller presents
- A request names the endpoint it is for, so a request composed for one host is refused when relayed to another rather than being indistinguishable from one meant for it
- An answer names its own author and is bound to a digest of the exact bytes that arrived; a requester checks the author against the peer the connection authenticated before reading anything else, then the request id, then the digest
- One refusal code covers out of scope, not registered and registered in another project, so a cross-project Pod is refused in exactly the same words as one that does not exist and a scope cannot be read as a directory of what the host serves
- Admission and scope are decided before the registry is touched, so an unadmitted or out-of-scope requester cannot probe for Pods at all
- PodScope matches an exact capability and payload type pair, never a prefix or a wildcard, and it is the same pair PodRegistry::resolve keys on so no scope can be written that resolution would not honour
- One peer holds one entry: a second admit is refused rather than replacing the first, and a withdrawal takes effect on the peer's very next request because the scope is read from the table at every request
- Only Pure and Read Pods are reachable; an effectful Pod is refused with a code that names the action boundary, because this protocol commits no attempt and can report no uncertainty
- PodManifest::protocol_version is checked against the version the requester composed its payload for — the first place anything reads that field, and the only place where the caller and the Pod are not built together
- A Pod's output is verified by the host's verifier before it leaves, and a Pod that failed is a different refusal from an output the host will not vouch for; a verification report that carries any hard finding is refused as Unverified even when its status is Pass, on both the V1 and the V2 path
- The host holds no runtime and no ledger, so a Pod invoked over this wire cannot write to the host's history: a property of the type rather than a promise about the code
- Two outcomes rather than the execution wire's four, and no replay window and no at-most-once key, each absent by derivation rather than omission because nothing reachable here has an effect that could half-apply
- A request/answer frame format with distinct magics per direction, its own digest domain, an explicit layout version, bounded length-prefixed fields, an explicit two-way code table for every enumeration, and a distinct diagnostic code per refusal
- Every request is answered, including one this build cannot parse — its answer is bound to the bytes that arrived and reports no request id rather than inventing one
- An answer this side cannot encode becomes a refusal rather than no reply at all: the frame bound is enforced by trying to encode rather than by pre-checking one field, because a pre-check on the output bytes would miss the output type and every miss leaves a requester waiting on an open connection
- No background loop, timer or retry policy: the caller drives serve_once and request
- Generation-bound V2 frames (FORMAT_V2, and FORMAT_V3 when an address binding is present): encode_request_v2/decode_request_v2 and encode_answer_v2/decode_answer_v2 add artifact id, manifest hash, generation, revision, identity digest, session id and policy revision to the request, and the answer echoes the generation, revision, identity digest, session id and policy revision, bound to the exact request bytes; optional placement epoch and fencing token travel together or not at all, and V1 stays a separate, unchanged format
- V3 frames add a PodAddressBinding (source and destination pod address, semantic revision, trace id, hop limit, visited list of at most 64 distinct addresses, optional protocol and mesh endpoint binding) plus the execution-manifest digest; the binding is validated on encode and on decode, and an address binding without an execution manifest digest is refused
- PodHost::bind_v2 and its stateful, addressed and admitted variants build a PodWireV2Binding only from a pod actually registered in the host's registry; PodHost::serve_once_v2 answers a request whose artifact, manifest hash, generation, revision, state binding, admission binding or address binding differs from the binding as Unavailable before the Pod runs
- PodAdmissionBinding::from_identity refuses an IdentityContext whose digest does not validate, so the identity digest a V2 request must carry is checked before it enters a binding
- A V2 binding can carry an ExecutionManifest (with_execution_manifest, with_resolved_execution_manifest via the ExecutionManifestResolver trait, with_runtime_execution_manifest resolving the exact digest through ptr-runtime); a host given a live manifest registry (with_runtime_manifest_registry) re-resolves the digest at every request, so a manifest revoked after binding is refused before the Pod is invoked
- answer_bound runs exactly the Pod the binding captured, looked up by id and compared against the bound manifest, instead of selecting again by capability; it applies the same scope, Pure/Read-only, protocol-version and verifier checks as V1, and an output type the manifest does not declare it produces is refused as Unverified
- PodClient::request_v2 checks the V2 answer against the request (responder, request id, digest, generation, revision, state binding) and refuses an answer with the wrong generation or revision
- PodClient::connect_session returns a PodSession: one authenticated connection carrying many requests, each on its own stream with the same answer checks, with max_in_flight passed to the transport as the concurrency bound; PodHost::serve_session_v2 serves a caller-chosen, bounded number of V2 streams on one connection (zero is refused)
- PodSession records a request id as uncertain after a timeout, transport failure, undecodable answer or failed answer validation and refuses a retry of that id with RequestUncertain; an identical re-send of an answered id is served from a bounded (256 entry) session cache and conflicting bytes under a used id are refused as DuplicateRequest; PodSession::recover_uncertain_request only forwards a UncertainRequest to a caller-supplied StatefulRequestRecovery, it performs no recovery itself
- The whole composition is feature-gated, so iroh stays out of every other workspace build exactly as ptr-net keeps its own backend out
- A requester dials only what the deployment wrote down: request takes a ptr-net PeerAddress, which has no public constructor, so a bare address does not satisfy the signature and an address handed over by a peer or read out of a payload cannot be dialled

### Missing for the target architecture

- Authentication: the peer key is the one ptr-net reports the connection authenticated, and this crate takes the host's word that it passed that rather than something else
- Signed answers, and therefore any third-party evidence: inside one exchange the connection vouches for an answer's author and outside it nothing does
- A journaled access policy: the scope table is in memory, so it does not survive a restart and nothing records who was admitted to which project when
- Discovery: which address belongs to an id nobody told you is a mechanism choice with its own trust question, recorded as the owner's rather than invented
- An accept loop a deployment would run, including backpressure toward a peer that asks faster than the host can answer
- Session reuse by default: PodClient::request and request_v2 still open a connection per request; reuse is opt-in through PodSession, and the in-flight bound's Backpressure error is not exercised by any test in this crate
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
- v2_round_trip_binds_artifact_manifest_generation_and_revision and legacy_v2_round_trip_remains_available_without_address_binding (frame unit tests): V2 and V3 request and answer frames round trip with their bindings
- v2_binds_manifest_artifact_generation_and_revision_over_a_real_connection: a V2 exchange over a real connection is answered and the answer carries the bound generation and revision
- v2_rejects_a_stale_generation_before_invoking_the_pod and v2_rejects_a_protocol_binding_mismatch_before_invoking_the_pod: a stale generation or a mismatched protocol binding is refused with zero Pod invocations
- v2_binding_rejects_a_manifest_digest_that_is_not_the_registered_manifest: a binding cannot be built over a manifest digest other than the registered one
- v2_client_rejects_answers_with_wrong_generation_or_revision and v2_client_rejects_a_response_with_stale_state_binding: the client refuses an answer whose generation, revision or state binding does not match
- live_manifest_revocation_blocks_an_existing_binding_before_invoke: a manifest revoked in the live registry is refused although the binding already exists, and the Pod is not invoked
- v2_session_reuses_one_authenticated_connection, v2_session_replays_identical_request_ids_but_rejects_conflicting_bytes and v2_session_marks_an_unverifiable_answer_uncertain_and_blocks_retry: session reuse, the replay cache, the conflicting-bytes refusal and the uncertain-request block
- identity_admission_podwire_protected_kv_ledger_and_recovery_are_bound (tests/identity_recovery.rs): over one session a request with the bound identity, session, policy revision and placement epoch is answered and each of four requests with one of those changed is refused as Unavailable with the Pod run once; a wrong-generation answer marks the request uncertain and recover_uncertain_request drives ptr-runtime's RuntimeRecoveryAdapter through its cleanup steps, after which a new session with a new identity is answered
- compile-fail doctest: a bare endpoint address is not an argument to request
- frame codec unit tests: round trip of every outcome and refusal code, a request never read as an answer and never as the execution wire's frame, every prefix refused, no single bit change yielding the original request, trailing bytes, an unknown layout version, an invented length refused by the bound, an oversize frame refused before parsing and on the way out, an unknown outcome or refusal code refused rather than guessed, zero deliberately not a code, invalid UTF-8 refused rather than replaced, one distinct code per refusal, and this wire's digest asserted different from the execution wire's digest of the same bytes

<!-- PTR:STATUS:END -->

## Reusable V2 sessions

With `podwire-backend`, `PodClient::connect_session` creates a reusable
authenticated Iroh connection. Every request still uses its own bidirectional
stream and retains the normal responder, digest, generation and revision
checks. `max_in_flight` is a hard bound and overload is returned as typed
backpressure; the session does not add promotion or effect authority.

`PodHost::serve_session_v2` serves a caller-selected bounded number of V2
streams and waits for the peer to close after the final response. PodWire
remains effect-free; all external effects stay behind `ptr-execwire`.

If a request reaches an outcome-uncertain failure (transport failure, timeout,
malformed answer, or binding validation failure), the reusable session records
that request ID as uncertain and refuses a retry on the same session. The
caller must close the session and perform the surrounding runtime recovery
(lease revoke, resource cleanup, re-admission, and recompute where required)
before issuing a new stateful request. This keeps PodWire from silently
replaying an operation whose remote execution status cannot be established.

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

## Generation-bound PodWire V2

V1 remains the compatibility protocol for Pure/Read Pods whose invocation
contract is sufficient on its own. V2 is used when a request must be bound to a
specific admitted artifact and semantic generation. Its binary request frame
adds:

- `artifact_id`, so a request cannot be redirected to another implementation;
- the canonical `PodManifest` digest, so a non-empty arbitrary hash is not an
  admission proof;
- `generation` and `revision`, so stale artifact or runtime state is refused;
- `identity_digest`, `session_id` and `policy_revision`, so a stateful request
  cannot cross an authenticated identity/session or policy boundary;
- the existing capability, protocol and typed payload fields.

The binary answer echoes the complete admission binding and generation/revision
and remains bound to the exact request bytes, responder identity and request id.
`PodWireV2Binding` is created
by `PodHost::bind_v2` from an actually registered project/pod pair after
Artifact/Runtime Admission; its fields are opaque and the wire does not create a
second catalog or promotion authority. `PodHost::serve_once_v2` then reuses the
same access policy, project-scoped `PodRegistry`, verifier and effect boundary
as V1. A binding mismatch is answered as `Unavailable` before the Pod runs.
For the fully bound stateful path, callers use `PodHost::bind_v2_admitted` with
`PodAdmissionBinding::from_identity`; the identity digest is validated before
it enters the wire binding.

The endpoint is deliberately explicit: callers use `serve_once_v2` and
`request_v2` for generation-bound traffic, while `serve_once` and `request`
remain unchanged for V1. This makes protocol migration observable and prevents
an old V1 caller from accidentally claiming generation safety.

The real Iroh tests cover a successful V2 exchange and stale-generation
rejection with zero Pod invocation. The frame codec tests separately cover V2
binary round trips.

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
