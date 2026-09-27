# ptr-runtime — End-to-End Runtime Orchestrator

> **Role:** Compose PTR's semantic, model, execution, verification, security and authority layers into one request lifecycle.

<!-- PTR:STATUS:BEGIN -->
## Current implementation status

> **Generated section.** Source of truth: [`component.toml`](component.toml) plus code-derived metrics from `src/`. Run `python3 scripts/update_component_docs.py --write` after editing implementation metadata. Do not hand-edit inside this block.

**Maturity:** `prototype`  
**Last reviewed:** 2026-09-27  
**Code footprint:** 6 Rust source files · 4635 nonblank source lines · 20 integration-test files · 179 `#[test]` markers

### Implemented now

- bind_checkpoint resolves the slot encoding an artifact records and refuses a definition this build has no implementation of, because weights trained on vectors it cannot recompute are weights it cannot feed; the encoding is deliberately not part of a StateDeclaration, which says which committed facts a state came from rather than how an artifact was constructed
- AdmissionPolicy maps a transport-authenticated peer to one principal and one grant set; admit_peer takes the peer and nothing else, so no caller-supplied parameter can name either, and a second entry for a peer is refused rather than replacing the first
- A session admitted from the policy re-derives its authority at every use, so withdrawing a peer or replacing the policy stops live sessions at once instead of when a TTL runs out; a host-registered session carries no peer and is unaffected
- Pod resolution is scoped by project, and a Pod in another project is unavailable in exactly the same words as one that does not exist, so a refusal is not an existence oracle across the boundary
- Durable execution audit: an effect commits an EffectAttempted record before it is dispatched and an EffectSettled record after, so the window between them is described by committed history rather than by process memory
- The fence is that record: an attempt with no settlement fences execution and commits, is rebuilt on every open, and therefore survives a crash or panic inside the effect window
- reconcile_effect commits what an operator established from the receiving system; it never infers an outcome and never retries a possibly-applied effect
- prepare_execution_once adds an at-most-once key: a retry is answered from history without dispatching again, across restarts, while an attempt reconciled as not applied leaves the key free
- An applied key is bound to the project, principal, revision, generation and action digest its attempt recorded and to that attempt's index, set when the attempt is settled or reconciled: a permit under it is a retry when its project and principal match and its action digest, taken at the recorded revision and generation, matches the recorded one, so a retry carrying the current revision and generation is answered after either has moved, and any other permit is refused with KeyBoundToAnotherAction before anything is recorded, and a detached retry or a ResponseNotRetained refusal names the attempt that settled its own key rather than one inferred from response bytes or from the first attempt record under the key; a key reconciled as not applied binds nothing
- A response is retained up to MAX_RETAINED_RESPONSE and its digest unconditionally, so a retry whose response was not kept is refused rather than re-executed or answered with something else
- A denied preparation writes no effect record, and a fenced runtime yields no journal anchor, so no compaction floor can discard a record saying an effect may have applied
- Rustdoc covers the compacted-snapshot and neural-state APIs plus their canonical framing helpers and admission diagnostics
- Detached dispatch for work that does not finish inside the call: the attempt is committed and left unsettled, so the runtime is fenced from the moment the work is handed over until the adapter answers or a person reconciles it, and a refused dispatch writes no record
- A grant is synchronous or detached when it is issued, and each path refuses the other kind before the attempt is committed; both share one admission sequence, so there is one place where the order of checks and the attempt record is decided
- settle_detached records the adapter's own answer and is available only for an attempt this runtime dispatched; the dispatch kind is deliberately not rebuilt on reopen, so after a restart the fence stands and reconciliation is the way forward
- bind_checkpoint binds a real stored model artifact: it reads the shared CheckpointHeader, refuses an assignment this build cannot reproduce and an artifact whose codebook contradicts the declaration, then binds the opaque payload through the same bind_state path every other neural state uses
- PTRNEU01 neural/KV/checkpoint admission: opaque state bound to an anchored journal position, per-input semantic value digests, lifecycle generations, provenance and codebook version plus assignment fingerprint
- Admission is decided at every use rather than at insertion; a payload is reachable only through an admission decision, and a cache exposes no accessor that bypasses one
- Revocation, supersession, edited or removed inputs, foreign history with identical counters, unverifiable positions, changed codebook assignment and a fenced runtime each deny with their own stable code; failure to verify is a denial rather than an error
- bind_state derives every field from committed state and validates its own output through the same admission rules a later use applies
- PTRCS003 compacted materialized snapshot: committed state at a floor with canonical ascending-key sections, checked framing and an externally retained trusted anchor
- register_execution_session and AdmissionPolicy::admit refuse a principal longer than MAX_PRINCIPAL_BYTES, the compacted section's 4096-byte string bound, with InvalidSession, because a spent key's principal travels in every later snapshot and a longer one would make every export after its first keyed effect fail
- commit refuses a keyed EffectAttempted before append unless its key is an identifier of at most MAX_KEY_BYTES (InvalidEffectKey), its principal 1 to MAX_PRINCIPAL_BYTES bytes (InvalidEffectPrincipal) and its project 1 to MAX_PROJECT_BYTES bytes (InvalidEffectProject), all 4096, and admission appends its attempts through commit; prepare_execution_once refuses a key past MAX_KEY_BYTES with InvalidKey, because a settled key carries itself and, once applied, its attempt's project and principal into every later snapshot. The principal and project are held to that length and no more, so a principal no session could be registered as is recorded as given, and an unkeyed attempt enters no settled key and its principal and project are recorded as given. Replay does not apply these bounds, only the identifier check every build has made on a key: replay, and through it restore_recovery_snapshot, read_recovery_snapshot, restore_durable_snapshot and migrate_legacy_log; open_durable and open_durable_at; and the records restore_compacted replays above a floor. A log an earlier build wrote, which could hold a key up to the execution wire's 64 KiB field and a principal or project of any length, therefore opens as it did then: its key stays bound, a retry under an over-long key is refused by preparation before anything runs, and every export fails with PTR_COMPACTED_SECTION_LIMIT, as before, once such a key has settled (for an over-long principal or project, once it has applied). Refusing it on replay, which an earlier revision of this change did, would keep the runtime from opening after an upgrade
- install_admission_policy re-derives every policy-admitted session's grants and principal from the new table, so a replacement that narrows a peer narrows its live session instead of leaving the wider set in force until the TTL; a session whose peer is no longer admitted is kept so the refusal can still name PeerNotAdmitted, and the expiry is never extended
- A compacted snapshot carries the execution obligations a raised floor would otherwise discard (PTREX002): spent at-most-once keys with their outcomes, each applied one with the attempt that settled it and the project, principal, revision, generation and action digest that attempt recorded (a retained response encoded by byte length, so one up to MAX_RETAINED_RESPONSE round-trips rather than only one under the item bound), and unsettled attempts with the same five, so a restored runtime neither executes a spent key a second time nor answers another action under it. A PTRCS001 snapshot is refused by version rather than read as an empty obligation set, and a PTRCS002 snapshot or PTREX001 section rather than read as keys bound to no action
- restore_compacted installs floor state then replays the retained journal through the ordinary lifecycle/semantic validation path, keeping committed indices
- CompactedSnapshot::covers reports only the position it holds, so a compaction barrier cannot claim coverage the snapshot lacks
- Versioned replay-backed recovery snapshots bind complete journal, semantic revision, commit index and independently trusted SHA-256 anchor
- Snapshot validation and semantic/lifecycle replay finish before creating a durable destination; existing destinations are never replaced
- Strict anchored reopen and explicit legacy-to-v2 migration preserve history while restoring no process-local authority
- Ordered semantic journal publication before acknowledgment or model resume
- Semantic payload/dependency/revision reconstruction with schema and transition validation during replay
- Complete typed Pod bytes and source identity are revision-significant
- Fallible ingestion, optimistic base revision and canonical no-op handling
- Scoped synchronous execution gateway with opaque per-runtime sessions, exact grants, frozen ActionIR permits and registered verifier/executor binding
- Consume-time freshness, expiry, ownership and mandatory permission checks; permissions/commit/effect epochs invalidate pending permits
- Executor or ledger ambiguity fences subsequent execution and commits; process-local single-use authority is never replayed
- Ledger-only lifecycle mutation; invalid/rewinding/tombstoned/cross-project transitions rejected before append and during replay
- Central runtime object wiring configuration, SemDB, ledger, materialized state, events and permissions; an in-memory append that runs out of commit indices surfaces as a ledger error rather than a repeated index
- Durable standalone runtime constructor opens/replays FileLedger and reconstructs lifecycle/materialized state across restart
- Text ingestion into revisioned semantic state
- Current-revision, live-generation/revocation and capability/effect checks for ActionIR delegated through typed ptr-security authorization decisions
- Committed-event materialization and runtime event emission
- Committed-event replay rebuilds lifecycle state and preserves revocation
- Reference InferenceBackend request loop runs against a revisioned ModelRequest and emits completion event
- Typed model→PodRegistry→Pod→Verifier→SemDB observation loop for Pure/Read cognitive Pods
- Bounded multi-step model resume loop after verified Pod observations advances semantic revision before continuation
- apply_verified_semantic_delta prepares a delta, hands the verifier a view of the post-state's values and dependency sets and appends only on a Pass at the required level with no hard finding; a refusal writes nothing and names status, level and finding count
- apply_certified_semantic_delta is that path for work that relied on lifecycle generations, such as a certified branch's MergePlan: after verification and immediately before append, in the same &mut self call, it asks generation_validity about every relied (target, generation) and refuses any that is revoked, superseded or unknown (StaleReliance, PTR_RUNTIME_STALE_RELIANCE, naming each with its current validity), a no-op included, with nothing appended; it checks only the generations it is given, so passing all of them is the caller's obligation
- generation_validity combines the tombstone set with generation equality, so a revoked generation that is still the live one reads as Revoked rather than Live; ptr_search::retain_live takes it directly, so search hits are filtered by validity rather than by the live generation
- Neural-state input digests are computed from ptr-semdb canonical_input_bytes, byte-identical to the previous encoding (committed fixtures unchanged)

### Missing for the target architecture

- Durable compacted-snapshot publication and a runtime backed by an AcknowledgedLedger; compacted restore rebuilds an in-memory ledger above the floor and therefore retains no chain base, so it admits no neural state at all
- Any bound on how long detached work may stay outstanding, and any supervision of a detached adapter: there is no timeout, cancellation or retry, because a runtime that decided the work had failed would be deciding something it cannot observe
- A truly concurrent permission or lifecycle race inside one runtime: exclusive access prevents the interleaving by construction, so the concurrent case is two runtimes deciding at once, which ptr-execwire drives over a real connection rather than this crate
- Durable neural-anchor catalog and streamed or memory-mapped payloads; admission checks the producer's declared input set only, so an undeclared dependency stays invisible
- A trainer that binds a checkpoint as it saves one: bind_checkpoint attaches committed facts after the fact, so an artifact between saving and binding carries its assignment but no position
- An unforgeable peer identity: the NodeId passed to admit_peer is still taken on the caller's word here, and making it constructible only from an authenticated connection would require this crate to depend on ptr-net, which is an open decision; the execution wire itself now exists in ptr-execwire, where the forged-receipt and cross-runtime cases are tested over a real connection
- The admission policy is in memory: it is not journaled, so it does not survive a restart and a replayed history does not describe who was admitted when
- At-most-once memory is bounded by retention: a floor rising past a settled attempt discards its key, and reconciliation is a privileged host API whose caller this crate does not authenticate
- At-most-once keys scoped per principal or project: the memory is addressed by the key alone, so a refusal under a key tells a principal that another spent it and a principal that spends a key first makes it refused to every other; a caller has to choose keys nobody else can guess
- A spent key is refused at admission, not at validation: an EffectAttempted record a host commits directly under a spent key is accepted and replayed, and the key then holds what that attempt's settlement established
- Bounds on an attempt's target and operation: the materialized projection records both and the lifecycle section writes every materialized value as a string of 1 to 4096 bytes, so an attempt, keyed or not, with an empty or longer one leaves every later export failing; admission requires both to be identifiers but bounds neither's length, so a grant with an operation past 4096 bytes reaches it, and validation bounds no other record's materialized strings either
- A bound on the settled keys in total: every settled key goes into the one execution section, at most MAX_SECTION_BYTES (8 MiB) and 65,536 entries, an applied one with its retained response of up to MAX_RETAINED_RESPONSE (1 MiB), and nothing removes a settled key, so eight keyed effects each settled with a full-size response, or 65,537 settled keys, leave every later export failing with PTR_COMPACTED_SECTION_LIMIT although the fence has cleared, through admission alone; the per-string bounds on key, principal and project do not reach this
- Router-driven operator selection around the implemented bounded Pod-resume loop
- Async isolate scheduler integration
- Configured raft-engine/raft-rs/Turso backend composition for production runtime modes
- Streaming client response lifecycle and cancellation

### Next milestones

- Decide whether peer identity should be constructible only from an authenticated connection, which would make this crate depend on ptr-net; until then ptr-execwire is the one host that passes an authenticated key and nothing else
- Define the PodWire request framing, which is still undefined even though the execution wire is not
- Retaining a NeuralAnchor outside the artifact remains a deployment obligation; nothing here stores one
- Connect evaluated raft-engine/Turso adapters through typed backend config while preserving FileLedger reference mode
- Add opaque backend checkpoint handles and async streaming around the implemented observation resume contract
- Wire ptrd request handling beyond bootstrap ingestion

### Linked experiments

- [E001](../../experiments/system/E001-end-to-end/README.md) — `planned`
- [E003](../../experiments/system/E003-token-efficiency/README.md) — `planned`
- [E004](../../experiments/system/E004-long-horizon/README.md) — `planned`

### Technology evaluations

- None recorded.

### Decision records

- [ADR-0001-rust-runtime.md](../../research/decisions/ADR-0001-rust-runtime.md)
- [ADR-0012-hard-effect-boundary.md](../../research/decisions/ADR-0012-hard-effect-boundary.md)
- [ADR-0013-runtime-orchestrator.md](../../research/decisions/ADR-0013-runtime-orchestrator.md)

### Current automated checks

- the committed checkpoint fixture records the slot encoding it was produced under, and verification names both the codebook and the encoding
- detached work fences the runtime until the adapter answers: the fence names the attempt, no permit can be prepared, no journal anchor or compacted snapshot is produced, and the settlement carries the response digest
- an adapter that would not take the work still leaves the window open, because not accepted is not not-applied; a detached attempt survives a restart as a fence that only reconciliation moves, and the adapter's own answer is refused there
- each dispatch path refuses the other kind of grant with no record written and nothing verified; a settlement for an attempt this runtime did not hand out, and a second settlement of one it did, are refused
- an at-most-once key hands detached work out once: while the first is outstanding the fence refuses a retry by name, and after settlement a retry is answered from history without calling the adapter; an oversize answer keeps its digest without being retained and the next retry is refused rather than answered with something the effect never produced
- withdrawing one peer or replacing the policy in the prepare-to-consume window refuses that permit before the verifier while another session keeps working; a generation superseded or revoked in that window is refused before the verifier; one session's successful effect refuses another's in-flight permit without reaching the executor
- a real ptr-burn-a0 checkpoint binds to committed state, is admitted, seals and reopens with its weights unchanged, and is then refused once the generation it was bound under moves and once the semantic input it read is edited
- a checkpoint whose assignment moved, one whose codebook contradicts the declaration, one whose table width is not its family's cardinality and every prefix of one are each refused, with the intact fixture binding as the control
- an admitted peer executes only its granted action and the audited principal is the policy's; a different operation or project is denied before any check that could depend on what the caller claims
- withdrawing a peer refuses a permit issued beforehand, new preparations and re-admission without the TTL moving; replacing the policy kills live sessions and re-admitting does not revive them
- a Pod in another project is unavailable in a refusal equal to the one for a Pod that does not exist, while the same request inside its own project succeeds
- a runtime reopened after an executor error or panic is still fenced by the unsettled attempt it left, refuses to register a session and refuses to commit
- an applied effect commits its attempt before and its settlement after, with the action digest and admitting verification level asserted field by field
- a retry under an at-most-once key returns the first response without dispatching again, before and after a restart, while a different key does dispatch
- reconciliation lifts the fence for both outcomes and lets a key reconciled as unapplied execute again
- a fenced runtime yields no journal anchor and no compacted snapshot; a denied preparation writes no record
- settlements naming no live attempt, a second live attempt under one key, a malformed key, and a retained response that is oversize or contradicts its digest are each refused before append and during replay
- an empty runtime round-trips at the empty floor; a runtime restored at the index ceiling refuses a commit with PTR_LEDGER_INDEX_EXHAUSTED, stores nothing and fences the retry; revocation outranks every other binding mismatch simultaneously
- an admitted state reproduces the identical inference event sequence after a restart, with binding and opaque payload byte-for-byte equal
- revocation, supersession, edited/removed inputs, foreign history with identical counters, unverifiable position, compacted restore, contradicting revision, unknown/changed codebook, unknown target and a fenced runtime each denied
- every single-bit mutation of a sealed state rejected; wrong outer/binding magic, reserved field, length mismatch, byte removal/insertion and undersize rejected after resealing the digest
- an unrelated commit leaves a state admissible, so dependency-precise binding is asserted in both directions
- compacted reconstruction equals a full replay across semantic payloads, dependencies, supersession, constraints, procedures and revocations
- a revocation below the floor still denies after compaction; every single-bit mutation of a compacted snapshot rejected
- resealed noncanonical sections, reserved fields, length and trusted-identity mismatches, and records not above the floor all rejected
- snapshot exact-state/byte-corruption/rollback/revision/legacy/overwrite/authority rejection tests
- scoped execution positive/negative integration tests and non-forgeability/single-use compile-fail doctests
- runtime ingestion/revision/generation/capability/materialization integration tests
- typed authorization decision exposure and RuntimeError compatibility test
- reference model-loop integration test
- bounded verified Pod-observation resume-loop tests
- revocation replay/restart integration test
- durable FileLedger runtime reopen preserves revocation and materialized commit position
- workspace fmt/check/test/clippy
- tests/at_most_once_keys.rs: another payload and another principal under a spent key are refused with KeyBoundToAnotherAction, reach no executor and write no record, while the request that spent the key is still answered, and a detached dispatch under a spent key hands nothing out; a key reconciled as applied is bound to its attempt's action and one reconciled as not applied binds nothing; the binding holds for payload and principal after a restart and after a compacted round trip, and for the project, from an attempt written into the log directly, after a restart and after a compacted round trip; a detached retry names the attempt that settled its own key when another key settled equal bytes and after compaction, and a ResponseNotRetained refusal names it when the key was attempted twice and after compaction; a retry carrying the current revision and generation is answered after an unrelated delta moved the revision and after the capsule's generation was superseded, live, after a restart and after a compacted round trip, while another payload and another principal are still refused; a principal longer than MAX_PRINCIPAL_BYTES is refused by register_execution_session and AdmissionPolicy::admit, and the longest one allowed still exports and restores after spending a key; a keyed attempt whose principal is empty or past MAX_PRINCIPAL_BYTES is refused with InvalidEffectPrincipal at commit, while a history holding it, its key reconciled as applied, reconciled as not applied or left unsettled, opens with replay, open_durable, open_durable_at and restore_recovery_snapshot, and a log an earlier build could have written holding it reopens however the attempt ended, keeps its key bound and fails export with PTR_COMPACTED_SECTION_LIMIT once the key has applied; an unkeyed one naming it is recorded, exported and opened on each path, and the longest one allowed and ones padded or holding a control character, committed directly under a key, open on each path, export and restore with the key still bound; a key past MAX_KEY_BYTES, up to the execution wire's 64 KiB, is refused by prepare_execution_once and at commit, while a history or log holding it opens however it ended, a retry under it is refused by preparation, and export fails once it has settled either way; the longest allowed opens on each path and is spent, exported, restored and answered; a keyed attempt under an empty project or one past MAX_PROJECT_BYTES is refused at commit and through admission as Audit(InvalidEffectProject) with no record, no fence and no executor call, while a history or log holding it opens as the principal's does; the longest project allowed opens on each path, spends a key and round-trips; a PTRCS002 snapshot and a PTREX001 section are refused by version after resealing (a_spent_key_is_refused_for_another_principal_rather_than_answered_with_its_receipt, a_retry_is_answered_after_the_revision_and_generation_move_and_another_request_is_still_refused, a_principal_longer_than_a_snapshot_carries_is_refused_before_it_can_spend_a_key, a_keyed_attempt_whose_principal_no_snapshot_carries_is_refused_at_commit_and_a_logged_one_still_opens, a_key_longer_than_a_snapshot_carries_is_refused_at_preparation_and_at_commit_and_a_logged_one_still_opens, a_keyed_attempt_whose_project_no_snapshot_carries_is_refused_at_commit_and_at_admission_and_a_logged_one_still_opens)
- tests/verified_delta.rs: a verified delta commits the state its verifier saw; no failing score, shallow level or hard finding gets a delta past verification; a moved revision is refused before verification; revoked, superseded and unknown generations never read as Live; search hits filtered by generation_validity drop a revoked live generation; a certified delta is refused while a generation it relied on is revoked, superseded or unknown, a no-op included, and appends nothing (a_certified_delta_is_refused_while_a_generation_it_relied_on_is_not_live_and_appends_nothing)

<!-- PTR:STATUS:END -->

## Position in PTR

```mermaid
flowchart LR
  I["Ingress / API"] --> R["ptr-runtime"]
  R --> S["SemDB"]
  R --> M["Model / Router / Pods"]
  R --> V["Verifier / Security"]
  R --> L["Ledger / State"]
```

Dedicated diagram source: [`docs/diagrams/components/ptr-runtime.mmd`](../../docs/diagrams/components/ptr-runtime.mmd)

## Mission

Keep orchestration out of `ptrd` and out of individual domain crates. The runtime coordinates transitions while each component retains ownership of its semantics.

## Current boundary

The executable reference slice now wires configuration, semantic revisioning, a backend-neutral model call, semantic Pod resolution, Pure/Read Pod execution, verifier-gated observation promotion, action authorization, event emission, ledger materialization and durable FileLedger reopen/replay. The new scoped synchronous execution gateway binds opaque sessions and immutable ActionIR permits to host-registered verifiers/executors. Administrative session registration assumes the embedding host has authenticated the principal; it is not an HTTP authentication endpoint. See [P0.1 execution authority](../../docs/architecture/21-scoped-execution.md) for guarantees, counterexamples and the remaining durable/network gates. P0.2 now reconstructs the actual journaled semantic payloads, dependencies and revisions. The next persistence gate is record integrity and verified snapshot/checkpoint admission, not additional backend breadth.

## P0.2 semantic journal integration

[Durable semantic-state contract](../../docs/architecture/22-durable-semantic-state.md)
records the new code/codec, ownership and replay boundaries. Publication follows
successful journal append. Typed Pod bytes and source identity participate in
semantic revisions. Logical removals do not erase log history; neural checkpoints
and authenticated framing remain separate gates. Execution evidence is in the PR.

## Persistence contract update

See [P0.3 checked records and replay-backed recovery snapshots](../../docs/architecture/23-persistence-integrity.md)
for strict reopen, explicit legacy migration, independent anchors, create-new
restore and format/API compatibility. Old automatic crash-tail repair is replaced
by explicit anchored recovery. No authority or neural checkpoint is restored.
