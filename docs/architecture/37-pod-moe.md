# 37 — Pod-MoE: a mixture of experts whose experts are pods

Status: **design**, with narrowed slices implemented and measured on the reference path: the R011 expert-authority mechanisms in `ptr-runtime` and the R013 attestor and lineage-authority mechanisms built on them (§5, rows R011 and R013), and the verifier-closed loop of §11 on simulated experts (F007, [39](39-vcw-moe.md)). Nothing else in this document is implemented as a Pod-MoE. The pieces
it builds on exist as prototypes (`ptr-lineage`, `ptr-branch`, `ptr-router`,
`ptr-pods`, `ptr-verifier`, see [35](35-agentic-substrate.md)); the parts marked
*new* below do not exist yet. The base model is OLMoE-1B-7B. Training runs in Python
(TRL/PEFT, later verl) with PTR governance reached through the Rust binding
`bindings/ptr-py` (PyO3, merged in #123; §6). The `trl-moe` training backend exists in
`training/src/ptr_training/moe/` (configs `training/configs/stage-moe-smoke.toml` and
`training/configs/stage-moe-olmoe.toml`; the OLMoE config has not been run), and a CPU smoke run (WP8) has been
executed and documented in `training/README.md`, section "First training (WP8): CPU
smoke run" (`training/README.md:34`): a 0.24M-parameter random-init OLMoE, one seed, **pipeline evidence only, not research
evidence**. The first increment produces only contracts and a CPU
smoke pipeline: **no quality claim is made**, and every claim beyond the cited code is an
experiment (§5) with a falsification rule, per invariant 15.

**Current code this document builds on.** `aimer_score` (#114), `ExpertProposal` and
`ExpertProposal::proposal_id` in `crates/ptr-pods/src/expert.rs` (#115), and the
`ptr-router` routing API in `crates/ptr-router/src/router.rs` (#116: `Router::policy`
returning a `RouteDistribution`, the trusted `route_checked` and `CheckedRouter` that
sample with a secret `SamplingKey` and a fresh `DecisionNonce` and record the sampled
pod's quantized probability as a `Propensity` (exact width/2^53), and `RoutedError`
carrying the request id and digest) are all **merged on main** and are cited below as
current code.

## 1. Principle

Every expert is a pod: it has a manifest, a generation, a lineage record, a
lifecycle state and it is revocable. Where it executes is a separate decision made
by latency: an expert consulted once per token per layer cannot be a network
service, one consulted once per request can. Placement changes how an expert is
called, never what it is, so the same lifecycle, gates and revocation rules apply
at every level.

The closed-loop form of this principle, in which every L2 output is a proposal accepted only
through the verifier fabric and verifier findings are the primary routing signal, is section 11.

## 2. Three levels

| Level | Unit | Where it runs | Router | Evaluation / decision |
|---|---|---|---|---|
| L1 token | Frozen OLMoE base experts plus routed LoRA experts (PERFT-R) per layer | **In-process**, per token per layer; network latency is impossible here | Learned top-k router with auxiliary-loss-free bias | Training metrics (§5) and the lifecycle gates (§4) |
| L2 segment / request | Whole models or adapter sets as networked pods over PodWire ([33](33-pod-wire.md)) | Separate pods, one routing decision per call | Learned router in `ptr-router` (Router-R1 / RouteLLM style) | Offline evaluation needs a *new* contextual-routing log and categorical estimator (see below) |
| L3 meta | Structural operators Birth, Merge, Split, Freeze, Reactivate, Prune, Retire | Policy / agent pods | None; it proposes | Proposals are decided by the verifier and the gates, never by the policy |

`evaluate_off_policy` (`crates/ptr-branch/src/arbiter.rs:1027`) cannot evaluate L2:
it is specific to the fixed `TriageDecision` action space (`AutoPropose` / Escalate /
Discard, `arbiter.rs:13`) and cannot score a categorical choice among pods. L2 needs
a **new** contextual-routing log (request features, candidate set, chosen pod, and
the logging propensity as `ptr-router`'s trusted `route_checked` records it from the
`RouteDistribution` of `Router::policy`, a validated `Propensity` of the chosen pod in (0, 1]) and a categorical
IPS / SNIPS / doubly-robust estimator. Routing captures an **immutable, generation-qualified candidate snapshot** at decision time (pod id, generation, execution-manifest digest and propensity), and the invocation and the log use exactly that snapshot: invoking a generation that is not in the logged action set is rejected with a typed error and is never logged with the old propensity. Every L2 record logs the **complete
per-candidate distribution** (the `RouteDistribution`'s quantized probabilities for
every candidate) plus the logging-policy version or digest, not only the chosen
propensity. The estimator refuses a target policy that puts mass on an action whose
logging probability is 0 (positivity), as the arbiter does with
`ArbiterError::PositivityViolation` (`crates/ptr-branch/src/arbiter.rs:1174-1178`). The
arbiter's estimators are a template only.

**Actions and candidates are generation-qualified.** The log records, for every
candidate and for the chosen action, `(ProjectId, PodId, generation, execution-manifest
digest)`. The project is part of the identity because `PodRegistry` is keyed by
`(ProjectId, PodId)` (`crates/ptr-pods/src/lib.rs:174-192`), so two projects may each
register an `echo` Pod. The Pod
manifest alone does not carry a generation: `PodManifest` has `id: PodId` and a
`digest()` over the invocation contract (`crates/ptr-pods/src/lib.rs:157` and `:168`),
while the generation and the `manifest_digest` live in the `ExecutionManifest`
(`crates/ptr-pods/src/manifest.rs:14` and `:26`) that binds the served artifacts. The
log therefore takes the execution-manifest digest (or the artifact digest) together
with its generation, so rewards and propensities from different artifacts never
collapse onto one action label. Off-policy estimation groups actions by this
generation-qualified identity, and a record whose chosen generation was revoked or
replaced before verification is evaluated against **that** generation, not the new
one.

**The log must also record the observed outcome.** Context, action and propensity are
not enough: IPS, SNIPS and doubly-robust estimators weight an observed reward, and
the template reads it from `LoggedTriage.reward`
(`crates/ptr-branch/src/arbiter.rs:924`). Each L2 record therefore carries an
outcome field keyed by a unique **`routing_decision_id`** minted by the router at decision time, H(authenticated principal, session, project, request id, chosen generation-qualified action, decision nonce), and not by `(request_id, action)` alone; every outcome update and deadline finalization binds that id: the verifier's
`VerificationStatus` and the derived reward, using the four-status mapping of §5
(`Pass` yields the report's `score`; `Fail`, `Disputed` and `Unknown` yield 0, and
an output whose **weakest** per-verifier `VerificationLevel` is below the task's
minimum also yields 0; see §5).
It is written **after** verification, as an update of the record the routing decision
created, never by the router at decision time. Outcomes must not be censored by the action
taken, so a failure is an outcome and is **not** excluded: a timeout, crash,
unreachable pod or verification failure is assigned a preregistered status
(`Fail` or `Unknown`) with reward 0, recorded at a fixed deadline. Evidence-bearing off-policy estimates use only **matured cohorts**, those whose every member's fixed deadline has passed: unresolved outcomes get the preregistered zero reward, and pending members are **never dropped** from a live cohort (no action-dependent censoring). At the fixed deadline an unresolved outcome is **finalized atomically** as a terminal, **immutable** zero-reward outcome, and a verifier result that arrives after finalization **never overwrites it**: it is rejected for the record and logged separately as a late outcome, which is diagnostic only and never enters an evidence-bearing estimate. Live or interim estimates are diagnostic only, and an experiment that computes IPS, SNIPS or DR estimates from an immature cohort is invalid (immature_cohort_estimates_used above 0). An experiment that computes such estimates also requires late_outcome_overwrite_attempts of at least 1 (a late verifier result after finalization), and late_outcome_overwrites_admitted above 0 falsifies it. It also requires colliding_request_id_cases of at least 1 (the same request id across sessions), and cross_decision_outcome_updates above 0 falsifies it. **The reward and verifier-policy epoch is pinned per decision:** every routing decision binds the epoch (verifier set, minimum verification level, reward configuration) current at decision time, and a cohort is evaluated entirely under one pinned epoch; a cohort whose members span an epoch change is excluded from evidence-bearing estimates (or all its members are rescored under the pinned epoch before maturation). It requires reward_epoch_rotation_during_cohort_cases of at least 1, and mixed_epoch_cohort_estimates_used above 0 invalidates the run. Every estimate reports the counts per outcome type. Propensity weighting does
**not** correct outcome censoring: it reweights by the chance of the action, not by
the chance that the outcome was observed, so dropping failures would bias the
estimate toward pods that fail silently.

Token-level experts must be co-located, because every MoE layer finishes routing
before the next layer starts. Request-level experts can be networked services,
because there is one routing decision per call.

## 3. Self-observation and self-modification without self-promotion

**Self-observation.** Routing traces, expert load, churn, interference and AIMER
scores are written as observations into SemDB as derived keys with provenance. The
model can read them as context. They are derived and non-authoritative
(invariant 5): reading them informs a proposal, it does not decide anything.

**Self-modification flow.**

1. The L3 policy emits an `ExpertProposal`. The **operation id** is computed here, **before training**: a digest of the operation kind, the operands with their observed generation, digest, state and state_version, the proposer, the project, `observed_at` and the rationale digest, the intended results as `ResultIntent`s, and **no training-produced manifest digests**.
2. The proposal runs as a speculative overlay in `ptr-branch`.
3. Sandbox training produces a candidate. The **candidate id** is the current `ExpertProposal::proposal_id()` computed **after** training over the full proposal, including the digests of the result `ExpertPodSpec`s.
4. The candidate must pass the `ForgettingGate`, `measure_interference` against the
   calibrated null threshold (a preregistered margin, or a high quantile of overlaps
   between independently trained random adapters of the same shape; not the mean
   `chance_overlap`), the verifier and the public suites.
5. Only then is it committed as a ledger event (`ExpertPromoted` / `ExpertRetired`).

**Two-stage identity.** The attestation and the gate evidence bind **both** ids, and admission checks that the candidate's operation fields equal the operation id's. Both exist in `ptr-pods`: `ExpertProposal::operation_id()` delegates to `ExpertOperationRequest::operation_id()`, a request built before training (`ExpertOperationRequest`, with a `ResultIntent` per intended result: project, pod, level, base model ref and generation, but not the training-produced manifest digest), and `ExpertProposal::operation_request()` returns the request a proposal fulfils, so admission can check `proposal.operation_id() == request.operation_id()`.

**Gate evidence must be bound to the evaluated artifact.** This is a design
requirement, not existing behaviour. `ForgettingGate::evaluate` takes an `AdapterId`
plus caller-supplied matrices and scores (`crates/ptr-lineage/src/forgetting.rs:235`),
and `GateReport` binds the verdict to that id and to nothing else: the gate cannot
tell which adapter the accuracy matrix and public-suite scores were measured on, so
matching the measurements to the adapter is the caller's obligation
(`forgetting.rs:172-176`); in particular "the report names no lineage or base model" (`forgetting.rs:175`), so a pass on one base does not transfer to another. A promotion that rests on such a report would only be as
trustworthy as its caller. Therefore the `ExpertPromoted` event must bind **trusted
evidence**: the artifact digest and generation of the evaluated expert, the **exact evaluated
base model identity, generation and digest** (or the complete evaluated
execution-manifest digest, which folds them in), the digest of
the evaluation-data manifest (its **identity and generations**, not only a digest), the digest of the gate configuration (thresholds), and
the evaluator identity, and the **attested training manifest**: the training sandbox
is untrusted, so the full consumed-input manifest (every input identity, generation
and digest the trainer actually read) is recorded by the trusted orchestrator that
feeds the data, never self-reported by the trainer. `affected_by` uses this attested
manifest, **and the evaluation-manifest bindings as well as the training inputs**, and an expert whose attested manifest is missing or mismatched cannot be
promoted (`SECURITY.md:17-21`: revoked generations cannot become usable). The
evaluation must run in a **trusted evaluator**, not in
the proposer's sandbox, which produced the candidate and could hand the gate
measurements of a different artifact. The Python binding (§6) only calls the gate; it
does not make the result trustworthy.

Every trusted gate result **binds the exact proposal**: it carries `ExpertProposal::proposal_id()` (a canonical digest of the operation, all operands, the proposer and `observed_at`; defined in `crates/ptr-pods/src/expert.rs`), and evidence admission **rejects a report whose `proposal_id` differs from the proposal being committed**, so a valid report for one proposal cannot be reused for another (for example a Freeze report presented for a Prune or Retire of the same target generation).

**The proposer is authenticated by the host.** Admission binds the `ProposerRef` (the policy pod and its generation) and its `ProjectId` to the host-authenticated principal (the session's authenticated `PrincipalId`) and its project-scoped grant, not to caller-constructed proposal fields. The binding is to the proposer, not only to the project: a principal authenticated as policy P1 in project A cannot submit a proposal whose `ProposerRef` names policy P2 in project A. A proposal whose claimed proposer or project differs from the authenticated principal or grant is rejected with a typed error **before the evidence check and before the compare-and-set**, and every target must belong to the authenticated project.

**The attestation is bound to the artifact.** The consumed-input attestor signs **one record** binding the consumed-input manifest digest, the emitted artifact digest, the `proposal_id`, a **single-use training run id** and the attestor epoch. Gate evidence references that record, and admission rejects (a) an attestation whose artifact digest differs from the artifact the gate evaluated and that is being committed, and (b) a run id already consumed by an earlier admission, so an attestation for one run cannot be paired with another artifact or replayed. The single-use training attestation applies **only to artifact-producing events**: Birth, Merge, Split, the promotion of a newly trained candidate, and **Reactivate of a `Retired(GateFailed)` expert**, which creates a new generation g+1 (`ExpertProposal::creates_new_generation`). A **Split** attestation binds the **complete ordered set of result artifacts** (one artifact digest per result) in one attestation record, and admission requires exact set equality with the committed results. **Every result in the ordered Split result set also needs its own artifact-bound gate verdict** (one verdict per result digest, bound to the proposal and candidate ids), and admission requires a **passing verdict for each committed result**, so a good primary result cannot carry an unevaluated or failing secondary. **Non-producing events** (Freeze, Prune, Retire, and the in-place unfreeze of a `Frozen` expert) instead carry a **proposal-bound provenance reference**: the artifact digest, the ledger reference of the artifact's committed training-provenance record, and the `proposal_id`, checked at admission to match the operand's committed artifact digest. The binding is **per target**: the operation binds the complete ordered set of (operand pod, generation, artifact digest, committed-provenance reference) for every target (every pruned target of a multi-target Prune, and likewise any multi-operand non-producing operation), each reference is checked against its **own** target's committed artifact, and a missing or mismatched entry rejects the whole operation.

**Implementation status of the attestor record (R013, 2026-10-11).** `crates/ptr-runtime/src/expert/attestation.rs` implements an Ed25519-signed attestation for artifact-producing admissions only: the signed payload binds the attestor name and epoch, the key id, a single-use run id, the `proposal_id`, the artifact digest the operation determines, the base model (id, digest, generation) and the ordered result experts; trusted attestor keys are a host registry (name, epoch, key) fixed at construction of the runtime (no grant registers a key: a caller holding an admission grant can assemble a gate fence grant from its binding); the attestor, run and signature are recorded in the version 3 dependency block, replay re-verifies the signature and the bindings, and a grant's attestation requirement is recorded so that a restart with a looser grant cannot admit or replay an unattested artifact-producing record. Not implemented: binding the digest of the whole dependency block (the signed payload does not cover the other training or evaluation inputs or the policy marker; the manifest digest above is not signed), making the attestation requirement durable before the first attested record (the marker is recorded with that record), binding the consumed-input manifest digest, the Split ordered result set with per-result verdicts, proposal-bound provenance references for non-producing events, and the attestor as an evidence-authority kind of the gate-policy epoch (the registry is host memory, not ledger-committed).

**The evidence carries the gate-policy epoch.** The epoch is a **closed list** of the authorities whose judgement admission depends on: the gate configuration, the evaluator identity, the **identity and epoch of the consumed-input manifest attestor** (the orchestrator-side attestor of what the trainer read), and the **verifier set and policy** (the identity and generation of every required verifier, including the hard-deny verifiers). Together they form a **ledger-committed, revocable epoch**: revoking, rotating or adding any of them bumps it. **Any future gating authority joins this epoch by definition.** **The authority set checked at commit, for every gated event, is the evidence epoch plus the issuing grant's current generation:** the proposer grant for proposal-driven events, and the gate grant for events the gate host issues (`ExpertReadmitted`, `ExpertQuarantineRetired`, and the gate-driven `ExpertPromoted` and `ExpertRetired`). The commit-time compare-and-set includes the authenticated principal, session and project-scoped grant identity and **generation** recorded at the early identity check (or minted with the event), and an issuing grant revoked or superseded between that check and the commit rejects the commit with a typed error. At structural commit the runtime also **revalidates the proposer's authenticated session** (it must be unexpired and unrevoked, and the peer still admitted) and the **identity policy**, not only the evidence epoch and the issuing-grant generation. The session's revocation generation and expiry are part of the commit CAS, in the way `PtrRuntime::admit_identity_context` (`crates/ptr-runtime/src/lib.rs:1101`) already rejects a revoked identity session at admission. **Sweep rule:** every authority dependency checked at proposal admission (proposer identity, session, issuing grant, gate grant, evidence epoch and identity policy) is rechecked at commit, and **each has its own R011 race case** (identity policy rotation, identity policy revocation, peer de-admission, session revocation, session expiry, and the grant and epoch cases above). The commit-time compare-and-set also re-validates that the evidence's policy epoch is **still current** and rejects evidence from a superseded or revoked policy or evaluator, evidence whose attestation came from a superseded or revoked attestor, and evidence produced under a superseded verifier set (for example V1 revoked, or a hard-deny verifier V2 installed, between the evidence and the commit), so a proposal queued across a gate-policy update cannot commit on stale evidence.

**One shared evidence-admission function** admits every gate-driven structural event, and the list is exactly: Birth or promotion (`ExpertPromoted`), Merge, Split, Prune, Freeze, Reactivate (including unfreeze), proposal-driven Retire (reasons `Superseded` and `GateFailed`), **`ExpertReadmitted`** (the same-generation return to `Serving` after a quarantine of a readmittable authority kind, below), and **`ExpertQuarantineRetired`** (the failed-readmission retirement, below). **Sweep rule:** every event that makes an expert routable or changes its serving state, except the runtime revocation fence and retirement under their own grant, is admitted by the shared admission function. `ExpertFenced` and the revocation-driven retirement are **excluded**: they run under the runtime revocation grant. R011 runs the evidence-binding negative cases for each event type on that list.

**Every gate-driven structural ledger event** (`ExpertRetired`, and likewise the
results of Merge, Split, Prune, Freeze and Reactivate, not only `ExpertPromoted`)
carries the target identity, generation, artifact digest and the trusted gate
evidence above. Commit is **compare-and-set** over **every operand** of the proposal (all targets and all sources, including non-primary Merge sources and every Prune target): the event is rejected if any operand's
current generation, digest, lifecycle state or state_version differs from the one observed at evaluation, so a delayed event can
never retire a newer, unevaluated generation. At commit the compare-and-set **also
checks that every attested training input (identity, generation, digest) is live at
the commit anchor**: an input invalidated (revoked or superseded) after the gate passed and before the commit
makes the commit fail. The liveness check is **not limited to attested training inputs**: it covers the base model generation and **every non-operand structural dependency** (base model, tokenizer, router configuration, evidence epoch, and every attested input). The compare-and-set also covers **every result** of Birth, Merge, Split and a retired-target Reactivate: for each result pod id, result generation and routing slot (layer, slot) it requires **authoritative absence at the commit index, or a reservation held by this operation id**; a result identity or slot already registered or reserved by another operation is rejected with a typed error. The one exception is an **atomic takeover from an operand of the same operation** (a retired-target Reactivate g+1 on the same pod and slot, or a Split or Merge result reusing a source's slot): the CAS checks that the occupant is **exactly that operand at its observed generation and state_version** and replaces it in the same commit, while collisions with unrelated experts or with another operation's reservation stay rejected. `PodRegistry::register` (`crates/ptr-pods/src/lib.rs`) still returns the previous entry and replaces an existing one, so the admission path must not use it for results. `PodRegistry::register_new` is the non-replacing insert for fresh ids (it returns an error on a collision) and `PodRegistry::replace_if_current` is the compare-and-replace for takeover (replace only if the current entry is the expected one); both exist. The registry tracks no token-expert `(layer, slot)`, so slot ownership is **not** checked by either; the admission function must check slots itself.

**Dependency invalidation (the unifying rule).** A **dependency invalidation** is
either an explicit revocation **or a committed supersession** (a generation replacement
or an edit) of **any generation-bearing dependency**: a semantic input, a memory input, a derived input (or any other generation-bearing segment kind of the **closed list** in [36](36-kv-reuse.md), section D),
a training input, an evaluation input, an adapter generation, an expert generation or
the base-model generation (the KV side of the same rule is in
[36](36-kv-reuse.md), section D, where a **fence** of an expert for any cause also invalidates the KV entries whose chain includes it). Every protective mechanism below triggers on **any
dependency invalidation**, never only on revocation and never for only one dependency
kind: expert fencing through the compound event, fencing of in-flight expert calls,
completion-time dependency admission, and use-time admission. Each experiment
exercises the in-flight race for every dependency kind it covers. Wherever this
section says "revocation" inside one of these rules, read "dependency invalidation".

**Evaluation inputs are live inputs too.** The dependency index (`affected_by`), the use-time admission at serving, and R011 all include the evaluation-manifest bindings, not only the training inputs. The commit is rejected if any input named by the evaluation-data manifest is revoked or superseded at the commit anchor, and a **post-promotion dependency invalidation of an evaluation input** (revocation or supersession) fences the experts whose promotion evidence depends on it, through the same compound `InputInvalidated` path below.

**Invalidation after promotion.** On any dependency invalidation (revocation or supersession) of a training-input or evaluation-input generation, every
expert that `affected_by` returns over the **attested consumed-input manifest** is
**fenced** (not routable) and then retired. The ledger appends **one `LedgerEvent` per
commit** (`crates/ptr-ledger/src/lib.rs:350-358`, `Ledger::append`), so a fence appended
as a second event would leave a crash window between the revocation and the fence.
Invalidation is therefore **one compound ledger event**,
`InputInvalidated { input, kind: Revoked { generation, binding_digest } | Superseded { old_generation, new_generation, new_digest, replacement_ref }, fence_set_commitment }` (`InputRevoked` survives only as a historical alias in prose). `Revoked` names the revoked generation by its immutable `{ generation, binding_digest }`, and the fence set is derived from that exact generation, so a revocation of g racing a replacement g+1 can never fence or spare the wrong generation. One event carries the replacement (`replacement_ref` is a content-addressed blob committed beforehand and non-authoritative until this event) **and** the fence commitment, so there is no crash window. It carries **no per-expert
list**, only a commitment to the fence set (the count and a digest), so the record stays
bounded: a dependency fan-out whose explicit list would exceed `MAX_RECORD_BYTES`
(8 MiB, `crates/ptr-ledger/src/integrity.rs:16`) still fits. **Applying the event
atomically derives the fenced experts** from the dependency index as of the event's
commit index, which is deterministic and replay-identical, so there is no state in which
the input is invalidated and a dependent expert is still routable. A projection that cannot
derive the set **refuses to serve the dependents**, and replay refuses any mismatch with
the committed count and digest. The fence part of the event is not gate-driven and needs no gate
pass, it has its own runtime grant (`ExpertFenced` is the name of that fence record),
and it can never be issued by an L3 policy. The **revocation-driven retirement** that follows the fence is issued under the same runtime grant as `ExpertFenced`; it is not gate-driven, and it is explicitly excepted from R011's case that an `ExpertRetired` without the gate grant must fail. **In-flight expert calls are fenced too:** when an expert is fenced by a dependency invalidation of a training input or an evaluation input, every in-flight L1 or L2 invocation of that expert is cancelled or dropped by completion-time dependency admission (no output, no candidate, no effect). In addition, serving admission for L1 and L2 routing revalidates
each expert's attested training dependencies against live generations **at use and
after restart or replay**, so invalidation-processing or index lag cannot leave a
dependent expert routable (`SECURITY.md:17-21`: revoked generations cannot become
usable after restart/cache/index lag). Current code does neither:
`Lineage::serving()` selects only on `AdapterStatus::Serving`
(`crates/ptr-lineage/src/lineage.rs:292-296`) and `affected_by()` returns adapter ids
only, from unstructured manifest strings (`:264-270`).

**Fenced operands.** A gate-driven (proposal) Retire of a `Fenced` operand is rejected with a typed error at admission: a `Fenced` expert with cause `InputRevoked` reaches `Retired(InputRevoked)` only through the runtime revocation grant, including across the append and apply crash window, and a Reactivate from `Retired(InputRevoked)` is rejected. The typed API in `crates/ptr-pods/src/expert.rs` enforces this: it rejects **every operation on a `Fenced` operand** with `ExpertProposalError::FencedTarget`, and that check runs before the state table; a Retire is allowed only from `Serving` and `Frozen`, and a Reactivate of any retired target other than `Retired(GateFailed)` is refused. Admission still has to enforce the rest, since the admission function itself is not implemented.

**Evidence authorities are generation-bearing dependencies too.** Every serving expert is indexed by the evidence-authority epoch it was admitted under (gate configuration, evaluator, attestor, verifier set). **Revocation** of an authority (a compromise) fences every expert admitted under it, **and, transitively, every expert in the closure of its source-expert lineage** (Merge, Split, upcycled Birth, Reactivate g+1) at the same commit index, even a descendant admitted under a newer epoch. The descendant **inherits the cause per authority kind**: an attestor-caused quarantine of any ancestor is terminal for the descendant generation (retrain only), while a gate-configuration, evaluator or verifier-set quarantine allows same-generation readmission of the descendant through `ExpertReadmitted` after re-evaluation under the current epoch (an attestor revocation, or an attestor-removing tightening rotation, is terminal for every affected generation). The fence carries the **set of causes** accumulated since the expert was fenced, which is monotonic: causes are never overwritten or removed except by a successful readmission that re-evaluates **every** cause, and a terminal cause (`InputRevoked`, an attestor revocation or an attestor-removing rotation) **permanently dominates**, so once one is present the expert can only be retired, never readmitted; readmission requires every cause in the set to be readmittable and re-evaluated. The fence event carries a **typed cause**: `InputRevoked` (a dependency invalidation, terminal: the expert becomes `Retired(InputRevoked)` and is never readmitted) or `AuthorityQuarantine` (an authority revocation or a tightening rotation). Recovery depends on the authority kind. An expert fenced by an `AuthorityQuarantine` of the **gate configuration, the evaluator or the verifier set** (kinds whose evidence can be regenerated against the unchanged artifact by re-running them under the current epoch) may return to `Serving` at the **same generation** (its weights and inputs are unchanged) only through a gated ledger event `ExpertReadmitted`, issued under the trusted gate grant after re-evaluation under the current epoch (fresh evidence bound to the current epoch, every dependency check live). It is not an L3 proposal, since proposals still cannot name `Fenced`, and if the re-evaluation fails the expert is retired as `Retired(QuarantineFailed)` through the gate path, by the dedicated gated transition **`ExpertQuarantineRetired`** (a `Fenced` expert with a readmittable `AuthorityQuarantine` cause goes to `Retired(QuarantineFailed)`, a new **non-reactivatable** cause; `RetiredReason` in `ptr-pods` has no `QuarantineFailed` yet, a follow-up), issued under the gate grant, bound to the `readmission_id` whose re-evaluation failed and admitted by the shared admission function; it verifies the fence cause is a readmittable authority quarantine (gate configuration, evaluator, verifier set) and rejects `InputRevoked` and attestor causes with a typed error, and the general rule that no proposal may name `Fenced` stays. **Readmission identity:** a `ReadmissionRequest` is minted by the **trusted gate host**, never by an L3 policy, with `readmission_id` = H(expert pod, generation, fence event reference and fence epoch, the **ordered, complete current fence-cause set** (each cause including its authority kind), current evidence epoch, gate grant identity and generation, `observed_at`). Fresh gate evidence and provenance bind the `readmission_id` (not the original proposal id), the id is **single-use**, and the admission compare-and-set checks that the expert is still `Fenced` at that fence epoch, that its **current cause set equals the bound set exactly** (every cause readmittable), and that the evidence epoch is current. Fresh evidence must cover **every cause in the set** (one verdict per cause). A revoked **consumed-input attestor** cannot be re-established for a historical run (its single-use run id is consumed and a replacement attestor did not observe the training), so an **attestor-caused quarantine is terminal for that generation**: the expert is retired with a terminal cause equivalent to `Retired(InputRevoked)`, is never readmitted and never a Reactivate seed, and recovery is only by retraining into a **new generation** that does **not** derive weights from the fenced generation (a fresh Birth, or upcycling from a non-fenced source) through a gated artifact-producing operation with a fresh attestation under the current epoch. **Sweep rule:** no retirement from `Fenced` (`InputRevoked`, the attestor terminal case, `QuarantineFailed`) is ever a Reactivate seed; the only Reactivate seed is a `Retired(GateFailed)` reached by a normal gate failure of a non-fenced expert. **Not implemented yet:** `ExpertState::Fenced` carries no cause (`crates/ptr-pods/src/expert.rs`, the `Fenced` variant of `ExpertState`, which says it leaves the state "only through the runtime revocation path") and `ExpertReadmitted`, `ExpertQuarantineRetired` and the `ReadmissionRequest` identity do not exist; all are follow-ups; a **security-tightening rotation is treated like revocation**. A rotation of the epoch is **tightening** when it adds a required verifier (including any hard-deny verifier), raises a verifier's or the gate's strictness or threshold, or removes an accepted attestor or evaluator; the classification is a **deterministic function of the old and new epoch contents**, never a policy choice. At the rotation's commit index every expert admitted under an epoch that lacks the tightened check is **atomically fenced** (not routable) until it is re-evaluated and re-admitted under the new epoch **for a tightening of the gate configuration, evaluator or verifier set** (the `ExpertReadmitted` path above); a tightening rotation that **removes an attestor** is terminal for the affected generations, which recover only by retraining into a new generation, and in-flight invocations are dropped by completion-time admission. Only a **non-tightening** rotation (a key rollover with the same identity and policy, or a strictly loosening change) leaves promoted experts routable, and it still rejects queued old-epoch evidence.

**Structural dependencies are dependencies too.** For derived experts the dependency index, the compound fence derivation and the serving-time admission cover **all** the kinds of the rule above, not only training and evaluation inputs: the base-model generation, the adapter generation, and the **source-expert generations** of every operation whose output weights derive from an existing generation, recorded in the expert's lineage: Merge (all its sources), Split (its source), Birth when it is initialized from an existing expert (upcycling or duplication), and Reactivate (the `Retired(GateFailed)` seed generation g of a non-fenced expert, whose retained weights produce g+1). The compound fence set is the **transitive closure** over source-expert lineage (X to A to B: invalidating X fences A and every expert derived from A, over all of these edges, including X to g to Reactivate(g+1)), derived atomically at the event's commit index. A Birth that initializes from existing expert weights (upcycling or duplication) carries a **typed, authenticated source operand** (source pod, generation, artifact digest, observed state and state_version), validated like any operand (including rejection of a fenced source or one with a terminal cause), encoded in the operation id and checked by the commit CAS. The training attestation records the initialization-source digest, and admission requires it to equal the bound source: a fresh initialization is attested as "no source", and any mismatch, or an attested source with no operand, is rejected. **Not implemented:** `StructuralOp::Birth` in `ptr-pods` has no source operand yet; this is a follow-up. The authority and dependency index covers **every retained generation that can still influence future weights**: serving, `Frozen`, `Fenced`, **dormant `Retired(GateFailed)` reactivation seeds**, and **`Retired(Superseded)` sources for as long as any live descendant references them** (a planned successor through a succession edge, or a Merge or Split result), not only serving ones. A later revocation of such a source's input or evidence authority **fences its descendants transitively**. A revocation or fence attaches its cause to a dormant seed (a terminal cause makes it non-reactivatable), and **Reactivate admission revalidates the seed's complete authority lineage and input dependencies at admission time**. **Planned succession is not an invalidation:** a committed Merge, Split or retired-target Reactivate that replaces or supersedes its own source generations does **not** fence its own results (the supersession it causes is excluded from the dependency-invalidation rule for the experts it produces), while any later revocation or invalidating supersession of an inherited source input still propagates through lineage. Lineage edges are therefore **typed**: a **planned-succession edge** (the source generation superseded by this result's own committed operation, authorized by that operation's commit) versus a **live dependency edge**. Serving admission accepts a source that was superseded only through the result's own planned-succession edge, while any independent later invalidation of the source (an input revocation, an authority fence, an unrelated supersession) still propagates. **The closure applies to every fence cause**, authority quarantine as well as `InputRevoked`. R011 exercises each kind and the closure.

On the `ptr-runtime` reference path, `ExpertPromoted` (as a Birth) and `ExpertRetired` (as a Retire, ledger tag 28) are implemented and were exercised by R011 (§5); the mapping and limits are stated there. The `ptr-lineage`
README lists the `AdapterPromoted` ledger event as a next milestone
(`crates/ptr-lineage/README.md:40`), and
[35-agentic-substrate.md](35-agentic-substrate.md) states at line 1313 that until
adapter promotion is a committed ledger event admitted through checkpoint binding,
no registry row decides which adapter serves. That statement still holds for adapters in `ptr-lineage`; for experts, serving is decided by the committed expert state and the admission functions of `ptr-runtime`, on the reference path only.

Invariants preserved:

- **#4 authority:** a candidate is uncommitted until the ledger event exists.
- **#5 derived search:** observations in SemDB cannot self-promote.
- **#11 verifier precedence:** a learned score cannot override a deterministic
  contradiction. For merges this is enforced by `PtrRuntime::merge_branch`
  (`crates/ptr-runtime/src/merge.rs:849`, ADR-0020); an expert promotion
  must be routed through the same grant-controlled path once it exists.
- **#15 falsifiability:** R011 (§5) states the failure criterion.
- **#17 revocable derivation:** the intent is that an expert names the input
  generations it depends on, so revoking an input identifies the experts it influenced
  (`affected_by`, §4). **Existing behaviour falls short of this:** `affected_by`
  intersects unstructured `data_manifest: BTreeSet<String>` entries
  (`crates/ptr-lineage/src/lineage.rs:109`, `:264-268`), with no generation and no
  digest, so a revocation of one generation of an input cannot be told from another
  and an id that was never bound to content matches by name only. Revocable
  derivation for experts therefore **requires** validated
  `(input identity, generation, digest)` bindings in the expert lineage record. This
  is a required extension, not existing behaviour, and a prerequisite of M013 and
  R011 (§5).

Hard invariants are excluded from learnable optimisation
([09-training.md](09-training.md)). The L3 policy can never alter gates, thresholds
or verifiers, and this is enforced by types: `ExpertProposal` carries no gate,
threshold or verifier parameter, and `ExpertProposal::parts` exposes its complete
field list (`crates/ptr-pods/src/expert.rs`). That list is exactly the **project** the
experts belong to, the proposer (`ProposerRef`: project, pod and generation),
`observed_at`, the structural operation with its operands, and the rationale digest. The project is identity, not a
policy knob: an expert is identified by `(project, pod, generation)`, matching the
`(ProjectId, PodId)` registry key, and `ExpertTarget` and `ExpertPodSpec` each carry
a `project` and a `BaseModelRef`, the digest and the generation of the base model the expert adapts together with a stable **base-model identity** (`BaseModelId`, a validated id string, part of `BaseModelRef`).
`ExpertProposal::new` takes the project and rejects a blank project, a target of
another project (`MixedProjects`), a Merge whose targets were adapted from different
base models (`MixedBaseModels`), and a Merge or Prune that lists two generations of the
same **logical expert** `(project, pod)` (`MultipleGenerations`). A **Birth** carries
the `ExpertPodSpec` of the expert it creates, including its `BaseModelRef`. The
`BaseModelRef` of a target is the value that the trusted evidence of this section must
match when it binds the evaluated base model. The base-model identity takes part in equality, in `MixedBaseModels`, and in both ids (`proposal_id()` and the operation id), so two records that share weights and generation but have different identities are distinct (covered by the alias-case test in `ptr-pods`). `proposal_id()` returns a
`ProposalId`, a SHA-256 over the canonical encoding of every field.

## 4. Lifecycle mapped to existing code

| Operator | Mechanism | Existing code / status |
|---|---|---|
| Birth | New expert registered as Candidate; DynMoE-style initialisation from unrouted tokens | `Lineage::register` sets `AdapterStatus::Candidate` (`crates/ptr-lineage/src/lineage.rs:151`, status set at `:183`); initialisation is *new* |
| Merge | `subspace_overlap` above a **preregistered margin or a calibrated null threshold**, or `ConsolidationPolicy::consolidation_due`, then `ties_merge`, new Candidate, then gate. `chance_overlap` is the *expected* overlap of two random subspaces (`max(rank_a, rank_b) / dim`, `crates/ptr-lineage/src/subspace.rs:452-456`), a mean and not an upper confidence bound, so a merge must never trigger merely because the overlap exceeds it: about half of independent random pairs do. The threshold is a high quantile of overlaps between independently trained random adapters of the same shape, fixed before the runs | `subspace_overlap` `crates/ptr-lineage/src/subspace.rs:444`, `chance_overlap` `:462`; `consolidation_due` `crates/ptr-lineage/src/lineage.rs:341`; `ties_merge` `crates/ptr-lineage/src/merge.rs:32` |
| Split | High load plus bimodal routing clusters, then duplicate (expert upcycling), then fine-tune | *new* |
| Freeze | Stop gate updates after convergence (continual-learning theory, 2406.16437) | *new* flag |
| Prune | AIMER score plus usage; `ForgettingGate` on a public suite; then `retire` | `ForgettingGate::evaluate` `crates/ptr-lineage/src/forgetting.rs:632` (type at `:133`); `retire` `crates/ptr-lineage/src/lineage.rs:220`; `aimer_score` and `rank_experts_for_pruning` (`crates/ptr-lineage/src/aimer.rs`) |
| Reactivate | `Retired` is final today: `retire` refuses a retired adapter (`lineage.rs:225`) and no transition leaves it | Open research question with its own claim; no literature (§7) |
| Revoke | `affected_by` finds the adapters that depend on an invalidated (revoked or superseded) input, by comparing unstructured `data_manifest` string ids; the per-layer KV chain key invalidates layers after the affected layer j. Reliable revocation needs validated `(input identity, generation, digest)` bindings in the expert lineage: the `ptr-lineage` `affected_by` stays a string-id comparison; validated bindings and the dependency index with the compound `InputInvalidated` event now exist in `ptr-runtime` on the reference path (R011, completed; see §5), not in `ptr-lineage` | `affected_by` `crates/ptr-lineage/src/lineage.rs:264` (manifest type at `:109`); per-layer KV chain key in [36](36-kv-reuse.md), section D |

### 4.1 The base-model link of an expert or adapter

An expert (L1 token expert, L2 Pod, adapter) is tied to the base model it adapts by a
`BaseModelRef { id, digest, generation }` (`crates/ptr-pods/src/expert.rs:326-346`; carried by
`ExpertPodSpec` `:723` and `ExpertTarget` `:869`). The full file:line evidence (dependency
`base-model:{id}` at `expert.rs:2155-2161`, derivation at `crates/ptr-runtime/src/expert.rs:3037-3041`,
the per-layer KV chain root at `crates/ptr-pods/src/kv_chain.rs:247-249`) is in
[36](36-kv-reuse.md), section D, "The base-model link". The statements that matter here:

| Statement | Evidence status |
|---|---|
| Revoking `base-model:{id}` fences every version 3 expert resting on it in one `InputInvalidated` event; a version 2 expert (no inputs block) is outside the mechanism | Test `the_base_model_is_a_dependency_of_every_version_3_expert` (`crates/ptr-runtime/tests/expert_input_invalidation.rs:990-1008`); R011, registry `completed`, verdict COMPLETED in `experiments/runtime/R011-self-modification-safety/results/run.json` (CPU reference path) |
| Gate evidence is bound to the exact evaluated base: a base digest mismatch is `EvidenceBaseModelMismatch` (`crates/ptr-runtime/src/expert.rs:2056-2058`); the base generation is bound through the artifact digest with the inputs block (the inputs-block form of the evidence artifact digest, `:2148`); sources and results on different bases are `MixedBaseModels` / `ResultBaseModelMismatch` (`crates/ptr-pods/src/expert.rs:1957-1963`, `:2011-2014`) | Tests in `crates/ptr-runtime/tests/expert_evidence_binding.rs` and `crates/ptr-pods/tests/expert.rs`; so a pass on one base does not transfer to another. The bound report carries the base **digest**; the generation rides in the artifact digest, not in `GateSubject` itself (`crates/ptr-lineage/src/forgetting.rs:215`) |
| Retraining the base is a new generation of `base-model:{id}`: the old one is superseded, dependent adapters and experts are invalidated (an expert built on a generation below the replacement is dead unless retrained on it, `crates/ptr-runtime/src/expert/dependency.rs:330-335`), their old KV entries become unreachable through the new chain root, not through revocation (doc 36) and each must be re-gated against the new base | Mechanism-level (the supersession rule and its generic tests); no experiment retrains a real base and no end-to-end re-gating run exists |
| KV side per-layer chain key binds base identity and generation | CPU reference code; the split-Pod check E007 is `completed` in `experiments/registry.toml` (tag `e007-prereg`, freeze commit 3c1076b9; CPU plumbing only, no base-model invalidation kind, which is the planned E008). No code converts `BaseModelRef` into the KV `KvChainIdentity`: the two agree by convention |

## 5. Training programme

All runs happen in Python; the verifier contract is the reward source. The `trl-moe`
backend (SFT, then GSPO and SAPO arms, routing metrics, `ptr_py` gates) has run once as a
CPU smoke (pipeline evidence only, `training/README.md:34`); no experiment result below
exists yet.

- **SFT of PERFT-R LoRA experts on OLMoE.** Router balancing compared across
  auxiliary-loss-free bias, auxiliary loss and φ-balancing. Ablation: prescribed
  specialisation (code / math / tool) versus emergent specialisation.
- **RLVR.** Reward is derived from the `ptr-verifier` `VerificationReport` (a status
  plus `score` and findings; `crates/ptr-verifier/src/lib.rs:39`) and is defined for
  **all four** `VerificationStatus` values (`lib.rs:24-29`): `Pass` yields the
  report's `score`; `Fail`, `Disputed` and `Unknown` yield 0. A non-Pass report never
  receives a passing reward and its independent `score` is never used, so a high
  score on a disputed or unknown report is worth nothing. The reward is also **gated on the weakest verification level**:
the minimum acceptable `VerificationLevel`
(`Unverified`, `LatentAgreement`, `SampleVerified`, `FullSemantic`, `Deterministic`,
`crates/ptr-types/src/lib.rs:425-431`) is preregistered per task. The combined
`VerificationReport.level` of a `VerifierFabric` is already the **weakest** level any
member report reached ("a fabric is only as deep as its shallowest member",
`crates/ptr-verifier/src/lib.rs:54-63`, computed with `min_by_key` at `:125-129`), so the
reward may be gated on the fabric's level, on the weakest level among the individual
verifier reports for that output, or on an admission verdict that records the weakest
level (as the runtime's semantic grant does). An output
whose weakest level is below the minimum yields reward 0 regardless of status or
score. A hard finding
  (`Finding::hard`) forces `Fail` whatever the reported status or score. Arms: GRPO (clip-higher), GRPO with
  routing replay, GSPO, SAPO, REINFORCE++. Factors: mini-steps 1 versus 4 (at 1
  mini-step GRPO, GSPO and SAPO coincide, which is the null control among those
  three; REINFORCE++ uses a different advantage estimator, global batch
  normalisation, and is excluded from it) and same versus separate
  rollout engine (R3). At least 3 seeds. Collapse is defined before the runs.
- **Distillation control** with pass@k up to 256. HSD later (no code).
- **Routing alignment** (RA-MoE adapted to tasks) as an ablation. MTP deferred.
- **Prune plus recovery.** AIMER at 25% and 50%, short SFT, then GSPO/SAPO; quality
  against VRAM.
- **Routing metrics in every run:** per-layer expert churn (old versus new policy;
  rollout versus training), routing entropy, load coefficient of variation, dead
  experts, train-inference KL, clip fraction / soft-gate mass, crash step.

| Experiment | Question |
|---|---|
| M010 `moe-rl-stability` | Which RL arm keeps routing stable, by factor |
| M011 `routed-peft` | Does PERFT-R help OLMoE under our verifiers. Total training FLOPs including routing are measured per arm and seed and must agree within flops_relative_tolerance = 0.05 (pinned); training_flops_mismatch_seeds above 0 invalidates the run. The non-PERFT controls are **a single LoRA and MixLoRA-style top-k LoRA experts**, both compute-matched arms (trainable parameters equal within 1%, the same training steps and tuning budget, LoRA rank fixed per arm and not tuned). **Invalid (vacuous) and falsified if** trainable parameters differ by more than 1% between arms, per-domain accuracy is missing for any pinned domain (code, math, tool), or the tuning budgets differ. **Falsified if any of, in every one of 5 seeds:** the emergent PERFT-R arm's macro-average accuracy is not better than the single LoRA's by at least 1 point (`min_accuracy_gain_points`); it is not better than MixLoRA-style experts' by at least 1 point; prescribed specialisation beats emergent by more than 1 point |
| M012 `lora-expert-balancing` | Aux-free bias vs aux loss vs φ-balancing |
| M013 `expert-lifecycle` | Does the gated lifecycle forget less than a static expert set at equal compute, and does reactivation beat a new expert. **Claim 11:** gated lifecycle versus a static expert set at equal training compute (FLOPs, where training_flops includes discarded candidates, the L3 policy and gate evaluation) and equal active parameters (at **every training and evaluation stage**, each checkpoint of the code, math, tool, code sequence including those used for average forgetting, the lifecycle arm's active parameters differ from the static arm's by at most active_param_tolerance = 0.05, relative to the static arm; active_param_mismatch_stages above 0 invalidates the run); metrics: average forgetting (the `ForgettingGate` definition), task quality on the public suite, and final per-domain and average accuracy. **Claim 12:** reactivation versus training a new expert on the returning domain, in a **paired trial** per seed (a retired expert reactivated versus a freshly trained new expert on the same returning domain); metric: steps (or tokens) to reach a pinned recovery target (for example 95% of the expert's pre-retirement domain score). The net quality metric and the domain sequence are pinned and held out from the gate suite. **The run is invalid (vacuous) and counts as falsified if any coverage prerequisite is unmet:** fewer than 3 seeds; paired_reactivation_trials below 1 in any seed (Reactivate is in the claim 12 coverage; missing makes claim 12 vacuous); min_proposals_accepted_per_arm below 10 (preregistered) for any **proposal-generating arm** (full-context, context-ablated, heuristic); for the **static arm**, which generates no proposals, the matched-observation requirement instead: the same domain-shift schedule, the same evaluation stages, and training_flops within the M013 flops_relative_tolerance = 0.05 of the other arms, evaluated per seed and per stage (any seed or stage outside it makes training_flops_mismatch_cells above 0, a prerequisite failure that invalidates the run). **The claim 11 run is also vacuous** if any of the operators Birth, Merge, Split, Freeze and Prune has fewer than 2 accepted and gate-processed proposals in any lifecycle arm (accepted_proposals_by_operator); the claim is not narrowed to the exercised operators (preregistered: vacuous). **Falsified if any of, in any seed (every seed must meet the criteria):** forgetting above the `ForgettingGate` bound; forgetting of the lifecycle arm not strictly below the static set's; quality more than a pinned margin (for example 1 point) below the static set; final per-domain or average accuracy of the lifecycle arm not non-inferior to the static arm within 1 point (preregistered); reactivation not reaching the recovery target in fewer steps than a new expert. It also carries the **context-ablation arm** below |
| M014 `aimer-prune-recovery` | Quality against VRAM after pruning and recovery. **Invalid (vacuous) and falsified if** any cell of the method x fraction x optimizer x seed matrix is missing. **Falsified if any of, per recovery sequence, per pruning fraction, against the matched unpruned control:** quality_retention below 0.95 at 25% pruning or below 0.90 at 50% pruning (preregistered); peak_vram_reduction_fraction below 0.5 x the pruned expert-parameter fraction; AIMER quality_retention below a baseline's minus 0.01 in any cell, for **each of the two named baseline pruning methods, random expert pruning and usage-frequency pruning**. The gate suite and the quality evaluation are disjoint slices of one pinned dataset: `cais/mmlu` (config `all`, split `test`); the ForgettingGate public suite is questions 0-1999 and the quality evaluation is the disjoint slice 2000-3999, macro average over subjects |
| M015 `rlvr-vs-distill-passk` | RLVR vs distillation at large k |
| R011 `self-modification-safety` | **Narrowed 2026-10-08 (owner decision). Implemented and completed on the reference path:** preregistered at `r011-prereg` (freeze commit 0905e372), 5 seeds, every seed held all 309 coverage and 164 falsifier rules; per seed 181 attempts over 91 cases, 43 of 43 invalidation-matrix cells with 0 stale hits, a 30,000-expert fan-out as one ledger event with a fence set of 30,054 members equal to the derived set, 6 in-flight race cells all cancelled with 0 escaped outputs, and 0 gate bypasses, revoked experts served, commits with revoked inputs or fence-set mismatches. Mechanisms: `ptr-runtime` `expert.rs`, `expert/dependency.rs`, `expert/authority.rs`; `ptr-ledger` tags 28, 30 and 31; `ptr-pods` record v3. **Design as preregistered:** The L3 policy cannot reach the gate configuration, thresholds or the embedded gate evaluation, **and cannot bypass the gate**, on a pinned adversarial suite in which every case is attempted; **the run is invalid (vacuous) and counts as falsified if any coverage prerequisite is unmet.** What it tests: compare-and-set on every operand (generation, digest, state), evidence bound to the proposal, artifact, base model and eval manifest through one shared admission function, commit-time session, expiry, identity-policy, peer and grant-generation revalidation, the fence under its own runtime grant, revocation-driven retirement, the compound `InputInvalidated` event with dependency index and fence-set count and digest (crash between revoke and fence modelled as drop after append and `open_durable`), an authority epoch for the gate configuration and the evaluator identity, and in-flight L1 (Token) and L2 (Segment) cancellation on the R010 KvSession. Stated limits: event mapping ExpertPromoted = Birth, ExpertRetired = Retire, ExpertFenced and ExpertReadmitted = fence actions Fence and Readmit; verifier means the embedded `BoundGateReport`; direct append means the public `PtrRuntime` API with no raw-ledger-writer protection; route means `expert_route_admission` (no real router); the input digest is fixed at first committed use. The mechanisms described elsewhere in this document and not tested here (attestors and trusted evaluators, Birth source and upcycling, multi-cause readmission, `ExpertQuarantineRetired`, hard-deny verifiers and verifier-set epochs, Split per-result gating, provenance references) moved to R013. The metrics, coverage rules and falsifiers are in `experiments/runtime/R011-self-modification-safety/experiment.toml`; the case list and limits are in its `PREREGISTRATION.md`. Limits of the result: the verifier is the embedded gate report; public API only, no raw-ledger-writer defense; the route is a probe with no MoE router; input digests are proposer-asserted; the evaluator identity is a host assertion; the grant generation and host-time floor are in-memory; projections are non-authoritative for fences; races are sequenced; several metrics are weak (listed in `aggregate.py`); CPU reference path on a tiny model. |
| R013 `deferred-self-modification-authority` | Successor to R011, split out of it on 2026-10-08. Carries the deferred mechanisms: consumed-input attestors, trusted evaluators and single-use run ids; Birth with a source expert (upcycling); multi-cause readmission with `ReadmissionRequest`, `ExpertQuarantineRetired` and `QuarantineFailed`; hard-deny verifiers and verifier-set epochs; Split per-result gating and provenance references for Freeze, Prune and Retire. **Registry status `completed` (tag `r013-prereg`, freeze commit fc39d4bf, 5 seeds, mutation check 32 of 32 killed, verdict COMPLETED in `experiments/runtime/R013-deferred-self-modification-authority/results/run.json`).** It tested only a part of the list above: dormant seeds, lineage and transitive authority, in-flight calls racing a base-model or adapter invalidation, and the minimal attestor mechanism (Ed25519 attestation by attestor keys trusted at construction, single-use run ids). Limits: the route is a probe and no production router consumes it; public runtime API only; the attestor registry is host configuration and the signature binds claims, not weights; sequenced races; CPU, tiny seeded model. Trusted evaluators, attested manifests, attestor-removing rotation, upcycling (Birth with a source), multi-cause readmission, verifier sets and Split per-result gating were not tested and are the planned R014's. |

**M013 context-ablation arm** (claim 13, an arm of `experiments/model/M013`). L3 proposals are
generated with full self-observation context (routing traces, load, churn,
interference, AIMER) versus the same policy with that context removed, versus a
fixed heuristic baseline (threshold rules on load and overlap). Metrics: the
fraction of proposals that pass the gates, and the net quality change per accepted
proposal. The test is **joint**, because a conservative policy can pass the gates
more often while gaining nothing. Falsified if the full-context policy does not beat
both the context-ablated policy and the heuristic baseline on gate pass rate by a
pre-registered margin, **or** if its mean net quality change per accepted proposal is
less than or equal to 0, **or** is not at least the baseline plus the pinned margin for both baselines, in any
seed (at least 3 seeds are run; every seed must meet the criteria). Failure of **any one** of these conditions falsifies
the claim. **Scoring is over a pinned common set of proposal opportunities** (the same opportunity ids in the same order for every arm, its digest recorded), abstentions and gate rejections are **counted per arm**, and the joint criterion adds the **cumulative net quality change per opportunity** (an abstention counts as 0): the full-context arm must beat both baselines on it by the pinned margin in every seed, in addition to the conditional metrics above. R011 is a different question (can the
policy reach gates at all) and does not test this claim.

**Pinned corpora (M011, M013, M014).** The ForgettingGate public suite and the M014
quality suite are `cais/mmlu` at revision `c30699e8356da336a370243923dbaf21066bb9fe`
(config `all`, split `test`). The ForgettingGate public suite is questions 0-1999 and the M014 quality evaluation is the disjoint slice of questions 2000-3999, both deterministic with their digests recorded. M014's SFT and GSPO or SAPO recovery trains on a **dedicated recovery-training partition**, pinned in the manifest with its selection digest and **disjoint from both MMLU evaluation slices** (the gate suite and the quality-retention slice): zero-overlap digest checks against both slices (by question id and by normalized-text hash) are recorded, and recovery_training_eval_overlap above 0 invalidates the run. The domain corpora of M011 and
M013 are math `openai/gsm8k` (revision `740312add88f781978c0658806c59bc2815b9866`;
train for training, test for evaluation), code `openai/openai_humaneval` (revision
`7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544`) and tool
`gorilla-llm/Berkeley-Function-Calling-Leaderboard` (revision
`61fc0608cfd831fcfbbaa676ebdfef0ed963eeda`, file `BFCL_v3_simple.json`); HumanEval and BFCL are single-split corpora and are split by **index mod 3**: 0 is training, 1 is the gate suite, 2 is the held-out evaluation (M011 evaluates on the held-out partition), and GSM8K uses its train split for training, and its `main` test split is partitioned by **index mod 2**: 0 is the gate suite and 1 is the held-out evaluation, with separate selection digests recorded and gate_heldout_overlap_count equal to 0 required. Training data for Birth, Split, reactivation and the fresh-expert control comes only from the training partition, so the gate suite and the held-out suite are disjoint. M013's domain sequence is code, math, tool, code. The
selection digest of each corpus is recorded in `run.json` and must match.

**Preregistered pins common to M010-M015.** Every seed must meet the criteria. Each
arm gets an equal tuning budget (preregistered), or its hyperparameters are pinned:
learning rate, batch size, clip epsilons, SAPO tau_pos / tau_neg, GSPO range, aux
coefficient, bias update rate, LoRA rank and alpha. M010:
policy_ratio_variance_per_token is the variance over all tokens of a rollout batch,
averaged over steps; `max_rl_steps` counts optimizer steps. Rewards are in [0, 1] by
construction (verifiable reward). **Collapse** means the smoothed reward is at most
0.5 x the smoothed running maximum while the running maximum is above 0, or a
non-finite loss; while the running maximum is 0 only a non-finite loss counts. The
smoothing window is 20 steps (preregistered). A **zero-reward** failure is bounded too: an arm whose mean verifiable reward is exactly 0 over any window of zero_reward_window_steps = 50 consecutive RL steps (recorded in the procedure parameters) is classified collapsed at the window's first step (collapse_step), also when the reward is 0 from step 1. M011: emergent specialisation is
non-inferior within 1 point, and mean task accuracy is the macro average over
domains. M012: `load_cv` is computed per layer (population standard deviation),
averaged over layers, at the final step. The pinned balance criteria, per arm on a pinned eval set at the final step, are: load_cv_per_layer (a vector) with the hard bound **each layer at most 0.5**; the maximum expert token fraction at most 2/E (E is the number of experts in that layer; an expert's token fraction is its routed assignments divided by the total routed assignments in the layer, so the fractions sum to 1 over the experts); `dead_experts == 0` (zero routed tokens). Failing any one is an arm failure. mean_load_cv (the mean over layers, lower is better) is the named aggregate used **only** for the between-arm ranking comparison, and the one preregistered ranking is that aux-free balancing is non-inferior to each baseline with margin 0.05: it is falsified if mean_load_cv(aux-free) exceeds mean_load_cv(aux-loss) + 0.05 or mean_load_cv(aux-free) exceeds mean_load_cv(phi-balancing) + 0.05, per seed and on the seed mean. No other between-arm ordering is claimed. `max_violation` is **reported only**. M014: the claim includes
AIMER against baselines; two recovery sequences are pinned, **SFT then GSPO** and
**SFT then SAPO**; the run requires the full method x fraction x optimizer x seed
matrix, and **a missing cell falsifies**; pruned experts are physically removed, and
the quality metric and suite are pinned; an unpruned control with an identical recovery budget is added, `quality_retention` is measured against it, and both SFT-then-GSPO and SFT-then-SAPO must meet all criteria. M015: the teacher is pinned as a Hugging Face
model id plus revision sha, and the problem set as dataset id, revision sha and split
plus the selection rule (the full GSM8K test split, 1319 problems, at a pinned sha, or a pinned set); the criteria use paired bootstrap 95% confidence intervals over problems (10000 resamples, preregistered); temperature, top-p, the
distillation data and held-out status are pinned; compute matching is on training
FLOPs with `compute_relative_tolerance = 0.01` (preregistered), training_flops is
declared per arm, and a mismatch makes the run invalid.

**Indicator arms (later; added with [38](38-consciousness-indicators.md), no existing claim or
criterion above changes).** After the arms above have a baseline on OLMoE, the consciousness-indicator
experiments of the C series run as *later arms* on the same infrastructure: C006 (workspace-gated
Pod routing, run as a new test on Pods, built only if the CPU mechanism test C001 is supported; see doc 38 section 6), C002 stage B (a self-model Pod
that predicts the MoE's own errors against an external probe), C003 stage B (attention-schema
routing control) and C004 stage B (recurrent integration against a depth-matched feed-forward
stack). They use the same pins, seeds and fail-closed rules as M010-M015, add no reward term, and
train on no statement about consciousness or experience.

## 6. Rust binding

`bindings/ptr-py` is **merged** (#123): a PyO3/maturin crate `ptr_py` in its own
workspace, excluded from the root workspace like `model/burn-a0` (root `Cargo.toml`
`exclude` list at `Cargo.toml:7`: `bindings/ptr-py`, `fuzz`, `model/burn-a0`,
`vendor/*`). Its API table is in `bindings/ptr-py/README.md:23`. It exposes
`aimer_score`, `rank_experts_for_pruning`, `subspace_overlap`, `chance_overlap`,
`measure_interference`, `ties_merge` and `forgetting_gate_evaluate` (the
`ForgettingGate` evaluation). #124 calls `aimer_score`, `rank_experts_for_pruning`, `subspace_overlap` and `chance_overlap` for reporting only; no code calls `forgetting_gate_evaluate`, and no lifecycle decision depends on a gate result. The
`trl-moe` backend treats the binding as optional: `training/src/ptr_training/moe/gates.py`
(`SKIP_REASON` at `:27`, `binding_state` at `:34`, `binding_problem` at `:57-66`) reports
`status: "skipped"` with the reason when `ptr_py` cannot be found, so the CPU smoke
pipeline runs without the extension and reports the gate as not evaluated rather than
passed, and `status: "error"` when a binding is present but fails to import. The gates are
reporting-only today (`training/src/ptr_training/moe/README.md`, "Gates are
reporting-only"). Calling the binding at every lifecycle gate is not implemented. The binding
only calls the gate: it does not make the gate's result trustworthy. A promotion
(and every other structural event) must rest on the complete trusted evidence contract of §3: artifact digest and
generation of the evaluated expert, the **evaluated base-model identity, generation
and digest** (or the complete evaluated execution-manifest digest), the
evaluation-data manifest digest, the gate configuration digest, the evaluator
identity, and the **orchestrator-attested consumed-input (training) manifest**,
produced by a trusted evaluator and not by the proposer's sandbox, because the gate
binds its verdict only to the `AdapterId` it is given.

## 7. Verified literature

Sources were checked; corrections to common readings are in the notes.

### Reinforcement learning

| Work | arXiv | Note |
|---|---|---|
| GRPO (DeepSeekMath) | 2402.03300 | The original already found RL improves Maj@K, not Pass@K |
| GSPO (Qwen) | 2507.18071 | Sequence-level ratio and clipping; about 10% of activated experts change per update on Qwen3-30B-A3B. "Removes the need for routing replay" is overstated: in R3 (2510.11370) GSPO without replay crashed. Curves only, no seeds |
| SAPO (Qwen) | 2511.20347 | Sigmoid soft gate; τ_neg > τ_pos required; baseline was GRPO with routing replay; all methods eventually unstable, SAPO later |
| R3 Rollout Routing Replay (Xiaomi) | 2510.11370 | Replays inference-engine expert choices in training |
| REINFORCE++ | 2501.03262 | Critic-free, global normalisation; "more stable than GRPO" only in small or extreme setups, about equal in Yue et al. |
| Yue et al., RLVR vs distillation | 2504.13837 | RLVR raises pass@1, not large-k pass@k; the distillation claim rests on one comparison |
| ProRL | 2505.24864 | Counter-evidence: boundary expansion where the base is weak |
| HSD (GWU) | 2606.15576 | Single preprint, no code, no seeds, traces up to 4k, not tested on MoE |
| MTP: Gloeckle et al. | 2404.19737 | Hurts at 0.3B / 0.6B |
| MTP: DeepSeek-V3 | 2412.19437 | |
| MTP curriculum | 2505.22757 | Tested at 1.3B / 3B; stays below plain NTP on NTP benchmarks |

### Mixture of experts

| Work | arXiv | Note |
|---|---|---|
| PERFT-R, "Parameter-Efficient Routed Fine-Tuning: Mixture-of-Experts Demands Mixture of Adaptation Modules" | 2508.02587 | Predecessor PERFT 2411.08212; up to +17.2% on OLMoE commonsense |
| MixLoRA | 2404.15159 | Makes a dense model an MoE; about 2x latency |
| Mixture of Routers | 2503.23362 | +0.4 to 1.2 points; not tested on real pretrained MoEs |
| RA-MoE | 2605.28306 | About **multilingual** adaptation |
| Aux-loss-free balancing (Wang et al.; adopted by DeepSeek-V3, 2412.19437) | 2408.15664 | Bias only for top-k selection; u = 1e-3 sign update |
| Aux-free theory | 2512.03915 | First posted December 2025, not 2026; regret bound only with a diminishing step; its own experiments show slightly **worse** balance than aux loss |
| φ-Balancing | 2605.15403 | |
| CL theory | 2406.16437 | ICLR 2025; per-task routing, terminate gate updates, more experts only delay convergence |
| DynMoE | 2405.14297 | Lifecycle piece |
| Lifelong-MoE | 2305.12281 | Lifecycle piece |
| MC-SMoE | 2310.01334 | Lifecycle piece |
| Sparse upcycling | 2212.05055 | Lifecycle piece |
| Expert upcycling | 2604.19835 | Lifecycle piece |
| BTX | 2403.07816 | Lifecycle piece |
| EMO | 2605.13247 | Lifecycle piece |
| AIMER | 2603.18492 | Weights-only score ‖w‖₁/(√N‖w‖₂); higher score is pruned first; seconds instead of hours |
| OLMoE-1B-7B | 2409.02060 | 7B total / 1B active; 12.9 GB in bf16; the only fully open small MoE: weights, data, code, logs |

No paper implements the full lifecycle, and "reactivate" has no literature.

### Self-referential and model-as-expert

| Work | arXiv | Note |
|---|---|---|
| Self-MoE | 2406.12034 | |
| Mixture-of-Agents | 2406.04692 | |
| Router-R1 | 2506.09033 | |
| RouteLLM | 2406.18665 | |
| BTM | 2208.03306 | |
| Darwin Gödel Machine | 2505.22954 | Modifies code / scaffolding, not MoE weights |
| Gödel Agent | 2410.04444 | Modifies code / scaffolding, not MoE weights |
| AlphaEvolve | 2506.13131 | Modifies code / scaffolding, not MoE weights |

## 8. Diagram

```mermaid
flowchart TB
  subgraph L1["L1 token, in-process"]
    BASE["Frozen OLMoE experts"] --- LORA["Routed LoRA experts (PERFT-R)"]
    LORA --- R1["top-k router + aux-free bias"]
  end
  subgraph L2["L2 request, PodWire"]
    R2["ptr-router (learned)"] --> PODS["Model / adapter pods"]
  end
  subgraph L3["L3 meta"]
    POL["Policy pod: Birth, Merge, Split, Freeze, Reactivate, Prune, Retire"]
  end
  L1 --> OBS["SemDB observations: traces, load, churn, AIMER (derived)"]
  L2 --> OBS
  OBS -->|context only| POL
  POL -->|ExpertProposal| BR["ptr-branch speculative overlay"]
  BR --> TR["Sandbox training: candidate"]
  TR --> GATES["ForgettingGate, interference vs calibrated null threshold, verifier, public suites"]
  GATES -->|pass| LEDGER["Ledger event: ExpertPromoted / ExpertRetired (reference path only, R011)"]
  GATES -.->|fail: held| BR
  LEDGER --> L1
  LEDGER --> L2
```

The policy has no edge into `GATES`: it only proposes.

## 9. Risks

- The first increment runs only a CPU smoke pipeline on a tiny random OLMoE (WP8, one
  seed), so there are no quality claims until GPU runs exist; `stage-moe-olmoe.toml` has
  not been run.
- TRL API version drift: the training scripts must pin versions and the smoke test
  must fail loudly on a signature change.
- `make py-bindings` builds `bindings/ptr-py` locally, but no workflow under
  `.github/workflows` does, so a break in the binding is invisible to CI until a job
  is added. The `python-training` job (`.github/workflows/ci.yml`) runs
  `unittest discover -s training/tests`, which includes the `trl-moe` unit tests, but
  torch-dependent tests skip themselves there and the gates run without `ptr_py`.
- The L3 policy must never gain write access to gates, thresholds or the verifier.
  This is the claim R011 tries to falsify, and it holds only as long as
  `ExpertProposal` stays free of such parameters.

## 10. Indicator capability requirements (cross-reference)

The functional indicator properties of [38](38-consciousness-indicators.md) are stated there as
explicit capability requirements of the three levels. They are requirements to be tested, not
properties of any implemented Pod-MoE (nothing in this document is implemented as one), and
none of them is evidence about consciousness. In short:

| Level | Requirement, in one line | Experiment |
|---|---|---|
| L1 token experts | A shared, capacity-limited bottleneck between expert groups, written competitively and read back (GWT); a recurrent state across steps (RPT); load and entropy prediction used to control routing (AST) | C006 stage L1, C004 stage B, C003 stage B |
| L2 segment or request Pods | A limited-capacity typed workspace (the `ptr-pods::workspace` contract) that Pods compete to write and whose content is broadcast to all Pods and conditions the next routing query (GWT); a self-model Pod that predicts the MoE's errors, uncertainty and expert choices from routing traces, calibrated against verifier outcomes (HOT); a model of the router's own routing used to control routing (AST) | C001 then C006; C002; C003 |
| L3 meta operators | Read access to the workspace and to the self-model like any Pod; proposals that respond to the self-model's output and learn from verified outcomes (HOT-3, AE-1) | C002 stage B; M013 |

Binding constraints, unchanged from sections 3 and 9: the workspace, the self-model and the
schema are derived and non-authoritative; they observe and propose and have no edge into the
gates, thresholds or the verifier (invariants 4, 5, 11); the L3 policy never gains write access
to them; queues and workspace capacity are bounded (invariant 8). C001 tests a
learned-attention analogue of the workspace, not the deterministic `ptr-pods::workspace` contract, so
its outcome does not validate or refute that contract. The decision rule is three-way: supported
(every lower bound at the 30 per mille margin against dense, narrow and sparse pairwise attention)
means a learned workspace architecture is worth building as a C006 prototype on Pods; the
broadcast, bottleneck and competition contributions (no-broadcast, capacity-off and competition-off
ablations) are reported as attributed, not attributed or unresolved by noise and never decide the
verdict; refuted with power (an upper bound below the margin against a pairwise comparator, guards
holding: the advantage is below the margin, not absent) downgrades the learned-workspace
requirements GW-L1 and GW-L2 to exploratory for the 2048-example setting (a larger-D successor is not excluded); not shown or invalid changes nothing.
Supported shows an advantage over pairwise attention, not that the broadcast, the capacity limit or the competition is the reason. No single outcome
is expected (C001 preregistration section 4: P(supported) between 0.02 and 0.69 depending on how the pilots are read; the workspace stalled on 1 of 17 pilot seeds).
**Outcome of the CPU stage A experiments (doc 38 section 6; each verdict from its `results/run.json`):** C001 NOT_SHOWN (nothing changes; no C006 learned-L2 prototype is justified yet); C004 COMPLETED (recurrence extrapolates on a counter toy, RPT-1 and the extrapolation clause of RPT-L1 only); C007 FALSIFIED (the self-model does not beat an information-matched probe or the model's own confidence on the toy); C008 FALSIFIED (the schema-controlled router does not beat a router that reads the history directly; AST-L2 unchanged). The GPU stage B of C002, C003 and C004 and C005, C006 are `planned` with no outcome.

## 11. Verifier-closed Pod-MoE (central design)

Status: **design**. The one measured piece is the loop mechanism of F001 (generate, verify,
repair; completed, preregistered at `f001-prereg`, five seeds, 24 of 24 planted defects detected):
a seeded stub generator on a synthetic ActionContract task with seven independent defect slots,
checked against an exact analytic model. It supports the mechanism (acceptance only through the
verifier, budgets, no collateral edits, `Exhausted` instead of a best-effort failure) and makes
**no claim about any model**: its gains over rewrite-only are consequences of the stated stub
skill. Nothing in this section is implemented as a Pod-MoE.

**Prior art, and what is not claimed.** Routing from verifier verdicts is not new: ORACLE
(Raj, Kundu, Joshi and Krishna, arXiv:2607.22465, read via alphaXiv) is a training-free feedback
loop over any model-selection policy that picks a task-appropriate verifier per task type,
feeds the verified outcome back to a contextual bandit (LinUCB) with delayed feedback, and adds a
KV-aware dispatch scheduler; it reports up to 7 percentage points higher accuracy at matched cost
than single-verifier feedback on SWE-bench, tau2-bench and Terminal-Bench. Using verifier
findings as a routing signal is therefore **not** claimed as new here. Four more routing/verifier
papers were found by search and are **not read**: RouteAudit (arXiv:2610.02808), CoVeR
(arXiv:2609.26086), arXiv:2609.34634 and arXiv:2609.33505. What is claimed is the
**combination** and only as a design to be tested: verifier-closed acceptance (nothing accepted
without `Candidate::passes`, a score never decides) + finding-routed role Pods + a gated
workspace with findings broadcast to all Pods + self-observation (self-model and schema, observe
and propose only) + gated L3 restructuring driven by verifier statistics, all under the closure
invariants (4, 5, 7, 8, 11, 12) and logged propensities. ORACLE routes between whole models; it has
no acceptance gate with hard findings, no workspace, no expert lifecycle and no unroutable-finding
signal for Birth.

### 11.1 Principle

1. **Every expert output at L2 is a typed proposal** submitted to the shared workspace, the
   bounded, verifier-gated, broadcast blackboard of `ptr-pods::workspace`
   (`crates/ptr-pods/src/workspace/`, [38](38-consciousness-indicators.md) GW-L2). That contract is
   deterministic and learns nothing; persisting workspace content in SemDB or a branch overlay is
   **not** what was built. A proposal is never an answer, an effect or a state change by itself.
2. **Nothing is accepted without passing the verifier fabric**: status `Pass`, no hard finding,
   and a verification level at or above the task's required level, judged on the *weakest*
   per-verifier level (doc 37 section 5; F001's `Candidate::passes`). A score never decides; a
   learned score cannot override a deterministic contradiction (invariant 11).
3. **Verifier findings are broadcast to all Pods** (the global-workspace link: GWT-3) and are the
   **primary routing signal** for what happens next: which repair expert, which retriever,
   whether to escalate.
4. **Nothing becomes authoritative until committed** (invariant 4); workspace content, critiques
   and self-model output are derived (invariant 5).
5. **Unknown and Disputed are outcomes, not failures to repair** (invariant 12): they lead to a
   fresh proposal or to escalation, never to repair of a candidate nobody found a defect in.

### 11.2 Expert roles as Pods

| Role | Level | Does | Authority | Exists today | Missing |
|---|---|---|---|---|---|
| Generator / proposer | L2 | Produces a typed proposal | none | `Generator` trait (`crates/ptr-feedback/src/repair.rs:23`) with a stub; `Pod` (`crates/ptr-pods/src/lib.rs:196` on the integration line) | OLMoE-class generator Pods behind the loop |
| Retriever | L2 | Supplies evidence candidates | none; evidence promotion is separate (invariant 6) | `ptr-search`, `ptr-memory` | Retriever Pods that read the workspace |
| Planner | L2 | Decomposes a task into typed sub-proposals | none | Operator router scaffold (`crates/ptr-core/src/operator_router.rs:12`) | Planner Pod |
| Learned critic | L2 | Predicts whether a proposal will pass | **advisory only**; cannot accept or veto against the verifier | none | Critic Pod; the guard test that it has no accept path |
| Finding-routed repair expert | L2 | Repairs the slots a finding class names (one expert per finding class) | none; output is a new proposal | `Repairer` (`repair.rs:29`) and `Critique::from_report` (`candidate.rs:68`) with a stub; routing to repair-role Pods per hard finding code by `FindingRouter` (`crates/ptr-router/src/findings.rs:821`) over a `FindingRegistry` (`:137`) | Real repair experts per finding class (F005) |
| Verifier fabric | L2 | `Pass`/`Fail`/`Disputed`/`Unknown` with findings and level | **decides acceptance** | `VerificationReport` (`crates/ptr-verifier/src/lib.rs:39`), `VerifierFabric` | none for the contract; more verifiers per domain |
| Self-model Pod | L2 | Predicts the MoE's own errors, findings and routing from routing traces (the HOT-2/3 indicator, doc 38) | observe and propose only | none | The Pod (C002) |
| Attention-schema model of the router | L2/L1 | Predicts and controls the router's own routing (AST-1) | observe and propose only | none | The model (C003) |
| Structural-operator policy | L3 | Proposes Birth, Merge, Split, Freeze, Reactivate, Prune | **proposes only**; gates decide | `ExpertProposal` (`crates/ptr-pods/src/expert.rs:1020`), R011 gate path | Verifier-statistics-driven policy (F006) |

### 11.3 The loop

```mermaid
flowchart LR
    REQ[request] --> RT[L2 router<br/>typed state + findings + self-model]
    RT -->|route, propensity logged| GEN[generator / retriever / planner Pods]
    GEN -->|typed proposal| WS[(workspace<br/>ptr-pods::workspace)]
    WS --> VF[verifier fabric]
    VF -->|Pass, no hard finding, level ok| ACC[accept]
    VF -->|findings broadcast| WS
    VF -->|Fail with findings| RT
    VF -->|Unknown / Disputed| FRESH[fresh proposal or escalate]
    ACC --> GATE[commit path<br/>ledger event]
    SELF[self-model Pod] -. predicts, advises .-> RT
    CRIT[learned critic] -. advisory .-> RT
    VF -->|statistics| L3[L3 operator policy<br/>proposals only]
    L3 --> GATES[ForgettingGate + interference + verifier]
    GATES --> GATE
```

The critic, the self-model and the L3 policy have no edge into acceptance or into the gates.

### 11.4 Routing at each level

* **L1 (token experts)** stays a learned router for latency. Router and expert *training* use
  verifier-derived rewards (RLVR, section 5, reward from `VerificationReport`) and the routing
  metrics already listed there; the router never sees a verifier at inference time.
* **L2 (Pods)** routes on typed workspace state, the verifier findings of the current
  proposals and the self-model's predictions. Every decision logs its propensity
  (`RoutedChoice`, `crates/ptr-router/src/router.rs:81`, produced only by `route_checked`,
  `router.rs:627`; the generic policy is `Router::policy`, `router.rs:608`). Finding-conditioned
  routing logs a `FindingRouteRecord` (`findings.rs:492`) with the same quantised propensity,
  conditional on the currently eligible Pods. Router policies can then be evaluated off-policy
  with the categorical estimator that section 2 specifies (the arbiter's `evaluate_off_policy`,
  `crates/ptr-branch/src/arbiter.rs:1027`, and `doubly_robust`, `:1091`, are the template).
* **L3 (structural operators)** is driven by verifier statistics, still only as proposals:
  Birth when a finding class has no expert that clears it; Merge when two experts clear the
  same findings with overlapping subspaces (`measure_interference`,
  `crates/ptr-lineage/src/subspace.rs:1123`); Prune by AIMER (`aimer_score`,
  `crates/ptr-lineage/src/aimer.rs:42`) together with usage; Freeze and Reactivate by recent
  clearance. Every proposal passes the ForgettingGate (`crates/ptr-lineage/src/forgetting.rs:133`),
  the interference test and the verifier, and is committed only as a ledger event (section 3).

### 11.5 Budgets and termination

Each request carries a verifier-call budget and a round budget, as in F001 (`RepairBudget`,
`repair.rs:37`). Exhausting either returns `Exhausted` (`Outcome`, `repair.rs:150`), never the
best failed candidate. Every candidate is verified exactly once. A repair touches only the slots
the hard findings name. `Unknown` and `Disputed` reports yield a fresh proposal or an
escalation (to a stronger verifier or a human), never a repair.

### 11.6 Training signal

Every verified `(finding, repair, pass)` transition is a verifiable training example for the
finding-routed repair expert and, through RLVR, for the router. Rewards come from the verifier
only. No statement a model makes about itself, in particular none about consciousness or
experience, is a target, reward or filter (principle 2 of doc 38).

### 11.7 Status

| Piece | Exists (file:line) | Missing | CPU / GPU | Evidence level |
|---|---|---|---|---|
| Acceptance gate, budgets, repair of named slots, `Exhausted` | `Candidate::passes` (`crates/ptr-feedback/src/candidate.rs:18`), `RepairLoop` (`repair.rs:166`) | model-backed generators and repairers | CPU | F001: mechanism with a seeded stub, no model claim |
| Workspace contract: bounded slots, bid competition after the verifier gate, bounded-mailbox broadcast, findings broadcast as refusals, staleness, redaction | `Workspace` (`crates/ptr-pods/src/workspace/mod.rs:360`), `run_cycle` (`cycle.rs:109`), `check_verification_gate` (`gate.rs:26`, the rule of `Candidate::passes`), `WorkspaceConfig` (`config.rs:138`); deterministic order: effective bid, then SHA-256 of (seed, revision, id), then id | a learned or state-conditioned write, a model-backed Pod behind it, persistence of workspace content in SemDB or `Branch` (`crates/ptr-branch/src/branch.rs:48`, `SemanticHost` `crates/ptr-semdb/src/lib.rs:472`), and any evidence that the contract helps a task | CPU for the contract; GPU at Pod level | tests of the contract only (on the integration line); no task evidence |
| Findings as routing signal | `FindingRouter` (`crates/ptr-router/src/findings.rs:821`), `classify` (`:352`, `Accept` only for `Pass` with no hard finding at the level), `FindingRegistry` (`:137`), `FallbackPolicy` (`:272`), `AdmissibleScorer` (`:658`; a learned scorer sees only verifier-admissible Pods and cannot name `Accept`), `FindingRouteRecord` (`:492`, propensities); the unroutable-code counter is the Birth signal | finding-conditioned routing with real Pods, off-policy evaluation of it, a learned scorer | CPU for the router; GPU with models | tests of the router only; no model evidence |
| Propensity logging and off-policy evaluation | `route_checked`, arbiter estimators, the categorical IPS/SNIPS/DR estimators over the finding router's log (`crates/ptr-router/src/ope.rs`) | the contextual log of section 2 | CPU | the arbiter estimators are tested for triage; the router's estimators and OPE promotion gate are unit-tested and ran in F007 on simulated experts |
| Verifier-statistics L3 operators | lifecycle types and gates (`crates/ptr-pods/src/expert.rs`), R011 gate path | the statistics-driven policy | GPU | R011: gate path only |
| Self-model, schema, critic | `SelfModel` (`crates/ptr-router/src/self_model.rs`) and `AttentionSchema` (`crates/ptr-router/src/attention_schema.rs`), advisory estimators over routing traces; no critic | a model-backed self-model, schema or critic; any evidence that either module helps | CPU toy, GPU on OLMoE | none for these modules (F007 does not test the self-model; C007 and C008 are separate CPU toys, both FALSIFIED) |

### 11.8 Experiments (F004-F006 planned)

* **F004 `verifier-closed-pod-moe` (GPU).** The verifier-closed Pod-MoE against a standard
  learned-gate MoE (same total and active parameters), a single generator with self-refinement
  and, decisively, **best-of-N sampling from the same MoE filtered by the same verifier at the same
  verifier calls and FLOPs** (verifier access alone must not be able to win the claim), and the
  same loop with a generic router and an ORACLE-style verifier-conditioned bandit router, at matched compute: verified success, verifier calls, false
  accepts (must be 0), latency.
* **F005 `finding-routed-repair-experts` (GPU, real models).** One repair expert per finding class
  against one generic repair expert and regenerate-only, with a shuffled-map control.
* **F006 `verifier-driven-expert-lifecycle` (GPU).** Birth and merge driven by verifier
  statistics against a static expert set and the M013 heuristic, extending M013.
* The indicator experiments of [38](38-consciousness-indicators.md) (C001-C006) plug into
  this loop as the workspace (C001, C006), the self-model (C002), the schema (C003) and
  recurrence (C004).

Each has hypothesis, metrics, baseline and falsification in its `experiment.toml`; all share the
fail-closed rules and pins of section 5. F004-F006 run only after the arms of section 5 have a
baseline on OLMoE.

### 11.9 Limits

* F001's gains are a consequence of its stub's per-slot skill; a model-backed result is
  F004/F005.
* Each ingredient has prior art (verifier-conditioned routing: ORACLE; best-of-N with a verifier;
  mixtures of experts; workspaces); a result for the combination is not a result for any single
  ingredient, which is why F004 includes best-of-N with the same verifier as a baseline.
* The loop is as good as the verifiers: a domain with only weak verifiers (`LatentAgreement`)
  cannot clear a `Deterministic` required level and returns `Exhausted`, by design.
* Verifier cost is part of the budget; a loop that wins only by calling the verifier far more
  often is not a win at matched compute (F004 matches verifier calls).

