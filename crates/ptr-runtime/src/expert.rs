//! Admission of expert structural changes (birth, merge, split, freeze, reactivate, prune,
//! retire) into the authoritative ledger.
//!
//! An [`ExpertProposal`] is inert data. It becomes authority only through
//! [`PtrRuntime::admit_expert_proposal`], which checks, in this order and before anything is
//! appended:
//!
//! 1. the proposal's project and proposer are the ones the host authenticated
//!    ([`ExpertAdmissionGrant`]), never the caller's claim;
//! 2. the gate report is bound to exactly this proposal and this operation, to the artifact the
//!    operation produces or acts on, to the expert's base model and to the evaluation manifest
//!    the host expects (see "Evidence binding");
//! 3. the report was produced under the gate configuration the grant requires;
//! 4. the report passed;
//! 5. a compare-and-set against [`ExpertRegistryProjection`] for every operand (generation,
//!    manifest digest, state, state version, level, base model) and for every result (pod and
//!    routing slot absent, except an atomic takeover from an operand of the same operation);
//! 6. immediately before the append, the authority behind the grant is still current (see
//!    "Commit-time authority").
//!
//! The record and the event's ids are then derived from the validated proposal, never taken from
//! the caller, and the event is appended through the runtime's internal path. The public
//! [`PtrRuntime::commit`] keeps refusing the event.
//!
//! The record committed is a version 2 [`ExpertCommitRecord`] (version 3 when it carries a
//! dependency block, see "Input dependencies"): besides the report digest it embeds
//! the report's canonical bytes and the gate configuration digest the grant required for its
//! gate epoch.
//!
//! Replay (`replay`, `open_durable`) runs the same record checks without a grant: it decodes the
//! record, re-derives the op kind, the proposal id, the operation id, the evidence digest and the
//! project, refuses any disagreement with the event, and repeats the compare-and-set against the
//! projection as of that point of the history. It also re-verifies the evidence itself: the
//! embedded report must decode canonically, its re-derived digest must equal the record's
//! evidence digest, it must have passed, its subject must be this record's proposal and
//! operation, it must have been produced under the gate configuration the record says was
//! required, and no two records of one gate epoch may disagree on that configuration. A tampered,
//! reordered or truncated history therefore fails to open instead of producing a different
//! expert registry, and a record whose digest was merely made self-consistent is refused.
//!
//! # Evidence binding
//!
//! One function, `check_report`, decides whether a report is evidence for an expectation, and
//! both [`PtrRuntime::admit_expert_proposal`] and [`PtrRuntime::readmit_expert`] call it (replay
//! calls it too, see below). It checks, in this order, the subject's proposal id, operation id,
//! artifact digest, base model digest and (at commit only) evaluation manifest digest, then the
//! gate configuration, then that the report passed. Each disagreement is its own typed refusal
//! and nothing is appended.
//!
//! * `artifact_digest` must equal [`expert_evidence_artifact_digest`]: the result's manifest
//!   digest for `Birth`, `Merge` and a `Reactivate` that creates a new generation; the operand's
//!   manifest digest for `Freeze`, `Retire`, a `Reactivate` of a frozen expert (the unfreeze)
//!   and a `Prune` of one target; for a `Split` and for a `Prune` of several targets a
//!   domain-separated SHA-256 over the ordered result (or operand) manifest digests. A readmission
//!   is bound to the manifest digest of the expert it readmits. For a version 3 record (one with
//!   a dependency block) that creates experts the artifact digest is extended to SHA-256 over
//!   the domain `ptr.expert.evidence.artifact-inputs.v1`, that digest and
//!   [`ExpertInputs::canonical_digest`] ([`expert_evidence_artifact_digest_with_inputs`]), so the
//!   gate attests the inputs the proposer names: a report produced for another block, or for none,
//!   is refused ([`ExpertAdmissionError::EvidenceArtifactMismatch`]) at admission and on replay.
//! * `base_model_digest` must equal the digest of the [`BaseModelRef`] every operand and result
//!   of the operation (or the readmitted expert) shares.
//! * `eval_manifest_digest` must equal the digest the host put in its grant
//!   ([`ExpertGateRequirement::eval_manifest_digest`]).
//!
//! A host that wants every expert to record its inputs sets
//! [`ExpertGateRequirement::require_inputs`]: an admission without a dependency block (the
//! version 2 path included) is then refused ([`ExpertAdmissionError::InputsRequired`]). That is
//! host policy at commit; replay cannot re-check it. What no mechanism checks is that a block is
//! complete: the gate attests the block it is shown, not that nothing was left out.
//!
//! The grant is not recorded, so replay re-checks the artifact and the base model binding but not
//! the evaluation manifest: the committed history cannot say which manifest the host expected.
//! A history written before these checks existed, whose reports carry arbitrary digests, is
//! refused on replay for the same reason a forged one is.
//!
//! # Commit-time authority
//!
//! Every grant carries an [`ExpertGrantBinding`]: the host's [`IdentityContext`], the host clock
//! reading `now` it hands the grant over at, the policy key id the grant's authority rests on
//! and the grant generation it was minted at. Immediately before the append (after the
//! compare-and-set, nothing awaits in between) the runtime refuses, with a typed error and
//! nothing appended, a grant whose generation is no longer current for its [`ExpertGrantKind`],
//! a revoked identity session, an identity context (expired, not yet issued or with a wrong
//! digest) not valid at the later of `now` and the highest host clock reading the runtime was
//! told of ([`PtrRuntime::observe_host_time`]; the runtime has no clock of its own, so a host
//! that never calls it gets only the grant's stamp), a policy key that is revoked or was never
//! installed, a transport peer ([`NodeId`]) the installed [`crate::execution::AdmissionPolicy`] no
//! longer admits and a principal (`issuer:subject`) the manifest authority registry no longer
//! admits.
//!
//! The grant generation registry is host authority held in memory:
//! [`PtrRuntime::expert_grant_generation`] reads it and [`PtrRuntime::supersede_expert_grant`]
//! bumps it, so every grant minted at an older generation is refused at its next commit. It is not written
//! to the ledger, not replayed, restarts at [`ExpertGrantGeneration::INITIAL`] and is not part
//! of the registry projection's equality. Session revocation is ledger-backed and policy key
//! revocation, peer admission and principal admission are runtime state; none of them is
//! re-checked on replay, because the grant is not recorded.
//!
//! # Route admission
//!
//! [`PtrRuntime::expert_route_admission`] is the use-time check an inference path asks before it
//! routes to an expert. It is not a router: no production router consumes it. It refuses an
//! expert that is unknown, at another generation, fenced, not `Serving`/`Frozen`, or, for an
//! expert admitted with a dependency block, or descending from experts that were, one of whose
//! recorded inputs or an ancestor's is revoked or superseded beyond the generation it was built
//! from ([`RouteRefusal::DependencyNotLive`]), whether or not a fence was ever applied.
//!
//! # Input dependencies and the compound invalidation
//!
//! [`PtrRuntime::admit_expert_proposal_with_inputs`] commits a version 3
//! [`ExpertCommitRecord`] whose typed dependency block names the training and the evaluation
//! inputs the produced expert(s) rest on, each as `(identity, generation, digest)`; the base
//! model of the expert is a dependency too, under the identity `base-model:{id}` (see
//! [`ptr_pods::ExpertInputRef::from_base_model`]). [`PtrRuntime::admit_expert_proposal`] keeps
//! writing version 2 records that name no inputs, so an expert admitted that way is outside every
//! mechanism below. The [`ExpertDependencyIndex`] inside the registry projection holds the
//! relation, the source edges of `Merge`, `Split` and `Reactivate` and the input-digest registry
//! (see `dependency.rs` for the full content).
//!
//! The digest a block asserts for an input is the proposer's word: nothing authenticates it
//! except the gate evidence bound to the block. The first committed use of an `(identity,
//! generation)` fixes it, so a proposer can fix a wrong one for an identity the lifecycle
//! authority does not know (later honest uses are refused as stale). It cannot block a
//! revocation, which does not depend on the registered digest; a supersession is still held to
//! it.
//!
//! Admission of a version 3 record, and its replay, refuse before anything is appended (typed,
//! for training and evaluation inputs and the base model alike):
//!
//! * an input revoked, by a committed `InputInvalidated`, by a committed `Revoked` or
//!   `ProcedureRevoked`, or as [`PtrRuntime::generation_validity`] reports it
//!   ([`ExpertAdmissionError::InputRevoked`]);
//! * an input generation below a supersession floor or below the live generation the lifecycle
//!   authority holds ([`ExpertAdmissionError::InputSuperseded`], the stale input generation);
//! * an input generation ahead of the live generation the lifecycle authority holds for an
//!   identity it knows ([`ExpertAdmissionError::InputGenerationMismatch`]): a record cannot name
//!   `(identity, live + 1)` and fix a digest for a generation that does not exist yet;
//! * for an expert created from source experts (`Merge`, `Split`, `Reactivate`), any input of a
//!   tracked ancestor, transitively, that is revoked (no exception: revoked data is not
//!   laundered by retraining) or superseded unless the new expert's own block, or an expert on the
//!   path from the ancestor, names the replacement, with or without a block of its own
//!   ([`ExpertInputRole::Inherited`]). The route probe and a readmission judge an expert the same
//!   way over its own inputs and its ancestors';
//! * an input digest that differs from the one the first committed use of that
//!   `(identity, generation)` fixed ([`ExpertAdmissionError::InputDigestStale`]). Only a
//!   committed `Superseded` invalidation moves a generation's digest forward, by fixing the digest
//!   of its new generation.
//!
//! [`PtrRuntime::invalidate_input`] is the compound event. Under a
//! [`ExpertFenceAuthority::GlobalRevocation`] grant ([`ExpertFenceGrant::global_revocation`]; a
//! project-scoped [`ExpertFenceAuthority::RuntimeRevocation`] grant is refused, because the
//! event reaches every project) it appends ONE
//! [`LedgerEvent::InputInvalidated`] (ledger tag 30) that revokes (or supersedes) the input and
//! fences every expert that rests on it. The fence set is derived at this commit index from the
//! index: every expert whose dependency block names the input (`Revoked`: at exactly that
//! generation; `Superseded`: at any generation below the new one), closed over the source edges
//! (a descendant of an affected expert is affected), keeping those that are `Serving`, `Frozen`
//! or fenced for `AuthorityQuarantine`. A `Superseded` closure does not follow a descendant whose
//! own block already names the input at the new generation or above (it was retrained on the
//! replacement); no other planned-succession typing exists. The event carries only the count and
//! a domain-separated SHA-256 over the canonical ordered set ([`expert_fence_set_digest`]);
//! applying it fences every member in one step with cause `InputRevoked` (a quarantined member is
//! escalated; a supersession uses the same cause, which is final), moves their fence epochs up by
//! one and, for a revocation, joins the revocation set exactly like a `Revoked` record. A
//! supersession reads only the new generation and the floor: the `old_generation` it carries is
//! evidence (it must be below the new generation and is held to the registered digest), not a
//! bound, so every generation below the new one is stale afterwards. The dependency index is
//! never pruned (revocations, floors, digests, blocks and edges only grow with history) and its
//! snapshot is size-limited (`CompactedError::SizeLimit`), so a very long history can outgrow it;
//! a supersession scans the older dependents of its identity, O(k * n) over k supersessions. Commit
//! and replay both re-derive the set at that point of the history and refuse a different count
//! or digest ([`ExpertAdmissionError::FenceSetMismatch`]), so a fan-out of any size is one small
//! record and costs O(set) to apply.
//!
//! The `expert:` rows of a plain projection are not authoritative after such a fence: the event
//! names the fenced experts by count and digest only, so a consumer that applies only
//! `ptr_state::projection_entries` (the reference backend, ptr-pg) still sees them `Serving`;
//! only the runtime's materialized state (and its registry) marks them. Nothing may route from
//! that projection; the same holds for an authority change (see "Evidence-authority epoch").
//!
//! The older pair, a plain `Revoked` followed by [`PtrRuntime::fence_expert`] per expert, keeps
//! working unchanged. Only the compound event is atomic: between a plain `Revoked` and the last
//! fence an expert is still `Serving`, which route admission closes only for experts that
//! recorded their inputs. A readmission of a quarantined expert is refused while any recorded
//! input is not live ([`ExpertAdmissionError::DependencyNotLive`]).
//!
//! # Evidence-authority epoch
//!
//! [`PtrRuntime::change_expert_authority`] moves the standing (epoch and identity) of the gate
//! configuration or of the evaluator with ONE [`LedgerEvent::ExpertAuthorityChanged`] (ledger tag
//! 31). Once a kind has a standing, admission, readmission and replay all require evidence under
//! it ([`ExpertAdmissionError::StalePolicyEvidence`]), and a `Revoke` or `TighteningRotation`
//! fences, in the same event and with cause `AuthorityQuarantine`, every `Serving` or `Frozen`
//! expert admitted under an earlier epoch of that kind (count and digest in the event, re-derived
//! on replay); a `NonTightening` rotation fences nothing. A quarantined expert is readmitted
//! through the gated path on evidence under the current standing. The standings and the
//! per-expert admission epochs are part of the registry projection and of the compacted snapshot.
//! See the `authority` module for the full rules and their limits.
//!
//! # Legacy records
//!
//! A version 1 record (written before evidence was embedded) carries only a digest. Rewriting
//! history is not possible and refusing it would make existing ledgers unopenable, so replay
//! accepts it, applies its compare-and-set and marks every expert it touched
//! [`ExpertEvidenceStatus::UnverifiedLegacy`] in [`ExpertRegistryProjection::evidence_status`]
//! until a later verified record rewrites the expert. Admission never writes version 1, so a
//! legacy record cannot be admitted again, and it takes part in no epoch consistency check.
//!
//! # What this does not check
//!
//! The report is trusted to be what the host's evaluator produced (see `ptr-lineage`
//! `BoundGateReport`): neither admission nor replay re-runs the gate, and the digest is an
//! unkeyed hash, so replay shows the committed evidence is the evidence that was admitted, not
//! that the measurements are true. The inputs a version 3 record names are checked for liveness
//! and digest (see "Input dependencies"); an expert admitted without a dependency block is not.
//! The gate epoch is held to the current standing once a kind has one (see "Evidence-authority
//! epoch"), not otherwise. The grant is a snapshot the host builds from its own
//! current authority; only the identity, key, principal and grant generation it rests on are
//! re-read at commit. A compacted snapshot carries neither the legacy marks nor the per-epoch
//! configuration digests, so both restart empty from one.
//!
//! # Post-operation state
//!
//! [`ptr_pods::StructuralOp::outcomes`] is the single definition of the state each operand and
//! result is left in; this module stores it, `ptr-state` projects it.
//!
//! # Fencing
//!
//! A fence is not a proposal: no proposal may name a fenced expert, and the gate does not judge
//! one. [`PtrRuntime::fence_expert`] and [`PtrRuntime::readmit_expert`] write
//! [`LedgerEvent::ExpertFenceCommitted`] under an [`ExpertFenceGrant`] the host builds from its
//! own authority: a fence needs [`ExpertFenceAuthority::RuntimeRevocation`], a readmission
//! [`ExpertFenceAuthority::Gate`]. Both compare-and-set the expert's generation and state
//! version and move the state version to the next value, the expert's **fence epoch**: a fence
//! takes a `Serving` or `Frozen` expert to `Fenced` at an unchanged generation, a readmission
//! returns a fenced expert to the state the fence left it in (`Serving` or `Frozen`, recorded in
//! the event and checked on replay). A readmission is accepted only for a fence of cause
//! [`ExpertFenceCause::AuthorityQuarantine`]; an `InputRevoked` fence is final. The cause only
//! tightens: [`PtrRuntime::escalate_expert_fence`] re-fences a quarantined expert for
//! `InputRevoked` (still `Fenced`, new epoch); the reverse is refused. The public
//! `commit` refuses the event like it refuses a structural record. A KV entry that recorded the
//! expert's earlier epoch is never revived, because the materialized value
//! (`<generation>:<state>:<epoch>`) no longer carries that epoch.
//!
//! # Revocation-driven retirement
//!
//! [`PtrRuntime::retire_revoked_expert`] takes a `Fenced` expert of cause `InputRevoked` to
//! `Retired(InputRevoked)` (fence action `Retire`, ledger code 3), compare-and-setting generation
//! and state version and moving the state version up by one. It is the runtime's reaction to a
//! revocation, not a gate decision: it needs a [`ExpertFenceAuthority::RuntimeRevocation`] grant
//! (revalidated at commit like a fence), the same committed generation-exact `Revoked` or
//! `ProcedureRevoked` the fence rests on, and it carries no gate evidence (the codec refuses
//! any). It is refused, typed and with nothing appended, for an unfenced expert
//! ([`ExpertAdmissionError::NotFenced`]), an already retired one, a fence of cause
//! `AuthorityQuarantine` ([`ExpertAdmissionError::NotRetirable`]), a wrong generation or state
//! version, an uncommitted revocation and a gate grant. The retired expert leaves the fence
//! registry, releases its routing slot, is never routable ([`RouteRefusal::NotServing`]) and is
//! terminal: readmission is refused (`NotFenced`), no proposal may name it as a reactivation seed
//! ([`ptr_pods::ExpertProposalError::SeedUnavailable`]) and no proposal may retire a fenced
//! expert ([`ptr_pods::ExpertProposalError::FencedTarget`]). Replay repeats the compare-and-set,
//! the cause check and the committed-revocation check; a compacted snapshot carries the state
//! and the fence memo; a KV entry that depended on the expert is dead from the fence on.
//! The transition does not run by itself: the compound invalidation fences, it does not retire,
//! so the host still issues the retirement. A `Superseded` invalidation fences with the same
//! cause but names no revocation: the retirement may cite the committed supersession instead,
//! `(input, a generation below the floor)`, which only a retirement accepts. For an expert that
//! recorded its inputs, any cited revocation (of a fence, an escalation or a retirement) must be
//! one of its inputs or of an ancestor's ([`ExpertAdmissionError::RevocationNotADependency`]);
//! an expert that recorded none may cite any committed revocation.
//!
//! `InputRevoked` is a ledger-backed cause: the event names the `(subject, generation)` of a
//! committed `Revoked` (or `ProcedureRevoked`) record, and commit and replay refuse it unless the
//! runtime's revocations, built in commit order, already hold it. `AuthorityQuarantine` stays a
//! cause the host asserts under its grant: the ledger holds no record of it to check.
//!
//! The registry records the inputs of an expert admitted with a version 3 record (see "Input
//! dependencies"), and a readmission of such an expert is refused while any of them is not live.
//! For an expert admitted without a dependency block the registry knows no inputs, so the
//! safeguard for an input revoked while the expert is quarantined stays the escalation above,
//! which the host issues when it sees the revocation, or the compound invalidation, which
//! escalates a quarantined dependent itself.
//!
//! Replay repeats the compare-and-set against the registry as of that point of the history; it
//! cannot re-check the grant, which is not recorded.
//!
//! A fence never waits for a gate. A readmission does: it needs a passed [`BoundGateReport`]
//! whose subject is bound to this expert, generation, state version and epoch
//! ([`expert_readmission_subject_ids`]) and produced under the gate configuration the
//! [`ExpertFenceAuthority::Gate`] grant requires. The report's canonical bytes, the gate epoch
//! and the required configuration digest travel in the event, and replay re-verifies them as for
//! a structural record (canonical decode, passed, subject, configuration, one configuration per
//! gate epoch). The gate epoch of a readmission may not be below the highest one an earlier
//! readmission was admitted under, at commit and on replay alike.
//!
//! The registry also keeps a fence memo: the highest epoch each expert was ever fenced at. A KV
//! dependency whose recorded epoch is `0` or at most the memo is never live, also after a
//! readmission, so an entry recorded before the fence (or with no epoch) is never revived.
//!
//! # The L3 policy
//!
//! [`StructurePolicy`] turns verifier statistics into [`ptr_pods::ExpertOperationRequest`]s
//! (Birth, Merge, Prune, Freeze, Reactivate). It is pure and holds no authority: it sees no
//! ledger, grant, gate report or verifier configuration. A request that creates an expert has no
//! manifest digest until the host completes it with the trained one
//! ([`StructureProposal::complete`]); only then is it an [`ExpertProposal`], which
//! [`PtrRuntime::admit_expert_proposal`] decides on a gate report. See the `structure` module
//! source for the operators and bounds.

mod attestation;
mod authority;
mod dependency;
mod structure;

use crate::manifest::ManifestBindingAuthority;
use crate::{PtrRuntime, RuntimeError};
pub use attestation::{
    ExpertAttestation, ATTESTOR_IDENTITY_PREFIX, RUN_GENERATION, RUN_IDENTITY_PREFIX,
};
pub(crate) use authority::AuthorityState;
pub use authority::{
    expert_authority_fence_set_digest, ExpertAdmissionEpochs, ExpertAuthorityChange,
    ExpertAuthorityClass, ExpertAuthorityKind, ExpertAuthorityOutcome, ExpertAuthorityStanding,
    StaleEvidenceReason,
};
use authority::{EvaluatorClaim, EvidenceClaim};
pub use dependency::ExpertDependencyIndex;
pub(crate) use dependency::ExpertInputSet;
use dependency::{
    DependencyUpdate, ExpertRef, InputKey, InputStatus, RecordedBlock, RecordedInput,
};
pub use ptr_ledger::InputInvalidationKind;
use ptr_ledger::{
    ExpertFenceAction, ExpertFenceCause, ExpertFenceEvidence, ExpertFenceResume,
    ExpertFenceRevocation, ExpertOpKind, LedgerEvent,
};
use ptr_lineage::BoundGateReport;
use ptr_pods::{
    BaseModelRef, ExpertCommitRecord, ExpertInputRef, ExpertInputs, ExpertLevel, ExpertProposal,
    ExpertState, ManifestDigest, OpKind, ProposerRef, RetiredReason, StructuralOp,
};
use ptr_types::{
    CommitIndex, Generation, IdentityContext, NodeId, PodId, PrincipalId, ProjectId, Timestamp,
    Validity,
};
use std::collections::{BTreeMap, BTreeSet};
pub use structure::{
    unroutable_delta, BirthContext, BirthSite, CodeWindowStats, EnabledOps, ObservationLimits,
    PodCodeStats, RegisteredExpert, StructureCompletionError, StructureConfigError,
    StructureObservation, StructureObservationError, StructurePolicy, StructurePolicyConfig,
    StructureProposal, StructureRationale, StructureReport, SubspaceOverlap, Suppression,
    WindowObservation, MAX_COUNT,
};

/// What the host expects of the evidence behind one grant: the gate epoch it stands at, the
/// digest of the gate configuration it requires and the digest of the evaluation manifest the
/// evaluation must have run on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpertGateRequirement {
    pub gate_epoch: u64,
    pub config_digest: [u8; 32],
    pub eval_manifest_digest: [u8; 32],
    /// Whether the host requires the dependency block: an admission without a version 3 inputs
    /// block (including every [`PtrRuntime::admit_expert_proposal`] call) is refused as
    /// [`ExpertAdmissionError::InputsRequired`]. Only the admission path reads it; a readmission
    /// grant ignores it. It is host policy at commit and is not recorded, so replay cannot
    /// re-check it (it does re-check that the evidence is bound to the block a record carries).
    pub require_inputs: bool,
}

/// The kind of host authority a grant stands on. Each kind has its own current grant generation.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ExpertGrantKind {
    /// An [`ExpertAdmissionGrant`]: the proposer's authority to admit a proposal.
    Admission,
    /// An [`ExpertFenceGrant`] of [`ExpertFenceAuthority::Gate`]: the gate host's authority to
    /// readmit.
    Gate,
    /// An [`ExpertFenceGrant`] of [`ExpertFenceAuthority::RuntimeRevocation`]: the runtime's
    /// authority to fence the experts of one project.
    Fence,
    /// An [`ExpertFenceGrant`] of [`ExpertFenceAuthority::GlobalRevocation`]: the authority to
    /// invalidate an input or change an evidence authority for every project at once.
    GlobalRevocation,
}

/// The generation of a host grant. A grant is refused at commit unless its generation is the
/// current one for its [`ExpertGrantKind`].
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ExpertGrantGeneration(u64);

impl ExpertGrantGeneration {
    /// The generation every kind starts at in a new runtime.
    pub const INITIAL: Self = Self(1);

    pub fn get(self) -> u64 {
        self.0
    }
}

/// The authority a grant rests on, re-read from the runtime immediately before the append.
///
/// Built by the host from its own authentication: the identity context of the principal the
/// grant speaks for, the host clock reading `now`, the id of the policy key the grant's
/// authority was signed under and the grant generation read from
/// [`PtrRuntime::expert_grant_generation`] when the grant was minted.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertGrantBinding {
    identity: IdentityContext,
    peer: NodeId,
    now: Timestamp,
    policy_key_id: String,
    generation: ExpertGrantGeneration,
}

impl ExpertGrantBinding {
    pub fn new(
        identity: IdentityContext,
        peer: NodeId,
        now: Timestamp,
        policy_key_id: impl Into<String>,
        generation: ExpertGrantGeneration,
    ) -> Self {
        Self {
            identity,
            peer,
            now,
            policy_key_id: policy_key_id.into(),
            generation,
        }
    }

    /// The same binding with the host clock reading `now`.
    #[must_use]
    pub fn at(mut self, now: Timestamp) -> Self {
        self.now = now;
        self
    }

    pub fn identity(&self) -> &IdentityContext {
        &self.identity
    }

    /// The transport peer the grant was authenticated over.
    pub fn peer(&self) -> &NodeId {
        &self.peer
    }

    pub fn now(&self) -> Timestamp {
        self.now
    }

    pub fn policy_key_id(&self) -> &str {
        &self.policy_key_id
    }

    pub fn generation(&self) -> ExpertGrantGeneration {
        self.generation
    }
}

/// What the host has authenticated for one expert admission.
///
/// Only the host constructs a grant, from its own authority: the project and policy pod the
/// session's principal is granted, the gate requirement it currently stands at and the
/// [`ExpertGrantBinding`] of the authority behind it. Nothing in a proposal or a report can
/// supply or widen one of these, and the runtime does not remember a grant between calls.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertAdmissionGrant {
    project: ProjectId,
    proposer: ProposerRef,
    requirement: ExpertGateRequirement,
    binding: ExpertGrantBinding,
    evaluator: Option<[u8; 32]>,
    attestation_required: bool,
}

impl ExpertAdmissionGrant {
    /// A grant for `proposer` (a policy pod and its generation, which carries its own project)
    /// in `project`. The proposer is named whole, so a principal authenticated as one policy
    /// pod cannot admit a proposal naming another.
    pub fn new(
        project: ProjectId,
        proposer: ProposerRef,
        requirement: ExpertGateRequirement,
        binding: ExpertGrantBinding,
    ) -> Self {
        Self {
            project,
            proposer,
            requirement,
            binding,
            evaluator: None,
            attestation_required: false,
        }
    }

    /// The same grant naming `evaluator_id_digest` as the evaluator the report comes from. Once
    /// the evaluator authority has a standing (see [`PtrRuntime::change_expert_authority`]) the
    /// name must equal its current identity or the admission is refused as stale policy
    /// evidence; a grant that names none is refused. The name is the host's assertion: the
    /// report does not carry an evaluator, and the record does not either.
    #[must_use]
    pub fn with_evaluator(mut self, evaluator_id_digest: [u8; 32]) -> Self {
        self.evaluator = Some(evaluator_id_digest);
        self
    }

    /// The same grant requiring an attestation ([`ExpertAttestation`]): an admission that carries
    /// none (every call of [`PtrRuntime::admit_expert_proposal`] and
    /// [`PtrRuntime::admit_expert_proposal_with_inputs`]) is refused as
    /// [`ExpertAdmissionError::AttestationRequired`], before any other check. Host policy at
    /// commit: it is not recorded, so replay cannot re-check it.
    #[must_use]
    pub fn with_required_attestation(mut self) -> Self {
        self.attestation_required = true;
        self
    }

    /// Whether the grant refuses an admission that carries no attestation.
    pub fn requires_attestation(&self) -> bool {
        self.attestation_required
    }

    /// The evaluator this grant names, if any.
    pub fn evaluator(&self) -> Option<&[u8; 32]> {
        self.evaluator.as_ref()
    }

    pub fn project(&self) -> &ProjectId {
        &self.project
    }

    pub fn proposer(&self) -> &ProposerRef {
        &self.proposer
    }

    pub fn gate_epoch(&self) -> u64 {
        self.requirement.gate_epoch
    }

    pub fn required_gate_config_digest(&self) -> &[u8; 32] {
        &self.requirement.config_digest
    }

    pub fn expected_eval_manifest_digest(&self) -> &[u8; 32] {
        &self.requirement.eval_manifest_digest
    }

    /// Whether the grant refuses an admission that carries no dependency block.
    pub fn requires_inputs(&self) -> bool {
        self.requirement.require_inputs
    }

    pub fn binding(&self) -> &ExpertGrantBinding {
        &self.binding
    }
}

/// The compare-and-set field that disagreed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpertStaleField {
    Generation,
    ManifestDigest,
    State,
    StateVersion,
    Level,
    BaseModel,
}

/// The event field that disagreed with the record it carries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpertEventField {
    Project,
    OpKind,
    ProposalId,
    OperationId,
    EvidenceDigest,
}

/// Why an expert structural change was refused.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExpertAdmissionError {
    /// The proposal names a project other than the one the host authenticated.
    ProjectMismatch,
    /// The proposal names a proposer other than the one the host authenticated.
    ProposerMismatch,
    /// The gate report is bound to another proposal.
    EvidenceProposalMismatch,
    /// The gate report is bound to another operation.
    EvidenceOperationMismatch,
    /// The gate report was produced under a gate configuration the grant (at admission) or the
    /// record (on replay) does not require.
    GateConfigMismatch,
    /// The embedded gate report does not decode canonically.
    EvidenceUndecodable(String),
    /// The embedded gate report's digest is not the record's evidence digest.
    EvidenceDigestMismatch,
    /// Two records of one gate epoch name different required gate configurations.
    EpochGateConfigConflict { epoch: u64 },
    /// The gate report did not pass.
    GateFailed,
    /// An operand is not in the registry.
    UnknownOperand { pod: String },
    /// An operand differs from the authoritative entry (compare-and-set failed).
    StaleOperand {
        pod: String,
        field: ExpertStaleField,
    },
    /// A result's pod id is already registered.
    ResultPodOccupied { pod: String },
    /// A result's routing slot is held by an expert the operation does not replace.
    ResultSlotOccupied { layer: u16, index: u16 },
    /// The record does not decode, or the evidence digest cannot be bound to it.
    InvalidRecord(String),
    /// The event cannot be framed by the ledger (size or count limits).
    Unencodable,
    /// The event disagrees with the record it carries.
    EventMismatch { field: ExpertEventField },
    /// The fence request names a project other than the one the grant authenticated.
    FenceProjectMismatch,
    /// The grant's authority does not allow this fence action.
    FenceAuthorityMismatch {
        action: ExpertFenceAction,
        required: ExpertFenceAuthority,
    },
    /// The expert is not `Serving` or `Frozen`, so it cannot be fenced.
    NotFenceable { pod: String, state: ExpertState },
    /// A readmission names an expert that is not `Fenced`.
    NotFenced { pod: String, state: ExpertState },
    /// A readmission names a different cause than the fence it would clear.
    FenceCauseMismatch { pod: String },
    /// An escalation names a fence whose cause is not `AuthorityQuarantine`: an `InputRevoked`
    /// fence is already final, and the cause never loosens. A fence of unknown cause (restored
    /// from a snapshot that did not record it) is not escalated either.
    NotEscalatable { pod: String },
    /// A retirement names a fence whose cause is not `InputRevoked`: only a revoked input retires
    /// an expert. An `AuthorityQuarantine` fence is readmitted or escalated, never retired, and a
    /// fence of unknown cause (restored from a snapshot that did not record it) is not retired
    /// either.
    NotRetirable { pod: String },
    /// An `InputRevoked` fence or escalation names a revocation that no committed `Revoked` (or
    /// `ProcedureRevoked`) record at or before it carries, or names none.
    FenceRevocationNotCommitted { pod: String },
    /// An escalation or readmission records a resume state other than the one the fence left, or
    /// a fence records one other than the state the expert is in.
    FenceResumeMismatch { pod: String },
    /// A readmission's gate epoch is below the highest gate epoch an earlier readmission was
    /// admitted under.
    GateEpochRegressed { pod: String, seen: u64, found: u64 },
    /// The fence was caused by a revoked input, or its cause is unknown (a snapshot restored
    /// from before causes were recorded): it is never readmitted.
    NotReadmittable { pod: String },
    /// A readmission carries no gate evidence.
    FenceEvidenceMissing { pod: String },
    /// A fence carries gate evidence: a fence never waits for a gate.
    FenceEvidenceUnexpected { pod: String },
    /// The event's epoch is not exactly one above the state version it compares against.
    FenceEpochMismatch { pod: String },
    /// The gate report's artifact digest is not the digest the operation (or the readmitted
    /// expert) determines, see [`expert_evidence_artifact_digest`].
    EvidenceArtifactMismatch,
    /// The gate report's base model digest is not the digest of the base model the operation
    /// (or the readmitted expert) is on.
    EvidenceBaseModelMismatch,
    /// The gate report's evaluation manifest digest is not the one the grant expects.
    EvidenceEvalManifestMismatch,
    /// The grant was minted at a generation that is no longer current for its kind.
    GrantGenerationStale {
        kind: ExpertGrantKind,
        grant: u64,
        current: u64,
    },
    /// The identity session behind the grant was revoked after the grant was minted.
    GrantSessionRevoked,
    /// The identity context behind the grant is not valid at the grant's `now`: expired, not yet
    /// issued, or its digest does not match its content.
    GrantIdentityInvalid,
    /// The policy key the grant's authority rests on is revoked or was never installed.
    GrantPolicyKeyUntrusted,
    /// The principal the grant's identity maps to is not (or no longer) admitted.
    GrantPrincipalNotAdmitted,
    /// The transport peer the grant was authenticated over is no longer admitted by the
    /// runtime's [`crate::execution::AdmissionPolicy`].
    GrantPeerNotAdmitted,
    /// The grant generation of this kind cannot be advanced any further.
    GrantGenerationExhausted { kind: ExpertGrantKind },
    /// The grant requires a version 3 inputs block ([`ExpertGateRequirement::require_inputs`])
    /// and the admission carries none.
    InputsRequired,
    /// An attestation names another proposal than the one being admitted.
    AttestationProposalMismatch,
    /// An attestation names another artifact than the one the proposal determines
    /// ([`expert_evidence_artifact_digest`]).
    AttestationArtifactMismatch,
    /// The grant requires an attestation ([`ExpertAdmissionGrant::with_required_attestation`])
    /// and the admission carries none.
    AttestationRequired,
    /// A record names a training run (`run:` input) that an earlier committed record already
    /// named: a run id is single-use.
    RunIdReused { identity: String },
    /// A host time observation is below the highest one already observed.
    HostTimeRegressed { floor: Timestamp, found: Timestamp },
    /// A fence, escalation or retirement cites a revocation that is not one of the inputs the
    /// expert rests on, itself or through its ancestors.
    RevocationNotADependency {
        pod: String,
        subject: String,
        generation: Generation,
    },
    /// A record names an input that is revoked: by a committed `InputInvalidated`, a committed
    /// `Revoked` or `ProcedureRevoked`, or as the lifecycle authority reports it.
    InputRevoked {
        role: ExpertInputRole,
        identity: String,
        generation: Generation,
    },
    /// A record names an input generation that is superseded: below a supersession floor or
    /// below the live generation of the lifecycle authority (a stale input generation).
    InputSuperseded {
        role: ExpertInputRole,
        identity: String,
        generation: Generation,
        floor: Generation,
    },
    /// A record names an input with a digest other than the one the first committed use of that
    /// `(identity, generation)` fixed (a stale input digest), or names one `(identity,
    /// generation)` with two digests.
    InputDigestStale {
        role: ExpertInputRole,
        identity: String,
        generation: Generation,
    },
    /// The input generation is already revoked by a committed `InputInvalidated`.
    InputAlreadyInvalidated {
        identity: String,
        generation: Generation,
    },
    /// The invalidation's binding digest (or a supersession's new digest) differs from the
    /// digest the registry fixed for that `(identity, generation)`.
    InputBindingMismatch {
        identity: String,
        generation: Generation,
    },
    /// The invalidation names a generation the lifecycle authority does not hold: a revocation
    /// of a generation ahead of the live one, or a supersession to a generation not yet
    /// committed there. `live` is the generation the authority holds.
    InputGenerationMismatch {
        identity: String,
        generation: Generation,
        live: Generation,
    },
    /// A supersession does not move the floor of its identity forward.
    StaleSupersession {
        identity: String,
        floor: Generation,
        new_generation: Generation,
    },
    /// The count or digest of the fence set an `InputInvalidated` event carries is not the set
    /// derived from the dependency index at that point of the history.
    FenceSetMismatch {
        derived_count: u32,
        event_count: u32,
        digest_matches: bool,
    },
    /// A readmission names an expert one of whose recorded inputs is revoked or superseded.
    DependencyNotLive {
        pod: String,
        identity: String,
        generation: Generation,
    },
    /// The evidence was issued under an authority that is no longer current: its gate epoch or
    /// gate configuration is not the current one, or the evaluator it comes from is not the
    /// current evaluator (see [`ExpertAuthorityKind`]).
    StalePolicyEvidence {
        kind: ExpertAuthorityKind,
        reason: StaleEvidenceReason,
    },
    /// An authority change names an epoch that does not exceed the current epoch of its kind.
    AuthorityEpochNotIncreasing {
        kind: ExpertAuthorityKind,
        current: u64,
        found: u64,
    },
    /// A `Revoke` names the identity that is current: a revoked identity cannot stay current.
    AuthorityIdentityUnchanged { kind: ExpertAuthorityKind },
}

/// The role an input plays in a version 3 record, as named in a refusal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpertInputRole {
    /// A training input.
    Training,
    /// An evaluation input.
    Evaluation,
    /// The base model, derived from the record's operation.
    BaseModel,
    /// An input of an ancestor expert (a `Merge`, `Split` or `Reactivate` source, transitively)
    /// that the new expert inherits.
    Inherited,
}

/// Why an expert may not be routed to.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RouteRefusal {
    /// The registry holds no such expert.
    UnknownExpert,
    /// The registry holds the expert at another generation than the one asked for.
    GenerationMismatch { current: Generation },
    /// The expert is fenced.
    Fenced,
    /// The expert is in a state other than `Serving` or `Frozen`.
    NotServing { state: ExpertState },
    /// One of the inputs the expert's record named is revoked or superseded beyond the
    /// generation the expert was built from, whether or not a fence was applied.
    DependencyNotLive {
        identity: String,
        generation: Generation,
    },
}

/// Which of the host's authorities a fence grant carries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpertFenceAuthority {
    /// The runtime's revocation authority over one project: it may fence, escalate and retire the
    /// experts of the grant's project.
    RuntimeRevocation,
    /// The gate host's authority: it may readmit a quarantined expert.
    Gate,
    /// The runtime-wide revocation authority. An input identity names no project and an evidence
    /// authority belongs to the whole runtime, so [`PtrRuntime::invalidate_input`] and
    /// [`PtrRuntime::change_expert_authority`] fence experts of every project; they need this
    /// authority and refuse a project-scoped [`ExpertFenceAuthority::RuntimeRevocation`] grant.
    /// It is not accepted for a per-expert fence, escalation or retirement.
    GlobalRevocation,
}

/// What the host has authenticated for one fence or readmission.
///
/// As with [`ExpertAdmissionGrant`], only the host constructs one, from its own authority, and
/// the runtime does not remember it between calls. A readmission grant also carries the gate
/// requirement: the epoch the host stands at, the digest of the gate configuration it requires
/// evidence from and the evaluation manifest digest it expects the evaluation to have run on.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertFenceGrant {
    /// `None` for a [`ExpertFenceAuthority::GlobalRevocation`] grant, which has no project.
    project: Option<ProjectId>,
    authority: ExpertFenceAuthority,
    gate: Option<ExpertGateRequirement>,
    binding: ExpertGrantBinding,
    evaluator: Option<[u8; 32]>,
}

impl ExpertFenceGrant {
    /// A grant of the runtime's revocation authority: it may fence, never readmit.
    pub fn revocation(project: ProjectId, binding: ExpertGrantBinding) -> Self {
        Self {
            project: Some(project),
            authority: ExpertFenceAuthority::RuntimeRevocation,
            gate: None,
            binding,
            evaluator: None,
        }
    }

    /// A grant of the runtime-wide revocation authority, for [`PtrRuntime::invalidate_input`]
    /// and [`PtrRuntime::change_expert_authority`], whose effect crosses projects. It has no
    /// project and is revalidated against its own grant generation
    /// ([`ExpertGrantKind::GlobalRevocation`]). It may not fence, escalate or retire a single
    /// expert, and a project-scoped [`ExpertFenceGrant::revocation`] may not invalidate an input.
    pub fn global_revocation(binding: ExpertGrantBinding) -> Self {
        Self {
            project: None,
            authority: ExpertFenceAuthority::GlobalRevocation,
            gate: None,
            binding,
            evaluator: None,
        }
    }

    /// A grant of the gate authority: it may readmit on a report produced under
    /// `requirement`, never fence.
    pub fn gate(
        project: ProjectId,
        requirement: ExpertGateRequirement,
        binding: ExpertGrantBinding,
    ) -> Self {
        Self {
            project: Some(project),
            authority: ExpertFenceAuthority::Gate,
            gate: Some(requirement),
            binding,
            evaluator: None,
        }
    }

    /// The same grant naming `evaluator_id_digest` as the evaluator the readmission report comes
    /// from; see [`ExpertAdmissionGrant::with_evaluator`].
    #[must_use]
    pub fn with_evaluator(mut self, evaluator_id_digest: [u8; 32]) -> Self {
        self.evaluator = Some(evaluator_id_digest);
        self
    }

    /// The evaluator this grant names, if any.
    pub fn evaluator(&self) -> Option<&[u8; 32]> {
        self.evaluator.as_ref()
    }

    /// The project the grant is scoped to; `None` for a global revocation grant.
    pub fn project(&self) -> Option<&ProjectId> {
        self.project.as_ref()
    }

    pub fn authority(&self) -> ExpertFenceAuthority {
        self.authority
    }

    pub fn binding(&self) -> &ExpertGrantBinding {
        &self.binding
    }

    fn kind(&self) -> ExpertGrantKind {
        match self.authority {
            ExpertFenceAuthority::RuntimeRevocation => ExpertGrantKind::Fence,
            ExpertFenceAuthority::Gate => ExpertGrantKind::Gate,
            ExpertFenceAuthority::GlobalRevocation => ExpertGrantKind::GlobalRevocation,
        }
    }
}

/// The expert a fence or readmission names, as the caller last observed it: the generation and
/// state version are the compare-and-set operands.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertFenceRequest {
    pub project: ProjectId,
    pub pod: PodId,
    pub generation: Generation,
    pub expected_state_version: u64,
    /// The cause of the fence, or for a readmission the cause of the fence it clears.
    pub cause: ExpertFenceCause,
    /// The committed revocation an `InputRevoked` fence or escalation rests on: the `subject` and
    /// `generation` of a `Revoked` (or `ProcedureRevoked`, subject `procedure:{id}`) record the
    /// ledger already holds. Required for exactly those two; the runtime verifies it against its
    /// committed revocations and refuses a fence naming one it does not hold.
    pub revocation: Option<ExpertFenceRevocation>,
}

/// One invalidation of an input: revoke it, or supersede it by a newer generation. Both fence
/// every expert that rests on the input in the same committed event; see
/// [`PtrRuntime::invalidate_input`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InputInvalidation {
    input: String,
    generation: Generation,
    binding_digest: [u8; 32],
    kind: InputInvalidationKind,
}

impl InputInvalidation {
    /// Revoke `input` at `generation`. `binding_digest` is the content digest of the input at
    /// that generation; it must equal the digest the first committed use fixed, when there was
    /// one.
    pub fn revoked(
        input: impl Into<String>,
        generation: Generation,
        binding_digest: [u8; 32],
    ) -> Self {
        Self {
            input: input.into(),
            generation,
            binding_digest,
            kind: InputInvalidationKind::Revoked,
        }
    }

    /// Supersede `input` at `old_generation` (whose content digest is `binding_digest`) by
    /// `new_generation` with content digest `new_digest`. `replacement_ref` is a digest of the
    /// replacement artifact, recorded as evidence only. Every generation below `new_generation`
    /// is stale afterwards.
    pub fn superseded(
        input: impl Into<String>,
        old_generation: Generation,
        binding_digest: [u8; 32],
        new_generation: Generation,
        new_digest: [u8; 32],
        replacement_ref: [u8; 32],
    ) -> Self {
        Self {
            input: input.into(),
            generation: old_generation,
            binding_digest,
            kind: InputInvalidationKind::Superseded {
                old_generation,
                new_generation,
                new_digest,
                replacement_ref,
            },
        }
    }

    pub fn input(&self) -> &str {
        &self.input
    }

    /// The invalidated generation (the old generation of a supersession).
    pub fn generation(&self) -> Generation {
        self.generation
    }

    pub fn binding_digest(&self) -> &[u8; 32] {
        &self.binding_digest
    }

    pub fn kind(&self) -> InputInvalidationKind {
        self.kind
    }
}

/// One expert a compound invalidation fences, as the fence set records it: the expert at its
/// generation and the state version the fence compares against. Its new fence epoch is
/// `expected_state_version + 1`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertFenceSetMember {
    pub project: ProjectId,
    pub pod: PodId,
    pub generation: Generation,
    pub expected_state_version: u64,
    /// `Fence` for a `Serving` or `Frozen` expert, `Escalate` for a quarantined one.
    pub action: ExpertFenceAction,
    /// The state the expert returns to when the fence is cleared (for a quarantined expert that
    /// is now final, the state it was fenced from).
    pub resume: ExpertFenceResume,
}

impl ExpertFenceSetMember {
    /// The fence epoch the expert stands at once fenced.
    pub fn epoch(&self) -> u64 {
        self.expected_state_version.saturating_add(1)
    }
}

/// What a committed [`PtrRuntime::invalidate_input`] did.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InputInvalidationOutcome {
    index: CommitIndex,
    fence_digest: [u8; 32],
    fenced: Vec<ExpertFenceSetMember>,
}

impl InputInvalidationOutcome {
    /// The commit index of the one `InputInvalidated` event.
    pub fn index(&self) -> CommitIndex {
        self.index
    }

    /// The number of experts the event fenced, as the event carries it.
    pub fn fence_count(&self) -> u32 {
        // The count was checked to fit the event's `u32` before the event was built.
        self.fenced.len() as u32
    }

    /// The digest the event carries over the fence set.
    pub fn fence_digest(&self) -> &[u8; 32] {
        &self.fence_digest
    }

    /// The fenced experts in canonical order. Held for the caller only: the event itself does
    /// not list them.
    pub fn fenced(&self) -> &[ExpertFenceSetMember] {
        &self.fenced
    }

    /// The fence epoch now in force per `expert:{project}/{pod}` identity, the map a standalone
    /// KV backend or catalog takes to make the fenced experts' entries dead (see
    /// [`expert_fence_epochs`], which is the same map for a plain fence event).
    pub fn fence_epochs(&self) -> BTreeMap<String, u64> {
        fence_set_epochs(&self.fenced)
    }
}

fn fence_set_epochs(members: &[ExpertFenceSetMember]) -> BTreeMap<String, u64> {
    members
        .iter()
        .map(|member| {
            (
                ptr_state::expert_key(&member.project.0, &member.pod.0),
                member.epoch(),
            )
        })
        .collect()
}

fn put_text(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

const FENCE_SET_DOMAIN: &str = "ptr.expert.input-invalidation.fence-set.v1";

/// The digest an `InputInvalidated` event carries over its fence set: SHA-256 over the
/// length-prefixed domain `ptr.expert.input-invalidation.fence-set.v1`, the length-prefixed
/// `input`, the invalidated `generation`, the member count and, for each member in the order
/// given (canonical order is ascending `(project, pod)`), the length-prefixed project and pod,
/// the expert generation, the expected state version, the action code and the resume code.
/// Integers are little-endian `u64`, codes are one byte. A harness uses this to check the digest
/// of an event against the set it independently derived.
pub fn expert_fence_set_digest(
    input: &str,
    generation: Generation,
    members: &[ExpertFenceSetMember],
) -> [u8; 32] {
    let mut bytes = Vec::new();
    put_text(&mut bytes, FENCE_SET_DOMAIN);
    put_text(&mut bytes, input);
    bytes.extend_from_slice(&generation.0.to_le_bytes());
    bytes.extend_from_slice(&(members.len() as u64).to_le_bytes());
    for member in members {
        put_text(&mut bytes, &member.project.0);
        put_text(&mut bytes, &member.pod.0);
        bytes.extend_from_slice(&member.generation.0.to_le_bytes());
        bytes.extend_from_slice(&member.expected_state_version.to_le_bytes());
        bytes.push(member.action.code());
        bytes.push(member.resume.code());
    }
    ptr_ledger::integrity::sha256(&bytes)
}

/// How the evidence behind an expert's latest structural record was checked on replay.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpertEvidenceStatus {
    /// The record embeds its gate report and replay re-verified it.
    Verified,
    /// The record is a version 1 record with only a digest; nothing about its evidence could be
    /// re-verified.
    UnverifiedLegacy,
}

/// The authoritative state of one expert, as projected from admitted records.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertEntry {
    pub state: ExpertState,
    pub state_version: u64,
    pub generation: Generation,
    pub manifest_digest: ManifestDigest,
    pub level: ExpertLevel,
    pub base_model: BaseModelRef,
}

/// Experts by `(project, pod)`, rebuilt from the ledger by applying admitted records in order.
///
/// It is a projection: it changes only when an admitted record is applied, which the runtime does
/// for every committed [`LedgerEvent::ExpertStructuralCommitted`], live and on replay.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ExpertRegistryProjection {
    entries: BTreeMap<(String, String), ExpertEntry>,
    /// Experts whose latest record is unverified legacy; every other entry is verified.
    legacy: BTreeSet<(String, String)>,
    /// The gate configuration digest the first verified record of each gate epoch required.
    epoch_gate_configs: BTreeMap<u64, [u8; 32]>,
    /// The cause of the fence every `Fenced` expert is under, and the state it returns to on
    /// readmission. An expert absent here but `Fenced` (restored from a snapshot that did not
    /// record causes) is neither readmitted nor escalated.
    fence_causes: BTreeMap<(String, String), FenceStanding>,
    /// The highest gate epoch any readmission was admitted under; a later readmission may not
    /// stand at a lower one. `0` before any readmission.
    readmit_gate_epoch: u64,
    /// The highest epoch each expert was ever fenced at. Never lowered, kept after a
    /// readmission; see [`ExpertRegistryProjection::highest_fenced_epoch`].
    fenced_epochs: BTreeMap<(String, String), u64>,
    /// Which experts rest on which inputs, the source edges between experts, the input-digest
    /// registry and the committed invalidations. Projected from version 3 records and
    /// `InputInvalidated` events; see [`ExpertDependencyIndex`].
    deps: ExpertDependencyIndex,
    /// The standing of the gate configuration and the evaluator and the epochs each expert was
    /// admitted under; see the `authority` module. Projected from `ExpertAuthorityChanged`
    /// events, structural records and readmissions.
    authority: AuthorityState,
    /// The current grant generation per [`ExpertGrantKind`]. Host authority in memory, not
    /// projected from the ledger: it takes no part in equality and a rebuilt registry starts at
    /// [`ExpertGrantGeneration::INITIAL`].
    grants: HostGrantGenerations,
}

/// The current grant generation of each [`ExpertGrantKind`].
#[derive(Clone, Copy, Debug)]
struct HostGrantGenerations {
    admission: u64,
    gate: u64,
    fence: u64,
    global: u64,
    /// The highest host clock reading the runtime has been told of, see
    /// [`PtrRuntime::observe_host_time`].
    time_floor: Timestamp,
}

impl Default for HostGrantGenerations {
    fn default() -> Self {
        let initial = ExpertGrantGeneration::INITIAL.get();
        Self {
            admission: initial,
            gate: initial,
            fence: initial,
            global: initial,
            time_floor: Timestamp(0),
        }
    }
}

impl HostGrantGenerations {
    fn slot(&mut self, kind: ExpertGrantKind) -> &mut u64 {
        match kind {
            ExpertGrantKind::Admission => &mut self.admission,
            ExpertGrantKind::Gate => &mut self.gate,
            ExpertGrantKind::Fence => &mut self.fence,
            ExpertGrantKind::GlobalRevocation => &mut self.global,
        }
    }

    fn current(&self, kind: ExpertGrantKind) -> u64 {
        match kind {
            ExpertGrantKind::Admission => self.admission,
            ExpertGrantKind::Gate => self.gate,
            ExpertGrantKind::Fence => self.fence,
            ExpertGrantKind::GlobalRevocation => self.global,
        }
    }
}

/// Grant generations are host authority, not projected state: two registries that agree on every
/// expert are equal whatever grants the host has superseded, so a registry compared with a fresh
/// replay of its ledger stays equal.
impl PartialEq for HostGrantGenerations {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Eq for HostGrantGenerations {}

/// Why a `Fenced` expert is fenced and what it returns to when readmitted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FenceStanding {
    pub(crate) cause: ExpertFenceCause,
    pub(crate) resume: ExpertFenceResume,
}

/// What applying one fence event changes.
struct FencePlan {
    key: (String, String),
    entry: ExpertEntry,
    /// `Some` when the expert is now fenced under that standing, `None` when the fence is
    /// cleared.
    cause: Option<FenceStanding>,
    /// `Some` when the event fenced the expert at that epoch.
    fenced_at: Option<u64>,
    verified: Option<VerifiedEvidence>,
}

type Plan = Vec<((String, String), ExpertEntry)>;

/// What applying one record changes: the entries, how its evidence was checked and what it adds
/// to the dependency index.
struct ExpertPlan {
    entries: Plan,
    verified: Option<VerifiedEvidence>,
    dependency: DependencyUpdate,
}

/// The experts an invalidation fences, with the count and digest the event commits to.
pub(crate) struct FenceSet {
    members: Vec<ExpertFenceSetMember>,
    count: u32,
    digest: [u8; 32],
}

/// The facts a verified record contributes to the projection.
struct VerifiedEvidence {
    epoch: u64,
    gate_config_digest: [u8; 32],
}

impl ExpertRegistryProjection {
    pub fn get(&self, project: &ProjectId, pod: &PodId) -> Option<&ExpertEntry> {
        self.entries.get(&(project.0.clone(), pod.0.clone()))
    }

    /// A registry holding exactly `entries`, with the evidence marks a compacted snapshot
    /// carries. The snapshot reader has already checked order, identities and that every mark
    /// names an entry, and cross-checks the result against the materialized `expert:` keys.
    pub(crate) fn from_parts(
        entries: Vec<((String, String), ExpertEntry)>,
        legacy: BTreeSet<(String, String)>,
        epoch_gate_configs: BTreeMap<u64, [u8; 32]>,
        fence_causes: BTreeMap<(String, String), FenceStanding>,
        fenced_epochs: BTreeMap<(String, String), u64>,
        readmit_gate_epoch: u64,
        deps: ExpertDependencyIndex,
    ) -> Self {
        Self {
            entries: entries.into_iter().collect(),
            legacy,
            epoch_gate_configs,
            fence_causes,
            fenced_epochs,
            readmit_gate_epoch,
            deps,
            authority: AuthorityState::default(),
            grants: HostGrantGenerations::default(),
        }
    }

    /// The same registry with the authority state a compacted snapshot carries.
    #[must_use]
    pub(crate) fn with_authority(mut self, authority: AuthorityState) -> Self {
        self.authority = authority;
        self
    }

    pub(crate) fn authority_state(&self) -> &AuthorityState {
        &self.authority
    }

    /// A registry restored from a snapshot that does not say how its evidence was checked
    /// (PTRCS005): every expert is unverified legacy and no epoch configuration is known.
    pub(crate) fn from_entries_all_legacy(entries: Vec<((String, String), ExpertEntry)>) -> Self {
        let legacy = entries.iter().map(|(key, _)| key.clone()).collect();
        // A snapshot from before the memo: every fenced expert was fenced at its current epoch.
        let fenced_epochs = fenced_at_current_epoch(&entries);
        Self::from_parts(
            entries,
            legacy,
            BTreeMap::new(),
            BTreeMap::new(),
            fenced_epochs,
            0,
            ExpertDependencyIndex::default(),
        )
    }

    /// The dependency index: recorded inputs, source edges, input digests and committed
    /// invalidations.
    pub fn dependencies(&self) -> &ExpertDependencyIndex {
        &self.deps
    }

    pub(crate) fn dependency_index(&self) -> &ExpertDependencyIndex {
        &self.deps
    }

    pub(crate) fn legacy_keys(&self) -> impl Iterator<Item = (&str, &str)> {
        self.legacy.iter().map(|(p, d)| (p.as_str(), d.as_str()))
    }

    pub(crate) fn fenced_epoch_entries(&self) -> impl Iterator<Item = (&str, &str, u64)> {
        self.fenced_epochs
            .iter()
            .map(|((project, pod), epoch)| (project.as_str(), pod.as_str(), *epoch))
    }

    /// The highest epoch the expert `identity` (`expert:{project}/{pod}`) was ever fenced at, or
    /// `None` when it never was. A dependency whose recorded fence epoch is `0` or at most this
    /// is never live: it was recorded before the fence ended it, and a readmission does not
    /// revive it.
    pub fn highest_fenced_epoch(&self, identity: &str) -> Option<u64> {
        let (project, pod) = identity
            .strip_prefix(ptr_state::EXPERT_PREFIX)?
            .split_once('/')?;
        self.fenced_epochs
            .get(&(project.to_owned(), pod.to_owned()))
            .copied()
    }

    pub(crate) fn fence_cause_entries(&self) -> impl Iterator<Item = (&str, &str, FenceStanding)> {
        self.fence_causes
            .iter()
            .map(|((project, pod), standing)| (project.as_str(), pod.as_str(), *standing))
    }

    pub(crate) fn readmit_gate_epoch(&self) -> u64 {
        self.readmit_gate_epoch
    }

    /// The cause `pod`'s fence was committed for, or `None` when it is not fenced or the cause
    /// is unknown.
    pub fn fence_cause(&self, project: &ProjectId, pod: &PodId) -> Option<ExpertFenceCause> {
        self.fence_causes
            .get(&(project.0.clone(), pod.0.clone()))
            .map(|standing| standing.cause)
    }

    /// The state `pod` returns to when readmitted, or `None` when it is not fenced or the fence
    /// predates recorded causes.
    pub fn fence_resume(&self, project: &ProjectId, pod: &PodId) -> Option<ExpertFenceResume> {
        self.fence_causes
            .get(&(project.0.clone(), pod.0.clone()))
            .map(|standing| standing.resume)
    }

    pub(crate) fn epoch_configs(&self) -> impl Iterator<Item = (u64, [u8; 32])> + '_ {
        self.epoch_gate_configs.iter().map(|(e, d)| (*e, *d))
    }

    /// How the evidence behind `pod`'s latest structural record was checked, or `None` when the
    /// registry holds no such expert. An expert restored from a compacted snapshot reads as
    /// verified: the snapshot does not carry the legacy marks.
    pub fn evidence_status(
        &self,
        project: &ProjectId,
        pod: &PodId,
    ) -> Option<ExpertEvidenceStatus> {
        let key = (project.0.clone(), pod.0.clone());
        if !self.entries.contains_key(&key) {
            return None;
        }
        Some(if self.legacy.contains(&key) {
            ExpertEvidenceStatus::UnverifiedLegacy
        } else {
            ExpertEvidenceStatus::Verified
        })
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether `pod` at `generation` may be routed to: it is in the registry at exactly that
    /// generation, `Serving` or `Frozen`, and not fenced.
    fn route_admission(
        &self,
        project: &ProjectId,
        pod: &PodId,
        generation: Generation,
    ) -> Result<(), RouteRefusal> {
        let entry = self.get(project, pod).ok_or(RouteRefusal::UnknownExpert)?;
        if entry.generation != generation {
            return Err(RouteRefusal::GenerationMismatch {
                current: entry.generation,
            });
        }
        match entry.state {
            ExpertState::Serving | ExpertState::Frozen => {}
            ExpertState::Fenced => return Err(RouteRefusal::Fenced),
            state @ (ExpertState::Candidate
            | ExpertState::Gated
            | ExpertState::Superseded
            | ExpertState::Retired(_)) => return Err(RouteRefusal::NotServing { state }),
        }
        self.dependencies_live(project, pod, entry)
    }

    /// The liveness of what an expert was built from, as the dependency index knows it: an
    /// expert admitted with a dependency block, or descending from experts that were, is refused
    /// when one of its own recorded inputs or of any ancestor's is revoked or below a
    /// supersession floor, fenced or not (a retrained successor is exempt for a supersession,
    /// see [`ExpertDependencyIndex::first_dead_input`]). An expert whose lineage named no inputs
    /// has nothing to check. (The runtime adds the lifecycle authority's own view in
    /// [`PtrRuntime::expert_route_admission`].)
    fn dependencies_live(
        &self,
        project: &ProjectId,
        pod: &PodId,
        entry: &ExpertEntry,
    ) -> Result<(), RouteRefusal> {
        let node = (project.0.clone(), pod.0.clone(), entry.generation);
        let status =
            |identity: &str, generation: Generation| self.deps.status(identity, generation);
        match self.deps.first_dead_input_of(&node, &status) {
            Some(dead) => Err(RouteRefusal::DependencyNotLive {
                identity: dead.identity,
                generation: dead.generation,
            }),
            None => Ok(()),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Every entry in key order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str, &ExpertEntry)> {
        self.entries
            .iter()
            .map(|((project, pod), entry)| (project.as_str(), pod.as_str(), entry))
    }

    /// Compare-and-set `record` against this projection; the entries it would write.
    fn plan(&self, record: &ExpertCommitRecord) -> Result<Plan, ExpertAdmissionError> {
        let proposal = record.proposal();
        let project = proposal.project();
        let op = proposal.op();
        let key = |pod: &PodId| (project.0.clone(), pod.0.clone());
        let sources = op.sources();
        for source in &sources {
            let entry = self.entries.get(&key(source.pod())).ok_or_else(|| {
                ExpertAdmissionError::UnknownOperand {
                    pod: source.pod().0.clone(),
                }
            })?;
            let stale = |field| ExpertAdmissionError::StaleOperand {
                pod: source.pod().0.clone(),
                field,
            };
            if entry.generation != source.generation() {
                return Err(stale(ExpertStaleField::Generation));
            }
            if entry.manifest_digest != *source.manifest_digest() {
                return Err(stale(ExpertStaleField::ManifestDigest));
            }
            if entry.state != source.state() {
                return Err(stale(ExpertStaleField::State));
            }
            if entry.state_version != source.state_version() {
                return Err(stale(ExpertStaleField::StateVersion));
            }
            if entry.level != source.level() {
                return Err(stale(ExpertStaleField::Level));
            }
            if entry.base_model != *source.base_model() {
                return Err(stale(ExpertStaleField::BaseModel));
            }
        }
        let takeover = match op.kind() {
            OpKind::Reactivate => sources.first().map(|target| target.pod().clone()),
            _ => None,
        };
        for result in op.results() {
            if takeover.as_ref() != Some(result.pod())
                && self.entries.contains_key(&key(result.pod()))
            {
                return Err(ExpertAdmissionError::ResultPodOccupied {
                    pod: result.pod().0.clone(),
                });
            }
            if let ExpertLevel::Token { layer, index } = result.level() {
                let occupied = self
                    .entries
                    .iter()
                    .any(|((entry_project, entry_pod), entry)| {
                        entry_project == &project.0
                            && entry.level == result.level()
                            && occupies_slot(entry.state)
                            && !sources.iter().any(|source| source.pod().0 == *entry_pod)
                    });
                if occupied {
                    return Err(ExpertAdmissionError::ResultSlotOccupied { layer, index });
                }
            }
        }
        Ok(op
            .outcomes()
            .into_iter()
            .map(|outcome| {
                (
                    key(&outcome.pod),
                    ExpertEntry {
                        state: outcome.state,
                        state_version: outcome.state_version,
                        generation: outcome.generation,
                        manifest_digest: outcome.manifest_digest,
                        level: outcome.level,
                        base_model: outcome.base_model,
                    },
                )
            })
            .collect())
    }

    /// Compare-and-set a fence event against this projection; the entry it would write.
    fn plan_fence(&self, event: &LedgerEvent) -> Result<FencePlan, ExpertAdmissionError> {
        let LedgerEvent::ExpertFenceCommitted {
            project,
            pod,
            action,
            cause,
            resume,
            generation,
            expected_state_version,
            epoch,
            evidence,
            ..
        } = event
        else {
            return Err(ExpertAdmissionError::InvalidRecord(
                "not a fence event".to_owned(),
            ));
        };
        let key = (project.0.clone(), pod.0.clone());
        let resume_mismatch = || ExpertAdmissionError::FenceResumeMismatch { pod: pod.0.clone() };
        let entry = self
            .entries
            .get(&key)
            .ok_or_else(|| ExpertAdmissionError::UnknownOperand { pod: pod.0.clone() })?;
        let stale = |field| ExpertAdmissionError::StaleOperand {
            pod: pod.0.clone(),
            field,
        };
        if entry.generation != *generation {
            return Err(stale(ExpertStaleField::Generation));
        }
        if entry.state_version != *expected_state_version {
            return Err(stale(ExpertStaleField::StateVersion));
        }
        if expected_state_version.checked_add(1) != Some(*epoch) {
            return Err(ExpertAdmissionError::FenceEpochMismatch { pod: pod.0.clone() });
        }
        let (state, recorded, verified) = match action {
            ExpertFenceAction::Fence => {
                if evidence.is_some() {
                    return Err(ExpertAdmissionError::FenceEvidenceUnexpected {
                        pod: pod.0.clone(),
                    });
                }
                if !matches!(entry.state, ExpertState::Serving | ExpertState::Frozen) {
                    return Err(ExpertAdmissionError::NotFenceable {
                        pod: pod.0.clone(),
                        state: entry.state,
                    });
                }
                // The fence records the state it leaves, which readmission restores.
                let leaves = match entry.state {
                    ExpertState::Frozen => ExpertFenceResume::Frozen,
                    _ => ExpertFenceResume::Serving,
                };
                if *resume != leaves {
                    return Err(resume_mismatch());
                }
                (
                    ExpertState::Fenced,
                    Some(FenceStanding {
                        cause: *cause,
                        resume: *resume,
                    }),
                    None,
                )
            }
            ExpertFenceAction::Escalate => {
                if evidence.is_some() {
                    return Err(ExpertAdmissionError::FenceEvidenceUnexpected {
                        pod: pod.0.clone(),
                    });
                }
                if entry.state != ExpertState::Fenced {
                    return Err(ExpertAdmissionError::NotFenced {
                        pod: pod.0.clone(),
                        state: entry.state,
                    });
                }
                // The cause only tightens: AuthorityQuarantine to InputRevoked, never back.
                let standing = match self.fence_causes.get(&key) {
                    Some(standing)
                        if standing.cause == ExpertFenceCause::AuthorityQuarantine
                            && *cause == ExpertFenceCause::InputRevoked =>
                    {
                        *standing
                    }
                    _ => {
                        return Err(ExpertAdmissionError::NotEscalatable { pod: pod.0.clone() });
                    }
                };
                if *resume != standing.resume {
                    return Err(resume_mismatch());
                }
                (
                    ExpertState::Fenced,
                    Some(FenceStanding {
                        cause: *cause,
                        resume: standing.resume,
                    }),
                    None,
                )
            }
            ExpertFenceAction::Retire => {
                if evidence.is_some() {
                    return Err(ExpertAdmissionError::FenceEvidenceUnexpected {
                        pod: pod.0.clone(),
                    });
                }
                if entry.state != ExpertState::Fenced {
                    return Err(ExpertAdmissionError::NotFenced {
                        pod: pod.0.clone(),
                        state: entry.state,
                    });
                }
                // Only a revoked input retires: the fence must be `InputRevoked` and so must the
                // event. The expert leaves the fence registry for good: `Retired` is terminal.
                let standing = match self.fence_causes.get(&key) {
                    Some(standing)
                        if standing.cause == ExpertFenceCause::InputRevoked
                            && *cause == ExpertFenceCause::InputRevoked =>
                    {
                        *standing
                    }
                    _ => {
                        return Err(ExpertAdmissionError::NotRetirable { pod: pod.0.clone() });
                    }
                };
                if *resume != standing.resume {
                    return Err(resume_mismatch());
                }
                (
                    ExpertState::Retired(RetiredReason::InputRevoked),
                    None,
                    None,
                )
            }
            ExpertFenceAction::Readmit => {
                if entry.state != ExpertState::Fenced {
                    return Err(ExpertAdmissionError::NotFenced {
                        pod: pod.0.clone(),
                        state: entry.state,
                    });
                }
                let standing = match self.fence_causes.get(&key) {
                    Some(standing) if standing.cause == ExpertFenceCause::AuthorityQuarantine => {
                        *standing
                    }
                    _ => {
                        return Err(ExpertAdmissionError::NotReadmittable { pod: pod.0.clone() });
                    }
                };
                if *cause != ExpertFenceCause::AuthorityQuarantine {
                    return Err(ExpertAdmissionError::FenceCauseMismatch { pod: pod.0.clone() });
                }
                if *resume != standing.resume {
                    return Err(resume_mismatch());
                }
                let evidence = evidence.as_ref().ok_or_else(|| {
                    ExpertAdmissionError::FenceEvidenceMissing { pod: pod.0.clone() }
                })?;
                let verified = self.verify_readmission_evidence(event, evidence, entry)?;
                let state = match standing.resume {
                    ExpertFenceResume::Serving => ExpertState::Serving,
                    ExpertFenceResume::Frozen => ExpertState::Frozen,
                };
                (state, None, Some(verified))
            }
        };
        Ok(FencePlan {
            key,
            entry: ExpertEntry {
                state,
                state_version: *epoch,
                ..entry.clone()
            },
            cause: recorded,
            fenced_at: matches!(
                action,
                ExpertFenceAction::Fence | ExpertFenceAction::Escalate
            )
            .then_some(*epoch),
            verified,
        })
    }

    /// Re-verify the evidence a readmission embeds, at commit and on replay alike: the report
    /// decodes canonically and is evidence for exactly this readmission of `entry` (the shared
    /// `check_report`: subject ids, artifact and base model of the expert, gate configuration
    /// the event says was required, passed), and it agrees with every other record of its gate
    /// epoch. The evaluation manifest is not re-checked: the grant that named it is not
    /// recorded.
    fn verify_readmission_evidence(
        &self,
        event: &LedgerEvent,
        evidence: &ExpertFenceEvidence,
        entry: &ExpertEntry,
    ) -> Result<VerifiedEvidence, ExpertAdmissionError> {
        let LedgerEvent::ExpertFenceCommitted {
            project,
            pod,
            generation,
            expected_state_version,
            epoch,
            cause,
            ..
        } = event
        else {
            return Err(ExpertAdmissionError::InvalidRecord(
                "not a fence event".to_owned(),
            ));
        };
        let report = BoundGateReport::decode_canonical(&evidence.report)
            .map_err(|error| ExpertAdmissionError::EvidenceUndecodable(error.to_string()))?;
        self.check_authority(&EvidenceClaim {
            gate_epoch: evidence.gate_epoch,
            gate_config_digest: evidence.required_gate_config_digest,
            evaluator: EvaluatorClaim::NotRechecked,
        })?;
        let expected = readmission_expectation(
            ReadmissionTarget {
                project,
                pod,
                generation: *generation,
                expected_state_version: *expected_state_version,
                epoch: *epoch,
                cause: *cause,
            },
            entry,
            evidence.required_gate_config_digest,
            None,
        );
        check_report(&report, &expected)?;
        if evidence.gate_epoch < self.readmit_gate_epoch {
            return Err(ExpertAdmissionError::GateEpochRegressed {
                pod: pod.0.clone(),
                seen: self.readmit_gate_epoch,
                found: evidence.gate_epoch,
            });
        }
        if let Some(known) = self.epoch_gate_configs.get(&evidence.gate_epoch) {
            if *known != evidence.required_gate_config_digest {
                return Err(ExpertAdmissionError::EpochGateConfigConflict {
                    epoch: evidence.gate_epoch,
                });
            }
        }
        Ok(VerifiedEvidence {
            epoch: evidence.gate_epoch,
            gate_config_digest: evidence.required_gate_config_digest,
        })
    }

    fn apply_fence(&mut self, plan: FencePlan) {
        if let Some(verified) = plan.verified {
            self.epoch_gate_configs
                .entry(verified.epoch)
                .or_insert(verified.gate_config_digest);
            self.readmit_gate_epoch = self.readmit_gate_epoch.max(verified.epoch);
            // A readmission is fresh evidence: the expert stands under the current epochs.
            self.authority.record_admission(&plan.key, true);
        }
        if let Some(epoch) = plan.fenced_at {
            let highest = self.fenced_epochs.entry(plan.key.clone()).or_insert(0);
            *highest = (*highest).max(epoch);
        }
        match plan.cause {
            Some(standing) => {
                self.fence_causes.insert(plan.key.clone(), standing);
            }
            None => {
                self.fence_causes.remove(&plan.key);
            }
        }
        self.entries.insert(plan.key, plan.entry);
    }

    /// The experts an invalidation of `input` at `generation` fences, derived from the
    /// dependency index and the registry as they stand now, in canonical `(project, pod)` order,
    /// with the digest the event commits to.
    ///
    /// The index yields every expert generation that names the input (see
    /// [`ExpertDependencyIndex`]), closed over source edges; of those, the registry's current
    /// generation of each pod is kept when it is `Serving`, `Frozen`, or fenced for
    /// `AuthorityQuarantine` (escalated). An expert already fenced for `InputRevoked`, or fenced
    /// with an unknown cause, is final and not part of the set; a candidate, gated, superseded or
    /// retired expert is not live.
    fn fence_set(
        &self,
        input: &str,
        generation: Generation,
        kind: &InputInvalidationKind,
    ) -> Result<FenceSet, ExpertAdmissionError> {
        let superseded_by = match kind {
            InputInvalidationKind::Revoked => None,
            InputInvalidationKind::Superseded { new_generation, .. } => Some(*new_generation),
        };
        let mut members = Vec::new();
        for (project, pod, expert_generation) in
            self.deps.affected(input, generation, superseded_by)
        {
            let key = (project, pod);
            let Some(entry) = self.entries.get(&key) else {
                continue;
            };
            if entry.generation != expert_generation {
                continue;
            }
            let (action, resume) = match entry.state {
                ExpertState::Serving => (ExpertFenceAction::Fence, ExpertFenceResume::Serving),
                ExpertState::Frozen => (ExpertFenceAction::Fence, ExpertFenceResume::Frozen),
                ExpertState::Fenced => match self.fence_causes.get(&key) {
                    Some(standing) if standing.cause == ExpertFenceCause::AuthorityQuarantine => {
                        (ExpertFenceAction::Escalate, standing.resume)
                    }
                    _ => continue,
                },
                ExpertState::Candidate
                | ExpertState::Gated
                | ExpertState::Superseded
                | ExpertState::Retired(_) => continue,
            };
            if entry.state_version.checked_add(1).is_none() {
                return Err(ExpertAdmissionError::FenceEpochMismatch { pod: key.1 });
            }
            members.push(ExpertFenceSetMember {
                project: ProjectId(key.0),
                pod: PodId(key.1),
                generation: entry.generation,
                expected_state_version: entry.state_version,
                action,
                resume,
            });
        }
        let count = u32::try_from(members.len()).map_err(|_| ExpertAdmissionError::Unencodable)?;
        let digest = expert_fence_set_digest(input, generation, &members);
        Ok(FenceSet {
            members,
            count,
            digest,
        })
    }

    /// Fence every member of `set` in one step: `Fenced` at the next state version, with
    /// `cause`, the fence memo raised. The set was derived against this very registry, so every
    /// member is present and at the state it compares against.
    fn apply_fence_set(&mut self, set: &FenceSet, cause: ExpertFenceCause) {
        for member in &set.members {
            let key = (member.project.0.clone(), member.pod.0.clone());
            let epoch = member.epoch();
            if let Some(entry) = self.entries.get_mut(&key) {
                entry.state = ExpertState::Fenced;
                entry.state_version = epoch;
            }
            self.fence_causes.insert(
                key.clone(),
                FenceStanding {
                    cause,
                    resume: member.resume,
                },
            );
            let highest = self.fenced_epochs.entry(key).or_insert(0);
            *highest = (*highest).max(epoch);
        }
    }

    fn apply(&mut self, plan: ExpertPlan) {
        self.deps.register(&plan.dependency);
        for (key, _) in &plan.entries {
            if plan.verified.is_some() {
                self.legacy.remove(key);
            } else {
                self.legacy.insert(key.clone());
            }
            self.authority
                .record_admission(key, plan.verified.is_some());
        }
        if let Some(verified) = plan.verified {
            self.epoch_gate_configs
                .entry(verified.epoch)
                .or_insert(verified.gate_config_digest);
        }
        self.entries.extend(plan.entries);
    }
}

/// Whether an expert in `state` still holds its routing slot. A withdrawn or retired expert has
/// released it; every other state, a fenced one included, keeps it.
fn occupies_slot(state: ExpertState) -> bool {
    !matches!(
        state,
        ExpertState::Superseded
            | ExpertState::Retired(RetiredReason::Pruned)
            | ExpertState::Retired(RetiredReason::Superseded)
            | ExpertState::Retired(RetiredReason::GateFailed)
            | ExpertState::Retired(RetiredReason::InputRevoked)
    )
}

/// The ledger code for a proposal-level operation kind.
pub fn ledger_op_kind(kind: OpKind) -> ExpertOpKind {
    match kind {
        OpKind::Birth => ExpertOpKind::Birth,
        OpKind::Merge => ExpertOpKind::Merge,
        OpKind::Split => ExpertOpKind::Split,
        OpKind::Freeze => ExpertOpKind::Freeze,
        OpKind::Reactivate => ExpertOpKind::Reactivate,
        OpKind::Prune => ExpertOpKind::Prune,
        OpKind::Retire => ExpertOpKind::Retire,
    }
}

/// The event for `record`, with every id derived from it.
fn event_for(record: &ExpertCommitRecord) -> LedgerEvent {
    let proposal = record.proposal();
    LedgerEvent::ExpertStructuralCommitted {
        project: proposal.project().clone(),
        op_kind: ledger_op_kind(proposal.op().kind()),
        proposal_id: *proposal.proposal_id().as_bytes(),
        operation_id: *proposal.operation_id().as_bytes(),
        evidence_digest: *record.evidence_digest(),
        record: record.encode_canonical(),
    }
}

/// What a gate report must be evidence for, as the runtime derives it from the operation (or
/// the readmitted expert) and the host's requirement, never from the report.
struct ExpectedEvidence {
    proposal_id: [u8; 32],
    operation_id: [u8; 32],
    artifact_digest: [u8; 32],
    base_model_digest: [u8; 32],
    gate_config_digest: [u8; 32],
    /// The evaluation manifest digest the host expects. `None` on replay, where the grant that
    /// named it is not available.
    eval_manifest_digest: Option<[u8; 32]>,
}

/// The one function that decides whether `report` is evidence for `expected`, for admission,
/// readmission and replay alike. Typed refusals, first failing check wins: subject proposal id,
/// operation id, artifact, base model, evaluation manifest (when expected), gate configuration,
/// then whether the report passed.
fn check_report(
    report: &BoundGateReport,
    expected: &ExpectedEvidence,
) -> Result<(), ExpertAdmissionError> {
    let subject = report.subject();
    if subject.proposal_id != expected.proposal_id {
        return Err(ExpertAdmissionError::EvidenceProposalMismatch);
    }
    if subject.operation_id != expected.operation_id {
        return Err(ExpertAdmissionError::EvidenceOperationMismatch);
    }
    if subject.artifact_digest != expected.artifact_digest {
        return Err(ExpertAdmissionError::EvidenceArtifactMismatch);
    }
    if subject.base_model_digest != expected.base_model_digest {
        return Err(ExpertAdmissionError::EvidenceBaseModelMismatch);
    }
    if let Some(eval) = expected.eval_manifest_digest {
        if subject.eval_manifest_digest != eval {
            return Err(ExpertAdmissionError::EvidenceEvalManifestMismatch);
        }
    }
    if report.gate_config_digest() != expected.gate_config_digest {
        return Err(ExpertAdmissionError::GateConfigMismatch);
    }
    if !report.passed() {
        return Err(ExpertAdmissionError::GateFailed);
    }
    Ok(())
}

const ARTIFACT_SET_DOMAIN_SPLIT: &str = "ptr.expert.evidence.artifact-set.split-results.v1";
const ARTIFACT_SET_DOMAIN_PRUNE: &str = "ptr.expert.evidence.artifact-set.prune-targets.v1";

/// SHA-256 over the length-prefixed `domain`, the count and each digest in order.
fn ordered_digest_set<'a>(
    domain: &str,
    digests: impl ExactSizeIterator<Item = &'a ManifestDigest>,
) -> [u8; 32] {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(domain.len() as u64).to_le_bytes());
    bytes.extend_from_slice(domain.as_bytes());
    bytes.extend_from_slice(&(digests.len() as u64).to_le_bytes());
    for digest in digests {
        bytes.extend_from_slice(digest.as_bytes());
    }
    ptr_ledger::integrity::sha256(&bytes)
}

/// The `artifact_digest` a [`GateSubject`](ptr_lineage::GateSubject) must carry for a gate report
/// to be evidence for `op`. A host's evaluator uses this to build the subject; admission and
/// replay use it to check one.
///
/// | operation | digest |
/// |---|---|
/// | `Birth`, `Merge` | the result's manifest digest |
/// | `Reactivate` creating a new generation | the result's manifest digest |
/// | `Reactivate` of a frozen expert (unfreeze), `Freeze`, `Retire` | the operand's manifest digest |
/// | `Prune` of one target | that target's manifest digest |
/// | `Prune` of several targets | SHA-256 over `"ptr.expert.evidence.artifact-set.prune-targets.v1"`, the target count and the target manifest digests in operand order |
/// | `Split` | SHA-256 over `"ptr.expert.evidence.artifact-set.split-results.v1"`, the result count and the result manifest digests in result order |
///
/// Each text is length-prefixed and each count is a little-endian `u64`, so no two sets share an
/// encoding and a set digest cannot equal a single manifest digest by construction of the input.
pub fn expert_evidence_artifact_digest(op: &StructuralOp) -> [u8; 32] {
    match op {
        StructuralOp::Birth { result } | StructuralOp::Merge { result, .. } => {
            *result.manifest_digest().as_bytes()
        }
        StructuralOp::Reactivate {
            result: Some(result),
            ..
        } => *result.manifest_digest().as_bytes(),
        StructuralOp::Reactivate {
            target,
            result: None,
        }
        | StructuralOp::Freeze { target }
        | StructuralOp::Retire { target, .. } => *target.manifest_digest().as_bytes(),
        StructuralOp::Prune { targets } => match targets.as_slice() {
            [only] => *only.manifest_digest().as_bytes(),
            many => ordered_digest_set(
                ARTIFACT_SET_DOMAIN_PRUNE,
                many.iter().map(|target| target.manifest_digest()),
            ),
        },
        StructuralOp::Split { results, .. } => ordered_digest_set(
            ARTIFACT_SET_DOMAIN_SPLIT,
            results.iter().map(|result| result.manifest_digest()),
        ),
    }
}

const ARTIFACT_INPUTS_DOMAIN: &str = "ptr.expert.evidence.artifact-inputs.v1";

/// The `artifact_digest` a [`GateSubject`](ptr_lineage::GateSubject) must carry for a gate report
/// to be evidence for `op` admitted with the dependency block `inputs` (a version 3 record).
///
/// For an operation that creates experts (`Birth`, `Merge`, `Split`, a `Reactivate` that creates
/// a new generation) it is SHA-256 over the length-prefixed domain
/// `ptr.expert.evidence.artifact-inputs.v1`, the 32 bytes of
/// [`expert_evidence_artifact_digest`]`(op)` and the 32 bytes of
/// [`ExpertInputs::canonical_digest`]: the gate therefore attests the inputs the proposer
/// names, and a report produced for another block, or for none, is not evidence for this one.
/// For an operation that creates no expert the block is refused as an invalid record and the
/// digest is the plain one.
pub fn expert_evidence_artifact_digest_with_inputs(
    op: &StructuralOp,
    inputs: &ExpertInputs,
) -> [u8; 32] {
    let plain = expert_evidence_artifact_digest(op);
    if op.results().is_empty() {
        return plain;
    }
    let mut bytes = Vec::new();
    put_text(&mut bytes, ARTIFACT_INPUTS_DOMAIN);
    bytes.extend_from_slice(&plain);
    bytes.extend_from_slice(&inputs.canonical_digest());
    ptr_ledger::integrity::sha256(&bytes)
}

/// The `base_model_digest` a [`GateSubject`](ptr_lineage::GateSubject) must carry for a gate
/// report to be evidence for `op`: the digest of the base model every operand and result is on
/// (proposal validation requires them to agree, so the first result, else the first operand,
/// stands for all). `None` only for an operation that names no expert, which no valid proposal
/// is.
pub fn expert_evidence_base_model_digest(op: &StructuralOp) -> Option<[u8; 32]> {
    let result = op
        .results()
        .first()
        .map(|r| *r.base_model().digest().as_bytes());
    result.or_else(|| {
        op.sources()
            .first()
            .map(|s| *s.base_model().digest().as_bytes())
    })
}

/// The expectation for a structural record: ids from the proposal, artifact and base model from
/// its operation.
fn proposal_expectation(
    proposal: &ExpertProposal,
    inputs: Option<&ExpertInputs>,
    gate_config_digest: [u8; 32],
    eval_manifest_digest: Option<[u8; 32]>,
) -> Result<ExpectedEvidence, ExpertAdmissionError> {
    // A proposal always names at least one result or operand; an operation that does not has
    // no base model to bind and is no evidence target.
    let base_model_digest = expert_evidence_base_model_digest(proposal.op()).ok_or_else(|| {
        ExpertAdmissionError::InvalidRecord("operation names no expert".to_owned())
    })?;
    Ok(ExpectedEvidence {
        proposal_id: *proposal.proposal_id().as_bytes(),
        operation_id: *proposal.operation_id().as_bytes(),
        artifact_digest: match inputs {
            Some(inputs) => expert_evidence_artifact_digest_with_inputs(proposal.op(), inputs),
            None => expert_evidence_artifact_digest(proposal.op()),
        },
        base_model_digest,
        gate_config_digest,
        eval_manifest_digest,
    })
}

/// The readmission a report must be evidence for.
struct ReadmissionTarget<'a> {
    project: &'a ProjectId,
    pod: &'a PodId,
    generation: Generation,
    expected_state_version: u64,
    epoch: u64,
    cause: ExpertFenceCause,
}

/// The expectation for a readmission of `entry`: ids from [`expert_readmission_subject_ids`],
/// artifact and base model from the expert itself.
fn readmission_expectation(
    target: ReadmissionTarget<'_>,
    entry: &ExpertEntry,
    gate_config_digest: [u8; 32],
    eval_manifest_digest: Option<[u8; 32]>,
) -> ExpectedEvidence {
    let (proposal_id, operation_id) = expert_readmission_subject_ids(
        target.project,
        target.pod,
        target.generation,
        target.expected_state_version,
        target.epoch,
        target.cause,
    );
    ExpectedEvidence {
        proposal_id,
        operation_id,
        artifact_digest: *entry.manifest_digest.as_bytes(),
        base_model_digest: *entry.base_model.digest().as_bytes(),
        gate_config_digest,
        eval_manifest_digest,
    }
}

fn check_evidence(
    registry: &ExpertRegistryProjection,
    proposal: &ExpertProposal,
    inputs: Option<&ExpertInputs>,
    report: &BoundGateReport,
    grant: &ExpertAdmissionGrant,
) -> Result<(), ExpertAdmissionError> {
    if proposal.project() != grant.project() {
        return Err(ExpertAdmissionError::ProjectMismatch);
    }
    if proposal.proposer() != grant.proposer() {
        return Err(ExpertAdmissionError::ProposerMismatch);
    }
    if grant.requirement.require_inputs && inputs.is_none() {
        return Err(ExpertAdmissionError::InputsRequired);
    }
    // The grant's epoch and evaluator must be the current authority (shared with readmission and
    // replay).
    registry.check_authority(&EvidenceClaim {
        gate_epoch: grant.gate_epoch(),
        gate_config_digest: *grant.required_gate_config_digest(),
        evaluator: EvaluatorClaim::Named(grant.evaluator().copied()),
    })?;
    let expected = proposal_expectation(
        proposal,
        inputs,
        *grant.required_gate_config_digest(),
        Some(*grant.expected_eval_manifest_digest()),
    )?;
    check_report(report, &expected)
}

/// Whether an admission carries an attestation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Attested {
    No,
    Yes,
}

impl PtrRuntime {
    /// The expert registry as projected from admitted structural records.
    pub fn expert_registry(&self) -> &ExpertRegistryProjection {
        &self.experts
    }

    /// Admit `proposal` on `report` under the host's `grant` and commit it.
    ///
    /// See the module documentation for the checks and their order. On success the record is
    /// committed and the registry updated; on any error nothing is appended and the registry is
    /// unchanged. This is the only path that writes
    /// [`LedgerEvent::ExpertStructuralCommitted`]; [`PtrRuntime::commit`] refuses it.
    ///
    /// # Errors
    /// [`RuntimeError::Expert`] with the first failing check, or the runtime's own errors when
    /// execution is fenced or the append fails.
    pub fn admit_expert_proposal(
        &mut self,
        proposal: &ExpertProposal,
        report: &BoundGateReport,
        grant: &ExpertAdmissionGrant,
    ) -> Result<CommitIndex, RuntimeError> {
        self.admit_expert_record(proposal, report, grant, None, Attested::No)
    }

    /// [`PtrRuntime::admit_expert_proposal`] for a proposal whose result expert(s) rest on the
    /// training and evaluation inputs in `inputs`: commits a version 3 record carrying the
    /// dependency block. Besides every check of the plain path it refuses, typed and with nothing
    /// appended, a record that names a revoked input, an input generation that is superseded or
    /// stale, or an input digest other than the one the first committed use fixed
    /// ([`ExpertAdmissionError::InputRevoked`], [`ExpertAdmissionError::InputSuperseded`],
    /// [`ExpertAdmissionError::InputDigestStale`]), for the base model as for the training and
    /// evaluation inputs. The proposal must create at least one expert (`Birth`, `Merge`,
    /// `Split`, or a `Reactivate` of a retired expert); inputs on any other operation are refused
    /// as an invalid record.
    ///
    /// # Errors
    /// [`RuntimeError::Expert`] with the first failing check, or the runtime's own errors when
    /// execution is fenced or the append fails.
    pub fn admit_expert_proposal_with_inputs(
        &mut self,
        proposal: &ExpertProposal,
        report: &BoundGateReport,
        grant: &ExpertAdmissionGrant,
        inputs: ExpertInputs,
    ) -> Result<CommitIndex, RuntimeError> {
        self.admit_expert_record(proposal, report, grant, Some(inputs), Attested::No)
    }

    fn admit_expert_record(
        &mut self,
        proposal: &ExpertProposal,
        report: &BoundGateReport,
        grant: &ExpertAdmissionGrant,
        inputs: Option<ExpertInputs>,
        attested: Attested,
    ) -> Result<CommitIndex, RuntimeError> {
        if grant.attestation_required && attested == Attested::No {
            return Err(RuntimeError::Expert(
                ExpertAdmissionError::AttestationRequired,
            ));
        }
        check_evidence(&self.experts, proposal, inputs.as_ref(), report, grant)
            .map_err(RuntimeError::Expert)?;
        let invalid = |error: ptr_pods::ExpertProposalError| {
            RuntimeError::Expert(ExpertAdmissionError::InvalidRecord(error.to_string()))
        };
        let record = match inputs {
            None => ExpertCommitRecord::new_with_evidence(
                proposal.clone(),
                report.digest(),
                grant.gate_epoch(),
                *grant.required_gate_config_digest(),
                report.encode_canonical(),
            ),
            Some(inputs) => ExpertCommitRecord::new_with_inputs(
                proposal.clone(),
                report.digest(),
                grant.gate_epoch(),
                *grant.required_gate_config_digest(),
                report.encode_canonical(),
                inputs,
            ),
        }
        .map_err(invalid)?;
        let event = event_for(&record);
        self.validate_expert_event(&event)?;
        self.revalidate_grant(ExpertGrantKind::Admission, grant.binding())?;
        self.append_prepared(event, None)
    }

    /// The current grant generation of `kind`. A host reads it when it mints a grant.
    pub fn expert_grant_generation(&self, kind: ExpertGrantKind) -> ExpertGrantGeneration {
        ExpertGrantGeneration(self.experts.grants.current(kind))
    }

    /// Tell the runtime the host clock now reads `now`. The runtime has no clock of its own: it
    /// keeps the highest reading it was told of, and [`ExpertGrantBinding::now`] only counts when
    /// it is above that floor, so a grant minted at `T0` and reused after its identity expired is
    /// refused at commit ([`ExpertAdmissionError::GrantIdentityInvalid`]) once the host has
    /// observed a time past the expiry. The floor is monotone and in memory (not ledgered, not
    /// replayed, restarts at `0`): a host calls this on every request it serves and again after a
    /// restart.
    ///
    /// # Errors
    /// [`ExpertAdmissionError::HostTimeRegressed`] for a reading below the floor; the floor is
    /// unchanged.
    pub fn observe_host_time(&mut self, now: Timestamp) -> Result<(), ExpertAdmissionError> {
        let floor = &mut self.experts.grants.time_floor;
        if now < *floor {
            return Err(ExpertAdmissionError::HostTimeRegressed {
                floor: *floor,
                found: now,
            });
        }
        *floor = now;
        Ok(())
    }

    /// The highest host clock reading observed with [`PtrRuntime::observe_host_time`], `0` when
    /// none was.
    pub fn host_time_floor(&self) -> Timestamp {
        self.experts.grants.time_floor
    }

    /// Revoke every grant of `kind` minted so far by moving its current generation up by one;
    /// the new generation is returned for the host to mint the replacement at. A grant of an
    /// older generation is refused at its next commit. Host authority held in memory: not
    /// written to the ledger and not replayed.
    ///
    /// # Errors
    /// [`ExpertAdmissionError::GrantGenerationExhausted`] when the generation cannot advance.
    pub fn supersede_expert_grant(
        &mut self,
        kind: ExpertGrantKind,
    ) -> Result<ExpertGrantGeneration, ExpertAdmissionError> {
        self.bump_grant_generation(kind)
    }

    fn bump_grant_generation(
        &mut self,
        kind: ExpertGrantKind,
    ) -> Result<ExpertGrantGeneration, ExpertAdmissionError> {
        let slot = self.experts.grants.slot(kind);
        *slot = slot
            .checked_add(1)
            .ok_or(ExpertAdmissionError::GrantGenerationExhausted { kind })?;
        Ok(ExpertGrantGeneration(*slot))
    }

    /// Re-read, from the runtime's own state, the authority a grant rests on: its generation is
    /// current for `kind`, its identity session is not revoked, its identity context is valid at
    /// the grant's `now`, its policy key is installed and not revoked and its principal is
    /// admitted. Called immediately before the append.
    fn revalidate_grant(
        &self,
        kind: ExpertGrantKind,
        binding: &ExpertGrantBinding,
    ) -> Result<(), RuntimeError> {
        let expert = RuntimeError::Expert;
        let current = self.experts.grants.current(kind);
        if binding.generation().get() != current {
            return Err(expert(ExpertAdmissionError::GrantGenerationStale {
                kind,
                grant: binding.generation().get(),
                current,
            }));
        }
        let identity = binding.identity();
        if self.revoked_sessions.contains(&identity.session_id) {
            return Err(expert(ExpertAdmissionError::GrantSessionRevoked));
        }
        // The binding's `now` was stamped by the host when it minted the grant; the runtime's
        // own floor of observed host time keeps a grant reused after expiry from being valid.
        let now = binding.now().max(self.experts.grants.time_floor);
        if !identity.is_valid_at(now) {
            return Err(expert(ExpertAdmissionError::GrantIdentityInvalid));
        }
        if !self.policy_trust.contains(binding.policy_key_id()) {
            return Err(expert(ExpertAdmissionError::GrantPolicyKeyUntrusted));
        }
        if !self.execution.peer_admitted(binding.peer()) {
            return Err(expert(ExpertAdmissionError::GrantPeerNotAdmitted));
        }
        let principal = PrincipalId(format!("{}:{}", identity.issuer, identity.subject));
        // A poisoned lock reads as not admitted: authority fails closed.
        let admitted = self
            .manifest_authority
            .read()
            .map(|authority| authority.principal_admitted(&principal))
            .unwrap_or(false);
        if !admitted {
            return Err(expert(ExpertAdmissionError::GrantPrincipalNotAdmitted));
        }
        Ok(())
    }

    /// Whether the runtime may route to the expert `pod` of `project` at `generation` now: it is
    /// in the registry at exactly that generation, `Serving` or `Frozen`, and not fenced. A
    /// restart that rebuilds the registry from the ledger answers the same.
    ///
    /// This is the one use-time admission check; no production router consumes it.
    ///
    /// # Errors
    /// [`RouteRefusal`] naming the first failing condition.
    pub fn expert_route_admission(
        &self,
        project: &ProjectId,
        pod: &PodId,
        generation: Generation,
    ) -> Result<(), RouteRefusal> {
        self.experts.route_admission(project, pod, generation)?;
        // The index knows the invalidations it committed; the lifecycle authority also knows
        // the plain revocations and supersessions, which no fence need have followed yet.
        match self.first_dead_dependency(project, pod, generation) {
            Some((identity, generation)) => Err(RouteRefusal::DependencyNotLive {
                identity,
                generation,
            }),
            None => Ok(()),
        }
    }

    /// Fence the expert `request` names: a committed, ledger-ordered withdrawal from serving at
    /// an unchanged generation, moving its fence epoch (state version) up by one. Needs a grant
    /// of [`ExpertFenceAuthority::RuntimeRevocation`] for the request's project.
    ///
    /// # Errors
    /// [`RuntimeError::Expert`] with the first failing check; nothing is appended on error.
    pub fn fence_expert(
        &mut self,
        request: &ExpertFenceRequest,
        grant: &ExpertFenceGrant,
    ) -> Result<CommitIndex, RuntimeError> {
        self.commit_expert_fence(ExpertFenceAction::Fence, request, None, grant)
    }

    /// Escalate the `AuthorityQuarantine` fence of the `Fenced` expert `request` names to
    /// `InputRevoked`: the expert stays fenced at its generation, the fence epoch moves up by one
    /// and the cause is now final. The request must name `InputRevoked` and a revocation the
    /// ledger already holds. Needs a grant of [`ExpertFenceAuthority::RuntimeRevocation`]. The
    /// reverse, a fence of cause `InputRevoked` re-fenced for `AuthorityQuarantine`, is refused.
    ///
    /// # Errors
    /// [`RuntimeError::Expert`] with the first failing check; nothing is appended on error.
    pub fn escalate_expert_fence(
        &mut self,
        request: &ExpertFenceRequest,
        grant: &ExpertFenceGrant,
    ) -> Result<CommitIndex, RuntimeError> {
        self.commit_expert_fence(ExpertFenceAction::Escalate, request, None, grant)
    }

    /// Retire the `Fenced` expert `request` names for good: `Retired(InputRevoked)` at its
    /// unchanged generation, the fence epoch (state version) up by one. The expert must be fenced
    /// with cause [`ExpertFenceCause::InputRevoked`]; the request must name that cause and the
    /// `(subject, generation)` of a `Revoked` (or `ProcedureRevoked`) record the ledger already
    /// holds, exactly as the fence did. Needs a grant of
    /// [`ExpertFenceAuthority::RuntimeRevocation`]; a gate grant is refused, and no gate evidence
    /// exists or is accepted: this is the runtime's reaction to a revocation, never a gate
    /// decision. The state is terminal: no readmission, no reactivation seeds from it and no
    /// proposal may name it. An `AuthorityQuarantine` fence, an unfenced, a retired or a
    /// wrong-generation expert is refused.
    ///
    /// # Errors
    /// [`RuntimeError::Expert`] with the first failing check; nothing is appended on error.
    pub fn retire_revoked_expert(
        &mut self,
        request: &ExpertFenceRequest,
        grant: &ExpertFenceGrant,
    ) -> Result<CommitIndex, RuntimeError> {
        self.commit_expert_fence(ExpertFenceAction::Retire, request, None, grant)
    }

    /// Readmit a fenced expert at its unchanged generation, in the state the fence left it
    /// (`Serving` or `Frozen`), at a new fence epoch,
    /// so no entry recorded under an earlier epoch is revived. Needs a grant of
    /// [`ExpertFenceAuthority::Gate`], a fence of cause
    /// [`ExpertFenceCause::AuthorityQuarantine`] and a passed `report` bound to this readmission
    /// ([`expert_readmission_subject_ids`]) under the grant's gate configuration.
    ///
    /// # Errors
    /// [`RuntimeError::Expert`] with the first failing check; nothing is appended on error.
    pub fn readmit_expert(
        &mut self,
        request: &ExpertFenceRequest,
        report: &BoundGateReport,
        grant: &ExpertFenceGrant,
    ) -> Result<CommitIndex, RuntimeError> {
        self.commit_expert_fence(ExpertFenceAction::Readmit, request, Some(report), grant)
    }

    /// Invalidate an input and fence every expert that rests on it, in ONE committed
    /// [`LedgerEvent::InputInvalidated`]. Needs a grant of
    /// [`ExpertFenceAuthority::GlobalRevocation`] ([`ExpertFenceGrant::global_revocation`]),
    /// revalidated at commit like a fence; a project-scoped
    /// [`ExpertFenceAuthority::RuntimeRevocation`] grant is refused
    /// ([`ExpertAdmissionError::FenceAuthorityMismatch`]).
    ///
    /// The fence set is derived here, deterministically, from the dependency index at this
    /// commit index (see "Input dependencies" in the module documentation): the event carries
    /// its count and digest only, never a list, and applying the event fences every member
    /// atomically. For [`InputInvalidationKind::Revoked`] the input also joins the revocation set
    /// (as a `Revoked` record does), so [`PtrRuntime::generation_validity`] reports it revoked
    /// and no record may name it again; for `Superseded` the floor of the input moves up and the
    /// digest registry takes the new generation's digest.
    ///
    /// An input identity names no project, so the invalidation reaches the dependents in every
    /// project, like a plain `Revoked` record does for everything that cites its subject. That
    /// reach is why the grant is the runtime-wide revocation authority and not a project's.
    ///
    /// Refused, typed and with nothing appended: a gate grant; a revocation already committed
    /// through this event ([`ExpertAdmissionError::InputAlreadyInvalidated`]); a binding digest
    /// other than the one the first committed use fixed
    /// ([`ExpertAdmissionError::InputBindingMismatch`]); a generation the lifecycle authority
    /// does not hold (a revocation ahead of the live generation, a supersession to a generation
    /// it has not committed: [`ExpertAdmissionError::InputGenerationMismatch`]); a supersession
    /// that does not move the floor forward ([`ExpertAdmissionError::StaleSupersession`]).
    /// Commit and replay both re-derive the set and refuse a different count or digest
    /// ([`ExpertAdmissionError::FenceSetMismatch`]).
    ///
    /// # Errors
    /// [`RuntimeError::Expert`] with the first failing check, or the runtime's own errors when
    /// execution is fenced or the append fails.
    pub fn invalidate_input(
        &mut self,
        grant: &ExpertFenceGrant,
        request: InputInvalidation,
    ) -> Result<InputInvalidationOutcome, RuntimeError> {
        let expert = RuntimeError::Expert;
        if grant.authority() != ExpertFenceAuthority::GlobalRevocation {
            return Err(expert(ExpertAdmissionError::FenceAuthorityMismatch {
                action: ExpertFenceAction::Fence,
                required: ExpertFenceAuthority::GlobalRevocation,
            }));
        }
        let set = self
            .experts
            .fence_set(&request.input, request.generation, &request.kind)
            .map_err(expert)?;
        let event = LedgerEvent::InputInvalidated {
            input: request.input,
            generation: request.generation,
            binding_digest: request.binding_digest,
            kind: request.kind,
            fence_count: set.count,
            fence_digest: set.digest,
        };
        // The same validation replay applies, including re-deriving the set.
        let planned = self.plan_input_invalidation(&event)?;
        self.revalidate_grant(grant.kind(), grant.binding())?;
        let index = self.append_prepared(event, None)?;
        Ok(InputInvalidationOutcome {
            index,
            fence_digest: planned.digest,
            fenced: planned.members,
        })
    }

    /// Replay-time and commit-time check of an `InputInvalidated` event against the history as of
    /// now; the fence set it would apply.
    pub(crate) fn validate_input_invalidated_event(
        &self,
        event: &LedgerEvent,
    ) -> Result<(), RuntimeError> {
        self.plan_input_invalidation(event).map(|_| ())
    }

    fn plan_input_invalidation(&self, event: &LedgerEvent) -> Result<FenceSet, RuntimeError> {
        let LedgerEvent::InputInvalidated {
            input,
            generation,
            binding_digest,
            kind,
            fence_count,
            fence_digest,
        } = event
        else {
            return Err(RuntimeError::Expert(ExpertAdmissionError::InvalidRecord(
                "not an input invalidation".to_owned(),
            )));
        };
        let expert = RuntimeError::Expert;
        ptr_ledger::check_encodable(event)
            .map_err(|_| expert(ExpertAdmissionError::Unencodable))?;
        let index = self.experts.dependency_index();
        let (identity, generation) = (input.clone(), *generation);
        if matches!(kind, InputInvalidationKind::Revoked) && index.is_revoked(input, generation) {
            return Err(expert(ExpertAdmissionError::InputAlreadyInvalidated {
                identity,
                generation,
            }));
        }
        // The digest the first committed use fixed for the invalidated generation, if any. A
        // dependency-block digest is the proposer's assertion, so the first use of an identity
        // the lifecycle authority does not know can fix a wrong one; a revocation must not wait
        // on it (the event records the digest the revoker asserted), only a supersession, which
        // moves the registry forward from that digest, is held to it.
        if matches!(kind, InputInvalidationKind::Superseded { .. })
            && index
                .registered_digest(input, generation)
                .is_some_and(|fixed| fixed != *binding_digest)
        {
            return Err(expert(ExpertAdmissionError::InputBindingMismatch {
                identity,
                generation,
            }));
        }
        // A generation the lifecycle authority holds: a revocation cannot be ahead of it, and a
        // supersession needs its new generation committed there.
        let live = self.live_generation(input);
        match kind {
            InputInvalidationKind::Revoked => {
                if let Some(live) = live.filter(|live| generation > *live) {
                    return Err(expert(ExpertAdmissionError::InputGenerationMismatch {
                        identity,
                        generation,
                        live,
                    }));
                }
            }
            InputInvalidationKind::Superseded {
                new_generation,
                new_digest,
                ..
            } => {
                if let Some(live) = live.filter(|live| new_generation > live) {
                    return Err(expert(ExpertAdmissionError::InputGenerationMismatch {
                        identity,
                        generation: *new_generation,
                        live,
                    }));
                }
                if let Some(floor) = index
                    .supersession_floor(input)
                    .filter(|floor| new_generation <= floor)
                {
                    return Err(expert(ExpertAdmissionError::StaleSupersession {
                        identity,
                        floor,
                        new_generation: *new_generation,
                    }));
                }
                if index
                    .registered_digest(input, *new_generation)
                    .is_some_and(|fixed| fixed != *new_digest)
                {
                    return Err(expert(ExpertAdmissionError::InputBindingMismatch {
                        identity,
                        generation: *new_generation,
                    }));
                }
            }
        }
        let set = self
            .experts
            .fence_set(input, generation, kind)
            .map_err(expert)?;
        if set.count != *fence_count || set.digest != *fence_digest {
            return Err(expert(ExpertAdmissionError::FenceSetMismatch {
                derived_count: set.count,
                event_count: *fence_count,
                digest_matches: set.digest == *fence_digest,
            }));
        }
        Ok(set)
    }

    /// Apply a committed `InputInvalidated` event: the revocation or the supersession floor, the
    /// fences of every member of the derived set in one step, the materialized `expert:` keys of
    /// the fenced experts and the KV catalog entries that rest on them.
    pub(crate) fn apply_input_invalidated_event(
        &mut self,
        event: &LedgerEvent,
    ) -> Result<(), RuntimeError> {
        let LedgerEvent::InputInvalidated {
            input,
            generation,
            kind,
            ..
        } = event
        else {
            return Ok(());
        };
        // Everything fallible happens before the first mutation.
        let set = self.plan_input_invalidation(event)?;
        self.invalidate_kv_expert_fences(&fence_set_epochs(&set.members));
        match kind {
            InputInvalidationKind::Revoked => {
                self.experts.deps.record_revocation(input, *generation);
                self.revoked_generations
                    .insert((input.clone(), *generation));
                // A revocation of a semantic key is a floor over every citation of it.
                let held = self.semdb.snapshot().value(input).is_some();
                self.semantic_edits
                    .record_revocation(input, held, self.semdb.revision());
            }
            InputInvalidationKind::Superseded {
                new_generation,
                new_digest,
                ..
            } => self
                .experts
                .deps
                .record_supersession(input, *new_generation, *new_digest),
        }
        self.experts
            .apply_fence_set(&set, ExpertFenceCause::InputRevoked);
        for member in &set.members {
            self.state.values.insert(
                ptr_state::expert_key(&member.project.0, &member.pod.0),
                ptr_state::expert_entry(member.generation.0, ExpertState::Fenced, member.epoch()),
            );
        }
        Ok(())
    }

    fn commit_expert_fence(
        &mut self,
        action: ExpertFenceAction,
        request: &ExpertFenceRequest,
        report: Option<&BoundGateReport>,
        grant: &ExpertFenceGrant,
    ) -> Result<CommitIndex, RuntimeError> {
        let expert = RuntimeError::Expert;
        let required = match action {
            ExpertFenceAction::Fence | ExpertFenceAction::Escalate | ExpertFenceAction::Retire => {
                ExpertFenceAuthority::RuntimeRevocation
            }
            ExpertFenceAction::Readmit => ExpertFenceAuthority::Gate,
        };
        if grant.authority() != required {
            return Err(expert(ExpertAdmissionError::FenceAuthorityMismatch {
                action,
                required,
            }));
        }
        if grant.project() != Some(&request.project) {
            return Err(expert(ExpertAdmissionError::FenceProjectMismatch));
        }
        let epoch = request
            .expected_state_version
            .checked_add(1)
            .ok_or_else(|| {
                expert(ExpertAdmissionError::FenceEpochMismatch {
                    pod: request.pod.0.clone(),
                })
            })?;
        let evidence = match (action, report, grant.gate) {
            (
                ExpertFenceAction::Fence | ExpertFenceAction::Escalate | ExpertFenceAction::Retire,
                _,
                _,
            ) => None,
            (ExpertFenceAction::Readmit, Some(report), Some(requirement)) => {
                // Readmission needs evidence under the current authority, as admission does.
                self.experts
                    .check_authority(&EvidenceClaim {
                        gate_epoch: requirement.gate_epoch,
                        gate_config_digest: requirement.config_digest,
                        evaluator: EvaluatorClaim::Named(grant.evaluator().copied()),
                    })
                    .map_err(expert)?;
                let entry = self
                    .experts
                    .get(&request.project, &request.pod)
                    .ok_or_else(|| {
                        expert(ExpertAdmissionError::UnknownOperand {
                            pod: request.pod.0.clone(),
                        })
                    })?;
                let expected = readmission_expectation(
                    ReadmissionTarget {
                        project: &request.project,
                        pod: &request.pod,
                        generation: request.generation,
                        expected_state_version: request.expected_state_version,
                        epoch,
                        cause: request.cause,
                    },
                    entry,
                    requirement.config_digest,
                    Some(requirement.eval_manifest_digest),
                );
                check_report(report, &expected).map_err(expert)?;
                Some(ExpertFenceEvidence {
                    gate_epoch: requirement.gate_epoch,
                    required_gate_config_digest: requirement.config_digest,
                    report: report.encode_canonical(),
                })
            }
            (ExpertFenceAction::Readmit, _, _) => {
                return Err(expert(ExpertAdmissionError::FenceEvidenceMissing {
                    pod: request.pod.0.clone(),
                }));
            }
        };
        // The state the fence leaves (a fence) or restores (an escalation, a readmission). An
        // unknown expert or one with no recorded resume reads as `Serving`: the compare-and-set
        // below refuses it before the value matters.
        let resume = match action {
            ExpertFenceAction::Fence => match self.experts.get(&request.project, &request.pod) {
                Some(entry) if entry.state == ExpertState::Frozen => ExpertFenceResume::Frozen,
                _ => ExpertFenceResume::Serving,
            },
            ExpertFenceAction::Escalate
            | ExpertFenceAction::Retire
            | ExpertFenceAction::Readmit => self
                .experts
                .fence_resume(&request.project, &request.pod)
                .unwrap_or(ExpertFenceResume::Serving),
        };
        if action == ExpertFenceAction::Escalate && request.cause != ExpertFenceCause::InputRevoked
        {
            return Err(expert(ExpertAdmissionError::NotEscalatable {
                pod: request.pod.0.clone(),
            }));
        }
        if action == ExpertFenceAction::Retire && request.cause != ExpertFenceCause::InputRevoked {
            return Err(expert(ExpertAdmissionError::NotRetirable {
                pod: request.pod.0.clone(),
            }));
        }
        let wants_revocation =
            action != ExpertFenceAction::Readmit && request.cause == ExpertFenceCause::InputRevoked;
        if request.revocation.is_some() != wants_revocation {
            return Err(expert(ExpertAdmissionError::FenceRevocationNotCommitted {
                pod: request.pod.0.clone(),
            }));
        }
        let event = LedgerEvent::ExpertFenceCommitted {
            project: request.project.clone(),
            pod: request.pod.clone(),
            action,
            cause: request.cause,
            resume,
            generation: request.generation,
            expected_state_version: request.expected_state_version,
            epoch,
            revocation: request.revocation.clone(),
            evidence,
        };
        self.validate_expert_fence_event(&event)?;
        self.revalidate_grant(grant.kind(), grant.binding())?;
        self.append_prepared(event, None)
    }

    /// Replay-time and commit-time check of a fence event: the ledger can frame it and the
    /// compare-and-set holds against the registry as of now.
    pub(crate) fn validate_expert_fence_event(
        &self,
        event: &LedgerEvent,
    ) -> Result<(), RuntimeError> {
        ptr_ledger::check_encodable(event)
            .map_err(|_| RuntimeError::Expert(ExpertAdmissionError::Unencodable))?;
        self.experts
            .plan_fence(event)
            .map_err(RuntimeError::Expert)?;
        // A readmission puts an expert back to work: none of the inputs its record named may be
        // revoked or superseded meanwhile.
        if let LedgerEvent::ExpertFenceCommitted {
            project,
            pod,
            generation,
            action: ExpertFenceAction::Readmit,
            ..
        } = event
        {
            if let Some((identity, input_generation)) =
                self.first_dead_dependency(project, pod, *generation)
            {
                return Err(RuntimeError::Expert(
                    ExpertAdmissionError::DependencyNotLive {
                        pod: pod.0.clone(),
                        identity,
                        generation: input_generation,
                    },
                ));
            }
        }
        // An `InputRevoked` fence rests on a revocation the ledger holds at or before it: the
        // runtime's revoked set is built in commit order, at commit and on replay alike. A
        // retirement may also cite the committed supersession that fenced the expert (a
        // `Superseded` invalidation names no revocation, and without this the expert would stay
        // fenced for ever). For an expert that recorded its inputs, the cited pair must be one
        // of them or of an ancestor's, so a revocation of unrelated data cannot be used to
        // retire it.
        if let LedgerEvent::ExpertFenceCommitted {
            project,
            pod,
            generation,
            action,
            revocation: Some(revocation),
            ..
        } = event
        {
            let cited = (revocation.subject.clone(), revocation.generation);
            let revoked = self.revoked_generations.contains(&cited);
            let superseded = *action == ExpertFenceAction::Retire
                && matches!(
                    self.experts
                        .dependency_index()
                        .status(&revocation.subject, revocation.generation),
                    Some(InputStatus::Superseded { .. })
                );
            if !revoked && !superseded {
                return Err(RuntimeError::Expert(
                    ExpertAdmissionError::FenceRevocationNotCommitted { pod: pod.0.clone() },
                ));
            }
            let lineage =
                self.experts
                    .dependency_index()
                    .lineage_inputs(&project.0, &pod.0, *generation);
            if !lineage.is_empty() && !lineage.contains(&cited) {
                return Err(RuntimeError::Expert(
                    ExpertAdmissionError::RevocationNotADependency {
                        pod: pod.0.clone(),
                        subject: revocation.subject.clone(),
                        generation: revocation.generation,
                    },
                ));
            }
        }
        Ok(())
    }

    /// Apply a committed fence event to the registry.
    pub(crate) fn apply_expert_fence_event(
        &mut self,
        event: &LedgerEvent,
    ) -> Result<(), RuntimeError> {
        let plan = self
            .experts
            .plan_fence(event)
            .map_err(RuntimeError::Expert)?;
        self.experts.apply_fence(plan);
        Ok(())
    }

    /// Replay-time and admission-time check of a structural event: the record decodes, every id
    /// the event carries is re-derived from it, the ledger can frame it, and the compare-and-set
    /// holds against the registry as of now.
    pub(crate) fn validate_expert_event(&self, event: &LedgerEvent) -> Result<(), RuntimeError> {
        self.plan_expert_event(event).map(|_| ())
    }

    fn plan_expert_event(&self, event: &LedgerEvent) -> Result<ExpertPlan, RuntimeError> {
        let LedgerEvent::ExpertStructuralCommitted {
            project,
            op_kind,
            proposal_id,
            operation_id,
            evidence_digest,
            record,
        } = event
        else {
            return Err(RuntimeError::Expert(ExpertAdmissionError::InvalidRecord(
                "not a structural event".to_owned(),
            )));
        };
        let expert = RuntimeError::Expert;
        let decoded = ExpertCommitRecord::decode_canonical(record)
            .map_err(|error| expert(ExpertAdmissionError::InvalidRecord(error.to_string())))?;
        let proposal = decoded.proposal();
        let mismatch = |field| expert(ExpertAdmissionError::EventMismatch { field });
        if project != proposal.project() {
            return Err(mismatch(ExpertEventField::Project));
        }
        if *op_kind != ledger_op_kind(proposal.op().kind()) {
            return Err(mismatch(ExpertEventField::OpKind));
        }
        if proposal_id != proposal.proposal_id().as_bytes() {
            return Err(mismatch(ExpertEventField::ProposalId));
        }
        if operation_id != proposal.operation_id().as_bytes() {
            return Err(mismatch(ExpertEventField::OperationId));
        }
        if evidence_digest != decoded.evidence_digest() {
            return Err(mismatch(ExpertEventField::EvidenceDigest));
        }
        ptr_ledger::check_encodable(event)
            .map_err(|_| expert(ExpertAdmissionError::Unencodable))?;
        let verified = self.verify_expert_evidence(&decoded).map_err(expert)?;
        let entries = self.experts.plan(&decoded).map_err(expert)?;
        let dependency = self.plan_dependencies(&decoded).map_err(expert)?;
        Ok(ExpertPlan {
            entries,
            verified,
            dependency,
        })
    }

    /// What `record` adds to the dependency index, after checking the inputs it names against
    /// the history as of now: each must be live (not revoked, not superseded, by the index or by
    /// the lifecycle authority, and not ahead of the lifecycle authority's generation) and carry
    /// the digest the first committed use fixed. The digest a block asserts is the proposer's
    /// word: nothing authenticates it except the gate evidence, which is bound to the block
    /// (see [`expert_evidence_artifact_digest_with_inputs`]).
    ///
    /// An expert created from source experts (`Merge`, `Split`, `Reactivate`) also inherits what
    /// they rest on: when any input of a tracked ancestor is revoked, or superseded without the
    /// new expert's own block naming the replacement, the record is refused
    /// ([`ExpertInputRole::Inherited`]), with or without a block of its own. A record without a
    /// dependency block contributes its source edges.
    fn plan_dependencies(
        &self,
        record: &ExpertCommitRecord,
    ) -> Result<DependencyUpdate, ExpertAdmissionError> {
        let proposal = record.proposal();
        let project = &proposal.project().0;
        let op = proposal.op();
        let key =
            |pod: &PodId, generation: Generation| (project.clone(), pod.0.clone(), generation);
        let results: Vec<ExpertRef> = op
            .results()
            .iter()
            .map(|result| key(result.pod(), result.generation()))
            .collect();
        let sources: Vec<ExpertRef> = op
            .sources()
            .iter()
            .map(|source| key(source.pod(), source.generation()))
            .collect();
        let Some(block) = record.inputs() else {
            self.check_inherited_inputs(&[], &results, &sources)?;
            return Ok(DependencyUpdate {
                results,
                sources,
                inputs: None,
            });
        };
        let base = op
            .results()
            .first()
            .map(|result| ExpertInputRef::from_base_model(result.base_model()))
            .ok_or_else(|| {
                ExpertAdmissionError::InvalidRecord("inputs without a result expert".to_owned())
            })?;
        let named = block
            .training()
            .iter()
            .map(|input| (ExpertInputRole::Training, input))
            .chain(
                block
                    .evaluation()
                    .iter()
                    .map(|input| (ExpertInputRole::Evaluation, input)),
            )
            .chain(std::iter::once((ExpertInputRole::BaseModel, &base)));
        let mut seen: BTreeMap<InputKey, [u8; 32]> = BTreeMap::new();
        let mut recorded = Vec::new();
        for (role, input) in named {
            let identity = input.identity();
            let generation = input.generation();
            self.check_input_live(role, identity, generation)?;
            if role == ExpertInputRole::Training
                && identity.starts_with(RUN_IDENTITY_PREFIX)
                && self
                    .experts
                    .dependency_index()
                    .registered_digest(identity, generation)
                    .is_some()
            {
                return Err(ExpertAdmissionError::RunIdReused {
                    identity: identity.to_owned(),
                });
            }
            let stale = || ExpertAdmissionError::InputDigestStale {
                role,
                identity: identity.to_owned(),
                generation,
            };
            if self
                .experts
                .dependency_index()
                .registered_digest(identity, generation)
                .is_some_and(|fixed| fixed != *input.digest())
            {
                return Err(stale());
            }
            // One record cannot name one pair with two digests (the base model derives a pair
            // that an explicit input may collide with).
            let known = seen
                .entry((identity.to_owned(), generation))
                .or_insert(*input.digest());
            if *known != *input.digest() {
                return Err(stale());
            }
            recorded.push(RecordedInput {
                identity: identity.to_owned(),
                generation,
                digest: *input.digest(),
            });
        }
        let own: Vec<InputKey> = seen.keys().cloned().collect();
        self.check_inherited_inputs(&own, &results, &sources)?;
        let split = |inputs: &[ExpertInputRef]| -> Vec<RecordedInput> {
            inputs
                .iter()
                .map(|input| RecordedInput {
                    identity: input.identity().to_owned(),
                    generation: input.generation(),
                    digest: *input.digest(),
                })
                .collect()
        };
        let base = recorded
            .pop()
            .ok_or_else(|| ExpertAdmissionError::InvalidRecord("no base model".to_owned()))?;
        Ok(DependencyUpdate {
            results,
            sources,
            inputs: Some(RecordedBlock {
                training: split(block.training()),
                evaluation: split(block.evaluation()),
                base,
                commitment: block.canonical_digest(),
            }),
        })
    }

    /// Refuse a record that creates experts from `sources` when what they rest on, directly or
    /// through their ancestors, is no longer live (see the first-dead-input rule in
    /// [`ExpertDependencyIndex`]). `own` is the new experts' own dependency set, which can
    /// exempt a superseded ancestor input by naming its replacement.
    fn check_inherited_inputs(
        &self,
        own: &[InputKey],
        results: &[ExpertRef],
        sources: &[ExpertRef],
    ) -> Result<(), ExpertAdmissionError> {
        if results.is_empty() || sources.is_empty() {
            return Ok(());
        }
        let status =
            |identity: &str, generation: Generation| self.input_status(identity, generation);
        let Some(dead) = self
            .experts
            .dependency_index()
            .first_dead_input(own, sources, &status)
        else {
            return Ok(());
        };
        let role = if dead.inherited {
            ExpertInputRole::Inherited
        } else {
            ExpertInputRole::Training
        };
        Err(match dead.status {
            InputStatus::Revoked => ExpertAdmissionError::InputRevoked {
                role,
                identity: dead.identity,
                generation: dead.generation,
            },
            InputStatus::Superseded { floor } => ExpertAdmissionError::InputSuperseded {
                role,
                identity: dead.identity,
                generation: dead.generation,
                floor,
            },
        })
    }

    /// Why `identity` at `generation` is not live now, as the dependency index and the lifecycle
    /// authority together know it, or `None`. A revocation is reported before a supersession; for
    /// the lifecycle authority a superseded generation's floor is its live generation.
    fn input_status(&self, identity: &str, generation: Generation) -> Option<InputStatus> {
        if let Some(status) = self.experts.dependency_index().status(identity, generation) {
            return Some(status);
        }
        match self.generation_validity(identity, generation) {
            Some(Validity::Live) | None => None,
            Some(Validity::Revoked | Validity::Disputed) => Some(InputStatus::Revoked),
            Some(Validity::Superseded) => Some(InputStatus::Superseded {
                floor: self.live_generation(identity).unwrap_or(generation),
            }),
        }
    }

    /// Whether `identity` at `generation` may be built on or served from now: neither revoked nor
    /// superseded, by a committed `InputInvalidated`, a committed `Revoked`/`ProcedureRevoked`,
    /// or the lifecycle authority's own generation (see [`PtrRuntime::generation_validity`]), and
    /// not ahead of the generation the lifecycle authority holds for an identity it knows. The
    /// last rule keeps a record from naming `(identity, live + 1)` and fixing a digest for a
    /// generation that does not exist yet.
    fn check_input_live(
        &self,
        role: ExpertInputRole,
        identity: &str,
        generation: Generation,
    ) -> Result<(), ExpertAdmissionError> {
        match self.input_status(identity, generation) {
            Some(InputStatus::Revoked) => {
                return Err(ExpertAdmissionError::InputRevoked {
                    role,
                    identity: identity.to_owned(),
                    generation,
                })
            }
            Some(InputStatus::Superseded { floor }) => {
                return Err(ExpertAdmissionError::InputSuperseded {
                    role,
                    identity: identity.to_owned(),
                    generation,
                    floor,
                })
            }
            None => {}
        }
        match self.live_generation(identity) {
            Some(live) if generation > live => Err(ExpertAdmissionError::InputGenerationMismatch {
                identity: identity.to_owned(),
                generation,
                live,
            }),
            Some(_) | None => Ok(()),
        }
    }

    /// The first input of the expert `pod` of `project` at `generation`, or of any ancestor, that
    /// is not live now, if any.
    fn first_dead_dependency(
        &self,
        project: &ProjectId,
        pod: &PodId,
        generation: Generation,
    ) -> Option<(String, Generation)> {
        let node = (project.0.clone(), pod.0.clone(), generation);
        let status = |identity: &str, input_generation: Generation| {
            self.input_status(identity, input_generation)
        };
        self.experts
            .dependency_index()
            .first_dead_input_of(&node, &status)
            .map(|dead| (dead.identity, dead.generation))
    }

    /// Re-verify the evidence a record embeds. `None` for a legacy record, which embeds none.
    fn verify_expert_evidence(
        &self,
        record: &ExpertCommitRecord,
    ) -> Result<Option<VerifiedEvidence>, ExpertAdmissionError> {
        let Some(evidence) = record.evidence() else {
            return Ok(None);
        };
        let report = BoundGateReport::decode_canonical(evidence.report())
            .map_err(|error| ExpertAdmissionError::EvidenceUndecodable(error.to_string()))?;
        if report.digest() != *record.evidence_digest() {
            return Err(ExpertAdmissionError::EvidenceDigestMismatch);
        }
        self.experts.check_authority(&EvidenceClaim {
            gate_epoch: record.gate_epoch(),
            gate_config_digest: *evidence.required_gate_config_digest(),
            evaluator: EvaluatorClaim::NotRechecked,
        })?;
        let expected = proposal_expectation(
            record.proposal(),
            record.inputs(),
            *evidence.required_gate_config_digest(),
            None,
        )?;
        check_report(&report, &expected)?;
        let epoch = record.gate_epoch();
        if let Some(known) = self.experts.epoch_gate_configs.get(&epoch) {
            if known != evidence.required_gate_config_digest() {
                return Err(ExpertAdmissionError::EpochGateConfigConflict { epoch });
            }
        }
        Ok(Some(VerifiedEvidence {
            epoch,
            gate_config_digest: *evidence.required_gate_config_digest(),
        }))
    }

    /// Apply a committed structural event to the registry. The event passed validation before it
    /// was appended, so the compare-and-set holding again is an invariant; a failure surfaces as
    /// an error instead of a silently diverging registry.
    pub(crate) fn apply_expert_event(&mut self, event: &LedgerEvent) -> Result<(), RuntimeError> {
        let plan = self.plan_expert_event(event)?;
        self.experts.apply(plan);
        Ok(())
    }
}

/// The `(proposal_id, operation_id)` a gate report must be bound to for it to justify the
/// readmission of `pod` at `generation`, from state version `expected_state_version` to `epoch`,
/// out of a fence of `cause`. Both are domain-separated digests of every one of those fields, so
/// a report for another expert, generation, epoch or cause does not match.
pub fn expert_readmission_subject_ids(
    project: &ProjectId,
    pod: &PodId,
    generation: Generation,
    expected_state_version: u64,
    epoch: u64,
    cause: ExpertFenceCause,
) -> ([u8; 32], [u8; 32]) {
    let digest = |domain: &str| {
        let mut bytes = Vec::new();
        for text in [domain, project.0.as_str(), pod.0.as_str()] {
            bytes.extend_from_slice(&(text.len() as u64).to_le_bytes());
            bytes.extend_from_slice(text.as_bytes());
        }
        bytes.extend_from_slice(&generation.0.to_le_bytes());
        bytes.extend_from_slice(&expected_state_version.to_le_bytes());
        bytes.extend_from_slice(&epoch.to_le_bytes());
        bytes.push(cause.code());
        ptr_ledger::integrity::sha256(&bytes)
    };
    (
        digest("ptr.expert.readmission.proposal.v1"),
        digest("ptr.expert.readmission.operation.v1"),
    )
}

/// The fence memo of a registry that predates it: each `Fenced` expert was fenced at the epoch
/// it now stands at, and nothing is known about any earlier fence.
pub(crate) fn fenced_at_current_epoch(
    entries: &[((String, String), ExpertEntry)],
) -> BTreeMap<(String, String), u64> {
    entries
        .iter()
        .filter(|(_, entry)| entry.state == ExpertState::Fenced)
        .map(|(key, entry)| (key.clone(), entry.state_version))
        .collect()
}

/// The fence epoch now in force, by `expert:{project}/{pod}` identity, for a committed fence or
/// readmission event; empty for any other event. This is the map
/// [`crate::KvPrefixCatalog::invalidate_expert_fences`] takes: an entry that recorded an epoch
/// below it, or none, is tombstoned.
pub fn expert_fence_epochs(event: &LedgerEvent) -> BTreeMap<String, u64> {
    match event {
        LedgerEvent::ExpertFenceCommitted {
            project,
            pod,
            epoch,
            ..
        } => BTreeMap::from([(ptr_state::expert_key(&project.0, &pod.0), *epoch)]),
        _ => BTreeMap::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_proposal_kind_has_its_own_ledger_code() {
        let kinds = [
            OpKind::Birth,
            OpKind::Merge,
            OpKind::Split,
            OpKind::Freeze,
            OpKind::Reactivate,
            OpKind::Prune,
            OpKind::Retire,
        ];
        let codes: Vec<u8> = kinds.iter().map(|k| ledger_op_kind(*k).code()).collect();
        assert_eq!(codes, vec![0, 1, 2, 3, 4, 5, 6]);
    }
}
