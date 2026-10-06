//! Pod-MoE experts and structural proposals.
//!
//! In a pod-MoE every expert is a pod with a manifest, a generation, lineage and a lifecycle
//! (see `docs/architecture/37-pod-moe.md`). A meta-level policy may *propose* structural changes
//! (birth, merge, split, freeze, reactivate, prune, retire); this module only defines the shape
//! of such proposals and checks that a proposal is well formed.
//!
//! # Proposals are inert data
//!
//! An [`ExpertProposal`] changes nothing. Whether it is accepted is decided elsewhere, by the
//! forgetting and interference gates, the verifier and a ledger commit; `ptr-runtime`
//! `admit_expert_proposal` is that acceptance path, and it commits an [`ExpertCommitRecord`] that
//! embeds the gate evidence so replay re-verifies it. This crate defines the shapes and the
//! record codec and decides no admission. Every observed fact in a proposal (a target's `state`, `state_version`,
//! `generation`, `manifest_digest`, and the `observed_at` ledger index) is a caller-observed
//! claim; the commit re-checks all of them against authority (compare-and-swap) and rejects the
//! proposal if any is stale.
//!
//! # Typed operations
//!
//! [`StructuralOp`] carries its operands, so a malformed shape (a merge with one source, a
//! birth with a target, a result on a freeze) is unrepresentable or rejected with a specific
//! error rather than hidden in optional parameters:
//!
//! | operation | operands | result expert(s) |
//! |---|---|---|
//! | `Birth` | none | one new candidate [`ExpertPodSpec`] |
//! | `Merge` | at least two sources, Token level, all `Serving` or all `Frozen` | one new candidate |
//! | `Split` | one source (`Serving` or `Frozen`) | at least two new candidates |
//! | `Freeze` | one `Serving` target | none |
//! | `Reactivate` | one `Frozen` or `Retired(GateFailed)` target | `Frozen`: none; `Retired`: one new generation |
//! | `Prune` | at least one `Serving` target, one Token layer | none |
//! | `Retire` | one `Serving` or `Frozen` target and a proposable [`RetiredReason`] (`Superseded`, `GateFailed`) | none |
//!
//! # Expert states and the lineage mapping
//!
//! [`ExpertState`] is the proposal-level view. It maps onto `ptr-lineage` `AdapterStatus` and
//! onto [`ArtifactLifecycle`] / [`LifecycleGate`](crate::LifecycleGate) as follows (see
//! [`ExpertState::adapter_status_name`] and [`ExpertState::artifact_lifecycles`], which encode
//! this table and are tested against it):
//!
//! | `ExpertState` | `AdapterStatus` | `ArtifactLifecycle` |
//! |---|---|---|
//! | `Candidate` | `Candidate` | `Candidate`, `Trained`, `Evaluated` |
//! | `Gated` | `Gated` | `Approved` |
//! | `Serving` | `Serving` | `Active` |
//! | `Frozen` | `Serving` (serving with router and gate updates stopped) | `Active` |
//! | `Fenced` | `Serving` (lineage is unchanged; the fencing token is external authority) | `Active` |
//! | `Superseded` | `Retired` (withdrawn, kept for audit) | `Superseded` |
//! | `Retired(reason)` | `Retired` | `Retired`; `Revoked` for `RetiredReason::InputRevoked` |
//!
//! `ptr-lineage` has no `Frozen` or `Fenced`; those are expert-layer overlays on a serving
//! expert, hence `Serving`/`Active`. `LifecycleGate::transition` allows `Retired` only from
//! `Active`; a proposal may still retire only `Serving` and `Frozen`, never `Fenced`. `Lineage` itself only gates
//! `Candidate` to `Gated` and serves `Gated` to `Serving`, so no expert-layer overlay has a
//! lineage transition: unfreezing a `Frozen` expert leaves lineage at `Serving` / `Active`
//! (see Reactivation). `ptr-pods` does not depend on `ptr-lineage`, so the adapter status is
//! exposed by name.
//!
//! # Allowed (operation, source state) pairs
//!
//! [`OpKind::allows_state`] is the single source of truth, used by validation and checked
//! pair by pair in `tests/expert.rs`. A cell is the state a source/target may be in:
//!
//! | state | Merge | Split | Freeze | Reactivate | Prune | Retire |
//! |---|---|---|---|---|---|---|
//! | `Candidate` | no | no | no | no | no | no |
//! | `Gated` | no | no | no | no | no | no |
//! | `Serving` | yes | yes | yes | no | yes | yes |
//! | `Frozen` | yes | yes | no | yes | no | yes |
//! | `Fenced` | no | no | no | no | no | no |
//! | `Superseded` | no | no | no | no | no | no |
//! | `Retired(GateFailed)` | no | no | no | yes | no | no |
//! | `Retired(Pruned \| Superseded \| InputRevoked)` | no | no | no | no | no | no |
//!
//! `Birth` has no source. `Retire` is allowed exactly from `Serving` and `Frozen`. A `Candidate`
//! or `Gated` expert is not retired by a proposal (a failed candidate is dropped by its gate),
//! and a `Superseded` expert is already withdrawn.
//!
//! # Fenced experts are outside the proposal API
//!
//! A `Fenced` expert has had its inputs revoked or its authority withdrawn. No proposal may name
//! it as an operand, for any operation: validation rejects it with
//! [`ExpertProposalError::FencedTarget`] before the generic state check. In particular a policy
//! cannot retire a fenced expert as `Retired(GateFailed)` and later reactivate it from its
//! revoked weights. A fenced expert reaches `Retired(InputRevoked)` only through the runtime
//! revocation path (a runtime-grant ledger event), never through an [`ExpertProposal`], and
//! `InputRevoked` is also not a proposable reason or a valid reactivation seed. A
//! `Merge` additionally needs all its sources in the *same* state. A `Reactivate` of a `Retired`
//! target with any reason other than `GateFailed` is reported as
//! [`ExpertProposalError::SeedUnavailable`] (not as a generic state error): `Pruned` weights are
//! physically removed, `Superseded` already has a successor, and `InputRevoked` must never seed.
//!
//! # Reactivation
//!
//! Retirement is final (`Lineage::retire`; `LifecycleGate` has no transition out of `Retired`).
//! `Reactivate` of a `Frozen` target is an UNFREEZE overlay on the same generation: it carries no
//! result (`UnexpectedResult` if one is given), lineage stays `Serving` / `Active` and no lineage
//! transition happens. The check that gates the unfreeze is recorded by the commit path, not as a
//! lineage transition. `Reactivate` of a `Retired(GateFailed)` target leaves that generation
//! `Retired` forever and requires a result (`MissingResult` otherwise): a spec for a NEW
//! candidate generation of the same logical expert (same project and pod, exactly the source
//! generation plus one, same level including the Token index), seeded from the retained weights.
//! It enters lineage as `Candidate` and goes `Candidate`, `Gated`, `Serving` through the normal
//! gate like any birth. Other retired targets are
//! not seeds: `Pruned` weights are gone, `Superseded` has a successor, `InputRevoked` is barred.
//!
//! # Retirement reasons
//!
//! `Retire` of a `Serving` or `Frozen` target accepts only `Superseded` and `GateFailed`. `Pruned` has its own operation
//! ([`StructuralOp::Prune`]) and `InputRevoked` is a runtime-grant ledger event, never a proposal
//! (rejected with [`ExpertProposalError::ReasonNotProposable`]).
//!
//! # Results
//!
//! A fresh expert (`Birth`, `Merge`, `Split` result) must be generation 1 and a new pod. A split
//! result may reuse a source's Token `(layer, index)` slot, since sources are replaced, but two
//! results of one operation may not share a slot ([`ExpertProposalError::DuplicateResultSlot`]).
//! For `Merge`/`Split` the result level need only share the sources' kind and Token layer.
//!
//! # Identifiers
//!
//! Project and pod ids are 1 to 128 bytes of ASCII `[A-Za-z0-9._:-]`. Whitespace-only ids are
//! blank; anything else outside the set (interior spaces, control characters, zero-width
//! characters, non-ASCII, over-long) is non-canonical.
//!
//! # Validation order
//!
//! [`ExpertProposal::new`] reports the first failing rule in this order, so errors are
//! deterministic:
//!
//! 1. the proposal project id is canonical;
//! 2. the proposer belongs to the proposal project;
//! 3. operand counts (Merge and Prune sources, Split results);
//! 4. every source and result belongs to the proposal project;
//! 5. source states: a `Fenced` operand is `FencedTarget` for every operation; then the table
//!    above (a non-`GateFailed` retired Reactivate target is `SeedUnavailable` here);
//! 6. operation structure, per operation:
//!    * Merge: Token level, duplicates and multiple generations, one layer, one base model,
//!      distinct Token slots (`DuplicateSourceSlot`), one shared state (`MixedStates`);
//!    * Prune: Token level, duplicates and multiple generations, one layer, one base model,
//!      distinct Token slots (`DuplicateSourceSlot`);
//!    * Reactivate: result present iff the target is `Retired`;
//!    * Retire: the reason is proposable;
//! 7. result relations, per result: pod, generation, base model, level;
//! 8. result distinctness: pods (`DuplicateResult`), then Token slots (`DuplicateResultSlot`).
//!
//! Ids are validated when [`ExpertPodSpec`], [`ExpertTarget`] and [`ProposerRef`] are built.
//!
//! # Identity
//!
//! An expert is identified by `(project, pod)`, matching the `(ProjectId, PodId)` registry key,
//! and a specific incarnation by its generation. Two generations of one logical expert are never
//! listed together as sources. Project and pod ids must be canonical.
//!
//! # No override surface (deliberate type-level guarantee)
//!
//! An [`ExpertProposal`] has no threshold, gate, verifier, score-override or policy-parameter
//! field, and no field that could hold one. It carries exactly the project, the proposer, the
//! ledger index it was built against, the operation with its operands and a digest of the
//! proposer's rationale. A proposer therefore cannot loosen a gate or switch off a verifier by
//! what it submits. This protects invariant #4 (uncommitted state is never authoritative: a
//! proposal is not a change) and invariant #11 (a learned positive score cannot override a
//! deterministic contradiction) in `docs/INVARIANTS.md`. [`ExpertProposal::parts`] exposes the
//! complete field list as [`ProposalParts`], so a reviewer (and the test in `tests/expert.rs`)
//! can see that adding a field is a visible, breaking change.
//!
//! # Two-stage identity
//!
//! [`ExpertOperationRequest::operation_id`] exists before training: it identifies the requested
//! operation (kind, operands with their observed generation, manifest digest, state and state
//! version, retire reason, proposer, project, `observed_at`, rationale digest) and the intended
//! results, in order, as [`ResultIntent`]s: every [`ExpertPodSpec`] field (project, pod,
//! generation, level with Token layer and slot, base model ref) except the manifest digest,
//! which only training produces. So two requests that would create different experts have
//! different ids, and the id is unchanged by the manifest digest of a trained result. It is
//! SHA-256 over a length-prefixed canonical encoding with domain separator
//! `ptr.pods.expert-operation.v2`.
//!
//! An [`ExpertOperationRequest`] (with [`RequestedOp`], a [`StructuralOp`] whose results are
//! [`ResultIntent`]s) is the validated pre-result form: its constructor runs exactly the rules
//! of [`ExpertProposal::new`], through the same code, so an orchestrator mints the id without
//! fabricating a trained result. [`ExpertProposal::operation_request`] derives the request from
//! a proposal and [`ExpertProposal::operation_id`] delegates to it, so both paths agree.
//! Admission checks `proposal.operation_id() == request.operation_id()`, which covers the
//! intended results; no constructor binds a request to results.
//!
//! [`ExpertProposal::proposal_id`] exists after training and identifies the candidate: SHA-256
//! over a length-prefixed canonical encoding of every field including results (domain separator
//! `ptr.pods.expert-proposal.v2`). It is deterministic, order-sensitive in operand lists, and
//! changes with any field, including `observed_at` and the result specs.

use crate::manifest::ArtifactLifecycle;
use ptr_types::{CommitIndex, Digest, Generation, PodId, ProjectId};
use sha2::{Digest as ShaDigest, Sha256};
use std::collections::BTreeMap;
use std::fmt;

const ZERO_DIGEST: Digest = [0; 32];

/// A digest that was all zero.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ZeroDigest;

impl fmt::Display for ZeroDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("digest is all zero")
    }
}

impl std::error::Error for ZeroDigest {}

macro_rules! digest_newtype {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
        pub struct $name(Digest);

        impl $name {
            /// Wrap a digest. Rejects the all-zero digest.
            pub fn new(digest: Digest) -> Result<Self, ZeroDigest> {
                if digest == ZERO_DIGEST {
                    Err(ZeroDigest)
                } else {
                    Ok(Self(digest))
                }
            }

            pub fn as_bytes(&self) -> &Digest {
                &self.0
            }
        }
    };
}

digest_newtype!(
    /// Digest of an expert pod's execution manifest (non-zero).
    ManifestDigest
);
digest_newtype!(
    /// Digest of a base model (non-zero).
    BaseModelDigest
);
digest_newtype!(
    /// Digest of the proposer's rationale (non-zero).
    RationaleDigest
);
/// Deterministic identity of a requested operation before any training, minted only by
/// [`ExpertProposal::operation_id`] and [`ExpertOperationRequest::operation_id`]. It includes
/// each result's [`ResultIntent`] (project, pod, level with Token layer and slot, base model
/// ref, generation); the only thing excluded is each result's training-produced manifest
/// digest, so it is stable across the training that produces the result. [`ProposalId`]
/// identifies the candidate after training.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct OperationId(Digest);

impl OperationId {
    pub fn as_bytes(&self) -> &Digest {
        &self.0
    }
}

/// Deterministic identity of a proposal (the candidate, including its result specs), minted only
/// by [`ExpertProposal::proposal_id`]. Exists after training; see [`OperationId`] for the
/// pre-training identity.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ProposalId(Digest);

impl ProposalId {
    pub fn as_bytes(&self) -> &Digest {
        &self.0
    }
}

/// Stable identity of a logical base-model record: 1 to 128 bytes of ASCII `[A-Za-z0-9._:-]`,
/// the same rules as project and pod ids. Two records may share weights (a digest) and a
/// generation yet be different base models; the id tells them apart.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BaseModelId(String);

impl BaseModelId {
    /// Validate and wrap an id. Rejects blank (whitespace-only) and non-canonical ids.
    pub fn new(id: impl Into<String>) -> Result<Self, ExpertIdentityError> {
        let id = id.into();
        check_id(
            &id,
            ExpertIdentityError::BlankBaseModel,
            ExpertIdentityError::NonCanonicalBaseModel,
        )?;
        Ok(Self(id))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for BaseModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// The base model an expert adapts: its stable identity, its weights digest and the base
/// model's own generation. All three take part in equality, so two records with the same digest
/// and generation but different ids are different base models (a `MixedBaseModels` for Merge and
/// Prune, a result base-model mismatch, and a different proposal and operation id).
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct BaseModelRef {
    id: BaseModelId,
    digest: BaseModelDigest,
    generation: u64,
}

impl BaseModelRef {
    /// Bind a base model id and digest to the base model's generation. Rejects generation zero
    /// (never a committed generation). The id and digest are valid by type.
    pub fn new(
        id: BaseModelId,
        digest: BaseModelDigest,
        generation: u64,
    ) -> Result<Self, ExpertIdentityError> {
        if generation == 0 {
            Err(ExpertIdentityError::ZeroBaseModelGeneration)
        } else {
            Ok(Self {
                id,
                digest,
                generation,
            })
        }
    }

    pub fn id(&self) -> &BaseModelId {
        &self.id
    }

    pub fn digest(&self) -> &BaseModelDigest {
        &self.digest
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }
}

/// Where an expert lives.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ExpertLevel {
    /// An in-process expert inside one MoE layer; `index` is its slot within the layer.
    Token { layer: u16, index: u16 },
    /// A whole model or adapter set reachable as a networked pod.
    Segment,
}

impl ExpertLevel {
    /// Same kind of level, and the same layer for Token level. The expert index may differ.
    fn compatible_with(self, other: Self) -> bool {
        match (self, other) {
            (Self::Segment, Self::Segment) => true,
            (Self::Token { layer: a, .. }, Self::Token { layer: b, .. }) => a == b,
            (Self::Segment, Self::Token { .. }) | (Self::Token { .. }, Self::Segment) => false,
        }
    }
}

/// Why an expert was retired. A retired target carries its reason.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RetiredReason {
    Pruned,
    Superseded,
    GateFailed,
    /// An input (evidence, knowledge, adapter source) was revoked. Never a valid seed.
    InputRevoked,
}

/// Lifecycle of an expert as seen by a proposal. See the module documentation for the mapping
/// to `ptr-lineage` and [`ArtifactLifecycle`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ExpertState {
    Candidate,
    Gated,
    Serving,
    Frozen,
    /// Fenced off by authority. No proposal may name it ([`ExpertProposalError::FencedTarget`]);
    /// it leaves this state only through the runtime revocation path.
    Fenced,
    /// Replaced by a successor generation; only `Retire` may name it.
    Superseded,
    Retired(RetiredReason),
}

impl ExpertState {
    /// `ptr-lineage` `AdapterStatus` this state maps to, by lowercase name.
    pub fn adapter_status_name(self) -> &'static str {
        match self {
            Self::Candidate => "candidate",
            Self::Gated => "gated",
            Self::Serving | Self::Frozen | Self::Fenced => "serving",
            Self::Superseded | Self::Retired(_) => "retired",
        }
    }

    /// [`ArtifactLifecycle`] values this state maps to.
    pub fn artifact_lifecycles(self) -> &'static [ArtifactLifecycle] {
        match self {
            Self::Candidate => &[
                ArtifactLifecycle::Candidate,
                ArtifactLifecycle::Trained,
                ArtifactLifecycle::Evaluated,
            ],
            Self::Gated => &[ArtifactLifecycle::Approved],
            Self::Serving | Self::Frozen | Self::Fenced => &[ArtifactLifecycle::Active],
            Self::Superseded => &[ArtifactLifecycle::Superseded],
            Self::Retired(RetiredReason::InputRevoked) => &[ArtifactLifecycle::Revoked],
            Self::Retired(
                RetiredReason::Pruned | RetiredReason::Superseded | RetiredReason::GateFailed,
            ) => &[ArtifactLifecycle::Retired],
        }
    }
}

/// The kind of a [`StructuralOp`], without operands.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum OpKind {
    Birth,
    Merge,
    Split,
    Freeze,
    Reactivate,
    Prune,
    Retire,
}

impl OpKind {
    /// Whether a source/target in `state` may be named by this operation (the table in the
    /// module documentation). `Birth` has no source, so no state is allowed.
    pub fn allows_state(self, state: ExpertState) -> bool {
        match self {
            Self::Birth => false,
            Self::Merge | Self::Split => {
                matches!(state, ExpertState::Serving | ExpertState::Frozen)
            }
            Self::Freeze | Self::Prune => matches!(state, ExpertState::Serving),
            Self::Reactivate => matches!(
                state,
                ExpertState::Frozen | ExpertState::Retired(RetiredReason::GateFailed)
            ),
            // `Fenced` maps to `Active` but is revoked by the runtime, never retired by a proposal.
            Self::Retire => matches!(state, ExpertState::Serving | ExpertState::Frozen),
        }
    }
}

impl fmt::Display for OpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Birth => "birth",
            Self::Merge => "merge",
            Self::Split => "split",
            Self::Freeze => "freeze",
            Self::Reactivate => "reactivate",
            Self::Prune => "prune",
            Self::Retire => "retire",
        };
        f.write_str(name)
    }
}

/// A structural operation with its operands. See the module documentation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StructuralOp {
    Birth {
        result: ExpertPodSpec,
    },
    Merge {
        sources: Vec<ExpertTarget>,
        result: ExpertPodSpec,
    },
    Split {
        source: ExpertTarget,
        results: Vec<ExpertPodSpec>,
    },
    Freeze {
        target: ExpertTarget,
    },
    /// `result` is required iff `target` is `Retired` and forbidden iff it is `Frozen`.
    Reactivate {
        target: ExpertTarget,
        result: Option<ExpertPodSpec>,
    },
    Prune {
        targets: Vec<ExpertTarget>,
    },
    Retire {
        target: ExpertTarget,
        reason: RetiredReason,
    },
}

impl StructuralOp {
    pub fn kind(&self) -> OpKind {
        match self {
            Self::Birth { .. } => OpKind::Birth,
            Self::Merge { .. } => OpKind::Merge,
            Self::Split { .. } => OpKind::Split,
            Self::Freeze { .. } => OpKind::Freeze,
            Self::Reactivate { .. } => OpKind::Reactivate,
            Self::Prune { .. } => OpKind::Prune,
            Self::Retire { .. } => OpKind::Retire,
        }
    }

    /// Existing experts the operation names.
    pub fn sources(&self) -> Vec<&ExpertTarget> {
        match self {
            Self::Birth { .. } => Vec::new(),
            Self::Merge { sources, .. } => sources.iter().collect(),
            Self::Split { source, .. } => vec![source],
            Self::Freeze { target }
            | Self::Reactivate { target, .. }
            | Self::Retire { target, .. } => vec![target],
            Self::Prune { targets } => targets.iter().collect(),
        }
    }

    /// New expert generations the operation would create (empty for in-place operations).
    pub fn results(&self) -> Vec<&ExpertPodSpec> {
        match self {
            Self::Birth { result } | Self::Merge { result, .. } => vec![result],
            Self::Split { results, .. } => results.iter().collect(),
            Self::Reactivate { result, .. } => result.iter().collect(),
            Self::Freeze { .. } | Self::Prune { .. } | Self::Retire { .. } => Vec::new(),
        }
    }
}

/// Whether an [`ExpertOutcome`] is an existing operand or a newly created result.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum OutcomeKind {
    /// An existing expert the operation names, now in its post-operation state.
    Operand,
    /// A new expert generation the operation creates.
    Result,
}

/// The authoritative state one expert is left in by an admitted [`StructuralOp`].
///
/// This is the single definition of the post-operation state mapping, shared by every consumer
/// of an admitted record (the runtime registry and the state projection), so none re-implements
/// it. It is pure data derived from the operation: it proves nothing about admission.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertOutcome {
    pub kind: OutcomeKind,
    pub pod: PodId,
    pub generation: Generation,
    pub level: ExpertLevel,
    pub base_model: BaseModelRef,
    pub manifest_digest: ManifestDigest,
    pub state: ExpertState,
    /// An operand's observed `state_version` plus one; `1` for a result.
    pub state_version: u64,
}

impl StructuralOp {
    /// The state every expert the operation touches is left in once it is admitted, operands
    /// first (in operand order) and then results (in result order).
    ///
    /// | operation | operand after | result |
    /// |---|---|---|
    /// | `Birth` | | `Serving` |
    /// | `Merge`, `Split` | `Superseded` | `Serving` |
    /// | `Freeze` | `Frozen` | |
    /// | `Reactivate` of `Frozen` | `Serving` (same generation) | |
    /// | `Reactivate` of `Retired(GateFailed)` | (kept `Retired`; replaced by the result, same pod) | `Serving`, generation + 1 |
    /// | `Prune` | `Retired(Pruned)` | |
    /// | `Retire` | `Retired(reason)` | |
    ///
    /// A result is `Serving` with `state_version` 1 because admission requires a passing
    /// artifact-bound gate verdict for it, which is the `Candidate` to `Gated` to `Serving`
    /// path collapsed into the one committed event. A retired reactivation source yields no
    /// operand outcome: its pod now names the new generation.
    pub fn outcomes(&self) -> Vec<ExpertOutcome> {
        let operand_state = match self {
            Self::Birth { .. } => None,
            Self::Merge { .. } | Self::Split { .. } => Some(ExpertState::Superseded),
            Self::Freeze { .. } => Some(ExpertState::Frozen),
            Self::Reactivate { target, .. } => match target.state() {
                ExpertState::Retired(_) => None,
                _ => Some(ExpertState::Serving),
            },
            Self::Prune { .. } => Some(ExpertState::Retired(RetiredReason::Pruned)),
            Self::Retire { reason, .. } => Some(ExpertState::Retired(*reason)),
        };
        let mut out = Vec::new();
        if let Some(state) = operand_state {
            for source in self.sources() {
                out.push(ExpertOutcome {
                    kind: OutcomeKind::Operand,
                    pod: source.pod().clone(),
                    generation: source.generation(),
                    level: source.level(),
                    base_model: source.base_model().clone(),
                    manifest_digest: *source.manifest_digest(),
                    state,
                    state_version: source.state_version().saturating_add(1),
                });
            }
        }
        for result in self.results() {
            out.push(ExpertOutcome {
                kind: OutcomeKind::Result,
                pod: result.pod().clone(),
                generation: result.generation(),
                level: result.level(),
                base_model: result.base_model().clone(),
                manifest_digest: *result.manifest_digest(),
                state: ExpertState::Serving,
                state_version: 1,
            });
        }
        out
    }
}

const MAX_ID_BYTES: usize = 128;

/// Project and pod ids: 1 to 128 bytes of ASCII `[A-Za-z0-9._:-]`. Whitespace-only is blank.
fn check_id(
    value: &str,
    blank: ExpertIdentityError,
    non_canonical: ExpertIdentityError,
) -> Result<(), ExpertIdentityError> {
    if value.trim().is_empty() {
        Err(blank)
    } else if value.len() > MAX_ID_BYTES
        || !value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b':' | b'-'))
    {
        Err(non_canonical)
    } else {
        Ok(())
    }
}

fn check_project_id(project: &ProjectId) -> Result<(), ExpertIdentityError> {
    check_id(
        &project.0,
        ExpertIdentityError::BlankProject,
        ExpertIdentityError::NonCanonicalProject,
    )
}

fn check_pod_id(pod: &PodId) -> Result<(), ExpertIdentityError> {
    check_id(
        &pod.0,
        ExpertIdentityError::BlankPod,
        ExpertIdentityError::NonCanonicalPod,
    )
}

/// An identity field (project, pod, base model, generation) was malformed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ExpertIdentityError {
    BlankProject,
    NonCanonicalProject,
    BlankPod,
    NonCanonicalPod,
    BlankBaseModel,
    NonCanonicalBaseModel,
    ZeroGeneration,
    ZeroBaseModelGeneration,
}

impl fmt::Display for ExpertIdentityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::BlankProject => "project id is blank",
            Self::NonCanonicalProject => "project id is not 1-128 bytes of ASCII [A-Za-z0-9._:-]",
            Self::BlankPod => "pod id is blank",
            Self::NonCanonicalPod => "pod id is not 1-128 bytes of ASCII [A-Za-z0-9._:-]",
            Self::BlankBaseModel => "base model id is blank",
            Self::NonCanonicalBaseModel => {
                "base model id is not 1-128 bytes of ASCII [A-Za-z0-9._:-]"
            }
            Self::ZeroGeneration => "generation is zero",
            Self::ZeroBaseModelGeneration => "base model generation is zero",
        })
    }
}

impl std::error::Error for ExpertIdentityError {}

fn check_generation(generation: Generation) -> Result<(), ExpertIdentityError> {
    if generation == Generation::default() {
        Err(ExpertIdentityError::ZeroGeneration)
    } else {
        Ok(())
    }
}

/// Identity and binding of one expert pod, used for the result of an operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertPodSpec {
    project: ProjectId,
    pod: PodId,
    manifest_digest: ManifestDigest,
    level: ExpertLevel,
    base_model: BaseModelRef,
    generation: Generation,
}

impl ExpertPodSpec {
    /// Build a spec. Rejects blank or non-canonical project and pod ids and generation zero
    /// (the default value, never a committed generation). The digests are non-zero by type.
    pub fn new(
        project: ProjectId,
        pod: PodId,
        manifest_digest: ManifestDigest,
        level: ExpertLevel,
        base_model: BaseModelRef,
        generation: Generation,
    ) -> Result<Self, ExpertIdentityError> {
        check_project_id(&project)?;
        check_pod_id(&pod)?;
        check_generation(generation)?;
        Ok(Self {
            project,
            pod,
            manifest_digest,
            level,
            base_model,
            generation,
        })
    }

    pub fn project(&self) -> &ProjectId {
        &self.project
    }

    pub fn pod(&self) -> &PodId {
        &self.pod
    }

    pub fn manifest_digest(&self) -> &ManifestDigest {
        &self.manifest_digest
    }

    pub fn level(&self) -> ExpertLevel {
        self.level
    }

    pub fn base_model(&self) -> &BaseModelRef {
        &self.base_model
    }

    pub fn generation(&self) -> Generation {
        self.generation
    }

    /// The spec without its training-produced manifest digest.
    pub fn intent(&self) -> ResultIntent {
        ResultIntent {
            project: self.project.clone(),
            pod: self.pod.clone(),
            level: self.level,
            base_model: self.base_model.clone(),
            generation: self.generation,
        }
    }
}

/// The pre-training intent of a result expert: every field of an [`ExpertPodSpec`] except the
/// manifest digest, which only training produces. It carries the project, pod id, generation,
/// level (including the Token layer and slot) and base model ref.
///
/// A request names its intended results this way, so the [`OperationId`] separates two requests
/// that differ in what they would create, while staying stable across training.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResultIntent {
    project: ProjectId,
    pod: PodId,
    level: ExpertLevel,
    base_model: BaseModelRef,
    generation: Generation,
}

impl ResultIntent {
    /// Build an intent. Rejects blank or non-canonical project and pod ids and generation zero,
    /// like [`ExpertPodSpec::new`].
    pub fn new(
        project: ProjectId,
        pod: PodId,
        level: ExpertLevel,
        base_model: BaseModelRef,
        generation: Generation,
    ) -> Result<Self, ExpertIdentityError> {
        check_project_id(&project)?;
        check_pod_id(&pod)?;
        check_generation(generation)?;
        Ok(Self {
            project,
            pod,
            level,
            base_model,
            generation,
        })
    }

    pub fn project(&self) -> &ProjectId {
        &self.project
    }

    pub fn pod(&self) -> &PodId {
        &self.pod
    }

    pub fn level(&self) -> ExpertLevel {
        self.level
    }

    pub fn base_model(&self) -> &BaseModelRef {
        &self.base_model
    }

    pub fn generation(&self) -> Generation {
        self.generation
    }
}

/// Fields of an [`ExpertTarget`], validated by [`ExpertTarget::new`].
#[derive(Clone, Debug)]
pub struct ExpertTargetInit {
    pub project: ProjectId,
    pub pod: PodId,
    pub generation: Generation,
    pub level: ExpertLevel,
    pub base_model: BaseModelRef,
    pub manifest_digest: ManifestDigest,
    pub state: ExpertState,
    pub state_version: u64,
}

/// An existing expert a proposal refers to, pinned to a generation, manifest and observed state.
///
/// `state` and `state_version` (the transition epoch, so a compare-and-swap also covers a state
/// change at the same generation) are caller-observed claims. They are not authority: the commit
/// re-checks them against the authoritative record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertTarget {
    project: ProjectId,
    pod: PodId,
    generation: Generation,
    level: ExpertLevel,
    base_model: BaseModelRef,
    manifest_digest: ManifestDigest,
    state: ExpertState,
    state_version: u64,
}

impl ExpertTarget {
    /// Build a target. Rejects blank or non-canonical project and pod ids and generation zero.
    pub fn new(init: ExpertTargetInit) -> Result<Self, ExpertIdentityError> {
        let ExpertTargetInit {
            project,
            pod,
            generation,
            level,
            base_model,
            manifest_digest,
            state,
            state_version,
        } = init;
        check_project_id(&project)?;
        check_pod_id(&pod)?;
        check_generation(generation)?;
        Ok(Self {
            project,
            pod,
            generation,
            level,
            base_model,
            manifest_digest,
            state,
            state_version,
        })
    }

    pub fn project(&self) -> &ProjectId {
        &self.project
    }

    pub fn pod(&self) -> &PodId {
        &self.pod
    }

    pub fn generation(&self) -> Generation {
        self.generation
    }

    pub fn level(&self) -> ExpertLevel {
        self.level
    }

    pub fn base_model(&self) -> &BaseModelRef {
        &self.base_model
    }

    pub fn manifest_digest(&self) -> &ManifestDigest {
        &self.manifest_digest
    }

    /// Caller-observed state; re-checked by the commit.
    pub fn state(&self) -> ExpertState {
        self.state
    }

    /// Caller-observed transition epoch; re-checked by the commit.
    pub fn state_version(&self) -> u64 {
        self.state_version
    }
}

/// The pod that makes a proposal.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProposerRef {
    project: ProjectId,
    pod: PodId,
    generation: Generation,
}

impl ProposerRef {
    pub fn new(
        project: ProjectId,
        pod: PodId,
        generation: Generation,
    ) -> Result<Self, ExpertIdentityError> {
        check_project_id(&project)?;
        check_pod_id(&pod)?;
        check_generation(generation)?;
        Ok(Self {
            project,
            pod,
            generation,
        })
    }

    pub fn project(&self) -> &ProjectId {
        &self.project
    }

    pub fn pod(&self) -> &PodId {
        &self.pod
    }

    pub fn generation(&self) -> Generation {
        self.generation
    }
}

/// Number of operands an operation needs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OperandCount {
    Exactly(usize),
    AtLeast(usize),
}

impl OperandCount {
    fn accepts(self, n: usize) -> bool {
        match self {
            Self::Exactly(k) => n == k,
            Self::AtLeast(k) => n >= k,
        }
    }
}

impl fmt::Display for OperandCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exactly(k) => write!(f, "exactly {k}"),
            Self::AtLeast(k) => write!(f, "at least {k}"),
        }
    }
}

/// Borrowed view of every field of an [`ExpertProposal`].
///
/// The field list is the complete content of a proposal. It is deliberately free of thresholds,
/// gates, verifier settings and policy parameters (see the module documentation).
#[derive(Clone, Copy, Debug)]
pub struct ProposalParts<'a> {
    pub project: &'a ProjectId,
    pub proposer: &'a ProposerRef,
    pub observed_at: CommitIndex,
    pub op: &'a StructuralOp,
    pub rationale: &'a RationaleDigest,
}

/// A validated, inert request for a structural change.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertProposal {
    project: ProjectId,
    proposer: ProposerRef,
    observed_at: CommitIndex,
    op: StructuralOp,
    rationale: RationaleDigest,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ExpertProposalError {
    /// The proposal project id is blank.
    BlankProject,
    /// The proposal project id is not 1-128 bytes of ASCII [A-Za-z0-9._:-].
    NonCanonicalProject,
    /// The proposer belongs to another project.
    ProposerProjectMismatch {
        expected: ProjectId,
        found: ProjectId,
    },
    /// The operation needs a different number of source experts.
    WrongSourceCount {
        op: OpKind,
        expected: OperandCount,
        found: usize,
    },
    /// The operation needs a different number of result experts.
    WrongResultCount {
        op: OpKind,
        expected: OperandCount,
        found: usize,
    },
    /// A source or result belongs to a project other than the proposal's project.
    MixedProjects {
        expected: ProjectId,
        found: ProjectId,
    },
    /// An operand is observed `Fenced`. No proposal may name a fenced expert: it reaches
    /// `Retired(InputRevoked)` only through the runtime revocation path.
    FencedTarget { op: OpKind, pod: PodId },
    /// A source's state does not allow the operation.
    WrongTargetState {
        op: OpKind,
        pod: PodId,
        state: ExpertState,
    },
    /// Merge or Prune lists the same (project, pod, generation) twice.
    DuplicateTarget { pod: PodId, generation: Generation },
    /// Merge or Prune lists two generations of the same logical expert.
    MultipleGenerations {
        pod: PodId,
        first: Generation,
        second: Generation,
    },
    /// Merge and Prune are defined only for Token-level experts.
    NotTokenLevel { op: OpKind },
    /// Sources mix levels, or Token experts of different layers.
    MixedLevels,
    /// Merge or Prune combines experts adapted from different base models.
    MixedBaseModels,
    /// Merge combines sources in different states.
    MixedStates,
    /// Reactivate of a `Retired` target that cannot seed (any reason but `GateFailed`).
    SeedUnavailable { pod: PodId, reason: RetiredReason },
    /// Retire with a reason that is not a proposal: `Pruned` (use Prune) or `InputRevoked`
    /// (a runtime-grant ledger event).
    ReasonNotProposable { reason: RetiredReason },
    /// Reactivate of a `Retired` target needs a result spec.
    MissingResult { op: OpKind },
    /// Reactivate of a `Frozen` target must not carry a result spec.
    UnexpectedResult { op: OpKind },
    /// A result reuses the pod id of a source (Merge, Split, Birth results are new experts).
    ResultPodIsSource { pod: PodId },
    /// A reactivation result must be a new generation of the retired expert itself.
    ResultNotSameExpert { expected: PodId, found: PodId },
    /// A fresh expert (Birth, Merge, Split result) must be generation 1.
    FreshResultGeneration { pod: PodId, found: Generation },
    /// A reactivation result must be exactly the source generation plus one; `expected` is
    /// `None` when that would overflow.
    ResultGenerationNotNext {
        pod: PodId,
        result: Generation,
        expected: Option<Generation>,
    },
    /// A result's base model differs from its sources'.
    ResultBaseModelMismatch { pod: PodId },
    /// A result's level is incompatible with its sources' level.
    ResultLevelMismatch { pod: PodId },
    /// A result pod is listed twice.
    DuplicateResult { pod: PodId },
    /// Two results claim the same Token `(layer, index)` slot.
    DuplicateResultSlot { layer: u16, index: u16 },
    /// Two different source experts occupy the same Token `(layer, index)` slot.
    DuplicateSourceSlot { layer: u16, index: u16 },
    /// A decoded record held an identity field that is not valid.
    Identity(ExpertIdentityError),
    /// The bytes are not the canonical encoding of an [`ExpertCommitRecord`].
    MalformedRecord(&'static str),
}

impl fmt::Display for ExpertProposalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BlankProject => f.write_str("proposal project id is blank"),
            Self::NonCanonicalProject => {
                f.write_str("proposal project id is not 1-128 bytes of ASCII [A-Za-z0-9._:-]")
            }
            Self::ProposerProjectMismatch { expected, found } => write!(
                f,
                "proposer belongs to project {found}, but the proposal is for project {expected}"
            ),
            Self::WrongSourceCount {
                op,
                expected,
                found,
            } => write!(f, "{op} needs {expected} source(s), found {found}"),
            Self::WrongResultCount {
                op,
                expected,
                found,
            } => write!(f, "{op} needs {expected} result(s), found {found}"),
            Self::MixedProjects { expected, found } => write!(
                f,
                "expert belongs to project {found}, but the proposal is for project {expected}"
            ),
            Self::FencedTarget { op, pod } => write!(
                f,
                "{op} cannot name fenced expert {pod}; only the runtime revocation path may \
                 retire it"
            ),
            Self::WrongTargetState { op, pod, state } => {
                write!(f, "{op} cannot target expert {pod} in state {state:?}")
            }
            Self::DuplicateTarget { pod, generation } => {
                write!(
                    f,
                    "expert {pod} generation {} is listed twice",
                    generation.0
                )
            }
            Self::MultipleGenerations { pod, first, second } => write!(
                f,
                "expert {pod} is listed at generations {} and {}; one logical expert may \
                 appear once",
                first.0, second.0
            ),
            Self::NotTokenLevel { op } => write!(f, "{op} is defined only for Token-level experts"),
            Self::MixedLevels => {
                f.write_str("sources must share one level (and one layer for token level)")
            }
            Self::MixedBaseModels => {
                f.write_str("sources must share one base model (digest and generation)")
            }
            Self::MixedStates => f.write_str("merge sources must all be in the same state"),
            Self::SeedUnavailable { pod, reason } => {
                write!(
                    f,
                    "expert {pod} retired as {reason:?} cannot seed a reactivation"
                )
            }
            Self::ReasonNotProposable { reason } => {
                write!(f, "retirement reason {reason:?} cannot be proposed")
            }
            Self::MissingResult { op } => write!(f, "{op} of a retired expert needs a result"),
            Self::UnexpectedResult { op } => {
                write!(f, "{op} of a frozen expert must not carry a result")
            }
            Self::ResultPodIsSource { pod } => {
                write!(f, "result reuses the pod id of source {pod}")
            }
            Self::ResultNotSameExpert { expected, found } => write!(
                f,
                "reactivation result must be expert {expected}, found {found}"
            ),
            Self::FreshResultGeneration { pod, found } => {
                write!(
                    f,
                    "fresh expert {pod} must be generation 1, found {}",
                    found.0
                )
            }
            Self::ResultGenerationNotNext {
                pod,
                result,
                expected,
            } => match expected {
                Some(expected) => write!(
                    f,
                    "result generation {} of {pod} must be {}",
                    result.0, expected.0
                ),
                None => write!(f, "result generation of {pod} would overflow"),
            },
            Self::ResultBaseModelMismatch { pod } => {
                write!(
                    f,
                    "result {pod} has a different base model than its sources"
                )
            }
            Self::ResultLevelMismatch { pod } => {
                write!(f, "result {pod} has a level incompatible with its sources")
            }
            Self::DuplicateResult { pod } => write!(f, "result {pod} is listed twice"),
            Self::DuplicateSourceSlot { layer, index } => {
                write!(
                    f,
                    "two sources occupy token slot layer {layer} index {index}"
                )
            }
            Self::DuplicateResultSlot { layer, index } => {
                write!(
                    f,
                    "two results claim token slot layer {layer} index {index}"
                )
            }
            Self::Identity(error) => write!(f, "record holds an invalid identity: {error}"),
            Self::MalformedRecord(reason) => write!(f, "malformed expert commit record: {reason}"),
        }
    }
}

impl std::error::Error for ExpertProposalError {}

/// A structural operation as requested, before any training: [`StructuralOp`] with each result
/// spec replaced by its [`ResultIntent`] (the spec without the training-produced manifest
/// digest).
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RequestedOp {
    Birth {
        result: ResultIntent,
    },
    Merge {
        sources: Vec<ExpertTarget>,
        result: ResultIntent,
    },
    Split {
        source: ExpertTarget,
        results: Vec<ResultIntent>,
    },
    Freeze {
        target: ExpertTarget,
    },
    /// `result` is required iff `target` is `Retired` and forbidden iff it is `Frozen`.
    Reactivate {
        target: ExpertTarget,
        result: Option<ResultIntent>,
    },
    Prune {
        targets: Vec<ExpertTarget>,
    },
    Retire {
        target: ExpertTarget,
        reason: RetiredReason,
    },
}

impl RequestedOp {
    pub fn kind(&self) -> OpKind {
        match self {
            Self::Birth { .. } => OpKind::Birth,
            Self::Merge { .. } => OpKind::Merge,
            Self::Split { .. } => OpKind::Split,
            Self::Freeze { .. } => OpKind::Freeze,
            Self::Reactivate { .. } => OpKind::Reactivate,
            Self::Prune { .. } => OpKind::Prune,
            Self::Retire { .. } => OpKind::Retire,
        }
    }

    /// Existing experts the operation names.
    pub fn sources(&self) -> Vec<&ExpertTarget> {
        match self {
            Self::Birth { .. } => Vec::new(),
            Self::Merge { sources, .. } => sources.iter().collect(),
            Self::Split { source: target, .. }
            | Self::Freeze { target }
            | Self::Reactivate { target, .. }
            | Self::Retire { target, .. } => vec![target],
            Self::Prune { targets } => targets.iter().collect(),
        }
    }

    /// Intended new expert generations, in order.
    pub fn results(&self) -> Vec<&ResultIntent> {
        match self {
            Self::Birth { result } | Self::Merge { result, .. } => vec![result],
            Self::Split { results, .. } => results.iter().collect(),
            Self::Reactivate { result, .. } => result.iter().collect(),
            Self::Freeze { .. } | Self::Prune { .. } | Self::Retire { .. } => Vec::new(),
        }
    }

    fn from_structural(op: &StructuralOp) -> Self {
        match op {
            StructuralOp::Birth { result } => Self::Birth {
                result: result.intent(),
            },
            StructuralOp::Merge { sources, result } => Self::Merge {
                sources: sources.clone(),
                result: result.intent(),
            },
            StructuralOp::Split { source, results } => Self::Split {
                source: source.clone(),
                results: results.iter().map(ExpertPodSpec::intent).collect(),
            },
            StructuralOp::Freeze { target } => Self::Freeze {
                target: target.clone(),
            },
            StructuralOp::Reactivate { target, result } => Self::Reactivate {
                target: target.clone(),
                result: result.as_ref().map(ExpertPodSpec::intent),
            },
            StructuralOp::Prune { targets } => Self::Prune {
                targets: targets.clone(),
            },
            StructuralOp::Retire { target, reason } => Self::Retire {
                target: target.clone(),
                reason: *reason,
            },
        }
    }
}

/// A validated, inert request for a structural change that exists before training: everything
/// in an [`ExpertProposal`] except the training-produced manifest digest of each result, which
/// is replaced by [`ResultIntent`]s. It lets an orchestrator mint the pre-training
/// [`OperationId`] without fabricating a trained result.
///
/// [`ExpertOperationRequest::new`] runs exactly the rules [`ExpertProposal::new`] runs, in the
/// order documented there, through the same code: none of them needs the manifest digest. There
/// is no constructor binding a request to results: admission checks
/// `proposal.operation_id() == request.operation_id()`, which covers the intended results.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertOperationRequest {
    project: ProjectId,
    proposer: ProposerRef,
    observed_at: CommitIndex,
    op: RequestedOp,
    rationale: RationaleDigest,
}

impl ExpertOperationRequest {
    /// Validate and build a request. See the type documentation for the rules checked.
    pub fn new(
        project: ProjectId,
        proposer: ProposerRef,
        observed_at: CommitIndex,
        op: RequestedOp,
        rationale: RationaleDigest,
    ) -> Result<Self, ExpertProposalError> {
        check_proposal_header(&project, &proposer)?;
        validate_requested(&op, &project)?;
        Ok(Self {
            project,
            proposer,
            observed_at,
            op,
            rationale,
        })
    }

    pub fn project(&self) -> &ProjectId {
        &self.project
    }

    pub fn proposer(&self) -> &ProposerRef {
        &self.proposer
    }

    /// The committed ledger index the request was built against. A caller-observed claim.
    pub fn observed_at(&self) -> CommitIndex {
        self.observed_at
    }

    pub fn op(&self) -> &RequestedOp {
        &self.op
    }

    pub fn rationale(&self) -> &RationaleDigest {
        &self.rationale
    }

    /// Pre-training identity: SHA-256 over the canonical encoding of the project, proposer,
    /// `observed_at`, rationale digest, operation kind, operands (with observed generation,
    /// manifest digest, state and state version), for Retire the reason, and each intended
    /// result's [`ResultIntent`] in order (project, pod, level with Token layer and slot, base
    /// model ref, generation), under domain separator `ptr.pods.expert-operation.v2`. The only
    /// thing excluded is each result's training-produced manifest digest.
    pub fn operation_id(&self) -> OperationId {
        let mut e = Encoder::default();
        e.str("ptr.pods.expert-operation.v2");
        e.str(&self.project.0);
        e.str(&self.proposer.project.0);
        e.str(&self.proposer.pod.0);
        e.u64(self.proposer.generation.0);
        e.u64(self.observed_at.0);
        e.digest(self.rationale.as_bytes());
        e.u64(match self.op.kind() {
            OpKind::Birth => 0,
            OpKind::Merge => 1,
            OpKind::Split => 2,
            OpKind::Freeze => 3,
            OpKind::Reactivate => 4,
            OpKind::Prune => 5,
            OpKind::Retire => 6,
        });
        let sources = self.op.sources();
        e.u64(sources.len() as u64);
        for source in sources {
            e.target(source);
        }
        // The retirement reason is a request parameter, not a result.
        if let RequestedOp::Retire { reason, .. } = &self.op {
            e.reason(*reason);
        }
        // The intended results, in order; the manifest digest is not part of an intent.
        let results = self.op.results();
        e.u64(results.len() as u64);
        for result in results {
            e.intent(result);
        }
        OperationId(Sha256::digest(e.bytes).into())
    }
}

/// Rules 1 and 2: the proposal project id is canonical and the proposer belongs to it.
fn check_proposal_header(
    project: &ProjectId,
    proposer: &ProposerRef,
) -> Result<(), ExpertProposalError> {
    match check_project_id(project) {
        Ok(()) => {}
        Err(ExpertIdentityError::BlankProject) => return Err(ExpertProposalError::BlankProject),
        Err(_) => return Err(ExpertProposalError::NonCanonicalProject),
    }
    if &proposer.project != project {
        return Err(ExpertProposalError::ProposerProjectMismatch {
            expected: project.clone(),
            found: proposer.project.clone(),
        });
    }
    Ok(())
}

impl ExpertProposal {
    /// Validate and build a proposal. See the module documentation for what a proposal is not
    /// and for the order in which rules are checked.
    pub fn new(
        project: ProjectId,
        proposer: ProposerRef,
        observed_at: CommitIndex,
        op: StructuralOp,
        rationale: RationaleDigest,
    ) -> Result<Self, ExpertProposalError> {
        check_proposal_header(&project, &proposer)?;
        validate_requested(&RequestedOp::from_structural(&op), &project)?;
        Ok(Self {
            project,
            proposer,
            observed_at,
            op,
            rationale,
        })
    }

    pub fn project(&self) -> &ProjectId {
        &self.project
    }

    pub fn proposer(&self) -> &ProposerRef {
        &self.proposer
    }

    /// The committed ledger index the proposal was built against. A caller-observed claim.
    pub fn observed_at(&self) -> CommitIndex {
        self.observed_at
    }

    pub fn op(&self) -> &StructuralOp {
        &self.op
    }

    pub fn rationale(&self) -> &RationaleDigest {
        &self.rationale
    }

    /// Whether accepting this proposal would create a new expert generation: true for birth,
    /// merge, split and the reactivation of a `Retired` target; false for freeze, prune, retire
    /// and the reactivation of a `Frozen` target (the same generation is unfrozen in place, with
    /// no lineage transition).
    pub fn creates_new_generation(&self) -> bool {
        !self.op.results().is_empty()
    }

    /// Every field of the proposal; adding a field to the proposal means changing this struct.
    pub fn parts(&self) -> ProposalParts<'_> {
        // Compile-time guard: this destructure names every field and has no `..`. Adding any
        // field to `ExpertProposal` breaks this line and forces a review of the new field
        // against invariants #4 and #11 before `ProposalParts` is updated.
        let ExpertProposal {
            project,
            proposer,
            observed_at,
            op,
            rationale,
        } = self;
        ProposalParts {
            project,
            proposer,
            observed_at: *observed_at,
            op,
            rationale,
        }
    }

    /// The pre-result request this proposal fulfils: the same project, proposer, `observed_at`,
    /// rationale, operation kind, operands, retirement reason and the [`ResultIntent`] of each
    /// result (its spec without the training-produced manifest digest). It always validates, because the proposal already passed every rule the request checks.
    pub fn operation_request(&self) -> ExpertOperationRequest {
        ExpertOperationRequest {
            project: self.project.clone(),
            proposer: self.proposer.clone(),
            observed_at: self.observed_at,
            op: RequestedOp::from_structural(&self.op),
            rationale: self.rationale,
        }
    }

    /// Pre-training identity of the requested operation; delegates to
    /// [`ExpertOperationRequest::operation_id`], so a request and the proposal that fulfils it
    /// give the same id. It includes each result's [`ResultIntent`] (project, pod, level with
    /// Token layer and slot, base model ref, generation) and excludes only each result's
    /// training-produced manifest digest. The candidate is identified by
    /// [`ExpertProposal::proposal_id`] after training.
    pub fn operation_id(&self) -> OperationId {
        self.operation_request().operation_id()
    }

    /// Deterministic identity of the candidate: SHA-256 over the canonical encoding of every
    /// field, results included. Exists after training; see [`ExpertProposal::operation_id`] for
    /// the identity that exists before.
    pub fn proposal_id(&self) -> ProposalId {
        let ProposalParts {
            project,
            proposer,
            observed_at,
            op,
            rationale,
        } = self.parts();
        let mut e = Encoder::default();
        e.str("ptr.pods.expert-proposal.v2");
        e.str(&project.0);
        e.str(&proposer.project.0);
        e.str(&proposer.pod.0);
        e.u64(proposer.generation.0);
        e.u64(observed_at.0);
        e.digest(rationale.as_bytes());
        match op {
            StructuralOp::Birth { result } => {
                e.u64(0);
                e.spec(result);
            }
            StructuralOp::Merge { sources, result } => {
                e.u64(1);
                e.targets(sources);
                e.spec(result);
            }
            StructuralOp::Split { source, results } => {
                e.u64(2);
                e.target(source);
                e.u64(results.len() as u64);
                for result in results {
                    e.spec(result);
                }
            }
            StructuralOp::Freeze { target } => {
                e.u64(3);
                e.target(target);
            }
            StructuralOp::Reactivate { target, result } => {
                e.u64(4);
                e.target(target);
                match result {
                    Some(result) => {
                        e.u64(1);
                        e.spec(result);
                    }
                    None => e.u64(0),
                }
            }
            StructuralOp::Prune { targets } => {
                e.u64(5);
                e.targets(targets);
            }
            StructuralOp::Retire { target, reason } => {
                e.u64(6);
                e.target(target);
                e.reason(*reason);
            }
        }
        ProposalId(Sha256::digest(e.bytes).into())
    }
}

#[derive(Default)]
struct Encoder {
    bytes: Vec<u8>,
}

impl Encoder {
    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn str(&mut self, value: &str) {
        self.u64(value.len() as u64);
        self.bytes.extend_from_slice(value.as_bytes());
    }

    fn digest(&mut self, value: &Digest) {
        self.bytes.extend_from_slice(value);
    }

    fn level(&mut self, level: ExpertLevel) {
        match level {
            ExpertLevel::Token { layer, index } => {
                self.u64(0);
                self.u64(u64::from(layer));
                self.u64(u64::from(index));
            }
            ExpertLevel::Segment => self.u64(1),
        }
    }

    fn reason(&mut self, reason: RetiredReason) {
        self.u64(match reason {
            RetiredReason::Pruned => 0,
            RetiredReason::Superseded => 1,
            RetiredReason::GateFailed => 2,
            RetiredReason::InputRevoked => 3,
        });
    }

    fn state(&mut self, state: ExpertState) {
        match state {
            ExpertState::Candidate => self.u64(0),
            ExpertState::Gated => self.u64(1),
            ExpertState::Serving => self.u64(2),
            ExpertState::Frozen => self.u64(3),
            ExpertState::Fenced => self.u64(4),
            ExpertState::Superseded => self.u64(5),
            ExpertState::Retired(reason) => {
                self.u64(6);
                self.reason(reason);
            }
        }
    }

    fn base_model(&mut self, base: &BaseModelRef) {
        self.str(&base.id.0);
        self.digest(base.digest.as_bytes());
        self.u64(base.generation);
    }

    fn spec(&mut self, spec: &ExpertPodSpec) {
        self.str(&spec.project.0);
        self.str(&spec.pod.0);
        self.digest(spec.manifest_digest.as_bytes());
        self.level(spec.level);
        self.base_model(&spec.base_model);
        self.u64(spec.generation.0);
    }

    fn intent(&mut self, intent: &ResultIntent) {
        self.str(&intent.project.0);
        self.str(&intent.pod.0);
        self.level(intent.level);
        self.base_model(&intent.base_model);
        self.u64(intent.generation.0);
    }

    fn target(&mut self, target: &ExpertTarget) {
        self.str(&target.project.0);
        self.str(&target.pod.0);
        self.u64(target.generation.0);
        self.level(target.level);
        self.base_model(&target.base_model);
        self.digest(target.manifest_digest.as_bytes());
        self.state(target.state);
        self.u64(target.state_version);
    }

    fn targets(&mut self, targets: &[ExpertTarget]) {
        self.u64(targets.len() as u64);
        for target in targets {
            self.target(target);
        }
    }
}

/// Rules 3 to 8 over a requested operation: the single implementation behind both
/// [`ExpertProposal::new`] (which passes the intents of its result specs) and
/// [`ExpertOperationRequest::new`].
fn validate_requested(op: &RequestedOp, project: &ProjectId) -> Result<(), ExpertProposalError> {
    let kind = op.kind();
    let sources = op.sources();
    let results = op.results();

    // 3. operand counts (a Split's result count is a result rule, checked here in order).
    check_source_counts(kind, &sources)?;
    if let RequestedOp::Split { .. } = op {
        let expected = OperandCount::AtLeast(2);
        if !expected.accepts(results.len()) {
            return Err(ExpertProposalError::WrongResultCount {
                op: kind,
                expected,
                found: results.len(),
            });
        }
    }

    // 4. projects, sources first and then results.
    check_source_projects(&sources, project)?;
    if let Some(found) = results.iter().map(|r| &r.project).find(|p| *p != project) {
        return Err(ExpertProposalError::MixedProjects {
            expected: project.clone(),
            found: found.clone(),
        });
    }

    // 5. source states and 6. source-only operation structure.
    let reason = match op {
        RequestedOp::Retire { reason, .. } => Some(*reason),
        _ => None,
    };
    check_source_states(kind, &sources)?;
    check_source_structure(kind, &sources, reason)?;

    // 6 (continued). A reactivation result is present iff the target is `Retired`.
    if let RequestedOp::Reactivate { target, result } = op {
        match (target.state, result) {
            (ExpertState::Retired(_), None) => {
                return Err(ExpertProposalError::MissingResult { op: kind });
            }
            (ExpertState::Frozen, Some(_)) => {
                return Err(ExpertProposalError::UnexpectedResult { op: kind });
            }
            _ => {}
        }
    }

    // 7. result relations.
    for result in &results {
        check_result(kind, &sources, result)?;
    }

    // 8. distinct result pods, then distinct Token slots (a source's slot may be reused).
    let mut pods = BTreeMap::new();
    for result in &results {
        if pods.insert(&result.pod, ()).is_some() {
            return Err(ExpertProposalError::DuplicateResult {
                pod: result.pod.clone(),
            });
        }
    }
    let mut slots = BTreeMap::new();
    for result in &results {
        if let ExpertLevel::Token { layer, index } = result.level {
            if slots.insert((layer, index), ()).is_some() {
                return Err(ExpertProposalError::DuplicateResultSlot { layer, index });
            }
        }
    }
    Ok(())
}

/// Rule 3, source part: Merge needs at least two sources, Prune at least one.
fn check_source_counts(kind: OpKind, sources: &[&ExpertTarget]) -> Result<(), ExpertProposalError> {
    match kind {
        OpKind::Merge => count_sources(kind, sources, OperandCount::AtLeast(2)),
        OpKind::Prune => count_sources(kind, sources, OperandCount::AtLeast(1)),
        OpKind::Birth | OpKind::Split | OpKind::Freeze | OpKind::Reactivate | OpKind::Retire => {
            Ok(())
        }
    }
}

/// Rule 4, source part.
fn check_source_projects(
    sources: &[&ExpertTarget],
    project: &ProjectId,
) -> Result<(), ExpertProposalError> {
    match sources.iter().map(|t| &t.project).find(|p| *p != project) {
        Some(found) => Err(ExpertProposalError::MixedProjects {
            expected: project.clone(),
            found: found.clone(),
        }),
        None => Ok(()),
    }
}

/// Rule 5: a non-`GateFailed` retired reactivation target is `SeedUnavailable`, a `Fenced`
/// operand is `FencedTarget`, then the (operation, state) table.
fn check_source_states(kind: OpKind, sources: &[&ExpertTarget]) -> Result<(), ExpertProposalError> {
    if kind == OpKind::Reactivate {
        if let Some(target) = sources.first() {
            if let ExpertState::Retired(reason) = target.state {
                if reason != RetiredReason::GateFailed {
                    return Err(ExpertProposalError::SeedUnavailable {
                        pod: target.pod.clone(),
                        reason,
                    });
                }
            }
        }
    }
    if let Some(target) = sources.iter().find(|t| t.state == ExpertState::Fenced) {
        return Err(ExpertProposalError::FencedTarget {
            op: kind,
            pod: target.pod.clone(),
        });
    }
    if let Some(target) = sources.iter().find(|t| !kind.allows_state(t.state)) {
        return Err(ExpertProposalError::WrongTargetState {
            op: kind,
            pod: target.pod.clone(),
            state: target.state,
        });
    }
    Ok(())
}

/// Rule 6, the part that involves no result: Merge and Prune source structure and the
/// retirement reason.
fn check_source_structure(
    kind: OpKind,
    sources: &[&ExpertTarget],
    reason: Option<RetiredReason>,
) -> Result<(), ExpertProposalError> {
    match kind {
        OpKind::Merge => {
            require_token_level(kind, sources)?;
            check_unique(sources)?;
            check_same_layer(sources)?;
            check_same_base(sources)?;
            check_distinct_slots(sources)?;
            if sources.iter().any(|t| t.state != sources[0].state) {
                return Err(ExpertProposalError::MixedStates);
            }
        }
        OpKind::Prune => {
            require_token_level(kind, sources)?;
            check_unique(sources)?;
            check_same_layer(sources)?;
            check_same_base(sources)?;
            check_distinct_slots(sources)?;
        }
        OpKind::Retire => match reason {
            Some(RetiredReason::Superseded | RetiredReason::GateFailed) | None => {}
            Some(reason @ (RetiredReason::Pruned | RetiredReason::InputRevoked)) => {
                return Err(ExpertProposalError::ReasonNotProposable { reason });
            }
        },
        OpKind::Birth | OpKind::Split | OpKind::Freeze | OpKind::Reactivate => {}
    }
    Ok(())
}

fn count_sources(
    op: OpKind,
    sources: &[&ExpertTarget],
    expected: OperandCount,
) -> Result<(), ExpertProposalError> {
    if expected.accepts(sources.len()) {
        Ok(())
    } else {
        Err(ExpertProposalError::WrongSourceCount {
            op,
            expected,
            found: sources.len(),
        })
    }
}

fn require_token_level(op: OpKind, sources: &[&ExpertTarget]) -> Result<(), ExpertProposalError> {
    if sources
        .iter()
        .any(|t| matches!(t.level, ExpertLevel::Segment))
    {
        Err(ExpertProposalError::NotTokenLevel { op })
    } else {
        Ok(())
    }
}

/// One logical expert is (project, pod). Two generations of it are never listed together, and
/// the same generation is not listed twice. All sources share the proposal project by now.
fn check_unique(sources: &[&ExpertTarget]) -> Result<(), ExpertProposalError> {
    let mut seen = BTreeMap::new();
    for target in sources {
        if let Some(first) = seen.insert((&target.project, &target.pod), target.generation) {
            return Err(if first == target.generation {
                ExpertProposalError::DuplicateTarget {
                    pod: target.pod.clone(),
                    generation: target.generation,
                }
            } else {
                ExpertProposalError::MultipleGenerations {
                    pod: target.pod.clone(),
                    first,
                    second: target.generation,
                }
            });
        }
    }
    Ok(())
}

/// Two different sources cannot occupy one Token slot (the same pod twice is `check_unique`'s).
fn check_distinct_slots(sources: &[&ExpertTarget]) -> Result<(), ExpertProposalError> {
    let mut seen = BTreeMap::new();
    for target in sources {
        if let ExpertLevel::Token { layer, index } = target.level {
            if seen.insert((layer, index), &target.pod).is_some() {
                return Err(ExpertProposalError::DuplicateSourceSlot { layer, index });
            }
        }
    }
    Ok(())
}

fn check_same_layer(sources: &[&ExpertTarget]) -> Result<(), ExpertProposalError> {
    let first = sources[0].level;
    if sources.iter().any(|t| !t.level.compatible_with(first)) {
        Err(ExpertProposalError::MixedLevels)
    } else {
        Ok(())
    }
}

fn check_same_base(sources: &[&ExpertTarget]) -> Result<(), ExpertProposalError> {
    let first = &sources[0].base_model;
    if sources.iter().any(|t| &t.base_model != first) {
        Err(ExpertProposalError::MixedBaseModels)
    } else {
        Ok(())
    }
}

fn check_result(
    kind: OpKind,
    sources: &[&ExpertTarget],
    result: &ResultIntent,
) -> Result<(), ExpertProposalError> {
    match kind {
        OpKind::Reactivate => {
            // A new generation of the retired expert itself, exactly one above the source.
            let source = sources[0];
            if result.pod != source.pod {
                return Err(ExpertProposalError::ResultNotSameExpert {
                    expected: source.pod.clone(),
                    found: result.pod.clone(),
                });
            }
            let expected = source.generation.0.checked_add(1).map(Generation);
            if expected != Some(result.generation) {
                return Err(ExpertProposalError::ResultGenerationNotNext {
                    pod: result.pod.clone(),
                    result: result.generation,
                    expected,
                });
            }
        }
        OpKind::Merge | OpKind::Split => {
            if sources.iter().any(|t| t.pod == result.pod) {
                return Err(ExpertProposalError::ResultPodIsSource {
                    pod: result.pod.clone(),
                });
            }
        }
        OpKind::Birth | OpKind::Freeze | OpKind::Prune | OpKind::Retire => {}
    }
    if matches!(kind, OpKind::Birth | OpKind::Merge | OpKind::Split)
        && result.generation != Generation(1)
    {
        return Err(ExpertProposalError::FreshResultGeneration {
            pod: result.pod.clone(),
            found: result.generation,
        });
    }
    if let Some(first) = sources.first() {
        if sources.iter().any(|t| t.base_model != result.base_model) {
            return Err(ExpertProposalError::ResultBaseModelMismatch {
                pod: result.pod.clone(),
            });
        }
        // Reactivation restores the exact slot; Merge and Split need only the same layer.
        let level_ok = match kind {
            OpKind::Reactivate => result.level == first.level,
            _ => result.level.compatible_with(first.level),
        };
        if !level_ok {
            return Err(ExpertProposalError::ResultLevelMismatch {
                pod: result.pod.clone(),
            });
        }
    }
    Ok(())
}

/// Leading magic and version of a version 1 [`ExpertCommitRecord`] encoding, which carries only
/// the evidence digest. A change to the layout below needs a new magic, so an old reader refuses
/// new bytes instead of misreading them.
const RECORD_MAGIC_V1: &[u8; 8] = b"PTREXR01";

/// Leading magic of a version 2 record, which also embeds the evidence and the gate
/// configuration digest the host required for its epoch.
const RECORD_MAGIC_V2: &[u8; 8] = b"PTREXR02";

/// The largest encoded [`ExpertCommitRecord`] [`ExpertCommitRecord::decode_canonical`] reads.
/// It equals the ledger's `MAX_EXPERT_RECORD_BYTES` (1 MiB), the most a structural event may
/// carry, and longer input is refused before any parsing.
pub const MAX_EXPERT_COMMIT_RECORD_BYTES: usize = 1 << 20;

/// The evidence a version 2 [`ExpertCommitRecord`] embeds so a replaying reader can re-verify it
/// instead of trusting the digest: the gate report's canonical bytes (opaque to this crate;
/// `ptr-lineage` `BoundGateReport::decode_canonical` reads them) and the digest of the gate
/// configuration the host required for the record's gate epoch when it admitted the proposal.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertEvidence {
    required_gate_config_digest: [u8; 32],
    report: Vec<u8>,
}

impl ExpertEvidence {
    pub fn required_gate_config_digest(&self) -> &[u8; 32] {
        &self.required_gate_config_digest
    }

    /// The canonical bytes of the embedded gate report.
    pub fn report(&self) -> &[u8] {
        &self.report
    }
}

/// An [`ExpertProposal`] as committed to the ledger, with the digest of the trusted evidence it
/// was admitted on and the gate-policy epoch that evidence was issued under.
///
/// The record is the opaque payload of the ledger's structural-commit event. This type only
/// defines and re-validates the bytes: it does not decide admission, and holding one proves
/// nothing about the evidence. The fields are private, and [`ExpertCommitRecord::decode_canonical`]
/// rebuilds the proposal through [`ExpertProposal::new`], so bytes that decode always describe a
/// proposal that validates.
///
/// # Versions
///
/// A version 2 record ([`ExpertCommitRecord::new_with_evidence`], the only kind admission
/// writes) embeds the gate report and the required gate configuration digest, so replay can
/// re-derive the report digest, check that it passed, is bound to this proposal and operation and
/// was produced under the configuration the host required. A version 1 record
/// ([`ExpertCommitRecord::new`]) carries only the evidence digest; it still decodes, because
/// histories written before version 2 exist, but replay can verify nothing about its evidence
/// and treats it as unverified legacy (see `ptr-runtime`'s `ExpertEvidenceStatus`). Nothing new
/// writes version 1.
///
/// # Canonical encoding
///
/// All integers are little-endian. `record_v1 = magic "PTREXR01" | proposal | evidence_digest[32]
/// | gate_epoch u64`; `record_v2 = magic "PTREXR02" | proposal | evidence_digest[32] |
/// gate_epoch u64 | required_gate_config_digest[32] | report (u32 length, bytes)`. A string is a
/// `u32` byte length and its UTF-8 bytes, a list a `u32` count and
/// its items, an option a `0`/`1` byte. The proposal is its project, proposer (project, pod,
/// generation), `observed_at`, rationale digest and a one-byte operation tag (0 Birth, 1 Merge,
/// 2 Split, 3 Freeze, 4 Reactivate, 5 Prune, 6 Retire) with its operands, in field order. A level
/// is a tag byte (0 Token with `layer u16`, `index u16`; 1 Segment); a state is a tag byte (0
/// Candidate, 1 Gated, 2 Serving, 3 Frozen, 4 Fenced, 5 Superseded, 6 Retired followed by a
/// reason byte); a reason byte is 0 Pruned, 1 Superseded, 2 GateFailed, 3 InputRevoked.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertCommitRecord {
    proposal: ExpertProposal,
    evidence_digest: [u8; 32],
    gate_epoch: u64,
    evidence: Option<ExpertEvidence>,
}

impl ExpertCommitRecord {
    /// A legacy version 1 record: a proposal bound to the digest of the evidence it was admitted
    /// on and the gate-policy epoch, with no evidence embedded. Rejects an all-zero evidence
    /// digest. Admission does not use this; see [`ExpertCommitRecord::new_with_evidence`].
    pub fn new(
        proposal: ExpertProposal,
        evidence_digest: [u8; 32],
        gate_epoch: u64,
    ) -> Result<Self, ExpertProposalError> {
        if evidence_digest == ZERO_DIGEST {
            return Err(ExpertProposalError::MalformedRecord(
                "evidence digest is all zero",
            ));
        }
        Ok(Self {
            proposal,
            evidence_digest,
            gate_epoch,
            evidence: None,
        })
    }

    /// A version 2 record embedding the gate report (`report`, its canonical bytes) and the gate
    /// configuration digest the host required for `gate_epoch`. This type does not parse the
    /// report: the reader that can (`ptr-lineage`) checks it against `evidence_digest`.
    ///
    /// # Errors
    /// Rejects an all-zero evidence digest or configuration digest and an empty report.
    pub fn new_with_evidence(
        proposal: ExpertProposal,
        evidence_digest: [u8; 32],
        gate_epoch: u64,
        required_gate_config_digest: [u8; 32],
        report: Vec<u8>,
    ) -> Result<Self, ExpertProposalError> {
        if required_gate_config_digest == ZERO_DIGEST {
            return Err(ExpertProposalError::MalformedRecord(
                "required gate configuration digest is all zero",
            ));
        }
        if report.is_empty() {
            return Err(ExpertProposalError::MalformedRecord(
                "embedded gate report is empty",
            ));
        }
        let mut record = Self::new(proposal, evidence_digest, gate_epoch)?;
        record.evidence = Some(ExpertEvidence {
            required_gate_config_digest,
            report,
        });
        Ok(record)
    }

    pub fn proposal(&self) -> &ExpertProposal {
        &self.proposal
    }

    pub fn evidence_digest(&self) -> &[u8; 32] {
        &self.evidence_digest
    }

    pub fn gate_epoch(&self) -> u64 {
        self.gate_epoch
    }

    /// The embedded evidence of a version 2 record; `None` for a legacy version 1 record.
    pub fn evidence(&self) -> Option<&ExpertEvidence> {
        self.evidence.as_ref()
    }

    /// The canonical bytes; see the type documentation for the layout.
    pub fn encode_canonical(&self) -> Vec<u8> {
        let mut w = Writer::default();
        w.bytes.extend_from_slice(match self.evidence {
            None => RECORD_MAGIC_V1,
            Some(_) => RECORD_MAGIC_V2,
        });
        w.proposal(&self.proposal);
        w.bytes.extend_from_slice(&self.evidence_digest);
        w.u64(self.gate_epoch);
        if let Some(evidence) = &self.evidence {
            w.bytes
                .extend_from_slice(&evidence.required_gate_config_digest);
            w.count(evidence.report.len());
            w.bytes.extend_from_slice(&evidence.report);
        }
        w.bytes
    }

    /// Decode and re-validate. The proposal is rebuilt through [`ExpertProposal::new`] and every
    /// identity through its constructor.
    ///
    /// # Errors
    /// Rejects a wrong magic, truncated or trailing bytes, an unknown tag, an invalid identity or
    /// digest, a proposal that does not validate, and any byte string that is not exactly the
    /// canonical encoding of the record it decodes to.
    pub fn decode_canonical(bytes: &[u8]) -> Result<Self, ExpertProposalError> {
        if bytes.len() > MAX_EXPERT_COMMIT_RECORD_BYTES {
            return Err(ExpertProposalError::MalformedRecord(
                "record exceeds the size limit",
            ));
        }
        let mut r = Reader { bytes };
        let magic = r.take(RECORD_MAGIC_V1.len())?;
        let embeds_evidence = if magic == RECORD_MAGIC_V1 {
            false
        } else if magic == RECORD_MAGIC_V2 {
            true
        } else {
            return Err(ExpertProposalError::MalformedRecord("unknown magic"));
        };
        let proposal = r.proposal()?;
        let evidence_digest = r.array32()?;
        let gate_epoch = r.u64()?;
        let record = if embeds_evidence {
            let required = r.array32()?;
            let len = r.count()?;
            let report = r.take(len)?.to_vec();
            Self::new_with_evidence(proposal, evidence_digest, gate_epoch, required, report)?
        } else {
            Self::new(proposal, evidence_digest, gate_epoch)?
        };
        if !r.bytes.is_empty() {
            return Err(ExpertProposalError::MalformedRecord("trailing bytes"));
        }
        if record.encode_canonical() != bytes {
            return Err(ExpertProposalError::MalformedRecord(
                "non-canonical encoding",
            ));
        }
        Ok(record)
    }
}

#[derive(Default)]
struct Writer {
    bytes: Vec<u8>,
}

impl Writer {
    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn count(&mut self, value: usize) {
        // Ids are at most 128 bytes and no operation holds anywhere near 2^32 operands; a value
        // that did not fit would encode a truncated count and fail to decode, never alias.
        self.bytes.extend_from_slice(&(value as u32).to_le_bytes());
    }

    fn str(&mut self, value: &str) {
        self.count(value.len());
        self.bytes.extend_from_slice(value.as_bytes());
    }

    fn digest(&mut self, value: &Digest) {
        self.bytes.extend_from_slice(value);
    }

    fn level(&mut self, level: ExpertLevel) {
        match level {
            ExpertLevel::Token { layer, index } => {
                self.u8(0);
                self.u16(layer);
                self.u16(index);
            }
            ExpertLevel::Segment => self.u8(1),
        }
    }

    fn reason(&mut self, reason: RetiredReason) {
        self.u8(match reason {
            RetiredReason::Pruned => 0,
            RetiredReason::Superseded => 1,
            RetiredReason::GateFailed => 2,
            RetiredReason::InputRevoked => 3,
        });
    }

    fn state(&mut self, state: ExpertState) {
        match state {
            ExpertState::Candidate => self.u8(0),
            ExpertState::Gated => self.u8(1),
            ExpertState::Serving => self.u8(2),
            ExpertState::Frozen => self.u8(3),
            ExpertState::Fenced => self.u8(4),
            ExpertState::Superseded => self.u8(5),
            ExpertState::Retired(reason) => {
                self.u8(6);
                self.reason(reason);
            }
        }
    }

    fn base_model(&mut self, base: &BaseModelRef) {
        self.str(&base.id.0);
        self.digest(base.digest.as_bytes());
        self.u64(base.generation);
    }

    fn spec(&mut self, spec: &ExpertPodSpec) {
        self.str(&spec.project.0);
        self.str(&spec.pod.0);
        self.digest(spec.manifest_digest.as_bytes());
        self.level(spec.level);
        self.base_model(&spec.base_model);
        self.u64(spec.generation.0);
    }

    fn target(&mut self, target: &ExpertTarget) {
        self.str(&target.project.0);
        self.str(&target.pod.0);
        self.u64(target.generation.0);
        self.level(target.level);
        self.base_model(&target.base_model);
        self.digest(target.manifest_digest.as_bytes());
        self.state(target.state);
        self.u64(target.state_version);
    }

    fn targets(&mut self, targets: &[ExpertTarget]) {
        self.count(targets.len());
        for target in targets {
            self.target(target);
        }
    }

    fn proposal(&mut self, proposal: &ExpertProposal) {
        let ProposalParts {
            project,
            proposer,
            observed_at,
            op,
            rationale,
        } = proposal.parts();
        self.str(&project.0);
        self.str(&proposer.project.0);
        self.str(&proposer.pod.0);
        self.u64(proposer.generation.0);
        self.u64(observed_at.0);
        self.digest(rationale.as_bytes());
        match op {
            StructuralOp::Birth { result } => {
                self.u8(0);
                self.spec(result);
            }
            StructuralOp::Merge { sources, result } => {
                self.u8(1);
                self.targets(sources);
                self.spec(result);
            }
            StructuralOp::Split { source, results } => {
                self.u8(2);
                self.target(source);
                self.count(results.len());
                for result in results {
                    self.spec(result);
                }
            }
            StructuralOp::Freeze { target } => {
                self.u8(3);
                self.target(target);
            }
            StructuralOp::Reactivate { target, result } => {
                self.u8(4);
                self.target(target);
                match result {
                    Some(result) => {
                        self.u8(1);
                        self.spec(result);
                    }
                    None => self.u8(0),
                }
            }
            StructuralOp::Prune { targets } => {
                self.u8(5);
                self.targets(targets);
            }
            StructuralOp::Retire { target, reason } => {
                self.u8(6);
                self.target(target);
                self.reason(*reason);
            }
        }
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
}

fn malformed<T>(reason: &'static str) -> Result<T, ExpertProposalError> {
    Err(ExpertProposalError::MalformedRecord(reason))
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], ExpertProposalError> {
        if self.bytes.len() < n {
            return malformed("truncated");
        }
        let (head, tail) = self.bytes.split_at(n);
        self.bytes = tail;
        Ok(head)
    }

    fn u8(&mut self) -> Result<u8, ExpertProposalError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, ExpertProposalError> {
        Ok(u16::from_le_bytes(
            self.take(2)?.try_into().expect("2 bytes"),
        ))
    }

    fn u32(&mut self) -> Result<u32, ExpertProposalError> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }

    fn u64(&mut self) -> Result<u64, ExpertProposalError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("8 bytes"),
        ))
    }

    fn array32(&mut self) -> Result<[u8; 32], ExpertProposalError> {
        Ok(self.take(32)?.try_into().expect("32 bytes"))
    }

    /// A list count. Every item takes at least one byte, so a count above the remaining length
    /// is refused before anything is allocated.
    fn count(&mut self) -> Result<usize, ExpertProposalError> {
        let n = self.u32()? as usize;
        if n > self.bytes.len() {
            return malformed("count exceeds the remaining bytes");
        }
        Ok(n)
    }

    fn string(&mut self) -> Result<String, ExpertProposalError> {
        let n = self.count()?;
        match std::str::from_utf8(self.take(n)?) {
            Ok(s) => Ok(s.to_owned()),
            Err(_) => malformed("string is not UTF-8"),
        }
    }

    fn level(&mut self) -> Result<ExpertLevel, ExpertProposalError> {
        match self.u8()? {
            0 => Ok(ExpertLevel::Token {
                layer: self.u16()?,
                index: self.u16()?,
            }),
            1 => Ok(ExpertLevel::Segment),
            _ => malformed("unknown level tag"),
        }
    }

    fn reason(&mut self) -> Result<RetiredReason, ExpertProposalError> {
        match self.u8()? {
            0 => Ok(RetiredReason::Pruned),
            1 => Ok(RetiredReason::Superseded),
            2 => Ok(RetiredReason::GateFailed),
            3 => Ok(RetiredReason::InputRevoked),
            _ => malformed("unknown retirement reason"),
        }
    }

    fn state(&mut self) -> Result<ExpertState, ExpertProposalError> {
        match self.u8()? {
            0 => Ok(ExpertState::Candidate),
            1 => Ok(ExpertState::Gated),
            2 => Ok(ExpertState::Serving),
            3 => Ok(ExpertState::Frozen),
            4 => Ok(ExpertState::Fenced),
            5 => Ok(ExpertState::Superseded),
            6 => Ok(ExpertState::Retired(self.reason()?)),
            _ => malformed("unknown state tag"),
        }
    }

    fn manifest_digest(&mut self) -> Result<ManifestDigest, ExpertProposalError> {
        match ManifestDigest::new(self.array32()?) {
            Ok(d) => Ok(d),
            Err(ZeroDigest) => malformed("manifest digest is all zero"),
        }
    }

    fn base_model(&mut self) -> Result<BaseModelRef, ExpertProposalError> {
        let id = BaseModelId::new(self.string()?).map_err(ExpertProposalError::Identity)?;
        let digest = match BaseModelDigest::new(self.array32()?) {
            Ok(d) => d,
            Err(ZeroDigest) => return malformed("base model digest is all zero"),
        };
        BaseModelRef::new(id, digest, self.u64()?).map_err(ExpertProposalError::Identity)
    }

    fn spec(&mut self) -> Result<ExpertPodSpec, ExpertProposalError> {
        let project = ProjectId(self.string()?);
        let pod = PodId(self.string()?);
        let manifest = self.manifest_digest()?;
        let level = self.level()?;
        let base = self.base_model()?;
        let generation = Generation(self.u64()?);
        ExpertPodSpec::new(project, pod, manifest, level, base, generation)
            .map_err(ExpertProposalError::Identity)
    }

    fn target(&mut self) -> Result<ExpertTarget, ExpertProposalError> {
        let project = ProjectId(self.string()?);
        let pod = PodId(self.string()?);
        let generation = Generation(self.u64()?);
        let level = self.level()?;
        let base_model = self.base_model()?;
        let manifest_digest = self.manifest_digest()?;
        let state = self.state()?;
        let state_version = self.u64()?;
        ExpertTarget::new(ExpertTargetInit {
            project,
            pod,
            generation,
            level,
            base_model,
            manifest_digest,
            state,
            state_version,
        })
        .map_err(ExpertProposalError::Identity)
    }

    fn targets(&mut self) -> Result<Vec<ExpertTarget>, ExpertProposalError> {
        let n = self.count()?;
        (0..n).map(|_| self.target()).collect()
    }

    fn proposal(&mut self) -> Result<ExpertProposal, ExpertProposalError> {
        let project = ProjectId(self.string()?);
        let proposer = ProposerRef::new(
            ProjectId(self.string()?),
            PodId(self.string()?),
            Generation(self.u64()?),
        )
        .map_err(ExpertProposalError::Identity)?;
        let observed_at = CommitIndex(self.u64()?);
        let rationale = match RationaleDigest::new(self.array32()?) {
            Ok(d) => d,
            Err(ZeroDigest) => return malformed("rationale digest is all zero"),
        };
        let op = match self.u8()? {
            0 => StructuralOp::Birth {
                result: self.spec()?,
            },
            1 => StructuralOp::Merge {
                sources: self.targets()?,
                result: self.spec()?,
            },
            2 => {
                let source = self.target()?;
                let n = self.count()?;
                let results = (0..n).map(|_| self.spec()).collect::<Result<Vec<_>, _>>()?;
                StructuralOp::Split { source, results }
            }
            3 => StructuralOp::Freeze {
                target: self.target()?,
            },
            4 => {
                let target = self.target()?;
                let result = match self.u8()? {
                    0 => None,
                    1 => Some(self.spec()?),
                    _ => return malformed("unknown option tag"),
                };
                StructuralOp::Reactivate { target, result }
            }
            5 => StructuralOp::Prune {
                targets: self.targets()?,
            },
            6 => StructuralOp::Retire {
                target: self.target()?,
                reason: self.reason()?,
            },
            _ => return malformed("unknown operation tag"),
        };
        ExpertProposal::new(project, proposer, observed_at, op, rationale)
    }
}
