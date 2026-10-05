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
//! forgetting and interference gates, the verifier and a ledger commit. That acceptance path is
//! not implemented yet. Every observed fact in a proposal (a target's `state`, `state_version`,
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
//! | `Retire` | one target and a proposable [`RetiredReason`] (`Superseded`, `GateFailed`) | none |
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
//! expert, hence `Serving`/`Active`. That is also why `Retire` can name them:
//! `LifecycleGate::transition` allows `Retired` only from `Active`. Reactivating a `Frozen`
//! expert re-enters it as `Gated`, a revalidation recorded by the ledger rather than a
//! `LifecycleGate` edge. `ptr-pods` does not depend on `ptr-lineage`, so the adapter status is exposed
//! by name.
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
//! | `Fenced` | no | no | no | no | no | yes |
//! | `Superseded` | no | no | no | no | no | no |
//! | `Retired(GateFailed)` | no | no | no | yes | no | no |
//! | `Retired(Pruned \| Superseded \| InputRevoked)` | no | no | no | no | no | no |
//!
//! `Birth` has no source. `Retire` is allowed exactly from `Serving`, `Frozen` and `Fenced`, the
//! states that map to `Active`, mirroring `LifecycleGate::transition`. A `Candidate` or `Gated`
//! expert is not retired by a proposal (a failed candidate is dropped by its gate), and a
//! `Superseded` expert is already withdrawn. `Fenced` is valid only for `Retire`. A
//! `Merge` additionally needs all its sources in the *same* state. A `Reactivate` of a `Retired`
//! target with any reason other than `GateFailed` is reported as
//! [`ExpertProposalError::SeedUnavailable`] (not as a generic state error): `Pruned` weights are
//! physically removed, `Superseded` already has a successor, and `InputRevoked` must never seed.
//!
//! # Reactivation
//!
//! Retirement is final (`Lineage::retire`; `LifecycleGate` has no transition out of `Retired`).
//! `Reactivate` of a `Frozen` target unfreezes the same generation, but the expert re-enters as
//! `Gated`: it must be revalidated before it is `Serving` again. `Reactivate` of a
//! `Retired(GateFailed)` target leaves that generation `Retired` forever; the proposal carries a
//! result spec for a NEW candidate generation of the same logical expert (same project and pod,
//! exactly the source generation plus one, same level including the Token index), seeded from the
//! retained weights, which must pass the gates again like any birth. Other retired targets are
//! not seeds: `Pruned` weights are gone, `Superseded` has a successor, `InputRevoked` is barred.
//!
//! # Retirement reasons
//!
//! `Retire` accepts only `Superseded` and `GateFailed`. `Pruned` has its own operation
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
//! 5. source states against the table above (a non-`GateFailed` retired Reactivate target is
//!    `SeedUnavailable` here);
//! 6. operation structure, per operation:
//!    * Merge: Token level, duplicates and multiple generations, one layer, one base model, one
//!      shared state (`MixedStates`);
//!    * Prune: Token level, duplicates and multiple generations, one layer, one base model;
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
//! # Proposal id
//!
//! [`ExpertProposal::proposal_id`] is SHA-256 over a length-prefixed canonical encoding of every
//! field (domain separator `ptr.pods.expert-proposal.v1`). It is deterministic, order-sensitive
//! in operand lists, and changes with any field, including `observed_at`.

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
/// Deterministic identity of a proposal, minted only by [`ExpertProposal::proposal_id`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ProposalId(Digest);

impl ProposalId {
    pub fn as_bytes(&self) -> &Digest {
        &self.0
    }
}

/// The base model an expert adapts: its digest and the base model's own generation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BaseModelRef {
    digest: BaseModelDigest,
    generation: u64,
}

impl BaseModelRef {
    /// Bind a base model digest to the base model's generation. Rejects generation zero (never
    /// a committed generation). The digest is non-zero by type.
    pub fn new(digest: BaseModelDigest, generation: u64) -> Result<Self, ExpertIdentityError> {
        if generation == 0 {
            Err(ExpertIdentityError::ZeroBaseModelGeneration)
        } else {
            Ok(Self { digest, generation })
        }
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
    /// Fenced off by authority; only `Retire` may name it.
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
            // Mirrors `LifecycleGate::transition`: retirement only from `Active`.
            Self::Retire => matches!(
                state,
                ExpertState::Serving | ExpertState::Frozen | ExpertState::Fenced
            ),
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

/// An identity field (project, pod, generation) was malformed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ExpertIdentityError {
    BlankProject,
    NonCanonicalProject,
    BlankPod,
    NonCanonicalPod,
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
            Self::DuplicateResultSlot { layer, index } => {
                write!(
                    f,
                    "two results claim token slot layer {layer} index {index}"
                )
            }
        }
    }
}

impl std::error::Error for ExpertProposalError {}

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
        match check_project_id(&project) {
            Ok(()) => {}
            Err(ExpertIdentityError::BlankProject) => {
                return Err(ExpertProposalError::BlankProject)
            }
            Err(_) => return Err(ExpertProposalError::NonCanonicalProject),
        }
        if proposer.project != project {
            return Err(ExpertProposalError::ProposerProjectMismatch {
                expected: project,
                found: proposer.project,
            });
        }
        validate_op(&op, &project)?;
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
    /// and the reactivation of a `Frozen` target (the same generation is unfrozen and re-enters
    /// as `Gated`).
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

    /// Deterministic identity: SHA-256 over the canonical encoding of every field.
    pub fn proposal_id(&self) -> ProposalId {
        let ProposalParts {
            project,
            proposer,
            observed_at,
            op,
            rationale,
        } = self.parts();
        let mut e = Encoder::default();
        e.str("ptr.pods.expert-proposal.v1");
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
        self.u64(targets.len() as u64);
        for target in targets {
            self.target(target);
        }
    }
}

fn validate_op(op: &StructuralOp, project: &ProjectId) -> Result<(), ExpertProposalError> {
    let kind = op.kind();
    let sources = op.sources();
    let results = op.results();

    // 3. operand counts.
    match op {
        StructuralOp::Merge { .. } => count_sources(kind, &sources, OperandCount::AtLeast(2))?,
        StructuralOp::Prune { .. } => count_sources(kind, &sources, OperandCount::AtLeast(1))?,
        StructuralOp::Split { .. } => {
            let expected = OperandCount::AtLeast(2);
            if !expected.accepts(results.len()) {
                return Err(ExpertProposalError::WrongResultCount {
                    op: kind,
                    expected,
                    found: results.len(),
                });
            }
        }
        StructuralOp::Birth { .. }
        | StructuralOp::Freeze { .. }
        | StructuralOp::Reactivate { .. }
        | StructuralOp::Retire { .. } => {}
    }

    // 4. projects.
    let source_projects = sources.iter().map(|t| &t.project);
    let result_projects = results.iter().map(|r| &r.project);
    if let Some(found) = source_projects
        .chain(result_projects)
        .find(|p| *p != project)
    {
        return Err(ExpertProposalError::MixedProjects {
            expected: project.clone(),
            found: found.clone(),
        });
    }

    // 5. source states. A retired reactivation target that cannot seed gets its own error.
    if let StructuralOp::Reactivate { target, .. } = op {
        if let ExpertState::Retired(reason) = target.state {
            if reason != RetiredReason::GateFailed {
                return Err(ExpertProposalError::SeedUnavailable {
                    pod: target.pod.clone(),
                    reason,
                });
            }
        }
    }
    if let Some(target) = sources.iter().find(|t| !kind.allows_state(t.state)) {
        return Err(ExpertProposalError::WrongTargetState {
            op: kind,
            pod: target.pod.clone(),
            state: target.state,
        });
    }

    // 6. operation structure.
    match op {
        StructuralOp::Merge { .. } => {
            require_token_level(kind, &sources)?;
            check_unique(&sources)?;
            check_same_layer(&sources)?;
            check_same_base(&sources)?;
            if sources.iter().any(|t| t.state != sources[0].state) {
                return Err(ExpertProposalError::MixedStates);
            }
        }
        StructuralOp::Prune { .. } => {
            require_token_level(kind, &sources)?;
            check_unique(&sources)?;
            check_same_layer(&sources)?;
            check_same_base(&sources)?;
        }
        StructuralOp::Reactivate { target, result } => match (target.state, result) {
            (ExpertState::Retired(_), None) => {
                return Err(ExpertProposalError::MissingResult { op: kind });
            }
            (ExpertState::Frozen, Some(_)) => {
                return Err(ExpertProposalError::UnexpectedResult { op: kind });
            }
            _ => {}
        },
        StructuralOp::Retire { reason, .. } => match reason {
            RetiredReason::Superseded | RetiredReason::GateFailed => {}
            RetiredReason::Pruned | RetiredReason::InputRevoked => {
                return Err(ExpertProposalError::ReasonNotProposable { reason: *reason });
            }
        },
        StructuralOp::Birth { .. } | StructuralOp::Split { .. } | StructuralOp::Freeze { .. } => {}
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
    result: &ExpertPodSpec,
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
