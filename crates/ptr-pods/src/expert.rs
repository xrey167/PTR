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
//! not implemented yet.
//!
//! # No override surface (deliberate type-level guarantee)
//!
//! An [`ExpertProposal`] has no threshold, gate, verifier, score-override or policy-parameter
//! field, and no field that could hold one. It carries exactly the operation, the experts it
//! targets, the level of a newborn expert and a digest of the proposer's rationale. A proposer
//! therefore cannot loosen a gate or switch off a verifier by what it submits. This protects
//! invariant #4 (uncommitted state is never authoritative: a proposal is not a change) and
//! invariant #11 (a learned positive score cannot override a deterministic contradiction) in
//! `docs/INVARIANTS.md`. [`ExpertProposal::parts`] exposes the complete field list as
//! [`ProposalParts`], so a reviewer (and the test in `tests/expert.rs`) can see that adding a
//! field is a visible, breaking change.

use ptr_types::{Digest, Generation, PodId};
use std::collections::BTreeSet;
use std::fmt;

const ZERO_DIGEST: Digest = [0; 32];

/// Where an expert lives.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ExpertLevel {
    /// An in-process expert inside one MoE layer.
    Token { layer: u16 },
    /// A whole model or adapter set reachable as a networked pod.
    Segment,
}

/// Lifecycle of an expert as seen by a proposal.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ExpertState {
    Candidate,
    Serving,
    Frozen,
    Retired,
}

/// A structural operation a proposal asks for.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StructuralOp {
    Birth,
    Merge,
    Split,
    Freeze,
    Reactivate,
    Prune,
    Retire,
}

impl fmt::Display for StructuralOp {
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

/// Identity and binding of one expert pod.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertPodSpec {
    pod: PodId,
    manifest_digest: Digest,
    level: ExpertLevel,
    base_model: Digest,
    generation: Generation,
}

impl ExpertPodSpec {
    /// Build a spec. Rejects a blank pod id, an all-zero manifest or base-model digest and
    /// generation zero (the default value, never a committed generation).
    pub fn new(
        pod: PodId,
        manifest_digest: Digest,
        level: ExpertLevel,
        base_model: Digest,
        generation: Generation,
    ) -> Result<Self, ExpertSpecError> {
        if pod.0.trim().is_empty() {
            return Err(ExpertSpecError::BlankPod);
        }
        if manifest_digest == ZERO_DIGEST {
            return Err(ExpertSpecError::ZeroManifestDigest);
        }
        if base_model == ZERO_DIGEST {
            return Err(ExpertSpecError::ZeroBaseModel);
        }
        if generation == Generation::default() {
            return Err(ExpertSpecError::ZeroGeneration);
        }
        Ok(Self {
            pod,
            manifest_digest,
            level,
            base_model,
            generation,
        })
    }

    pub fn pod(&self) -> &PodId {
        &self.pod
    }

    pub fn manifest_digest(&self) -> &Digest {
        &self.manifest_digest
    }

    pub fn level(&self) -> ExpertLevel {
        self.level
    }

    pub fn base_model(&self) -> &Digest {
        &self.base_model
    }

    pub fn generation(&self) -> Generation {
        self.generation
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExpertSpecError {
    BlankPod,
    ZeroManifestDigest,
    ZeroBaseModel,
    ZeroGeneration,
}

impl fmt::Display for ExpertSpecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BlankPod => f.write_str("expert pod id is blank"),
            Self::ZeroManifestDigest => f.write_str("expert manifest digest is all zero"),
            Self::ZeroBaseModel => f.write_str("expert base model digest is all zero"),
            Self::ZeroGeneration => f.write_str("expert generation is zero"),
        }
    }
}

impl std::error::Error for ExpertSpecError {}

/// An existing expert a proposal refers to, pinned to a generation and observed state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertTarget {
    pub pod: PodId,
    pub generation: Generation,
    pub level: ExpertLevel,
    pub state: ExpertState,
}

/// Borrowed view of every field of an [`ExpertProposal`].
///
/// The field list is the complete content of a proposal. It is deliberately free of thresholds,
/// gates, verifier settings and policy parameters (see the module documentation).
#[derive(Clone, Copy, Debug)]
pub struct ProposalParts<'a> {
    pub op: StructuralOp,
    pub targets: &'a [ExpertTarget],
    pub birth_level: Option<ExpertLevel>,
    pub rationale_digest: &'a Digest,
}

/// A validated, inert request for a structural change.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertProposal {
    op: StructuralOp,
    targets: Vec<ExpertTarget>,
    birth_level: Option<ExpertLevel>,
    rationale_digest: Digest,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExpertProposalError {
    /// The operation needs a different number of targets.
    WrongTargetCount {
        op: StructuralOp,
        expected: &'static str,
        found: usize,
    },
    /// Birth must name the level of the new expert.
    BirthWithoutLevel,
    /// Only Birth may carry a birth level.
    UnexpectedBirthLevel { op: StructuralOp },
    /// A target's state does not allow the operation.
    WrongTargetState {
        op: StructuralOp,
        pod: PodId,
        state: ExpertState,
    },
    /// Merge lists the same (pod, generation) twice.
    DuplicateTarget { pod: PodId, generation: Generation },
    /// Merge mixes levels, or Token experts of different layers.
    MixedLevels,
    /// A target has a blank pod id or generation zero.
    InvalidTarget { pod: PodId },
    /// The rationale digest is all zero.
    ZeroRationaleDigest,
}

impl fmt::Display for ExpertProposalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongTargetCount {
                op,
                expected,
                found,
            } => write!(f, "{op} needs {expected} target(s), found {found}"),
            Self::BirthWithoutLevel => f.write_str("birth must name the level of the new expert"),
            Self::UnexpectedBirthLevel { op } => {
                write!(f, "{op} must not carry a birth level; only birth may")
            }
            Self::WrongTargetState { op, pod, state } => {
                write!(f, "{op} cannot target expert {pod} in state {state:?}")
            }
            Self::DuplicateTarget { pod, generation } => {
                write!(
                    f,
                    "merge lists expert {pod} generation {} twice",
                    generation.0
                )
            }
            Self::MixedLevels => {
                f.write_str("merge targets must share one level (and one layer for token level)")
            }
            Self::InvalidTarget { pod } => {
                write!(f, "target {pod} has a blank pod id or generation zero")
            }
            Self::ZeroRationaleDigest => f.write_str("rationale digest is all zero"),
        }
    }
}

impl std::error::Error for ExpertProposalError {}

impl ExpertProposal {
    /// Validate and build a proposal. See the module documentation for what a proposal is not.
    pub fn new(
        op: StructuralOp,
        targets: Vec<ExpertTarget>,
        birth_level: Option<ExpertLevel>,
        rationale_digest: Digest,
    ) -> Result<Self, ExpertProposalError> {
        if rationale_digest == ZERO_DIGEST {
            return Err(ExpertProposalError::ZeroRationaleDigest);
        }
        for target in &targets {
            if target.pod.0.trim().is_empty() || target.generation == Generation::default() {
                return Err(ExpertProposalError::InvalidTarget {
                    pod: target.pod.clone(),
                });
            }
        }
        match (op, birth_level) {
            (StructuralOp::Birth, None) => return Err(ExpertProposalError::BirthWithoutLevel),
            (StructuralOp::Birth, Some(_)) => {}
            (_, Some(_)) => return Err(ExpertProposalError::UnexpectedBirthLevel { op }),
            (_, None) => {}
        }
        const LIVE: &[ExpertState] = &[ExpertState::Serving, ExpertState::Frozen];
        match op {
            StructuralOp::Birth => check_count(op, &targets, "no", |n| n == 0)?,
            StructuralOp::Merge => {
                check_count(op, &targets, "at least two", |n| n >= 2)?;
                check_states(op, &targets, LIVE)?;
                check_merge(&targets)?;
            }
            StructuralOp::Split | StructuralOp::Prune => {
                check_count(op, &targets, "exactly one", |n| n == 1)?;
                check_states(op, &targets, LIVE)?;
            }
            // A candidate that fails its gate is retired, as `Lineage::retire` allows; only a
            // retired expert cannot be retired again.
            StructuralOp::Retire => {
                check_count(op, &targets, "exactly one", |n| n == 1)?;
                check_states(
                    op,
                    &targets,
                    &[
                        ExpertState::Candidate,
                        ExpertState::Serving,
                        ExpertState::Frozen,
                    ],
                )?;
            }
            StructuralOp::Freeze => {
                check_count(op, &targets, "exactly one", |n| n == 1)?;
                check_states(op, &targets, &[ExpertState::Serving])?;
            }
            StructuralOp::Reactivate => {
                check_count(op, &targets, "exactly one", |n| n == 1)?;
                check_states(op, &targets, &[ExpertState::Retired])?;
            }
        }
        Ok(Self {
            op,
            targets,
            birth_level,
            rationale_digest,
        })
    }

    pub fn op(&self) -> StructuralOp {
        self.op
    }

    pub fn targets(&self) -> &[ExpertTarget] {
        &self.targets
    }

    pub fn birth_level(&self) -> Option<ExpertLevel> {
        self.birth_level
    }

    pub fn rationale_digest(&self) -> &Digest {
        &self.rationale_digest
    }

    /// Every field of the proposal; adding a field to the proposal means changing this struct.
    pub fn parts(&self) -> ProposalParts<'_> {
        ProposalParts {
            op: self.op,
            targets: &self.targets,
            birth_level: self.birth_level,
            rationale_digest: &self.rationale_digest,
        }
    }
}

fn check_count(
    op: StructuralOp,
    targets: &[ExpertTarget],
    expected: &'static str,
    ok: impl Fn(usize) -> bool,
) -> Result<(), ExpertProposalError> {
    if ok(targets.len()) {
        Ok(())
    } else {
        Err(ExpertProposalError::WrongTargetCount {
            op,
            expected,
            found: targets.len(),
        })
    }
}

fn check_states(
    op: StructuralOp,
    targets: &[ExpertTarget],
    allowed: &[ExpertState],
) -> Result<(), ExpertProposalError> {
    match targets.iter().find(|t| !allowed.contains(&t.state)) {
        Some(target) => Err(ExpertProposalError::WrongTargetState {
            op,
            pod: target.pod.clone(),
            state: target.state,
        }),
        None => Ok(()),
    }
}

fn check_merge(targets: &[ExpertTarget]) -> Result<(), ExpertProposalError> {
    let mut seen = BTreeSet::new();
    for target in targets {
        if !seen.insert((&target.pod, target.generation)) {
            return Err(ExpertProposalError::DuplicateTarget {
                pod: target.pod.clone(),
                generation: target.generation,
            });
        }
    }
    let first = targets[0].level;
    if targets.iter().any(|t| t.level != first) {
        return Err(ExpertProposalError::MixedLevels);
    }
    Ok(())
}
