//! The L3 structural-proposal policy: verifier statistics in, operation requests out.
//!
//! [`StructurePolicy::propose`] is a pure function of a bounded [`StructureObservation`] and a
//! validated [`StructurePolicyConfig`]. It has no authority, by construction:
//!
//! * it takes no ledger, no [`ExpertAdmissionGrant`](super::ExpertAdmissionGrant), no gate
//!   report, no `ForgettingGate` and no verifier configuration. A unit test checks the module's
//!   `use` list against an allow-list (only `ptr_pods`, `ptr_router`, `ptr_types`, `sha2`, `std`
//!   and two items of the parent module, no glob import) and pins the signature of `propose`;
//! * every output is an [`ExpertOperationRequest`], built through the `ptr-pods` constructor so
//!   its validation runs: a request that would name a fenced expert, a state the operation does
//!   not allow ([`OpKind::allows_state`]) or a mixed project or base model never leaves here;
//! * a request carries only what [`ExpertProposal::parts`] lists, with the result experts as
//!   [`ResultIntent`]s. The thresholds that triggered it reach it only through the rationale
//!   digest, which commits to the policy configuration and to the whole observation.
//!
//! # Pending and ready proposals
//!
//! Birth, Merge and Reactivate create an expert, and only training produces its manifest digest.
//! The policy therefore never builds an [`ExpertProposal`] for them: [`StructureProposal`] holds
//! the pre-training [`ExpertOperationRequest`] (its [`OperationId`](ptr_pods::OperationId) is
//! final), and the host completes it with [`StructureProposal::complete`] once training has
//! produced the manifest digest. Only Freeze and Prune, which create nothing, are available
//! as a proposal directly ([`StructureProposal::ready`]). A request is not accepted by
//! `admit_expert_proposal`, which takes an [`ExpertProposal`] only.
//!
//! # Operators
//!
//! | Operator   | Triggers when                                                                   |
//! |------------|----------------------------------------------------------------------------------|
//! | Birth      | a finding code was unroutable, or routed and never cleared, `>= min_unroutable` times in each of the last `birth_windows` windows |
//! | Reactivate | the same trigger for a code whose former expert is `Retired(GateFailed)`          |
//! | Prune      | a serving, mature Token expert with usage `< prune_usage_floor` in each of the last `prune_windows` windows and an AIMER score, and every code it is relied on for has another expert with demonstrated clears |
//! | Merge      | two Token experts with equal code sets, clear rates within a margin and above a floor, and subspace overlap above chance by a margin |
//! | Freeze     | a serving expert whose usage and clear rate are stable, and the clear rate above a floor, over `freeze_windows` windows |
//!
//! Split and Retire are never proposed. A backup that protects a code from a Prune must be
//! registered for it **and** have cleared it at least `backup_min_cleared` times in the last
//! `backup_clear_windows` windows (a `Frozen` expert counts, since it is routable); a bare
//! registration protects nothing.
//!
//! The Merge margin is a placeholder preregistered-margin rule, not the calibrated null
//! threshold doc 37 section 4 asks for (a high quantile of the overlaps of independently trained
//! adapters of the same shape): `chance_overlap` is a mean, so an excess over it is weak
//! evidence. M013's preregistration must fix the margin or replace it with that threshold.
//! Comparisons carry a `1e-12` tolerance so a threshold that is not exactly representable
//! (0.1 and 0.2, say) still triggers at its boundary.
//!
//! Candidates are generated in a fixed priority order (Birth, Reactivate, Prune, Merge, Freeze),
//! each in a fixed key order (Births by evidence, then code), then cut by the sub-cap on Births,
//! the per-window cap and the per-expert cap, so the same observation always yields the same
//! list.
//!
//! # Provenance
//!
//! Every number in an observation is a claim of whoever assembled it. The host must build it from
//! router and ledger state, never from the proposer's own data. The rationale digest commits to
//! [`StructureObservation::digest`], so a reviewer can recompute why a request was made.
//! The generator is stateless: the host passes the codes of requests still outstanding in
//! [`StructureObservation::outstanding`] so a Birth is not re-requested every window, and dedupes
//! by operation id.
//!
//! # Bounds
//!
//! The observation is checked against [`ObservationLimits`] before anything is computed (counts
//! above [`MAX_COUNT`] are refused), so the work and the memory of one call are bounded by the
//! limits and the output by `max_proposals_per_window`. All arithmetic on counts saturates.

use super::{occupies_slot, ExpertRegistryProjection};
use ptr_pods::{
    BaseModelRef, ExpertIdentityError, ExpertLevel, ExpertOperationRequest, ExpertPodSpec,
    ExpertProposal, ExpertProposalError, ExpertState, ExpertTarget, ExpertTargetInit,
    ManifestDigest, OpKind, ProposerRef, RationaleDigest, RequestedOp, ResultIntent, RetiredReason,
    StructuralOp,
};
use ptr_router::FindingCode;
use ptr_types::{CommitIndex, Generation, PodId, ProjectId};
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

const RATIONALE_DOMAIN: &[u8] = b"ptr.runtime.expert-structure.rationale.v2";
const CONFIG_DOMAIN: &[u8] = b"ptr.runtime.expert-structure.config.v2";
const OBSERVATION_DOMAIN: &[u8] = b"ptr.runtime.expert-structure.observation.v1";

/// Tolerance of every threshold comparison on a computed fraction or variance.
const TOLERANCE: f64 = 1e-12;

/// The largest count the policy accepts in an observation (2^40): large enough for any real
/// window, small enough that sums over the limits stay exact in `u64` and `f64`.
pub const MAX_COUNT: u64 = 1 << 40;

// ---------------------------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------------------------

/// Which operators the policy may propose. Turning one off is an ablation lever; it never
/// enables an operator the policy does not implement (Split, Retire).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EnabledOps {
    pub birth: bool,
    pub merge: bool,
    pub prune: bool,
    pub freeze: bool,
    pub reactivate: bool,
}

impl EnabledOps {
    /// Every implemented operator.
    pub const ALL: Self = Self {
        birth: true,
        merge: true,
        prune: true,
        freeze: true,
        reactivate: true,
    };

    fn any(self) -> bool {
        self.birth || self.merge || self.prune || self.freeze || self.reactivate
    }
}

/// Upper bounds on what one observation may contain. An observation over a bound is refused, not
/// truncated, so a caller cannot get a policy decision on data the policy did not look at.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObservationLimits {
    pub max_windows: usize,
    pub max_codes_per_window: usize,
    pub max_pods_per_code: usize,
    pub max_experts: usize,
    pub max_codes_per_expert: usize,
    pub max_overlap_pairs: usize,
}

impl Default for ObservationLimits {
    fn default() -> Self {
        Self {
            max_windows: 16,
            max_codes_per_window: 256,
            max_pods_per_code: 64,
            max_experts: 256,
            max_codes_per_expert: 64,
            max_overlap_pairs: 4096,
        }
    }
}

/// Every threshold of the policy. This is policy configuration, not gate configuration: nothing
/// here reaches a gate, a verifier or an admission grant, and none of it appears in a request
/// except through the rationale digest.
#[derive(Clone, Debug, PartialEq)]
pub struct StructurePolicyConfig {
    pub ops: EnabledOps,
    /// Birth and Reactivate: unroutable (or routed and never cleared) reports of one code in a
    /// window at which that window counts. Triggers at exactly this count.
    pub min_unroutable: u64,
    /// Birth and Reactivate: the number of most recent windows that must each reach
    /// `min_unroutable`, so one burst does not trigger.
    pub birth_windows: usize,
    /// At most this many Births per call, taken by evidence, so junk codes cannot occupy the
    /// whole window cap. Keep it below `max_proposals_per_window`.
    pub max_births_per_window: usize,
    /// Merge: largest clear-rate gap, per shared code, still counted as the same behavior
    /// (inclusive), in `[0, 1]`.
    pub merge_clear_rate_margin: f64,
    /// Merge: how far the subspace overlap must exceed chance (inclusive), in `(0, 1]`. A zero
    /// margin would merge at chance, which carries no evidence of interference. An uncalibrated
    /// placeholder (see the module documentation).
    pub merge_overlap_margin: f64,
    /// Merge: routed reports of a code an expert needs in the latest window for its clear rate
    /// to count.
    pub merge_min_routed: u64,
    /// Merge: both experts' clear rate on every shared code must reach this, in `[0, 1]`
    /// (inclusive), so two experts that repair nothing are not merged.
    pub merge_min_clear_rate: f64,
    /// Prune: an expert whose usage is strictly below this in each of the last `prune_windows`
    /// windows is a candidate.
    pub prune_usage_floor: u64,
    /// Prune: the number of most recent windows the low usage must hold in.
    pub prune_windows: usize,
    /// Prune: an expert must be at least this many commits old (`observed_at` minus its
    /// `last_changed_at`). An expert with no known age is too young while this is above zero.
    pub prune_min_age_commits: u64,
    /// Prune: clears of a code a backup must show to protect it from a Prune of its only other
    /// clearer, counted over the last `backup_clear_windows` windows.
    pub backup_min_cleared: u64,
    /// Prune: the number of most recent windows the backup's clears are counted over.
    pub backup_clear_windows: usize,
    /// Freeze: number of most recent windows that must all be stable.
    pub freeze_windows: usize,
    /// Freeze: largest population variance of usage over those windows (inclusive).
    pub freeze_max_usage_variance: f64,
    /// Freeze: largest population variance of the clear rate over those windows (inclusive).
    pub freeze_max_clear_rate_variance: f64,
    /// Freeze: clear rate every one of those windows must reach, in `[0, 1]` (inclusive), so a
    /// stable expert that repairs nothing is not frozen as converged.
    pub freeze_min_clear_rate: f64,
    /// Freeze: usage every one of those windows must reach. Not below `prune_usage_floor`, so an
    /// expert is never both a freeze and a prune candidate.
    pub freeze_min_usage: u64,
    pub max_proposals_per_window: usize,
    pub max_proposals_per_expert: usize,
    pub limits: ObservationLimits,
}

impl Default for StructurePolicyConfig {
    fn default() -> Self {
        Self {
            ops: EnabledOps::ALL,
            min_unroutable: 8,
            birth_windows: 2,
            max_births_per_window: 4,
            merge_clear_rate_margin: 0.1,
            merge_overlap_margin: 0.2,
            merge_min_routed: 4,
            merge_min_clear_rate: 0.5,
            prune_usage_floor: 2,
            prune_windows: 2,
            prune_min_age_commits: 16,
            backup_min_cleared: 1,
            backup_clear_windows: 2,
            freeze_windows: 3,
            freeze_max_usage_variance: 4.0,
            freeze_max_clear_rate_variance: 0.01,
            freeze_min_clear_rate: 0.5,
            freeze_min_usage: 4,
            max_proposals_per_window: 8,
            max_proposals_per_expert: 1,
            limits: ObservationLimits::default(),
        }
    }
}

/// Why a [`StructurePolicyConfig`] is refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum StructureConfigError {
    /// No operator is enabled.
    NoOperatorEnabled,
    /// A count that must be at least one is zero.
    ZeroField { field: &'static str },
    /// A fraction is NaN, infinite or outside its range.
    BadFraction { field: &'static str },
    /// A variance bound is NaN, infinite or negative.
    BadVariance { field: &'static str },
    /// `freeze_windows` is below two: one window has no variance.
    FreezeNeedsTwoWindows,
    /// `limits.max_windows` is below `freeze_windows`, so Freeze could never trigger.
    HistoryShorterThanFreezeWindows,
    /// A window count is above `limits.max_windows`, so the operator could never trigger.
    WindowsExceedHistory { field: &'static str },
    /// `freeze_min_usage` is below `prune_usage_floor`.
    FreezeFloorBelowPruneFloor,
}

impl fmt::Display for StructureConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoOperatorEnabled => f.write_str("no structural operator is enabled"),
            Self::ZeroField { field } => write!(f, "{field} must be at least 1"),
            Self::BadFraction { field } => write!(f, "{field} is not a finite value in range"),
            Self::BadVariance { field } => {
                write!(f, "{field} is not a finite non-negative value")
            }
            Self::FreezeNeedsTwoWindows => f.write_str("freeze_windows must be at least 2"),
            Self::HistoryShorterThanFreezeWindows => {
                f.write_str("limits.max_windows is below freeze_windows")
            }
            Self::WindowsExceedHistory { field } => {
                write!(f, "{field} is above limits.max_windows")
            }
            Self::FreezeFloorBelowPruneFloor => {
                f.write_str("freeze_min_usage is below prune_usage_floor")
            }
        }
    }
}

impl std::error::Error for StructureConfigError {}

fn fraction_in(value: f64, low_open: bool) -> bool {
    value.is_finite() && value <= 1.0 && if low_open { value > 0.0 } else { value >= 0.0 }
}

impl StructurePolicyConfig {
    /// Check the configuration.
    ///
    /// # Errors
    /// The first violated rule, as a [`StructureConfigError`].
    pub fn validate(&self) -> Result<(), StructureConfigError> {
        if !self.ops.any() {
            return Err(StructureConfigError::NoOperatorEnabled);
        }
        let l = &self.limits;
        let counts: [(&'static str, u64); 15] = [
            ("min_unroutable", self.min_unroutable),
            ("birth_windows", self.birth_windows as u64),
            ("max_births_per_window", self.max_births_per_window as u64),
            ("merge_min_routed", self.merge_min_routed),
            ("prune_usage_floor", self.prune_usage_floor),
            ("prune_windows", self.prune_windows as u64),
            ("backup_min_cleared", self.backup_min_cleared),
            ("backup_clear_windows", self.backup_clear_windows as u64),
            (
                "max_proposals_per_window",
                self.max_proposals_per_window as u64,
            ),
            (
                "max_proposals_per_expert",
                self.max_proposals_per_expert as u64,
            ),
            ("limits.max_windows", l.max_windows as u64),
            ("limits.max_codes_per_window", l.max_codes_per_window as u64),
            ("limits.max_pods_per_code", l.max_pods_per_code as u64),
            ("limits.max_experts", l.max_experts as u64),
            ("limits.max_overlap_pairs", l.max_overlap_pairs as u64),
        ];
        if let Some((field, _)) = counts.iter().find(|(_, value)| *value == 0) {
            return Err(StructureConfigError::ZeroField { field });
        }
        if l.max_codes_per_expert == 0 {
            return Err(StructureConfigError::ZeroField {
                field: "limits.max_codes_per_expert",
            });
        }
        for (field, value) in [
            ("merge_clear_rate_margin", self.merge_clear_rate_margin),
            ("merge_min_clear_rate", self.merge_min_clear_rate),
            ("freeze_min_clear_rate", self.freeze_min_clear_rate),
        ] {
            if !fraction_in(value, false) {
                return Err(StructureConfigError::BadFraction { field });
            }
        }
        if !fraction_in(self.merge_overlap_margin, true) {
            return Err(StructureConfigError::BadFraction {
                field: "merge_overlap_margin",
            });
        }
        for (field, value) in [
            ("freeze_max_usage_variance", self.freeze_max_usage_variance),
            (
                "freeze_max_clear_rate_variance",
                self.freeze_max_clear_rate_variance,
            ),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(StructureConfigError::BadVariance { field });
            }
        }
        if self.freeze_windows < 2 {
            return Err(StructureConfigError::FreezeNeedsTwoWindows);
        }
        if l.max_windows < self.freeze_windows {
            return Err(StructureConfigError::HistoryShorterThanFreezeWindows);
        }
        for (field, windows) in [
            ("birth_windows", self.birth_windows),
            ("prune_windows", self.prune_windows),
            ("backup_clear_windows", self.backup_clear_windows),
        ] {
            if windows > l.max_windows {
                return Err(StructureConfigError::WindowsExceedHistory { field });
            }
        }
        if self.freeze_min_usage < self.prune_usage_floor {
            return Err(StructureConfigError::FreezeFloorBelowPruneFloor);
        }
        Ok(())
    }

    /// Canonical digest of the configuration, folded into every rationale digest so a request is
    /// tied to the policy version that produced it.
    pub fn digest(&self) -> [u8; 32] {
        let mut enc = Encoder::new(CONFIG_DOMAIN);
        for flag in [
            self.ops.birth,
            self.ops.merge,
            self.ops.prune,
            self.ops.freeze,
            self.ops.reactivate,
        ] {
            enc.u64(u64::from(flag));
        }
        let l = &self.limits;
        enc.u64(self.min_unroutable)
            .u64(self.birth_windows as u64)
            .u64(self.max_births_per_window as u64)
            .f64(self.merge_clear_rate_margin)
            .f64(self.merge_overlap_margin)
            .u64(self.merge_min_routed)
            .f64(self.merge_min_clear_rate)
            .u64(self.prune_usage_floor)
            .u64(self.prune_windows as u64)
            .u64(self.prune_min_age_commits)
            .u64(self.backup_min_cleared)
            .u64(self.backup_clear_windows as u64)
            .u64(self.freeze_windows as u64)
            .f64(self.freeze_max_usage_variance)
            .f64(self.freeze_max_clear_rate_variance)
            .f64(self.freeze_min_clear_rate)
            .u64(self.freeze_min_usage)
            .u64(self.max_proposals_per_window as u64)
            .u64(self.max_proposals_per_expert as u64)
            .u64(l.max_windows as u64)
            .u64(l.max_codes_per_window as u64)
            .u64(l.max_pods_per_code as u64)
            .u64(l.max_experts as u64)
            .u64(l.max_codes_per_expert as u64)
            .u64(l.max_overlap_pairs as u64);
        enc.finish()
    }
}

// ---------------------------------------------------------------------------------------------
// Observation
// ---------------------------------------------------------------------------------------------

/// What one repair Pod did with one finding code in one window.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PodCodeStats {
    /// Reports of the code routed to the Pod.
    pub routed: u64,
    /// Routed reports a later verification passed (repair, then Pass).
    pub cleared: u64,
}

/// One finding code in one window.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CodeWindowStats {
    /// Reports of the code with no eligible repair Pod.
    pub unroutable: u64,
    /// Per repair Pod.
    pub pods: BTreeMap<PodId, PodCodeStats>,
}

/// One observation window: the finding statistics and the expert usage inside it.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct WindowObservation {
    pub codes: BTreeMap<FindingCode, CodeWindowStats>,
    /// Invocations per expert.
    pub usage: BTreeMap<PodId, u64>,
    /// Unroutable reports of codes the router no longer tracked
    /// (`FindingRouter::unroutable_overflow`). Reported back, never acted on: there is no code to
    /// propose a Birth for.
    pub unroutable_untracked: u64,
}

/// The per-code growth between two cumulative `FindingRouter::unroutable_counts` readings.
/// A code absent from `previous` counts from zero; a counter that went down (a restarted
/// router) counts as zero rather than wrapping.
pub fn unroutable_delta(
    previous: &BTreeMap<FindingCode, u64>,
    current: &BTreeMap<FindingCode, u64>,
) -> BTreeMap<FindingCode, u64> {
    current
        .iter()
        .filter_map(|(code, now)| {
            let before = previous.get(code).copied().unwrap_or(0);
            let grown = now.saturating_sub(before);
            (grown > 0).then(|| (code.clone(), grown))
        })
        .collect()
}

/// Pairwise subspace overlap of two experts and the overlap two random subspaces of the same
/// ranks would have, both from `ptr-lineage` (`subspace_overlap`, `chance_overlap`), supplied by
/// the caller.
#[derive(Clone, Debug, PartialEq)]
pub struct SubspaceOverlap {
    pub first: PodId,
    pub second: PodId,
    pub overlap: f64,
    pub chance: f64,
}

/// An expert of the registry view and the finding codes it is registered to repair.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegisteredExpert {
    target: ExpertTarget,
    codes: BTreeSet<FindingCode>,
    last_changed_at: Option<CommitIndex>,
}

impl RegisteredExpert {
    pub fn new(target: ExpertTarget, codes: BTreeSet<FindingCode>) -> Self {
        Self {
            target,
            codes,
            last_changed_at: None,
        }
    }

    /// The commit index of the expert's last state change (its birth, reactivation or freeze).
    /// The registry projection does not record it, so the host supplies it; Prune treats an
    /// expert without one as too young while `prune_min_age_commits` is above zero.
    #[must_use]
    pub fn with_last_changed_at(mut self, at: CommitIndex) -> Self {
        self.last_changed_at = Some(at);
        self
    }

    /// The view of `pod` as the registry projection has it now, or `None` when it is not there
    /// or its entry is not a valid target. The projection does not know finding codes; the
    /// caller supplies them.
    pub fn from_projection(
        registry: &ExpertRegistryProjection,
        project: &ProjectId,
        pod: &PodId,
        codes: BTreeSet<FindingCode>,
    ) -> Option<Self> {
        let entry = registry.get(project, pod)?;
        let target = ExpertTarget::new(ExpertTargetInit {
            project: project.clone(),
            pod: pod.clone(),
            generation: entry.generation,
            level: entry.level,
            base_model: entry.base_model.clone(),
            manifest_digest: entry.manifest_digest,
            state: entry.state,
            state_version: entry.state_version,
        })
        .ok()?;
        Some(Self::new(target, codes))
    }

    pub fn target(&self) -> &ExpertTarget {
        &self.target
    }

    pub fn codes(&self) -> &BTreeSet<FindingCode> {
        &self.codes
    }

    pub fn last_changed_at(&self) -> Option<CommitIndex> {
        self.last_changed_at
    }
}

/// Where a Birth places its expert.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BirthSite {
    /// A networked Segment pod.
    Segment,
    /// A Token expert in `layer`, in the lowest of the first `slots` indices no registry entry
    /// that still holds its slot occupies.
    Token { layer: u16, slots: u16 },
}

/// What a Birth needs that no statistic can tell: the base model and the site.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BirthContext {
    pub base_model: BaseModelRef,
    pub site: BirthSite,
}

/// One bounded window of verifier statistics and the registry view they were taken against.
#[derive(Clone, Debug)]
pub struct StructureObservation {
    pub project: ProjectId,
    pub proposer: ProposerRef,
    /// The commit index the registry view was read at; becomes `observed_at` of every request.
    pub observed_at: CommitIndex,
    /// Oldest first; the last is the current window.
    pub windows: Vec<WindowObservation>,
    pub experts: Vec<RegisteredExpert>,
    /// AIMER score per expert (`ptr_lineage::aimer_score`); higher prunes first.
    pub aimer: BTreeMap<PodId, f64>,
    pub overlaps: Vec<SubspaceOverlap>,
    /// Without it no Birth is proposed.
    pub birth: Option<BirthContext>,
    /// Codes with a Birth or Reactivate request still outstanding (proposed, not yet decided).
    /// No new one is proposed for them. The host fills it from its pending requests.
    pub outstanding: BTreeSet<FindingCode>,
}

/// Why an observation is refused.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum StructureObservationError {
    TooManyWindows {
        found: usize,
        max: usize,
    },
    TooManyCodes {
        found: usize,
        max: usize,
    },
    TooManyPodsForCode {
        code: FindingCode,
        found: usize,
        max: usize,
    },
    TooManyExperts {
        found: usize,
        max: usize,
    },
    TooManyCodesForExpert {
        pod: PodId,
        found: usize,
        max: usize,
    },
    TooManyOverlaps {
        found: usize,
        max: usize,
    },
    /// A count is above [`MAX_COUNT`].
    CountTooLarge {
        field: &'static str,
    },
    /// A Pod cleared more reports of a code than were routed to it.
    ClearedExceedsRouted {
        code: FindingCode,
        pod: PodId,
    },
    /// An expert belongs to another project than the observation.
    ExpertProjectMismatch {
        pod: PodId,
    },
    DuplicateExpert {
        pod: PodId,
    },
    /// An AIMER score or an overlap value is NaN or infinite, or an overlap is outside `[0, 1]`.
    NonFiniteSignal {
        field: &'static str,
    },
    /// An overlap names one Pod twice.
    SelfOverlap {
        pod: PodId,
    },
    /// Two overlap entries name the same unordered pair.
    DuplicateOverlap {
        first: PodId,
        second: PodId,
    },
    /// An AIMER score or an overlap names a Pod that is not in the expert view.
    SignalForUnknownExpert {
        pod: PodId,
    },
    /// The proposer belongs to another project than the observation.
    ProposerProjectMismatch,
}

impl fmt::Display for StructureObservationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyWindows { found, max } => write!(f, "{found} windows, limit {max}"),
            Self::TooManyCodes { found, max } => {
                write!(f, "{found} codes in a window, limit {max}")
            }
            Self::TooManyPodsForCode { code, found, max } => {
                write!(f, "{found} pods for code {code}, limit {max}")
            }
            Self::TooManyExperts { found, max } => write!(f, "{found} experts, limit {max}"),
            Self::TooManyCodesForExpert { pod, found, max } => {
                write!(f, "{found} codes for expert {pod}, limit {max}")
            }
            Self::TooManyOverlaps { found, max } => write!(f, "{found} overlaps, limit {max}"),
            Self::CountTooLarge { field } => write!(f, "{field} is above {MAX_COUNT}"),
            Self::ClearedExceedsRouted { code, pod } => {
                write!(
                    f,
                    "pod {pod} cleared more of code {code} than was routed to it"
                )
            }
            Self::ExpertProjectMismatch { pod } => {
                write!(f, "expert {pod} belongs to another project")
            }
            Self::DuplicateExpert { pod } => write!(f, "expert {pod} listed twice"),
            Self::NonFiniteSignal { field } => write!(f, "{field} is not a finite value in range"),
            Self::SelfOverlap { pod } => write!(f, "overlap of {pod} with itself"),
            Self::DuplicateOverlap { first, second } => {
                write!(f, "overlap of {first} and {second} listed twice")
            }
            Self::SignalForUnknownExpert { pod } => {
                write!(f, "a signal names {pod}, which is not in the expert view")
            }
            Self::ProposerProjectMismatch => {
                f.write_str("proposer belongs to another project than the observation")
            }
        }
    }
}

impl std::error::Error for StructureObservationError {}

impl StructureObservation {
    fn validate(&self, limits: &ObservationLimits) -> Result<(), StructureObservationError> {
        use StructureObservationError as E;
        if self.proposer.project() != &self.project {
            return Err(E::ProposerProjectMismatch);
        }
        if self.windows.len() > limits.max_windows {
            return Err(E::TooManyWindows {
                found: self.windows.len(),
                max: limits.max_windows,
            });
        }
        for window in &self.windows {
            if window.codes.len() > limits.max_codes_per_window {
                return Err(E::TooManyCodes {
                    found: window.codes.len(),
                    max: limits.max_codes_per_window,
                });
            }
            if window.usage.len() > limits.max_experts {
                return Err(E::TooManyExperts {
                    found: window.usage.len(),
                    max: limits.max_experts,
                });
            }
            let usage = window.usage.values().copied();
            let counts = [window.unroutable_untracked]
                .into_iter()
                .chain(usage)
                .any(|n| n > MAX_COUNT);
            if counts {
                return Err(E::CountTooLarge {
                    field: "usage or untracked",
                });
            }
            for (code, stats) in &window.codes {
                if stats.pods.len() > limits.max_pods_per_code {
                    return Err(E::TooManyPodsForCode {
                        code: code.clone(),
                        found: stats.pods.len(),
                        max: limits.max_pods_per_code,
                    });
                }
                let big = stats.unroutable > MAX_COUNT
                    || stats
                        .pods
                        .values()
                        .any(|s| s.routed > MAX_COUNT || s.cleared > MAX_COUNT);
                if big {
                    return Err(E::CountTooLarge {
                        field: "code stats",
                    });
                }
                if let Some((pod, _)) = stats.pods.iter().find(|(_, s)| s.cleared > s.routed) {
                    return Err(E::ClearedExceedsRouted {
                        code: code.clone(),
                        pod: pod.clone(),
                    });
                }
            }
        }
        if self.experts.len() > limits.max_experts {
            return Err(E::TooManyExperts {
                found: self.experts.len(),
                max: limits.max_experts,
            });
        }
        let mut seen = BTreeSet::new();
        for expert in &self.experts {
            if expert.target.project() != &self.project {
                return Err(E::ExpertProjectMismatch {
                    pod: expert.target.pod().clone(),
                });
            }
            if !seen.insert(expert.target.pod()) {
                return Err(E::DuplicateExpert {
                    pod: expert.target.pod().clone(),
                });
            }
            if expert.codes.len() > limits.max_codes_per_expert {
                return Err(E::TooManyCodesForExpert {
                    pod: expert.target.pod().clone(),
                    found: expert.codes.len(),
                    max: limits.max_codes_per_expert,
                });
            }
        }
        if self.aimer.len() > limits.max_experts {
            return Err(E::TooManyExperts {
                found: self.aimer.len(),
                max: limits.max_experts,
            });
        }
        if self.aimer.values().any(|score| !score.is_finite()) {
            return Err(E::NonFiniteSignal { field: "aimer" });
        }
        if self.overlaps.len() > limits.max_overlap_pairs {
            return Err(E::TooManyOverlaps {
                found: self.overlaps.len(),
                max: limits.max_overlap_pairs,
            });
        }
        let mut pairs = BTreeSet::new();
        for overlap in &self.overlaps {
            if !fraction_in(overlap.overlap, false) || !fraction_in(overlap.chance, false) {
                return Err(E::NonFiniteSignal { field: "overlap" });
            }
            if overlap.first == overlap.second {
                return Err(E::SelfOverlap {
                    pod: overlap.first.clone(),
                });
            }
            let (low, high) = ordered(&overlap.first, &overlap.second);
            if !pairs.insert((low, high)) {
                return Err(E::DuplicateOverlap {
                    first: low.clone(),
                    second: high.clone(),
                });
            }
        }
        let unknown = self
            .aimer
            .keys()
            .chain(self.overlaps.iter().flat_map(|o| [&o.first, &o.second]))
            .find(|pod| !seen.contains(pod));
        if let Some(pod) = unknown {
            return Err(E::SignalForUnknownExpert { pod: pod.clone() });
        }
        Ok(())
    }

    /// Canonical digest of the whole observation (independent of the order of the expert and
    /// overlap lists), folded into every rationale digest so a reviewer can recompute why a
    /// request was made.
    pub fn digest(&self) -> [u8; 32] {
        let mut e = Encoder::new(OBSERVATION_DOMAIN);
        e.str(&self.project.0)
            .str(&self.proposer.project().0)
            .str(&self.proposer.pod().0)
            .u64(self.proposer.generation().0)
            .u64(self.observed_at.0)
            .u64(self.windows.len() as u64);
        for window in &self.windows {
            e.u64(window.codes.len() as u64);
            for (code, stats) in &window.codes {
                e.str(code.as_str())
                    .u64(stats.unroutable)
                    .u64(stats.pods.len() as u64);
                for (pod, s) in &stats.pods {
                    e.str(&pod.0).u64(s.routed).u64(s.cleared);
                }
            }
            e.u64(window.usage.len() as u64);
            for (pod, used) in &window.usage {
                e.str(&pod.0).u64(*used);
            }
            e.u64(window.unroutable_untracked);
        }
        let experts: BTreeMap<&PodId, &RegisteredExpert> =
            self.experts.iter().map(|x| (x.target.pod(), x)).collect();
        e.u64(experts.len() as u64);
        for expert in experts.values() {
            let t = &expert.target;
            e.str(&t.pod().0).u64(t.generation().0);
            match t.level() {
                ExpertLevel::Segment => e.u64(0),
                ExpertLevel::Token { layer, index } => {
                    e.u64(1).u64(u64::from(layer)).u64(u64::from(index))
                }
            };
            encode_base(&mut e, t.base_model());
            e.bytes(t.manifest_digest().as_bytes())
                .u64(state_tag(t.state()))
                .u64(t.state_version())
                .u64(expert.codes.len() as u64);
            for code in &expert.codes {
                e.str(code.as_str());
            }
            match expert.last_changed_at {
                Some(at) => e.u64(1).u64(at.0),
                None => e.u64(0),
            };
        }
        e.u64(self.aimer.len() as u64);
        for (pod, score) in &self.aimer {
            e.str(&pod.0).f64(*score);
        }
        let overlaps: BTreeMap<(&PodId, &PodId), &SubspaceOverlap> = self
            .overlaps
            .iter()
            .map(|o| (ordered(&o.first, &o.second), o))
            .collect();
        e.u64(overlaps.len() as u64);
        for ((a, b), o) in overlaps {
            e.str(&a.0).str(&b.0).f64(o.overlap).f64(o.chance);
        }
        match &self.birth {
            None => e.u64(0),
            Some(ctx) => {
                encode_base(&mut e, &ctx.base_model);
                match ctx.site {
                    BirthSite::Segment => e.u64(1),
                    BirthSite::Token { layer, slots } => {
                        e.u64(2).u64(u64::from(layer)).u64(u64::from(slots))
                    }
                }
            }
        };
        e.u64(self.outstanding.len() as u64);
        for code in &self.outstanding {
            e.str(code.as_str());
        }
        e.finish()
    }
}

fn encode_base(e: &mut Encoder, base: &BaseModelRef) {
    e.str(base.id().as_str())
        .bytes(base.digest().as_bytes())
        .u64(base.generation());
}

fn state_tag(state: ExpertState) -> u64 {
    match state {
        ExpertState::Candidate => 0,
        ExpertState::Gated => 1,
        ExpertState::Serving => 2,
        ExpertState::Frozen => 3,
        ExpertState::Fenced => 4,
        ExpertState::Superseded => 5,
        ExpertState::Retired(RetiredReason::Pruned) => 6,
        ExpertState::Retired(RetiredReason::Superseded) => 7,
        ExpertState::Retired(RetiredReason::GateFailed) => 8,
        ExpertState::Retired(RetiredReason::InputRevoked) => 9,
    }
}

fn ordered<'a>(a: &'a PodId, b: &'a PodId) -> (&'a PodId, &'a PodId) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

// ---------------------------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------------------------

/// Why the policy produced a request, with the numbers it looked at.
#[derive(Clone, Debug, PartialEq)]
pub enum StructureRationale {
    Birth {
        code: FindingCode,
        unroutable: u64,
        routed_uncleared: u64,
    },
    Reactivate {
        pod: PodId,
        code: FindingCode,
        unroutable: u64,
        routed_uncleared: u64,
    },
    Prune {
        pod: PodId,
        usage: u64,
        aimer: f64,
    },
    Merge {
        first: PodId,
        second: PodId,
        max_clear_rate_gap: f64,
        overlap_excess: f64,
    },
    Freeze {
        pod: PodId,
        windows: usize,
        usage_variance: f64,
        clear_rate_variance: f64,
    },
}

impl StructureRationale {
    /// Digest of the rationale under the configuration and the observation; the
    /// [`RationaleDigest`] of the request.
    fn digest(&self, config_digest: &[u8; 32], observation_digest: &[u8; 32]) -> [u8; 32] {
        let mut enc = Encoder::new(RATIONALE_DOMAIN);
        enc.bytes(config_digest).bytes(observation_digest);
        match self {
            Self::Birth {
                code,
                unroutable,
                routed_uncleared,
            } => {
                enc.u64(1)
                    .str(code.as_str())
                    .u64(*unroutable)
                    .u64(*routed_uncleared);
            }
            Self::Reactivate {
                pod,
                code,
                unroutable,
                routed_uncleared,
            } => {
                enc.u64(2)
                    .str(&pod.0)
                    .str(code.as_str())
                    .u64(*unroutable)
                    .u64(*routed_uncleared);
            }
            Self::Prune { pod, usage, aimer } => {
                enc.u64(3).str(&pod.0).u64(*usage).f64(*aimer);
            }
            Self::Merge {
                first,
                second,
                max_clear_rate_gap,
                overlap_excess,
            } => {
                enc.u64(4)
                    .str(&first.0)
                    .str(&second.0)
                    .f64(*max_clear_rate_gap)
                    .f64(*overlap_excess);
            }
            Self::Freeze {
                pod,
                windows,
                usage_variance,
                clear_rate_variance,
            } => {
                enc.u64(5)
                    .str(&pod.0)
                    .u64(*windows as u64)
                    .f64(*usage_variance)
                    .f64(*clear_rate_variance);
            }
        }
        enc.finish()
    }
}

/// Why a request could not be turned into a proposal.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum StructureCompletionError {
    /// The operation creates no expert, so there is nothing for training to produce; use
    /// [`StructureProposal::ready`].
    NoResultToTrain,
    /// The operation creates an expert; use [`StructureProposal::complete`].
    NeedsTrainedManifest,
    /// The result identity was refused.
    Identity(ExpertIdentityError),
    /// The completed proposal was refused by `ptr-pods`.
    Proposal(ExpertProposalError),
    /// Completion changed the operation id, which it must not.
    OperationIdChanged,
    /// The policy never proposes this operation.
    Unsupported(OpKind),
}

impl fmt::Display for StructureCompletionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoResultToTrain => f.write_str("the operation creates no expert to train"),
            Self::NeedsTrainedManifest => f.write_str("the operation needs a trained manifest"),
            Self::Identity(e) => write!(f, "result identity refused: {e}"),
            Self::Proposal(e) => write!(f, "proposal refused: {e}"),
            Self::OperationIdChanged => f.write_str("completion changed the operation id"),
            Self::Unsupported(kind) => write!(f, "{kind} is never proposed by the policy"),
        }
    }
}

impl std::error::Error for StructureCompletionError {}

/// A request the policy made and the typed rationale behind its digest. Inert: it still needs
/// training (when it creates an expert), a gate report and admission.
#[derive(Clone, Debug, PartialEq)]
pub struct StructureProposal {
    request: ExpertOperationRequest,
    rationale: StructureRationale,
}

impl StructureProposal {
    /// The pre-training request. Its operation id is final.
    pub fn request(&self) -> &ExpertOperationRequest {
        &self.request
    }

    pub fn rationale(&self) -> &StructureRationale {
        &self.rationale
    }

    pub fn kind(&self) -> OpKind {
        self.request.op().kind()
    }

    /// Whether the operation creates an expert whose manifest digest only training produces.
    pub fn needs_training(&self) -> bool {
        !self.request.op().results().is_empty()
    }

    /// The proposal of an operation that creates nothing (Freeze, Prune); `None` for one that
    /// needs training.
    pub fn ready(&self) -> Option<ExpertProposal> {
        if self.needs_training() {
            return None;
        }
        self.assemble(None).ok()
    }

    /// The proposal of an operation that creates an expert, with the manifest digest training
    /// produced. The operation id is unchanged.
    ///
    /// # Errors
    /// [`StructureCompletionError`] when the operation creates no expert or `ptr-pods` refuses
    /// the result.
    pub fn complete(
        &self,
        trained: ManifestDigest,
    ) -> Result<ExpertProposal, StructureCompletionError> {
        if !self.needs_training() {
            return Err(StructureCompletionError::NoResultToTrain);
        }
        self.assemble(Some(trained))
    }

    fn assemble(
        &self,
        trained: Option<ManifestDigest>,
    ) -> Result<ExpertProposal, StructureCompletionError> {
        let spec = |intent: &ResultIntent| -> Result<ExpertPodSpec, StructureCompletionError> {
            let manifest = trained.ok_or(StructureCompletionError::NeedsTrainedManifest)?;
            ExpertPodSpec::new(
                intent.project().clone(),
                intent.pod().clone(),
                manifest,
                intent.level(),
                intent.base_model().clone(),
                intent.generation(),
            )
            .map_err(StructureCompletionError::Identity)
        };
        let op = match self.request.op() {
            RequestedOp::Birth { result } => StructuralOp::Birth {
                result: spec(result)?,
            },
            RequestedOp::Merge { sources, result } => StructuralOp::Merge {
                sources: sources.clone(),
                result: spec(result)?,
            },
            RequestedOp::Reactivate { target, result } => StructuralOp::Reactivate {
                target: target.clone(),
                result: result.as_ref().map(spec).transpose()?,
            },
            RequestedOp::Freeze { target } => StructuralOp::Freeze {
                target: target.clone(),
            },
            RequestedOp::Prune { targets } => StructuralOp::Prune {
                targets: targets.clone(),
            },
            RequestedOp::Split { .. } => {
                return Err(StructureCompletionError::Unsupported(OpKind::Split))
            }
            RequestedOp::Retire { .. } => {
                return Err(StructureCompletionError::Unsupported(OpKind::Retire))
            }
        };
        let proposal = ExpertProposal::new(
            self.request.project().clone(),
            self.request.proposer().clone(),
            self.request.observed_at(),
            op,
            *self.request.rationale(),
        )
        .map_err(StructureCompletionError::Proposal)?;
        if proposal.operation_id() != self.request.operation_id() {
            return Err(StructureCompletionError::OperationIdChanged);
        }
        Ok(proposal)
    }
}

/// Why a candidate did not become a request. Counted, not listed, so the report stays bounded.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Suppression {
    /// `max_proposals_per_window` was reached.
    WindowCap,
    /// `max_births_per_window` was reached.
    BirthCap,
    /// An operand already sits in `max_proposals_per_expert` requests.
    ExpertCap,
    /// A Birth was triggered but the observation has no [`BirthContext`].
    NoBirthContext,
    /// The derived result pod already exists in the registry view or earlier in this call.
    ResultPodExists,
    /// A Token Birth found no free slot.
    NoFreeSlot,
    /// A request for the code is outstanding or an earlier birth of it is still in flight.
    CoolDown,
    /// A Reactivate seed already serves another triggered code of this call.
    SeedAlreadyUsed,
    /// The operation does not allow the target's state ([`OpKind::allows_state`]) or the target
    /// is fenced.
    StateNotAllowed,
    /// The operation is defined only for Token experts and the target is a Segment expert.
    NotTokenLevel,
    /// Pruning would leave a code with no expert that demonstrably clears it.
    LastClearer,
    /// Prune needs an AIMER score and the expert has none.
    MissingAimerScore,
    /// Prune: the expert changed too recently (or its age is unknown).
    TooYoung,
    /// The pair or expert cannot be combined (different layer, base model or code set).
    Incompatible,
    /// `ExpertOperationRequest::new` refused the candidate.
    ProposalRejected,
}

/// The result of one [`StructurePolicy::propose`] call.
#[derive(Clone, Debug, PartialEq)]
pub struct StructureReport {
    proposals: Vec<StructureProposal>,
    suppressed: BTreeMap<Suppression, u64>,
    untracked_unroutable: u64,
}

impl StructureReport {
    /// In priority order: Birth, Reactivate, Prune, Merge, Freeze, each in key order.
    pub fn proposals(&self) -> &[StructureProposal] {
        &self.proposals
    }

    pub fn suppressed(&self) -> &BTreeMap<Suppression, u64> {
        &self.suppressed
    }

    /// Unroutable reports of untracked codes in the latest window; no request can name them.
    pub fn untracked_unroutable(&self) -> u64 {
        self.untracked_unroutable
    }

    pub fn count(&self, kind: OpKind) -> usize {
        self.proposals.iter().filter(|p| p.kind() == kind).count()
    }
}

// ---------------------------------------------------------------------------------------------
// Policy
// ---------------------------------------------------------------------------------------------

/// The L3 policy. Holds a validated configuration and nothing else.
#[derive(Clone, Debug)]
pub struct StructurePolicy {
    config: StructurePolicyConfig,
    config_digest: [u8; 32],
}

struct Candidate {
    rationale: StructureRationale,
    operands: Vec<PodId>,
    op: RequestedOp,
}

impl StructurePolicy {
    /// # Errors
    /// [`StructureConfigError`] when `config` is invalid.
    pub fn new(config: StructurePolicyConfig) -> Result<Self, StructureConfigError> {
        config.validate()?;
        let config_digest = config.digest();
        Ok(Self {
            config,
            config_digest,
        })
    }

    pub fn config(&self) -> &StructurePolicyConfig {
        &self.config
    }

    /// Requests for `observation`. Pure: the same observation gives the same report.
    ///
    /// # Errors
    /// [`StructureObservationError`] when the observation is over a limit or inconsistent.
    pub fn propose(
        &self,
        observation: &StructureObservation,
    ) -> Result<StructureReport, StructureObservationError> {
        observation.validate(&self.config.limits)?;
        let mut run = Run::new(self, observation);
        let (reactivations, births) = run.birth_and_reactivate();
        let prunes = run.prune();
        let merges = run.merge();
        let freezes = run.freeze();
        for candidate in births
            .into_iter()
            .chain(reactivations)
            .chain(prunes)
            .chain(merges)
            .chain(freezes)
        {
            run.emit(candidate);
        }
        Ok(StructureReport {
            proposals: run.out,
            suppressed: run.suppressed,
            untracked_unroutable: observation
                .windows
                .last()
                .map_or(0, |w| w.unroutable_untracked),
        })
    }
}

/// One call's working state.
struct Run<'a> {
    policy: &'a StructurePolicy,
    obs: &'a StructureObservation,
    observation_digest: [u8; 32],
    experts: BTreeMap<&'a PodId, &'a RegisteredExpert>,
    out: Vec<StructureProposal>,
    claims: BTreeMap<PodId, usize>,
    suppressed: BTreeMap<Suppression, u64>,
}

/// Evidence of one triggered code in the latest window.
struct Trigger<'a> {
    code: &'a FindingCode,
    unroutable: u64,
    routed_uncleared: u64,
}

impl Trigger<'_> {
    fn evidence(&self) -> u64 {
        self.unroutable.saturating_add(self.routed_uncleared)
    }
}

/// `(unroutable, routed with no clear anywhere)` of one code in one window.
fn code_evidence(stats: &CodeWindowStats) -> (u64, u64) {
    let routed = stats
        .pods
        .values()
        .fold(0u64, |sum, p| sum.saturating_add(p.routed));
    let cleared = stats
        .pods
        .values()
        .fold(0u64, |sum, p| sum.saturating_add(p.cleared));
    (stats.unroutable, if cleared == 0 { routed } else { 0 })
}

impl<'a> Run<'a> {
    fn new(policy: &'a StructurePolicy, obs: &'a StructureObservation) -> Self {
        Self {
            policy,
            obs,
            observation_digest: obs.digest(),
            experts: obs.experts.iter().map(|e| (e.target.pod(), e)).collect(),
            out: Vec::new(),
            claims: BTreeMap::new(),
            suppressed: BTreeMap::new(),
        }
    }

    fn cfg(&self) -> &'a StructurePolicyConfig {
        &self.policy.config
    }

    fn suppress(&mut self, why: Suppression) {
        *self.suppressed.entry(why).or_insert(0) += 1;
    }

    fn current(&self) -> Option<&'a WindowObservation> {
        self.obs.windows.last()
    }

    /// The last `n` windows, or `None` when there are fewer.
    fn recent(&self, n: usize) -> Option<&'a [WindowObservation]> {
        let windows = &self.obs.windows;
        windows.len().checked_sub(n).map(|start| &windows[start..])
    }

    fn rationale_digest(&self, rationale: &StructureRationale) -> [u8; 32] {
        rationale.digest(&self.policy.config_digest, &self.observation_digest)
    }

    fn emit(&mut self, candidate: Candidate) {
        if self.out.len() >= self.cfg().max_proposals_per_window {
            self.suppress(Suppression::WindowCap);
            return;
        }
        let cap = self.cfg().max_proposals_per_expert;
        if candidate
            .operands
            .iter()
            .any(|pod| self.claims.get(pod).copied().unwrap_or(0) >= cap)
        {
            self.suppress(Suppression::ExpertCap);
            return;
        }
        let digest = self.rationale_digest(&candidate.rationale);
        let Ok(rationale) = RationaleDigest::new(digest) else {
            self.suppress(Suppression::ProposalRejected);
            return;
        };
        let built = ExpertOperationRequest::new(
            self.obs.project.clone(),
            self.obs.proposer.clone(),
            self.obs.observed_at,
            candidate.op,
            rationale,
        );
        let Ok(request) = built else {
            self.suppress(Suppression::ProposalRejected);
            return;
        };
        for pod in &candidate.operands {
            *self.claims.entry(pod.clone()).or_insert(0) += 1;
        }
        self.out.push(StructureProposal {
            request,
            rationale: candidate.rationale,
        });
    }

    /// Codes that reach `min_unroutable` in each of the last `birth_windows` windows, with the
    /// latest window's evidence, in code order.
    fn triggers(&self) -> Vec<Trigger<'a>> {
        let (Some(window), Some(recent)) = (self.current(), self.recent(self.cfg().birth_windows))
        else {
            return Vec::new();
        };
        let min = self.cfg().min_unroutable;
        window
            .codes
            .iter()
            .filter_map(|(code, stats)| {
                let persistent = recent.iter().all(|w| {
                    w.codes.get(code).is_some_and(|s| {
                        let (unroutable, uncleared) = code_evidence(s);
                        unroutable.saturating_add(uncleared) >= min
                    })
                });
                let (unroutable, routed_uncleared) = code_evidence(stats);
                persistent.then_some(Trigger {
                    code,
                    unroutable,
                    routed_uncleared,
                })
            })
            .collect()
    }

    /// Slots still held by an expert in the view.
    fn held_slots(&self) -> BTreeSet<(u16, u16)> {
        self.obs
            .experts
            .iter()
            .filter(|e| occupies_slot(e.target.state()))
            .filter_map(|e| match e.target.level() {
                ExpertLevel::Token { layer, index } => Some((layer, index)),
                ExpertLevel::Segment => None,
            })
            .collect()
    }

    /// Reactivations and births, each in key order. A code whose former expert is
    /// `Retired(GateFailed)` gets a Reactivate (when enabled) and no Birth.
    #[allow(clippy::type_complexity)]
    fn birth_and_reactivate(&mut self) -> (Vec<Candidate>, Vec<Candidate>) {
        let mut triggers = self.triggers();
        triggers.retain(|t| {
            if self.obs.outstanding.contains(t.code) {
                *self.suppressed.entry(Suppression::CoolDown).or_insert(0) += 1;
                false
            } else {
                true
            }
        });
        let mut reactivations = Vec::new();
        let mut pending_births = Vec::new();
        let mut reactivated: BTreeSet<&PodId> = BTreeSet::new();
        for trigger in triggers {
            if self.cfg().ops.reactivate {
                let seed = self
                    .experts
                    .values()
                    .copied()
                    .filter(|e| {
                        e.target.state() == ExpertState::Retired(RetiredReason::GateFailed)
                            && e.codes.contains(trigger.code)
                    })
                    .map(|e| e.target.pod())
                    .next();
                if let Some(pod) = seed {
                    if reactivated.insert(pod) {
                        if let Some(candidate) = self.reactivation(pod, &trigger) {
                            reactivations.push(candidate);
                        }
                    } else {
                        self.suppress(Suppression::SeedAlreadyUsed);
                    }
                    continue;
                }
            }
            if self.cfg().ops.birth {
                pending_births.push(trigger);
            }
        }
        // Births by evidence, then code; the sub-cap keeps the largest.
        pending_births.sort_by(|a, b| b.evidence().cmp(&a.evidence()).then(a.code.cmp(b.code)));
        let mut used_slots = self.held_slots();
        let mut born = BTreeSet::new();
        let mut births = Vec::new();
        for trigger in &pending_births {
            if births.len() >= self.cfg().max_births_per_window {
                self.suppress(Suppression::BirthCap);
                continue;
            }
            if let Some(candidate) = self.birth(trigger, &mut used_slots, &mut born) {
                births.push(candidate);
            }
        }
        (reactivations, births)
    }

    fn reactivation(&mut self, pod: &PodId, trigger: &Trigger<'_>) -> Option<Candidate> {
        let expert = self.experts.get(pod)?;
        let target = expert.target.clone();
        let rationale = StructureRationale::Reactivate {
            pod: pod.clone(),
            code: trigger.code.clone(),
            unroutable: trigger.unroutable,
            routed_uncleared: trigger.routed_uncleared,
        };
        let intent = target.generation().0.checked_add(1).and_then(|next| {
            ResultIntent::new(
                target.project().clone(),
                pod.clone(),
                target.level(),
                target.base_model().clone(),
                Generation(next),
            )
            .ok()
        });
        let Some(intent) = intent else {
            self.suppress(Suppression::ProposalRejected);
            return None;
        };
        Some(Candidate {
            rationale,
            operands: vec![pod.clone()],
            op: RequestedOp::Reactivate {
                target,
                result: Some(intent),
            },
        })
    }

    fn birth(
        &mut self,
        trigger: &Trigger<'_>,
        used_slots: &mut BTreeSet<(u16, u16)>,
        born: &mut BTreeSet<PodId>,
    ) -> Option<Candidate> {
        let Some(context) = self.obs.birth.as_ref() else {
            self.suppress(Suppression::NoBirthContext);
            return None;
        };
        // An earlier birth of this code that is still a Candidate or Gated is in flight.
        let stem = pod_id_stem("repair", &[trigger.code.as_str()]);
        let earlier: Vec<&RegisteredExpert> = self
            .obs
            .experts
            .iter()
            .filter(|e| e.target.pod().0.starts_with(&stem))
            .collect();
        if earlier.iter().any(|e| {
            matches!(
                e.target.state(),
                ExpertState::Candidate | ExpertState::Gated
            )
        }) {
            self.suppress(Suppression::CoolDown);
            return None;
        }
        // A pruned or retired repairer of the code leaves its id behind: the next birth takes
        // the next number.
        let pod = PodId(format!("{stem}{}", earlier.len() + 1));
        if self.experts.contains_key(&pod) || !born.insert(pod.clone()) {
            self.suppress(Suppression::ResultPodExists);
            return None;
        }
        let level = match context.site {
            BirthSite::Segment => ExpertLevel::Segment,
            BirthSite::Token { layer, slots } => {
                let free = (0..slots).find(|index| !used_slots.contains(&(layer, *index)));
                let Some(index) = free else {
                    self.suppress(Suppression::NoFreeSlot);
                    return None;
                };
                used_slots.insert((layer, index));
                ExpertLevel::Token { layer, index }
            }
        };
        let rationale = StructureRationale::Birth {
            code: trigger.code.clone(),
            unroutable: trigger.unroutable,
            routed_uncleared: trigger.routed_uncleared,
        };
        let intent = ResultIntent::new(
            self.obs.project.clone(),
            pod,
            level,
            context.base_model.clone(),
            Generation(1),
        );
        let Ok(result) = intent else {
            self.suppress(Suppression::ProposalRejected);
            return None;
        };
        Some(Candidate {
            rationale,
            operands: Vec::new(),
            op: RequestedOp::Birth { result },
        })
    }

    fn usage_of(window: &WindowObservation, pod: &PodId) -> u64 {
        window.usage.get(pod).copied().unwrap_or(0)
    }

    /// The codes an expert is relied on for: registered, or cleared in any window.
    fn relied_codes(&self, expert: &'a RegisteredExpert) -> BTreeSet<&'a FindingCode> {
        let pod = expert.target.pod();
        let observed = self.obs.windows.iter().flat_map(|w| {
            w.codes
                .iter()
                .filter(move |(_, s)| s.pods.get(pod).is_some_and(|p| p.cleared > 0))
                .map(|(code, _)| code)
        });
        expert.codes.iter().chain(observed).collect()
    }

    /// Whether some expert other than `expert` and `also_excluded` (the Prune candidates already
    /// chosen) demonstrably clears `code`: it is `Serving` or `Frozen` (both routable),
    /// registered for the code, and cleared it at least `backup_min_cleared` times in the last
    /// `backup_clear_windows` windows.
    fn has_demonstrated_backup(
        &self,
        expert: &PodId,
        code: &FindingCode,
        also_excluded: &BTreeSet<&PodId>,
    ) -> bool {
        let Some(recent) = self.recent(self.cfg().backup_clear_windows) else {
            // Less history than the demonstration needs: nobody is demonstrated.
            return false;
        };
        self.obs.experts.iter().any(|other| {
            let pod = other.target.pod();
            let cleared = recent
                .iter()
                .filter_map(|w| w.codes.get(code))
                .filter_map(|s| s.pods.get(pod))
                .fold(0u64, |sum, p| sum.saturating_add(p.cleared));
            pod != expert
                && !also_excluded.contains(pod)
                && matches!(
                    other.target.state(),
                    ExpertState::Serving | ExpertState::Frozen
                )
                && other.codes.contains(code)
                && cleared >= self.cfg().backup_min_cleared
        })
    }

    fn prune(&mut self) -> Vec<Candidate> {
        if !self.cfg().ops.prune {
            return Vec::new();
        }
        let (Some(window), Some(recent)) = (self.current(), self.recent(self.cfg().prune_windows))
        else {
            return Vec::new();
        };
        let floor = self.cfg().prune_usage_floor;
        let mut ranked: Vec<(&RegisteredExpert, u64, Option<f64>)> = self
            .obs
            .experts
            .iter()
            .filter(|e| {
                recent
                    .iter()
                    .all(|w| Self::usage_of(w, e.target.pod()) < floor)
            })
            .map(|e| {
                (
                    e,
                    Self::usage_of(window, e.target.pod()),
                    self.obs.aimer.get(e.target.pod()).copied(),
                )
            })
            .collect();
        ranked.sort_by(|a, b| {
            let score = |x: Option<f64>| x.unwrap_or(f64::NEG_INFINITY);
            score(b.2)
                .total_cmp(&score(a.2))
                .then_with(|| a.0.target.pod().cmp(b.0.target.pod()))
        });
        let mut out: Vec<Candidate> = Vec::new();
        for (expert, usage, aimer) in ranked {
            let target = &expert.target;
            if !OpKind::Prune.allows_state(target.state()) {
                // Only a `Serving` expert has a meaningful low usage worth reporting.
                if matches!(target.state(), ExpertState::Frozen) {
                    self.suppress(Suppression::StateNotAllowed);
                }
                continue;
            }
            if matches!(target.level(), ExpertLevel::Segment) {
                self.suppress(Suppression::NotTokenLevel);
                continue;
            }
            let min_age = self.cfg().prune_min_age_commits;
            let age = expert
                .last_changed_at
                .map(|at| self.obs.observed_at.0.saturating_sub(at.0));
            if min_age > 0 && age.is_none_or(|age| age < min_age) {
                self.suppress(Suppression::TooYoung);
                continue;
            }
            let Some(aimer) = aimer else {
                self.suppress(Suppression::MissingAimerScore);
                continue;
            };
            let pod = target.pod();
            // Candidates chosen so far are not emitted yet, so they are excluded here.
            let already: BTreeSet<&PodId> = out.iter().flat_map(|c| c.operands.iter()).collect();
            let last = self
                .relied_codes(expert)
                .into_iter()
                .any(|code| !self.has_demonstrated_backup(pod, code, &already));
            if last {
                self.suppress(Suppression::LastClearer);
                continue;
            }
            out.push(Candidate {
                rationale: StructureRationale::Prune {
                    pod: pod.clone(),
                    usage,
                    aimer,
                },
                operands: vec![pod.clone()],
                op: RequestedOp::Prune {
                    targets: vec![target.clone()],
                },
            });
        }
        out
    }

    fn clear_rate(
        window: &WindowObservation,
        pod: &PodId,
        code: &FindingCode,
    ) -> Option<(u64, f64)> {
        let stats = window.codes.get(code)?.pods.get(pod)?;
        (stats.routed > 0).then(|| (stats.routed, stats.cleared as f64 / stats.routed as f64))
    }

    fn merge(&mut self) -> Vec<Candidate> {
        if !self.cfg().ops.merge {
            return Vec::new();
        }
        let Some(window) = self.current() else {
            return Vec::new();
        };
        let cfg = self.cfg();
        let mut found: Vec<(f64, Candidate)> = Vec::new();
        for overlap in &self.obs.overlaps {
            let excess = overlap.overlap - overlap.chance;
            if excess + TOLERANCE < cfg.merge_overlap_margin {
                continue;
            }
            let (first, second) = ordered(&overlap.first, &overlap.second);
            let (Some(a), Some(b)) = (self.experts.get(first), self.experts.get(second)) else {
                continue;
            };
            let (a, b) = (&a.target, &b.target);
            if !(OpKind::Merge.allows_state(a.state()) && a.state() == b.state()) {
                self.suppress(Suppression::StateNotAllowed);
                continue;
            }
            let (ExpertLevel::Token { layer: la, index }, ExpertLevel::Token { layer: lb, .. }) =
                (a.level(), b.level())
            else {
                self.suppress(Suppression::NotTokenLevel);
                continue;
            };
            let codes_a = &self.experts[first].codes;
            if la != lb
                || a.base_model() != b.base_model()
                || codes_a != &self.experts[second].codes
                || codes_a.is_empty()
            {
                self.suppress(Suppression::Incompatible);
                continue;
            }
            let mut max_gap = 0.0_f64;
            let comparable = codes_a.iter().all(|code| {
                let (ra, rb) = (
                    Self::clear_rate(window, first, code),
                    Self::clear_rate(window, second, code),
                );
                match (ra, rb) {
                    (Some((na, rate_a)), Some((nb, rate_b)))
                        if na >= cfg.merge_min_routed
                            && nb >= cfg.merge_min_routed
                            && rate_a + TOLERANCE >= cfg.merge_min_clear_rate
                            && rate_b + TOLERANCE >= cfg.merge_min_clear_rate =>
                    {
                        let gap = (rate_a - rate_b).abs();
                        max_gap = max_gap.max(gap);
                        gap <= cfg.merge_clear_rate_margin + TOLERANCE
                    }
                    _ => false,
                }
            });
            if !comparable {
                continue;
            }
            let pod = PodId(format!(
                "{}1",
                pod_id_stem("merged", &[&first.0, &second.0])
            ));
            if self.experts.contains_key(&pod) {
                self.suppress(Suppression::ResultPodExists);
                continue;
            }
            let rationale = StructureRationale::Merge {
                first: first.clone(),
                second: second.clone(),
                max_clear_rate_gap: max_gap,
                overlap_excess: excess,
            };
            let intent = ResultIntent::new(
                self.obs.project.clone(),
                pod,
                ExpertLevel::Token { layer: la, index },
                a.base_model().clone(),
                Generation(1),
            );
            let Ok(result) = intent else {
                self.suppress(Suppression::ProposalRejected);
                continue;
            };
            found.push((
                excess,
                Candidate {
                    rationale,
                    operands: vec![first.clone(), second.clone()],
                    op: RequestedOp::Merge {
                        sources: vec![a.clone(), b.clone()],
                        result,
                    },
                },
            ));
        }
        found.sort_by(|x, y| {
            y.0.total_cmp(&x.0)
                .then_with(|| x.1.operands.cmp(&y.1.operands))
        });
        found.into_iter().map(|(_, candidate)| candidate).collect()
    }

    fn freeze(&mut self) -> Vec<Candidate> {
        let cfg = self.cfg();
        if !cfg.ops.freeze {
            return Vec::new();
        }
        let Some(recent) = self.recent(cfg.freeze_windows) else {
            return Vec::new();
        };
        let k = cfg.freeze_windows;
        let mut experts: Vec<&RegisteredExpert> = self.obs.experts.iter().collect();
        experts.sort_by(|a, b| a.target.pod().cmp(b.target.pod()));
        let mut out = Vec::new();
        for expert in experts {
            let target = &expert.target;
            if !OpKind::Freeze.allows_state(target.state()) {
                continue;
            }
            let pod = target.pod();
            let mut usage = Vec::with_capacity(k);
            let mut rates = Vec::with_capacity(k);
            for window in recent {
                let used = Self::usage_of(window, pod);
                let (routed, cleared) = window
                    .codes
                    .values()
                    .filter_map(|s| s.pods.get(pod))
                    .fold((0u64, 0u64), |(r, c), s| {
                        (r.saturating_add(s.routed), c.saturating_add(s.cleared))
                    });
                if used < cfg.freeze_min_usage || routed == 0 {
                    break;
                }
                let rate = cleared as f64 / routed as f64;
                if rate + TOLERANCE < cfg.freeze_min_clear_rate {
                    break;
                }
                usage.push(used as f64);
                rates.push(rate);
            }
            if usage.len() < k {
                continue;
            }
            let (usage_variance, clear_rate_variance) = (variance(&usage), variance(&rates));
            if usage_variance > cfg.freeze_max_usage_variance + TOLERANCE
                || clear_rate_variance > cfg.freeze_max_clear_rate_variance + TOLERANCE
            {
                continue;
            }
            out.push(Candidate {
                rationale: StructureRationale::Freeze {
                    pod: pod.clone(),
                    windows: k,
                    usage_variance,
                    clear_rate_variance,
                },
                operands: vec![pod.clone()],
                op: RequestedOp::Freeze {
                    target: target.clone(),
                },
            });
        }
        out
    }
}

/// Population variance; the input is non-empty.
fn variance(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n
}

/// The canonical stem of a derived pod id: `prefix.<sanitized parts>.<32 hex of a hash of the
/// parts>.` (with the trailing dot). The caller appends a number, so an id is never reused after
/// its expert was retired.
fn pod_id_stem(prefix: &str, parts: &[&str]) -> String {
    let mut hash = Sha256::new();
    for part in parts {
        hash.update((part.len() as u64).to_be_bytes());
        hash.update(part.as_bytes());
    }
    let digest = hash.finalize();
    let mut id = String::from(prefix);
    for part in parts {
        id.push('.');
        id.extend(part.chars().take(16).map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' {
                c
            } else {
                '_'
            }
        }));
    }
    id.push('.');
    for byte in &digest[..16] {
        id.push_str(&format!("{byte:02x}"));
    }
    id.push('.');
    id
}

/// Domain-separated, length-prefixed digest builder.
struct Encoder(Sha256);

impl Encoder {
    fn new(domain: &[u8]) -> Self {
        let mut hash = Sha256::new();
        hash.update((domain.len() as u64).to_be_bytes());
        hash.update(domain);
        Self(hash)
    }

    fn bytes(&mut self, value: &[u8]) -> &mut Self {
        self.0.update(value);
        self
    }

    fn u64(&mut self, value: u64) -> &mut Self {
        self.0.update(value.to_be_bytes());
        self
    }

    fn f64(&mut self, value: f64) -> &mut Self {
        self.u64(value.to_bits())
    }

    fn str(&mut self, value: &str) -> &mut Self {
        self.u64(value.len() as u64);
        self.0.update(value.as_bytes());
        self
    }

    fn finish(self) -> [u8; 32] {
        self.0.finalize().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The non-test source with comment lines removed.
    fn code() -> String {
        let source = include_str!("structure.rs");
        let code = source.split("#[cfg(test)]").next().unwrap_or(source);
        code.lines()
            .filter(|line| !line.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// The policy has no authority: it may import only the type crates it computes with.
    #[test]
    fn the_module_imports_only_an_allow_list() {
        let code = code();
        let allowed_roots = ["ptr_pods", "ptr_router", "ptr_types", "sha2", "std"];
        let mut uses = 0;
        for statement in code.split(';').map(str::trim) {
            let Some(path) = statement
                .lines()
                .find(|l| l.starts_with("use "))
                .map(|_| statement.trim_start_matches("use ").trim())
            else {
                continue;
            };
            let path = path.split_whitespace().collect::<Vec<_>>().join(" ");
            uses += 1;
            assert!(!path.contains('*'), "glob import: {path}");
            let ok = allowed_roots
                .iter()
                .any(|root| path.starts_with(&format!("{root}::")))
                || (path.starts_with("super::{")
                    && path.contains("occupies_slot")
                    && path.contains("ExpertRegistryProjection")
                    && path.matches(',').count() == 1);
            assert!(ok, "import outside the allow-list: {path}");
        }
        assert!(uses >= 6, "the scan found {uses} imports");
        // No other path into the crate.
        assert!(!code.contains("crate::"), "a crate:: path");
        assert_eq!(code.matches("super::").count(), 1, "a super:: path");
    }

    #[test]
    fn the_module_source_names_no_authority_type() {
        let code = code();
        for forbidden in [
            "ExpertAdmissionGrant",
            "ExpertGateRequirement",
            "ForgettingGate",
            "BoundGateReport",
            "GateSubject",
            "LedgerEvent",
            "FileLedger",
            "ExpertCommitRecord",
            "PtrRuntime",
            "ptr_verifier",
            "ptr_ledger",
            "ptr_security",
            "ptr_exec",
        ] {
            assert!(
                !code.contains(forbidden),
                "{forbidden} appears in the policy"
            );
        }
    }

    /// The only way in is a configuration and an observation, and the only way out a report.
    #[test]
    fn propose_has_the_pinned_signature() {
        let _: fn(
            &StructurePolicy,
            &StructureObservation,
        ) -> Result<StructureReport, StructureObservationError> = StructurePolicy::propose;
    }

    #[test]
    fn derived_pod_ids_are_canonical_wide_and_distinct() {
        let a = pod_id_stem("repair", &["target"]);
        let b = pod_id_stem("repair", &["target "]);
        let long = pod_id_stem("repair", &[&"x".repeat(500)]);
        assert_ne!(a, b);
        assert!(long.len() <= 100);
        assert!(a
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "._:-".contains(c)));
        // Two codes the sanitizer maps to the same text still differ in the hash.
        assert_ne!(
            pod_id_stem("repair", &["a b"]),
            pod_id_stem("repair", &["a_b"])
        );
    }
}
