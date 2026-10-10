//! Routing by verifier findings.
//!
//! A [`FindingRouter`] turns a [`VerificationReport`] into a typed [`RouteOutcome`] over
//! role-tagged Pods. The verifier alone decides the *class* of the outcome, in [`classify`]:
//!
//! - `Pass` with no hard finding at the required level is `Accept`, and nothing else is;
//! - `Fail` with at least one hard finding is `Repair`, one route per distinct hard code in
//!   report order, to the repair-role Pods registered for that code (a code with none is an
//!   explicit [`CodeRoute::Unroutable`], counted in [`FindingRouter::unroutable_counts`]; that
//!   count is the signal a later structural step uses to birth an expert);
//! - `Unknown`, `Disputed`, `Fail` without a hard finding, a `Pass` that carries a hard finding
//!   (contradicts itself) and a `Pass` below the required level are never repaired and never
//!   accepted (the same rule as `Critique::from_report` in `ptr-feedback`). A typed
//!   [`FallbackPolicy`] sends each of them to `Regenerate` (a fresh proposal-role Pod) or
//!   `Escalate`.
//!
//! State is bounded (invariant 8): [`RouterLimits`] caps hard-finding codes per report, code
//! length, tracked unroutable counters (with an explicit overflow bucket) and the registry; a
//! report beyond a cap is a typed fallback ([`FallbackReason::TooManyCodes`],
//! [`FallbackReason::OversizedCode`]), never silently truncated. Only Pods the caller's
//! [`PodEligibility`] view reports as currently eligible are admissible, so a revoked or
//! quarantined Pod is never chosen and the propensity is conditional on the eligible set.
//!
//! A learned or scored component ([`AdmissibleScorer`]) receives only the verifier-determined
//! admissible Pods of one class and returns weights over them. It cannot name an outcome class,
//! so it cannot route anything to `Accept`, and a weight on a Pod outside the admissible set is
//! refused ([`FindingRouteError::OutsideAdmissibleSet`]). Invariant 11: a learned score cannot
//! override a deterministic verifier contradiction.
//!
//! When several Pods are admissible, the choice is sampled with the same hidden keyed draw and
//! quantized probabilities as [`crate::route_checked`]; the [`Propensity`] of every
//! [`FindingRouteRecord`] is the sampled Pod's interval width over `2^53`, so the records are
//! usable by inverse-propensity estimators.

use crate::{
    DecisionNonce, DistributionError, NonceExhausted, NonceSequence, Propensity, PropensityError,
    RouteDistribution, RouteError, SamplingKey,
};
use ptr_types::{Digest, Generation, PodId, Probability, RequestId, VerificationLevel};
use ptr_verifier::{VerificationReport, VerificationStatus};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::sync::Mutex;

/// The stable code of a hard verifier finding (for example `"target"`).
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct FindingCode(String);

impl FindingCode {
    pub fn new(code: impl Into<String>) -> Self {
        Self(code.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for FindingCode {
    fn from(code: &str) -> Self {
        Self::new(code)
    }
}

impl fmt::Display for FindingCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Version of a [`FindingRegistry`]; every accepted change increases it by one.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RegistryGeneration(u64);

impl RegistryGeneration {
    pub const INITIAL: Self = Self(0);

    pub fn get(self) -> u64 {
        self.0
    }
}

/// A typed change of the finding registry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RegistryChange {
    AddRepair { code: FindingCode, pod: PodId },
    RemoveRepair { code: FindingCode, pod: PodId },
    AddProposer { pod: PodId },
    RemoveProposer { pod: PodId },
}

/// A change together with the generation the caller based it on.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegistryUpdate {
    pub expected: RegistryGeneration,
    pub change: RegistryChange,
}

/// Why a registry update was refused. A refused update changes nothing.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RegistryError {
    /// The update was based on an older (or newer) generation than the registry has.
    StaleGeneration {
        expected: RegistryGeneration,
        current: RegistryGeneration,
    },
    AlreadyRegistered,
    NotRegistered,
    GenerationExhausted,
    /// The registry already holds the maximum number of codes or Pods per role.
    CapacityExceeded,
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StaleGeneration { expected, current } => write!(
                f,
                "registry update based on generation {} but the registry is at {}",
                expected.0, current.0
            ),
            Self::AlreadyRegistered => f.write_str("pod is already registered for that role"),
            Self::NotRegistered => f.write_str("pod is not registered for that role"),
            Self::GenerationExhausted => f.write_str("registry generation is exhausted"),
            Self::CapacityExceeded => f.write_str("registry capacity is exceeded"),
        }
    }
}

impl std::error::Error for RegistryError {}

/// Deterministic registry of role-tagged Pods: repair-role Pods per finding code and
/// proposal-role Pods for fresh proposals. Iteration order is `PodId` order.
#[derive(Clone, Debug)]
pub struct FindingRegistry {
    max_codes: usize,
    max_pods_per_role: usize,
    generation: u64,
    repair: BTreeMap<FindingCode, BTreeSet<PodId>>,
    proposers: BTreeSet<PodId>,
}

impl Default for FindingRegistry {
    fn default() -> Self {
        Self::with_capacity(1024, 64)
    }
}

impl FindingRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// A registry holding at most `max_codes` codes and `max_pods_per_role` Pods per code (and
    /// as many proposers).
    pub fn with_capacity(max_codes: usize, max_pods_per_role: usize) -> Self {
        Self {
            max_codes,
            max_pods_per_role,
            generation: 0,
            repair: BTreeMap::new(),
            proposers: BTreeSet::new(),
        }
    }

    pub fn generation(&self) -> RegistryGeneration {
        RegistryGeneration(self.generation)
    }

    /// Repair-role Pods registered for `code`, in `PodId` order.
    pub fn repair_pods(&self, code: &FindingCode) -> Vec<PodId> {
        self.repair
            .get(code)
            .map_or_else(Vec::new, |pods| pods.iter().cloned().collect())
    }

    /// Proposal-role Pods, in `PodId` order.
    pub fn proposers(&self) -> Vec<PodId> {
        self.proposers.iter().cloned().collect()
    }

    /// Applies `update` if it is based on the current generation and changes something, and
    /// returns the new generation.
    pub fn apply(&mut self, update: RegistryUpdate) -> Result<RegistryGeneration, RegistryError> {
        let current = self.generation();
        if update.expected != current {
            return Err(RegistryError::StaleGeneration {
                expected: update.expected,
                current,
            });
        }
        let next = self
            .generation
            .checked_add(1)
            .ok_or(RegistryError::GenerationExhausted)?;
        match update.change {
            RegistryChange::AddRepair { code, pod } => {
                match self.repair.get(&code) {
                    None if self.repair.len() >= self.max_codes => {
                        return Err(RegistryError::CapacityExceeded);
                    }
                    Some(pods) if pods.contains(&pod) => {
                        return Err(RegistryError::AlreadyRegistered);
                    }
                    Some(pods) if pods.len() >= self.max_pods_per_role => {
                        return Err(RegistryError::CapacityExceeded);
                    }
                    _ => {}
                }
                self.repair.entry(code).or_default().insert(pod);
            }
            RegistryChange::RemoveRepair { code, pod } => {
                let pods = self
                    .repair
                    .get_mut(&code)
                    .ok_or(RegistryError::NotRegistered)?;
                if !pods.remove(&pod) {
                    return Err(RegistryError::NotRegistered);
                }
                if pods.is_empty() {
                    self.repair.remove(&code);
                }
            }
            RegistryChange::AddProposer { pod } => {
                if self.proposers.contains(&pod) {
                    return Err(RegistryError::AlreadyRegistered);
                }
                if self.proposers.len() >= self.max_pods_per_role {
                    return Err(RegistryError::CapacityExceeded);
                }
                self.proposers.insert(pod);
            }
            RegistryChange::RemoveProposer { pod } => {
                if !self.proposers.remove(&pod) {
                    return Err(RegistryError::NotRegistered);
                }
            }
        }
        self.generation = next;
        Ok(self.generation())
    }
}

/// Why a report is neither accepted nor repairable.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FallbackReason {
    Unknown,
    Disputed,
    FailWithoutHardFinding,
    /// A `Pass` that carries a hard finding: self-contradictory, so none of its findings is a
    /// repair target.
    ForgedPass,
    /// A hard-finding-free `Pass` below the required verification level.
    BelowLevel,
    /// A `Fail` with more distinct hard codes than [`RouterLimits::max_codes_per_report`].
    TooManyCodes,
    /// A `Fail` with a hard code longer than [`RouterLimits::max_code_len`].
    OversizedCode,
}

/// What to do with a report that is not repairable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FallbackAction {
    Regenerate,
    Escalate,
}

/// The typed fallback policy: one action per [`FallbackReason`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FallbackPolicy {
    pub unknown: FallbackAction,
    pub disputed: FallbackAction,
    pub fail_without_hard_finding: FallbackAction,
    pub forged_pass: FallbackAction,
    /// A below-level `Pass` is a valid but too shallow result. The typed outcomes have no
    /// "re-verify deeper" action, so the default is `Regenerate` (a fresh proposal is verified
    /// at the required level), the same as `ptr-feedback`'s loop, which replaces a candidate
    /// that does not pass at the required level; set `Escalate` to hand it to a deeper
    /// verifier instead.
    pub below_level: FallbackAction,
    pub too_many_codes: FallbackAction,
    pub oversized_code: FallbackAction,
}

impl Default for FallbackPolicy {
    fn default() -> Self {
        Self {
            unknown: FallbackAction::Regenerate,
            disputed: FallbackAction::Escalate,
            fail_without_hard_finding: FallbackAction::Regenerate,
            forged_pass: FallbackAction::Escalate,
            below_level: FallbackAction::Regenerate,
            too_many_codes: FallbackAction::Escalate,
            oversized_code: FallbackAction::Escalate,
        }
    }
}

impl FallbackPolicy {
    pub fn action(&self, reason: FallbackReason) -> FallbackAction {
        match reason {
            FallbackReason::Unknown => self.unknown,
            FallbackReason::Disputed => self.disputed,
            FallbackReason::FailWithoutHardFinding => self.fail_without_hard_finding,
            FallbackReason::ForgedPass => self.forged_pass,
            FallbackReason::BelowLevel => self.below_level,
            FallbackReason::TooManyCodes => self.too_many_codes,
            FallbackReason::OversizedCode => self.oversized_code,
        }
    }
}

/// Typed bounds on the state and per-report work of a [`FindingRouter`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RouterLimits {
    /// Distinct hard-finding codes one `Fail` report may route.
    pub max_codes_per_report: usize,
    /// Bytes of one hard-finding code.
    pub max_code_len: usize,
    /// Distinct codes with their own unroutable counter; later codes share the overflow bucket.
    pub max_tracked_unroutable_codes: usize,
}

impl Default for RouterLimits {
    fn default() -> Self {
        Self {
            max_codes_per_report: 32,
            max_code_len: 128,
            max_tracked_unroutable_codes: 1024,
        }
    }
}

/// Proof that a report was accepted. It can only be built by [`classify`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Accepted(());

/// The verifier-determined class of a report, before any Pod is chosen.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReportClass {
    Accept(Accepted),
    /// Distinct hard-finding codes in report order, at most `max_codes_per_report`.
    Repair(Vec<FindingCode>),
    Fallback(FallbackReason),
}

/// Classifies `report` against `required`. Pure and deterministic; the only place `Accept` is
/// decided (it returns the sealed [`Accepted`]). Work is bounded by `limits`: at most
/// `max_codes_per_report + 1` distinct codes are held.
pub fn classify(
    report: &VerificationReport,
    required: VerificationLevel,
    limits: &RouterLimits,
) -> ReportClass {
    let hard = |report: &VerificationReport| report.findings.iter().any(|finding| finding.hard);
    match report.status {
        VerificationStatus::Pass if hard(report) => {
            ReportClass::Fallback(FallbackReason::ForgedPass)
        }
        VerificationStatus::Pass if level_rank(report.level) < level_rank(required) => {
            ReportClass::Fallback(FallbackReason::BelowLevel)
        }
        VerificationStatus::Pass => ReportClass::Accept(Accepted(())),
        VerificationStatus::Unknown => ReportClass::Fallback(FallbackReason::Unknown),
        VerificationStatus::Disputed => ReportClass::Fallback(FallbackReason::Disputed),
        VerificationStatus::Fail => {
            let mut seen: BTreeSet<&str> = BTreeSet::new();
            let mut codes: Vec<FindingCode> = Vec::new();
            for finding in report.findings.iter().filter(|finding| finding.hard) {
                if finding.code.len() > limits.max_code_len {
                    return ReportClass::Fallback(FallbackReason::OversizedCode);
                }
                if seen.insert(finding.code.as_str()) {
                    if codes.len() >= limits.max_codes_per_report {
                        return ReportClass::Fallback(FallbackReason::TooManyCodes);
                    }
                    codes.push(FindingCode::new(finding.code.clone()));
                }
            }
            if codes.is_empty() {
                ReportClass::Fallback(FallbackReason::FailWithoutHardFinding)
            } else {
                ReportClass::Repair(codes)
            }
        }
    }
}

/// A view of which Pods are eligible right now (for example not revoked or quarantined in the
/// Pod registry), supplied by the caller for each routing.
pub trait PodEligibility {
    /// The current generation of `pod`, or `None` when it must not be routed to.
    fn current_generation(&self, pod: &PodId) -> Option<Generation>;
}

/// An in-memory [`PodEligibility`] snapshot.
#[derive(Clone, Debug, Default)]
pub struct EligibilitySnapshot {
    pods: BTreeMap<PodId, Generation>,
}

impl EligibilitySnapshot {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, pod: PodId, generation: Generation) {
        self.pods.insert(pod, generation);
    }

    pub fn revoke(&mut self, pod: &PodId) {
        self.pods.remove(pod);
    }
}

impl PodEligibility for EligibilitySnapshot {
    fn current_generation(&self, pod: &PodId) -> Option<Generation> {
        self.pods.get(pod).copied()
    }
}

/// Router-owned exploration: the final distribution is `(1 - epsilon) * scorer + epsilon *
/// uniform` over the eligible Pods, so every eligible Pod keeps propensity at least
/// `epsilon / n` whatever a learned scorer says (positivity for off-policy estimation).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Exploration(f64);

/// A rejected exploration rate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExplorationError(pub f64);

impl fmt::Display for ExplorationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "exploration rate {} is not 0 or in [1e-6, 1]", self.0)
    }
}

impl Error for ExplorationError {}

impl Exploration {
    /// No floor: the scorer alone decides (and may starve a Pod).
    pub const NONE: Self = Self(0.0);

    pub fn new(epsilon: f64) -> Result<Self, ExplorationError> {
        if epsilon == 0.0 || (epsilon.is_finite() && (1e-6..=1.0).contains(&epsilon)) {
            Ok(Self(epsilon))
        } else {
            Err(ExplorationError(epsilon))
        }
    }

    pub fn epsilon(self) -> f64 {
        self.0
    }
}

impl Default for Exploration {
    fn default() -> Self {
        Self(0.05)
    }
}

/// Everything configurable about a [`FindingRouter`] except its key, nonces and scorer.
#[derive(Clone, Copy, Debug, Default)]
pub struct RouterConfig {
    pub fallback: FallbackPolicy,
    pub limits: RouterLimits,
    pub exploration: Exploration,
}

fn level_rank(level: VerificationLevel) -> u8 {
    match level {
        VerificationLevel::Unverified => 0,
        VerificationLevel::LatentAgreement => 1,
        VerificationLevel::SampleVerified => 2,
        VerificationLevel::FullSemantic => 3,
        VerificationLevel::Deterministic => 4,
    }
}

/// The role a Pod was chosen for.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RouteClass {
    Repair(FindingCode),
    Regenerate(FallbackReason),
}

/// A sampled Pod choice with everything an off-policy estimator and a replay need.
#[derive(Clone, Debug, PartialEq)]
pub struct FindingRouteRecord {
    request_id: RequestId,
    request_digest: Digest,
    class: RouteClass,
    support: Vec<(PodId, u64)>,
    pod: PodId,
    pod_generation: Generation,
    propensity: Propensity,
    draw: u64,
    nonce: DecisionNonce,
    key_id: [u8; 8],
    generation: RegistryGeneration,
}

impl FindingRouteRecord {
    pub fn request_id(&self) -> &RequestId {
        &self.request_id
    }

    pub fn request_digest(&self) -> Digest {
        self.request_digest
    }

    pub fn class(&self) -> &RouteClass {
        &self.class
    }

    /// The sampling support: every eligible Pod with positive probability and its interval
    /// width out of `2^53`, in `PodId` order. This is the exact distribution the draw used.
    pub fn support(&self) -> &[(PodId, u64)] {
        &self.support
    }

    /// The generation of the chosen Pod reported by the eligibility view at route time.
    pub fn pod_generation(&self) -> Generation {
        self.pod_generation
    }

    pub fn pod(&self) -> &PodId {
        &self.pod
    }

    /// Probability, in `(0, 1]`, with which the logging policy sampled [`Self::pod`].
    pub fn propensity(&self) -> Propensity {
        self.propensity
    }

    pub fn draw(&self) -> u64 {
        self.draw
    }

    pub fn nonce(&self) -> DecisionNonce {
        self.nonce
    }

    pub fn key_id(&self) -> [u8; 8] {
        self.key_id
    }

    pub fn generation(&self) -> RegistryGeneration {
        self.generation
    }

    /// The inverse-propensity weight `target_probability / propensity` for a target policy
    /// giving the logged Pod that probability. In `[0, 2^53]`, always finite.
    pub fn ips_weight(&self, target_probability: Probability) -> f64 {
        f64::from(target_probability.get()) / self.propensity.get()
    }
}

/// The outcome of routing one repair class: a Pod, or an explicit absence of one.
#[derive(Clone, Debug, PartialEq)]
pub enum CodeRoute {
    Routed(FindingRouteRecord),
    /// No repair-role Pod is registered for the code.
    Unroutable(FindingCode),
}

/// Why a report was escalated.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EscalationReason {
    /// The fallback policy chose escalation.
    Policy(FallbackReason),
    /// The policy chose regeneration but no proposal-role Pod is registered.
    NoProposer(FallbackReason),
}

/// The typed routing decision for one report.
#[derive(Clone, Debug, PartialEq)]
pub enum RouteOutcome {
    /// Verified at the required level with no hard finding. The [`Accepted`] proof can only
    /// be produced by [`classify`].
    Accept(Accepted),
    Repair {
        routes: Vec<CodeRoute>,
    },
    Regenerate {
        reason: FallbackReason,
        proposer: FindingRouteRecord,
    },
    Escalate {
        reason: EscalationReason,
    },
}

impl RouteOutcome {
    pub fn is_accept(&self) -> bool {
        matches!(self, Self::Accept(_))
    }

    pub fn is_repair(&self) -> bool {
        matches!(self, Self::Repair { .. })
    }

    /// The codes of a repair outcome that have no registered Pod.
    pub fn unroutable_codes(&self) -> Vec<&FindingCode> {
        match self {
            Self::Repair { routes } => routes
                .iter()
                .filter_map(|route| match route {
                    CodeRoute::Unroutable(code) => Some(code),
                    CodeRoute::Routed(_) => None,
                })
                .collect(),
            Self::Accept(_) | Self::Regenerate { .. } | Self::Escalate { .. } => Vec::new(),
        }
    }

    /// Every sampled choice of the outcome, for logging.
    pub fn records(&self) -> Vec<&FindingRouteRecord> {
        match self {
            Self::Repair { routes } => routes
                .iter()
                .filter_map(|route| match route {
                    CodeRoute::Routed(record) => Some(record),
                    CodeRoute::Unroutable(_) => None,
                })
                .collect(),
            Self::Regenerate { proposer, .. } => vec![proposer],
            Self::Accept(_) | Self::Escalate { .. } => Vec::new(),
        }
    }
}

/// Everything a scorer may see for one class. It names no outcome, only the admissible Pods.
#[non_exhaustive]
pub struct ScoringContext<'a> {
    pub class: &'a RouteClass,
    /// The verifier-determined admissible Pods that are currently eligible, in `PodId` order.
    /// Never empty.
    pub admissible: &'a [PodId],
    pub request_id: &'a RequestId,
    pub request_digest: Digest,
}

/// A learned or scored re-ranking within the admissible set. It returns non-negative weights
/// for admissible Pods only; it cannot select an outcome class.
///
/// # Purity
///
/// Off-policy estimation (`ptr_router::estimate_policy`) rebuilds a scorer's distribution on
/// logged decisions with `request_id = "ope"` and a zero `request_digest`. A scorer used with
/// it must therefore be a function of `class` and `admissible` (and of its own immutable state)
/// only, and must ignore `request_id` and `request_digest`; otherwise the estimated probability
/// of an action is not the probability the router sampled it with. `LearnedPolicy`,
/// `UniformScorer` and `ActiveScorer` are pure in this sense.
pub trait AdmissibleScorer {
    fn weights(&self, context: &ScoringContext<'_>) -> Result<Vec<(PodId, f64)>, RouteError>;

    /// The exploration rate this scorer's evidence was computed under, if it is bound to one
    /// (a learned policy is: its promotion evidence assumed that floor). A router refuses a
    /// scorer whose bound rate is not its own, so a learned policy cannot be deployed without
    /// the floor it was evaluated with. `None` for a scorer that carries no such evidence.
    fn bound_exploration(&self) -> Option<Exploration> {
        None
    }
}

/// Equal weight for every admissible Pod.
#[derive(Clone, Copy, Debug, Default)]
pub struct UniformScorer;

impl AdmissibleScorer for UniformScorer {
    fn weights(&self, context: &ScoringContext<'_>) -> Result<Vec<(PodId, f64)>, RouteError> {
        Ok(context
            .admissible
            .iter()
            .map(|pod| (pod.clone(), 1.0))
            .collect())
    }
}

/// Why a router could not be built.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum RouterBuildError {
    /// The scorer is bound to one exploration rate and the router has another (including none).
    ExplorationMismatch {
        scorer: Exploration,
        router: Exploration,
    },
}

impl fmt::Display for RouterBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExplorationMismatch { scorer, router } => write!(
                f,
                "the scorer is bound to exploration {} and the router uses {}",
                scorer.epsilon(),
                router.epsilon()
            ),
        }
    }
}

impl Error for RouterBuildError {}

/// Why a report could not be routed.
#[derive(Debug)]
#[non_exhaustive]
pub enum FindingRouteErrorKind {
    Scorer(RouteError),
    Distribution(DistributionError),
    InvalidPropensity(PropensityError),
    /// The scorer named a Pod the verifier did not admit (or that is not eligible).
    OutsideAdmissibleSet {
        pod: PodId,
    },
    /// The scorer gave no admissible Pod any weight.
    NoMass,
    NonceExhausted(NonceExhausted),
}

/// A routing failure with the request it happened for. No outcome is produced: the caller must
/// treat it as "not accepted" (escalate), never as a pass.
#[derive(Debug)]
pub struct FindingRouteError {
    request_id: RequestId,
    request_digest: Digest,
    kind: FindingRouteErrorKind,
}

impl FindingRouteError {
    fn new(request: &FindingRouteRequest<'_>, kind: FindingRouteErrorKind) -> Self {
        Self {
            request_id: request.request_id.clone(),
            request_digest: request.request_digest,
            kind,
        }
    }

    pub fn request_id(&self) -> &RequestId {
        &self.request_id
    }

    pub fn request_digest(&self) -> Digest {
        self.request_digest
    }

    pub fn kind(&self) -> &FindingRouteErrorKind {
        &self.kind
    }
}

impl fmt::Display for FindingRouteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "finding routing failed for request {:?}: ",
            self.request_id
        )?;
        match &self.kind {
            FindingRouteErrorKind::Scorer(_) => f.write_str("scorer failed"),
            FindingRouteErrorKind::Distribution(error) => {
                write!(f, "scorer weights rejected: {error}")
            }
            FindingRouteErrorKind::InvalidPropensity(error) => write!(f, "{error}"),
            FindingRouteErrorKind::OutsideAdmissibleSet { pod } => {
                write!(f, "scorer named a pod outside the admissible set: {pod:?}")
            }
            FindingRouteErrorKind::NoMass => {
                f.write_str("scorer gave no admissible pod any weight")
            }
            FindingRouteErrorKind::NonceExhausted(error) => write!(f, "{error}"),
        }
    }
}

impl Error for FindingRouteError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.kind {
            FindingRouteErrorKind::Scorer(error) => Some(error),
            FindingRouteErrorKind::Distribution(error) => Some(error),
            FindingRouteErrorKind::InvalidPropensity(error) => Some(error),
            FindingRouteErrorKind::NonceExhausted(error) => Some(error),
            FindingRouteErrorKind::OutsideAdmissibleSet { .. } | FindingRouteErrorKind::NoMass => {
                None
            }
        }
    }
}

/// One report to route.
#[derive(Clone, Copy)]
pub struct FindingRouteRequest<'a> {
    pub request_id: &'a RequestId,
    pub request_digest: Digest,
    pub report: &'a VerificationReport,
    /// The verification level an accepted candidate must reach.
    pub required: VerificationLevel,
    /// Which registered Pods may be chosen right now.
    pub eligibility: &'a dyn PodEligibility,
}

#[derive(Default)]
struct Counters {
    unroutable: BTreeMap<FindingCode, u64>,
    unroutable_overflow: u64,
    ineligible_skipped: u64,
}

/// Routes verifier reports to role Pods. Owns the sampling key, a nonce sequence, the registry,
/// the configuration and the bounded counters.
pub struct FindingRouter<S = UniformScorer> {
    registry: FindingRegistry,
    config: RouterConfig,
    scorer: S,
    key: SamplingKey,
    nonces: NonceSequence,
    counters: Mutex<Counters>,
}

impl<S> fmt::Debug for FindingRouter<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FindingRouter")
            .field("registry", &self.registry)
            .field("config", &self.config)
            .field("key", &self.key)
            .finish_non_exhaustive()
    }
}

impl<S: AdmissibleScorer> FindingRouter<S> {
    /// # Errors
    /// [`RouterBuildError::ExplorationMismatch`] when `scorer` is bound to an exploration rate
    /// (see [`AdmissibleScorer::bound_exploration`]) other than `config.exploration`. A learned
    /// policy promoted with a floor of `epsilon` cannot be deployed under another floor, nor
    /// under none.
    pub fn new(
        registry: FindingRegistry,
        config: RouterConfig,
        scorer: S,
        key: SamplingKey,
        nonces: NonceSequence,
    ) -> Result<Self, RouterBuildError> {
        if let Some(bound) = scorer.bound_exploration() {
            if bound != config.exploration {
                return Err(RouterBuildError::ExplorationMismatch {
                    scorer: bound,
                    router: config.exploration,
                });
            }
        }
        Ok(Self {
            registry,
            config,
            scorer,
            key,
            nonces,
            counters: Mutex::new(Counters::default()),
        })
    }

    pub fn registry(&self) -> &FindingRegistry {
        &self.registry
    }

    /// Applies a registry update; stale generations are refused.
    pub fn update_registry(
        &mut self,
        update: RegistryUpdate,
    ) -> Result<RegistryGeneration, RegistryError> {
        self.registry.apply(update)
    }

    pub fn key_id(&self) -> [u8; 8] {
        self.key.key_id()
    }

    fn counters(&self) -> std::sync::MutexGuard<'_, Counters> {
        self.counters
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// How often each of the first `max_tracked_unroutable_codes` codes was reported with no
    /// eligible repair Pod, since construction.
    pub fn unroutable_counts(&self) -> BTreeMap<FindingCode, u64> {
        self.counters().unroutable.clone()
    }

    /// Unroutable reports of codes beyond the tracked set.
    pub fn unroutable_overflow(&self) -> u64 {
        self.counters().unroutable_overflow
    }

    /// Registered Pods skipped because the eligibility view did not list them.
    pub fn ineligible_skipped(&self) -> u64 {
        self.counters().ineligible_skipped
    }

    /// Routes `request.report`. Counters change only when the whole routing succeeds; nonces
    /// are consumed by every sampled choice.
    pub fn route(
        &self,
        request: &FindingRouteRequest<'_>,
    ) -> Result<RouteOutcome, FindingRouteError> {
        match classify(request.report, request.required, &self.config.limits) {
            ReportClass::Accept(proof) => Ok(RouteOutcome::Accept(proof)),
            ReportClass::Repair(codes) => {
                let mut routes = Vec::with_capacity(codes.len());
                let mut skipped = 0u64;
                for code in codes {
                    let registered = self.registry.repair_pods(&code);
                    let (eligible, ineligible) = filter_eligible(&registered, request.eligibility);
                    skipped = skipped.saturating_add(ineligible);
                    if eligible.is_empty() {
                        routes.push(CodeRoute::Unroutable(code));
                    } else {
                        let class = RouteClass::Repair(code);
                        routes.push(CodeRoute::Routed(self.sample(request, class, eligible)?));
                    }
                }
                let mut counters = self.counters();
                counters.ineligible_skipped = counters.ineligible_skipped.saturating_add(skipped);
                for route in &routes {
                    if let CodeRoute::Unroutable(code) = route {
                        counters.count_unroutable(code, &self.config.limits);
                    }
                }
                Ok(RouteOutcome::Repair { routes })
            }
            ReportClass::Fallback(reason) => match self.config.fallback.action(reason) {
                FallbackAction::Escalate => Ok(RouteOutcome::Escalate {
                    reason: EscalationReason::Policy(reason),
                }),
                FallbackAction::Regenerate => {
                    let registered = self.registry.proposers();
                    let (eligible, ineligible) = filter_eligible(&registered, request.eligibility);
                    if eligible.is_empty() {
                        let mut counters = self.counters();
                        counters.ineligible_skipped =
                            counters.ineligible_skipped.saturating_add(ineligible);
                        return Ok(RouteOutcome::Escalate {
                            reason: EscalationReason::NoProposer(reason),
                        });
                    }
                    let proposer =
                        self.sample(request, RouteClass::Regenerate(reason), eligible)?;
                    let mut counters = self.counters();
                    counters.ineligible_skipped =
                        counters.ineligible_skipped.saturating_add(ineligible);
                    Ok(RouteOutcome::Regenerate { reason, proposer })
                }
            },
        }
    }

    fn sample(
        &self,
        request: &FindingRouteRequest<'_>,
        class: RouteClass,
        admissible: Vec<PodId>,
    ) -> Result<FindingRouteRecord, FindingRouteError> {
        let fail = |kind| FindingRouteError::new(request, kind);
        let distribution = mixed_distribution(
            &self.scorer,
            &ScoringContext {
                class: &class,
                admissible: &admissible,
                request_id: request.request_id,
                request_digest: request.request_digest,
            },
            self.config.exploration.epsilon(),
        )
        .map_err(fail)?;
        let nonce = self
            .nonces
            .next_nonce()
            .map_err(|error| fail(FindingRouteErrorKind::NonceExhausted(error)))?;
        // Domain separation from `CheckedRouter`: the draw input is a purpose-labelled id.
        let label = match &class {
            RouteClass::Repair(code) => format!("repair/{}", code.as_str()),
            RouteClass::Regenerate(_) => "regenerate".to_string(),
        };
        let domain_id = RequestId::from(
            format!("ptr-router/finding/v1/{label}/{}", request.request_id.0).as_str(),
        );
        let draw = self.key.draw(&nonce, &domain_id, &request.request_digest);
        let support: Vec<(PodId, u64)> = distribution
            .iter()
            .map(|(pod, _)| (pod.clone(), distribution.width(pod)))
            .collect();
        let mut upper = 0u64;
        let mut chosen = None;
        for (pod, width) in &support {
            upper += width;
            if draw < upper {
                chosen = Some((pod.clone(), *width));
                break;
            }
        }
        // Unreachable: the widths sum to 2^53 and the draw is below it. Kept fail-closed.
        let (pod, width) = chosen.ok_or_else(|| fail(FindingRouteErrorKind::NoMass))?;
        let propensity = Propensity::new(width as f64 / RouteDistribution::RESOLUTION as f64)
            .map_err(|error| fail(FindingRouteErrorKind::InvalidPropensity(error)))?;
        let pod_generation = request
            .eligibility
            .current_generation(&pod)
            .ok_or_else(|| {
                fail(FindingRouteErrorKind::OutsideAdmissibleSet { pod: pod.clone() })
            })?;
        Ok(FindingRouteRecord {
            request_id: request.request_id.clone(),
            request_digest: request.request_digest,
            class,
            support,
            pod,
            pod_generation,
            propensity,
            draw,
            nonce,
            key_id: self.key.key_id(),
            generation: self.registry.generation(),
        })
    }
}

/// The distribution a router samples from for one class: the scorer's weights over the
/// admissible Pods, quantized, mixed with the uniform distribution at the router-owned
/// exploration rate `epsilon`. One function for the router and for the off-policy estimator
/// (`ope`), so the probability an estimator assigns to a policy is the probability the router
/// would sample with.
///
/// # Errors
/// The scorer's failure, a weight on a Pod outside `context.admissible`, weights the
/// distribution refuses, or no mass at all.
pub(crate) fn mixed_distribution<S: AdmissibleScorer + ?Sized>(
    scorer: &S,
    context: &ScoringContext<'_>,
    epsilon: f64,
) -> Result<RouteDistribution, FindingRouteErrorKind> {
    let admissible = context.admissible;
    let weights = scorer
        .weights(context)
        .map_err(FindingRouteErrorKind::Scorer)?;
    if let Some((pod, _)) = weights.iter().find(|(pod, _)| !admissible.contains(pod)) {
        return Err(FindingRouteErrorKind::OutsideAdmissibleSet { pod: pod.clone() });
    }
    let scored = RouteDistribution::new(weights).map_err(FindingRouteErrorKind::Distribution)?;
    if scored.is_empty() {
        return Err(FindingRouteErrorKind::NoMass);
    }
    // Router-owned exploration floor: mix the scorer's quantized distribution with uniform.
    let uniform = 1.0 / admissible.len() as f64;
    let mixed: Vec<(PodId, f64)> = admissible
        .iter()
        .map(|pod| {
            let scorer_mass = scored.probability(pod);
            (
                pod.clone(),
                (1.0 - epsilon) * scorer_mass + epsilon * uniform,
            )
        })
        .collect();
    RouteDistribution::new(mixed).map_err(FindingRouteErrorKind::Distribution)
}

impl Counters {
    fn count_unroutable(&mut self, code: &FindingCode, limits: &RouterLimits) {
        if let Some(count) = self.unroutable.get_mut(code) {
            *count = count.saturating_add(1);
        } else if self.unroutable.len() < limits.max_tracked_unroutable_codes {
            self.unroutable.insert(code.clone(), 1);
        } else {
            self.unroutable_overflow = self.unroutable_overflow.saturating_add(1);
        }
    }
}

/// The eligible subset of `registered` (order kept) and the number of Pods skipped.
fn filter_eligible(registered: &[PodId], view: &dyn PodEligibility) -> (Vec<PodId>, u64) {
    let eligible: Vec<PodId> = registered
        .iter()
        .filter(|pod| view.current_generation(pod).is_some())
        .cloned()
        .collect();
    let skipped = (registered.len() - eligible.len()) as u64;
    (eligible, skipped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhausted_generation_is_refused_and_changes_nothing() {
        let mut registry = FindingRegistry::new();
        registry.generation = u64::MAX;
        let update = RegistryUpdate {
            expected: registry.generation(),
            change: RegistryChange::AddProposer {
                pod: PodId::from("p"),
            },
        };
        assert_eq!(
            registry.apply(update),
            Err(RegistryError::GenerationExhausted)
        );
        assert!(registry.proposers().is_empty());
    }

    #[test]
    fn removing_the_last_pod_of_a_code_leaves_no_entry() {
        let mut registry = FindingRegistry::new();
        let code = FindingCode::new("c");
        let pod = PodId::from("p");
        let add = RegistryUpdate {
            expected: registry.generation(),
            change: RegistryChange::AddRepair {
                code: code.clone(),
                pod: pod.clone(),
            },
        };
        registry.apply(add).unwrap();
        let remove = RegistryUpdate {
            expected: registry.generation(),
            change: RegistryChange::RemoveRepair {
                code: code.clone(),
                pod,
            },
        };
        registry.apply(remove).unwrap();
        assert!(registry.repair.is_empty());
        assert!(registry.repair_pods(&code).is_empty());
    }
}
