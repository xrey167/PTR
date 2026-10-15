//! Fallback plan and honest reflex counters.
//!
//! A router may decline (abstain), have nothing to score, or pick a pod that then fails. This
//! module holds the pure data for what happens next: the [`DefaultPodPlan`] a host configures, the
//! [`MissReason`] that says why the primary path produced no answer, and the [`ReflexCounters`]
//! that count what happened.
//!
//! The counters are telemetry. They are not authoritative (they are not ledger records, a
//! restarted runtime starts at zero), they never decide anything, and the only claim they make
//! is the one [`ReflexCounters::coverage`] spells out: whether every miss they saw was caught by
//! the default pod. With nothing observed they say [`FailoverCoverage::NoObservations`], never
//! "covered".

use crate::{AbstentionReason, RouteError};
use ptr_types::PodId;
use std::collections::{BTreeMap, VecDeque};
use std::fmt;

/// Most dispatch attempts one plan may allow per pod request. Only one default pod exists, so a
/// larger budget would only repeat attempts.
pub const MAX_FALLBACK_ATTEMPTS: u8 = 4;

/// How many latency samples [`LatencyWindow`] keeps (invariant 8). Older samples are dropped
/// before a new one is added.
pub const LATENCY_WINDOW: usize = 1024;

/// What to do when the primary path yields no usable pod: dispatch to one explicit default pod,
/// within an attempt budget. The plan names a pod; whether that pod may serve a given request is
/// decided at every use (registry candidate, live generation, effect class, verifier).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DefaultPodPlan {
    default_pod: Option<PodId>,
    max_attempts: u8,
}

/// A [`DefaultPodPlan`] that cannot be built.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DefaultPodPlanError {
    /// A plan must allow at least one dispatch attempt.
    ZeroAttempts,
    /// More attempts than [`MAX_FALLBACK_ATTEMPTS`].
    TooManyAttempts { max: u8, actual: u8 },
}

impl fmt::Display for DefaultPodPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroAttempts => f.write_str("a fallback plan must allow at least one attempt"),
            Self::TooManyAttempts { max, actual } => {
                write!(f, "fallback plan allows {actual} attempts, at most {max}")
            }
        }
    }
}

impl std::error::Error for DefaultPodPlanError {}

impl DefaultPodPlan {
    /// A plan with `default_pod` (or none) and a budget of `max_attempts` dispatch attempts per
    /// pod request. The attempt that went to a routed pod counts; a miss before any dispatch
    /// (an abstention, say) has used none. `max_attempts = 1` therefore lets the default serve
    /// only a request that was not dispatched before.
    ///
    /// # Errors
    /// [`DefaultPodPlanError`] when `max_attempts` is zero or above [`MAX_FALLBACK_ATTEMPTS`].
    pub fn new(default_pod: Option<PodId>, max_attempts: u8) -> Result<Self, DefaultPodPlanError> {
        if max_attempts == 0 {
            return Err(DefaultPodPlanError::ZeroAttempts);
        }
        if max_attempts > MAX_FALLBACK_ATTEMPTS {
            return Err(DefaultPodPlanError::TooManyAttempts {
                max: MAX_FALLBACK_ATTEMPTS,
                actual: max_attempts,
            });
        }
        Ok(Self {
            default_pod,
            max_attempts,
        })
    }

    pub fn default_pod(&self) -> Option<&PodId> {
        self.default_pod.as_ref()
    }

    pub fn max_attempts(&self) -> u8 {
        self.max_attempts
    }
}

/// Why the primary path produced no answer.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MissReason {
    /// The registry offers no candidate for the request, or the chosen pod is not in it.
    NoCandidate,
    /// A learned router declined to choose, for the stated reason.
    Abstained(AbstentionReason),
    /// No eligible candidate had a learned representation (nothing was scored at all). Distinct
    /// from an abstention, where pods were scored.
    NoRepresentation,
    /// The routed pod was invoked and failed.
    DispatchError,
}

impl MissReason {
    /// The miss a routing error stands for, or `None` when the error is a failure of the router
    /// (an invalid distribution, an ineligible choice, ...) that must not be papered over by a
    /// fallback.
    pub fn from_route_error(error: &RouteError) -> Option<Self> {
        match error {
            RouteError::Abstained { reason, .. } => Some(Self::Abstained(*reason)),
            RouteError::NoRepresentation { .. } | RouteError::InactiveRepresentation { .. } => {
                Some(Self::NoRepresentation)
            }
            RouteError::Policy { .. }
            | RouteError::InvalidPropensity(_)
            | RouteError::InvalidDistribution(_)
            | RouteError::InvalidFeatures { .. }
            | RouteError::IneligibleChoice { .. }
            | RouteError::MissingChoice { .. }
            | RouteError::NonceExhausted(_)
            | RouteError::InvalidRepresentation { .. } => None,
        }
    }

    /// A stable, pod-free code for telemetry.
    pub fn code(self) -> &'static str {
        match self {
            Self::NoCandidate => "no_candidate",
            Self::Abstained(AbstentionReason::LowScore) => "abstained_low_score",
            Self::Abstained(AbstentionReason::SmallMargin) => "abstained_small_margin",
            Self::Abstained(AbstentionReason::PartialCoverage) => "abstained_partial_coverage",
            Self::NoRepresentation => "no_representation",
            Self::DispatchError => "dispatch_error",
        }
    }
}

/// Why a miss was not caught by the default pod.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum UncoveredReason {
    /// No plan is configured for the request.
    NoPlan,
    /// The plan names no default pod.
    NoDefault,
    /// The plan's attempt budget was already used by the primary dispatch.
    AttemptsExhausted,
    /// The default pod is the pod that just failed.
    DefaultIsFailedPod,
    /// The default pod is not a registry candidate of the request's project, capability and
    /// input type.
    NotACandidate,
    /// The default pod's generation is revoked, superseded or not live.
    GenerationNotLive,
    /// The default pod declares an effect other than pure or read.
    EffectNotAllowed,
    /// The registry holds another implementation of the default pod than the revision the plan
    /// names and the lifecycle checked: a different namespace or generation, or one registered
    /// without them. The generation that was checked is not the code that would run.
    StaleFallback,
    /// The default pod was dispatched and failed.
    DefaultFailed,
    /// The default pod answered and the answer was not promoted: the verifier contradicted it,
    /// or a taint or type check refused it.
    DefaultRejected,
}

impl UncoveredReason {
    /// A stable, pod-free code for telemetry.
    pub fn code(self) -> &'static str {
        match self {
            Self::NoPlan => "no_plan",
            Self::NoDefault => "no_default",
            Self::AttemptsExhausted => "attempts_exhausted",
            Self::DefaultIsFailedPod => "default_is_failed_pod",
            Self::NotACandidate => "not_a_candidate",
            Self::GenerationNotLive => "generation_not_live",
            Self::EffectNotAllowed => "effect_not_allowed",
            Self::StaleFallback => "stale_fallback",
            Self::DefaultFailed => "default_failed",
            Self::DefaultRejected => "default_rejected",
        }
    }
}

/// What happened to one pod request on the primary path and, after a miss, on the failover path.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReflexOutcome {
    /// The routed pod answered and its output was promoted.
    Hit { pod: PodId },
    /// The primary path produced no answer.
    Miss { reason: MissReason },
    /// The default pod answered a missed request and its output was promoted.
    Failover { to: PodId },
}

/// What the counters can honestly say about failover.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FailoverCoverage {
    /// Nothing was observed: no statement, in particular not "covered".
    NoObservations,
    /// Requests were observed and none missed. Coverage is vacuous: the failover path was never
    /// exercised.
    NoMisses { hits: u64 },
    /// Every one of `misses` misses was caught by the default pod.
    Covered { misses: u64 },
    /// Some misses were not caught (or the counts are inconsistent).
    Uncovered { misses: u64, failovers: u64 },
}

/// One latency sample: how long a pod request took from selection to settlement, and whether it
/// was served.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LatencySample {
    pub micros: u64,
    /// `false` for a request that ended in an error. Failures are timed too: leaving them out
    /// would make a path that fails more look faster.
    pub served: bool,
}

/// A bounded window of the most recent [`LATENCY_WINDOW`] samples, with all-time totals.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LatencyWindow {
    samples: VecDeque<LatencySample>,
    recorded: u64,
    failed: u64,
}

impl LatencyWindow {
    /// Adds a sample; when the window is full the oldest one is dropped first.
    pub fn record(&mut self, sample: LatencySample) {
        if self.samples.len() >= LATENCY_WINDOW {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
        self.recorded = self.recorded.saturating_add(1);
        if !sample.served {
            self.failed = self.failed.saturating_add(1);
        }
    }

    /// Samples in the window.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Samples ever recorded, including those that left the window.
    pub fn recorded(&self) -> u64 {
        self.recorded
    }

    /// Failed samples ever recorded.
    pub fn failed(&self) -> u64 {
        self.failed
    }

    /// Failed samples in the window.
    pub fn failed_in_window(&self) -> usize {
        self.samples.iter().filter(|sample| !sample.served).count()
    }

    pub fn max_micros(&self) -> Option<u64> {
        self.samples.iter().map(|sample| sample.micros).max()
    }

    /// The nearest-rank quantile of the window, `permille` in `0..=1000`; `None` when empty.
    pub fn quantile_micros(&self, permille: u16) -> Option<u64> {
        if self.samples.is_empty() {
            return None;
        }
        let mut micros: Vec<u64> = self.samples.iter().map(|sample| sample.micros).collect();
        micros.sort_unstable();
        let permille = usize::from(permille.min(1000));
        let rank = (micros.len() * permille).div_ceil(1000).max(1);
        micros.get(rank - 1).copied()
    }
}

/// Counters of the reflex path. Telemetry only: not authoritative, never read to decide.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ReflexCounters {
    hits: u64,
    misses: u64,
    failovers: u64,
    rejected: u64,
    inconsistent: u64,
    miss_reasons: BTreeMap<MissReason, u64>,
    uncovered_reasons: BTreeMap<UncoveredReason, u64>,
    latency: LatencyWindow,
}

impl ReflexCounters {
    /// Counts one outcome. A [`ReflexOutcome::Failover`] is counted only against a miss that is
    /// not yet settled; otherwise it is recorded as an inconsistency and never as coverage.
    pub fn record(&mut self, outcome: &ReflexOutcome) {
        match outcome {
            ReflexOutcome::Hit { .. } => self.hits = self.hits.saturating_add(1),
            ReflexOutcome::Miss { reason } => {
                self.misses = self.misses.saturating_add(1);
                let count = self.miss_reasons.entry(*reason).or_insert(0);
                *count = count.saturating_add(1);
            }
            ReflexOutcome::Failover { .. } => {
                if self.has_open_miss() {
                    self.failovers = self.failovers.saturating_add(1);
                } else {
                    self.inconsistent = self.inconsistent.saturating_add(1);
                }
            }
        }
    }

    /// A miss that the default pod did not catch, and why.
    pub fn record_uncovered(&mut self, reason: UncoveredReason) {
        if self.has_open_miss() {
            let count = self.uncovered_reasons.entry(reason).or_insert(0);
            *count = count.saturating_add(1);
        } else {
            self.inconsistent = self.inconsistent.saturating_add(1);
        }
    }

    /// The routed pod answered but its output was not promoted (the verifier or a check refused
    /// it). Neither a hit nor a miss: the pod answered and the answer was not accepted.
    pub fn record_rejected(&mut self) {
        self.rejected = self.rejected.saturating_add(1);
    }

    /// Times one pod request, served or not.
    pub fn record_latency(&mut self, sample: LatencySample) {
        self.latency.record(sample);
    }

    fn has_open_miss(&self) -> bool {
        self.failovers.saturating_add(self.uncovered()) < self.misses
    }

    pub fn hits(&self) -> u64 {
        self.hits
    }

    pub fn misses(&self) -> u64 {
        self.misses
    }

    pub fn failovers(&self) -> u64 {
        self.failovers
    }

    pub fn rejected(&self) -> u64 {
        self.rejected
    }

    /// Recordings that contradicted the counts (a failover or an uncovered record without an open
    /// miss). Non-zero means the counters are not to be trusted.
    pub fn inconsistent(&self) -> u64 {
        self.inconsistent
    }

    pub fn miss_reasons(&self) -> &BTreeMap<MissReason, u64> {
        &self.miss_reasons
    }

    pub fn uncovered_reasons(&self) -> &BTreeMap<UncoveredReason, u64> {
        &self.uncovered_reasons
    }

    /// Misses that were recorded as not caught, with their reason.
    pub fn uncovered(&self) -> u64 {
        self.uncovered_reasons
            .values()
            .fold(0u64, |total, count| total.saturating_add(*count))
    }

    pub fn latency(&self) -> &LatencyWindow {
        &self.latency
    }

    /// Requests that reached an outcome: hits, misses and rejected answers.
    pub fn observations(&self) -> u64 {
        self.hits
            .saturating_add(self.misses)
            .saturating_add(self.rejected)
    }

    /// What the counters say about failover; see [`FailoverCoverage`].
    pub fn coverage(&self) -> FailoverCoverage {
        if self.observations() == 0 {
            FailoverCoverage::NoObservations
        } else if self.inconsistent > 0 || self.failovers != self.misses {
            FailoverCoverage::Uncovered {
                misses: self.misses,
                failovers: self.failovers,
            }
        } else if self.misses == 0 {
            FailoverCoverage::NoMisses { hits: self.hits }
        } else {
            FailoverCoverage::Covered {
                misses: self.misses,
            }
        }
    }

    /// Every miss that was observed was caught by the default pod. False when nothing was
    /// observed. True with no misses is vacuous: use [`ReflexCounters::coverage`] to tell it from
    /// a failover path that was exercised.
    pub fn all_misses_covered(&self) -> bool {
        matches!(
            self.coverage(),
            FailoverCoverage::NoMisses { .. } | FailoverCoverage::Covered { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pod(name: &str) -> PodId {
        PodId::from(name)
    }

    fn miss(counters: &mut ReflexCounters, reason: MissReason) {
        counters.record(&ReflexOutcome::Miss { reason });
    }

    mod plan {
        use super::*;

        #[test]
        fn zero_attempts_are_refused() {
            assert_eq!(
                DefaultPodPlan::new(Some(pod("a")), 0),
                Err(DefaultPodPlanError::ZeroAttempts)
            );
        }

        #[test]
        fn attempts_above_the_cap_report_the_cap_and_the_request() {
            assert_eq!(
                DefaultPodPlan::new(None, MAX_FALLBACK_ATTEMPTS + 1),
                Err(DefaultPodPlanError::TooManyAttempts {
                    max: MAX_FALLBACK_ATTEMPTS,
                    actual: MAX_FALLBACK_ATTEMPTS + 1
                })
            );
        }

        #[test]
        fn a_plan_keeps_its_default_and_budget() {
            let plan = DefaultPodPlan::new(Some(pod("a")), 2).expect("valid plan");
            assert_eq!(plan.default_pod(), Some(&pod("a")));
            assert_eq!(plan.max_attempts(), 2);
        }
    }

    mod reasons {
        use super::*;

        #[test]
        fn abstention_keeps_its_reason_and_is_not_no_representation() {
            let abstained = RouteError::Abstained {
                best: pod("a"),
                runner_up: None,
                best_score: 0.4,
                margin: 0.4,
                reason: AbstentionReason::PartialCoverage,
            };
            assert_eq!(
                MissReason::from_route_error(&abstained),
                Some(MissReason::Abstained(AbstentionReason::PartialCoverage))
            );
            assert_eq!(
                MissReason::from_route_error(&RouteError::NoRepresentation { candidates: 2 }),
                Some(MissReason::NoRepresentation)
            );
        }

        #[test]
        fn router_failures_are_not_misses() {
            for error in [
                RouteError::MissingChoice { candidates: 1 },
                RouteError::IneligibleChoice { pod: pod("a") },
                RouteError::InvalidFeatures {
                    message: String::new(),
                },
            ] {
                assert_eq!(MissReason::from_route_error(&error), None);
            }
        }

        #[test]
        fn every_miss_reason_has_its_own_code() {
            let codes: std::collections::BTreeSet<_> = [
                MissReason::Abstained(AbstentionReason::LowScore),
                MissReason::Abstained(AbstentionReason::SmallMargin),
                MissReason::Abstained(AbstentionReason::PartialCoverage),
                MissReason::NoRepresentation,
                MissReason::NoCandidate,
                MissReason::DispatchError,
            ]
            .map(MissReason::code)
            .into_iter()
            .collect();
            assert_eq!(codes.len(), 6);
        }
    }

    mod counters {
        use super::*;

        #[test]
        fn nothing_observed_is_not_covered() {
            let counters = ReflexCounters::default();
            assert_eq!(counters.coverage(), FailoverCoverage::NoObservations);
            assert!(!counters.all_misses_covered());
        }

        #[test]
        fn hits_without_misses_are_vacuously_covered_and_say_so() {
            let mut counters = ReflexCounters::default();
            counters.record(&ReflexOutcome::Hit { pod: pod("a") });
            assert_eq!(counters.coverage(), FailoverCoverage::NoMisses { hits: 1 });
            assert!(counters.all_misses_covered());
        }

        #[test]
        fn a_miss_is_covered_only_by_a_failover() {
            let mut counters = ReflexCounters::default();
            miss(&mut counters, MissReason::NoRepresentation);
            assert!(!counters.all_misses_covered());
            counters.record(&ReflexOutcome::Failover { to: pod("d") });
            assert_eq!(counters.coverage(), FailoverCoverage::Covered { misses: 1 });
            assert!(counters.all_misses_covered());
        }

        #[test]
        fn one_uncovered_miss_among_covered_ones_is_not_covered() {
            let mut counters = ReflexCounters::default();
            miss(&mut counters, MissReason::NoRepresentation);
            counters.record(&ReflexOutcome::Failover { to: pod("d") });
            miss(&mut counters, MissReason::NoCandidate);
            counters.record_uncovered(UncoveredReason::NotACandidate);
            assert_eq!(
                counters.coverage(),
                FailoverCoverage::Uncovered {
                    misses: 2,
                    failovers: 1
                }
            );
            assert_eq!(counters.uncovered(), 1);
        }

        #[test]
        fn a_failover_without_a_miss_is_an_inconsistency_never_coverage() {
            let mut counters = ReflexCounters::default();
            counters.record(&ReflexOutcome::Hit { pod: pod("a") });
            counters.record(&ReflexOutcome::Failover { to: pod("d") });
            assert_eq!(counters.failovers(), 0);
            assert_eq!(counters.inconsistent(), 1);
            assert!(!counters.all_misses_covered());
        }

        #[test]
        fn an_uncovered_record_without_an_open_miss_is_an_inconsistency() {
            let mut counters = ReflexCounters::default();
            counters.record_uncovered(UncoveredReason::NoPlan);
            assert_eq!(counters.uncovered(), 0);
            assert_eq!(counters.inconsistent(), 1);
        }

        #[test]
        fn a_second_failover_for_one_miss_is_not_counted() {
            let mut counters = ReflexCounters::default();
            miss(&mut counters, MissReason::DispatchError);
            counters.record(&ReflexOutcome::Failover { to: pod("d") });
            counters.record(&ReflexOutcome::Failover { to: pod("d") });
            assert_eq!(counters.failovers(), 1);
            assert_eq!(counters.inconsistent(), 1);
            assert!(!counters.all_misses_covered());
        }

        #[test]
        fn miss_reasons_are_counted_apart() {
            let mut counters = ReflexCounters::default();
            let low = MissReason::Abstained(AbstentionReason::LowScore);
            miss(&mut counters, low);
            miss(&mut counters, low);
            miss(&mut counters, MissReason::NoRepresentation);
            assert_eq!(counters.miss_reasons().get(&low), Some(&2));
            assert_eq!(
                counters.miss_reasons().get(&MissReason::NoRepresentation),
                Some(&1)
            );
            assert_eq!(counters.misses(), 3);
        }

        #[test]
        fn a_rejected_answer_is_an_observation_but_not_a_miss() {
            let mut counters = ReflexCounters::default();
            counters.record_rejected();
            assert_eq!(counters.observations(), 1);
            assert_eq!(counters.misses(), 0);
            assert_eq!(counters.coverage(), FailoverCoverage::NoMisses { hits: 0 });
        }

        #[test]
        fn counts_saturate_instead_of_wrapping() {
            let mut counters = ReflexCounters {
                hits: u64::MAX,
                ..ReflexCounters::default()
            };
            counters.record(&ReflexOutcome::Hit { pod: pod("a") });
            assert_eq!(counters.hits(), u64::MAX);
        }
    }

    mod latency {
        use super::*;

        fn sample(micros: u64, served: bool) -> LatencySample {
            LatencySample { micros, served }
        }

        #[test]
        fn failures_are_timed_too() {
            let mut window = LatencyWindow::default();
            window.record(sample(10, true));
            window.record(sample(900, false));
            assert_eq!(window.len(), 2);
            assert_eq!(window.failed(), 1);
            assert_eq!(window.failed_in_window(), 1);
            assert_eq!(window.max_micros(), Some(900));
        }

        #[test]
        fn the_window_is_bounded_and_drops_the_oldest_first() {
            let mut window = LatencyWindow::default();
            for index in 0..(LATENCY_WINDOW as u64 + 10) {
                window.record(sample(index, index % 2 == 0));
            }
            assert_eq!(window.len(), LATENCY_WINDOW);
            assert_eq!(window.recorded(), LATENCY_WINDOW as u64 + 10);
            assert_eq!(window.quantile_micros(0), Some(10));
        }

        #[test]
        fn quantiles_use_nearest_rank_and_an_empty_window_has_none() {
            let mut window = LatencyWindow::default();
            assert_eq!(window.quantile_micros(500), None);
            for micros in [10, 20, 30, 40] {
                window.record(sample(micros, true));
            }
            assert_eq!(window.quantile_micros(500), Some(20));
            assert_eq!(window.quantile_micros(1000), Some(40));
            assert_eq!(window.quantile_micros(2000), Some(40));
        }

        #[test]
        fn counters_keep_the_failed_samples_of_the_failover_path() {
            let mut counters = ReflexCounters::default();
            counters.record_latency(sample(5, false));
            assert_eq!(counters.latency().failed(), 1);
        }
    }
}
