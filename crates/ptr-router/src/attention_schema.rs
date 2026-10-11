//! A model of the router's own routing that can steer it only inside the admissible set.
//!
//! This is the functional analogue of the attention-schema indicator (AST-1 in
//! `docs/architecture/38-consciousness-indicators.md`): a predictive model of the system's own
//! attention (here, its Pod selections) that is used to control that attention. It is a
//! statistical estimator; it makes no claim about experience.
//!
//! # The model
//!
//! An [`AttentionSchema`] is fed the [`FindingRouteRecord`]s of the router's actual choices
//! ([`AttentionSchema::observe`]) and, later, their verifier outcomes
//! ([`AttentionSchema::settle`]). Per routing class (a repair code or a regeneration reason) it
//! keeps
//!
//! - an exponential moving average (rate 1/16) of the *logged composite attention*: the
//!   quantized sampling widths in [`FindingRouteRecord::support`], i.e. the exact distribution
//!   the router (inner scorer, schema correction and exploration floor together) sampled from.
//!   This is the schema's forecast `q` of its own attention;
//! - an inverse-propensity pass-rate estimate per (class, Pod) from settled records, the
//!   verifier-grounded part of the target attention.
//!
//! Before it learns from a record it forecasts the record's sampling support and scores the
//! forecast against the logged widths. **Schema accuracy is the expected hit rate under the
//! logged policy**: the logged probability of the Pod the schema predicted, averaged over
//! records. It has no sampling noise, and it is reported next to the chance level (the mean of
//! `1/n`) and the oracle level (the mean of the largest logged probability), as a skill score
//! `(expected - chance) / (oracle - chance)`. Raw hits are kept as a diagnostic only: they are
//! dependent, so no interval is given for them. Records with a single-Pod support carry no
//! information and are counted separately. A duplicate record (same request digest and class)
//! is ignored and counted.
//!
//! # How it controls, and what it cannot do
//!
//! Control is a wrapper, [`ControlledScorer<S>`], around the inner scorer (uniform or learned)
//! and it implements [`AdmissibleScorer`], the only interface the router offers. The final
//! weight for each admissible Pod is
//!
//! `w_i = inner_i * c_i`, with `c_i = clamp(balance_i * hint_i, 1/R_i, R_i)`,
//!
//! where `balance_i = (t_i / q_i)^beta` pushes the forecast composite attention `q` towards the
//! target `t`, `hint_i` is `hint_boost` for the Pod a still-valid, validated
//! [`SelfModelHint::PreferRepairPod`] names (and 1 otherwise), and `R_i = min(R, 1 + seen_i / k)`
//! limits the effect on a Pod the schema has barely seen (a new or returning Pod is not
//! flooded). `t` is the verifier-grounded target
//! `(1 - lambda) * r^p / sum(r^p) + lambda / n` with `r_i` the shrunk inverse-propensity pass
//! rate of the Pod (a power-normalised target is used instead of a softmax so replay needs no
//! `exp`), and uniform until enough outcomes are settled. `beta` is one of 0, 1/2, 1 and 2, so
//! every operation (`+ - * / sqrt`) is exactly rounded and bit-identical across platforms.
//!
//! **Ratio bound.** Because the hint boost is inside the clamp, `w_i / inner_i` always lies in
//! `[1/R, R]`, whatever beta, the hint and the forecast do. Between two Pods the final weight
//! ratio is at most `R^2` times the inner ratio. [`ControlConfig::new`] requires
//! `1 <= hint_boost <= R <= 64`. Consumers should run the router with an exploration floor of at
//! least 0.05: then every eligible Pod keeps propensity of at least `0.05 / n` regardless of the
//! schema.
//!
//! The schema:
//!
//! - cannot make, change or see a class decision (`Accept`, `Repair`, `Fallback`): the router
//!   does not consult a scorer for those (invariant 11);
//! - cannot add a Pod: weights for Pods the inner scorer did not name are not created, a Pod
//!   outside the admissible set is still refused by the router (fail closed), and a zero inner
//!   weight stays zero;
//! - keeps the router's exploration floor, hidden draw and propensity logging in force, so
//!   off-policy evaluation of the logged records stays valid. The record's support, not the
//!   schema, is the replay contract.
//!
//! # Arms for ablation (C003)
//!
//! [`ControlArm`] is a typed configuration: `InnerOnly` (the schema is not consulted at all),
//! `ObserveOnly` (identical to the inner scorer, the schema still models), `BalanceOnly` (the
//! balance rule towards a uniform target), `HintOnly` (the hint boost without a forecast),
//! `Full` and `FullMismatched` (the full rule with the forecast rotated by one Pod: the
//! falsifier for "forecast accuracy matters causally").
//!
//! # Bounds and determinism
//!
//! Contexts, Pods per context, the accuracy window, the preference table (replaced wholesale
//! by [`AttentionSchema::apply_hints`], with a time-to-live in observed records), the duplicate
//! ring and the generation table are capped ([`SchemaLimits`]); counters saturate. Statistics
//! are dropped when a Pod's generation changes, and a hint applies only to the generation most
//! recently seen. No clock, no randomness. Diagnostic entropy uses `ln` and is not part of the
//! digest. Interior state is behind a mutex; a poisoned lock is recovered.

use crate::self_model::DigestWriter;
use crate::{
    AdmissibleScorer, FallbackReason, FindingCode, FindingRouteRecord, RouteClass, RouteError,
    ScoringContext, SelfModelHint,
};
use ptr_types::{Digest, Generation, PodId};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

/// Rate of the exponential moving average of logged attention.
const EMA_RATE: f64 = 0.0625;
/// Quantization resolution of the router's sampling widths.
const RESOLUTION: f64 = 9_007_199_254_740_992.0;
/// Pods whose generation is tracked (further Pods can never match a hint).
const MAX_TRACKED_GENERATIONS: usize = 1024;

/// Capacity bounds of an [`AttentionSchema`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SchemaLimits {
    /// Distinct routing classes tracked; further classes share one overflow context.
    pub max_contexts: usize,
    /// Distinct Pods tracked per class; further Pods are forecast at the floor.
    pub max_pods_per_context: usize,
    /// Length of the windowed accuracy (at least 1).
    pub window: usize,
    /// Repair-code preferences kept from hints.
    pub max_preferences: usize,
    /// Observed records after which a stored preference expires.
    pub preference_ttl: u64,
    /// Recent (request digest, class, kind) triples remembered to ignore duplicates.
    pub dedup_window: usize,
    /// Longest code or Pod id accepted in a hint.
    pub max_id_len: usize,
}

impl Default for SchemaLimits {
    fn default() -> Self {
        Self {
            max_contexts: 128,
            max_pods_per_context: 32,
            window: 256,
            max_preferences: 64,
            preference_ttl: 2048,
            dedup_window: 1024,
            max_id_len: 128,
        }
    }
}

/// The exponent `beta` of the balance rule. Only exactly-rounded powers are offered.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BalanceStrength {
    /// `beta = 0`: no balancing.
    Off,
    /// `beta = 1/2` (a square root).
    Half,
    /// `beta = 1`.
    One,
    /// `beta = 2`.
    Two,
}

/// The exponent `p` of the power-normalised target.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TargetPower {
    One,
    Two,
    Four,
}

/// Tuning of the control rule. See the module header for the bound.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ControlConfig {
    balance: BalanceStrength,
    hint_boost: f64,
    max_ratio: f64,
    target_mix: f64,
    target_power: TargetPower,
    min_settled: u64,
    newcomer_k: u64,
}

impl ControlConfig {
    /// `hint_boost` in `[1, max_ratio]`, `max_ratio` in `[1, 64]`, `target_mix` (the uniform
    /// share of the target) in `[0, 1]`, all finite. `min_settled` is the number of settled
    /// outcomes a class needs before its target stops being uniform and `newcomer_k` (at least 1)
    /// the number of observed choices that raises a Pod's ratio cap by one.
    pub fn new(
        balance: BalanceStrength,
        hint_boost: f64,
        max_ratio: f64,
        target_mix: f64,
        target_power: TargetPower,
        min_settled: u64,
        newcomer_k: u64,
    ) -> Result<Self, SchemaError> {
        let in_range =
            |value: f64, low: f64, high: f64| value.is_finite() && (low..=high).contains(&value);
        if !in_range(max_ratio, 1.0, 64.0) {
            return Err(SchemaError::InvalidControl { field: "max_ratio" });
        }
        if !in_range(hint_boost, 1.0, max_ratio) {
            return Err(SchemaError::InvalidControl {
                field: "hint_boost",
            });
        }
        if !in_range(target_mix, 0.0, 1.0) {
            return Err(SchemaError::InvalidControl {
                field: "target_mix",
            });
        }
        if newcomer_k == 0 {
            return Err(SchemaError::InvalidControl {
                field: "newcomer_k",
            });
        }
        Ok(Self {
            balance,
            hint_boost,
            max_ratio,
            target_mix,
            target_power,
            min_settled,
            newcomer_k,
        })
    }

    pub fn balance(&self) -> BalanceStrength {
        self.balance
    }

    pub fn hint_boost(&self) -> f64 {
        self.hint_boost
    }

    /// `R`: the bound of the final-to-inner weight ratio, `[1/R, R]`.
    pub fn max_ratio(&self) -> f64 {
        self.max_ratio
    }

    pub fn target_mix(&self) -> f64 {
        self.target_mix
    }
}

impl Default for ControlConfig {
    fn default() -> Self {
        Self {
            balance: BalanceStrength::One,
            hint_boost: 2.0,
            max_ratio: 8.0,
            target_mix: 0.25,
            target_power: TargetPower::Two,
            min_settled: 30,
            newcomer_k: 8,
        }
    }
}

/// Which parts of the control rule are active. See the module header.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ControlArm {
    /// The wrapper returns the inner weights and never consults the schema.
    InnerOnly,
    /// The wrapper returns the inner weights; the schema still models (exact equality with the
    /// inner scorer).
    ObserveOnly,
    /// Balance towards a uniform target.
    BalanceOnly,
    /// Hint boost only.
    HintOnly,
    /// Balance towards the verifier-grounded target, plus the hint boost.
    Full,
    /// `Full` with the forecast rotated by one Pod: the falsifier control.
    FullMismatched,
}

impl ControlArm {
    fn uses_balance(self) -> bool {
        matches!(self, Self::BalanceOnly | Self::Full | Self::FullMismatched)
    }

    fn uses_hints(self) -> bool {
        matches!(self, Self::HintOnly | Self::Full | Self::FullMismatched)
    }

    fn uses_verifier_target(self) -> bool {
        matches!(self, Self::Full | Self::FullMismatched)
    }
}

/// A rejected schema configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SchemaError {
    InvalidLimits { field: &'static str },
    InvalidControl { field: &'static str },
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLimits { field } => write!(f, "invalid schema limit `{field}`"),
            Self::InvalidControl { field } => write!(f, "invalid control parameter `{field}`"),
        }
    }
}

impl Error for SchemaError {}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum SchemaKey {
    Repair(FindingCode),
    Regenerate(u8),
}

impl SchemaKey {
    fn of(class: &RouteClass) -> Self {
        match class {
            RouteClass::Repair(code) => Self::Repair(code.clone()),
            RouteClass::Regenerate(reason) => Self::Regenerate(match reason {
                FallbackReason::Unknown => 0,
                FallbackReason::Disputed => 1,
                FallbackReason::FailWithoutHardFinding => 2,
                FallbackReason::ForgedPass => 3,
                FallbackReason::BelowLevel => 4,
                FallbackReason::TooManyCodes => 5,
                FallbackReason::OversizedCode => 6,
            }),
        }
    }

    fn write(&self, out: &mut DigestWriter) {
        match self {
            Self::Repair(code) => {
                out.u64(0);
                out.str(code.as_str());
            }
            Self::Regenerate(reason) => {
                out.u64(1);
                out.u64(u64::from(*reason));
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct PodStat {
    /// Records observed with this Pod chosen.
    seen: u64,
    /// Settled records with this Pod chosen.
    settled: u64,
    /// Sum of `pass / propensity` over settled records.
    ips_sum: f64,
}

#[derive(Clone, Debug, Default)]
struct ClassState {
    ema: BTreeMap<PodId, f64>,
    pods: BTreeMap<PodId, PodStat>,
    /// Settled records of the class (the denominator of the inverse-propensity estimates).
    settled: u64,
    observed: u64,
}

impl ClassState {
    fn purge(&mut self, pod: &PodId) {
        self.ema.remove(pod);
        self.pods.remove(pod);
    }

    fn write(&self, out: &mut DigestWriter) {
        out.u64(self.settled);
        out.u64(self.observed);
        out.u64(self.ema.len() as u64);
        for (pod, value) in &self.ema {
            out.str(&pod.0);
            out.f64(*value);
        }
        out.u64(self.pods.len() as u64);
        for (pod, stat) in &self.pods {
            out.str(&pod.0);
            out.u64(stat.seen);
            out.u64(stat.settled);
            out.f64(stat.ips_sum);
        }
    }
}

#[derive(Clone, Debug)]
struct Preference {
    pod: PodId,
    generation: Generation,
    stamp: u64,
}

#[derive(Debug, Default)]
struct State {
    contexts: BTreeMap<SchemaKey, ClassState>,
    overflow: ClassState,
    overflow_observations: u64,
    epoch: u64,
    generations: BTreeMap<PodId, Generation>,
    seen_ring: VecDeque<(Digest, SchemaKey, u8)>,
    seen_set: BTreeSet<(Digest, SchemaKey, u8)>,
    duplicates: u64,
    observations: u64,
    trivial: u64,
    raw_hits: u64,
    expected_sum: f64,
    oracle_sum: f64,
    chance_sum: f64,
    tv_sum: f64,
    brier_sum: f64,
    window: VecDeque<f64>,
    preferences: BTreeMap<FindingCode, Preference>,
}

/// Forecast of attention over `pods`: the moving average restricted to them and normalized, or
/// uniform without information.
fn forecast_over(state: &ClassState, pods: &[PodId]) -> Vec<(PodId, f64)> {
    let raw: Vec<f64> = pods
        .iter()
        .map(|pod| state.ema.get(pod).copied().unwrap_or(0.0))
        .collect();
    let sum: f64 = raw.iter().sum();
    let n = pods.len().max(1) as f64;
    pods.iter()
        .zip(raw)
        .map(|(pod, value)| (pod.clone(), if sum > 0.0 { value / sum } else { 1.0 / n }))
        .collect()
}

fn argmax(distribution: &[(PodId, f64)]) -> Option<(PodId, f64)> {
    let mut best: Option<&(PodId, f64)> = None;
    for item in distribution {
        let better = match best {
            None => true,
            Some(current) => item.1 > current.1 || (item.1 == current.1 && item.0 < current.0),
        };
        if better {
            best = Some(item);
        }
    }
    best.cloned()
}

/// The attention schema. See the module header.
#[derive(Debug)]
pub struct AttentionSchema {
    limits: SchemaLimits,
    state: Mutex<State>,
}

/// What an [`AttentionSchema`] has learned about the router's routing, read in one snapshot.
#[derive(Clone, Debug, PartialEq)]
pub struct SchemaReport {
    /// Scored forecasts (records with at least two Pods in the support).
    pub observations: u64,
    /// Records with a single-Pod support, counted but not scored.
    pub trivial_skipped: u64,
    /// Duplicate records ignored.
    pub duplicates_ignored: u64,
    /// Schema accuracy: the mean logged probability of the Pod the schema predicted.
    pub expected_accuracy: Option<f64>,
    /// The mean logged probability of the most likely Pod (the best any predictor can do).
    pub oracle_accuracy: Option<f64>,
    /// The mean of `1 / n` over the supports (a uniform guess).
    pub chance_accuracy: Option<f64>,
    /// `(expected - chance) / (oracle - chance)`; `None` when the oracle is at chance.
    pub skill: Option<f64>,
    /// Mean of the expected hit over the last [`SchemaLimits::window`] scored forecasts.
    pub windowed_expected_accuracy: Option<f64>,
    /// Mean total variation distance between the forecast and the logged support.
    pub mean_total_variation: Option<f64>,
    /// Mean squared error between the forecast and the logged support.
    pub mean_forecast_brier: Option<f64>,
    /// Raw top-1 hits (a noisy, dependent diagnostic; no interval is given).
    pub raw_hits: u64,
    /// Mean Shannon entropy (nats) of the moving-average attention, weighted by records.
    pub attention_entropy: Option<f64>,
    pub contexts_tracked: usize,
    pub overflow_observations: u64,
    pub preferences: usize,
    /// Observed records so far.
    pub epoch: u64,
    pub digest: [u8; 32],
}

impl AttentionSchema {
    pub fn new(limits: SchemaLimits) -> Result<Self, SchemaError> {
        if limits.window == 0 {
            return Err(SchemaError::InvalidLimits { field: "window" });
        }
        Ok(Self {
            limits,
            state: Mutex::new(State::default()),
        })
    }

    pub fn limits(&self) -> &SchemaLimits {
        &self.limits
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The schema's forecast of the router's attention over `admissible` for `class`. Pure.
    pub fn forecast(&self, class: &RouteClass, admissible: &[PodId]) -> Vec<(PodId, f64)> {
        let state = self.state();
        let empty = ClassState::default();
        forecast_over(
            Self::class_state(&state, &SchemaKey::of(class), &self.limits, &empty),
            admissible,
        )
    }

    /// Predicts which of `admissible` the router will pick for `class`, with its forecast
    /// probability. `None` without admissible Pods.
    pub fn predicted_choice(
        &self,
        class: &RouteClass,
        admissible: &[PodId],
    ) -> Option<(PodId, f64)> {
        argmax(&self.forecast(class, admissible))
    }

    fn class_state<'a>(
        state: &'a State,
        key: &SchemaKey,
        limits: &SchemaLimits,
        empty: &'a ClassState,
    ) -> &'a ClassState {
        match state.contexts.get(key) {
            Some(class) => class,
            None if state.contexts.len() < limits.max_contexts => empty,
            None => &state.overflow,
        }
    }

    fn class_state_mut<'a>(
        state: &'a mut State,
        key: SchemaKey,
        limits: &SchemaLimits,
    ) -> &'a mut ClassState {
        if state.contexts.contains_key(&key) || state.contexts.len() < limits.max_contexts {
            state.contexts.entry(key).or_default()
        } else {
            &mut state.overflow
        }
    }

    /// Records the record as seen. Returns false for a duplicate.
    fn first_sight(&self, state: &mut State, record: &FindingRouteRecord, kind: u8) -> bool {
        let entry = (record.request_digest(), SchemaKey::of(record.class()), kind);
        if state.seen_set.contains(&entry) {
            state.duplicates = state.duplicates.saturating_add(1);
            return false;
        }
        if self.limits.dedup_window > 0 {
            while state.seen_ring.len() >= self.limits.dedup_window {
                if let Some(old) = state.seen_ring.pop_front() {
                    state.seen_set.remove(&old);
                }
            }
            state.seen_set.insert(entry.clone());
            state.seen_ring.push_back(entry);
        }
        true
    }

    /// Drops the statistics of a Pod whose generation changed, then tracks the new generation.
    fn track_generation(&self, state: &mut State, pod: &PodId, generation: Generation) {
        match state.generations.get(pod).copied() {
            Some(known) if known == generation => {}
            Some(_) => {
                state.generations.insert(pod.clone(), generation);
                for class in state.contexts.values_mut() {
                    class.purge(pod);
                }
                state.overflow.purge(pod);
            }
            None if state.generations.len() < MAX_TRACKED_GENERATIONS => {
                state.generations.insert(pod.clone(), generation);
            }
            None => {}
        }
    }

    /// Learns from one actual router choice: forecasts it first, scores the forecast against
    /// the logged support, then updates the moving average. The record is the router's own log
    /// entry, so the schema sees exactly the distribution the router sampled from. A duplicate
    /// record is ignored.
    pub fn observe(&self, record: &FindingRouteRecord) {
        let mut state = self.state();
        if !self.first_sight(&mut state, record, 0) {
            return;
        }
        self.track_generation(&mut state, record.pod(), record.pod_generation());
        let key = SchemaKey::of(record.class());
        let support: Vec<(PodId, f64)> = record
            .support()
            .iter()
            .map(|(pod, width)| (pod.clone(), *width as f64 / RESOLUTION))
            .collect();
        let pods: Vec<PodId> = support.iter().map(|(pod, _)| pod.clone()).collect();
        let empty = ClassState::default();
        let forecast = forecast_over(Self::class_state(&state, &key, &self.limits, &empty), &pods);
        if support.len() < 2 {
            state.trivial = state.trivial.saturating_add(1);
        } else if let Some((top, _)) = argmax(&forecast) {
            let logged = |pod: &PodId| {
                support
                    .iter()
                    .find(|(p, _)| p == pod)
                    .map_or(0.0, |(_, p)| *p)
            };
            let expected = logged(&top);
            let oracle = support.iter().map(|(_, p)| *p).fold(0.0_f64, f64::max);
            let (mut tv, mut brier) = (0.0, 0.0);
            for ((_, q), (_, p)) in forecast.iter().zip(&support) {
                tv += (q - p).abs();
                brier += (q - p) * (q - p);
            }
            state.observations = state.observations.saturating_add(1);
            state.raw_hits = state
                .raw_hits
                .saturating_add(u64::from(top == *record.pod()));
            state.expected_sum += expected;
            state.oracle_sum += oracle;
            state.chance_sum += 1.0 / support.len() as f64;
            state.tv_sum += tv / 2.0;
            state.brier_sum += brier;
            if state.window.len() >= self.limits.window {
                state.window.pop_front();
            }
            state.window.push_back(expected);
        }
        let max_pods = self.limits.max_pods_per_context;
        let overflowed =
            !state.contexts.contains_key(&key) && state.contexts.len() >= self.limits.max_contexts;
        if overflowed {
            state.overflow_observations = state.overflow_observations.saturating_add(1);
        }
        let class = Self::class_state_mut(&mut state, key, &self.limits);
        class.observed = class.observed.saturating_add(1);
        for (pod, value) in class.ema.iter_mut() {
            let p = support
                .iter()
                .find(|(q, _)| q == pod)
                .map_or(0.0, |(_, p)| *p);
            *value += (p - *value) * EMA_RATE;
        }
        for (pod, p) in &support {
            if !class.ema.contains_key(pod) && class.ema.len() < max_pods {
                class.ema.insert(pod.clone(), *p);
            }
        }
        if class.pods.contains_key(record.pod()) || class.pods.len() < max_pods {
            let stat = class.pods.entry(record.pod().clone()).or_default();
            stat.seen = stat.seen.saturating_add(1);
        }
        state.epoch = state.epoch.saturating_add(1);
    }

    /// Credits the record's chosen Pod with the verifier outcome, as an inverse-propensity
    /// term `pass / propensity`, which is what makes the target immune to the extra traffic the
    /// schema itself sends to a Pod. A duplicate is ignored.
    pub fn settle(&self, record: &FindingRouteRecord, passed: bool) {
        let mut state = self.state();
        if !self.first_sight(&mut state, record, 1) {
            return;
        }
        self.track_generation(&mut state, record.pod(), record.pod_generation());
        let key = SchemaKey::of(record.class());
        let max_pods = self.limits.max_pods_per_context;
        let class = Self::class_state_mut(&mut state, key, &self.limits);
        class.settled = class.settled.saturating_add(1);
        if class.pods.contains_key(record.pod()) || class.pods.len() < max_pods {
            let stat = class.pods.entry(record.pod().clone()).or_default();
            stat.settled = stat.settled.saturating_add(1);
            if passed {
                stat.ips_sum += 1.0 / record.propensity().get();
            }
        }
    }

    /// Replaces the stored preferences wholesale with the valid `PreferRepairPod` hints in
    /// `hints` and returns how many were stored. A hint is valid when its code and Pod ids are
    /// at most [`SchemaLimits::max_id_len`] long, it has at least one observation, and its lower
    /// bound is a probability that exceeds the runner-up's upper bound. At most
    /// [`SchemaLimits::max_preferences`] are kept (smallest codes first; a repeated code keeps
    /// the larger lower bound). `EscalateEarly` is a host decision and is ignored. A stored
    /// preference expires after [`SchemaLimits::preference_ttl`] observed records and applies only
    /// to the generation of the Pod most recently seen. Whether a preference acts depends on the
    /// [`ControlArm`] of the scorer.
    pub fn apply_hints(&self, hints: &[SelfModelHint]) -> usize {
        let mut state = self.state();
        let mut valid: BTreeMap<FindingCode, (f64, Preference)> = BTreeMap::new();
        for hint in hints {
            let SelfModelHint::PreferRepairPod {
                code,
                pod,
                generation,
                value_lcb,
                runner_up_ucb,
                evidence,
            } = hint
            else {
                continue;
            };
            let probability = |x: f64| x.is_finite() && (0.0..=1.0).contains(&x);
            if code.as_str().len() > self.limits.max_id_len
                || pod.0.len() > self.limits.max_id_len
                || *evidence == 0
                || !probability(*value_lcb)
                || !probability(*runner_up_ucb)
                || value_lcb <= runner_up_ucb
            {
                continue;
            }
            let keep = valid.get(code).is_none_or(|(old, _)| value_lcb > old);
            if keep {
                valid.insert(
                    code.clone(),
                    (
                        *value_lcb,
                        Preference {
                            pod: pod.clone(),
                            generation: *generation,
                            stamp: state.epoch,
                        },
                    ),
                );
            }
        }
        state.preferences = valid
            .into_iter()
            .take(self.limits.max_preferences)
            .map(|(code, (_, preference))| (code, preference))
            .collect();
        state.preferences.len()
    }

    /// Drops every stored preference.
    pub fn clear_preferences(&self) {
        self.state().preferences.clear();
    }

    /// The per-Pod correction factors `c_i` for `admissible`, in order, within `[1/R_i, R_i]`.
    fn corrections(
        &self,
        class: &RouteClass,
        admissible: &[PodId],
        arm: ControlArm,
        config: &ControlConfig,
    ) -> Vec<f64> {
        let state = self.state();
        let key = SchemaKey::of(class);
        let empty = ClassState::default();
        let class_state = Self::class_state(&state, &key, &self.limits, &empty);
        let n = admissible.len().max(1) as f64;
        let ratio = config.max_ratio;
        let mut q: Vec<f64> = forecast_over(class_state, admissible)
            .into_iter()
            .map(|(_, p)| p)
            .collect();
        if arm == ControlArm::FullMismatched && q.len() > 1 {
            q.rotate_left(1);
        }
        // Shrunk inverse-propensity pass rates and the power-normalised target.
        let target: Vec<f64> =
            if arm.uses_verifier_target() && class_state.settled >= config.min_settled {
                let settled = class_state.settled as f64;
                let scores: Vec<f64> = admissible
                    .iter()
                    .map(|pod| {
                        let sum = class_state.pods.get(pod).map_or(0.0, |s| s.ips_sum);
                        let rate = ((sum + 1.0) / (settled + 2.0)).clamp(0.0, 1.0);
                        match config.target_power {
                            TargetPower::One => rate,
                            TargetPower::Two => rate * rate,
                            TargetPower::Four => (rate * rate) * (rate * rate),
                        }
                    })
                    .collect();
                let total: f64 = scores.iter().sum();
                scores
                    .iter()
                    .map(|score| {
                        let share = if total > 0.0 { score / total } else { 1.0 / n };
                        (1.0 - config.target_mix) * share + config.target_mix / n
                    })
                    .collect()
            } else {
                vec![1.0 / n; admissible.len()]
            };
        admissible
            .iter()
            .enumerate()
            .map(|(i, pod)| {
                let floor = 1.0 / (ratio * n);
                let forecast = q[i].max(floor);
                let gap = target[i] / forecast;
                let balance = if arm.uses_balance() {
                    match config.balance {
                        BalanceStrength::Off => 1.0,
                        BalanceStrength::Half => gap.sqrt(),
                        BalanceStrength::One => gap,
                        BalanceStrength::Two => gap * gap,
                    }
                } else {
                    1.0
                };
                let hint = if arm.uses_hints() && self.preference_applies(&state, class, pod) {
                    config.hint_boost
                } else {
                    1.0
                };
                let seen = class_state.pods.get(pod).map_or(0, |s| s.seen);
                let cap = ratio.min(1.0 + (seen / config.newcomer_k) as f64);
                (balance * hint).clamp(1.0 / cap, cap)
            })
            .collect()
    }

    fn preference_applies(&self, state: &State, class: &RouteClass, pod: &PodId) -> bool {
        let RouteClass::Repair(code) = class else {
            return false;
        };
        state.preferences.get(code).is_some_and(|preference| {
            preference.pod == *pod
                && state.generations.get(pod) == Some(&preference.generation)
                && state.epoch.saturating_sub(preference.stamp) <= self.limits.preference_ttl
        })
    }

    /// A SHA-256 digest over the limits, every table, the accuracy sums and the preferences.
    pub fn digest(&self) -> [u8; 32] {
        self.digest_locked(&self.state())
    }

    fn digest_locked(&self, state: &State) -> [u8; 32] {
        let mut out = DigestWriter::new("ptr-router/attention-schema/v2");
        out.u64(self.limits.max_contexts as u64);
        out.u64(self.limits.max_pods_per_context as u64);
        out.u64(self.limits.window as u64);
        out.u64(self.limits.max_preferences as u64);
        out.u64(self.limits.preference_ttl);
        out.u64(self.limits.dedup_window as u64);
        out.u64(self.limits.max_id_len as u64);
        out.u64(state.contexts.len() as u64);
        for (key, class) in &state.contexts {
            key.write(&mut out);
            class.write(&mut out);
        }
        state.overflow.write(&mut out);
        out.u64(state.overflow_observations);
        out.u64(state.epoch);
        out.u64(state.generations.len() as u64);
        for (pod, generation) in &state.generations {
            out.str(&pod.0);
            out.u64(generation.0);
        }
        out.u64(state.duplicates);
        out.u64(state.observations);
        out.u64(state.trivial);
        out.u64(state.raw_hits);
        for value in [
            state.expected_sum,
            state.oracle_sum,
            state.chance_sum,
            state.tv_sum,
            state.brier_sum,
        ] {
            out.f64(value);
        }
        out.u64(state.window.len() as u64);
        for value in &state.window {
            out.f64(*value);
        }
        out.u64(state.preferences.len() as u64);
        for (code, preference) in &state.preferences {
            out.str(code.as_str());
            out.str(&preference.pod.0);
            out.u64(preference.generation.0);
            out.u64(preference.stamp);
        }
        out.finish()
    }

    /// The schema's accuracy and attention entropy so far, in one consistent snapshot.
    pub fn report(&self) -> SchemaReport {
        let state = self.state();
        let digest = self.digest_locked(&state);
        let n = state.observations as f64;
        let mean = |sum: f64| (state.observations > 0).then(|| sum / n);
        let expected = mean(state.expected_sum);
        let oracle = mean(state.oracle_sum);
        let chance = mean(state.chance_sum);
        let skill = match (expected, oracle, chance) {
            (Some(e), Some(o), Some(c)) if o - c > 1e-12 => Some((e - c) / (o - c)),
            _ => None,
        };
        let mut weighted = 0.0;
        let mut total = 0u64;
        for class in state
            .contexts
            .values()
            .chain(std::iter::once(&state.overflow))
        {
            let sum: f64 = class.ema.values().sum();
            if sum > 0.0 {
                let entropy: f64 = class
                    .ema
                    .values()
                    .map(|value| value / sum)
                    .filter(|p| *p > 0.0)
                    .map(|p| -p * p.ln())
                    .sum();
                weighted += entropy * class.observed as f64;
                total = total.saturating_add(class.observed);
            }
        }
        SchemaReport {
            observations: state.observations,
            trivial_skipped: state.trivial,
            duplicates_ignored: state.duplicates,
            expected_accuracy: expected,
            oracle_accuracy: oracle,
            chance_accuracy: chance,
            skill,
            windowed_expected_accuracy: (!state.window.is_empty())
                .then(|| state.window.iter().sum::<f64>() / state.window.len() as f64),
            mean_total_variation: mean(state.tv_sum),
            mean_forecast_brier: mean(state.brier_sum),
            raw_hits: state.raw_hits,
            attention_entropy: (total > 0).then(|| weighted / total as f64),
            contexts_tracked: state.contexts.len(),
            overflow_observations: state.overflow_observations,
            preferences: state.preferences.len(),
            epoch: state.epoch,
            digest,
        }
    }
}

/// An [`AdmissibleScorer`] that wraps an inner scorer and applies the schema's bounded
/// correction to its weights. See the module header for the rule, the arms and the bound.
#[derive(Debug)]
pub struct ControlledScorer<S> {
    inner: S,
    schema: Arc<AttentionSchema>,
    arm: ControlArm,
    config: ControlConfig,
}

impl<S> ControlledScorer<S> {
    pub fn new(
        inner: S,
        schema: Arc<AttentionSchema>,
        arm: ControlArm,
        config: ControlConfig,
    ) -> Self {
        Self {
            inner,
            schema,
            arm,
            config,
        }
    }

    pub fn arm(&self) -> ControlArm {
        self.arm
    }

    pub fn config(&self) -> &ControlConfig {
        &self.config
    }

    pub fn schema(&self) -> &Arc<AttentionSchema> {
        &self.schema
    }
}

impl<S: AdmissibleScorer> AdmissibleScorer for ControlledScorer<S> {
    fn bound_exploration(&self) -> Option<crate::Exploration> {
        self.inner.bound_exploration()
    }

    fn weights(&self, context: &ScoringContext<'_>) -> Result<Vec<(PodId, f64)>, RouteError> {
        let inner = self.inner.weights(context)?;
        if matches!(self.arm, ControlArm::InnerOnly | ControlArm::ObserveOnly) {
            return Ok(inner);
        }
        let corrections =
            self.schema
                .corrections(context.class, context.admissible, self.arm, &self.config);
        Ok(inner
            .into_iter()
            .map(|(pod, weight)| {
                // A Pod outside the admissible set keeps its weight untouched, so the router
                // still refuses it: the wrapper never repairs or hides an inner violation.
                let factor = context
                    .admissible
                    .iter()
                    .position(|p| *p == pod)
                    .map_or(1.0, |i| corrections[i]);
                (pod, weight * factor)
            })
            .collect())
    }
}
