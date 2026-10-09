//! A bounded, deterministic self-model of the routed Pods: it observes typed traces and predicts
//! the system's own next verifier outcome, hard finding and route. Advisory only.
//!
//! This is the functional analogue of the metacognitive-monitoring indicator (HOT-2/HOT-3 in
//! `docs/architecture/38-consciousness-indicators.md`): a model that predicts the system's own
//! reliability from its own traces, is scored against verifier outcomes, and whose output may
//! change what a host does. It is a statistical estimator. It is never trained on, and never
//! emits, a self-report about experience, and nothing here is evidence of consciousness.
//!
//! # What it does
//!
//! For a [`SelfContext`] (role, producing Pod, round, budget left, required verification level
//! and the finding or refusal codes of the last report) a [`SelfModel`] predicts, before the
//! outcome is known:
//!
//! 1. the probability that the next proposal passes the verifier gate (`classify` returns
//!    `Accept`), a Beta-Bernoulli estimate shrunk towards the global rate of the *other*
//!    contexts;
//! 2. a categorical distribution over the first hard finding code of the next report
//!    (`NoHardFinding`, a tracked code, or `Other`);
//! 3. the Pod (and Pod generation) the router will pick next, a categorical over tracked
//!    Pods and an `Other` bucket that is scored like any class.
//!
//! Every prediction is made before the update (prequential), logged with the realised
//! [`SelfOutcome`] and scored online with the *multiclass* Brier score, the *top-label*
//! expected calibration error over ten fixed equal-width bins (both as in
//! `ptr_analytics::brier_score` and `expected_calibration_error`), a class-conditional
//! reliability table for P(pass) and Wilson intervals ([`CalibrationReport`]).
//!
//! # Hints, and why their evidence is off-policy
//!
//! The router chose repair Pods on context, and a boosted Pod gets more traffic, so raw
//! per-Pod pass rates are confounded and self-reinforcing. Repair evidence is therefore kept as
//! an inverse-propensity estimate: for a finding code `c` with `N` logged repair steps, the
//! value of "always use Pod `X`" is `V(X) = (1/N) * sum(pass_i / propensity_i)` over the steps
//! where `X` was routed, with the router's logged [`Propensity`]. Extra traffic to a boosted
//! Pod comes with a larger propensity and so a smaller weight, which keeps `V` unbiased and the
//! hint from inflating itself. [`SelfModelHint::PreferRepairPod`] needs at least two Pods with
//! enough evidence, and the lower confidence bound of the best must exceed the upper bound of
//! the runner-up. Statistics are keyed by `(code, Pod, Pod generation)` (a superseded generation
//! never lends its record to its successor, invariants 3 and 17) and by the *routed* Pod, not
//! the Pod that produced the failing proposal.
//!
//! # Authority
//!
//! The self-model only observes and proposes. Its output is [`SelfModelHint`], plain advisory
//! data. It holds no reference to a router, registry, gate, threshold or verifier, it does not
//! implement [`AdmissibleScorer`](crate::AdmissibleScorer), and the only way to build an
//! `Accepted` proof is `classify`, which takes a verifier report and nothing else (invariants
//! 4, 5 and 11). A hint can act only through
//! [`AttentionSchema::apply_hints`](crate::AttentionSchema::apply_hints) (which validates it and
//! lets it re-weight inside the verifier-determined admissible set) or through a host decision.
//!
//! A hint cannot be turned into an acceptance:
//!
//! ```compile_fail,E0277
//! fn forge(hint: ptr_router::SelfModelHint) -> ptr_router::Accepted {
//!     hint.into()
//! }
//! ```
//!
//! positive twin: the same proof type is nameable and is produced by `classify` only.
//!
//! ```
//! fn takes_proof(_: Option<ptr_router::Accepted>) {}
//! takes_proof(None);
//! fn data_only(hint: ptr_router::SelfModelHint) -> Option<u64> {
//!     Some(hint.evidence())
//! }
//! let _ = data_only;
//! ```
//!
//! nor into a gate or registry change:
//!
//! ```compile_fail,E0277
//! fn forge(hint: ptr_router::SelfModelHint) -> ptr_router::RegistryUpdate {
//!     hint.into()
//! }
//! ```
//!
//! positive twin: the update type is nameable, and a registry change is built from explicit
//! fields by a host.
//!
//! ```
//! fn takes_update(_: Option<ptr_router::RegistryUpdate>) {}
//! takes_update(None);
//! ```
//!
//! and the proof type cannot be built outside `classify`:
//!
//! ```compile_fail,E0423
//! let forged = ptr_router::Accepted(());
//! ```
//!
//! The model itself is not a scorer, so it cannot be handed to a router:
//!
//! ```compile_fail,E0277
//! fn needs_scorer<S: ptr_router::AdmissibleScorer>(_: &S) {}
//! needs_scorer(&ptr_router::SelfModel::new(ptr_router::SelfModelLimits::default()));
//! ```
//!
//! The positive twin: the same helper accepts the controlled scorer, which is the one
//! sanctioned consumer of hints, and a hint is readable data.
//!
//! ```
//! use std::sync::Arc;
//! fn needs_scorer<S: ptr_router::AdmissibleScorer>(_: &S) {}
//! let schema = Arc::new(
//!     ptr_router::AttentionSchema::new(ptr_router::SchemaLimits::default()).unwrap(),
//! );
//! let controlled = ptr_router::ControlledScorer::new(
//!     ptr_router::UniformScorer,
//!     schema,
//!     ptr_router::ControlArm::ObserveOnly,
//!     ptr_router::ControlConfig::default(),
//! );
//! needs_scorer(&controlled);
//!
//! let hint = ptr_router::SelfModelHint::EscalateEarly {
//!     predicted_pass: 0.1,
//!     threshold: 0.2,
//!     evidence: 40,
//! };
//! assert_eq!(hint.evidence(), 40);
//! ```
//!
//! Stable rustdoc does not check the error codes of `compile_fail` blocks. They were verified by
//! compiling the snippets, and `tests/authority_surface.rs` additionally pins that these source
//! files never name a registry, an update or an `Accepted` constructor.
//!
//! # Bounds and determinism
//!
//! Every table is capped by [`SelfModelLimits`] with an explicit overflow bucket or an
//! eviction rule (invariant 8): contexts, finding codes and Pods per context, repair pairs
//! (evicted by least evidence, then least recent), repair codes and the prediction log (a ring;
//! evictions are counted). Counters saturate; the version saturates. There is no clock, no
//! randomness and no hash iteration, and the update path uses only IEEE `+ - * / sqrt`, so the
//! same trace gives the same predictions, [`SelfModel::digest`] and [`SelfModel::version`].
//! Diagnostic metrics that use `ln` (log loss) are not part of any digest. Codes and Pod ids
//! longer than [`SelfModelLimits::max_code_len`] are replaced by a marker that starts with
//! [`OVERSIZED_CODE`] and carries a hash prefix, so they stay distinct and bounded.

use crate::{classify, FindingCode, FindingRouteRecord, Propensity, ReportClass, RouterLimits};
use ptr_analytics::{wilson_interval, RateEstimate};
use ptr_types::{Generation, PodId, VerificationLevel};
use ptr_verifier::VerificationReport;
use sha2::{Digest as ShaDigest, Sha256};
use std::collections::{BTreeMap, VecDeque};
use std::error::Error;
use std::fmt;

/// Prefix of the marker stored in place of a code or Pod id longer than
/// [`SelfModelLimits::max_code_len`]; a hash prefix follows so long ids stay distinct.
pub const OVERSIZED_CODE: &str = "~oversized";

/// Number of fixed equal-width calibration bins.
const BINS: usize = 10;
/// Two-sided 95% normal quantile for the intervals.
const Z: f64 = 1.959_964;
/// Pseudo-observations pulling a context's pass rate towards the global rate.
const SHRINKAGE: f64 = 2.0;
/// Observations after which a repair pair is established: probationary pairs are evicted first,
/// so a flood of novel pairs displaces at most one established pair and then only itself.
const PROBATION: u64 = 4;
/// Additive smoothing of the categorical heads.
const ALPHA: f64 = 0.5;

/// Capacity bounds of a [`SelfModel`]. Zero is legal and sends everything to the overflow
/// bucket or the eviction path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SelfModelLimits {
    /// Distinct context keys tracked; further contexts share one overflow context.
    pub max_contexts: usize,
    /// Distinct hard-finding codes counted per context; further codes count as `Other`.
    pub max_codes_per_context: usize,
    /// Distinct routed Pods counted per context; further Pods count as `Other`.
    pub max_pods_per_context: usize,
    /// Distinct (code, Pod, generation) repair pairs tracked. When full, a probationary pair
    /// (fewer than 4 observations) is evicted first, else the pair with the least evidence, then
    /// the least recent.
    pub max_repair_pairs: usize,
    /// Distinct finding codes with repair totals tracked (same eviction rule).
    pub max_repair_codes: usize,
    /// Codes kept in a context key (sorted, distinct); more are summarised by a flag. The same
    /// canonical list is used for learning and for hints.
    pub max_context_codes: usize,
    /// Longest finding code or Pod id stored.
    pub max_code_len: usize,
    /// Predictions kept in the ring log.
    pub max_log: usize,
}

impl Default for SelfModelLimits {
    fn default() -> Self {
        Self {
            max_contexts: 256,
            max_codes_per_context: 16,
            max_pods_per_context: 16,
            max_repair_pairs: 128,
            max_repair_codes: 64,
            max_context_codes: 4,
            max_code_len: 128,
            max_log: 1024,
        }
    }
}

/// Which context features the model may use. All on by default; switching one off is the
/// feature ablation of C002 (a self-model without that information).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FeatureMask {
    pub role: bool,
    pub pod: bool,
    pub round: bool,
    pub budget: bool,
    pub codes: bool,
    pub required: bool,
}

impl Default for FeatureMask {
    fn default() -> Self {
        Self::ALL
    }
}

impl FeatureMask {
    pub const ALL: Self = Self {
        role: true,
        pod: true,
        round: true,
        budget: true,
        codes: true,
        required: true,
    };
}

/// A Pod together with the generation it was routed at. Statistics are never shared between
/// generations.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PodRef {
    pub pod: PodId,
    pub generation: Generation,
}

impl PodRef {
    pub fn new(pod: PodId, generation: Generation) -> Self {
        Self { pod, generation }
    }
}

/// Which kind of proposal a context describes.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum SelfRole {
    /// A fresh proposal.
    Proposal,
    /// A repair of a reported finding.
    Repair,
}

/// The typed features the self-model, and any external probe, sees for one prediction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SelfContext {
    pub role: SelfRole,
    /// The Pod that produced the last proposal, if any. This is a context feature only; the
    /// repair statistics are keyed by [`SelfOutcome::routed_pod`].
    pub producer: Option<PodRef>,
    /// Zero-based round index within the request.
    pub round: u32,
    /// Remaining attempt budget.
    pub budget_left: u32,
    /// The verification level a proposal must reach: the pass label depends on it.
    pub required: VerificationLevel,
    /// Hard finding codes of the last report and workspace refusal codes, any order.
    pub last_codes: Vec<FindingCode>,
}

impl SelfContext {
    /// Builds a context from the last verifier report: its distinct hard codes in report
    /// order.
    pub fn from_report(
        role: SelfRole,
        producer: Option<PodRef>,
        round: u32,
        budget_left: u32,
        required: VerificationLevel,
        report: &VerificationReport,
    ) -> Self {
        let mut last_codes: Vec<FindingCode> = Vec::new();
        for finding in report.findings.iter().filter(|finding| finding.hard) {
            let code = FindingCode::new(finding.code.clone());
            if !last_codes.contains(&code) {
                last_codes.push(code);
            }
        }
        Self {
            role,
            producer,
            round,
            budget_left,
            required,
            last_codes,
        }
    }

    /// Adds workspace refusal codes (findings are broadcast as refusals) to the context.
    pub fn with_refusal_codes<'a>(mut self, codes: impl IntoIterator<Item = &'a str>) -> Self {
        for code in codes {
            let code = FindingCode::new(code);
            if !self.last_codes.contains(&code) {
                self.last_codes.push(code);
            }
        }
        self
    }
}

/// What happened to the next proposal.
#[derive(Clone, Debug, PartialEq)]
pub struct SelfOutcome {
    /// The verifier gate passed: `classify` returned `Accept`.
    pub passed: bool,
    /// The first hard finding code of the report, if any.
    pub primary_code: Option<FindingCode>,
    /// The Pod and generation the router actually used for this proposal or repair.
    pub routed_pod: Option<PodRef>,
    /// The router's logged propensity of that choice. Without it the step still trains the
    /// three heads but is not repair evidence.
    pub propensity: Option<Propensity>,
}

impl SelfOutcome {
    /// Builds an outcome from the verifier report of the proposal and the router's record.
    /// Whether it passed is decided by `classify` alone.
    pub fn from_report(
        report: &VerificationReport,
        required: VerificationLevel,
        limits: &RouterLimits,
        routed: Option<&FindingRouteRecord>,
    ) -> Self {
        let passed = matches!(classify(report, required, limits), ReportClass::Accept(_));
        let primary_code = report
            .findings
            .iter()
            .find(|finding| finding.hard)
            .map(|finding| FindingCode::new(finding.code.clone()));
        Self {
            passed,
            primary_code,
            routed_pod: routed
                .map(|record| PodRef::new(record.pod().clone(), record.pod_generation())),
            propensity: routed.map(FindingRouteRecord::propensity),
        }
    }
}

/// One trace step: the context before a proposal and what followed. This is the only thing
/// the self-model and an external probe learn from.
#[derive(Clone, Debug, PartialEq)]
pub struct SelfTraceStep {
    pub context: SelfContext,
    pub outcome: SelfOutcome,
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

fn fnv(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

fn marker(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    let hash: [u8; 32] = hasher.finalize().into();
    let mut out = String::from(OVERSIZED_CODE);
    out.push(':');
    for byte in &hash[..8] {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn bound_code(code: &FindingCode, limits: &SelfModelLimits) -> FindingCode {
    if code.as_str().len() > limits.max_code_len {
        FindingCode::new(marker(code.as_str()))
    } else {
        code.clone()
    }
}

fn bound_pod(pod: &PodRef, limits: &SelfModelLimits) -> PodRef {
    if pod.pod.0.len() > limits.max_code_len {
        PodRef::new(PodId(marker(&pod.pod.0)), pod.generation)
    } else {
        pod.clone()
    }
}

/// The canonical code list of a context: bounded, sorted, distinct, at most
/// `max_context_codes`, plus whether more were dropped. Used for the key, for learning and for
/// hints, so all three agree.
fn canonical_codes(codes: &[FindingCode], limits: &SelfModelLimits) -> (Vec<FindingCode>, bool) {
    let mut codes: Vec<FindingCode> = codes.iter().map(|code| bound_code(code, limits)).collect();
    codes.sort();
    codes.dedup();
    let more = codes.len() > limits.max_context_codes;
    codes.truncate(limits.max_context_codes);
    (codes, more)
}

/// A bucketed, bounded context key.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ContextKey {
    role: u8,
    producer: Option<PodRef>,
    round: u8,
    budget: u8,
    required: u8,
    codes: Vec<FindingCode>,
    more_codes: bool,
}

impl ContextKey {
    fn new(context: &SelfContext, limits: &SelfModelLimits, mask: &FeatureMask) -> Self {
        let (codes, more_codes) = if mask.codes {
            canonical_codes(&context.last_codes, limits)
        } else {
            (Vec::new(), false)
        };
        Self {
            role: if mask.role {
                match context.role {
                    SelfRole::Proposal => 0,
                    SelfRole::Repair => 1,
                }
            } else {
                0
            },
            producer: context
                .producer
                .as_ref()
                .filter(|_| mask.pod)
                .map(|pod| bound_pod(pod, limits)),
            round: if mask.round {
                u8::try_from(context.round.min(7)).unwrap_or(7)
            } else {
                0
            },
            budget: if mask.budget {
                match context.budget_left {
                    0 => 0,
                    1..=3 => 1,
                    4..=15 => 2,
                    _ => 3,
                }
            } else {
                0
            },
            required: if mask.required {
                level_rank(context.required)
            } else {
                0
            },
            codes,
            more_codes,
        }
    }

    fn write(&self, out: &mut DigestWriter) {
        out.u64(u64::from(self.role));
        match &self.producer {
            Some(pod) => {
                out.u64(1);
                write_pod(out, pod);
            }
            None => out.u64(0),
        }
        out.u64(u64::from(self.round));
        out.u64(u64::from(self.budget));
        out.u64(u64::from(self.required));
        out.u64(self.codes.len() as u64);
        for code in &self.codes {
            out.str(code.as_str());
        }
        out.u64(u64::from(self.more_codes));
    }

    fn digest(&self) -> [u8; 32] {
        let mut out = DigestWriter::new("ptr-router/self-model/context/v2");
        self.write(&mut out);
        out.finish()
    }
}

fn write_pod(out: &mut DigestWriter, pod: &PodRef) {
    out.str(&pod.pod.0);
    out.u64(pod.generation.0);
}

/// An outcome class of the finding-code head.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum CodeOutcome {
    /// The report has no hard finding.
    NoHardFinding,
    /// A tracked hard finding code.
    Code(FindingCode),
    /// Any code not tracked in the context.
    Other,
}

/// The self-model's prediction for one context.
#[derive(Clone, Debug, PartialEq)]
pub struct SelfPrediction {
    /// P(the next proposal passes the verifier gate).
    pub pass_probability: f64,
    /// Observations behind the pass estimate in this context (not counting the prior).
    pub evidence: u64,
    /// The most likely first hard-finding outcome and its probability.
    pub code: Option<(CodeOutcome, f64)>,
    /// The Pod the router is most likely to pick, and its probability.
    pub route: Option<(PodRef, f64)>,
}

/// A prediction with the outcome that followed it.
#[derive(Clone, Debug, PartialEq)]
pub struct LoggedPrediction {
    /// Position in the trace, from 0.
    pub seq: u64,
    /// The model version the prediction was made under.
    pub version: u64,
    /// SHA-256 of the bucketed context key.
    pub context_digest: [u8; 32],
    pub prediction: SelfPrediction,
    pub outcome: SelfOutcome,
}

/// A confidence or score outside its domain was given to a calibration tracker.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibrationError {
    value: f64,
}

impl CalibrationError {
    pub fn value(&self) -> f64 {
        self.value
    }
}

impl fmt::Display for CalibrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} is not a valid probability or score", self.value)
    }
}

impl Error for CalibrationError {}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct BinStat {
    n: u64,
    confidence_sum: f64,
    correct: u64,
}

/// Online calibration of top-class predictions: the multiclass Brier score, the top-label
/// expected calibration error over ten fixed equal-width bins and the top-class accuracy.
/// State is constant.
#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationTracker {
    n: u64,
    correct: u64,
    brier_sum: f64,
    bins: [BinStat; BINS],
}

impl Default for CalibrationTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl CalibrationTracker {
    pub fn new() -> Self {
        Self {
            n: 0,
            correct: 0,
            brier_sum: 0.0,
            bins: [BinStat::default(); BINS],
        }
    }

    /// Records one prediction: the top-class `confidence` in `[0, 1]`, whether the top class
    /// was right and the item's multiclass Brier term (finite, non-negative). On error nothing
    /// changes.
    pub fn record(
        &mut self,
        confidence: f64,
        top_correct: bool,
        brier: f64,
    ) -> Result<(), CalibrationError> {
        if !(confidence.is_finite() && (0.0..=1.0).contains(&confidence)) {
            return Err(CalibrationError { value: confidence });
        }
        if !(brier.is_finite() && brier >= 0.0) {
            return Err(CalibrationError { value: brier });
        }
        let bin = ((confidence * BINS as f64) as usize).min(BINS - 1);
        let stat = &mut self.bins[bin];
        stat.n = stat.n.saturating_add(1);
        stat.confidence_sum += confidence;
        stat.correct = stat.correct.saturating_add(u64::from(top_correct));
        self.n = self.n.saturating_add(1);
        self.correct = self.correct.saturating_add(u64::from(top_correct));
        self.brier_sum += brier;
        Ok(())
    }

    /// Records a binary prediction `p_pass` against the realised `passed`. Class order is
    /// (fail, pass), the top class wins ties for pass, and the Brier term is the multiclass one
    /// (twice the binary Brier score).
    pub fn record_binary(&mut self, p_pass: f64, passed: bool) -> Result<(), CalibrationError> {
        if !(p_pass.is_finite() && (0.0..=1.0).contains(&p_pass)) {
            return Err(CalibrationError { value: p_pass });
        }
        let q_fail = 1.0 - p_pass;
        let (t_fail, t_pass) = if passed { (0.0, 1.0) } else { (1.0, 0.0) };
        let brier = (q_fail - t_fail) * (q_fail - t_fail) + (p_pass - t_pass) * (p_pass - t_pass);
        let predicted_pass = p_pass >= q_fail;
        let confidence = if predicted_pass { p_pass } else { q_fail };
        self.record(confidence, predicted_pass == passed, brier)
    }

    pub fn count(&self) -> u64 {
        self.n
    }

    pub fn report(&self) -> CalibrationReport {
        let n = self.n;
        let nf = n as f64;
        let mut ece = 0.0;
        let mut bins = Vec::new();
        for (index, stat) in self.bins.iter().enumerate() {
            if stat.n == 0 {
                continue;
            }
            let size = stat.n as f64;
            let confidence = stat.confidence_sum / size;
            let accuracy = stat.correct as f64 / size;
            ece += (size / nf) * (accuracy - confidence).abs();
            bins.push(BinReport {
                low: index as f64 / BINS as f64,
                high: (index + 1) as f64 / BINS as f64,
                n: stat.n,
                mean_confidence: confidence,
                accuracy,
            });
        }
        CalibrationReport {
            n,
            brier: (n > 0).then(|| self.brier_sum / nf),
            ece: (n > 0).then_some(ece),
            accuracy: wilson_interval(self.correct, n, Z).ok(),
            bins,
        }
    }

    pub(crate) fn write(&self, out: &mut DigestWriter) {
        out.u64(self.n);
        out.u64(self.correct);
        out.f64(self.brier_sum);
        for stat in &self.bins {
            out.u64(stat.n);
            out.f64(stat.confidence_sum);
            out.u64(stat.correct);
        }
    }
}

/// One non-empty calibration bin.
#[derive(Clone, Debug, PartialEq)]
pub struct BinReport {
    pub low: f64,
    pub high: f64,
    pub n: u64,
    pub mean_confidence: f64,
    pub accuracy: f64,
}

/// Calibration of one prediction head so far.
#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationReport {
    pub n: u64,
    /// Mean *multiclass* Brier score (sum over classes; twice the binary score for a binary
    /// head); `None` without predictions.
    pub brier: Option<f64>,
    /// *Top-label* expected calibration error over ten equal-width bins.
    pub ece: Option<f64>,
    /// Top-class accuracy with a 95% Wilson interval (valid for independent trials only).
    pub accuracy: Option<RateEstimate>,
    pub bins: Vec<BinReport>,
}

/// Class-conditional reliability of P(pass): per bin of the predicted probability, the observed
/// pass frequency.
#[derive(Clone, Debug, Default, PartialEq)]
struct Reliability {
    n: [u64; BINS],
    p_sum: [f64; BINS],
    pass: [u64; BINS],
}

impl Reliability {
    fn record(&mut self, p: f64, passed: bool) {
        let bin = ((p * BINS as f64) as usize).min(BINS - 1);
        self.n[bin] = self.n[bin].saturating_add(1);
        self.p_sum[bin] += p;
        self.pass[bin] = self.pass[bin].saturating_add(u64::from(passed));
    }

    fn report(&self) -> Vec<ReliabilityBin> {
        (0..BINS)
            .filter(|bin| self.n[*bin] > 0)
            .map(|bin| ReliabilityBin {
                low: bin as f64 / BINS as f64,
                high: (bin + 1) as f64 / BINS as f64,
                n: self.n[bin],
                mean_predicted: self.p_sum[bin] / self.n[bin] as f64,
                observed_pass_rate: self.pass[bin] as f64 / self.n[bin] as f64,
            })
            .collect()
    }

    fn write(&self, out: &mut DigestWriter) {
        for bin in 0..BINS {
            out.u64(self.n[bin]);
            out.f64(self.p_sum[bin]);
            out.u64(self.pass[bin]);
        }
    }
}

/// One non-empty bin of the P(pass) reliability table.
#[derive(Clone, Debug, PartialEq)]
pub struct ReliabilityBin {
    pub low: f64,
    pub high: f64,
    pub n: u64,
    pub mean_predicted: f64,
    pub observed_pass_rate: f64,
}

/// The calibration state of a [`SelfModel`], read in one consistent snapshot.
#[derive(Clone, Debug, PartialEq)]
pub struct SelfModelReport {
    pub version: u64,
    pub digest: [u8; 32],
    pub pass: CalibrationReport,
    /// Observed pass frequency per predicted-probability bin.
    pub pass_reliability: Vec<ReliabilityBin>,
    pub code: CalibrationReport,
    /// The route head, including the `Other` bucket as a class.
    pub route: CalibrationReport,
    pub contexts_tracked: usize,
    /// Observations that went to the shared overflow context.
    pub overflow_observations: u64,
    pub log_evicted: u64,
    pub repair_pairs_evicted: u64,
}

#[derive(Clone, Debug, Default)]
struct Stats {
    pass: u64,
    fail: u64,
    none: u64,
    other_codes: u64,
    codes: BTreeMap<FindingCode, u64>,
    routes_other: u64,
    routes: BTreeMap<PodRef, u64>,
}

impl Stats {
    fn n(&self) -> u64 {
        self.pass.saturating_add(self.fail)
    }

    fn code_total(&self) -> u64 {
        self.codes
            .values()
            .fold(self.none.saturating_add(self.other_codes), |a, c| {
                a.saturating_add(*c)
            })
    }

    fn route_total(&self) -> u64 {
        self.routes
            .values()
            .fold(self.routes_other, |a, c| a.saturating_add(*c))
    }

    fn write(&self, out: &mut DigestWriter) {
        out.u64(self.pass);
        out.u64(self.fail);
        out.u64(self.none);
        out.u64(self.other_codes);
        out.u64(self.codes.len() as u64);
        for (code, count) in &self.codes {
            out.str(code.as_str());
            out.u64(*count);
        }
        out.u64(self.routes_other);
        out.u64(self.routes.len() as u64);
        for (pod, count) in &self.routes {
            write_pod(out, pod);
            out.u64(*count);
        }
    }

    fn learn(&mut self, outcome: &SelfOutcome, max_codes: usize, max_pods: usize) {
        if outcome.passed {
            self.pass = self.pass.saturating_add(1);
        } else {
            self.fail = self.fail.saturating_add(1);
        }
        match &outcome.primary_code {
            None => self.none = self.none.saturating_add(1),
            Some(code) => {
                if let Some(count) = self.codes.get_mut(code) {
                    *count = count.saturating_add(1);
                } else if self.codes.len() < max_codes {
                    self.codes.insert(code.clone(), 1);
                } else {
                    self.other_codes = self.other_codes.saturating_add(1);
                }
            }
        }
        if let Some(pod) = &outcome.routed_pod {
            if let Some(count) = self.routes.get_mut(pod) {
                *count = count.saturating_add(1);
            } else if self.routes.len() < max_pods {
                self.routes.insert(pod.clone(), 1);
            } else {
                self.routes_other = self.routes_other.saturating_add(1);
            }
        }
    }
}

fn smoothed<K>(counts: Vec<(K, u64)>) -> Vec<(K, f64)> {
    let total = counts.iter().fold(0u64, |a, (_, c)| a.saturating_add(*c)) as f64;
    let denominator = total + ALPHA * counts.len() as f64;
    counts
        .into_iter()
        .map(|(key, count)| (key, (count as f64 + ALPHA) / denominator))
        .collect()
}

fn code_distribution(stats: &Stats, global: &Stats) -> Vec<(CodeOutcome, f64)> {
    let source = if stats.code_total() == 0 {
        global
    } else {
        stats
    };
    let mut counts: Vec<(CodeOutcome, u64)> = vec![(CodeOutcome::NoHardFinding, source.none)];
    counts.extend(
        source
            .codes
            .iter()
            .map(|(code, count)| (CodeOutcome::Code(code.clone()), *count)),
    );
    counts.push((CodeOutcome::Other, source.other_codes));
    smoothed(counts)
}

fn route_distribution(stats: &Stats, global: &Stats) -> Vec<(Option<PodRef>, f64)> {
    let source = if stats.route_total() == 0 {
        global
    } else {
        stats
    };
    if source.routes.is_empty() {
        return Vec::new();
    }
    let mut counts: Vec<(Option<PodRef>, u64)> = source
        .routes
        .iter()
        .map(|(pod, count)| (Some(pod.clone()), *count))
        .collect();
    counts.push((None, source.routes_other));
    smoothed(counts)
}

fn top<K: Clone>(distribution: &[(K, f64)]) -> Option<(K, f64)> {
    let mut best: Option<&(K, f64)> = None;
    for item in distribution {
        if best.is_none_or(|b| item.1 > b.1) {
            best = Some(item);
        }
    }
    best.cloned()
}

fn brier_term<K: PartialEq>(distribution: &[(K, f64)], realised: &K) -> f64 {
    distribution
        .iter()
        .map(|(key, q)| {
            let target = if key == realised { 1.0 } else { 0.0 };
            (q - target) * (q - target)
        })
        .sum()
}

/// Inverse-propensity totals of one (code, Pod, generation) pair.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct PairStat {
    /// Steps where the Pod was routed.
    n: u64,
    /// Sum of `pass / propensity`.
    ips_sum: f64,
    /// Sum of `(pass / propensity)^2`.
    ips_sq: f64,
    last_seq: u64,
}

/// Value estimate of "always use this Pod" for a code with `total` logged steps: the mean and a
/// 95% normal interval of the inverse-propensity terms.
fn ips_interval(stat: &PairStat, total: u64) -> (f64, f64, f64) {
    let n = total.max(1) as f64;
    let mean = stat.ips_sum / n;
    let variance = (stat.ips_sq / n - mean * mean).max(0.0);
    let se = (variance / n).sqrt();
    (
        mean.clamp(0.0, 1.0),
        (mean - Z * se).clamp(0.0, 1.0),
        (mean + Z * se).clamp(0.0, 1.0),
    )
}

/// Advisory tuning of [`SelfModel::hints`]. Non-finite thresholds disable the hints they gate;
/// a negative margin counts as zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HintPolicy {
    /// Emit [`SelfModelHint::EscalateEarly`] when the 95% Wilson upper bound of the context's
    /// observed pass rate is below this and the point prediction is too.
    pub escalate_below: f64,
    /// Observations a hint needs (per context, or per (code, Pod) pair and per code).
    pub min_evidence: u64,
    /// Required gap between the best Pod's lower bound and the runner-up's upper bound.
    pub prefer_margin: f64,
}

impl Default for HintPolicy {
    fn default() -> Self {
        Self {
            escalate_below: 0.2,
            min_evidence: 20,
            prefer_margin: 0.0,
        }
    }
}

/// Advice from the self-model. Plain data: it names no outcome class, holds no proof and has
/// no conversion into an acceptance, a gate, a threshold, a verifier or a registry change.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum SelfModelHint {
    /// Escalate early: the predicted pass probability is below `threshold` with `evidence`
    /// observations behind it.
    EscalateEarly {
        predicted_pass: f64,
        threshold: f64,
        evidence: u64,
    },
    /// Prefer `pod` (at `generation`) as the repair Pod for `code`: the lower bound of its
    /// inverse-propensity pass value exceeds the upper bound of every other Pod's.
    PreferRepairPod {
        code: FindingCode,
        pod: PodId,
        generation: Generation,
        value_lcb: f64,
        runner_up_ucb: f64,
        evidence: u64,
    },
}

impl SelfModelHint {
    /// The number of observations the hint rests on.
    pub fn evidence(&self) -> u64 {
        match self {
            Self::EscalateEarly { evidence, .. } | Self::PreferRepairPod { evidence, .. } => {
                *evidence
            }
        }
    }
}

/// A learner that predicts the pass probability from a [`SelfContext`] and learns from
/// [`SelfTraceStep`]s. [`SelfModel`] implements it, and so can an external probe: because both
/// see exactly the same context and step types, a comparison trains them on the same
/// information (the fair baseline of experiment C002).
pub trait TraceLearner {
    /// Predicts before the outcome is known. Must not learn.
    fn predict(&self, context: &SelfContext) -> SelfPrediction;
    /// Learns from one realised step.
    fn learn(&mut self, step: &SelfTraceStep);
}

/// Metrics of the pass head of one learner over a trace (prequential: predict, score, learn).
#[derive(Clone, Debug, PartialEq)]
pub struct PrequentialReport {
    pub n: u64,
    /// Observed pass rate over the whole trace (the reference predictor of the skill score).
    pub base_rate: f64,
    /// Binary Brier score (mean squared error of P(pass)).
    pub brier_binary: f64,
    /// `1 - brier / (base_rate * (1 - base_rate))`; `None` when the base rate is 0 or 1. A
    /// calibrated constant scores about 0, so unlike ECE this rewards resolution.
    pub brier_skill: Option<f64>,
    /// Mean negative log likelihood (diagnostic; uses `ln`, not part of any digest).
    pub log_loss: f64,
    /// Area under the ROC curve of P(pass); `None` when one class is absent.
    pub auroc: Option<f64>,
    /// Multiclass Brier, top-label ECE and Wilson accuracy.
    pub calibration: CalibrationReport,
}

/// Prequential evaluation of the pass head over `steps`: predict, score, then learn, step by
/// step. Run it for the self-model and for each probe on the same slice.
pub fn evaluate_prequential(
    learner: &mut dyn TraceLearner,
    steps: &[SelfTraceStep],
) -> Result<PrequentialReport, CalibrationError> {
    let mut tracker = CalibrationTracker::new();
    let mut scored: Vec<(f64, bool)> = Vec::with_capacity(steps.len());
    for step in steps {
        let prediction = learner.predict(&step.context);
        tracker.record_binary(prediction.pass_probability, step.outcome.passed)?;
        scored.push((prediction.pass_probability, step.outcome.passed));
        learner.learn(step);
    }
    let n = scored.len();
    let nf = n.max(1) as f64;
    let passes = scored.iter().filter(|(_, y)| *y).count();
    let base_rate = passes as f64 / nf;
    let brier_binary = scored
        .iter()
        .map(|(p, y)| {
            let t = if *y { 1.0 } else { 0.0 };
            (p - t) * (p - t)
        })
        .sum::<f64>()
        / nf;
    let reference = base_rate * (1.0 - base_rate);
    let log_loss = scored
        .iter()
        .map(|(p, y)| {
            let q = if *y { *p } else { 1.0 - *p };
            -q.clamp(1e-12, 1.0).ln()
        })
        .sum::<f64>()
        / nf;
    Ok(PrequentialReport {
        n: n as u64,
        base_rate,
        brier_binary,
        brier_skill: (reference > 0.0).then(|| 1.0 - brier_binary / reference),
        log_loss,
        auroc: auroc(&scored),
        calibration: tracker.report(),
    })
}

/// Area under the ROC curve with ties counted as half (Mann-Whitney).
fn auroc(scored: &[(f64, bool)]) -> Option<f64> {
    let positives = scored.iter().filter(|(_, y)| *y).count();
    let negatives = scored.len() - positives;
    if positives == 0 || negatives == 0 {
        return None;
    }
    let mut order: Vec<usize> = (0..scored.len()).collect();
    order.sort_by(|a, b| scored[*a].0.total_cmp(&scored[*b].0).then(a.cmp(b)));
    let mut rank_sum = 0.0;
    let mut i = 0;
    while i < order.len() {
        let mut j = i;
        while j + 1 < order.len() && scored[order[j + 1]].0 == scored[order[i]].0 {
            j += 1;
        }
        let average_rank = (i + j) as f64 / 2.0 + 1.0;
        for index in &order[i..=j] {
            if scored[*index].1 {
                rank_sum += average_rank;
            }
        }
        i = j + 1;
    }
    let p = positives as f64;
    Some((rank_sum - p * (p + 1.0) / 2.0) / (p * negatives as f64))
}

/// A copy of `steps` whose pass labels are permuted by a seeded shuffle: the negative control
/// of C002 (every learner must score about zero skill on it).
pub fn shuffled_labels(steps: &[SelfTraceStep], seed: u64) -> Vec<SelfTraceStep> {
    let mut labels: Vec<bool> = steps.iter().map(|s| s.outcome.passed).collect();
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    for i in (1..labels.len()).rev() {
        let j = (next() % (i as u64 + 1)) as usize;
        labels.swap(i, j);
    }
    steps
        .iter()
        .zip(labels)
        .map(|(step, passed)| {
            let mut step = step.clone();
            step.outcome.passed = passed;
            step
        })
        .collect()
}

fn pass_only(p: f64, evidence: u64) -> SelfPrediction {
    SelfPrediction {
        pass_probability: p.clamp(0.0, 1.0),
        evidence,
        code: None,
        route: None,
    }
}

/// A context-free baseline: the global Laplace-smoothed pass rate.
#[derive(Clone, Copy, Debug, Default)]
pub struct GlobalRateProbe {
    pass: u64,
    n: u64,
}

impl TraceLearner for GlobalRateProbe {
    fn predict(&self, _: &SelfContext) -> SelfPrediction {
        pass_only((self.pass as f64 + 1.0) / (self.n as f64 + 2.0), self.n)
    }

    fn learn(&mut self, step: &SelfTraceStep) {
        self.n = self.n.saturating_add(1);
        self.pass = self.pass.saturating_add(u64::from(step.outcome.passed));
    }
}

/// An equal-capacity probe on the raw features: a lookup table keyed by the exact (unbucketed)
/// context, with the same Laplace smoothing and the global rate as backoff, capped at
/// `capacity` entries.
#[derive(Clone, Debug)]
pub struct LookupTableProbe {
    capacity: usize,
    table: BTreeMap<String, (u64, u64)>,
    global: GlobalRateProbe,
}

impl LookupTableProbe {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            table: BTreeMap::new(),
            global: GlobalRateProbe::default(),
        }
    }

    fn key(context: &SelfContext) -> String {
        let mut codes: Vec<&str> = context.last_codes.iter().map(FindingCode::as_str).collect();
        codes.sort_unstable();
        codes.dedup();
        let producer = context
            .producer
            .as_ref()
            .map_or(String::new(), |p| format!("{}@{}", p.pod.0, p.generation.0));
        format!(
            "{:?}|{producer}|{}|{}|{}|{}",
            context.role,
            context.round,
            context.budget_left,
            level_rank(context.required),
            codes.join(",")
        )
    }
}

impl TraceLearner for LookupTableProbe {
    fn predict(&self, context: &SelfContext) -> SelfPrediction {
        let global = (self.global.pass as f64 + 1.0) / (self.global.n as f64 + 2.0);
        match self.table.get(&Self::key(context)) {
            Some((pass, n)) => pass_only(
                (*pass as f64 + SHRINKAGE * global) / (*n as f64 + SHRINKAGE),
                *n,
            ),
            None => pass_only(global, 0),
        }
    }

    fn learn(&mut self, step: &SelfTraceStep) {
        self.global.learn(step);
        let key = Self::key(&step.context);
        if let Some(entry) = self.table.get_mut(&key) {
            entry.0 = entry.0.saturating_add(u64::from(step.outcome.passed));
            entry.1 = entry.1.saturating_add(1);
        } else if self.table.len() < self.capacity {
            self.table.insert(key, (u64::from(step.outcome.passed), 1));
        }
    }
}

/// An equal-information linear probe: hashed one-hot features of the raw context, trained online
/// by squared-loss gradient steps through a soft-sign link (`p = 0.5 + 0.5 z / (1 + |z|)`,
/// exp-free so training is bit-exact across platforms).
#[derive(Clone, Debug)]
pub struct LinearProbe {
    weights: Vec<f64>,
    learning_rate: f64,
}

impl LinearProbe {
    /// `dimensions` hashed feature slots (at least 1) and a fixed `learning_rate`.
    pub fn new(dimensions: usize, learning_rate: f64) -> Self {
        Self {
            weights: vec![0.0; dimensions.max(1) + 1],
            learning_rate: if learning_rate.is_finite() {
                learning_rate.clamp(0.0, 1.0)
            } else {
                0.0
            },
        }
    }

    fn features(&self, context: &SelfContext) -> Vec<usize> {
        let dims = self.weights.len() - 1;
        let mut names = vec![
            format!("role:{:?}", context.role),
            format!("round:{}", context.round),
            format!("budget:{}", context.budget_left),
            format!("required:{}", level_rank(context.required)),
        ];
        if let Some(p) = &context.producer {
            names.push(format!("producer:{}@{}", p.pod.0, p.generation.0));
        }
        for code in &context.last_codes {
            names.push(format!("code:{}", code.as_str()));
        }
        let mut slots: Vec<usize> = names
            .iter()
            .map(|name| (fnv(name.as_bytes()) % dims as u64) as usize)
            .collect();
        slots.sort_unstable();
        slots.dedup();
        slots
    }

    fn score(&self, slots: &[usize]) -> f64 {
        let z = slots
            .iter()
            .fold(self.weights[self.weights.len() - 1], |a, s| {
                a + self.weights[*s]
            });
        0.5 + 0.5 * z / (1.0 + z.abs())
    }
}

impl TraceLearner for LinearProbe {
    fn predict(&self, context: &SelfContext) -> SelfPrediction {
        pass_only(self.score(&self.features(context)), 0)
    }

    fn learn(&mut self, step: &SelfTraceStep) {
        let slots = self.features(&step.context);
        let last = self.weights.len() - 1;
        let z = slots
            .iter()
            .fold(self.weights[last], |a, s| a + self.weights[*s]);
        let p = 0.5 + 0.5 * z / (1.0 + z.abs());
        let target = if step.outcome.passed { 1.0 } else { 0.0 };
        let slope = 0.5 / ((1.0 + z.abs()) * (1.0 + z.abs()));
        let delta = self.learning_rate * 2.0 * (target - p) * slope;
        for slot in slots {
            self.weights[slot] += delta;
        }
        self.weights[last] += delta;
    }
}

/// The self-model. See the module header.
#[derive(Clone, Debug)]
pub struct SelfModel {
    limits: SelfModelLimits,
    mask: FeatureMask,
    version: u64,
    seq: u64,
    global: Stats,
    empty: Stats,
    contexts: BTreeMap<ContextKey, Stats>,
    overflow: Stats,
    overflow_observations: u64,
    pairs: BTreeMap<(FindingCode, PodRef), PairStat>,
    pairs_evicted: u64,
    code_totals: BTreeMap<FindingCode, u64>,
    pass_cal: CalibrationTracker,
    pass_rel: Reliability,
    code_cal: CalibrationTracker,
    route_cal: CalibrationTracker,
    log: VecDeque<LoggedPrediction>,
    log_evicted: u64,
}

impl SelfModel {
    pub fn new(limits: SelfModelLimits) -> Self {
        Self::with_features(limits, FeatureMask::ALL)
    }

    /// A model that ignores the features switched off in `mask` (the C002 feature ablation).
    pub fn with_features(limits: SelfModelLimits, mask: FeatureMask) -> Self {
        Self {
            limits,
            mask,
            version: 0,
            seq: 0,
            global: Stats::default(),
            empty: Stats::default(),
            contexts: BTreeMap::new(),
            overflow: Stats::default(),
            overflow_observations: 0,
            pairs: BTreeMap::new(),
            pairs_evicted: 0,
            code_totals: BTreeMap::new(),
            pass_cal: CalibrationTracker::new(),
            pass_rel: Reliability::default(),
            code_cal: CalibrationTracker::new(),
            route_cal: CalibrationTracker::new(),
            log: VecDeque::new(),
            log_evicted: 0,
        }
    }

    pub fn limits(&self) -> &SelfModelLimits {
        &self.limits
    }

    /// The number of observed steps; changes with every observation.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Number of tracked contexts (at most [`SelfModelLimits::max_contexts`]).
    pub fn contexts_tracked(&self) -> usize {
        self.contexts.len()
    }

    /// Tracked (code, Pod, generation) repair pairs (at most
    /// [`SelfModelLimits::max_repair_pairs`]).
    pub fn repair_pairs_tracked(&self) -> usize {
        self.pairs.len()
    }

    /// Repair pairs evicted (or refused with zero capacity) so far.
    pub fn repair_pairs_evicted(&self) -> u64 {
        self.pairs_evicted
    }

    /// The most recent predictions with their realised outcomes, oldest first.
    pub fn log(&self) -> impl Iterator<Item = &LoggedPrediction> {
        self.log.iter()
    }

    /// Predictions dropped from the ring log.
    pub fn log_evicted(&self) -> u64 {
        self.log_evicted
    }

    /// The inverse-propensity value estimate of using `pod` for `code`, as
    /// `(estimate, lower bound, upper bound, pair observations)`; `None` without evidence.
    pub fn repair_value(&self, code: &FindingCode, pod: &PodRef) -> Option<(f64, f64, f64, u64)> {
        let code = bound_code(code, &self.limits);
        let pod = bound_pod(pod, &self.limits);
        let total = *self.code_totals.get(&code)?;
        let stat = self.pairs.get(&(code, pod))?;
        let (value, low, high) = ips_interval(stat, total);
        Some((value, low, high, stat.n))
    }

    fn stats_for(&self, key: &ContextKey) -> &Stats {
        match self.contexts.get(key) {
            Some(stats) => stats,
            None if self.contexts.len() < self.limits.max_contexts => &self.empty,
            None => &self.overflow,
        }
    }

    /// Pass estimate shrunk towards the rate of all *other* observations, so a context's own
    /// data is not counted twice.
    fn pass_estimate(&self, stats: &Stats) -> f64 {
        let other_pass = self.global.pass.saturating_sub(stats.pass) as f64;
        let other_n = self.global.n().saturating_sub(stats.n()) as f64;
        let prior = (other_pass + 1.0) / (other_n + 2.0);
        (stats.pass as f64 + SHRINKAGE * prior) / (stats.n() as f64 + SHRINKAGE)
    }

    /// Predicts for `context` from the current state. Pure.
    pub fn predict(&self, context: &SelfContext) -> SelfPrediction {
        let key = ContextKey::new(context, &self.limits, &self.mask);
        let stats = self.stats_for(&key);
        SelfPrediction {
            pass_probability: self.pass_estimate(stats),
            evidence: stats.n(),
            code: top(&code_distribution(stats, &self.global)),
            route: top(&route_distribution(stats, &self.global))
                .and_then(|(pod, p)| pod.map(|pod| (pod, p))),
        }
    }

    /// The full finding-code distribution for `context`: `NoHardFinding`, tracked codes in
    /// order and `Other`. Sums to 1.
    pub fn code_distribution(&self, context: &SelfContext) -> Vec<(CodeOutcome, f64)> {
        let key = ContextKey::new(context, &self.limits, &self.mask);
        code_distribution(self.stats_for(&key), &self.global)
    }

    /// Advisory hints for `context`. Pure; the hints do nothing by themselves.
    pub fn hints(&self, context: &SelfContext, policy: &HintPolicy) -> Vec<SelfModelHint> {
        let mut hints = Vec::new();
        let key = ContextKey::new(context, &self.limits, &self.mask);
        let stats = self.stats_for(&key);
        let prediction = self.predict(context);
        let n = stats.n();
        if n >= policy.min_evidence && prediction.pass_probability < policy.escalate_below {
            let upper = wilson_interval(stats.pass, n, Z).map_or(1.0, |r| r.high);
            if upper < policy.escalate_below {
                hints.push(SelfModelHint::EscalateEarly {
                    predicted_pass: prediction.pass_probability,
                    threshold: policy.escalate_below,
                    evidence: n,
                });
            }
        }
        let margin = if policy.prefer_margin > 0.0 {
            policy.prefer_margin
        } else {
            0.0
        };
        let (codes, _) = canonical_codes(&context.last_codes, &self.limits);
        for code in codes {
            let Some(total) = self.code_totals.get(&code).copied() else {
                continue;
            };
            if total < policy.min_evidence {
                continue;
            }
            // (pod, value, lcb, ucb, n) of every pair with enough evidence.
            let mut ranked: Vec<(&PodRef, f64, f64, f64, u64)> = self
                .pairs
                .iter()
                .filter(|((c, _), stat)| *c == code && stat.n >= policy.min_evidence)
                .map(|((_, pod), stat)| {
                    let (value, low, high) = ips_interval(stat, total);
                    (pod, value, low, high, stat.n)
                })
                .collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
            if ranked.len() < 2 {
                continue;
            }
            let best = ranked[0];
            let runner_up_ucb = ranked[1..]
                .iter()
                .map(|item| item.3)
                .fold(0.0_f64, f64::max);
            if best.2 - runner_up_ucb > margin {
                hints.push(SelfModelHint::PreferRepairPod {
                    code,
                    pod: best.0.pod.clone(),
                    generation: best.0.generation,
                    value_lcb: best.2,
                    runner_up_ucb,
                    evidence: best.4,
                });
            }
        }
        hints
    }

    fn bound_outcome(&self, outcome: &SelfOutcome) -> SelfOutcome {
        SelfOutcome {
            passed: outcome.passed,
            primary_code: outcome
                .primary_code
                .as_ref()
                .map(|code| bound_code(code, &self.limits)),
            routed_pod: outcome
                .routed_pod
                .as_ref()
                .map(|pod| bound_pod(pod, &self.limits)),
            propensity: outcome.propensity,
        }
    }

    /// Predicts for the step's context, logs the prediction with the realised outcome, scores
    /// it, and only then learns from it. Returns the prediction.
    pub fn observe(&mut self, step: &SelfTraceStep) -> SelfPrediction {
        let key = ContextKey::new(&step.context, &self.limits, &self.mask);
        let prediction = self.predict(&step.context);
        let outcome = self.bound_outcome(&step.outcome);

        // Score the three heads against the realised outcome (all probabilities are valid by
        // construction, so the trackers cannot refuse).
        let _ = self
            .pass_cal
            .record_binary(prediction.pass_probability, outcome.passed);
        self.pass_rel
            .record(prediction.pass_probability, outcome.passed);
        let (code_dist, route_dist) = {
            let stats = self.stats_for(&key);
            (
                code_distribution(stats, &self.global),
                route_distribution(stats, &self.global),
            )
        };
        let realised_code = match &outcome.primary_code {
            None => CodeOutcome::NoHardFinding,
            Some(code) => {
                let tracked = CodeOutcome::Code(code.clone());
                if code_dist.iter().any(|(o, _)| *o == tracked) {
                    tracked
                } else {
                    CodeOutcome::Other
                }
            }
        };
        if let Some((top_code, confidence)) = top(&code_dist) {
            let _ = self.code_cal.record(
                confidence,
                top_code == realised_code,
                brier_term(&code_dist, &realised_code),
            );
        }
        // The route head scores every class, `Other` included, so skipping the bucket does not
        // bias its calibration.
        if let (Some((top_class, confidence)), Some(pod)) =
            (top(&route_dist), outcome.routed_pod.as_ref())
        {
            let realised = route_dist
                .iter()
                .any(|(p, _)| p.as_ref() == Some(pod))
                .then(|| pod.clone());
            let _ = self.route_cal.record(
                confidence,
                realised == top_class,
                brier_term(&route_dist, &realised),
            );
        }

        let entry = LoggedPrediction {
            seq: self.seq,
            version: self.version,
            context_digest: key.digest(),
            prediction: prediction.clone(),
            outcome: outcome.clone(),
        };
        if self.limits.max_log == 0 {
            self.log_evicted = self.log_evicted.saturating_add(1);
        } else {
            while self.log.len() >= self.limits.max_log {
                self.log.pop_front();
                self.log_evicted = self.log_evicted.saturating_add(1);
            }
            self.log.push_back(entry);
        }

        // Learn.
        let (max_codes, max_pods) = (
            self.limits.max_codes_per_context,
            self.limits.max_pods_per_context,
        );
        self.global.learn(&outcome, max_codes, max_pods);
        if self.contexts.contains_key(&key) || self.contexts.len() < self.limits.max_contexts {
            self.contexts
                .entry(key)
                .or_default()
                .learn(&outcome, max_codes, max_pods);
        } else {
            self.overflow_observations = self.overflow_observations.saturating_add(1);
            self.overflow.learn(&outcome, max_codes, max_pods);
        }
        if step.context.role == SelfRole::Repair {
            self.learn_repair(&step.context, &outcome);
        }
        self.version = self.version.saturating_add(1);
        self.seq = self.seq.saturating_add(1);
        prediction
    }

    /// Credits the routed Pod, for every canonical code of the context, with the inverse
    /// propensity of the pass outcome. Needs a routed Pod and a logged propensity.
    fn learn_repair(&mut self, context: &SelfContext, outcome: &SelfOutcome) {
        let (Some(pod), Some(propensity)) = (&outcome.routed_pod, outcome.propensity) else {
            return;
        };
        let weight = 1.0 / propensity.get();
        let term = if outcome.passed { weight } else { 0.0 };
        let (codes, _) = canonical_codes(&context.last_codes, &self.limits);
        for code in codes {
            self.bump_code_total(&code);
            let pair = (code, pod.clone());
            if !self.pairs.contains_key(&pair) {
                if self.limits.max_repair_pairs == 0 {
                    self.pairs_evicted = self.pairs_evicted.saturating_add(1);
                    continue;
                }
                if self.pairs.len() >= self.limits.max_repair_pairs {
                    let victim = self
                        .pairs
                        .iter()
                        .min_by(|a, b| {
                            (a.1.n >= PROBATION, a.1.n, a.1.last_seq, a.0).cmp(&(
                                b.1.n >= PROBATION,
                                b.1.n,
                                b.1.last_seq,
                                b.0,
                            ))
                        })
                        .map(|(key, _)| key.clone());
                    if let Some(victim) = victim {
                        self.pairs.remove(&victim);
                        self.pairs_evicted = self.pairs_evicted.saturating_add(1);
                    }
                }
            }
            let stat = self.pairs.entry(pair).or_default();
            stat.n = stat.n.saturating_add(1);
            stat.ips_sum += term;
            stat.ips_sq += term * term;
            stat.last_seq = self.seq;
        }
    }

    fn bump_code_total(&mut self, code: &FindingCode) {
        if let Some(total) = self.code_totals.get_mut(code) {
            *total = total.saturating_add(1);
            return;
        }
        if self.limits.max_repair_codes == 0 {
            return;
        }
        if self.code_totals.len() >= self.limits.max_repair_codes {
            let victim = self
                .code_totals
                .iter()
                .min_by(|a, b| (a.1, a.0).cmp(&(b.1, b.0)))
                .map(|(code, _)| code.clone());
            if let Some(victim) = victim {
                self.code_totals.remove(&victim);
            }
        }
        self.code_totals.insert(code.clone(), 1);
    }

    /// A SHA-256 digest over the version, limits, feature mask, every table, the calibration
    /// state and the prediction log. Equal traces give equal digests.
    pub fn digest(&self) -> [u8; 32] {
        let mut out = DigestWriter::new("ptr-router/self-model/v2");
        for value in [
            self.limits.max_contexts,
            self.limits.max_codes_per_context,
            self.limits.max_pods_per_context,
            self.limits.max_repair_pairs,
            self.limits.max_repair_codes,
            self.limits.max_context_codes,
            self.limits.max_code_len,
            self.limits.max_log,
        ] {
            out.u64(value as u64);
        }
        for flag in [
            self.mask.role,
            self.mask.pod,
            self.mask.round,
            self.mask.budget,
            self.mask.codes,
            self.mask.required,
        ] {
            out.u64(u64::from(flag));
        }
        out.u64(self.version);
        out.u64(self.seq);
        self.global.write(&mut out);
        out.u64(self.contexts.len() as u64);
        for (key, stats) in &self.contexts {
            key.write(&mut out);
            stats.write(&mut out);
        }
        self.overflow.write(&mut out);
        out.u64(self.overflow_observations);
        out.u64(self.pairs.len() as u64);
        for ((code, pod), stat) in &self.pairs {
            out.str(code.as_str());
            write_pod(&mut out, pod);
            out.u64(stat.n);
            out.f64(stat.ips_sum);
            out.f64(stat.ips_sq);
            out.u64(stat.last_seq);
        }
        out.u64(self.pairs_evicted);
        out.u64(self.code_totals.len() as u64);
        for (code, total) in &self.code_totals {
            out.str(code.as_str());
            out.u64(*total);
        }
        self.pass_cal.write(&mut out);
        self.pass_rel.write(&mut out);
        self.code_cal.write(&mut out);
        self.route_cal.write(&mut out);
        out.u64(self.log_evicted);
        out.u64(self.log.len() as u64);
        for entry in &self.log {
            out.u64(entry.seq);
            out.u64(entry.version);
            out.f64(entry.prediction.pass_probability);
            out.u64(u64::from(entry.outcome.passed));
        }
        out.finish()
    }

    /// Calibration of the three heads and the table sizes (one consistent snapshot, since the
    /// model is borrowed for the whole call).
    pub fn report(&self) -> SelfModelReport {
        SelfModelReport {
            version: self.version,
            digest: self.digest(),
            pass: self.pass_cal.report(),
            pass_reliability: self.pass_rel.report(),
            code: self.code_cal.report(),
            route: self.route_cal.report(),
            contexts_tracked: self.contexts.len(),
            overflow_observations: self.overflow_observations,
            log_evicted: self.log_evicted,
            repair_pairs_evicted: self.pairs_evicted,
        }
    }
}

impl TraceLearner for SelfModel {
    fn predict(&self, context: &SelfContext) -> SelfPrediction {
        SelfModel::predict(self, context)
    }

    fn learn(&mut self, step: &SelfTraceStep) {
        self.observe(step);
    }
}

/// Length-prefixed canonical encoding for digests.
pub(crate) struct DigestWriter(Sha256);

impl DigestWriter {
    pub(crate) fn new(domain: &str) -> Self {
        let mut out = Self(Sha256::new());
        out.str(domain);
        out
    }

    pub(crate) fn u64(&mut self, value: u64) {
        self.0.update(value.to_be_bytes());
    }

    pub(crate) fn f64(&mut self, value: f64) {
        self.0.update(value.to_bits().to_be_bytes());
    }

    pub(crate) fn str(&mut self, value: &str) {
        self.u64(value.len() as u64);
        self.0.update(value.as_bytes());
    }

    pub(crate) fn finish(self) -> [u8; 32] {
        self.0.finalize().into()
    }
}
