//! The finding router of `ptr-router`, with its propensity log and the off-policy estimators.
//!
//! The router is built once from a seed and a roster and cannot be reconfigured from
//! Python: there is no registry update, no scorer, no fallback policy, no limit and no
//! eligibility handle that outlives a call. A report can only be one a verifier of
//! `ptr_py` produced, and a reward can only come from `classify` on such a report
//! (`VerifierReward` has no public constructor), so Python can log propensities and
//! estimate policies but cannot make a decision accept, or forge a reward.

use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard};

use ptr_router::{
    classify, estimate_policy, AdmissibleScorer, CodeRoute, EligibilitySnapshot, EscalationReason,
    Exploration, FallbackReason, FindingCode, FindingRegistry, FindingRouteRecord,
    FindingRouteRequest, FindingRouter, LoggedDecision, NonceSequence, PolicyEstimate,
    RegistryChange, RegistryUpdate, ReportClass, RewardModel, RouteClass, RouteDistribution,
    RouteError, RouteOutcome, RouterConfig, RouterLimits, SamplingKey, ScoringContext,
    UniformScorer, VerifierReward, MAX_LOG,
};
use ptr_types::{Generation, PodId, RequestId, VerificationLevel};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sha2::{Digest as _, Sha256};

use crate::convert::{hex, PtrRouterError};
use crate::verify::{PyReport, TaskSpec};

/// The level every accepted candidate must reach. It is the level of the deterministic
/// verifiers and is not a parameter of any function: it cannot be lowered from Python.
const REQUIRED: VerificationLevel = VerificationLevel::Deterministic;

/// Decisions that may wait for a verdict at once.
const MAX_PENDING: usize = 65_536;

pub(crate) fn fallback_name(reason: FallbackReason) -> &'static str {
    match reason {
        FallbackReason::Unknown => "unknown",
        FallbackReason::Disputed => "disputed",
        FallbackReason::FailWithoutHardFinding => "fail_without_hard_finding",
        FallbackReason::ForgedPass => "forged_pass",
        FallbackReason::BelowLevel => "below_level",
        FallbackReason::TooManyCodes => "too_many_codes",
        FallbackReason::OversizedCode => "oversized_code",
    }
}

pub(crate) fn escalation_names(reason: EscalationReason) -> (&'static str, &'static str) {
    match reason {
        EscalationReason::Policy(reason) => ("policy", fallback_name(reason)),
        EscalationReason::NoProposer(reason) => ("no_proposer", fallback_name(reason)),
    }
}

/// The posterior label of a route class: `repair/<code>` or `regenerate`.
fn class_label(class: &RouteClass) -> String {
    match class {
        RouteClass::Repair(code) => format!("repair/{}", code.as_str()),
        RouteClass::Regenerate(_) => "regenerate".to_owned(),
    }
}

/// One sampled routing record as a dict. `propensity` is the sampled Pod's interval width
/// over `2**53`, and equals its entry in `support`.
pub(crate) fn record_dict<'py>(
    py: Python<'py>,
    record: &FindingRouteRecord,
    decision_id: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    match record.class() {
        RouteClass::Repair(code) => {
            out.set_item("class", "repair")?;
            out.set_item("code", code.as_str())?;
        }
        RouteClass::Regenerate(reason) => {
            out.set_item("class", "regenerate")?;
            out.set_item("reason", fallback_name(*reason))?;
        }
    }
    out.set_item("label", class_label(record.class()))?;
    out.set_item("pod", &record.pod().0)?;
    out.set_item("pod_generation", record.pod_generation().0)?;
    out.set_item("propensity", record.propensity().get())?;
    let support = PyList::empty(py);
    let probabilities = PyList::empty(py);
    let total = RouteDistribution::RESOLUTION as f64;
    for (pod, width) in record.support() {
        support.append((&pod.0, *width))?;
        probabilities.append((&pod.0, *width as f64 / total))?;
    }
    out.set_item("support", support)?;
    out.set_item("support_probabilities", probabilities)?;
    out.set_item("support_total", RouteDistribution::RESOLUTION)?;
    out.set_item("draw", record.draw())?;
    out.set_item("nonce", hex(record.nonce().as_bytes()))?;
    out.set_item("key_id", hex(&record.key_id()))?;
    out.set_item("registry_generation", record.generation().get())?;
    out.set_item("request_id", &record.request_id().0)?;
    out.set_item("decision_id", decision_id)?;
    Ok(out)
}

/// Classify a report against a required level: the deterministic part of routing.
///
/// Returns `{"class": "accept"}`, `{"class": "repair", "codes": [...]}` (distinct hard
/// codes in report order) or `{"class": "fallback", "reason": ...}`. Only a `pass` report
/// with no hard finding at the `deterministic` level is `accept`; unknown, disputed, a
/// forged pass and a below-level pass never are. The level is fixed and not a parameter.
/// Read-only.
#[pyfunction]
pub(crate) fn classify_report<'py>(
    py: Python<'py>,
    report: &PyReport,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    match classify(&report.inner, REQUIRED, &RouterLimits::default()) {
        ReportClass::Accept(_) => out.set_item("class", "accept")?,
        ReportClass::Repair(codes) => {
            out.set_item("class", "repair")?;
            out.set_item(
                "codes",
                codes
                    .iter()
                    .map(|code| code.to_string())
                    .collect::<Vec<_>>(),
            )?;
        }
        ReportClass::Fallback(reason) => {
            out.set_item("class", "fallback")?;
            out.set_item("reason", fallback_name(reason))?;
        }
    }
    Ok(out)
}

/// A target policy given as a table. Pure: it ignores the request fields, as the
/// estimators require. A class absent from the table is uniform over the admissible set; a
/// Pod absent from its class has weight zero.
struct TableScorer {
    weights: BTreeMap<String, BTreeMap<String, f64>>,
}

impl AdmissibleScorer for TableScorer {
    fn bound_exploration(&self) -> Option<Exploration> {
        None
    }

    fn weights(&self, context: &ScoringContext<'_>) -> Result<Vec<(PodId, f64)>, RouteError> {
        let table = self.weights.get(&class_label(context.class));
        Ok(context
            .admissible
            .iter()
            .map(|pod| {
                let weight = table.map_or(1.0, |row| row.get(&pod.0).copied().unwrap_or(0.0));
                (pod.clone(), weight)
            })
            .collect())
    }
}

/// A reward model given as a table with a default.
struct TableReward {
    predictions: BTreeMap<String, BTreeMap<String, f64>>,
    default: f64,
}

impl RewardModel for TableReward {
    fn predict(&self, class: &RouteClass, pod: &PodId) -> f64 {
        self.predictions
            .get(&class_label(class))
            .and_then(|row| row.get(&pod.0))
            .copied()
            .unwrap_or(self.default)
    }
}

fn estimate_dict<'py>(py: Python<'py>, estimate: &PolicyEstimate) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("n", estimate.n)?;
    out.set_item("ips", estimate.ips)?;
    out.set_item("snips", estimate.snips)?;
    out.set_item("doubly_robust", estimate.doubly_robust)?;
    out.set_item("se_ips", estimate.se_ips)?;
    out.set_item("se_snips", estimate.se_snips)?;
    out.set_item("se_doubly_robust", estimate.se_doubly_robust)?;
    out.set_item("effective_sample_size", estimate.effective_sample_size)?;
    Ok(out)
}

/// A routed decision waiting for its candidate: the record and the task the routed report
/// was verified against.
struct Pending {
    record: FindingRouteRecord,
    spec: TaskSpec,
}

struct State {
    /// Unsettled decisions by id. A settled decision is removed, so the cap counts only
    /// decisions that still await a candidate; ids are never reused.
    pending: BTreeMap<usize, Pending>,
    next_id: usize,
    log: Vec<LoggedDecision>,
    /// SHA-256 of the verifier report and of the candidate behind each logged reward, in log
    /// order.
    digests: Vec<([u8; 32], [u8; 32])>,
}

/// A digest of everything a report holds, so a logged reward can be traced to its report.
fn report_digest(report: &ptr_verifier::VerificationReport) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(format!(
        "{:?}|{:?}|{}",
        report.status,
        report.level,
        report.score.get().to_bits()
    ));
    for finding in &report.findings {
        hasher.update(format!(
            "|{}:{}:{}",
            finding.code, finding.hard, finding.message
        ));
    }
    hasher.finalize().into()
}

/// A finding router over a fixed roster, with a seeded sampling key and a propensity log.
///
/// `FindingRouter(seed, proposers, repairers, *, exploration=0.05, instance=0, generations=None)`:
/// `proposers` is a list of Pod ids, `repairers` a list of `(pod, [codes])`. The seed
/// derives the sampling key, so the same seed, roster, report sequence and `instance`
/// give the same draws; it is a reproducibility key for training logs, not a secret. The
/// scorer is uniform and fixed. `exploration` is the uniform floor mixed into every
/// distribution, in `[0, 1)`. A positive floor is what makes a log usable off-policy.
///
/// `generations` maps a registered Pod to its generation (default 1, never 0); it is
/// validated here and fixed. The roster, eligibility, generations, scorer, fallback policy,
/// required level and limits cannot be changed after construction, and no `route` call can
/// introduce a Pod or a generation.
#[pyclass(frozen, name = "FindingRouter", module = "ptr_py")]
pub(crate) struct PyFindingRouter {
    router: FindingRouter<UniformScorer>,
    exploration: Exploration,
    registered: Vec<String>,
    eligibility: EligibilitySnapshot,
    state: Mutex<State>,
}

impl PyFindingRouter {
    fn state(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// The most decisions routing `report` can leave pending: one per distinct hard code that has a
/// registered repair Pod, one proposer for a fallback when a proposer is registered, none for an
/// accept. Codes and fallbacks no Pod can take come back `Unroutable` or `Escalate` and leave
/// nothing pending, so they never count against the capacity.
fn max_records(registry: &FindingRegistry, report: &ptr_verifier::VerificationReport) -> usize {
    match classify(report, REQUIRED, &RouterLimits::default()) {
        ReportClass::Accept(_) => 0,
        ReportClass::Repair(codes) => codes
            .iter()
            .filter(|code| !registry.repair_pods(code).is_empty())
            .count(),
        ReportClass::Fallback(_) => usize::from(!registry.proposers().is_empty()),
    }
}

fn router_error(error: impl std::fmt::Display) -> PyErr {
    PtrRouterError::new_err(error.to_string())
}

#[pymethods]
impl PyFindingRouter {
    #[new]
    #[pyo3(signature = (seed, proposers, repairers, *, exploration=0.05, instance=0, generations=None))]
    fn new(
        seed: u64,
        proposers: Vec<String>,
        repairers: Vec<(String, Vec<String>)>,
        exploration: f64,
        instance: u64,
        generations: Option<BTreeMap<String, u64>>,
    ) -> PyResult<Self> {
        let exploration = Exploration::new(exploration).map_err(router_error)?;
        let mut registry = FindingRegistry::new();
        let mut registered: Vec<String> = Vec::new();
        let mut apply = |change: RegistryChange| -> PyResult<()> {
            let expected = registry.generation();
            registry
                .apply(RegistryUpdate { expected, change })
                .map(|_| ())
                .map_err(router_error)
        };
        for pod in &proposers {
            apply(RegistryChange::AddProposer {
                pod: PodId::from(pod.as_str()),
            })?;
            if !registered.contains(pod) {
                registered.push(pod.clone());
            }
        }
        for (pod, codes) in &repairers {
            for code in codes {
                apply(RegistryChange::AddRepair {
                    code: FindingCode::new(code.clone()),
                    pod: PodId::from(pod.as_str()),
                })?;
            }
            if !registered.contains(pod) {
                registered.push(pod.clone());
            }
        }
        let generations = generations.unwrap_or_default();
        if let Some(unknown) = generations.keys().find(|pod| !registered.contains(pod)) {
            return Err(PtrRouterError::new_err(format!(
                "generations names {unknown:?}, which is not a registered Pod"
            )));
        }
        if generations.values().any(|generation| *generation == 0) {
            return Err(PtrRouterError::new_err(
                "a Pod generation must be at least 1",
            ));
        }
        let mut eligibility = EligibilitySnapshot::new();
        for pod in &registered {
            let generation = generations.get(pod).copied().unwrap_or(1);
            eligibility.set(PodId::from(pod.as_str()), Generation(generation));
        }
        let mut key = [0u8; 32];
        for chunk in key.chunks_mut(8) {
            chunk.copy_from_slice(&seed.to_le_bytes());
        }
        let router = FindingRouter::new(
            registry,
            RouterConfig {
                exploration,
                ..RouterConfig::default()
            },
            UniformScorer,
            SamplingKey::from_bytes(key),
            NonceSequence::starting_at(instance.to_le_bytes(), 0),
        )
        .map_err(|error| PtrRouterError::new_err(format!("{error:?}")))?;
        Ok(Self {
            router,
            exploration,
            registered,
            eligibility,
            state: Mutex::new(State {
                pending: BTreeMap::new(),
                next_id: 0,
                log: Vec::new(),
                digests: Vec::new(),
            }),
        })
    }

    /// Route one verification report over the roster.
    ///
    /// Returns a dict with `kind`: `"accept"`; `"repair"` with `routes`, one entry per
    /// distinct hard code (a route dict, or `{"unroutable": code}` when no eligible repair
    /// Pod exists); `"regenerate"` with `reason` and `route`; or `"escalate"` with
    /// `reason` and `cause`. A route dict has `pod`, `propensity` (the sampled Pod's
    /// support width over `2**53`; equal to its `support` entry), `support`,
    /// `support_probabilities`, `draw`, `nonce` and `decision_id`, which `settle` takes.
    ///
    /// The roster, eligibility and generations are those fixed at construction, and the
    /// required level is `deterministic`; none is a parameter. Only the verifier decides
    /// the class.
    ///
    /// Raises PtrRouterError when routing fails (it is then not an accept) and when too
    /// many decisions await a candidate.
    #[pyo3(signature = (report, *, request_id))]
    fn route<'py>(
        &self,
        py: Python<'py>,
        report: &PyReport,
        request_id: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let id = RequestId::from(request_id);
        let digest: [u8; 32] = Sha256::digest(request_id.as_bytes()).into();
        // Capacity is checked before the router samples: routing spends a nonce, so a refusal
        // after it would make a retry draw differently. The state lock is held across routing
        // so no other call can take the room in between.
        let mut state = self.state();
        if state
            .pending
            .len()
            .saturating_add(max_records(self.router.registry(), &report.inner))
            > MAX_PENDING
        {
            return Err(PtrRouterError::new_err(
                "too many routing decisions await a candidate; settle some first",
            ));
        }
        let outcome = self
            .router
            .route(&FindingRouteRequest {
                request_id: &id,
                request_digest: digest,
                report: &report.inner,
                required: REQUIRED,
                eligibility: &self.eligibility,
            })
            .map_err(router_error)?;
        let records = outcome.records();
        let mut ids: BTreeMap<*const FindingRouteRecord, usize> = BTreeMap::new();
        for record in records {
            let id = state.next_id;
            state.next_id += 1;
            state.pending.insert(
                id,
                Pending {
                    record: record.clone(),
                    spec: report.spec.clone(),
                },
            );
            ids.insert(record as *const _, id);
        }
        drop(state);
        let out = PyDict::new(py);
        let route = |record: &FindingRouteRecord| {
            record_dict(py, record, ids.get(&(record as *const _)).copied())
        };
        match &outcome {
            RouteOutcome::Accept(_) => out.set_item("kind", "accept")?,
            RouteOutcome::Repair { routes } => {
                out.set_item("kind", "repair")?;
                let list = PyList::empty(py);
                for item in routes {
                    match item {
                        CodeRoute::Routed(record) => list.append(route(record)?)?,
                        CodeRoute::Unroutable(code) => {
                            let entry = PyDict::new(py);
                            entry.set_item("unroutable", code.as_str())?;
                            list.append(entry)?;
                        }
                    }
                }
                out.set_item("routes", list)?;
            }
            RouteOutcome::Regenerate { reason, proposer } => {
                out.set_item("kind", "regenerate")?;
                out.set_item("reason", fallback_name(*reason))?;
                out.set_item("route", route(proposer)?)?;
            }
            RouteOutcome::Escalate { reason } => {
                let (cause, why) = escalation_names(*reason);
                out.set_item("kind", "escalate")?;
                out.set_item("reason", why)?;
                out.set_item("cause", cause)?;
            }
        }
        Ok(out)
    }

    /// Settle one routing decision: verify a candidate for it inside Rust and log the
    /// verdict as its reward.
    ///
    /// `candidate` is what the routed Pod produced. It is verified against the same task
    /// and verifier kind as the report the decision was routed for (text for arithmetic,
    /// json_schema and code; a list of `(subject, attribute, value)` claims for facts).
    /// The reward is 1 exactly when `classify` accepts that report; the report's digest is
    /// stored in the log (`log_digests`). A report object passed from Python is not
    /// accepted. Python chooses which decisions it settles, the task of the report it routes
    /// and the candidate text; it cannot state or forge the reward, which is bound to the
    /// routed report's task and computed in Rust. The log is therefore only as honest as the
    /// harness that produces candidates from the routed Pod and settles every decision. The
    /// report digest and the candidate digest are stored (`log_digests`). A decision is
    /// settled once. Returns True when the reward was 1.
    ///
    /// Raises PtrRouterError for an unknown or already settled decision, a router with no
    /// exploration floor, or a full log, and PtrVerifierError for a candidate of the wrong
    /// shape.
    fn settle(
        &self,
        py: Python<'_>,
        decision_id: usize,
        candidate: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let spec = {
            let state = self.state();
            if state.log.len() >= MAX_LOG {
                return Err(PtrRouterError::new_err("the off-policy log is full"));
            }
            state
                .pending
                .get(&decision_id)
                .map(|pending| pending.spec.clone())
                .ok_or_else(|| {
                    PtrRouterError::new_err(format!(
                        "decision {decision_id} is unknown or already settled"
                    ))
                })?
        };
        let candidate = spec.candidate(candidate)?;
        let report = py.detach(|| spec.verify(&candidate))?;
        let candidate_digest: [u8; 32] = Sha256::digest(candidate.canonical_bytes()).into();
        let mut state = self.state();
        // The verification ran without the lock: check the bound again before pushing.
        if state.log.len() >= MAX_LOG {
            return Err(PtrRouterError::new_err("the off-policy log is full"));
        }
        let pending = state.pending.get(&decision_id).ok_or_else(|| {
            PtrRouterError::new_err(format!(
                "decision {decision_id} is unknown or already settled"
            ))
        })?;
        let class = classify(&report, REQUIRED, &RouterLimits::default());
        let reward = VerifierReward::from_class(&class);
        let seq = state.log.len() as u64;
        let decision = LoggedDecision::from_record(seq, &pending.record, reward, self.exploration)
            .map_err(router_error)?;
        state.log.push(decision);
        state
            .digests
            .push((report_digest(&report), candidate_digest));
        state.pending.remove(&decision_id);
        Ok(reward.passed())
    }

    /// For each logged reward, in log order, `{"report": hex, "candidate": hex}`: the
    /// SHA-256 of the verifier report and of the candidate's canonical bytes (the text, or
    /// length-prefixed claim fields), so a logged reward can be audited by re-verifying.
    /// The report digest covers the report only; the candidate digest covers what Python
    /// chose to submit.
    fn log_digests<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let digests = self.state().digests.clone();
        digests
            .iter()
            .map(|(report, candidate)| {
                let item = PyDict::new(py);
                item.set_item("report", hex(report))?;
                item.set_item("candidate", hex(candidate))?;
                Ok(item)
            })
            .collect()
    }

    /// The number of settled decisions in the off-policy log.
    fn log_len(&self) -> usize {
        self.state().log.len()
    }

    /// Estimate a target policy from the settled log: IPS, SNIPS and doubly robust.
    ///
    /// `target` maps a class label (`"repair/<code>"` or `"regenerate"`) to
    /// `{pod: weight}`; a class that is absent is uniform and a Pod absent from its class
    /// has weight zero. `reward_model` maps a label to `{pod: p}` with `default_reward`
    /// elsewhere. Returns `{"n", "ips", "snips", "doubly_robust", "se_ips", "se_snips",
    /// "se_doubly_robust", "effective_sample_size"}`. This is an estimate only: promotion
    /// of a policy is a host action that is not exposed here.
    ///
    /// Raises PtrRouterError for an empty log, a target naming a Pod outside a logged
    /// support, a target without mass, a reward outside [0, 1] or a non-finite estimate.
    #[pyo3(signature = (target, *, reward_model=None, default_reward=0.5))]
    fn estimate<'py>(
        &self,
        py: Python<'py>,
        target: BTreeMap<String, BTreeMap<String, f64>>,
        reward_model: Option<BTreeMap<String, BTreeMap<String, f64>>>,
        default_reward: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let model = TableReward {
            predictions: reward_model.unwrap_or_default(),
            default: default_reward,
        };
        let state = self.state();
        let estimate = estimate_policy(&state.log, &TableScorer { weights: target }, &model)
            .map_err(router_error)?;
        estimate_dict(py, &estimate)
    }

    /// How often each code was reported with no eligible repair Pod.
    fn unroutable_counts(&self) -> BTreeMap<String, u64> {
        self.router
            .unroutable_counts()
            .into_iter()
            .map(|(code, count)| (code.to_string(), count))
            .collect()
    }

    /// The exploration floor this router was built with.
    #[getter]
    fn exploration(&self) -> f64 {
        self.exploration.epsilon()
    }

    /// The registered Pod ids, in registration order.
    fn pods(&self) -> Vec<String> {
        self.registered.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptr_types::Probability;
    use ptr_verifier::{Finding, VerificationReport, VerificationStatus};

    fn report(
        status: VerificationStatus,
        level: VerificationLevel,
        hard: bool,
    ) -> VerificationReport {
        VerificationReport {
            status,
            level,
            score: Probability::new(1.0).unwrap(),
            findings: if hard {
                vec![Finding {
                    code: "x".into(),
                    message: String::new(),
                    hard: true,
                }]
            } else {
                vec![]
            },
        }
    }

    fn class(report: &VerificationReport) -> ReportClass {
        classify(report, REQUIRED, &RouterLimits::default())
    }

    #[test]
    fn the_fixed_level_is_the_deterministic_one() {
        assert_eq!(REQUIRED, VerificationLevel::Deterministic);
    }

    #[test]
    fn a_below_level_pass_is_never_an_accept() {
        for level in [
            VerificationLevel::Unverified,
            VerificationLevel::LatentAgreement,
            VerificationLevel::SampleVerified,
            VerificationLevel::FullSemantic,
        ] {
            let pass = report(VerificationStatus::Pass, level, false);
            assert!(matches!(
                class(&pass),
                ReportClass::Fallback(FallbackReason::BelowLevel)
            ));
        }
        let pass = report(VerificationStatus::Pass, REQUIRED, false);
        assert!(matches!(class(&pass), ReportClass::Accept(_)));
    }

    #[test]
    fn a_forged_pass_is_never_an_accept() {
        let forged = report(VerificationStatus::Pass, REQUIRED, true);
        assert!(matches!(
            class(&forged),
            ReportClass::Fallback(FallbackReason::ForgedPass)
        ));
    }

    #[test]
    fn unknown_and_disputed_are_never_an_accept() {
        for status in [VerificationStatus::Unknown, VerificationStatus::Disputed] {
            assert!(!matches!(
                class(&report(status, REQUIRED, false)),
                ReportClass::Accept(_)
            ));
        }
    }
}
