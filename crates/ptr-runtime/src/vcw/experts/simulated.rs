//! Programmatic experts with seeded, configurable error and repair rates.
//!
//! These are the deterministic baseline: a proposer whose candidate carries each defect code
//! with a configured probability, and repairers that clear the one code they are asked about
//! with a configured probability. They exist for tests and for experiments on routing, where
//! the repair skill of each Pod per code must be known exactly.
//!
//! Every draw is a pure function of `(seed, role, request, Pod, code, occurrence)`, where
//! `occurrence` counts the earlier calls with the same `(request, Pod, role, code)`. It does not
//! depend on the round, on other requests, Pods or codes, or on any global call count: an extra,
//! skipped or declined call elsewhere does not shift a draw, and two arms of an experiment that
//! make the same call get the same draw. Request ids must therefore not be reused with more than
//! [`MAX_TRACKED_REQUESTS`] other requests in between: the counters of an evicted id restart at
//! zero and its draws repeat. The counters are kept for the last
//! [`MAX_TRACKED_REQUESTS`] requests; a request seen again after that many others restarts its
//! occurrence indexes at zero.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::sync::Arc;

use ptr_pods::{DynPod, PodManifest, TypedPayload};
use ptr_router::FindingCode;
use ptr_types::{Digest, PodId, Probability, ProjectId, TypeId, VerificationLevel};
use ptr_verifier::{Finding, VerificationReport, VerificationStatus, Verifier};
use sha2::{Digest as _, Sha256};

use ptr_router::{PodEligibility, PodRef};

use super::guard::{Guarded, GuardedExperts};
use super::manifest::{expert_manifest, ExpertRole};
use super::{CallOutcome, CallRecord, ExpertCall, RosterError, MAX_LOGGED_CALLS};
use crate::vcw::{ProposeCall, RepairCall, Roster, VcwPayload};

/// Requests whose occurrence counters are kept.
pub const MAX_TRACKED_REQUESTS: usize = 4096;

/// A payload with a known set of defects that a programmatic expert can plant and clear.
pub trait DefectPayload: VcwPayload {
    /// A defect-free candidate for `task`.
    fn clean(task: &str) -> Self;

    /// The defect codes present.
    fn defects(&self) -> BTreeSet<String>;

    /// This candidate with `code` planted.
    #[must_use]
    fn with_defect(&self, code: &str) -> Self;

    /// This candidate with `code` cleared.
    #[must_use]
    fn without_defect(&self, code: &str) -> Self;
}

/// The simulation payload: a task and the defect codes it currently carries.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DefectDoc {
    task: String,
    defects: BTreeSet<String>,
}

impl DefectDoc {
    pub const TYPE_ID: &'static str = "vcw.sim.defect_doc";

    /// The payload type id, for manifests.
    #[must_use]
    pub fn clean_type() -> TypeId {
        TypeId::from(Self::TYPE_ID)
    }

    #[must_use]
    pub fn task(&self) -> &str {
        &self.task
    }
}

const DEFECT_DOC_DIGEST_DOMAIN: &[u8] = b"ptr.runtime.vcw.sim.defect_doc.v2";

/// One field of a digest: its length as `u64` little-endian, then its bytes.
fn put_field(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

impl VcwPayload for DefectDoc {
    fn payload_type(&self) -> TypeId {
        TypeId::from(Self::TYPE_ID)
    }

    /// v2 (issue #141): a domain tag, the task and every defect length-prefixed, and the defect
    /// count. v1 joined them with a 0 byte, so a task holding a 0 byte aliased a task with one
    /// more defect.
    fn payload_digest(&self) -> Digest {
        let mut hasher = Sha256::new();
        hasher.update(DEFECT_DOC_DIGEST_DOMAIN);
        put_field(&mut hasher, self.task.as_bytes());
        hasher.update((self.defects.len() as u64).to_le_bytes());
        for defect in &self.defects {
            put_field(&mut hasher, defect.as_bytes());
        }
        hasher.finalize().into()
    }

    fn confined_to(&self, edited: &Self, named: &[String]) -> bool {
        let rest = |doc: &Self| -> BTreeSet<String> {
            doc.defects
                .iter()
                .filter(|code| !named.contains(code))
                .cloned()
                .collect()
        };
        self.task == edited.task && rest(self) == rest(edited)
    }
}

impl DefectPayload for DefectDoc {
    fn clean(task: &str) -> Self {
        Self {
            task: task.to_owned(),
            defects: BTreeSet::new(),
        }
    }

    fn defects(&self) -> BTreeSet<String> {
        self.defects.clone()
    }

    fn with_defect(&self, code: &str) -> Self {
        let mut next = self.clone();
        next.defects.insert(code.to_owned());
        next
    }

    fn without_defect(&self, code: &str) -> Self {
        let mut next = self.clone();
        next.defects.remove(code);
        next
    }
}

/// A verifier for [`DefectDoc`] that reports each defect as a hard finding. It is part of the
/// simulation (the defects are the ground truth the simulation planted), not a verifier of any
/// real task.
#[derive(Clone, Copy, Debug, Default)]
pub struct DefectDocVerifier;

impl Verifier<DefectDoc> for DefectDocVerifier {
    fn verify(&self, candidate: &DefectDoc) -> VerificationReport {
        let findings: Vec<Finding> = candidate
            .defects
            .iter()
            .map(|code| Finding {
                code: code.clone(),
                message: String::new(),
                hard: true,
            })
            .collect();
        let clean = findings.is_empty();
        VerificationReport {
            status: if clean {
                VerificationStatus::Pass
            } else {
                VerificationStatus::Fail
            },
            level: VerificationLevel::Deterministic,
            score: Probability::new(if clean { 1.0 } else { 0.0 }).expect("0 and 1 are valid"),
            findings,
        }
    }
}

/// Why a [`SimulationConfig`] is invalid.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SimulationError {
    /// A rate is NaN or outside `0..=1`.
    BadProbability { what: String },
    /// `pod_repair_success` names a Pod that is not a repairer of the roster.
    UnknownPod { pod: String },
    /// `pod_repair_success` names a code the Pod does not repair.
    UnknownCode { pod: String, code: String },
}

impl fmt::Display for SimulationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadProbability { what } => write!(f, "{what} is not a probability in 0..=1"),
            Self::UnknownPod { pod } => {
                write!(f, "pod_repair_success names {pod}, which is not a repairer")
            }
            Self::UnknownCode { pod, code } => {
                write!(
                    f,
                    "pod_repair_success names code {code}, which {pod} does not repair"
                )
            }
        }
    }
}

impl std::error::Error for SimulationError {}

/// Rates and seed of the simulation.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct SimulationConfig {
    pub seed: u64,
    /// Probability that a proposal carries each defect code.
    pub error_rates: BTreeMap<String, f64>,
    /// Probability that a repairer clears a code, unless overridden per Pod.
    pub repair_success: BTreeMap<String, f64>,
    /// Probability for a code absent from `repair_success`.
    pub default_repair_success: f64,
    /// Per `(Pod, code)` overrides of the repair probability.
    pub pod_repair_success: BTreeMap<(String, String), f64>,
    /// A failed repair returns nothing (a declined repair) instead of the unchanged candidate.
    pub decline_on_failure: bool,
}

impl SimulationConfig {
    /// A configuration with no errors and certain repair.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            error_rates: BTreeMap::new(),
            repair_success: BTreeMap::new(),
            default_repair_success: 1.0,
            pod_repair_success: BTreeMap::new(),
            decline_on_failure: false,
        }
    }

    #[must_use]
    pub fn with_error_rate(mut self, code: &str, rate: f64) -> Self {
        self.error_rates.insert(code.to_owned(), rate);
        self
    }

    #[must_use]
    pub fn with_repair_success(mut self, code: &str, probability: f64) -> Self {
        self.repair_success.insert(code.to_owned(), probability);
        self
    }

    #[must_use]
    pub fn with_pod_repair_success(mut self, pod: &str, code: &str, probability: f64) -> Self {
        self.pod_repair_success
            .insert((pod.to_owned(), code.to_owned()), probability);
        self
    }

    /// # Errors
    /// [`SimulationError::BadProbability`] naming the first offending rate.
    pub fn validate(&self) -> Result<(), SimulationError> {
        let ok = |p: f64| p.is_finite() && (0.0..=1.0).contains(&p);
        let bad = |what: String| Err(SimulationError::BadProbability { what });
        if !ok(self.default_repair_success) {
            return bad("default_repair_success".into());
        }
        for (code, p) in &self.error_rates {
            if !ok(*p) {
                return bad(format!("error rate of {code}"));
            }
        }
        for (code, p) in &self.repair_success {
            if !ok(*p) {
                return bad(format!("repair success of {code}"));
            }
        }
        for ((pod, code), p) in &self.pod_repair_success {
            if !ok(*p) {
                return bad(format!("repair success of {pod}/{code}"));
            }
        }
        Ok(())
    }

    fn success_for(&self, pod: &str, code: &str) -> f64 {
        self.pod_repair_success
            .get(&(pod.to_owned(), code.to_owned()))
            .or_else(|| self.repair_success.get(code))
            .copied()
            .unwrap_or(self.default_repair_success)
    }
}

/// A uniform draw in `[0, 1)` keyed by the parts.
fn keyed_draw(seed: u64, parts: &[&[u8]]) -> f64 {
    let mut hasher = Sha256::new();
    hasher.update(seed.to_le_bytes());
    for part in parts {
        hasher.update((part.len() as u64).to_le_bytes());
        hasher.update(part);
    }
    let digest = hasher.finalize();
    let mut head = [0_u8; 8];
    head.copy_from_slice(&digest[..8]);
    // 53 bits give an exact `f64` in [0, 1).
    (u64::from_le_bytes(head) >> 11) as f64 / (1_u64 << 53) as f64
}

/// How many times each `(request, Pod, role, code)` has been drawn for, for the last
/// [`MAX_TRACKED_REQUESTS`] requests.
#[derive(Default)]
struct Occurrences {
    by_request: BTreeMap<String, BTreeMap<(String, ExpertRole, String), u64>>,
    order: VecDeque<String>,
}

impl Occurrences {
    /// The index of this draw (0 for the first), counting it.
    fn next(&mut self, request: &str, pod: &str, role: ExpertRole, code: &str) -> u64 {
        if !self.by_request.contains_key(request) {
            if self.order.len() >= MAX_TRACKED_REQUESTS {
                if let Some(oldest) = self.order.pop_front() {
                    self.by_request.remove(&oldest);
                }
            }
            self.order.push_back(request.to_owned());
        }
        let counter = self
            .by_request
            .entry(request.to_owned())
            .or_default()
            .entry((pod.to_owned(), role, code.to_owned()))
            .or_insert(0);
        let index = *counter;
        *counter += 1;
        index
    }
}

/// The programmatic experts for one task. One value serves every simulated Pod of the roster;
/// which Pod answers is the loop's routing decision.
pub struct SimulatedExperts<P: DefectPayload> {
    config: SimulationConfig,
    task: String,
    proposers: BTreeSet<PodId>,
    repairers: BTreeMap<PodId, BTreeSet<String>>,
    occurrences: Occurrences,
    calls: VecDeque<ExpertCall>,
    marker: std::marker::PhantomData<fn() -> P>,
}

impl<P: DefectPayload> SimulatedExperts<P> {
    /// # Errors
    /// [`SimulationError`] when `config` holds a rate outside `0..=1`.
    pub fn new(config: SimulationConfig, task: &str) -> Result<Self, SimulationError> {
        config.validate()?;
        Ok(Self {
            config,
            task: task.to_owned(),
            proposers: BTreeSet::new(),
            repairers: BTreeMap::new(),
            occurrences: Occurrences::default(),
            calls: VecDeque::new(),
            marker: std::marker::PhantomData,
        })
    }

    pub fn set_task(&mut self, task: &str) {
        task.clone_into(&mut self.task);
    }

    fn holds(&self, pod: &PodId) -> bool {
        self.proposers.contains(pod) || self.repairers.contains_key(pod)
    }

    /// # Errors
    /// [`RosterError::DuplicatePod`] when a Pod with this id is already held.
    pub fn add_proposer(&mut self, pod: PodId) -> Result<(), RosterError> {
        if self.holds(&pod) {
            return Err(RosterError::DuplicatePod(pod));
        }
        self.proposers.insert(pod);
        Ok(())
    }

    /// # Errors
    /// [`RosterError::DuplicatePod`] when a Pod with this id is already held.
    pub fn add_repairer(&mut self, pod: PodId, codes: &[&str]) -> Result<(), RosterError> {
        if self.holds(&pod) {
            return Err(RosterError::DuplicatePod(pod));
        }
        self.repairers
            .insert(pod, codes.iter().map(|c| (*c).to_owned()).collect());
        Ok(())
    }

    /// Set the probability that `pod` clears `code`, for a repairer that is in the roster and
    /// repairs that code. This is how a host applies a specialist's configured skill (and the
    /// simulated collateral of its "training" on other Pods) to the programmatic experts.
    ///
    /// # Errors
    /// [`SimulationError::BadProbability`] outside `0..=1`, [`SimulationError::UnknownPod`] or
    /// [`SimulationError::UnknownCode`] when the roster does not hold the pair.
    pub fn set_pod_repair_success(
        &mut self,
        pod: &PodId,
        code: &str,
        probability: f64,
    ) -> Result<(), SimulationError> {
        if !(probability.is_finite() && (0.0..=1.0).contains(&probability)) {
            return Err(SimulationError::BadProbability {
                what: format!("repair success of {}/{code}", pod.0),
            });
        }
        let Some(codes) = self.repairers.get(pod) else {
            return Err(SimulationError::UnknownPod { pod: pod.0.clone() });
        };
        if !codes.contains(code) {
            return Err(SimulationError::UnknownCode {
                pod: pod.0.clone(),
                code: code.to_owned(),
            });
        }
        self.config
            .pod_repair_success
            .insert((pod.0.clone(), code.to_owned()), probability);
        Ok(())
    }

    /// Every `pod_repair_success` entry must name a repairer of this roster and a code it
    /// repairs: a typo would otherwise fall back to the default rate without a sign.
    ///
    /// # Errors
    /// [`SimulationError::UnknownPod`] or [`SimulationError::UnknownCode`] for the first entry
    /// that does not match.
    pub fn validate_roster(&self) -> Result<(), SimulationError> {
        for (pod, code) in self.config.pod_repair_success.keys() {
            let Some(codes) = self.repairers.get(&PodId::from(pod.as_str())) else {
                return Err(SimulationError::UnknownPod { pod: pod.clone() });
            };
            if !codes.contains(code) {
                return Err(SimulationError::UnknownCode {
                    pod: pod.clone(),
                    code: code.clone(),
                });
            }
        }
        Ok(())
    }

    /// The roster to hand to [`VcwParts`](crate::vcw::VcwParts), after
    /// [`Self::validate_roster`].
    ///
    /// # Errors
    /// The error of [`Self::validate_roster`].
    pub fn roster(&self) -> Result<Roster, SimulationError> {
        self.validate_roster()?;
        Ok(Roster {
            proposers: self.proposers.iter().cloned().collect(),
            repairers: self
                .repairers
                .iter()
                .map(|(pod, codes)| {
                    (
                        pod.clone(),
                        codes.iter().map(|c| FindingCode::new(c.as_str())).collect(),
                    )
                })
                .collect(),
        })
    }

    /// This adapter as the loop's [`RolePods`](crate::vcw::RolePods), checking each Pod's
    /// eligibility in `standing` at call time. `standing` must be the view passed to the loop's
    /// `run` for the same request.
    #[must_use]
    pub fn guarded<'a>(&'a mut self, standing: &'a dyn PodEligibility) -> Guarded<'a, Self> {
        Guarded::new(self, standing)
    }

    /// Pods for the registry, with manifests of the payload type `P`.
    #[must_use]
    pub fn pods(&self, project: &ProjectId, payload: &TypeId) -> Vec<Arc<dyn DynPod>> {
        let proposers = self.proposers.iter().map(|pod| (pod, ExpertRole::Proposer));
        let repairers = self.repairers.keys().map(|pod| (pod, ExpertRole::Repairer));
        proposers
            .chain(repairers)
            .map(|(pod, role)| {
                Arc::new(SimulatedPod {
                    manifest: expert_manifest(project, pod, role, payload),
                }) as Arc<dyn DynPod>
            })
            .collect()
    }

    #[must_use]
    pub fn calls(&self) -> &VecDeque<ExpertCall> {
        &self.calls
    }

    fn log(&mut self, call: ExpertCall) {
        if self.calls.len() >= MAX_LOGGED_CALLS {
            self.calls.pop_front();
        }
        self.calls.push_back(call);
    }
}

impl<P: DefectPayload> SimulatedExperts<P> {
    fn record(
        &mut self,
        role: ExpertRole,
        pod: &PodRef,
        round: u32,
        code: Option<&str>,
        outcome: CallOutcome,
    ) {
        self.log(
            CallRecord {
                role,
                pod,
                round,
                code,
                outcome,
                new_tokens: 0,
                answer: String::new(),
                model: None,
            }
            .into_call(),
        );
    }
}

impl<P: DefectPayload> GuardedExperts<P> for SimulatedExperts<P> {
    fn propose_while(&mut self, call: ProposeCall<'_>, eligible: &dyn Fn() -> bool) -> Option<P> {
        if !self.proposers.contains(&call.pod.pod) {
            return None;
        }
        if !eligible() {
            // No draw is consumed: an ineligible call must not shift later draws.
            self.record(
                ExpertRole::Proposer,
                call.pod,
                call.round,
                None,
                CallOutcome::IneligibleAtCallTime,
            );
            return None;
        }
        let mut candidate = P::clean(&self.task);
        for (code, rate) in &self.config.error_rates {
            let occurrence = self.occurrences.next(
                &call.request_id.0,
                &call.pod.pod.0,
                ExpertRole::Proposer,
                code,
            );
            let draw = keyed_draw(
                self.config.seed,
                &[
                    b"propose",
                    call.request_id.0.as_bytes(),
                    call.pod.pod.0.as_bytes(),
                    code.as_bytes(),
                    &occurrence.to_le_bytes(),
                ],
            );
            if draw < *rate {
                candidate = candidate.with_defect(code);
            }
        }
        self.record(
            ExpertRole::Proposer,
            call.pod,
            call.round,
            None,
            CallOutcome::Produced,
        );
        Some(candidate)
    }

    fn repair_while(&mut self, call: RepairCall<'_, P>, eligible: &dyn Fn() -> bool) -> Option<P> {
        let codes = self.repairers.get(&call.pod.pod)?;
        if !codes.contains(call.code.as_str()) {
            return None;
        }
        if !eligible() {
            self.record(
                ExpertRole::Repairer,
                call.pod,
                call.round,
                Some(call.code.as_str()),
                CallOutcome::IneligibleAtCallTime,
            );
            return None;
        }
        let occurrence = self.occurrences.next(
            &call.request_id.0,
            &call.pod.pod.0,
            ExpertRole::Repairer,
            call.code.as_str(),
        );
        let draw = keyed_draw(
            self.config.seed,
            &[
                b"repair",
                call.request_id.0.as_bytes(),
                call.pod.pod.0.as_bytes(),
                call.code.as_str().as_bytes(),
                &occurrence.to_le_bytes(),
            ],
        );
        let success = draw < self.config.success_for(&call.pod.pod.0, call.code.as_str());
        let result = if success {
            Some(call.previous.without_defect(call.code.as_str()))
        } else if self.config.decline_on_failure {
            None
        } else {
            Some(call.previous.clone())
        };
        let outcome = if result.is_some() {
            CallOutcome::Produced
        } else {
            CallOutcome::Declined
        };
        self.record(
            ExpertRole::Repairer,
            call.pod,
            call.round,
            Some(call.code.as_str()),
            outcome,
        );
        result
    }
}

/// A registry entry for a programmatic expert. The loop drives it through [`SimulatedExperts`];
/// invoking it directly is refused.
pub struct SimulatedPod {
    manifest: PodManifest,
}

impl DynPod for SimulatedPod {
    fn manifest(&self) -> &PodManifest {
        &self.manifest
    }

    fn invoke(&self, _input: TypedPayload) -> Result<TypedPayload, String> {
        Err("a simulated expert is driven through RolePods".to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Issue #141: v1 joined the task and the defects with a 0 byte, so a task ending in
    /// `\0x` read the same as the task with the defect `x`.
    #[test]
    fn a_zero_byte_in_the_task_does_not_alias_a_defect() {
        let embedded = DefectDoc::clean("t\0x");
        let defect = DefectDoc::clean("t").with_defect("x");
        assert_ne!(embedded.payload_digest(), defect.payload_digest());
        let two = DefectDoc::clean("t").with_defect("x\0y");
        let split = DefectDoc::clean("t").with_defect("x").with_defect("y");
        assert_ne!(two.payload_digest(), split.payload_digest());
    }

    #[test]
    fn rates_outside_zero_one_are_refused() {
        for bad in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
            assert!(SimulationConfig::new(1)
                .with_error_rate("a", bad)
                .validate()
                .is_err());
            assert!(SimulationConfig::new(1)
                .with_repair_success("a", bad)
                .validate()
                .is_err());
            assert!(SimulationConfig::new(1)
                .with_pod_repair_success("p", "a", bad)
                .validate()
                .is_err());
        }
        assert!(SimulationConfig::new(1)
            .with_error_rate("a", 0.0)
            .validate()
            .is_ok());
        assert!(SimulationConfig::new(1)
            .with_error_rate("a", 1.0)
            .validate()
            .is_ok());
    }

    #[test]
    fn a_pod_override_beats_the_code_rate_which_beats_the_default() {
        let config = SimulationConfig::new(1)
            .with_repair_success("a", 0.25)
            .with_pod_repair_success("p", "a", 0.75);
        assert!((config.success_for("p", "a") - 0.75).abs() < f64::EPSILON);
        assert!((config.success_for("q", "a") - 0.25).abs() < f64::EPSILON);
        assert!((config.success_for("q", "z") - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn keyed_draws_are_uniform_enough_and_depend_on_every_part() {
        let mut below = 0;
        let n = 4000;
        for i in 0..n {
            let d = keyed_draw(7, &[b"x", &(i as u64).to_le_bytes()]);
            assert!((0.0..1.0).contains(&d));
            if d < 0.3 {
                below += 1;
            }
        }
        let share = f64::from(below) / f64::from(n);
        assert!((share - 0.3).abs() < 0.03, "{share}");
        assert_ne!(keyed_draw(1, &[b"a"]), keyed_draw(2, &[b"a"]));
        assert_ne!(keyed_draw(1, &[b"a", b"b"]), keyed_draw(1, &[b"ab"]));
    }

    #[test]
    fn a_defect_doc_is_confined_to_the_named_codes() {
        let base = DefectDoc::clean("t").with_defect("a").with_defect("b");
        let fixed_a = base.without_defect("a");
        assert!(base.confined_to(&fixed_a, &["a".to_owned()]));
        assert!(!base.confined_to(&base.without_defect("b"), &["a".to_owned()]));
        assert_ne!(base.payload_digest(), fixed_a.payload_digest());
    }

    use ptr_types::{Generation, RequestId};

    fn pod_ref(name: &str) -> PodRef {
        PodRef::new(PodId::from(name), Generation(1))
    }

    fn experts(seed: u64) -> SimulatedExperts<DefectDoc> {
        let config = SimulationConfig::new(seed)
            .with_error_rate("a", 0.5)
            .with_error_rate("b", 0.5)
            .with_repair_success("a", 0.5)
            .with_repair_success("b", 0.5);
        let mut experts = SimulatedExperts::new(config, "t").unwrap();
        experts.add_proposer(PodId::from("gen")).unwrap();
        experts.add_proposer(PodId::from("gen2")).unwrap();
        experts
            .add_repairer(PodId::from("fix"), &["a", "b"])
            .unwrap();
        experts
            .add_repairer(PodId::from("fix2"), &["a", "b"])
            .unwrap();
        experts
    }

    fn propose(experts: &mut SimulatedExperts<DefectDoc>, request: &str, pod: &str) -> DefectDoc {
        let (id, pod) = (RequestId::from(request), pod_ref(pod));
        let call = ProposeCall {
            pod: &pod,
            request_id: &id,
            request_digest: [0; 32],
            round: 0,
            reason: None,
            snapshot: None,
        };
        experts.propose_while(call, &|| true).unwrap()
    }

    fn repair(
        experts: &mut SimulatedExperts<DefectDoc>,
        request: &str,
        pod: &str,
        code: &str,
        round: u32,
        eligible: bool,
    ) -> Option<DefectDoc> {
        let (id, pod, code) = (
            RequestId::from(request),
            pod_ref(pod),
            FindingCode::new(code),
        );
        let previous = DefectDoc::clean("t").with_defect("a").with_defect("b");
        let critique = ptr_feedback::Critique {
            candidate: ptr_types::CandidateId::from("c"),
            issues: vec![code.as_str().to_owned()],
            repairable: true,
        };
        let call = RepairCall {
            pod: &pod,
            request_id: &id,
            round,
            code: &code,
            critique: &critique,
            previous: &previous,
            snapshot: None,
        };
        experts.repair_while(call, &|| eligible)
    }

    #[test]
    fn an_extra_call_elsewhere_does_not_shift_a_draw() {
        // Arm one makes only the calls of the requests under study. Arm two makes many other
        // calls in between (other requests, Pods, codes, a different round, ineligible calls).
        let mut quiet = experts(5);
        let mut busy = experts(5);
        for i in 0..200 {
            let request = format!("r{i}");
            // Noise in the busy arm only.
            propose(&mut busy, &format!("noise{i}"), "gen2");
            repair(&mut busy, &format!("noise{i}"), "fix2", "a", 3, true);
            repair(&mut busy, &request, "fix2", "b", 0, true);
            repair(&mut busy, &request, "fix", "a", 0, false);
            assert_eq!(
                propose(&mut quiet, &request, "gen"),
                propose(&mut busy, &request, "gen"),
                "proposal of {request}"
            );
            assert_eq!(
                repair(&mut quiet, &request, "fix", "a", 0, true),
                // The round differs between the arms; the draw does not depend on it.
                repair(&mut busy, &request, "fix", "a", 5, true),
                "first repair of {request}"
            );
            assert_eq!(
                repair(&mut quiet, &request, "fix", "a", 1, true),
                repair(&mut busy, &request, "fix", "a", 6, true),
                "second repair of {request}"
            );
        }
        // Both arms drew both outcomes, so the comparison above was not vacuous.
        let mut probe = experts(5);
        let outcomes: BTreeSet<bool> = (0..50)
            .map(|i| {
                repair(&mut probe, &format!("p{i}"), "fix", "a", 0, true)
                    .unwrap()
                    .defects()
                    .contains("a")
            })
            .collect();
        assert_eq!(outcomes.len(), 2);
    }

    #[test]
    fn the_occurrence_index_makes_repeated_calls_independent_draws() {
        let mut experts = experts(9);
        let draws: Vec<bool> = (0..40)
            .map(|_| {
                repair(&mut experts, "same", "fix", "a", 0, true)
                    .unwrap()
                    .defects()
                    .contains("a")
            })
            .collect();
        assert!(draws.contains(&true) && draws.contains(&false));
    }

    #[test]
    fn an_ineligible_call_is_logged_declined_and_consumes_no_draw() {
        let mut experts = experts(1);
        assert!(repair(&mut experts, "r", "fix", "a", 0, false).is_none());
        let call = experts.calls().back().unwrap();
        assert_eq!(call.outcome, CallOutcome::IneligibleAtCallTime);
        assert!(!call.produced);
        assert_eq!(experts.occurrences.by_request.len(), 0);
    }

    #[test]
    fn pod_repair_rates_for_unknown_pods_or_codes_are_flagged() {
        let base = || {
            let mut e = SimulatedExperts::<DefectDoc>::new(
                SimulationConfig::new(1).with_pod_repair_success("fix", "a", 0.5),
                "t",
            )
            .unwrap();
            e.add_repairer(PodId::from("fix"), &["a"]).unwrap();
            e
        };
        assert!(base().roster().is_ok());
        let typo_pod = SimulatedExperts::<DefectDoc>::new(
            SimulationConfig::new(1).with_pod_repair_success("fxi", "a", 0.5),
            "t",
        )
        .unwrap();
        assert!(matches!(
            typo_pod.roster(),
            Err(SimulationError::UnknownPod { .. })
        ));
        let mut typo_code = SimulatedExperts::<DefectDoc>::new(
            SimulationConfig::new(1).with_pod_repair_success("fix", "zz", 0.5),
            "t",
        )
        .unwrap();
        typo_code.add_repairer(PodId::from("fix"), &["a"]).unwrap();
        assert!(matches!(
            typo_code.roster(),
            Err(SimulationError::UnknownCode { .. })
        ));
    }

    #[test]
    fn a_pod_id_cannot_be_added_twice() {
        let mut experts = experts(1);
        assert_eq!(
            experts.add_proposer(PodId::from("gen")),
            Err(RosterError::DuplicatePod(PodId::from("gen")))
        );
        assert!(experts.add_repairer(PodId::from("gen"), &["a"]).is_err());
        assert!(experts.add_proposer(PodId::from("fix")).is_err());
    }

    #[test]
    fn the_request_counters_are_bounded() {
        let mut occurrences = Occurrences::default();
        for i in 0..(MAX_TRACKED_REQUESTS + 10) {
            occurrences.next(&format!("r{i}"), "p", ExpertRole::Repairer, "a");
        }
        assert_eq!(occurrences.by_request.len(), MAX_TRACKED_REQUESTS);
        // The oldest request restarted from zero.
        assert_eq!(occurrences.next("r0", "p", ExpertRole::Repairer, "a"), 0);
    }
}
