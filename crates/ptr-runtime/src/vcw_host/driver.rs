//! The host-side L3 driver: closed windows in, structural changes through admission out.
//!
//! ```text
//!   VcwLoop::structure_step ---> StructureProposal (inert request)
//!                                     |
//!        needs training?  --yes--> SpecialistTrainer::train --> complete(manifest digest)
//!                         --no---> ready()
//!                                     |
//!   HeldOutSuite: clear rates before / after the change (verifier on held-out tasks)
//!                                     |
//!   ForgettingGate::evaluate(matrix, public suite).bind(subject of THIS proposal)
//!                                     |
//!   PtrRuntime::admit_expert_proposal  (the only way to ExpertStructuralCommitted)
//!                                     |
//!   Ok  : register the Pod, VcwLoop::sync_roster / retire_dead_experts
//!   Err : VcwLoop::withdraw_request; nothing was committed, nothing is enrolled
//!   Ok but the install failed: the change IS committed; the request stays outstanding in the
//!         loop, the driver keeps a PendingInstall (observable, retried at the start of every
//!         step and by retry_installs) and the outcome is CommittedNotInstalled, never Refused
//! ```
//!
//! # What the gate subject binds
//!
//! The evaluation manifest in the subject is [`HeldOutSuite::manifest_digest`] of the suite the
//! measurement was made with, so the report is bound to its proposal *and* to the suite that
//! produced it; admission compares it with the digest in the host's grant. A suite that does not
//! measure every code of the serving roster and of the candidate is refused before anything is
//! bound ([`MeasurementError::Coverage`]).
//!
//! The driver is the host, not the loop: it holds the runtime and the admission grant, which the
//! loop never does (the `vcw` source scan keeps the loop's module free of both). It decides
//! nothing about admission. It submits every proposal with the gate report measured for it,
//! including a report that failed, because the runtime's admission is the authority that refuses
//! one: a driver that filtered failing reports would be a second, unaudited gate.
//!
//! Limits, stated: the structure policy supplies no expert ages, so Prune needs
//! `prune_min_age_commits = 0`; the AIMER scores come from the caller; Merge and Reactivate go
//! through the same path but are exercised by the unit tests of their parts only.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use ptr_lineage::{AdapterId, ForgettingGate, GateSubject};
use ptr_pods::{
    BaseModelRef, ExpertProposal, OpKind, PodRegistrationError, PodRegistry, ProposerRef,
    RequestedOp,
};
use ptr_router::{AdmissibleScorer, FindingCode};
use ptr_types::{CommitIndex, PodId, ProjectId};

use super::holdout::{
    check_coverage, gate_inputs, HeldOutSuite, MeasurementError, RosterChange, RosterView,
    SpecialistIdentity, SuiteError,
};
use super::specialist::{SpecialistTrainer, TrainedSpecialist, TrainingError, TrainingRequest};
use crate::expert::{
    expert_evidence_artifact_digest, expert_evidence_artifact_digest_with_inputs,
    expert_evidence_base_model_digest, BirthContext, BirthSite, ExpertAdmissionError,
    ExpertAdmissionGrant, Suppression,
};
use crate::expert::{StructureProposal, StructureRationale};
use crate::vcw::{StructureHost, VcwError, VcwLoop};
use crate::{PtrRuntime, RuntimeError};

/// Everything the driver needs from the host for one step.
pub struct DriverHost<'a> {
    pub runtime: &'a mut PtrRuntime,
    pub pods: &'a mut PodRegistry,
    /// Mints the admission grant for the runtime as it is now (its grant generation moves).
    pub grant: &'a dyn Fn(&PtrRuntime) -> ExpertAdmissionGrant,
    pub trainer: &'a mut dyn SpecialistTrainer,
    pub suite: &'a dyn HeldOutSuite,
    /// AIMER scores of the experts, computed by the host from the lineage crate.
    pub aimer: &'a BTreeMap<PodId, f64>,
}

/// Why a proposal was not admitted. Nothing was committed or enrolled.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Refusal {
    Training(TrainingError),
    Completion(String),
    Suite(SuiteError),
    Measurement(MeasurementError),
    /// The gate could not be evaluated or bound.
    Gate(String),
    /// The runtime's admission refused the proposal.
    Admission {
        /// The refusal was the gate report not passing.
        gate_failed: bool,
        message: String,
    },
}

/// What became of one request.
#[derive(Clone, Debug)]
pub enum StructureResult {
    Admitted {
        commit: CommitIndex,
        /// The trained specialist of a Birth, Reactivate or Merge (it is registered and enrolled).
        specialist: Option<Box<TrainedSpecialist>>,
        /// Experts taken out of routing by the change.
        removed: Vec<PodId>,
    },
    Refused(Refusal),
    /// The change is committed in the ledger but the host could not finish it (register the Pod,
    /// enrol it, or take dead experts out of routing). This is not a refusal: a committed
    /// expert exists and is not routable (or a retired one is still routable) until the
    /// install is retried. The driver holds a [`PendingInstall`] for it
    /// ([`StructureDriver::pending_installs`]) and retries it at the start of every step.
    CommittedNotInstalled {
        commit: CommitIndex,
        message: String,
    },
}

/// One request and its fate.
#[derive(Clone, Debug)]
pub struct StructureOutcome {
    pub kind: OpKind,
    pub rationale: StructureRationale,
    pub result: StructureResult,
}

impl StructureOutcome {
    pub fn admitted(&self) -> bool {
        matches!(self.result, StructureResult::Admitted { .. })
    }

    /// Committed in the ledger but not yet installed.
    pub fn committed_not_installed(&self) -> bool {
        matches!(self.result, StructureResult::CommittedNotInstalled { .. })
    }
}

/// The result of one driver step.
#[derive(Clone, Debug, Default)]
pub struct DriverReport {
    /// Windows closed when the step ran.
    pub windows_closed: u64,
    /// False when no new window had closed and nothing was asked of the policy.
    pub ran_policy: bool,
    pub outcomes: Vec<StructureOutcome>,
    pub suppressed: BTreeMap<Suppression, u64>,
    /// Pods whose pending install from an earlier step was completed at the start of this one
    /// (the specialist installed, or for a Prune the pruned ones taken out of routing).
    pub installed_late: Vec<PodId>,
}

impl DriverReport {
    pub fn admitted(&self, kind: OpKind) -> usize {
        self.outcomes
            .iter()
            .filter(|o| o.kind == kind && o.admitted())
            .count()
    }

    /// Outcomes that were refused (nothing committed). A committed change that is not yet
    /// installed is not counted here.
    pub fn refused(&self) -> usize {
        self.outcomes
            .iter()
            .filter(|o| matches!(o.result, StructureResult::Refused(_)))
            .count()
    }

    /// Outcomes committed in the ledger whose install is pending.
    pub fn pending_install(&self) -> usize {
        self.outcomes
            .iter()
            .filter(|o| o.committed_not_installed())
            .count()
    }
}

/// Why a step could not run at all (not a refusal of one proposal).
#[derive(Debug)]
#[non_exhaustive]
pub enum DriverError {
    Loop(VcwError),
}

impl fmt::Display for DriverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Loop(error) => write!(f, "structure step: {error}"),
        }
    }
}

impl std::error::Error for DriverError {}

/// Most pending installs the driver remembers (further ones are still reported in their
/// outcome, but only this many are retried automatically).
const MAX_PENDING: usize = 64;

/// A change that is committed in the ledger and not yet reflected in the loop and the Pod
/// registry. Holds what a retry needs; observable through
/// [`StructureDriver::pending_installs`].
#[derive(Clone, Debug)]
pub struct PendingInstall {
    commit: CommitIndex,
    proposal: ExpertProposal,
    specialist: Option<TrainedSpecialist>,
    removed: Vec<PodId>,
    message: String,
}

impl PendingInstall {
    pub fn commit(&self) -> CommitIndex {
        self.commit
    }

    /// The Pods the change is about: the specialist, or the experts being removed.
    pub fn pods(&self) -> Vec<PodId> {
        match &self.specialist {
            Some(specialist) => vec![specialist.pod.clone()],
            None => self.removed.clone(),
        }
    }

    /// Why the last attempt failed.
    pub fn message(&self) -> &str {
        &self.message
    }
}

/// The host's L3 driver for one project.
pub struct StructureDriver {
    project: ProjectId,
    proposer: ProposerRef,
    base: BaseModelRef,
    site: BirthSite,
    gate: ForgettingGate,
    processed_windows: u64,
    pending: Vec<PendingInstall>,
}

impl StructureDriver {
    /// `gate` is the host's gate configuration, which the driver does not change. The
    /// evaluation manifest of every bound subject is the digest of the suite passed in
    /// [`DriverHost`], which the host's admission grant must expect.
    #[must_use]
    pub fn new(
        project: ProjectId,
        proposer: ProposerRef,
        base: BaseModelRef,
        site: BirthSite,
        gate: ForgettingGate,
    ) -> Self {
        Self {
            project,
            proposer,
            base,
            site,
            gate,
            processed_windows: 0,
            pending: Vec::new(),
        }
    }

    /// Committed changes whose install is still pending.
    pub fn pending_installs(&self) -> &[PendingInstall] {
        &self.pending
    }

    /// Retry every pending install; those that now succeed are dropped. Returns the Pods they
    /// concerned. Idempotent.
    pub fn retry_installs<S: AdmissibleScorer>(
        &mut self,
        vcw: &mut VcwLoop<S>,
        runtime: &PtrRuntime,
        pods: &mut PodRegistry,
    ) -> Vec<PodId> {
        let mut done = Vec::new();
        let project = self.project.clone();
        self.pending.retain_mut(|pending| {
            match finish(
                &project,
                vcw,
                runtime,
                pods,
                &pending.proposal,
                pending.specialist.as_ref(),
            ) {
                Ok(()) => {
                    done.extend(pending.pods());
                    false
                }
                Err(message) => {
                    pending.message = message;
                    true
                }
            }
        });
        done
    }

    /// Windows already handed to the policy.
    pub fn processed_windows(&self) -> u64 {
        self.processed_windows
    }

    /// If a window has closed since the last step, ask the policy and resolve each request it
    /// makes, in the policy's order.
    ///
    /// # Errors
    /// [`DriverError`] when the loop itself refuses the step (its observation limits, a full
    /// outstanding table). A refused *proposal* is an outcome, not an error.
    pub fn step<S: AdmissibleScorer>(
        &mut self,
        vcw: &mut VcwLoop<S>,
        host: DriverHost<'_>,
    ) -> Result<DriverReport, DriverError> {
        let DriverHost {
            runtime,
            pods,
            grant,
            trainer,
            suite,
            aimer,
        } = host;
        // A committed change that could not be installed last time is finished first.
        let installed_late = self.retry_installs(vcw, runtime, pods);
        let closed = vcw.windows_closed();
        if closed == self.processed_windows {
            return Ok(DriverReport {
                windows_closed: closed,
                installed_late,
                ..DriverReport::default()
            });
        }
        let observed_at = runtime
            .committed_events()
            .last()
            .map_or(CommitIndex(0), |event| event.index);
        let report = {
            let registry = runtime.expert_registry();
            // A score for an expert the loop no longer holds (pruned, merged away) is stale
            // host state, not a signal; the policy would refuse an observation that names it.
            let aimer: BTreeMap<PodId, f64> = aimer
                .iter()
                .filter(|(pod, _)| {
                    vcw.repairers().contains_key(*pod) && registry.get(&self.project, pod).is_some()
                })
                .map(|(pod, score)| (pod.clone(), *score))
                .collect();
            vcw.structure_step(StructureHost {
                project: self.project.clone(),
                proposer: self.proposer.clone(),
                observed_at,
                registry,
                birth: Some(BirthContext {
                    base_model: self.base.clone(),
                    site: self.site,
                }),
                aimer,
                overlaps: Vec::new(),
            })
            .map_err(DriverError::Loop)?
        };
        self.processed_windows = closed;
        let mut outcomes = Vec::new();
        for proposal in report.proposals() {
            let result = self.resolve(vcw, proposal, runtime, pods, grant, trainer, suite);
            outcomes.push(StructureOutcome {
                kind: proposal.kind(),
                rationale: proposal.rationale().clone(),
                result,
            });
        }
        Ok(DriverReport {
            windows_closed: closed,
            ran_policy: true,
            outcomes,
            suppressed: report.suppressed().clone(),
            installed_late,
        })
    }

    /// The codes a request that creates an expert is for.
    fn codes_of<S: AdmissibleScorer>(
        vcw: &VcwLoop<S>,
        rationale: &StructureRationale,
    ) -> BTreeSet<FindingCode> {
        let held = |pod: &PodId| vcw.repairers().get(pod).cloned().unwrap_or_default();
        match rationale {
            StructureRationale::Birth { code, .. } => BTreeSet::from([code.clone()]),
            StructureRationale::Reactivate { pod, code, .. } => {
                let mut codes = held(pod);
                codes.insert(code.clone());
                codes
            }
            StructureRationale::Merge { first, second, .. } => {
                let mut codes = held(first);
                codes.extend(held(second));
                codes
            }
            StructureRationale::Prune { .. } | StructureRationale::Freeze { .. } => BTreeSet::new(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn resolve<S: AdmissibleScorer>(
        &mut self,
        vcw: &mut VcwLoop<S>,
        proposal: &StructureProposal,
        runtime: &mut PtrRuntime,
        pods: &mut PodRegistry,
        grant: &dyn Fn(&PtrRuntime) -> ExpertAdmissionGrant,
        trainer: &mut dyn SpecialistTrainer,
        suite: &dyn HeldOutSuite,
    ) -> StructureResult {
        let creates = proposal.needs_training();
        // 1. The expert, if the request creates one.
        let specialist = if creates {
            let result_pods: Vec<PodId> = proposal
                .request()
                .op()
                .results()
                .iter()
                .map(|r| r.pod().clone())
                .collect();
            let Some(pod) = result_pods.first().cloned() else {
                return StructureResult::Refused(Refusal::Completion("no result pod".into()));
            };
            let sources = match proposal.request().op() {
                RequestedOp::Merge { sources, .. } => {
                    sources.iter().map(|s| s.pod().clone()).collect()
                }
                RequestedOp::Reactivate { target, .. } => vec![target.pod().clone()],
                _ => Vec::new(),
            };
            let request = TrainingRequest {
                kind: proposal.kind(),
                pod,
                codes: Self::codes_of(vcw, proposal.rationale()),
                sources,
            };
            match trainer.train(&request) {
                Ok(specialist) => Some(specialist),
                Err(error) => {
                    vcw.withdraw_structure_request(proposal);
                    return StructureResult::Refused(Refusal::Training(error));
                }
            }
        } else {
            None
        };
        let expert_proposal = match (&specialist, creates) {
            (Some(specialist), true) => match proposal.complete(specialist.manifest_digest) {
                Ok(p) => p,
                Err(error) => {
                    vcw.withdraw_structure_request(proposal);
                    return StructureResult::Refused(Refusal::Completion(error.to_string()));
                }
            },
            _ => match proposal.ready() {
                Some(p) => p,
                None => {
                    vcw.withdraw_structure_request(proposal);
                    return StructureResult::Refused(Refusal::Completion(
                        "a request without a result produced no proposal".into(),
                    ));
                }
            },
        };
        let withdraw = |vcw: &mut VcwLoop<S>| {
            vcw.withdraw_structure_request(proposal);
        };
        // 2. The held-out measurement, before and after the change.
        let roster = RosterView::of(vcw);
        let removed_pods: Vec<PodId> = match proposal.request().op() {
            RequestedOp::Prune { targets } => targets.iter().map(|t| t.pod().clone()).collect(),
            RequestedOp::Merge { sources, .. } => sources.iter().map(|s| s.pod().clone()).collect(),
            _ => Vec::new(),
        };
        // The suite is told who the candidate is, not what the trainer says it can do.
        let identity = specialist.as_ref().map(SpecialistIdentity::of);
        let change = match (proposal.kind(), &identity) {
            (OpKind::Prune, _) => RosterChange::Remove(&removed_pods),
            (OpKind::Merge, Some(add)) => RosterChange::Replace {
                remove: &removed_pods,
                add,
            },
            (OpKind::Birth | OpKind::Reactivate, Some(add)) => RosterChange::Add(add),
            _ => RosterChange::Keep,
        };
        let measured = suite
            .clear_rates(&roster, &RosterChange::Keep)
            .and_then(|before| {
                suite
                    .clear_rates(&roster, &change)
                    .map(|after| (before, after))
            });
        let (before, after) = match measured {
            Ok(rates) => rates,
            Err(error) => {
                withdraw(vcw);
                return StructureResult::Refused(Refusal::Suite(error));
            }
        };
        // Every code of the serving roster and of the candidate must have been measured, in both
        // arms: a code the suite skips would be "nothing lost".
        if let Err(error) = check_coverage(&before, &roster, &change)
            .and_then(|()| check_coverage(&after, &roster, &change))
        {
            withdraw(vcw);
            return StructureResult::Refused(Refusal::Measurement(error));
        }
        let inputs = match gate_inputs(&before, &after, &change.focus()) {
            Ok(inputs) => inputs,
            Err(error) => {
                withdraw(vcw);
                return StructureResult::Refused(Refusal::Measurement(error));
            }
        };
        // 3. The gate report, bound to this proposal.
        let Some(base_digest) = expert_evidence_base_model_digest(expert_proposal.op()) else {
            withdraw(vcw);
            return StructureResult::Refused(Refusal::Gate("operation has no base model".into()));
        };
        let inputs_declared = specialist.as_ref().and_then(|s| s.inputs.clone());
        let artifact = match &inputs_declared {
            Some(inputs) => {
                expert_evidence_artifact_digest_with_inputs(expert_proposal.op(), inputs)
            }
            None => expert_evidence_artifact_digest(expert_proposal.op()),
        };
        let subject = GateSubject::new(
            *expert_proposal.proposal_id().as_bytes(),
            *expert_proposal.operation_id().as_bytes(),
            artifact,
            base_digest,
            suite.manifest_digest(),
        );
        let adapter = AdapterId::from(
            specialist
                .as_ref()
                .map_or("roster-change", |s| s.pod.0.as_str()),
        );
        let bound = match self
            .gate
            .evaluate(&adapter, &inputs.0, inputs.1)
            .map(|report| report.bind(subject))
        {
            Ok(bound) => bound,
            Err(error) => {
                withdraw(vcw);
                return StructureResult::Refused(Refusal::Gate(error.to_string()));
            }
        };
        // 4. Admission: the only way to a structural commit.
        let grant = grant(runtime);
        let admitted = match inputs_declared {
            Some(inputs) => {
                runtime.admit_expert_proposal_with_inputs(&expert_proposal, &bound, &grant, inputs)
            }
            None => runtime.admit_expert_proposal(&expert_proposal, &bound, &grant),
        };
        let commit = match admitted {
            Ok(commit) => commit,
            Err(error) => {
                withdraw(vcw);
                let gate_failed = matches!(
                    error,
                    RuntimeError::Expert(ExpertAdmissionError::GateFailed)
                );
                return StructureResult::Refused(Refusal::Admission {
                    gate_failed,
                    message: format!("{error:?}"),
                });
            }
        };
        // 5. Committed. Install the Pod and tell the loop. A failure here is not a refusal: the
        // change is in the ledger. The request stays outstanding in the loop, the driver keeps
        // what a retry needs, and the outcome says so.
        if let Err(message) = finish(
            &self.project,
            vcw,
            runtime,
            pods,
            &expert_proposal,
            specialist.as_ref(),
        ) {
            if self.pending.len() < MAX_PENDING {
                self.pending.push(PendingInstall {
                    commit,
                    proposal: expert_proposal.clone(),
                    specialist: specialist.clone(),
                    removed: removed_pods.clone(),
                    message: message.clone(),
                });
            }
            return StructureResult::CommittedNotInstalled { commit, message };
        }
        StructureResult::Admitted {
            commit,
            specialist: specialist.map(Box::new),
            removed: removed_pods,
        }
    }
}

/// Finish a committed change: register the specialist's Pod and enrol it (or, for a change with
/// no specialist, take the dead experts out of routing). Idempotent, so a retry after a partial
/// failure completes it.
fn finish<S: AdmissibleScorer>(
    project: &ProjectId,
    vcw: &mut VcwLoop<S>,
    runtime: &PtrRuntime,
    pods: &mut PodRegistry,
    proposal: &ExpertProposal,
    specialist: Option<&TrainedSpecialist>,
) -> Result<(), String> {
    match specialist {
        Some(specialist) => {
            install(pods, specialist).map_err(|error| format!("install: {error}"))?;
            vcw.sync_roster(project, runtime.expert_registry(), proposal)
                .map(|_| ())
                .map_err(|error| format!("sync_roster: {error}"))
        }
        None => vcw
            .retire_dead_experts(project, runtime.expert_registry())
            .map(|_| ())
            .map_err(|error| format!("retire_dead_experts: {error}")),
    }
}

/// Register the specialist's Pod; a reactivated expert takes over its own id. A Pod already
/// registered with this very manifest is left alone (a retry).
fn install(
    pods: &mut PodRegistry,
    specialist: &TrainedSpecialist,
) -> Result<(), PodRegistrationError> {
    let pod = specialist.executable();
    match pods.get(&specialist.manifest.project, &specialist.pod) {
        Some(existing) if existing.manifest().digest() == specialist.manifest.digest() => Ok(()),
        Some(existing) => pods.replace_if_current(&existing, pod).map(|_| ()),
        None => pods.register_new(pod),
    }
}
