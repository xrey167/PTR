//! The per-request verifier-closed loop.
//!
//! See the [module documentation](super) for the state, the data flow and the constraints. This
//! file holds the types a host sees to run one request ([`VcwLoop::run`]) and the loop itself.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use ptr_feedback::{Candidate, Critique, ExhaustionReason};
use ptr_pods::{
    Bid, CycleHost, PayloadRef, Proposal, ProposalInit, SubscribeRequest, SubscriberId,
    VerifierIdentity, Workspace, WorkspaceConfig, WorkspaceConfigInit, WorkspaceProposalId,
    WorkspaceSnapshot, VERIFIER_PANIC_CODE,
};
use ptr_protocol::TypedPayload;
use ptr_router::{
    classify, Accepted, AdmissibleScorer, AttentionSchema, CodeRoute, EscalationReason,
    FallbackAction, FallbackReason, FindingCode, FindingRegistry, FindingRouteRecord,
    FindingRouteRequest, FindingRouter, HintPolicy, NonceSequence, PodEligibility, PodRef,
    RegistryChange, RegistryUpdate, ReportClass, RouteClass, RouteOutcome, RouterConfig,
    RouterLimits, SamplingKey, SelfContext, SelfModel, SelfModelHint, SelfOutcome, SelfRole,
    SelfTraceStep, UniformScorer,
};
use ptr_types::{
    CandidateId, Digest, Generation, PodId, Probability, ProjectId, RequestId, TypeId,
    VerificationLevel,
};
use ptr_verifier::{Finding, VerificationReport, VerificationStatus, Verifier};
use sha2::{Digest as _, Sha256};

use super::budget::{BudgetMeter, RequestBudget};
use super::eligibility::AuthorStanding;
use super::l3::{OutstandingRequests, RequestTally, WindowRecorder};
use super::trace::{
    CandidateTrace, ClassSummary, OutcomeTag, ReportSummary, SealParts, SelfStepTrace,
    StepDecision, StepOrigin, TraceStep, VcwTrace, WorkspaceOutcome,
};
use super::VcwError;
use crate::expert::{unroutable_delta, StructurePolicy};

/// The payload of a proposal: anything the host's verifier can judge and that has a type and a
/// content digest. The loop never reads the bytes; it carries the value between Pods and the
/// verifier, and shows the workspace its type and digest.
pub trait VcwPayload: Clone {
    fn payload_type(&self) -> TypeId;

    /// SHA-256 (or any 32-byte digest) of the content. Equal content must give equal digests.
    fn payload_digest(&self) -> Digest;

    /// Whether `edited` differs from `self` only in the parts the finding codes `named` point
    /// at. A repair that touches anything else is discarded as a declined repair. The default
    /// accepts every edit, which leaves "only the named parts change" to the verifier, whose
    /// findings then name the collateral damage.
    fn confined_to(&self, edited: &Self, named: &[String]) -> bool {
        let _ = (edited, named);
        true
    }
}

impl VcwPayload for TypedPayload {
    fn payload_type(&self) -> TypeId {
        self.type_id.clone()
    }

    fn payload_digest(&self) -> Digest {
        Sha256::digest(&self.bytes).into()
    }
}

/// What a request needs, besides the Pods and the verifier.
#[derive(Clone, Debug)]
pub struct VcwRequest {
    pub request_id: RequestId,
    pub project: ProjectId,
    /// Digest of the request content; keys the router's keyed draws.
    pub request_digest: Digest,
    /// The verification level an accepted candidate must reach.
    pub required: VerificationLevel,
    pub budget: RequestBudget,
}

/// Why a Pod is being asked for a proposal.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct ProposeCall<'a> {
    pub pod: &'a PodRef,
    pub request_id: &'a RequestId,
    pub request_digest: Digest,
    pub round: u32,
    /// `None` for the first proposal; otherwise the fallback that asked for a fresh one.
    pub reason: Option<FallbackReason>,
    /// The latest broadcast: admitted content and refusals with their finding codes.
    pub snapshot: Option<&'a WorkspaceSnapshot>,
}

/// A repair Pod is asked to fix exactly one finding code of a rejected candidate.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct RepairCall<'a, P> {
    pub pod: &'a PodRef,
    pub request_id: &'a RequestId,
    pub round: u32,
    pub code: &'a FindingCode,
    /// The critique restricted to `code`: `issues` is that one code and `repairable` is true.
    pub critique: &'a Critique,
    pub previous: &'a P,
    pub snapshot: Option<&'a WorkspaceSnapshot>,
}

/// An escalated request: a stronger Pod or a human may produce a candidate. Whatever it returns
/// is verified like any other candidate.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct EscalateCall<'a, P> {
    pub request_id: &'a RequestId,
    pub round: u32,
    pub reason: EscalationReason,
    pub previous: Option<&'a P>,
    pub snapshot: Option<&'a WorkspaceSnapshot>,
}

/// The Pods of the roles, supplied by the host. In tests these are stubs; in a deployment they
/// invoke the registered Pods. A method returns `None` when the Pod produced nothing.
pub trait RolePods<P> {
    fn propose(&mut self, call: ProposeCall<'_>) -> Option<P>;

    fn repair(&mut self, call: RepairCall<'_, P>) -> Option<P>;

    /// A candidate from an escalation target, with the Pod that authored it. The author must be
    /// eligible, or the candidate is dropped. The default has no escalation target.
    fn escalate(&mut self, call: EscalateCall<'_, P>) -> Option<(PodRef, P)> {
        let _ = call;
        None
    }
}

/// Whether the loop may pass the self-model's hints on.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum HintMode {
    /// Hints go into the trace only. This is the default: the self-model observes.
    #[default]
    TraceOnly,
    /// After each step the loop calls `AttentionSchema::apply_hints` with the hints. They then
    /// act only if the scorer's arm uses them, and only inside the admissible set.
    ApplyToSchema,
}

/// Configuration of the loop. None of it can be changed by an observer.
#[derive(Clone, Debug)]
pub struct VcwConfig {
    /// The workspace parameters. `required_level` is replaced per request by the request's level,
    /// so the workspace gate and the classification always use one level.
    pub workspace: WorkspaceConfigInit,
    /// Name and configuration digest of the host verifier, recorded in every snapshot.
    pub verifier_name: String,
    pub verifier_config_digest: Digest,
    /// The bid every proposal asks for, and the ceiling the author registry grants.
    pub bid: Bid,
    /// Requests per L3 observation window.
    pub window_requests: u32,
    /// Closed windows kept for the structure policy (at most the policy's `max_windows`).
    pub max_windows: usize,
    pub hint_policy: HintPolicy,
    pub hint_mode: HintMode,
    /// Birth and Reactivate requests that may be outstanding at once.
    pub max_outstanding: usize,
    /// Windows after which an unresolved request expires (counted from the window count at which
    /// it was handed out), so a host that never resolves requests cannot block the policy.
    pub outstanding_ttl_windows: u32,
}

impl VcwConfig {
    /// A configuration with a small conservative workspace.
    pub fn new(verifier_name: &str, verifier_config_digest: Digest) -> Self {
        Self {
            workspace: WorkspaceConfigInit::conservative(4),
            verifier_name: verifier_name.to_owned(),
            verifier_config_digest,
            bid: Bid::new(1),
            window_requests: 8,
            max_windows: 8,
            hint_policy: HintPolicy::default(),
            hint_mode: HintMode::TraceOnly,
            max_outstanding: 64,
            outstanding_ttl_windows: 16,
        }
    }
}

/// The Pods the router may choose among, by role. Registered at construction; later growth only
/// through [`VcwLoop::sync_roster`], which requires the ledger-backed expert projection to show
/// the expert.
#[derive(Clone, Debug, Default)]
pub struct Roster {
    pub proposers: Vec<PodId>,
    pub repairers: Vec<(PodId, Vec<FindingCode>)>,
}

/// Everything [`VcwLoop::new`] needs.
pub struct VcwParts<S = UniformScorer> {
    pub config: VcwConfig,
    pub roster: Roster,
    pub router: RouterConfig,
    pub scorer: S,
    pub key: SamplingKey,
    pub nonces: NonceSequence,
    pub self_model: SelfModel,
    /// Observed (every routing record, every verified outcome). Whether it controls routing is
    /// decided by the scorer the host built, not here.
    pub schema: Option<Arc<AttentionSchema>>,
    pub structure: StructurePolicy,
}

/// Counters of the loop, saturating.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub struct VcwCounters {
    pub requests: u64,
    pub verified: u64,
    pub exhausted: u64,
    pub escalated: u64,
    pub candidates: u64,
    pub verifier_calls: u64,
    pub pod_calls: u64,
    pub repairs_applied: u64,
    pub repairs_declined: u64,
    pub repairs_out_of_scope: u64,
    pub regenerations: u64,
    pub unroutable_codes: u64,
    pub author_no_longer_eligible: u64,
    pub escalation_author_ineligible: u64,
    pub workspace_not_submitted: u64,
    pub windows_closed: u64,
    pub requests_refused_outstanding_full: u64,
    pub outstanding_expired: u64,
    /// Expired requests whose experts were found committed and were enrolled instead.
    pub outstanding_reconciled: u64,
}

/// A candidate that the verifier fabric accepted through `classify`, with what it was accepted
/// for. It is **provisional** (invariant 4): it is not committed, promoted or acted on.
/// [`admit_verified_answer`](super::admit_verified_answer) is the way into the runtime's normal
/// output admission, which verifies and checks authority again.
///
/// Only the loop can build one: the fields are private and it carries the sealed [`Accepted`]
/// proof that only `classify` produces.
///
/// ```compile_fail,E0451
/// use ptr_runtime::vcw::VerifiedAnswer;
/// fn forge(payload: ptr_protocol::TypedPayload) -> VerifiedAnswer<ptr_protocol::TypedPayload> {
///     VerifiedAnswer { payload }
/// }
/// ```
/// It is not `Clone`: [`admit_verified_answer`](super::admit_verified_answer) consumes it, so
/// one acceptance is one admission attempt.
#[derive(Debug)]
pub struct VerifiedAnswer<P> {
    payload: P,
    /// The Pod that made the last edit.
    author: PodRef,
    /// Every Pod whose output is in the payload, in the order it contributed (the last one
    /// included). Never empty.
    authors: Vec<PodRef>,
    project: ProjectId,
    request_id: RequestId,
    request_digest: Digest,
    required: VerificationLevel,
    report: ReportSummary,
    workspace: WorkspaceOutcome,
    _proof: Accepted,
}

impl<P> VerifiedAnswer<P> {
    pub fn payload(&self) -> &P {
        &self.payload
    }

    pub fn into_payload(self) -> P {
        self.payload
    }

    /// The Pod that made the last edit (the proposer when no repair applied).
    pub fn author(&self) -> &PodRef {
        &self.author
    }

    /// Every Pod whose output is in the payload. The bridge requires each to be live.
    pub fn authors(&self) -> &[PodRef] {
        &self.authors
    }

    /// The project of the request the answer was accepted for.
    pub fn project(&self) -> &ProjectId {
        &self.project
    }

    pub fn request_id(&self) -> &RequestId {
        &self.request_id
    }

    pub fn request_digest(&self) -> &Digest {
        &self.request_digest
    }

    /// The level the request required and the report reached.
    pub fn required(&self) -> VerificationLevel {
        self.required
    }

    pub fn report(&self) -> &ReportSummary {
        &self.report
    }

    /// What the workspace did with the accepted candidate (it may be refused as a duplicate or
    /// outbid without affecting the acceptance).
    pub fn workspace(&self) -> WorkspaceOutcome {
        self.workspace
    }
}

/// How a request ended. No variant carries a rejected candidate.
#[derive(Debug)]
pub enum VcwOutcome<P> {
    Verified(VerifiedAnswer<P>),
    /// A budget was spent without an accepted candidate.
    Exhausted {
        reason: ExhaustionReason,
    },
    /// The fallback policy, or the lack of an eligible proposer, or an escalation with no
    /// target, hands the request to the host. Nothing is returned in place of an answer.
    Escalated {
        reason: EscalationReason,
    },
}

/// A finished request: the outcome and the trace.
#[derive(Debug)]
pub struct VcwRun<P> {
    pub outcome: VcwOutcome<P>,
    pub trace: VcwTrace,
}

/// The verifier-closed loop, with its router, self-model, schema observer, structure policy and
/// L3 window state. It owns no gate: the verifier, the required level and the workspace
/// parameters are per-request inputs and configuration it never mutates.
pub struct VcwLoop<S = UniformScorer> {
    pub(super) config: VcwConfig,
    pub(super) router: FindingRouter<S>,
    pub(super) limits: RouterLimits,
    pub(super) self_model: SelfModel,
    pub(super) schema: Option<Arc<AttentionSchema>>,
    pub(super) structure: StructurePolicy,
    pub(super) proposers: BTreeSet<PodId>,
    pub(super) repairers: BTreeMap<PodId, BTreeSet<FindingCode>>,
    pub(super) windows: WindowRecorder,
    pub(super) outstanding: OutstandingRequests,
    pub(super) hint_book: BTreeMap<FindingCode, SelfModelHint>,
    pub(super) failed_trace: Option<VcwTrace>,
    pub(super) counters: VcwCounters,
    /// Fault injection for the unit tests of the error paths.
    #[cfg(test)]
    pub(super) fault: Faults,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Faults {
    pub(super) retire: bool,
    pub(super) judge: bool,
}

/// Preference hints the loop remembers for the schema (the schema itself keeps at most its
/// `max_preferences`).
const MAX_HINT_BOOK: usize = 64;

/// Most repair experts the roster holds (matches the structure policy's expert limit).
pub(super) const MAX_ROSTER: usize = 256;

impl<S: AdmissibleScorer> VcwLoop<S> {
    /// # Errors
    /// [`VcwError`] for an invalid configuration: workspace parameters, verifier name, window
    /// sizes, a fallback policy that cannot ask for a first proposal, or a roster over the
    /// registry's capacity.
    pub fn new(parts: VcwParts<S>) -> Result<Self, VcwError> {
        let VcwParts {
            config,
            roster,
            router,
            scorer,
            key,
            nonces,
            self_model,
            schema,
            structure,
        } = parts;
        WorkspaceConfig::new(config.workspace).map_err(VcwError::WorkspaceConfig)?;
        if VerifierIdentity::new(&config.verifier_name, config.verifier_config_digest).is_none() {
            return Err(VcwError::BadVerifierIdentity);
        }
        if config.window_requests == 0 {
            return Err(VcwError::Config("window_requests is zero"));
        }
        let policy_windows = structure.config().limits.max_windows;
        if config.max_windows == 0 || config.max_windows > policy_windows {
            return Err(VcwError::Config(
                "max_windows is zero or above the structure policy's limit",
            ));
        }
        if config.outstanding_ttl_windows == 0 {
            return Err(VcwError::Config("outstanding_ttl_windows is zero"));
        }
        if config.max_outstanding == 0 || config.max_outstanding > MAX_ROSTER {
            return Err(VcwError::Config("max_outstanding is zero or above 256"));
        }
        // The first proposal is drawn through a regenerate route of an `Unknown` report; a
        // policy that escalates `Unknown` could never ask a proposer.
        if router.fallback.unknown != FallbackAction::Regenerate {
            return Err(VcwError::Config(
                "the fallback policy must regenerate Unknown (the first proposal is drawn so)",
            ));
        }
        let mut registry = FindingRegistry::new();
        let mut proposers = BTreeSet::new();
        let mut repairers: BTreeMap<PodId, BTreeSet<FindingCode>> = BTreeMap::new();
        for pod in roster.proposers {
            if proposers.len() >= MAX_ROSTER {
                return Err(VcwError::Config("more than 256 proposers"));
            }
            registry
                .apply(RegistryUpdate {
                    expected: registry.generation(),
                    change: RegistryChange::AddProposer { pod: pod.clone() },
                })
                .map_err(VcwError::Registry)?;
            proposers.insert(pod);
        }
        for (pod, codes) in roster.repairers {
            if repairers.len() >= MAX_ROSTER && !repairers.contains_key(&pod) {
                return Err(VcwError::Config("more than 256 repair experts"));
            }
            for code in codes {
                registry
                    .apply(RegistryUpdate {
                        expected: registry.generation(),
                        change: RegistryChange::AddRepair {
                            code: code.clone(),
                            pod: pod.clone(),
                        },
                    })
                    .map_err(VcwError::Registry)?;
                repairers.entry(pod.clone()).or_default().insert(code);
            }
        }
        let limits = router.limits;
        let windows = WindowRecorder::new(
            config.window_requests,
            config.max_windows,
            &structure.config().limits,
        );
        let outstanding = OutstandingRequests::new(config.max_outstanding);
        Ok(Self {
            router: FindingRouter::new(registry, router, scorer, key, nonces),
            limits,
            self_model,
            schema,
            structure,
            proposers,
            repairers,
            windows,
            outstanding,
            hint_book: BTreeMap::new(),
            failed_trace: None,
            #[cfg(test)]
            fault: Faults::default(),
            counters: VcwCounters::default(),
            config,
        })
    }

    pub fn counters(&self) -> VcwCounters {
        self.counters
    }

    pub fn router(&self) -> &FindingRouter<S> {
        &self.router
    }

    pub fn self_model(&self) -> &SelfModel {
        &self.self_model
    }

    pub fn schema(&self) -> Option<&Arc<AttentionSchema>> {
        self.schema.as_ref()
    }

    /// The partial trace of the most recent request that failed with a [`VcwError`] (one slot;
    /// the next failure replaces it). Budget spent by a failed request is in the counters.
    pub fn failed_trace(&self) -> Option<&VcwTrace> {
        self.failed_trace.as_ref()
    }

    /// Closed observation windows so far (the host runs the structure policy after one closes).
    pub fn windows_closed(&self) -> u64 {
        self.counters.windows_closed
    }

    /// The Pods registered as proposers.
    pub fn proposers(&self) -> &BTreeSet<PodId> {
        &self.proposers
    }

    /// The Pods registered as repairers and the codes each repairs.
    pub fn repairers(&self) -> &BTreeMap<PodId, BTreeSet<FindingCode>> {
        &self.repairers
    }

    /// Run one request.
    ///
    /// `standing` is the Pod registry view (see [`RuntimeStanding`](super::RuntimeStanding)):
    /// the router chooses only Pods it lists and the workspace admits only authors it lists at
    /// their generation. `verifier` is the host's fabric over the payload type; each candidate
    /// is verified once.
    ///
    /// # Errors
    /// [`VcwError`] when the loop itself fails (workspace, routing or internal errors). A request
    /// that merely fails to find an answer is `Ok` with an `Exhausted` or `Escalated` outcome.
    pub fn run<P: VcwPayload>(
        &mut self,
        request: &VcwRequest,
        standing: &dyn PodEligibility,
        verifier: &dyn Verifier<P>,
        pods: &mut dyn RolePods<P>,
    ) -> Result<VcwRun<P>, VcwError> {
        self.counters.requests = self.counters.requests.saturating_add(1);
        let unroutable_before = self.router.unroutable_counts();
        let overflow_before = self.router.unroutable_overflow();
        let mut session = Session::new(self, request, standing, verifier)?;
        let result = self.drive(&mut session, pods);
        // The tally is closed even when the loop failed: what was routed was routed.
        let unroutable_after = self.router.unroutable_counts();
        session.tally.unroutable = unroutable_delta(&unroutable_before, &unroutable_after);
        session.tally.untracked = self
            .router
            .unroutable_overflow()
            .saturating_sub(overflow_before);
        let closed = self.windows.record(std::mem::take(&mut session.tally));
        if closed {
            self.counters.windows_closed = self.counters.windows_closed.saturating_add(1);
        }
        // What was spent is accounted whether or not the request finished.
        let spent = session.meter.used();
        self.counters.verifier_calls = self
            .counters
            .verifier_calls
            .saturating_add(u64::from(spent.verifier_calls));
        self.counters.pod_calls = self
            .counters
            .pod_calls
            .saturating_add(u64::from(spent.pod_calls));
        let seal = |tag: OutcomeTag, session: Session<'_, P>, model: [u8; 32]| {
            VcwTrace::seal(SealParts {
                request_id: request.request_id.clone(),
                request_digest: request.request_digest,
                required: request.required,
                budget: session.meter.budget(),
                steps: session.steps,
                spent,
                outcome: tag,
                final_snapshot_digest: session.latest.as_ref().map(|s| *s.digest()),
                self_model_digest: model,
            })
        };
        let model = self.self_model.digest();
        let finished = match result {
            Ok(finished) => finished,
            Err(error) => {
                self.failed_trace = Some(seal(OutcomeTag::Failed, session, model));
                return Err(error);
            }
        };
        let (outcome, tag) = match finished {
            Finished::Verified(answer) => {
                self.counters.verified = self.counters.verified.saturating_add(1);
                (VcwOutcome::Verified(*answer), OutcomeTag::Verified)
            }
            Finished::Exhausted(reason) => {
                self.counters.exhausted = self.counters.exhausted.saturating_add(1);
                (
                    VcwOutcome::Exhausted { reason },
                    OutcomeTag::Exhausted(reason),
                )
            }
            Finished::Escalated(reason) => {
                self.counters.escalated = self.counters.escalated.saturating_add(1);
                (VcwOutcome::Escalated { reason }, OutcomeTag::Escalated)
            }
        };
        let trace = seal(tag, session, model);
        Ok(VcwRun { outcome, trace })
    }
}

enum Finished<P> {
    Verified(Box<VerifiedAnswer<P>>),
    Exhausted(ExhaustionReason),
    Escalated(EscalationReason),
}

/// The verifier reports the loop already obtained, by proposal. The workspace cycle's host
/// verifier is this cache, so a candidate is never verified twice and the workspace judges
/// exactly the report `classify` saw. An unknown proposal gets `Unknown`.
#[derive(Default)]
struct ReportCache(RefCell<BTreeMap<WorkspaceProposalId, VerificationReport>>);

impl Verifier<Proposal> for ReportCache {
    fn verify(&self, candidate: &Proposal) -> VerificationReport {
        self.0
            .borrow()
            .get(&candidate.id())
            .cloned()
            .unwrap_or_else(|| unknown_report(Vec::new()))
    }
}

fn unknown_report(findings: Vec<Finding>) -> VerificationReport {
    VerificationReport {
        status: VerificationStatus::Unknown,
        level: VerificationLevel::Unverified,
        score: Probability::default(),
        findings,
    }
}

/// A report that carries no evidence, used only to ask the router for a Pod.
fn synthetic(status: VerificationStatus) -> VerificationReport {
    VerificationReport {
        status,
        level: VerificationLevel::Unverified,
        score: Probability::default(),
        findings: Vec::new(),
    }
}

struct Previous {
    report: VerificationReport,
    refusal_codes: Vec<String>,
    author: PodRef,
}

struct Session<'a, P: VcwPayload> {
    request: &'a VcwRequest,
    standing: &'a dyn PodEligibility,
    verifier: &'a dyn Verifier<P>,
    authors: AuthorStanding<'a>,
    identity: VerifierIdentity,
    workspace: Workspace,
    cache: ReportCache,
    subscribers: BTreeMap<(PodId, Generation), SubscriberId>,
    latest: Option<Arc<WorkspaceSnapshot>>,
    meter: BudgetMeter,
    steps: Vec<TraceStep>,
    prev: Option<Previous>,
    tally: RequestTally,
}

impl<'a, P: VcwPayload> Session<'a, P> {
    fn new<S: AdmissibleScorer>(
        owner: &VcwLoop<S>,
        request: &'a VcwRequest,
        standing: &'a dyn PodEligibility,
        verifier: &'a dyn Verifier<P>,
    ) -> Result<Self, VcwError> {
        let mut init = owner.config.workspace;
        init.required_level = request.required;
        let workspace =
            Workspace::new(WorkspaceConfig::new(init).map_err(VcwError::WorkspaceConfig)?);
        let identity = VerifierIdentity::new(
            &owner.config.verifier_name,
            owner.config.verifier_config_digest,
        )
        .ok_or(VcwError::BadVerifierIdentity)?;
        Ok(Self {
            request,
            standing,
            verifier,
            authors: AuthorStanding::new(standing, owner.config.bid),
            identity,
            workspace,
            cache: ReportCache::default(),
            subscribers: BTreeMap::new(),
            latest: None,
            meter: BudgetMeter::new(request.budget),
            steps: Vec::new(),
            prev: None,
            tally: RequestTally::default(),
        })
    }

    fn snapshot(&self) -> Option<&WorkspaceSnapshot> {
        self.latest.as_deref()
    }

    /// Subscribe a Pod generation to the broadcast; a full or duplicate subscription is not an
    /// error of the request.
    fn subscribe(&mut self, pod: &PodRef) {
        let key = (pod.pod.clone(), pod.generation);
        if self.subscribers.contains_key(&key) {
            return;
        }
        if let Ok(id) = self.workspace.subscribe(SubscribeRequest {
            pod: pod.pod.clone(),
            generation: pod.generation,
        }) {
            self.subscribers.insert(key, id);
        }
    }

    /// Take everything waiting in the mailboxes; the newest snapshot is the Pods' view.
    fn drain_mailboxes(&mut self) {
        for id in self.subscribers.values() {
            while let Ok(Some(snapshot)) = self.workspace.receive(*id) {
                self.latest = Some(snapshot);
            }
        }
    }
}

enum Plan<P> {
    Propose {
        pod: PodRef,
        reason: Option<FallbackReason>,
    },
    Repair {
        previous: P,
        actions: Vec<(PodRef, FindingCode)>,
    },
    Escalate {
        reason: EscalationReason,
        previous: Option<P>,
    },
}

struct Attempt<P> {
    origin: StepOrigin,
    /// What the self-model is told the step is. A repair routing that the Pod declined and that
    /// was replaced by a fresh draw keeps `Repair`: the outcome belongs to the routing decision.
    role: SelfRole,
    routes: Vec<FindingRouteRecord>,
    unroutable: Vec<FindingCode>,
    plan: Plan<P>,
}

enum Produced<P> {
    /// A candidate and every Pod whose output is in it: the proposer, the repairers whose edits
    /// were applied (in order), or the escalation author. Never empty.
    Candidate {
        value: P,
        authors: Vec<PodRef>,
    },
    Nothing,
    /// Every repair Pod declined (or its edit left the named parts).
    Declined,
}

impl<S: AdmissibleScorer> VcwLoop<S> {
    fn drive<P: VcwPayload>(
        &mut self,
        session: &mut Session<'_, P>,
        pods: &mut dyn RolePods<P>,
    ) -> Result<Finished<P>, VcwError> {
        let mut attempt = self.route_fallback(session, FallbackReason::Unknown, None)?;
        if matches!(attempt.plan, Plan::Propose { .. }) {
            attempt.origin = StepOrigin::Initial;
        }
        let mut first = true;
        loop {
            if !first {
                if let Err(reason) = session.meter.charge_round() {
                    return Ok(Finished::Exhausted(reason));
                }
            }
            first = false;
            let round = session.meter.used().rounds;
            let mut produced = self.execute(session, pods, &attempt, round);
            if matches!(produced, Produced::Declined) {
                // ptr-feedback's rule: a declined repair is followed by a fresh draw in the same
                // round, as a fallback of a failing report with no usable finding. The fallback
                // policy decides: Regenerate draws now, Escalate hands over in the next round.
                self.counters.regenerations = self.counters.regenerations.saturating_add(1);
                let previous = match &attempt.plan {
                    Plan::Repair { previous, .. } => Some(previous.clone()),
                    Plan::Propose { .. } | Plan::Escalate { .. } => None,
                };
                let mut next =
                    self.route_fallback(session, FallbackReason::FailWithoutHardFinding, previous)?;
                next.role = attempt.role;
                if matches!(next.plan, Plan::Escalate { .. }) {
                    self.push_step(session, &attempt, round, &[], None, StepDecision::Escalate);
                    attempt = next;
                    continue;
                }
                next.routes
                    .splice(0..0, std::mem::take(&mut attempt.routes));
                next.unroutable
                    .splice(0..0, std::mem::take(&mut attempt.unroutable));
                attempt = next;
                produced = self.execute(session, pods, &attempt, round);
            }
            let (value, authors) = match produced {
                Produced::Candidate { value, authors } => (value, authors),
                Produced::Nothing | Produced::Declined => {
                    self.push_step(
                        session,
                        &attempt,
                        round,
                        &[],
                        None,
                        StepDecision::NoCandidate,
                    );
                    // An escalation nobody answered is handed to the host.
                    if let Plan::Escalate { reason, .. } = &attempt.plan {
                        return Ok(Finished::Escalated(*reason));
                    }
                    attempt = self.route_fallback(session, FallbackReason::Unknown, None)?;
                    continue;
                }
            };
            self.counters.candidates = self.counters.candidates.saturating_add(1);
            let judged = match self.judge(session, &attempt, &value, &authors) {
                Ok(judged) => judged,
                Err(error) => {
                    // The candidate was produced (and usually verified) and spent budget: keep
                    // it in the partial trace.
                    self.push_step(
                        session,
                        &attempt,
                        round,
                        &authors,
                        None,
                        StepDecision::JudgeFailed,
                    );
                    return Err(error);
                }
            };
            let mut decision = StepDecision::NoCandidate;
            if let ReportClass::Accept(proof) = &judged.class {
                if judged.authors_eligible {
                    let author = authors
                        .last()
                        .cloned()
                        .ok_or(VcwError::Internal("a candidate without an author"))?;
                    let answer = VerifiedAnswer {
                        payload: value,
                        author,
                        authors: authors.clone(),
                        project: session.request.project.clone(),
                        request_id: session.request.request_id.clone(),
                        request_digest: session.request.request_digest,
                        required: session.request.required,
                        report: judged.summary.clone(),
                        workspace: judged.workspace,
                        _proof: *proof,
                    };
                    self.push_step(
                        session,
                        &attempt,
                        round,
                        &authors,
                        Some(judged.trace),
                        StepDecision::Accepted,
                    );
                    return Ok(Finished::Verified(Box::new(answer)));
                }
                self.counters.author_no_longer_eligible =
                    self.counters.author_no_longer_eligible.saturating_add(1);
                decision = StepDecision::AuthorNoLongerEligible;
            }
            if let Some(reason) = session.meter.stop_after_verdict() {
                self.push_step(
                    session,
                    &attempt,
                    round,
                    &authors,
                    Some(judged.trace),
                    StepDecision::Stopped(reason),
                );
                return Ok(Finished::Exhausted(reason));
            }
            let (next, next_decision) = match self.route_next(session, &judged, value) {
                Ok(routed) => routed,
                Err(error) => {
                    // The candidate was verified and spent budget; keep it in the partial trace.
                    self.push_step(
                        session,
                        &attempt,
                        round,
                        &authors,
                        Some(judged.trace),
                        StepDecision::RoutingFailed,
                    );
                    return Err(error);
                }
            };
            if decision != StepDecision::AuthorNoLongerEligible {
                decision = next_decision;
            }
            self.push_step(
                session,
                &attempt,
                round,
                &authors,
                Some(judged.trace),
                decision,
            );
            attempt = next;
        }
    }

    /// Record an attempt. `authors` are the Pods whose output is in the candidate (empty when
    /// the attempt produced none), not the Pods the plan named.
    fn push_step<P: VcwPayload>(
        &mut self,
        session: &mut Session<'_, P>,
        attempt: &Attempt<P>,
        round: u32,
        authors: &[PodRef],
        candidate: Option<CandidateTrace>,
        decision: StepDecision,
    ) {
        session.steps.push(TraceStep {
            round,
            origin: attempt.origin,
            authors: authors.to_vec(),
            routes: attempt.routes.clone(),
            unroutable: attempt.unroutable.clone(),
            candidate,
            decision,
            budget_after: session.meter.used(),
        });
    }

    /// Ask the router for a Pod through a fallback of a report that carries no evidence. The
    /// fallback policy decides between a proposer (Regenerate) and the escalation target.
    fn route_fallback<P: VcwPayload>(
        &mut self,
        session: &mut Session<'_, P>,
        reason: FallbackReason,
        previous: Option<P>,
    ) -> Result<Attempt<P>, VcwError> {
        let status = match reason {
            FallbackReason::FailWithoutHardFinding => VerificationStatus::Fail,
            _ => VerificationStatus::Unknown,
        };
        let report = synthetic(status);
        let outcome = self.router_route(session, &report)?;
        match outcome {
            RouteOutcome::Regenerate { reason, proposer } => {
                self.observe_records(std::slice::from_ref(&proposer));
                Ok(Attempt {
                    origin: StepOrigin::Regenerated,
                    role: SelfRole::Proposal,
                    plan: Plan::Propose {
                        pod: PodRef::new(proposer.pod().clone(), proposer.pod_generation()),
                        reason: Some(reason),
                    },
                    routes: vec![proposer],
                    unroutable: Vec::new(),
                })
            }
            // Every escalation, whichever path decided it, goes to the escalation target.
            RouteOutcome::Escalate { reason } => Ok(Attempt {
                origin: StepOrigin::Escalated,
                role: SelfRole::Proposal,
                plan: Plan::Escalate { reason, previous },
                routes: Vec::new(),
                unroutable: Vec::new(),
            }),
            RouteOutcome::Accept(_) | RouteOutcome::Repair { .. } => {
                Err(VcwError::Internal("a report without evidence was routed"))
            }
        }
    }

    fn router_route<P: VcwPayload>(
        &self,
        session: &Session<'_, P>,
        report: &VerificationReport,
    ) -> Result<RouteOutcome, VcwError> {
        self.router
            .route(&FindingRouteRequest {
                request_id: &session.request.request_id,
                request_digest: session.request.request_digest,
                report,
                required: session.request.required,
                eligibility: session.standing,
            })
            .map_err(VcwError::Route)
    }

    fn observe_records(&self, records: &[FindingRouteRecord]) {
        if let Some(schema) = &self.schema {
            for record in records {
                schema.observe(record);
            }
        }
    }

    fn execute<P: VcwPayload>(
        &mut self,
        session: &mut Session<'_, P>,
        pods: &mut dyn RolePods<P>,
        attempt: &Attempt<P>,
        round: u32,
    ) -> Produced<P> {
        match &attempt.plan {
            Plan::Propose { pod, reason } => {
                session.subscribe(pod);
                session.meter.count_pod_call();
                let call = ProposeCall {
                    pod,
                    request_id: &session.request.request_id,
                    request_digest: session.request.request_digest,
                    round,
                    reason: *reason,
                    snapshot: session.snapshot(),
                };
                match pods.propose(call) {
                    Some(value) => Produced::Candidate {
                        value,
                        authors: vec![pod.clone()],
                    },
                    None => Produced::Nothing,
                }
            }
            Plan::Repair { previous, actions } => {
                let mut current = previous.clone();
                let mut applied: Vec<PodRef> = Vec::new();
                for (pod, code) in actions {
                    session.subscribe(pod);
                    session.meter.count_pod_call();
                    let critique = Critique {
                        candidate: CandidateId::from(format!("r{round}").as_str()),
                        issues: vec![code.as_str().to_owned()],
                        repairable: true,
                    };
                    let edit = pods.repair(RepairCall {
                        pod,
                        request_id: &session.request.request_id,
                        round,
                        code,
                        critique: &critique,
                        previous: &current,
                        snapshot: session.snapshot(),
                    });
                    match edit {
                        Some(edited) if current.confined_to(&edited, &critique.issues) => {
                            self.counters.repairs_applied =
                                self.counters.repairs_applied.saturating_add(1);
                            current = edited;
                            if !applied.contains(pod) {
                                applied.push(pod.clone());
                            }
                        }
                        Some(_) => {
                            self.counters.repairs_out_of_scope =
                                self.counters.repairs_out_of_scope.saturating_add(1);
                        }
                        None => {
                            self.counters.repairs_declined =
                                self.counters.repairs_declined.saturating_add(1);
                        }
                    }
                }
                if applied.is_empty() {
                    Produced::Declined
                } else {
                    Produced::Candidate {
                        value: current,
                        authors: applied,
                    }
                }
            }
            Plan::Escalate { reason, previous } => {
                session.meter.count_pod_call();
                let call = EscalateCall {
                    request_id: &session.request.request_id,
                    round,
                    reason: *reason,
                    previous: previous.as_ref(),
                    snapshot: session.snapshot(),
                };
                match pods.escalate(call) {
                    Some((author, value))
                        if session.standing.current_generation(&author.pod)
                            == Some(author.generation) =>
                    {
                        session.subscribe(&author);
                        Produced::Candidate {
                            value,
                            authors: vec![author],
                        }
                    }
                    Some(_) => {
                        self.counters.escalation_author_ineligible =
                            self.counters.escalation_author_ineligible.saturating_add(1);
                        Produced::Nothing
                    }
                    None => Produced::Nothing,
                }
            }
        }
    }
}

struct Judged {
    class: ReportClass,
    summary: ReportSummary,
    workspace: WorkspaceOutcome,
    trace: CandidateTrace,
    report: VerificationReport,
    /// Every Pod whose output is in the candidate is still eligible at its generation.
    authors_eligible: bool,
}

impl<S: AdmissibleScorer> VcwLoop<S> {
    /// Verify the candidate once, classify the report, let the workspace judge the same report
    /// and let the observers see the outcome.
    ///
    /// Acceptance does not depend on what the workspace did with the candidate: the workspace
    /// integrates and broadcasts, the verifier is the gate. A workspace refusal for a reason
    /// other than authority (capacity, bid, duplicate) is recorded in the trace, never hidden
    /// and never a veto.
    fn judge<P: VcwPayload>(
        &mut self,
        session: &mut Session<'_, P>,
        attempt: &Attempt<P>,
        value: &P,
        authors: &[PodRef],
    ) -> Result<Judged, VcwError> {
        let author = authors
            .last()
            .ok_or(VcwError::Internal("a candidate without an author"))?;
        let required = session.request.required;
        let calls_left = session.meter.calls_left();
        let context = self.context_for(session, attempt, calls_left);
        session
            .meter
            .charge_verifier()
            .map_err(|_| VcwError::Internal("verifier budget breached"))?;
        #[cfg(test)]
        if self.fault.judge {
            return Err(VcwError::Internal("injected judge fault"));
        }
        let verdict = catch_unwind(AssertUnwindSafe(|| session.verifier.verify(value)));
        let report = verdict.unwrap_or_else(|_| {
            unknown_report(vec![Finding {
                code: VERIFIER_PANIC_CODE.to_owned(),
                message: String::new(),
                hard: true,
            }])
        });

        // The classification. `Accept` is decided here and nowhere else.
        let class = classify(&report, required, &self.limits);
        // Defence in depth: the sibling restatement of the accept predicate must agree.
        let passes = Candidate {
            id: CandidateId::from("c"),
            value: (),
            verification: Some(report.clone()),
        }
        .passes(required);
        if matches!(class, ReportClass::Accept(_)) != passes {
            return Err(VcwError::GateDisagreement);
        }

        // The workspace judges the report just obtained.
        let payload_digest = value.payload_digest();
        let payload_type = value.payload_type();
        let proposal_id = Proposal::new(ProposalInit {
            author: author.pod.clone(),
            author_generation: author.generation,
            content: payload_digest,
            payload: PayloadRef::new(payload_type.clone(), payload_digest),
            computed_against: session.workspace.revision(),
            bid: self.config.bid,
        })
        .map_err(|_| "invalid-proposal")
        .and_then(|proposal| {
            let id = proposal.id();
            session.cache.0.borrow_mut().insert(id, report.clone());
            session
                .workspace
                .submit(proposal)
                .map(|_| id)
                .map_err(|_| "submit-refused")
        });
        let cycle = session
            .workspace
            .run_cycle(CycleHost {
                verifier: &session.cache,
                identity: &session.identity,
                authors: &session.authors,
            })
            .map_err(VcwError::Workspace)?;
        session
            .workspace
            .check_invariants()
            .map_err(VcwError::Workspace)?;
        for revoked in &cycle.revoked_subscribers {
            session.subscribers.retain(|_, id| id != revoked);
        }
        let snapshot = cycle.snapshot;
        let (workspace, refusal_codes) = match &proposal_id {
            Err(why) => {
                self.counters.workspace_not_submitted =
                    self.counters.workspace_not_submitted.saturating_add(1);
                (WorkspaceOutcome::NotSubmitted(why), Vec::new())
            }
            Ok(id) if snapshot.admitted().contains(id) => (WorkspaceOutcome::Admitted, Vec::new()),
            Ok(id) => match snapshot.refusals().iter().find(|r| r.proposal() == *id) {
                Some(refusal) => (
                    WorkspaceOutcome::Refused(refusal.reason().kind()),
                    refusal
                        .findings()
                        .iter()
                        .map(|finding| finding.code().to_owned())
                        .collect(),
                ),
                None => (WorkspaceOutcome::Unlisted, Vec::new()),
            },
        };
        session.latest = Some(Arc::clone(&snapshot));
        session.drain_mailboxes();

        // Liveness of every Pod whose output is in the candidate.
        let authors_eligible = authors
            .iter()
            .all(|a| session.standing.current_generation(&a.pod) == Some(a.generation));

        // Observers. The self-model predicts first, then learns (prequential); its label is the
        // verifier's verdict. The schema and the L3 tally credit a routed Pod only when the
        // candidate was accepted *and* can be returned.
        let passed = matches!(class, ReportClass::Accept(_));
        let returned = passed && authors_eligible;
        let outcome =
            SelfOutcome::from_report(&report, required, &self.limits, attempt.routes.first());
        let prediction = self.self_model.observe(&SelfTraceStep { context, outcome });
        let next_context = SelfContext::from_report(
            SelfRole::Proposal,
            Some(author.clone()),
            session.meter.used().rounds,
            session.meter.calls_left(),
            required,
            &report,
        )
        .with_refusal_codes(refusal_codes.iter().map(String::as_str));
        let hints = self
            .self_model
            .hints(&next_context, &self.config.hint_policy);
        self.apply_hints_if_configured(&next_context, &hints);
        if let Some(schema) = &self.schema {
            for record in &attempt.routes {
                schema.settle(record, returned);
            }
        }
        session.tally.note(&attempt.routes, returned);

        let summary = ReportSummary::of(&report);
        let class_summary = match &class {
            ReportClass::Accept(_) => ClassSummary::Accept,
            ReportClass::Repair(codes) => ClassSummary::Repair(codes.clone()),
            ReportClass::Fallback(reason) => ClassSummary::Fallback(*reason),
        };
        let trace = CandidateTrace {
            payload_type,
            payload_digest,
            report: summary.clone(),
            class: class_summary,
            workspace,
            workspace_revision: Some(snapshot.revision().get()),
            snapshot_digest: Some(*snapshot.digest()),
            self_model: SelfStepTrace {
                prediction,
                passed,
                hints: hints.len(),
            },
        };
        session.prev = Some(Previous {
            report: report.clone(),
            refusal_codes,
            author: author.clone(),
        });
        Ok(Judged {
            class,
            summary,
            workspace,
            trace,
            report,
            authors_eligible,
        })
    }

    /// Under [`HintMode::ApplyToSchema`], pass the hints to `AttentionSchema::apply_hints`, the
    /// only way a hint can reach routing. `apply_hints` replaces the stored preferences
    /// wholesale, so the loop keeps a bounded book of the latest `PreferRepairPod` hint per code:
    /// the codes of this report are refreshed (a code whose hint no longer qualifies drops out),
    /// the others stay until the schema expires them.
    fn apply_hints_if_configured(&mut self, context: &SelfContext, hints: &[SelfModelHint]) {
        if self.config.hint_mode != HintMode::ApplyToSchema {
            return;
        }
        let Some(schema) = &self.schema else {
            return;
        };
        for code in &context.last_codes {
            self.hint_book.remove(code);
        }
        for hint in hints {
            if let SelfModelHint::PreferRepairPod { code, .. } = hint {
                self.hint_book.insert(code.clone(), hint.clone());
            }
        }
        while self.hint_book.len() > MAX_HINT_BOOK {
            self.hint_book.pop_last();
        }
        let book: Vec<SelfModelHint> = self.hint_book.values().cloned().collect();
        schema.apply_hints(&book);
    }

    /// The self-model's context for the candidate about to be verified.
    fn context_for<P: VcwPayload>(
        &self,
        session: &Session<'_, P>,
        attempt: &Attempt<P>,
        calls_left: u32,
    ) -> SelfContext {
        let role = attempt.role;
        let round = session.meter.used().rounds;
        let required = session.request.required;
        match &session.prev {
            Some(prev) => SelfContext::from_report(
                role,
                Some(prev.author.clone()),
                round,
                calls_left,
                required,
                &prev.report,
            )
            .with_refusal_codes(prev.refusal_codes.iter().map(String::as_str)),
            None => SelfContext {
                role,
                producer: None,
                round,
                budget_left: calls_left,
                required,
                last_codes: Vec::new(),
            },
        }
    }

    /// Route the step that follows a rejected (or ineligible-author) verdict. Every outcome is
    /// an attempt, an escalation included.
    fn route_next<P: VcwPayload>(
        &mut self,
        session: &mut Session<'_, P>,
        judged: &Judged,
        value: P,
    ) -> Result<(Attempt<P>, StepDecision), VcwError> {
        let escalating = |attempt: &Attempt<P>| matches!(attempt.plan, Plan::Escalate { .. });
        // An accepted candidate whose author is no longer eligible is replaced by a fresh draw.
        if matches!(judged.class, ReportClass::Accept(_)) {
            self.counters.regenerations = self.counters.regenerations.saturating_add(1);
            let attempt = self.route_fallback(session, FallbackReason::Unknown, Some(value))?;
            return Ok((attempt, StepDecision::AuthorNoLongerEligible));
        }
        let outcome = self.router_route(session, &judged.report)?;
        match outcome {
            RouteOutcome::Accept(_) => Err(VcwError::GateDisagreement),
            RouteOutcome::Regenerate { reason, proposer } => {
                self.counters.regenerations = self.counters.regenerations.saturating_add(1);
                self.observe_records(std::slice::from_ref(&proposer));
                Ok((
                    Attempt {
                        origin: StepOrigin::Regenerated,
                        role: SelfRole::Proposal,
                        plan: Plan::Propose {
                            pod: PodRef::new(proposer.pod().clone(), proposer.pod_generation()),
                            reason: Some(reason),
                        },
                        routes: vec![proposer],
                        unroutable: Vec::new(),
                    },
                    StepDecision::Regenerate,
                ))
            }
            RouteOutcome::Escalate { reason } => Ok((
                // The escalation target gets the previous candidate; it is not returned.
                Attempt {
                    origin: StepOrigin::Escalated,
                    role: SelfRole::Proposal,
                    plan: Plan::Escalate {
                        reason,
                        previous: Some(value),
                    },
                    routes: Vec::new(),
                    unroutable: Vec::new(),
                },
                StepDecision::Escalate,
            )),
            RouteOutcome::Repair { routes } => {
                let mut records = Vec::new();
                let mut actions = Vec::new();
                let mut unroutable = Vec::new();
                for route in routes {
                    match route {
                        CodeRoute::Routed(record) => {
                            if let RouteClass::Repair(code) = record.class() {
                                actions.push((
                                    PodRef::new(record.pod().clone(), record.pod_generation()),
                                    code.clone(),
                                ));
                            }
                            records.push(record);
                        }
                        CodeRoute::Unroutable(code) => {
                            self.counters.unroutable_codes =
                                self.counters.unroutable_codes.saturating_add(1);
                            unroutable.push(code);
                        }
                    }
                }
                self.observe_records(&records);
                if actions.is_empty() {
                    // Nothing can repair any code: a failing report with no usable finding.
                    self.counters.regenerations = self.counters.regenerations.saturating_add(1);
                    let mut attempt = self.route_fallback(
                        session,
                        FallbackReason::FailWithoutHardFinding,
                        Some(value),
                    )?;
                    attempt.unroutable = unroutable;
                    let decision = if escalating(&attempt) {
                        StepDecision::Escalate
                    } else {
                        StepDecision::Regenerate
                    };
                    return Ok((attempt, decision));
                }
                Ok((
                    Attempt {
                        origin: StepOrigin::Repaired,
                        role: SelfRole::Repair,
                        plan: Plan::Repair {
                            previous: value,
                            actions,
                        },
                        routes: records,
                        unroutable,
                    },
                    StepDecision::Repair,
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{make_loop, request, view, CodeVerifier, Proposer};
    use super::*;
    use crate::expert::StructurePolicyConfig;

    #[test]
    fn a_judge_error_keeps_the_verified_candidate_in_the_partial_trace() {
        let mut vcw = make_loop(8, &[], StructurePolicyConfig::default());
        vcw.fault.judge = true;
        let result = vcw.run(
            &request("judge-fault", 3, 3),
            &view(&["gen"]),
            &CodeVerifier,
            &mut Proposer("x"),
        );
        assert!(matches!(result, Err(VcwError::Internal(_))));
        let partial = vcw.failed_trace().expect("a partial trace");
        assert_eq!(partial.outcome(), OutcomeTag::Failed);
        assert_eq!(partial.steps().len(), 1);
        let step = &partial.steps()[0];
        assert_eq!(step.decision, StepDecision::JudgeFailed);
        assert_eq!(
            step.authors,
            vec![PodRef::new(PodId::from("gen"), Generation(1))]
        );
        assert_eq!(step.budget_after.verifier_calls, 1);
        assert_eq!(partial.spent().verifier_calls, 1);
        assert_eq!(vcw.counters().verifier_calls, 1);
        assert_eq!(&partial.compute_digest(), partial.digest());
    }
}
