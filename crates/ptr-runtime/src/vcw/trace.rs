//! The bounded, typed per-request trace of the verifier-closed loop.
//!
//! A trace holds one [`TraceStep`] per attempt to produce a candidate: the routing decisions that
//! chose the Pods (with their support and propensity), the candidate's digest, the verifier
//! report in summary, the class `classify` gave it, what the workspace did with it, the
//! self-model's prediction made before the outcome was known, the budget spent, and what the loop
//! decided. The trace carries no payload and no finding message, so it cannot leak evidence, and
//! its size is bounded by the round budget and by [`MAX_TRACE_CODES`].
//!
//! [`VcwTrace::digest`] is SHA-256 over a canonical encoding of every field. It is a function of
//! the request, the seeds (sampling key, nonce start), the Pods' answers and the verifier's
//! reports, so a replay with the same inputs reproduces it (invariant 14).

use ptr_feedback::ExhaustionReason;
use ptr_pods::RefusalKind;
use ptr_router::{
    FallbackReason, FindingCode, FindingRouteRecord, PodRef, RouteClass, SelfPrediction,
};
use ptr_types::{Digest, RequestId, TypeId, VerificationLevel};
use ptr_verifier::{VerificationReport, VerificationStatus};
use sha2::{Digest as _, Sha256};

use super::budget::{BudgetUse, RequestBudget};

/// Hard-finding codes one step records; the rest are counted in
/// [`ReportSummary::hard_total`].
pub const MAX_TRACE_CODES: usize = 32;
/// Bytes of one recorded code.
pub const MAX_TRACE_CODE_BYTES: usize = 128;

/// v2: a repaired candidate's `authors` keep its proposer and earlier repairers (issue #141). The
/// encoding is unchanged; the version separates digests of traces recorded under the v1 meaning
/// (only the repairers of the last round) from the current one.
const TRACE_DOMAIN: &[u8] = b"ptr.runtime.vcw.trace.v2";

/// The verifier report of one candidate, without messages.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct ReportSummary {
    pub status: VerificationStatus,
    pub level: VerificationLevel,
    /// The score as `f32` bits. It is recorded, never read by any decision.
    pub score_bits: u32,
    /// Distinct hard-finding codes in report order, at most [`MAX_TRACE_CODES`], each cut to
    /// [`MAX_TRACE_CODE_BYTES`].
    pub hard_codes: Vec<String>,
    /// Hard findings in the report, counted before the cap.
    pub hard_total: u32,
    pub soft_findings: u32,
}

impl ReportSummary {
    pub(crate) fn of(report: &VerificationReport) -> Self {
        let mut hard_codes: Vec<String> = Vec::new();
        let mut hard_total = 0u32;
        let mut soft = 0u32;
        for finding in &report.findings {
            if finding.hard {
                hard_total = hard_total.saturating_add(1);
                let code = cut(&finding.code);
                if hard_codes.len() < MAX_TRACE_CODES && !hard_codes.contains(&code) {
                    hard_codes.push(code);
                }
            } else {
                soft = soft.saturating_add(1);
            }
        }
        Self {
            status: report.status,
            level: report.level,
            score_bits: report.score.get().to_bits(),
            hard_codes,
            hard_total,
            soft_findings: soft,
        }
    }
}

fn cut(code: &str) -> String {
    let mut end = code.len().min(MAX_TRACE_CODE_BYTES);
    while !code.is_char_boundary(end) {
        end -= 1;
    }
    code[..end].to_owned()
}

/// What `classify` made of a report.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ClassSummary {
    Accept,
    Repair(Vec<FindingCode>),
    Fallback(FallbackReason),
}

/// How an attempt came to produce its candidate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StepOrigin {
    /// The first proposal.
    Initial,
    /// Repair Pods edited the previous candidate, one per routed finding code.
    Repaired,
    /// A fresh proposal after a regenerate fallback (or a repair nothing could perform).
    Regenerated,
    /// A stronger Pod or a human produced the candidate after an escalation.
    Escalated,
}

/// What the workspace did with a candidate in the cycle that followed its verdict.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum WorkspaceOutcome {
    /// The candidate could not be put in the pending area.
    NotSubmitted(&'static str),
    /// The candidate holds a slot in the snapshot.
    Admitted,
    /// The candidate was refused, with the workspace's reason class.
    Refused(RefusalKind),
    /// The cycle neither admitted nor refused it (cannot happen unless the cycle was skipped).
    Unlisted,
}

/// What the loop did after a verdict.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StepDecision {
    Accepted,
    /// `classify` accepted but the author's generation was no longer eligible: the candidate is
    /// not returned and a fresh proposal is drawn.
    AuthorNoLongerEligible,
    Repair,
    Regenerate,
    Escalate,
    Stopped(ExhaustionReason),
    /// The attempt produced no candidate (a Pod declined).
    NoCandidate,
    /// Routing the next step failed with a [`VcwError`](super::VcwError); the request ended
    /// with that error and this trace is the partial one.
    RoutingFailed,
    /// Judging the candidate (verification, classification, workspace cycle) failed with a
    /// [`VcwError`](super::VcwError); the candidate's budget is spent and the trace is partial.
    JudgeFailed,
}

/// The self-model's step, predicted before the outcome was applied.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct SelfStepTrace {
    pub prediction: SelfPrediction,
    /// The verifier gate passed (`classify` returned `Accept`): the label the model learned.
    pub passed: bool,
    /// Advisory hints computed for the next context. They change nothing by themselves.
    pub hints: usize,
}

/// The candidate of an attempt.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct CandidateTrace {
    pub payload_type: TypeId,
    pub payload_digest: Digest,
    pub report: ReportSummary,
    pub class: ClassSummary,
    pub workspace: WorkspaceOutcome,
    /// Revision and digest of the snapshot of the cycle that judged the candidate.
    pub workspace_revision: Option<u64>,
    pub snapshot_digest: Option<Digest>,
    pub self_model: SelfStepTrace,
}

/// One attempt to produce a candidate.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct TraceStep {
    pub round: u32,
    pub origin: StepOrigin,
    /// Every Pod whose output is in the candidate, in the order it last contributed: the
    /// proposer; for a repaired candidate also every repairer whose edit was applied, in this
    /// round or an earlier one (the proposer stays listed); or the escalation author. Empty when
    /// the attempt produced no candidate.
    pub authors: Vec<PodRef>,
    /// Every sampled routing decision that led to this attempt, with support and propensity.
    pub routes: Vec<FindingRouteRecord>,
    /// Finding codes of the routing that had no eligible repair Pod.
    pub unroutable: Vec<FindingCode>,
    pub candidate: Option<CandidateTrace>,
    pub decision: StepDecision,
    pub budget_after: BudgetUse,
}

/// How a request ended.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutcomeTag {
    Verified,
    Exhausted(ExhaustionReason),
    Escalated,
    /// The loop itself failed with a [`VcwError`](super::VcwError); the trace is partial.
    Failed,
}

/// The fields of a trace before it is sealed.
pub(crate) struct SealParts {
    pub(crate) request_id: RequestId,
    pub(crate) request_digest: Digest,
    pub(crate) required: VerificationLevel,
    pub(crate) budget: RequestBudget,
    pub(crate) steps: Vec<TraceStep>,
    pub(crate) spent: BudgetUse,
    pub(crate) outcome: OutcomeTag,
    pub(crate) final_snapshot_digest: Option<Digest>,
    pub(crate) self_model_digest: Digest,
}

/// The trace of one request. Read-only: the fields are private, so the digest it carries is
/// always that of what it shows.
#[derive(Clone, Debug, PartialEq)]
pub struct VcwTrace {
    request_id: RequestId,
    request_digest: Digest,
    required: VerificationLevel,
    budget: RequestBudget,
    steps: Vec<TraceStep>,
    spent: BudgetUse,
    outcome: OutcomeTag,
    final_snapshot_digest: Option<Digest>,
    self_model_digest: Digest,
    digest: Digest,
}

impl VcwTrace {
    pub(crate) fn seal(parts: SealParts) -> Self {
        let mut trace = Self {
            request_id: parts.request_id,
            request_digest: parts.request_digest,
            required: parts.required,
            budget: parts.budget,
            steps: parts.steps,
            spent: parts.spent,
            outcome: parts.outcome,
            final_snapshot_digest: parts.final_snapshot_digest,
            self_model_digest: parts.self_model_digest,
            digest: [0; 32],
        };
        trace.digest = trace.compute_digest();
        trace
    }

    pub fn request_id(&self) -> &RequestId {
        &self.request_id
    }

    pub fn request_digest(&self) -> &Digest {
        &self.request_digest
    }

    pub fn required(&self) -> VerificationLevel {
        self.required
    }

    pub fn budget(&self) -> RequestBudget {
        self.budget
    }

    /// One step per attempt, at most one more than the round budget.
    pub fn steps(&self) -> &[TraceStep] {
        &self.steps
    }

    /// What the request spent. `pod_calls` is bounded by `(rounds + 1) * (MAX_TRACE_CODES + 2)`
    /// (a repair round calls at most one Pod per code, a declined repair adds one fresh
    /// proposal) and is metered, not budgeted separately.
    pub fn spent(&self) -> BudgetUse {
        self.spent
    }

    pub fn outcome(&self) -> OutcomeTag {
        self.outcome
    }

    /// Digest of the last workspace snapshot, if a cycle ran.
    pub fn final_snapshot_digest(&self) -> Option<&Digest> {
        self.final_snapshot_digest.as_ref()
    }

    /// Digest of the self-model after the request.
    pub fn self_model_digest(&self) -> &Digest {
        &self.self_model_digest
    }

    /// SHA-256 over the canonical encoding of every field except this digest.
    pub fn digest(&self) -> &Digest {
        &self.digest
    }

    /// Recompute the digest from the fields; equal to [`VcwTrace::digest`] for an unmodified
    /// trace.
    pub fn compute_digest(&self) -> Digest {
        let mut enc = Enc::new();
        enc.str(&self.request_id.0).bytes(&self.request_digest);
        enc.u8(level_tag(self.required));
        enc.u32(self.budget.verifier_calls.get())
            .u32(self.budget.rounds.get());
        enc.u64(self.steps.len() as u64);
        for step in &self.steps {
            enc.step(step);
        }
        enc.budget(self.spent);
        match self.outcome {
            OutcomeTag::Verified => enc.u8(0),
            OutcomeTag::Exhausted(reason) => enc.u8(1).u8(exhaustion_tag(reason)),
            OutcomeTag::Escalated => enc.u8(2),
            OutcomeTag::Failed => enc.u8(3),
        };
        enc.opt_digest(self.final_snapshot_digest.as_ref());
        enc.bytes(&self.self_model_digest);
        enc.finish()
    }

    /// The candidates the request verified, in order.
    pub fn candidates(&self) -> impl Iterator<Item = &CandidateTrace> {
        self.steps.iter().filter_map(|step| step.candidate.as_ref())
    }

    /// Every sampled routing decision of the request, in order.
    pub fn routes(&self) -> impl Iterator<Item = &FindingRouteRecord> {
        self.steps.iter().flat_map(|step| step.routes.iter())
    }
}

fn level_tag(level: VerificationLevel) -> u8 {
    match level {
        VerificationLevel::Unverified => 0,
        VerificationLevel::LatentAgreement => 1,
        VerificationLevel::SampleVerified => 2,
        VerificationLevel::FullSemantic => 3,
        VerificationLevel::Deterministic => 4,
    }
}

fn status_tag(status: VerificationStatus) -> u8 {
    match status {
        VerificationStatus::Pass => 0,
        VerificationStatus::Fail => 1,
        VerificationStatus::Disputed => 2,
        VerificationStatus::Unknown => 3,
    }
}

fn exhaustion_tag(reason: ExhaustionReason) -> u8 {
    match reason {
        ExhaustionReason::VerifierCalls => 0,
        ExhaustionReason::Rounds => 1,
    }
}

pub(crate) fn fallback_tag(reason: FallbackReason) -> u8 {
    match reason {
        FallbackReason::Unknown => 0,
        FallbackReason::Disputed => 1,
        FallbackReason::FailWithoutHardFinding => 2,
        FallbackReason::ForgedPass => 3,
        FallbackReason::BelowLevel => 4,
        FallbackReason::TooManyCodes => 5,
        FallbackReason::OversizedCode => 6,
    }
}

fn refusal_tag(kind: RefusalKind) -> u8 {
    match kind {
        RefusalKind::Stale => 0,
        RefusalKind::AuthorRevoked => 1,
        RefusalKind::NotPassed => 2,
        RefusalKind::HardFinding => 3,
        RefusalKind::BelowRequiredLevel => 4,
        RefusalKind::Duplicate => 5,
        RefusalKind::AuthorSlotCap => 6,
        RefusalKind::Outbid => 7,
        // `RefusalKind` is non-exhaustive in another crate.
        _ => 255,
    }
}

/// Length-prefixed canonical encoder.
struct Enc(Sha256);

impl Enc {
    fn new() -> Self {
        let mut hasher = Sha256::new();
        hasher.update(TRACE_DOMAIN);
        Self(hasher)
    }

    fn u8(&mut self, v: u8) -> &mut Self {
        self.0.update([v]);
        self
    }

    fn u32(&mut self, v: u32) -> &mut Self {
        self.0.update(v.to_le_bytes());
        self
    }

    fn u64(&mut self, v: u64) -> &mut Self {
        self.0.update(v.to_le_bytes());
        self
    }

    fn bytes(&mut self, v: &[u8]) -> &mut Self {
        self.u64(v.len() as u64);
        self.0.update(v);
        self
    }

    fn str(&mut self, v: &str) -> &mut Self {
        self.bytes(v.as_bytes())
    }

    fn opt_digest(&mut self, v: Option<&Digest>) -> &mut Self {
        match v {
            Some(d) => self.u8(1).bytes(d),
            None => self.u8(0),
        }
    }

    fn budget(&mut self, b: BudgetUse) -> &mut Self {
        self.u32(b.verifier_calls).u32(b.rounds).u32(b.pod_calls)
    }

    fn pod_ref(&mut self, pod: &PodRef) -> &mut Self {
        self.str(&pod.pod.0).u64(pod.generation.0)
    }

    fn route(&mut self, record: &FindingRouteRecord) -> &mut Self {
        self.str(&record.request_id().0)
            .bytes(&record.request_digest());
        match record.class() {
            RouteClass::Repair(code) => self.u8(0).str(code.as_str()),
            RouteClass::Regenerate(reason) => self.u8(1).u8(fallback_tag(*reason)),
        };
        self.u64(record.support().len() as u64);
        for (pod, width) in record.support() {
            self.str(&pod.0).u64(*width);
        }
        self.str(&record.pod().0)
            .u64(record.pod_generation().0)
            .u64(record.propensity().get().to_bits())
            .u64(record.draw())
            .bytes(record.nonce().as_bytes())
            .bytes(&record.key_id())
            .u64(record.generation().get())
    }

    fn step(&mut self, step: &TraceStep) -> &mut Self {
        self.u32(step.round);
        self.u8(match step.origin {
            StepOrigin::Initial => 0,
            StepOrigin::Repaired => 1,
            StepOrigin::Regenerated => 2,
            StepOrigin::Escalated => 3,
        });
        self.u64(step.authors.len() as u64);
        for author in &step.authors {
            self.pod_ref(author);
        }
        self.u64(step.routes.len() as u64);
        for record in &step.routes {
            self.route(record);
        }
        self.u64(step.unroutable.len() as u64);
        for code in &step.unroutable {
            self.str(code.as_str());
        }
        match &step.candidate {
            Some(candidate) => self.u8(1).candidate(candidate),
            None => self.u8(0),
        };
        match step.decision {
            StepDecision::Accepted => self.u8(0),
            StepDecision::AuthorNoLongerEligible => self.u8(1),
            StepDecision::Repair => self.u8(2),
            StepDecision::Regenerate => self.u8(3),
            StepDecision::Escalate => self.u8(4),
            StepDecision::Stopped(reason) => self.u8(5).u8(exhaustion_tag(reason)),
            StepDecision::NoCandidate => self.u8(6),
            StepDecision::RoutingFailed => self.u8(7),
            StepDecision::JudgeFailed => self.u8(8),
        };
        self.budget(step.budget_after)
    }

    fn candidate(&mut self, c: &CandidateTrace) -> &mut Self {
        self.str(&c.payload_type.0).bytes(&c.payload_digest);
        self.u8(status_tag(c.report.status))
            .u8(level_tag(c.report.level))
            .u32(c.report.score_bits)
            .u32(c.report.hard_total)
            .u32(c.report.soft_findings)
            .u64(c.report.hard_codes.len() as u64);
        for code in &c.report.hard_codes {
            self.str(code);
        }
        match &c.class {
            ClassSummary::Accept => {
                self.u8(0);
            }
            ClassSummary::Repair(codes) => {
                self.u8(1).u64(codes.len() as u64);
                for code in codes {
                    self.str(code.as_str());
                }
            }
            ClassSummary::Fallback(reason) => {
                self.u8(2).u8(fallback_tag(*reason));
            }
        }
        match c.workspace {
            WorkspaceOutcome::NotSubmitted(why) => self.u8(0).str(why),
            WorkspaceOutcome::Admitted => self.u8(1),
            WorkspaceOutcome::Refused(kind) => self.u8(2).u8(refusal_tag(kind)),
            WorkspaceOutcome::Unlisted => self.u8(3),
        };
        match c.workspace_revision {
            Some(revision) => self.u8(1).u64(revision),
            None => self.u8(0),
        };
        self.opt_digest(c.snapshot_digest.as_ref());
        let prediction = &c.self_model.prediction;
        self.u64(prediction.pass_probability.to_bits())
            .u64(prediction.evidence)
            .u8(u8::from(c.self_model.passed))
            .u64(c.self_model.hints as u64)
    }

    fn finish(self) -> Digest {
        self.0.finalize().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vcw::{
        ProposeCall, RepairCall, RolePods, Roster, VcwConfig, VcwLoop, VcwOutcome, VcwParts,
        VcwPayload, VcwRequest,
    };
    use ptr_router::{
        EligibilitySnapshot, NonceSequence, RouterConfig, SamplingKey, SelfModel, SelfModelLimits,
        UniformScorer,
    };
    use ptr_types::{Generation, PodId, Probability};
    use ptr_verifier::{Finding, Verifier};
    use std::cell::Cell;

    #[derive(Clone)]
    struct Text(String);

    impl VcwPayload for Text {
        fn payload_type(&self) -> TypeId {
            TypeId::from("text")
        }

        fn payload_digest(&self) -> Digest {
            Sha256::digest(self.0.as_bytes()).into()
        }
    }

    /// Fails the first candidate with a routable code, an unroutable code and a soft finding, and
    /// passes the second.
    struct Script(Cell<u32>);

    impl Verifier<Text> for Script {
        fn verify(&self, _: &Text) -> VerificationReport {
            self.0.set(self.0.get() + 1);
            let finding = |code: &str, hard: bool| Finding {
                code: code.to_owned(),
                message: String::new(),
                hard,
            };
            if self.0.get() == 1 {
                VerificationReport {
                    status: VerificationStatus::Fail,
                    level: VerificationLevel::Deterministic,
                    score: Probability::new(0.25).unwrap(),
                    findings: vec![finding("a", true), finding("zz", true), finding("s", false)],
                }
            } else {
                VerificationReport {
                    status: VerificationStatus::Pass,
                    level: VerificationLevel::Deterministic,
                    score: Probability::new(1.0).unwrap(),
                    findings: Vec::new(),
                }
            }
        }
    }

    struct Pods;

    impl RolePods<Text> for Pods {
        fn propose(&mut self, _: ProposeCall<'_>) -> Option<Text> {
            Some(Text("draft".to_owned()))
        }

        fn repair(&mut self, _: RepairCall<'_, Text>) -> Option<Text> {
            Some(Text("repaired".to_owned()))
        }
    }

    fn sample() -> VcwTrace {
        let mut view = EligibilitySnapshot::new();
        view.set(PodId::from("gen"), Generation(1));
        view.set(PodId::from("fix"), Generation(1));
        let mut vcw = VcwLoop::new(VcwParts {
            config: VcwConfig::new("digest-test", [1; 32]),
            roster: Roster {
                proposers: vec![PodId::from("gen")],
                repairers: vec![(PodId::from("fix"), vec![ptr_router::FindingCode::from("a")])],
            },
            router: RouterConfig::default(),
            scorer: UniformScorer,
            key: SamplingKey::from_bytes([3; 32]),
            nonces: NonceSequence::starting_at([2; 8], 0),
            self_model: SelfModel::new(SelfModelLimits::default()),
            schema: None,
            structure: crate::expert::StructurePolicy::new(
                crate::expert::StructurePolicyConfig::default(),
            )
            .unwrap(),
        })
        .unwrap();
        let request = VcwRequest {
            request_id: RequestId::from("digest"),
            project: ptr_types::ProjectId::from("p"),
            request_digest: [9; 32],
            required: VerificationLevel::SampleVerified,
            budget: RequestBudget::new(3, 3).unwrap(),
        };
        let run = vcw
            .run(&request, &view, &Script(Cell::new(0)), &mut Pods)
            .unwrap();
        assert!(matches!(run.outcome, VcwOutcome::Verified(_)));
        run.trace
    }

    type Mutation = (&'static str, fn(&mut VcwTrace));

    fn flip(d: &mut Digest) {
        d[0] ^= 1;
    }

    fn cand(t: &mut VcwTrace) -> &mut CandidateTrace {
        t.steps[0].candidate.as_mut().unwrap()
    }

    #[test]
    fn every_encoded_field_changes_the_digest() {
        let base = sample();
        // The sample exercises what the mutations touch.
        assert_eq!(base.steps.len(), 2);
        assert!(!base.steps[1].routes.is_empty());
        assert_eq!(base.steps[1].unroutable.len(), 1);
        assert_eq!(
            base.steps[0]
                .candidate
                .as_ref()
                .unwrap()
                .report
                .soft_findings,
            1
        );
        assert_eq!(&base.compute_digest(), base.digest());
        let mutations: Vec<Mutation> = vec![
            ("request_id", |t| t.request_id = RequestId::from("other")),
            ("request_digest", |t| flip(&mut t.request_digest)),
            ("required", |t| {
                t.required = VerificationLevel::Deterministic
            }),
            ("budget.calls", |t| {
                t.budget = RequestBudget::new(4, 3).unwrap()
            }),
            ("budget.rounds", |t| {
                t.budget = RequestBudget::new(3, 4).unwrap()
            }),
            ("spent.calls", |t| t.spent.verifier_calls += 1),
            ("spent.rounds", |t| t.spent.rounds += 1),
            ("spent.pods", |t| t.spent.pod_calls += 1),
            ("outcome", |t| t.outcome = OutcomeTag::Escalated),
            ("outcome.failed", |t| t.outcome = OutcomeTag::Failed),
            ("final_snapshot", |t| t.final_snapshot_digest = None),
            ("self_model_digest", |t| flip(&mut t.self_model_digest)),
            ("step.round", |t| t.steps[1].round += 1),
            ("step.origin", |t| t.steps[1].origin = StepOrigin::Escalated),
            ("step.authors", |t| t.steps[1].authors.clear()),
            ("step.author.generation", |t| {
                t.steps[1].authors[0].generation = Generation(9)
            }),
            ("step.author.pod", |t| {
                t.steps[1].authors[0].pod = PodId::from("other")
            }),
            ("step.routes", |t| t.steps[1].routes.clear()),
            ("step.unroutable", |t| t.steps[1].unroutable.clear()),
            ("step.decision", |t| {
                t.steps[0].decision = StepDecision::Escalate
            }),
            ("step.decision.stopped", |t| {
                t.steps[0].decision = StepDecision::Stopped(ExhaustionReason::Rounds)
            }),
            ("step.budget.calls", |t| {
                t.steps[0].budget_after.verifier_calls += 1
            }),
            ("step.budget.rounds", |t| {
                t.steps[0].budget_after.rounds += 1
            }),
            ("step.budget.pods", |t| {
                t.steps[0].budget_after.pod_calls += 1
            }),
            ("candidate.absent", |t| t.steps[0].candidate = None),
            ("candidate.type", |t| {
                cand(t).payload_type = TypeId::from("other")
            }),
            ("candidate.digest", |t| flip(&mut cand(t).payload_digest)),
            ("report.status", |t| {
                cand(t).report.status = VerificationStatus::Unknown
            }),
            ("report.level", |t| {
                cand(t).report.level = VerificationLevel::Unverified
            }),
            ("report.score", |t| cand(t).report.score_bits ^= 1),
            ("report.codes", |t| {
                cand(t).report.hard_codes.push("x".into())
            }),
            ("report.total", |t| cand(t).report.hard_total += 1),
            ("report.soft", |t| cand(t).report.soft_findings += 1),
            ("class", |t| cand(t).class = ClassSummary::Accept),
            ("class.fallback", |t| {
                cand(t).class = ClassSummary::Fallback(FallbackReason::Unknown)
            }),
            ("workspace", |t| {
                cand(t).workspace = WorkspaceOutcome::Admitted
            }),
            ("workspace.refused", |t| {
                cand(t).workspace = WorkspaceOutcome::Refused(RefusalKind::Outbid)
            }),
            ("workspace.revision", |t| cand(t).workspace_revision = None),
            ("workspace.snapshot", |t| cand(t).snapshot_digest = None),
            ("self.pass_probability", |t| {
                cand(t).self_model.prediction.pass_probability += 0.125
            }),
            ("self.evidence", |t| {
                cand(t).self_model.prediction.evidence += 1
            }),
            ("self.passed", |t| {
                cand(t).self_model.passed = !cand(t).self_model.passed
            }),
            ("self.hints", |t| cand(t).self_model.hints += 1),
        ];
        for (name, mutate) in mutations {
            let mut changed = base.clone();
            mutate(&mut changed);
            assert_ne!(
                changed.compute_digest(),
                *base.digest(),
                "mutating {name} left the digest unchanged"
            );
        }
    }
}
