//! The verifier-closed loop through its public API, with stub Pods and a scripted verifier:
//! acceptance only through `classify`, repair by finding code, regenerate and escalate, budgets
//! and `Exhausted`, eligibility (revoked and stale-generation Pods), observers that cannot change
//! the gate, L3 requests surfaced to the host and completed through the existing admission path,
//! the way an accepted answer reaches the runtime's output admission, and replay determinism.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::rc::Rc;
use std::sync::Arc;

use ptr_config::PtrConfig;
use ptr_feedback::{Candidate, ExhaustionReason};
use ptr_lineage::{
    AccuracyMatrix, AdapterId, BoundGateReport, ForgettingGate, GateSubject, PublicSuite,
};
use ptr_pods::{
    BaseModelDigest, BaseModelId, BaseModelRef, DynPod, ExpertState, ManifestDigest, PodManifest,
    PodRegistry, ProposerRef, RefusalKind,
};
use ptr_protocol::TypedPayload;
use ptr_router::{
    AttentionSchema, ControlArm, ControlConfig, ControlledScorer, EligibilitySnapshot,
    EscalationReason, FallbackAction, FallbackPolicy, FallbackReason, FindingCode, HintPolicy,
    NonceSequence, PodEligibility, PodRef, RouteClass, RouterConfig, SamplingKey, SchemaLimits,
    SelfModel, SelfModelLimits, UniformScorer,
};
use ptr_router::{LearnerConfig, RoutingLearner};
use ptr_runtime::expert::{
    expert_evidence_artifact_digest, expert_evidence_base_model_digest, BirthContext, BirthSite,
    EnabledOps, ExpertAdmissionGrant, StructurePolicy, StructurePolicyConfig,
};
use ptr_runtime::vcw::{
    admit_verified_answer, ClassSummary, EscalateCall, HintMode, OutcomeTag, ProposeCall,
    RepairCall, RequestBudget, RolePods, Roster, RuntimeStanding, StepDecision, StructureHost,
    VcwAdmitError, VcwConfig, VcwError, VcwLoop, VcwOutcome, VcwParts, VcwPayload, VcwRequest,
    VcwRun, WorkspaceOutcome,
};
use ptr_runtime::{PtrRuntime, RuntimeError};
use ptr_types::{
    CapabilityId, CommitIndex, Digest, Effect, Generation, PodId, Probability, ProjectId,
    RequestId, TypeId, VerificationLevel,
};
use ptr_verifier::{Finding, VerificationReport, VerificationStatus, Verifier};
use sha2::{Digest as _, Sha256};

use ptr_ledger::{ExpertFenceCause, LedgerEvent};
use ptr_pods::{
    ExpertLevel, ExpertPodSpec, ExpertProposal, ExpertTarget, ExpertTargetInit, RationaleDigest,
    StructuralOp,
};
use ptr_router::{AdmissibleScorer, RouteError, ScoringContext};
use ptr_runtime::expert::{ExpertFenceGrant, ExpertFenceRequest};

mod expert_support;

use expert_support::{binding, prime, requirement, EVAL_MANIFEST};

// ---------------------------------------------------------------------------------------------
// Stubs
// ---------------------------------------------------------------------------------------------

/// A document with a tag and a list of defect codes. The scripted verifier reports each defect as
/// a hard finding.
#[derive(Clone, Debug, Eq, PartialEq)]
struct Doc {
    tag: String,
    defects: Vec<String>,
}

fn doc(tag: &str, defects: &[&str]) -> Doc {
    Doc {
        tag: tag.to_owned(),
        defects: defects.iter().map(|d| (*d).to_owned()).collect(),
    }
}

impl VcwPayload for Doc {
    fn payload_type(&self) -> TypeId {
        TypeId::from("doc")
    }

    fn payload_digest(&self) -> Digest {
        if self.tag == "zero" {
            return [0; 32];
        }
        let mut hasher = Sha256::new();
        hasher.update(self.tag.as_bytes());
        for defect in &self.defects {
            hasher.update([0]);
            hasher.update(defect.as_bytes());
        }
        hasher.finalize().into()
    }

    fn confined_to(&self, edited: &Self, named: &[String]) -> bool {
        // A "loose" document models a payload whose named part is the whole text: a repair may
        // replace one defect by another (the case the L3 credit rule must not reward).
        if self.tag.starts_with("loose") {
            return self.tag == edited.tag;
        }
        let rest = |d: &Doc| -> Vec<String> {
            d.defects
                .iter()
                .filter(|code| !named.contains(code))
                .cloned()
                .collect()
        };
        self.tag == edited.tag && rest(self) == rest(edited)
    }
}

fn report(
    status: VerificationStatus,
    level: VerificationLevel,
    findings: &[(&str, bool)],
) -> VerificationReport {
    VerificationReport {
        status,
        level,
        score: Probability::new(0.9).unwrap(),
        findings: findings
            .iter()
            .map(|(code, hard)| Finding {
                code: (*code).to_owned(),
                message: String::new(),
                hard: *hard,
            })
            .collect(),
    }
}

/// Reports each defect as a hard finding. A scripted report, when queued, overrides it once.
struct DocVerifier {
    calls: Cell<u32>,
    seen: RefCell<Vec<Digest>>,
    level: VerificationLevel,
    scripted: RefCell<VecDeque<VerificationReport>>,
    always: RefCell<Option<VerificationReport>>,
}

impl DocVerifier {
    fn new() -> Self {
        Self {
            calls: Cell::new(0),
            seen: RefCell::new(Vec::new()),
            level: VerificationLevel::Deterministic,
            scripted: RefCell::new(VecDeque::new()),
            always: RefCell::new(None),
        }
    }

    fn always(report: VerificationReport) -> Self {
        let verifier = Self::new();
        *verifier.always.borrow_mut() = Some(report);
        verifier
    }
}

impl Verifier<Doc> for DocVerifier {
    fn verify(&self, candidate: &Doc) -> VerificationReport {
        self.calls.set(self.calls.get() + 1);
        self.seen.borrow_mut().push(candidate.payload_digest());
        if let Some(scripted) = self.scripted.borrow_mut().pop_front() {
            return scripted;
        }
        if let Some(always) = self.always.borrow().clone() {
            return always;
        }
        if candidate.defects.is_empty() {
            report(VerificationStatus::Pass, self.level, &[])
        } else {
            let findings: Vec<(&str, bool)> = candidate
                .defects
                .iter()
                .map(|d| (d.as_str(), true))
                .collect();
            report(VerificationStatus::Fail, self.level, &findings)
        }
    }
}

type MakeDoc = Box<dyn FnMut(&str, u32) -> Doc>;
type ProposeHook = Box<dyn FnMut(&str)>;

/// Stub role Pods: proposals from a factory, repairs that remove the named defect when the Pod
/// can fix that code.
struct Stub {
    n: u32,
    make: MakeDoc,
    fixes: BTreeMap<String, BTreeSet<String>>,
    /// A Pod that, when it repairs, also introduces this other defect.
    introduces: BTreeMap<String, String>,
    overreach: bool,
    none_proposals: bool,
    escalation: Option<(PodRef, Doc)>,
    on_propose: Option<ProposeHook>,
    on_repair: Option<ProposeHook>,
    log: Vec<String>,
    asked_codes: Vec<(String, Vec<String>)>,
    snapshots: Vec<Option<(u64, Vec<String>)>>,
}

impl Stub {
    fn new(make: impl FnMut(&str, u32) -> Doc + 'static) -> Self {
        Self {
            n: 0,
            make: Box::new(make),
            fixes: BTreeMap::new(),
            introduces: BTreeMap::new(),
            overreach: false,
            none_proposals: false,
            escalation: None,
            on_propose: None,
            on_repair: None,
            log: Vec::new(),
            asked_codes: Vec::new(),
            snapshots: Vec::new(),
        }
    }

    fn introducing(mut self, pod: &str, code: &str) -> Self {
        self.introduces.insert(pod.to_owned(), code.to_owned());
        self
    }

    fn fixing(mut self, pod: &str, codes: &[&str]) -> Self {
        self.fixes.insert(
            pod.to_owned(),
            codes.iter().map(|c| (*c).to_owned()).collect(),
        );
        self
    }
}

fn snapshot_view(snapshot: Option<&ptr_pods::WorkspaceSnapshot>) -> Option<(u64, Vec<String>)> {
    snapshot.map(|s| {
        (
            s.revision().get(),
            s.refusals()
                .iter()
                .flat_map(|r| r.findings().iter().map(|f| f.code().to_owned()))
                .collect(),
        )
    })
}

impl RolePods<Doc> for Stub {
    fn propose(&mut self, call: ProposeCall<'_>) -> Option<Doc> {
        self.log.push(format!("propose:{}", call.pod.pod.0));
        self.snapshots.push(snapshot_view(call.snapshot));
        if let Some(hook) = self.on_propose.as_mut() {
            hook(&call.pod.pod.0);
        }
        if self.none_proposals {
            return None;
        }
        self.n += 1;
        Some((self.make)(&call.pod.pod.0, self.n))
    }

    fn repair(&mut self, call: RepairCall<'_, Doc>) -> Option<Doc> {
        self.log
            .push(format!("repair:{}:{}", call.pod.pod.0, call.code.as_str()));
        self.asked_codes
            .push((call.pod.pod.0.clone(), call.critique.issues.clone()));
        if let Some(hook) = self.on_repair.as_mut() {
            hook(&call.pod.pod.0);
        }
        let can = self
            .fixes
            .get(&call.pod.pod.0)
            .is_some_and(|codes| codes.contains(call.code.as_str()));
        if !can {
            return None;
        }
        let mut edited = call.previous.clone();
        if self.overreach {
            edited.defects.clear();
        } else {
            edited.defects.retain(|d| d != call.code.as_str());
        }
        if let Some(new) = self.introduces.get(&call.pod.pod.0) {
            if !edited.defects.contains(new) {
                edited.defects.push(new.clone());
            }
        }
        Some(edited)
    }

    fn escalate(&mut self, call: EscalateCall<'_, Doc>) -> Option<(PodRef, Doc)> {
        self.log.push(format!("escalate:{:?}", call.reason));
        self.escalation.take()
    }
}

// ---------------------------------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------------------------------

fn pod(name: &str) -> PodId {
    PodId::from(name)
}

fn code(name: &str) -> FindingCode {
    FindingCode::from(name)
}

fn project() -> ProjectId {
    ProjectId::from("proj")
}

fn digest_of(text: &str) -> Digest {
    Sha256::digest(text.as_bytes()).into()
}

fn request(id: &str, calls: u32, rounds: u32, required: VerificationLevel) -> VcwRequest {
    VcwRequest {
        request_id: RequestId::from(id),
        project: project(),
        request_digest: digest_of(id),
        required,
        budget: RequestBudget::new(calls, rounds).unwrap(),
    }
}

fn eligibility(pods: &[&str]) -> EligibilitySnapshot {
    let mut view = EligibilitySnapshot::new();
    for name in pods {
        view.set(pod(name), Generation(1));
    }
    view
}

fn roster(proposers: &[&str], repairers: &[(&str, &[&str])]) -> Roster {
    Roster {
        proposers: proposers.iter().map(|p| pod(p)).collect(),
        repairers: repairers
            .iter()
            .map(|(p, codes)| (pod(p), codes.iter().map(|c| code(c)).collect()))
            .collect(),
    }
}

fn base_config() -> VcwConfig {
    VcwConfig::new("stub-verifier", [7; 32])
}

fn parts_with<S>(
    config: VcwConfig,
    roster: Roster,
    router: RouterConfig,
    scorer: S,
    key: u8,
    schema: Option<Arc<AttentionSchema>>,
) -> VcwParts<S> {
    VcwParts {
        config,
        roster,
        router,
        scorer,
        key: SamplingKey::from_bytes([key; 32]),
        nonces: NonceSequence::starting_at([1; 8], 0),
        self_model: SelfModel::new(SelfModelLimits::default()),
        schema,
        structure: StructurePolicy::new(StructurePolicyConfig::default()).unwrap(),
    }
}

fn plain(roster: Roster) -> VcwLoop {
    VcwLoop::new(parts_with(
        base_config(),
        roster,
        RouterConfig::default(),
        UniformScorer,
        7,
        None,
    ))
    .unwrap()
}

fn run_doc<S: ptr_router::AdmissibleScorer>(
    vcw: &mut VcwLoop<S>,
    req: &VcwRequest,
    view: &dyn PodEligibility,
    verifier: &DocVerifier,
    stub: &mut Stub,
) -> VcwRun<Doc> {
    vcw.run(req, view, verifier, stub).unwrap()
}

fn verified(run: &VcwRun<Doc>) -> &ptr_runtime::vcw::VerifiedAnswer<Doc> {
    match &run.outcome {
        VcwOutcome::Verified(answer) => answer,
        other => panic!("expected Verified, got {other:?}"),
    }
}

fn is_verified(run: &VcwRun<Doc>) -> bool {
    matches!(run.outcome, VcwOutcome::Verified(_))
}

/// Proposals with two defects, repaired by one Pod per code.
fn two_defect_stub() -> Stub {
    Stub::new(|_, n| doc(&format!("d{n}"), &["a", "b"]))
        .fixing("fix_a", &["a"])
        .fixing("fix_b", &["b"])
}

fn two_defect_roster() -> Roster {
    roster(&["gen"], &[("fix_a", &["a"]), ("fix_b", &["b"])])
}

fn two_defect_view() -> EligibilitySnapshot {
    eligibility(&["gen", "fix_a", "fix_b"])
}

// ---------------------------------------------------------------------------------------------
// Acceptance only through classify
// ---------------------------------------------------------------------------------------------

const STATUSES: [VerificationStatus; 4] = [
    VerificationStatus::Pass,
    VerificationStatus::Fail,
    VerificationStatus::Disputed,
    VerificationStatus::Unknown,
];

const LEVELS: [VerificationLevel; 5] = [
    VerificationLevel::Unverified,
    VerificationLevel::LatentAgreement,
    VerificationLevel::SampleVerified,
    VerificationLevel::FullSemantic,
    VerificationLevel::Deterministic,
];

fn finding_sets() -> Vec<Vec<(&'static str, bool)>> {
    vec![vec![], vec![("soft", false)], vec![("hard", true)]]
}

/// Over the full matrix of status x level x required x findings, a single-shot request is
/// `Verified` exactly when `Candidate::passes` holds, whatever the observers have learned.
fn accept_matrix<S: ptr_router::AdmissibleScorer>(mut vcw: VcwLoop<S>) {
    let view = eligibility(&["gen", "fix"]);
    let mut seen_accept = 0;
    let mut seen_reject = 0;
    for status in STATUSES {
        for level in LEVELS {
            for required in LEVELS {
                for findings in finding_sets() {
                    let rep = report(status, level, &findings);
                    let verifier = DocVerifier::always(rep.clone());
                    let mut stub = Stub::new(|_, n| doc(&format!("m{n}"), &[]));
                    let req = request("matrix", 1, 1, required);
                    let run = run_doc(&mut vcw, &req, &view, &verifier, &mut stub);
                    let passes = Candidate {
                        id: ptr_types::CandidateId::from("c"),
                        value: (),
                        verification: Some(rep),
                    }
                    .passes(required);
                    assert_eq!(
                        is_verified(&run),
                        passes,
                        "{status:?} {level:?} required {required:?} {findings:?}"
                    );
                    let class = &run.trace.candidates().next().unwrap().class;
                    assert_eq!(matches!(class, ClassSummary::Accept), passes);
                    assert_eq!(verifier.calls.get(), 1);
                    if passes {
                        seen_accept += 1;
                    } else {
                        seen_reject += 1;
                    }
                }
            }
        }
    }
    // The matrix has both kinds of case, so the equality above is not vacuous.
    assert!(seen_accept > 0 && seen_reject > 0);
}

/// Requests with hard findings and a sparse pass rate, to give the self-model something to
/// hint about.
fn warm_up<S: ptr_router::AdmissibleScorer>(vcw: &mut VcwLoop<S>) -> usize {
    let view = eligibility(&["gen", "fix", "fix2"]);
    let mut hints = 0;
    for i in 0..80 {
        let verifier = DocVerifier::new();
        let mut stub = Stub::new(|_, n| doc(&format!("w{n}"), &["a"]))
            .fixing("fix", &["a"])
            .fixing("fix2", &[]);
        let req = request(
            &format!("warm-{i}"),
            3,
            3,
            VerificationLevel::SampleVerified,
        );
        let run = run_doc(vcw, &req, &view, &verifier, &mut stub);
        hints += run
            .trace
            .candidates()
            .map(|c| c.self_model.hints)
            .sum::<usize>();
    }
    hints
}

#[test]
fn nothing_is_accepted_but_what_classify_accepts_over_the_full_matrix() {
    accept_matrix(plain(roster(&["gen"], &[("fix", &["hard"])])));
}

#[test]
fn trained_observers_and_a_controlling_schema_do_not_move_the_accept_predicate() {
    let schema = Arc::new(AttentionSchema::new(SchemaLimits::default()).unwrap());
    let scorer = ControlledScorer::new(
        UniformScorer,
        Arc::clone(&schema),
        ControlArm::Full,
        ControlConfig::default(),
    );
    let mut config = base_config();
    config.hint_mode = HintMode::ApplyToSchema;
    config.hint_policy = HintPolicy {
        escalate_below: 0.9,
        min_evidence: 5,
        prefer_margin: 0.0,
    };
    let mut vcw = VcwLoop::new(parts_with(
        config,
        roster(&["gen"], &[("fix", &["a", "hard"]), ("fix2", &["a"])]),
        RouterConfig::default(),
        scorer,
        7,
        Some(Arc::clone(&schema)),
    ))
    .unwrap();
    let hints = warm_up(&mut vcw);
    // Sensitivity control: the observers did form advice, so the matrix runs against a model
    // that would act if it could.
    assert!(hints > 0, "the warm-up produced no hint");
    accept_matrix(vcw);
}

#[test]
fn forged_pass_below_level_unknown_and_disputed_are_never_accepted() {
    let cases = [
        (
            "forged pass",
            report(
                VerificationStatus::Pass,
                VerificationLevel::Deterministic,
                &[("x", true)],
            ),
        ),
        (
            "below level",
            report(
                VerificationStatus::Pass,
                VerificationLevel::LatentAgreement,
                &[],
            ),
        ),
        (
            "unknown",
            report(
                VerificationStatus::Unknown,
                VerificationLevel::Deterministic,
                &[],
            ),
        ),
        (
            "disputed",
            report(
                VerificationStatus::Disputed,
                VerificationLevel::Deterministic,
                &[],
            ),
        ),
    ];
    for (name, rep) in cases {
        let mut vcw = plain(roster(&["gen"], &[]));
        let view = eligibility(&["gen"]);
        let verifier = DocVerifier::always(rep);
        let mut stub = Stub::new(|_, n| doc(&format!("f{n}"), &[]));
        let req = request(name, 4, 4, VerificationLevel::FullSemantic);
        let run = run_doc(&mut vcw, &req, &view, &verifier, &mut stub);
        assert!(!is_verified(&run), "{name}");
        // The candidate was a clean document with a fine score: nothing but the report decided.
        assert!(run
            .trace
            .candidates()
            .all(|c| !matches!(c.class, ClassSummary::Accept)));
    }
}

#[test]
fn unknown_regenerates_and_disputed_escalates_by_the_fallback_policy() {
    let view = eligibility(&["gen", "boss"]);
    // Unknown -> regenerate -> a clean second proposal is accepted.
    let mut vcw = plain(roster(&["gen"], &[]));
    let verifier = DocVerifier::new();
    verifier.scripted.borrow_mut().push_back(report(
        VerificationStatus::Unknown,
        VerificationLevel::Unverified,
        &[],
    ));
    let mut stub = Stub::new(|_, n| doc(&format!("u{n}"), &[]));
    let run = run_doc(
        &mut vcw,
        &request("unknown", 3, 3, VerificationLevel::SampleVerified),
        &view,
        &verifier,
        &mut stub,
    );
    assert_eq!(verified(&run).payload().tag, "u2");
    assert_eq!(run.trace.steps()[0].decision, StepDecision::Regenerate);
    assert_eq!(verifier.calls.get(), 2);

    // Disputed -> escalate (default policy): the host target authors a candidate, which is
    // verified like any other.
    let mut vcw = plain(roster(&["gen"], &[]));
    let verifier = DocVerifier::new();
    verifier.scripted.borrow_mut().push_back(report(
        VerificationStatus::Disputed,
        VerificationLevel::Deterministic,
        &[],
    ));
    let mut stub = Stub::new(|_, n| doc(&format!("e{n}"), &[]));
    stub.escalation = Some((PodRef::new(pod("boss"), Generation(1)), doc("boss", &[])));
    let run = run_doc(
        &mut vcw,
        &request("disputed", 3, 3, VerificationLevel::SampleVerified),
        &view,
        &verifier,
        &mut stub,
    );
    let answer = verified(&run);
    assert_eq!(answer.payload().tag, "boss");
    assert_eq!(answer.author().pod, pod("boss"));
    assert!(run.trace.steps()[1].origin == ptr_runtime::vcw::StepOrigin::Escalated);

    // No escalation target: the request is handed to the host, with no candidate in it.
    let mut vcw = plain(roster(&["gen"], &[]));
    let verifier = DocVerifier::always(report(
        VerificationStatus::Disputed,
        VerificationLevel::Deterministic,
        &[],
    ));
    let mut stub = Stub::new(|_, n| doc(&format!("e{n}"), &[]));
    let run = run_doc(
        &mut vcw,
        &request("nobody", 3, 3, VerificationLevel::SampleVerified),
        &view,
        &verifier,
        &mut stub,
    );
    assert!(matches!(
        run.outcome,
        VcwOutcome::Escalated {
            reason: EscalationReason::Policy(FallbackReason::Disputed)
        }
    ));
    assert_eq!(run.trace.outcome(), OutcomeTag::Escalated);

    // An escalation author that is not eligible is dropped, never verified.
    let mut vcw = plain(roster(&["gen"], &[]));
    let verifier = DocVerifier::always(report(
        VerificationStatus::Disputed,
        VerificationLevel::Deterministic,
        &[],
    ));
    let mut stub = Stub::new(|_, n| doc(&format!("e{n}"), &[]));
    stub.escalation = Some((PodRef::new(pod("ghost"), Generation(1)), doc("ghost", &[])));
    let run = run_doc(
        &mut vcw,
        &request("ghost", 3, 3, VerificationLevel::SampleVerified),
        &view,
        &verifier,
        &mut stub,
    );
    assert!(!is_verified(&run));
    assert_eq!(verifier.calls.get(), 1);
    assert_eq!(vcw.counters().escalation_author_ineligible, 1);
}

#[test]
fn fallback_policy_is_honoured_for_each_reason() {
    // Configure Unknown to regenerate (required) and FailWithoutHardFinding to escalate.
    let router = RouterConfig {
        fallback: FallbackPolicy {
            fail_without_hard_finding: FallbackAction::Escalate,
            ..FallbackPolicy::default()
        },
        ..RouterConfig::default()
    };
    let mut vcw = VcwLoop::new(parts_with(
        base_config(),
        roster(&["gen"], &[]),
        router,
        UniformScorer,
        7,
        None,
    ))
    .unwrap();
    let verifier = DocVerifier::always(report(
        VerificationStatus::Fail,
        VerificationLevel::Deterministic,
        &[("soft", false)],
    ));
    let mut stub = Stub::new(|_, n| doc(&format!("x{n}"), &[]));
    let run = run_doc(
        &mut vcw,
        &request("fwhf", 4, 4, VerificationLevel::SampleVerified),
        &eligibility(&["gen"]),
        &verifier,
        &mut stub,
    );
    assert!(stub.log.iter().any(|l| l.starts_with("escalate:")));
    assert!(!is_verified(&run));

    // A policy that escalates Unknown cannot draw the first proposal and is refused up front.
    let mut router = RouterConfig::default();
    router.fallback.unknown = FallbackAction::Escalate;
    let refused = VcwLoop::new(parts_with(
        base_config(),
        roster(&["gen"], &[]),
        router,
        UniformScorer,
        7,
        None,
    ));
    assert!(matches!(refused, Err(VcwError::Config(_))));
}

// ---------------------------------------------------------------------------------------------
// Repair by finding code
// ---------------------------------------------------------------------------------------------

#[test]
fn each_hard_code_goes_to_its_repair_pod_which_is_asked_only_for_that_code() {
    let mut vcw = plain(two_defect_roster());
    let verifier = DocVerifier::new();
    let mut stub = two_defect_stub();
    let run = run_doc(
        &mut vcw,
        &request("repair", 4, 4, VerificationLevel::SampleVerified),
        &two_defect_view(),
        &verifier,
        &mut stub,
    );
    let answer = verified(&run);
    assert!(answer.payload().defects.is_empty());
    // One proposal, one repair round, two verifier calls: the unrepaired draft and the repair.
    assert_eq!(verifier.calls.get(), 2);
    assert_eq!(
        stub.asked_codes,
        vec![
            ("fix_a".to_owned(), vec!["a".to_owned()]),
            ("fix_b".to_owned(), vec!["b".to_owned()])
        ]
    );
    // The first step's class is Repair([a, b]); the repaired step has two routing records, one
    // per code, each with a propensity and the exact support it was drawn from.
    let steps = run.trace.steps();
    assert!(matches!(
        &steps[0].candidate.as_ref().unwrap().class,
        ClassSummary::Repair(codes) if codes == &[code("a"), code("b")]
    ));
    assert_eq!(steps[1].routes.len(), 2);
    for record in &steps[1].routes {
        assert_eq!(
            record.support().iter().map(|(_, w)| w).sum::<u64>(),
            1 << 53
        );
        assert!(record.propensity().get() > 0.0);
        assert!(matches!(record.class(), RouteClass::Repair(_)));
    }
    assert_eq!(answer.author().pod, pod("fix_b"));
    assert_eq!(run.trace.spent().verifier_calls, 2);
    assert_eq!(run.trace.spent().rounds, 1);
}

#[test]
fn a_repair_that_touches_unnamed_parts_is_discarded_and_replaced_by_a_fresh_draw() {
    let mut vcw = plain(two_defect_roster());
    let verifier = DocVerifier::new();
    let mut stub = two_defect_stub();
    stub.overreach = true;
    let run = run_doc(
        &mut vcw,
        &request("scope", 3, 3, VerificationLevel::SampleVerified),
        &two_defect_view(),
        &verifier,
        &mut stub,
    );
    // "a" is fixed but the edit also drops "b": out of scope. Nothing but the verifier's
    // verdict on a fresh draw can end the request, and the fresh draws have the same defects.
    assert!(!is_verified(&run));
    assert!(vcw.counters().repairs_out_of_scope > 0);
    assert_eq!(vcw.counters().repairs_applied, 0);
    // The out-of-scope edit was never verified: every verified candidate is a proposal draft.
    assert!(verifier
        .seen
        .borrow()
        .iter()
        .all(|d| *d != doc("d1", &[]).payload_digest()));
}

#[test]
fn a_code_with_no_eligible_repair_pod_is_counted_and_the_request_regenerates() {
    let mut vcw = plain(roster(&["gen"], &[("fix_a", &["a"])]));
    let view = eligibility(&["gen", "fix_a"]);
    let verifier = DocVerifier::new();
    let mut stub = Stub::new(|_, n| doc(&format!("z{n}"), &["zzz"])).fixing("fix_a", &["a"]);
    let run = run_doc(
        &mut vcw,
        &request("unroutable", 3, 2, VerificationLevel::SampleVerified),
        &view,
        &verifier,
        &mut stub,
    );
    assert!(!is_verified(&run));
    let counts = vcw.router().unroutable_counts();
    // Two rejected drafts were routed before the budget ended the request.
    assert_eq!(counts.get(&code("zzz")), Some(&2));
    assert_eq!(vcw.counters().unroutable_codes, 2);
    assert!(run.trace.steps()[0].unroutable.is_empty());
    assert_eq!(run.trace.steps()[1].unroutable, vec![code("zzz")]);
    // Nothing was repaired: no repair Pod was ever called.
    assert!(stub.log.iter().all(|l| l.starts_with("propose:")));
}

// ---------------------------------------------------------------------------------------------
// Budgets
// ---------------------------------------------------------------------------------------------

#[test]
fn budgets_are_never_exceeded_and_exhausted_is_returned_with_no_candidate() {
    for calls in 1..=5u32 {
        for rounds in 1..=5u32 {
            let mut vcw = plain(two_defect_roster());
            let verifier = DocVerifier::new();
            // Repair Pods that never fix anything: every candidate is a failing draft.
            let mut stub = Stub::new(|_, n| doc(&format!("b{n}"), &["a"]));
            let run = run_doc(
                &mut vcw,
                &request("budget", calls, rounds, VerificationLevel::SampleVerified),
                &two_defect_view(),
                &verifier,
                &mut stub,
            );
            let spent = run.trace.spent();
            assert!(
                verifier.calls.get() <= calls,
                "calls {calls} rounds {rounds}"
            );
            assert!(spent.rounds <= rounds);
            assert!(verifier.calls.get() <= 1 + rounds);
            assert_eq!(spent.verifier_calls, verifier.calls.get());
            // After a rejected verdict the calls are checked first, then the rounds; a request
            // holds at most 1 + rounds candidates.
            let expected = if calls <= rounds + 1 {
                ExhaustionReason::VerifierCalls
            } else {
                ExhaustionReason::Rounds
            };
            assert!(
                matches!(run.outcome, VcwOutcome::Exhausted { reason } if reason == expected),
                "calls {calls} rounds {rounds}: {:?}",
                run.outcome
            );
            // Each candidate was verified exactly once.
            let seen = verifier.seen.borrow();
            let distinct: BTreeSet<_> = seen.iter().collect();
            assert_eq!(distinct.len(), seen.len());
        }
    }
}

#[test]
fn a_pod_that_produces_nothing_spends_rounds_and_ends_exhausted() {
    let mut vcw = plain(roster(&["gen"], &[]));
    let verifier = DocVerifier::new();
    let mut stub = Stub::new(|_, n| doc(&format!("n{n}"), &[]));
    stub.none_proposals = true;
    let run = run_doc(
        &mut vcw,
        &request("nothing", 5, 3, VerificationLevel::SampleVerified),
        &eligibility(&["gen"]),
        &verifier,
        &mut stub,
    );
    assert!(matches!(
        run.outcome,
        VcwOutcome::Exhausted {
            reason: ExhaustionReason::Rounds
        }
    ));
    assert_eq!(verifier.calls.get(), 0);
    assert_eq!(run.trace.spent().rounds, 3);
    assert!(run
        .trace
        .steps()
        .iter()
        .all(|s| s.decision == StepDecision::NoCandidate));
}

#[test]
fn no_eligible_proposer_hands_the_request_to_the_host() {
    let mut vcw = plain(roster(&["gen"], &[]));
    let run = run_doc(
        &mut vcw,
        &request("alone", 3, 3, VerificationLevel::SampleVerified),
        &eligibility(&[]),
        &DocVerifier::new(),
        &mut Stub::new(|_, n| doc(&format!("n{n}"), &[])),
    );
    assert!(matches!(
        run.outcome,
        VcwOutcome::Escalated {
            reason: EscalationReason::NoProposer(FallbackReason::Unknown)
        }
    ));
}

// ---------------------------------------------------------------------------------------------
// Eligibility: one view for the router and the workspace
// ---------------------------------------------------------------------------------------------

#[test]
fn revoked_and_stale_generation_pods_are_never_chosen() {
    let mut vcw = plain(roster(
        &["gen"],
        &[("r1", &["a"]), ("r2", &["a"]), ("r3", &["a"])],
    ));
    let mut view = eligibility(&["gen", "r2", "r3"]);
    // r1 is revoked (absent); r2 moved on to generation 2, so generation 1 is stale.
    view.set(pod("r2"), Generation(2));
    let mut chosen = BTreeSet::new();
    for i in 0..60 {
        let verifier = DocVerifier::new();
        let mut stub = Stub::new(|_, n| doc(&format!("e{n}"), &["a"]))
            .fixing("r1", &["a"])
            .fixing("r2", &["a"])
            .fixing("r3", &["a"]);
        let run = run_doc(
            &mut vcw,
            &request(
                &format!("elig-{i}"),
                3,
                3,
                VerificationLevel::SampleVerified,
            ),
            &view,
            &verifier,
            &mut stub,
        );
        assert!(is_verified(&run));
        for record in run.trace.routes() {
            chosen.insert((record.pod().0.clone(), record.pod_generation()));
            assert_ne!(record.pod(), &pod("r1"));
            if record.pod() == &pod("r2") {
                assert_eq!(record.pod_generation(), Generation(2));
            }
            assert!(record.support().iter().all(|(p, _)| p != &pod("r1")));
        }
    }
    // Both live repair Pods were used, so the exclusion is not an artefact of one Pod winning.
    assert!(chosen.contains(&("r2".to_owned(), Generation(2))));
    assert!(chosen.contains(&("r3".to_owned(), Generation(1))));
    assert!(vcw.router().ineligible_skipped() > 0);
}

/// An eligibility view the test can change while a request runs.
#[derive(Clone, Default)]
struct LiveView(Arc<RefCell<BTreeMap<PodId, Generation>>>);

impl LiveView {
    fn with(pods: &[&str]) -> Self {
        let view = Self::default();
        for p in pods {
            view.0.borrow_mut().insert(pod(p), Generation(1));
        }
        view
    }
}

impl PodEligibility for LiveView {
    fn current_generation(&self, p: &PodId) -> Option<Generation> {
        self.0.borrow().get(p).copied()
    }
}

#[test]
fn a_pod_revoked_mid_request_is_neither_admitted_to_the_workspace_nor_accepted() {
    let mut vcw = plain(roster(&["p1", "p2"], &[]));
    let view = LiveView::with(&["p1", "p2"]);
    let verifier = DocVerifier::new();
    let mut stub = Stub::new(|_, n| doc(&format!("m{n}"), &[]));
    let first = Rc::new(RefCell::new(None::<String>));
    {
        let view = view.clone();
        let first = Rc::clone(&first);
        stub.on_propose = Some(Box::new(move |who| {
            // The first Pod asked is revoked while it works.
            if first.borrow().is_none() {
                *first.borrow_mut() = Some(who.to_owned());
                view.0.borrow_mut().remove(&pod(who));
            }
        }));
    }
    let run = run_doc(
        &mut vcw,
        &request("mid", 4, 4, VerificationLevel::SampleVerified),
        &view,
        &verifier,
        &mut stub,
    );
    let revoked = first.borrow().clone().unwrap();
    let step0 = &run.trace.steps()[0];
    // The draft was clean and passed the verifier, but its author was gone: not accepted, and
    // the workspace refused it as written by a revoked author.
    assert!(matches!(
        step0.candidate.as_ref().unwrap().class,
        ClassSummary::Accept
    ));
    assert_eq!(step0.decision, StepDecision::AuthorNoLongerEligible);
    assert_eq!(
        step0.candidate.as_ref().unwrap().workspace,
        WorkspaceOutcome::Refused(RefusalKind::AuthorRevoked)
    );
    let answer = verified(&run);
    assert_ne!(answer.author().pod.0, revoked);
    assert_eq!(answer.workspace(), WorkspaceOutcome::Admitted);
    assert_eq!(vcw.counters().author_no_longer_eligible, 1);
}

// ---------------------------------------------------------------------------------------------
// Workspace broadcast
// ---------------------------------------------------------------------------------------------

#[test]
fn refusals_and_their_finding_codes_are_broadcast_to_the_next_pod() {
    let mut vcw = plain(two_defect_roster());
    let verifier = DocVerifier::new();
    let mut stub = two_defect_stub();
    let run = run_doc(
        &mut vcw,
        &request("broadcast", 4, 4, VerificationLevel::SampleVerified),
        &two_defect_view(),
        &verifier,
        &mut stub,
    );
    assert!(is_verified(&run));
    // The first proposal saw no snapshot; the workspace then judged the failing draft and
    // broadcast the refusal with its codes.
    assert_eq!(stub.snapshots[0], None);
    let step0 = run.trace.steps()[0].candidate.as_ref().unwrap();
    assert!(matches!(
        step0.workspace,
        WorkspaceOutcome::Refused(RefusalKind::HardFinding | RefusalKind::NotPassed)
    ));
    // Repair Pods ran after that cycle and the accepted answer holds a slot.
    let answer = verified(&run);
    assert_eq!(answer.workspace(), WorkspaceOutcome::Admitted);
    assert!(run.trace.final_snapshot_digest().is_some());
    assert_ne!(
        run.trace.steps()[0]
            .candidate
            .as_ref()
            .unwrap()
            .snapshot_digest,
        run.trace.steps()[1]
            .candidate
            .as_ref()
            .unwrap()
            .snapshot_digest
    );
}

// ---------------------------------------------------------------------------------------------
// Replay determinism and the trace
// ---------------------------------------------------------------------------------------------

fn scenario_digests(key: u8) -> Vec<Digest> {
    let mut vcw = VcwLoop::new(parts_with(
        base_config(),
        two_defect_roster(),
        RouterConfig::default(),
        UniformScorer,
        key,
        None,
    ))
    .unwrap();
    (0..12)
        .map(|i| {
            let verifier = DocVerifier::new();
            let mut stub = two_defect_stub();
            let run = run_doc(
                &mut vcw,
                &request(
                    &format!("replay-{i}"),
                    4,
                    4,
                    VerificationLevel::SampleVerified,
                ),
                &two_defect_view(),
                &verifier,
                &mut stub,
            );
            assert_eq!(&run.trace.compute_digest(), run.trace.digest());
            *run.trace.digest()
        })
        .collect()
}

#[test]
fn the_trace_digest_replays_and_depends_on_the_seed() {
    let first = scenario_digests(7);
    let second = scenario_digests(7);
    assert_eq!(first, second);
    assert_ne!(first, scenario_digests(8));
    // Different requests give different digests.
    assert_ne!(first[0], first[1]);
}

#[test]
fn the_trace_digest_covers_the_verdicts() {
    let run_with = |level: VerificationLevel| {
        let mut vcw = plain(two_defect_roster());
        let mut verifier = DocVerifier::new();
        verifier.level = level;
        let mut stub = two_defect_stub();
        run_doc(
            &mut vcw,
            &request("cover", 4, 4, VerificationLevel::SampleVerified),
            &two_defect_view(),
            &verifier,
            &mut stub,
        )
    };
    let a = run_with(VerificationLevel::Deterministic);
    let b = run_with(VerificationLevel::FullSemantic);
    assert!(is_verified(&a) && is_verified(&b));
    assert_ne!(a.trace.digest(), b.trace.digest());
}

#[test]
fn the_self_model_predicts_before_it_learns_at_every_candidate() {
    let mut vcw = plain(two_defect_roster());
    let verifier = DocVerifier::new();
    let mut stub = two_defect_stub();
    let run = run_doc(
        &mut vcw,
        &request("prequential", 4, 4, VerificationLevel::SampleVerified),
        &two_defect_view(),
        &verifier,
        &mut stub,
    );
    let candidates: Vec<_> = run.trace.candidates().collect();
    assert_eq!(candidates.len(), 2);
    assert_eq!(vcw.self_model().log().count(), candidates.len());
    // The first prediction was made with no evidence for the context; the labels it learned are
    // the verifier's.
    assert_eq!(candidates[0].self_model.prediction.evidence, 0);
    assert!(!candidates[0].self_model.passed);
    assert!(candidates[1].self_model.passed);
    let logged: Vec<bool> = vcw.self_model().log().map(|l| l.outcome.passed).collect();
    assert_eq!(logged, vec![false, true]);
    assert_eq!(run.trace.self_model_digest(), &vcw.self_model().digest());
}

// ---------------------------------------------------------------------------------------------
// Observers: schema arms and hints
// ---------------------------------------------------------------------------------------------

fn two_repairer_roster() -> Roster {
    roster(&["gen"], &[("good", &["a"]), ("bad", &["a"])])
}

fn run_series<S: ptr_router::AdmissibleScorer>(
    vcw: &mut VcwLoop<S>,
    n: usize,
) -> (Vec<Digest>, usize) {
    let view = eligibility(&["gen", "good", "bad"]);
    let mut hints = 0;
    let digests = (0..n)
        .map(|i| {
            let verifier = DocVerifier::new();
            // `good` repairs, `bad` always declines.
            let mut stub = Stub::new(|_, n| doc(&format!("s{n}"), &["a"]))
                .fixing("good", &["a"])
                .fixing("bad", &[]);
            let run = run_doc(
                vcw,
                &request(
                    &format!("series-{i}"),
                    4,
                    4,
                    VerificationLevel::SampleVerified,
                ),
                &view,
                &verifier,
                &mut stub,
            );
            hints += run
                .trace
                .candidates()
                .map(|c| c.self_model.hints)
                .sum::<usize>();
            *run.trace.digest()
        })
        .collect();
    (digests, hints)
}

fn schema() -> Arc<AttentionSchema> {
    Arc::new(AttentionSchema::new(SchemaLimits::default()).unwrap())
}

fn controlled(
    schema: &Arc<AttentionSchema>,
    arm: ControlArm,
    mode: HintMode,
) -> VcwLoop<ControlledScorer<UniformScorer>> {
    let mut config = base_config();
    config.hint_mode = mode;
    config.hint_policy = HintPolicy {
        escalate_below: 0.01,
        min_evidence: 5,
        prefer_margin: 0.0,
    };
    VcwLoop::new(parts_with(
        config,
        two_repairer_roster(),
        RouterConfig::default(),
        ControlledScorer::new(
            UniformScorer,
            Arc::clone(schema),
            arm,
            ControlConfig::default(),
        ),
        7,
        Some(Arc::clone(schema)),
    ))
    .unwrap()
}

#[test]
fn an_observe_only_schema_leaves_routing_exactly_as_the_inner_scorer() {
    let schema = schema();
    let mut observing = controlled(&schema, ControlArm::ObserveOnly, HintMode::TraceOnly);
    let mut config = base_config();
    config.hint_policy = HintPolicy {
        escalate_below: 0.01,
        min_evidence: 5,
        prefer_margin: 0.0,
    };
    let mut inner = VcwLoop::new(parts_with(
        config,
        two_repairer_roster(),
        RouterConfig::default(),
        UniformScorer,
        7,
        None,
    ))
    .unwrap();
    let (a, _) = run_series(&mut observing, 40);
    let (b, _) = run_series(&mut inner, 40);
    // The traces differ only if the self-model digest differs; the schema is not in the trace,
    // so equal digests mean equal routing records, verdicts and predictions.
    assert_eq!(a, b);
    // The schema did observe: it scored forecasts over the two-Pod supports.
    assert!(schema.report().observations > 0);
}

#[test]
fn hints_reach_routing_only_when_the_host_applies_them() {
    let trace_only = schema();
    let mut traced = controlled(&trace_only, ControlArm::Full, HintMode::TraceOnly);
    let (_, hints) = run_series(&mut traced, 160);
    // Sensitivity: the self-model formed advice, and the loop kept it out of the schema.
    assert!(hints > 0, "no hint was formed");
    assert_eq!(trace_only.report().preferences, 0);

    let applied = schema();
    let mut applying = controlled(&applied, ControlArm::Full, HintMode::ApplyToSchema);
    let (_, hints) = run_series(&mut applying, 160);
    assert!(hints > 0);
    assert!(applied.report().preferences > 0);
}

// ---------------------------------------------------------------------------------------------
// L3: a Birth request is surfaced, completed by the host and admitted through the ledger path
// ---------------------------------------------------------------------------------------------

fn base() -> BaseModelRef {
    BaseModelRef::new(
        BaseModelId::new("base-a").unwrap(),
        BaseModelDigest::new([9; 32]).unwrap(),
        1,
    )
    .unwrap()
}

fn proposer() -> ProposerRef {
    ProposerRef::new(project(), pod("meta"), Generation(1)).unwrap()
}

fn trained(seed: u8) -> ManifestDigest {
    ManifestDigest::new([seed; 32]).unwrap()
}

/// Thresholds with exactly representable values; only Birth is enabled.
fn just_birth() -> StructurePolicyConfig {
    StructurePolicyConfig {
        min_unroutable: 4,
        merge_clear_rate_margin: 0.25,
        merge_overlap_margin: 0.25,
        merge_min_routed: 4,
        prune_usage_floor: 3,
        freeze_windows: 2,
        freeze_max_usage_variance: 1.0,
        freeze_max_clear_rate_variance: 0.015625,
        freeze_min_usage: 5,
        max_proposals_per_window: 16,
        max_proposals_per_expert: 1,
        birth_windows: 1,
        max_births_per_window: 16,
        merge_min_clear_rate: 0.25,
        prune_windows: 1,
        prune_min_age_commits: 0,
        backup_min_cleared: 1,
        backup_clear_windows: 1,
        freeze_min_clear_rate: 0.25,
        ops: EnabledOps {
            birth: true,
            merge: false,
            prune: false,
            freeze: false,
            reactivate: false,
        },
        ..StructurePolicyConfig::default()
    }
}

fn gate() -> ForgettingGate {
    ForgettingGate {
        max_average_forgetting: 0.5,
        max_task_forgetting: 0.5,
        min_backward_transfer: -0.5,
        max_public_regression: 0.02,
    }
}

fn gate_report(proposal: &ptr_pods::ExpertProposal, public: PublicSuite) -> BoundGateReport {
    let subject = GateSubject::new(
        *proposal.proposal_id().as_bytes(),
        *proposal.operation_id().as_bytes(),
        expert_evidence_artifact_digest(proposal.op()),
        expert_evidence_base_model_digest(proposal.op()).unwrap(),
        EVAL_MANIFEST,
    );
    let matrix = AccuracyMatrix::new(vec![vec![0.9, 0.0], vec![0.9, 0.9]]).unwrap();
    gate()
        .evaluate(&AdapterId::from("candidate"), &matrix, public)
        .unwrap()
        .bind(subject)
}

fn passing() -> PublicSuite {
    PublicSuite {
        serving: 0.8,
        candidate: 0.8,
    }
}

fn failing() -> PublicSuite {
    PublicSuite {
        serving: 0.9,
        candidate: 0.1,
    }
}

fn grant() -> ExpertAdmissionGrant {
    ExpertAdmissionGrant::new(
        project(),
        proposer(),
        requirement(7, gate().config_digest().unwrap()),
        binding(),
    )
}

struct StubPod(PodManifest);

impl DynPod for StubPod {
    fn manifest(&self) -> &PodManifest {
        &self.0
    }

    fn invoke(&self, input: TypedPayload) -> Result<TypedPayload, String> {
        Ok(input)
    }
}

fn stub_pod(name: &str) -> Arc<dyn DynPod> {
    Arc::new(StubPod(PodManifest {
        project: project(),
        id: pod(name),
        capabilities: vec![CapabilityId::from("repair")],
        accepts: vec![TypeId::from("doc")],
        produces: vec![TypeId::from("doc")],
        effects: vec![Effect::Pure],
        protocol_version: 1,
    }))
}

/// A primed runtime in which the base Pods `gen` and `fixer_a` are live at generation 1 in the
/// lifecycle authority (a capsule commit per Pod id).
fn live_runtime() -> PtrRuntime {
    let mut rt = PtrRuntime::new(PtrConfig::default()).unwrap();
    prime(&mut rt);
    for name in ["gen", "fixer_a"] {
        make_live(&mut rt, &project(), name, 1);
    }
    rt
}

fn make_live(rt: &mut PtrRuntime, project: &ProjectId, name: &str, generation: u64) {
    rt.commit(ptr_ledger::LedgerEvent::CapsuleCommitted {
        project: project.clone(),
        capsule: ptr_types::CapsuleId::from(name),
        generation: Generation(generation),
    })
    .unwrap();
}

fn l3_loop() -> VcwLoop {
    let mut config = base_config();
    config.window_requests = 2;
    let mut parts = parts_with(
        config,
        roster(&["gen"], &[("fixer_a", &["a"])]),
        RouterConfig::default(),
        UniformScorer,
        7,
        None,
    );
    parts.structure = StructurePolicy::new(just_birth()).unwrap();
    VcwLoop::new(parts).unwrap()
}

fn run_in_runtime(
    vcw: &mut VcwLoop,
    rt: &PtrRuntime,
    pods: &PodRegistry,
    id: &str,
    stub: &mut Stub,
) -> VcwRun<Doc> {
    let project = project();
    let standing = RuntimeStanding::new(rt, pods, &project)
        .with_base_pod(pod("gen"), Generation(1))
        .with_base_pod(pod("fixer_a"), Generation(1));
    vcw.run(
        &request(id, 3, 2, VerificationLevel::SampleVerified),
        &standing,
        &DocVerifier::new(),
        stub,
    )
    .unwrap()
}

#[test]
fn a_birth_request_is_surfaced_completed_admitted_and_seen_by_the_next_request() {
    let mut rt = live_runtime();
    let mut pods = PodRegistry::default();
    pods.register_new(stub_pod("gen")).unwrap();
    pods.register_new(stub_pod("fixer_a")).unwrap();
    let mut vcw = l3_loop();

    // Two requests whose drafts carry a code no registered Pod repairs: unroutable every round.
    let mk = || Stub::new(|_, n| doc(&format!("n{n}"), &["newcode"])).fixing("fixer_a", &["a"]);
    for i in 0..2 {
        let run = run_in_runtime(&mut vcw, &rt, &pods, &format!("l3-{i}"), &mut mk());
        assert!(!is_verified(&run));
    }
    assert_eq!(vcw.windows_closed(), 1);
    // The window carries exactly the router's unroutable counter for the code.
    let window = vcw.closed_windows().next().unwrap().clone();
    let stats = &window.codes[&code("newcode")];
    assert_eq!(stats.unroutable, 4);
    assert_eq!(
        vcw.router().unroutable_counts().get(&code("newcode")),
        Some(&4)
    );
    assert_eq!(window.usage[&pod("gen")], 6);

    // The policy's request comes back to the host. Nothing was admitted by the loop.
    let events = rt.committed_events().len();
    fn host(rt: &PtrRuntime) -> StructureHost<'_> {
        StructureHost {
            project: project(),
            proposer: proposer(),
            observed_at: CommitIndex(1),
            registry: rt.expert_registry(),
            birth: Some(BirthContext {
                base_model: base(),
                site: BirthSite::Segment,
            }),
            aimer: BTreeMap::new(),
            overlaps: Vec::new(),
        }
    }
    let report = vcw.structure_step(host(&rt)).unwrap();
    assert_eq!(report.proposals().len(), 1);
    assert_eq!(rt.committed_events().len(), events);
    assert_eq!(vcw.outstanding_requests(), 1);
    let request = &report.proposals()[0];
    assert!(request.needs_training());
    assert_eq!(request.ready(), None);
    let born = request.request().op().results()[0].pod().clone();
    // The same Birth is not proposed again while this one is outstanding.
    assert!(vcw
        .structure_step(host(&rt))
        .unwrap()
        .proposals()
        .is_empty());

    // The host trains (a digest), completes the proposal and binds a gate report to it.
    let birth = request.complete(trained(0x42)).unwrap();
    // Not committed yet, so it cannot be enrolled.
    assert!(matches!(
        vcw.sync_roster(&project(), rt.expert_registry(), &birth),
        Err(VcwError::NotCommitted { .. })
    ));
    // A failing gate report commits nothing and enrols nothing.
    assert_eq!(
        rt.admit_expert_proposal(&birth, &gate_report(&birth, failing()), &grant()),
        Err(RuntimeError::Expert(
            ptr_runtime::expert::ExpertAdmissionError::GateFailed
        ))
    );
    assert_eq!(rt.committed_events().len(), events);
    assert!(vcw
        .sync_roster(&project(), rt.expert_registry(), &birth)
        .is_err());

    // The existing admission path commits the structural event.
    rt.admit_expert_proposal(&birth, &gate_report(&birth, passing()), &grant())
        .unwrap();
    assert_eq!(rt.committed_events().len(), events + 1);
    assert!(matches!(
        rt.committed_events().last().unwrap().event,
        ptr_ledger::LedgerEvent::ExpertStructuralCommitted { .. }
    ));
    assert_eq!(
        rt.expert_registry().get(&project(), &born).unwrap().state,
        ExpertState::Serving
    );

    // The expert is committed but the Pod registry has no executable Pod for it yet: the
    // eligibility view does not list it, so it is enrolled for the code and still never chosen.
    let sync = vcw
        .sync_roster(&project(), rt.expert_registry(), &birth)
        .unwrap();
    assert_eq!(sync.enrolled, vec![(born.clone(), vec![code("newcode")])]);
    assert_eq!(vcw.outstanding_requests(), 0);
    let mut stub = mk().fixing(&born.0, &["newcode"]);
    let run = run_in_runtime(&mut vcw, &rt, &pods, "after-commit", &mut stub);
    assert!(!is_verified(&run));
    assert!(run.trace.routes().all(|r| r.pod() != &born));
    assert!(vcw.router().ineligible_skipped() > 0);

    // The host registers the Pod; the next request's view lists it at the committed generation
    // and the router routes the code to it.
    pods.register_new(stub_pod(&born.0)).unwrap();
    let mut stub = mk().fixing(&born.0, &["newcode"]);
    let run = run_in_runtime(&mut vcw, &rt, &pods, "after-register", &mut stub);
    let answer = verified(&run);
    assert_eq!(answer.author().pod, born);
    let record = run
        .trace
        .routes()
        .find(|r| matches!(r.class(), RouteClass::Repair(_)))
        .unwrap();
    assert_eq!(record.pod(), &born);
    assert_eq!(record.pod_generation(), Generation(1));
    assert!(matches!(record.class(), RouteClass::Repair(c) if c == &code("newcode")));
}

/// Every `.rs` file under `dir`, including those of submodule directories.
fn rust_sources(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut found = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            found.extend(rust_sources(&path));
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            found.push(path);
        }
    }
    found
}

#[test]
fn the_loop_holds_no_handle_that_could_admit_commit_or_change_a_gate() {
    // The structure step takes a read-only projection: its signature has no runtime in it.
    let _step: fn(
        &mut VcwLoop<UniformScorer>,
        StructureHost<'_>,
    ) -> Result<ptr_runtime::expert::StructureReport, VcwError> =
        VcwLoop::<UniformScorer>::structure_step;

    // Source scan: outside comments, the module never touches admission, fencing, commit or
    // authority state, and only the output-admission bridge calls into the runtime at all.
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/vcw");
    let forbidden = [
        "admit_expert_proposal",
        "fence_expert",
        "readmit_expert",
        ".commit(",
        "commit_",
        "append_",
        "permissions_mut",
        "install_policy_key",
        "revoke_",
        "activate_policy",
        "set_live_generation",
        "ExpertAdmissionGrant",
        "BoundGateReport",
        "ForgettingGate",
        "&mut PtrRuntime",
        "&mut self.runtime",
    ];
    let mut files = 0;
    for path in rust_sources(&dir) {
        let text = std::fs::read_to_string(&path).unwrap();
        files += 1;
        let source: String = text
            .lines()
            .filter(|line| !line.trim_start().starts_with("//"))
            .map(|line| format!("{line}\n"))
            .collect();
        // Exemptions are by path relative to `src/vcw`, so a same-named file in a subdirectory
        // does not inherit them.
        let name = path
            .strip_prefix(&dir)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        for token in forbidden {
            // The one place that holds the runtime mutably is the output-admission bridge.
            if name == "admit.rs" && token == "&mut PtrRuntime" {
                continue;
            }
            assert!(!source.contains(token), "{name} mentions {token}");
        }
        if name != "admit.rs" {
            assert!(!source.contains("admit_pod_output"), "{name}");
        }
        // The eligibility view reads the runtime through a shared reference; nothing else in the
        // module names it.
        if name != "admit.rs" && name != "eligibility.rs" {
            assert!(!source.contains("PtrRuntime"), "{name}");
        }
    }
    assert!(files >= 6);
}

// ---------------------------------------------------------------------------------------------
// Bounded tables
// ---------------------------------------------------------------------------------------------

#[test]
fn windows_outstanding_requests_and_traces_are_bounded() {
    let mut config = base_config();
    config.window_requests = 1;
    config.max_windows = 2;
    config.max_outstanding = 1;
    let mut parts = parts_with(
        config,
        roster(&["gen"], &[("fixer_a", &["a"])]),
        RouterConfig::default(),
        UniformScorer,
        7,
        None,
    );
    parts.structure = StructurePolicy::new(StructurePolicyConfig {
        min_unroutable: 1,
        ..just_birth()
    })
    .unwrap();
    let mut vcw = VcwLoop::new(parts).unwrap();
    let rt = live_runtime();
    let mut pods = PodRegistry::default();
    pods.register_new(stub_pod("gen")).unwrap();
    pods.register_new(stub_pod("fixer_a")).unwrap();
    // Five requests, each with two codes of its own that no Pod repairs.
    for i in 0..5 {
        let (first, second) = (format!("odd{i}a"), format!("odd{i}b"));
        let mut stub =
            Stub::new(move |_, n| doc(&format!("o{n}"), &[first.as_str(), second.as_str()]));
        let run = run_in_runtime(&mut vcw, &rt, &pods, &format!("bound-{i}"), &mut stub);
        // At most 1 + rounds attempts, whatever happened.
        assert!(run.trace.steps().len() <= 3);
    }
    assert_eq!(vcw.windows_closed(), 5);
    assert_eq!(vcw.closed_windows().count(), 2);
    let host = StructureHost {
        project: project(),
        proposer: proposer(),
        observed_at: CommitIndex(1),
        registry: rt.expert_registry(),
        birth: Some(BirthContext {
            base_model: base(),
            site: BirthSite::Segment,
        }),
        aimer: BTreeMap::new(),
        overlaps: Vec::new(),
    };
    // Two windows with distinct codes would want two Births; the table holds one request, so the
    // step is refused whole and records nothing.
    let result = vcw.structure_step(host);
    assert!(
        matches!(result, Err(VcwError::OutstandingFull)),
        "{result:?}"
    );
    assert_eq!(vcw.outstanding_requests(), 0);
    assert_eq!(vcw.counters().requests_refused_outstanding_full, 1);
}

#[test]
fn a_trace_records_at_most_a_bounded_number_of_codes() {
    let many: Vec<(String, bool)> = (0..100).map(|i| (format!("c{i}"), true)).collect();
    let findings: Vec<(&str, bool)> = many.iter().map(|(c, h)| (c.as_str(), *h)).collect();
    let rep = report(
        VerificationStatus::Fail,
        VerificationLevel::Deterministic,
        &findings,
    );
    let mut vcw = plain(roster(&["gen"], &[]));
    let verifier = DocVerifier::always(rep);
    let mut stub = Stub::new(|_, n| doc(&format!("t{n}"), &[]));
    let run = run_doc(
        &mut vcw,
        &request("many", 1, 1, VerificationLevel::SampleVerified),
        &eligibility(&["gen"]),
        &verifier,
        &mut stub,
    );
    let summary = &run.trace.candidates().next().unwrap().report;
    assert_eq!(summary.hard_total, 100);
    assert!(summary.hard_codes.len() <= ptr_runtime::vcw::MAX_TRACE_CODES);
    // More distinct codes than the router routes is a typed fallback, never a truncation.
    assert!(matches!(
        run.trace.candidates().next().unwrap().class,
        ClassSummary::Fallback(FallbackReason::TooManyCodes)
    ));
}

// ---------------------------------------------------------------------------------------------
// From a verified answer into the runtime's output admission
// ---------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------
// Review round: authors, liveness, escalation, accounting, roster, workspace independence
// ---------------------------------------------------------------------------------------------

fn stub_pod_in(project_name: &str, name: &str) -> Arc<dyn DynPod> {
    Arc::new(StubPod(PodManifest {
        project: ProjectId::from(project_name),
        id: pod(name),
        capabilities: vec![CapabilityId::from("repair")],
        accepts: vec![TypeId::from("doc")],
        produces: vec![TypeId::from("doc")],
        effects: vec![Effect::Pure],
        protocol_version: 1,
    }))
}

#[test]
fn the_trace_and_the_answer_name_the_pods_whose_edits_are_in_the_candidate() {
    let mut vcw = plain(two_defect_roster());
    let verifier = DocVerifier::new();
    verifier.scripted.borrow_mut().push_back(report(
        VerificationStatus::Fail,
        VerificationLevel::Deterministic,
        &[("a", true), ("b", true)],
    ));
    verifier.scripted.borrow_mut().push_back(report(
        VerificationStatus::Pass,
        VerificationLevel::Deterministic,
        &[],
    ));
    // fix_a repairs `a`; fix_b is asked about `b` and declines.
    let mut stub = Stub::new(|_, n| doc(&format!("d{n}"), &["a", "b"]))
        .fixing("fix_a", &["a"])
        .fixing("fix_b", &[]);
    let run = run_doc(
        &mut vcw,
        &request("authors", 4, 4, VerificationLevel::SampleVerified),
        &two_defect_view(),
        &verifier,
        &mut stub,
    );
    let answer = verified(&run);
    let fix_a = PodRef::new(pod("fix_a"), Generation(1));
    assert!(stub.log.contains(&"repair:fix_b:b".to_owned()));
    assert_eq!(answer.authors(), std::slice::from_ref(&fix_a));
    assert_eq!(answer.author(), &fix_a);
    assert_eq!(
        run.trace.steps()[0].authors,
        vec![PodRef::new(pod("gen"), Generation(1))]
    );
    assert_eq!(run.trace.steps()[1].authors, vec![fix_a]);
    // The routing records still show both routed Pods.
    assert_eq!(run.trace.steps()[1].routes.len(), 2);
}

#[test]
fn an_accepted_escalated_candidate_has_its_author_in_the_trace_and_the_digest() {
    let run_with = |boss: &str| {
        let mut vcw = plain(roster(&["gen"], &[]));
        let verifier = DocVerifier::new();
        verifier.scripted.borrow_mut().push_back(report(
            VerificationStatus::Disputed,
            VerificationLevel::Deterministic,
            &[],
        ));
        let mut stub = Stub::new(|_, n| doc(&format!("e{n}"), &[]));
        stub.escalation = Some((PodRef::new(pod(boss), Generation(1)), doc("same", &[])));
        run_doc(
            &mut vcw,
            &request("esc", 3, 3, VerificationLevel::SampleVerified),
            &eligibility(&["gen", "boss1", "boss2"]),
            &verifier,
            &mut stub,
        )
    };
    let one = run_with("boss1");
    let two = run_with("boss2");
    assert_eq!(
        verified(&one).authors(),
        &[PodRef::new(pod("boss1"), Generation(1))]
    );
    assert_eq!(
        one.trace.steps()[1].authors,
        vec![PodRef::new(pod("boss1"), Generation(1))]
    );
    // Same payload, same verdicts, a different escalation author: the digest tells them apart.
    assert_ne!(one.trace.digest(), two.trace.digest());
}

#[test]
fn an_earlier_repairer_revoked_before_acceptance_blocks_acceptance_and_is_not_credited() {
    let mut config = base_config();
    config.window_requests = 1;
    let mut vcw = VcwLoop::new(parts_with(
        config,
        two_defect_roster(),
        RouterConfig::default(),
        UniformScorer,
        7,
        None,
    ))
    .unwrap();
    let view = LiveView::with(&["gen", "fix_a", "fix_b"]);
    let verifier = DocVerifier::new();
    let mut stub = two_defect_stub();
    {
        let view = view.clone();
        stub.on_repair = Some(Box::new(move |who| {
            // fix_a is revoked right after it made its edit.
            if who == "fix_a" {
                view.0.borrow_mut().remove(&pod("fix_a"));
            }
        }));
    }
    let run = run_doc(
        &mut vcw,
        &request("earlier", 4, 4, VerificationLevel::SampleVerified),
        &view,
        &verifier,
        &mut stub,
    );
    let step1 = &run.trace.steps()[1];
    // Both repairers' edits are in the payload and the payload passed; fix_a is gone.
    assert_eq!(step1.authors.len(), 2);
    assert!(matches!(
        step1.candidate.as_ref().unwrap().class,
        ClassSummary::Accept
    ));
    assert_eq!(step1.decision, StepDecision::AuthorNoLongerEligible);
    if let VcwOutcome::Verified(answer) = &run.outcome {
        assert!(answer.authors().iter().all(|p| p.pod != pod("fix_a")));
    }
    // The revoked repairer is credited with the routing and not with a clearing.
    let window = vcw.closed_windows().next().unwrap();
    let stats = window.codes[&code("a")].pods[&pod("fix_a")];
    assert_eq!(stats.routed, 1);
    assert_eq!(stats.cleared, 0);
}

/// Total (routed, cleared) credit of `pod` for `code` over all closed windows.
fn credit(vcw: &VcwLoop<impl AdmissibleScorer>, pod_name: &str, code_name: &str) -> (u64, u64) {
    vcw.closed_windows()
        .filter_map(|w| w.codes.get(&code(code_name)))
        .filter_map(|c| c.pods.get(&pod(pod_name)))
        .fold((0, 0), |(r, c), s| (r + s.routed, c + s.cleared))
}

fn one_request_windows() -> VcwConfig {
    let mut config = base_config();
    config.window_requests = 1;
    config
}

#[test]
fn a_pod_that_removes_the_code_but_introduces_another_is_not_credited_with_a_clearing() {
    // fix_a removes `a` and adds `z`; nobody repairs `z`. The verifier then reports only `z`:
    // `a` is gone from the report, but the candidate is worse in a way the routed code does not
    // explain, so fix_a is credited with the routing and with nothing else.
    let mut vcw = VcwLoop::new(parts_with(
        one_request_windows(),
        roster(&["gen"], &[("fix_a", &["a"])]),
        RouterConfig::default(),
        UniformScorer,
        7,
        None,
    ))
    .unwrap();
    let mut stub = Stub::new(|_, n| doc(&format!("loose{n}"), &["a"]))
        .fixing("fix_a", &["a"])
        .introducing("fix_a", "z");
    let run = run_doc(
        &mut vcw,
        &request("introduces", 6, 6, VerificationLevel::SampleVerified),
        &eligibility(&["gen", "fix_a"]),
        &DocVerifier::new(),
        &mut stub,
    );
    assert!(!is_verified(&run));
    let (routed, cleared) = credit(&vcw, "fix_a", "a");
    assert!(routed >= 1, "fix_a was routed");
    assert_eq!(
        cleared, 0,
        "a Pod that introduces a new code clears nothing"
    );
}

#[test]
fn a_flip_flop_between_two_codes_is_credited_to_neither_pod() {
    // fix_a turns `a` into `z`, fix_z turns `z` into `a`: every round one code "disappears" and
    // another appears. Neither Pod cleared anything.
    let mut vcw = VcwLoop::new(parts_with(
        one_request_windows(),
        roster(&["gen"], &[("fix_a", &["a"]), ("fix_z", &["z"])]),
        RouterConfig::default(),
        UniformScorer,
        7,
        None,
    ))
    .unwrap();
    let mut stub = Stub::new(|_, n| doc(&format!("loose{n}"), &["a"]))
        .fixing("fix_a", &["a"])
        .introducing("fix_a", "z")
        .fixing("fix_z", &["z"])
        .introducing("fix_z", "a");
    let run = run_doc(
        &mut vcw,
        &request("flip-flop", 6, 6, VerificationLevel::SampleVerified),
        &eligibility(&["gen", "fix_a", "fix_z"]),
        &DocVerifier::new(),
        &mut stub,
    );
    assert!(!is_verified(&run));
    for (pod_name, code_name) in [("fix_a", "a"), ("fix_z", "z")] {
        let (routed, cleared) = credit(&vcw, pod_name, code_name);
        assert!(routed >= 1, "{pod_name} was routed");
        assert_eq!(cleared, 0, "{pod_name} was credited in a flip-flop");
    }
}

#[test]
fn a_pod_that_clears_its_code_while_another_stays_defective_is_still_credited() {
    // Two defects, `a` fixed by fix_a, `b` declined by fix_b: the report after the round has
    // only `b`, which was routed, so fix_a cleared `a` and fix_b did not clear `b`.
    let mut vcw = VcwLoop::new(parts_with(
        one_request_windows(),
        two_defect_roster(),
        RouterConfig::default(),
        UniformScorer,
        7,
        None,
    ))
    .unwrap();
    let mut stub = Stub::new(|_, n| doc(&format!("d{n}"), &["a", "b"])).fixing("fix_a", &["a"]);
    run_doc(
        &mut vcw,
        &request("partial", 3, 3, VerificationLevel::SampleVerified),
        &two_defect_view(),
        &DocVerifier::new(),
        &mut stub,
    );
    assert_eq!(credit(&vcw, "fix_a", "a"), (1, 1));
    let (routed, cleared) = credit(&vcw, "fix_b", "b");
    assert!(routed >= 1);
    assert_eq!(cleared, 0, "fix_b declined; `b` is still reported");
}

#[test]
fn an_accepted_candidate_whose_repairer_was_revoked_is_no_reward_for_the_learner() {
    let vcw = VcwLoop::new(parts_with(
        one_request_windows(),
        two_defect_roster(),
        RouterConfig::default(),
        UniformScorer,
        7,
        None,
    ))
    .unwrap();
    let learner = Arc::new(RoutingLearner::new(LearnerConfig::new("docs", 3)).unwrap());
    let mut vcw = vcw.with_learner(Arc::clone(&learner)).unwrap();
    let view = LiveView::with(&["gen", "fix_a", "fix_b"]);
    let mut stub = two_defect_stub();
    {
        let view = view.clone();
        stub.on_repair = Some(Box::new(move |who| {
            if who == "fix_a" {
                view.0.borrow_mut().remove(&pod("fix_a"));
            }
        }));
    }
    run_doc(
        &mut vcw,
        &request("learner-revoked", 4, 4, VerificationLevel::SampleVerified),
        &view,
        &DocVerifier::new(),
        &mut stub,
    );
    // The tally gives the revoked repairer no clearing; the learner must agree: the candidate
    // that passed was not returned, so no route of that attempt is a reward.
    assert_eq!(credit(&vcw, "fix_a", "a").1, 0);
    let first_attempt: Vec<_> = learner
        .log_from(0)
        .into_iter()
        .filter(
            |d| matches!(d.class(), RouteClass::Repair(c) if c == &code("a") || c == &code("b")),
        )
        .take(2)
        .collect();
    assert_eq!(first_attempt.len(), 2);
    assert!(first_attempt.iter().all(|d| !d.reward().passed()));
}

#[test]
fn every_escalation_goes_through_the_escalation_target() {
    // Entry 1: no eligible proposer at the start.
    let mut vcw = plain(roster(&["gen"], &[]));
    let mut stub = Stub::new(|_, n| doc(&format!("n{n}"), &[]));
    stub.escalation = Some((PodRef::new(pod("boss"), Generation(1)), doc("boss", &[])));
    let run = run_doc(
        &mut vcw,
        &request("esc-start", 3, 3, VerificationLevel::SampleVerified),
        &eligibility(&["boss"]),
        &DocVerifier::new(),
        &mut stub,
    );
    assert_eq!(verified(&run).author().pod, pod("boss"));
    assert!(stub.log[0].starts_with("escalate:NoProposer"));
    assert_eq!(
        run.trace.steps()[0].origin,
        ptr_runtime::vcw::StepOrigin::Escalated
    );

    // Entry 2: a declined repair under a policy that escalates a Fail without usable finding.
    let router = RouterConfig {
        fallback: FallbackPolicy {
            fail_without_hard_finding: FallbackAction::Escalate,
            ..FallbackPolicy::default()
        },
        ..RouterConfig::default()
    };
    let mut vcw = VcwLoop::new(parts_with(
        base_config(),
        roster(&["gen"], &[("fix_a", &["a"])]),
        router,
        UniformScorer,
        7,
        None,
    ))
    .unwrap();
    let verifier = DocVerifier::new();
    let mut stub = Stub::new(|_, n| doc(&format!("d{n}"), &["a"])).fixing("fix_a", &[]);
    stub.escalation = Some((PodRef::new(pod("boss"), Generation(1)), doc("boss", &[])));
    let run = run_doc(
        &mut vcw,
        &request("esc-declined", 4, 4, VerificationLevel::SampleVerified),
        &eligibility(&["gen", "fix_a", "boss"]),
        &verifier,
        &mut stub,
    );
    assert_eq!(verified(&run).author().pod, pod("boss"));
    assert_eq!(stub.log[1], "repair:fix_a:a");
    assert!(stub.log[2].starts_with("escalate:Policy(FailWithoutHardFinding)"));
    assert_eq!(verifier.calls.get(), 2);

    // Entry 3: the policy escalates a report (covered above for Disputed) and nobody answers.
    let mut vcw = plain(roster(&["gen"], &[]));
    let run = run_doc(
        &mut vcw,
        &request("esc-none", 3, 3, VerificationLevel::SampleVerified),
        &eligibility(&[]),
        &DocVerifier::new(),
        &mut Stub::new(|_, n| doc(&format!("n{n}"), &[])),
    );
    assert!(matches!(run.outcome, VcwOutcome::Escalated { .. }));
}

/// A scorer that is honest once and then names a Pod outside the admissible set.
struct Rogue(Cell<u32>);

impl AdmissibleScorer for Rogue {
    fn bound_exploration(&self) -> Option<ptr_router::Exploration> {
        None
    }

    fn weights(&self, ctx: &ScoringContext<'_>) -> Result<Vec<(PodId, f64)>, RouteError> {
        self.0.set(self.0.get() + 1);
        if self.0.get() == 1 {
            Ok(ctx.admissible.iter().map(|p| (p.clone(), 1.0)).collect())
        } else {
            Ok(vec![(pod("ghost"), 1.0)])
        }
    }
}

#[test]
fn a_failed_request_still_accounts_what_it_spent_and_keeps_its_partial_trace() {
    let mut vcw = VcwLoop::new(parts_with(
        base_config(),
        roster(&["gen"], &[("fix_a", &["a"])]),
        RouterConfig::default(),
        Rogue(Cell::new(0)),
        7,
        None,
    ))
    .unwrap();
    let verifier = DocVerifier::new();
    let mut stub = Stub::new(|_, n| doc(&format!("d{n}"), &["a"])).fixing("fix_a", &["a"]);
    let result = vcw.run(
        &request("fail", 4, 4, VerificationLevel::SampleVerified),
        &eligibility(&["gen", "fix_a"]),
        &verifier,
        &mut stub,
    );
    assert!(matches!(result, Err(VcwError::Route(_))));
    assert_eq!(vcw.counters().verifier_calls, 1);
    assert_eq!(vcw.counters().pod_calls, 1);
    let partial = vcw.failed_trace().expect("a partial trace");
    assert_eq!(partial.outcome(), OutcomeTag::Failed);
    assert_eq!(partial.steps().len(), 1);
    assert_eq!(partial.steps()[0].decision, StepDecision::RoutingFailed);
    assert_eq!(partial.spent().verifier_calls, 1);
    assert_eq!(&partial.compute_digest(), partial.digest());
}

#[test]
fn a_workspace_refusal_for_a_non_authority_reason_neither_blocks_acceptance_nor_is_hidden() {
    // A payload whose digest is zero cannot become a workspace proposal. The verifier still
    // passes it, so it is accepted; the workspace outcome is visible in the answer and trace.
    let mut vcw = plain(roster(&["gen"], &[]));
    let verifier = DocVerifier::new();
    let mut stub = Stub::new(|_, _| doc("zero", &[]));
    let run = run_doc(
        &mut vcw,
        &request("no-slot", 2, 2, VerificationLevel::SampleVerified),
        &eligibility(&["gen"]),
        &verifier,
        &mut stub,
    );
    let answer = verified(&run);
    assert_eq!(
        answer.workspace(),
        WorkspaceOutcome::NotSubmitted("invalid-proposal")
    );
    assert_eq!(
        run.trace.candidates().next().unwrap().workspace,
        WorkspaceOutcome::NotSubmitted("invalid-proposal")
    );
    assert_eq!(vcw.counters().workspace_not_submitted, 1);
    assert_eq!(run.trace.steps()[0].decision, StepDecision::Accepted);
}

#[test]
fn a_controlling_schema_that_prefers_a_pod_never_routes_it_once_it_is_ineligible() {
    let applied = schema();
    let mut vcw = controlled(&applied, ControlArm::Full, HintMode::ApplyToSchema);
    let (_, hints) = run_series(&mut vcw, 160);
    assert!(hints > 0);
    assert!(applied.report().preferences > 0);
    // `good` is revoked; the schema still holds its preference.
    let view = eligibility(&["gen", "bad"]);
    let mut used_bad = 0;
    for i in 0..40 {
        let verifier = DocVerifier::new();
        let mut stub = Stub::new(|_, n| doc(&format!("r{n}"), &["a"]))
            .fixing("good", &["a"])
            .fixing("bad", &[]);
        let run = run_doc(
            &mut vcw,
            &request(
                &format!("after-{i}"),
                4,
                4,
                VerificationLevel::SampleVerified,
            ),
            &view,
            &verifier,
            &mut stub,
        );
        for record in run.trace.routes() {
            assert_ne!(record.pod(), &pod("good"));
            assert!(record.support().iter().all(|(p, _)| p != &pod("good")));
            used_bad += usize::from(record.pod() == &pod("bad"));
        }
    }
    assert!(
        used_bad > 0,
        "the control is vacuous without routing to bad"
    );
}

// ---- eligibility against the lifecycle authority and the expert projection ----

#[test]
fn a_base_pod_is_eligible_only_while_the_lifecycle_authority_reports_its_generation_live() {
    let mut rt = PtrRuntime::new(PtrConfig::default()).unwrap();
    prime(&mut rt);
    let project = project();
    let mut pods = PodRegistry::default();
    for name in [
        "live",
        "unknown",
        "ahead",
        "revoked",
        "superseded",
        "unregistered",
    ] {
        if name != "unregistered" {
            pods.register_new(stub_pod(name)).unwrap();
        }
    }
    for name in ["live", "ahead", "revoked", "superseded", "unregistered"] {
        make_live(&mut rt, &project, name, 1);
    }
    rt.commit(LedgerEvent::Revoked {
        subject: "revoked".to_owned(),
        generation: Generation(1),
    })
    .unwrap();
    rt.commit(LedgerEvent::CapsuleSuperseded {
        capsule: ptr_types::CapsuleId::from("superseded"),
        old: Generation(1),
        new: Generation(2),
    })
    .unwrap();
    let standing = RuntimeStanding::new(&rt, &pods, &project)
        .with_base_pod(pod("live"), Generation(1))
        .with_base_pod(pod("unknown"), Generation(1))
        .with_base_pod(pod("ahead"), Generation(2))
        .with_base_pod(pod("revoked"), Generation(1))
        .with_base_pod(pod("superseded"), Generation(1))
        .with_base_pod(pod("unregistered"), Generation(1));
    assert_eq!(
        standing.current_generation(&pod("live")),
        Some(Generation(1))
    );
    // Not known to the authority, ahead of it, revoked, superseded, not in the Pod registry,
    // never declared: all refused.
    for name in [
        "unknown",
        "ahead",
        "revoked",
        "superseded",
        "unregistered",
        "nobody",
    ] {
        assert_eq!(standing.current_generation(&pod(name)), None, "{name}");
    }
}

fn admit_op(rt: &mut PtrRuntime, op: StructuralOp) {
    let proposal = ExpertProposal::new(
        project(),
        proposer(),
        CommitIndex(1),
        op,
        RationaleDigest::new([7; 32]).unwrap(),
    )
    .unwrap();
    rt.admit_expert_proposal(&proposal, &gate_report(&proposal, passing()), &grant())
        .unwrap();
}

fn birth_token(rt: &mut PtrRuntime, name: &str) {
    let spec = ExpertPodSpec::new(
        project(),
        pod(name),
        ManifestDigest::new([name.as_bytes()[0]; 32]).unwrap(),
        ExpertLevel::Token {
            layer: 3,
            index: u16::from(name.as_bytes()[0]),
        },
        base(),
        Generation(1),
    )
    .unwrap();
    admit_op(rt, StructuralOp::Birth { result: spec });
}

fn target_of(rt: &PtrRuntime, name: &str) -> ExpertTarget {
    let entry = rt
        .expert_registry()
        .get(&project(), &pod(name))
        .unwrap()
        .clone();
    ExpertTarget::new(ExpertTargetInit {
        project: project(),
        pod: pod(name),
        generation: entry.generation,
        level: entry.level,
        base_model: entry.base_model,
        manifest_digest: entry.manifest_digest,
        state: entry.state,
        state_version: entry.state_version,
    })
    .unwrap()
}

fn fence(rt: &mut PtrRuntime, name: &str) {
    let version = rt
        .expert_registry()
        .get(&project(), &pod(name))
        .unwrap()
        .state_version;
    rt.fence_expert(
        &ExpertFenceRequest {
            project: project(),
            pod: pod(name),
            generation: Generation(1),
            expected_state_version: version,
            cause: ExpertFenceCause::AuthorityQuarantine,
            revocation: None,
        },
        &ExpertFenceGrant::revocation(project(), binding()),
    )
    .unwrap();
}

#[test]
fn an_expert_is_eligible_only_in_a_routable_state_and_roster_removal_follows_the_projection() {
    let mut rt = PtrRuntime::new(PtrConfig::default()).unwrap();
    prime(&mut rt);
    let mut pods = PodRegistry::default();
    for name in ["a", "b", "c", "d", "e"] {
        birth_token(&mut rt, name);
        if name != "e" {
            pods.register_new(stub_pod(name)).unwrap();
        }
    }
    let project = project();
    let view = |rt: &PtrRuntime, pods: &PodRegistry, name: &str| {
        RuntimeStanding::new(rt, pods, &project).current_generation(&pod(name))
    };
    for name in ["a", "b", "c", "d"] {
        assert_eq!(
            view(&rt, &pods, name),
            Some(Generation(1)),
            "{name} serving"
        );
    }
    // Committed and Serving but with no executable Pod in the registry: not eligible.
    assert_eq!(view(&rt, &pods, "e"), None);

    // Frozen is routable.
    let frozen = target_of(&rt, "b");
    admit_op(&mut rt, StructuralOp::Freeze { target: frozen });
    assert_eq!(
        rt.expert_registry().get(&project, &pod("b")).unwrap().state,
        ExpertState::Frozen
    );
    assert_eq!(view(&rt, &pods, "b"), Some(Generation(1)));
    // Fenced and Retired are not.
    fence(&mut rt, "c");
    assert_eq!(
        rt.expert_registry().get(&project, &pod("c")).unwrap().state,
        ExpertState::Fenced
    );
    assert_eq!(view(&rt, &pods, "c"), None);
    let pruned = target_of(&rt, "d");
    admit_op(
        &mut rt,
        StructuralOp::Prune {
            targets: vec![pruned],
        },
    );
    assert!(matches!(
        rt.expert_registry().get(&project, &pod("d")).unwrap().state,
        ExpertState::Retired(_)
    ));
    assert_eq!(view(&rt, &pods, "d"), None);
    // A lifecycle revocation of a Serving expert's generation also withdraws it.
    rt.commit(LedgerEvent::Revoked {
        subject: "expert:proj/a".to_owned(),
        generation: Generation(1),
    })
    .unwrap();
    assert_eq!(view(&rt, &pods, "a"), None);

    // Roster removal follows the projection: Retired goes, Fenced stays (it can be readmitted).
    let mut vcw = VcwLoop::new(parts_with(
        base_config(),
        roster(&["gen"], &[("c", &["y"]), ("d", &["z"]), ("b", &["w"])]),
        RouterConfig::default(),
        UniformScorer,
        7,
        None,
    ))
    .unwrap();
    let removed = vcw
        .retire_dead_experts(&project, rt.expert_registry())
        .unwrap();
    assert_eq!(removed, vec![pod("d")]);
    assert!(!vcw.repairers().contains_key(&pod("d")));
    assert!(vcw.repairers().contains_key(&pod("c")));
    assert!(vcw.router().registry().repair_pods(&code("z")).is_empty());
    assert_eq!(
        vcw.router().registry().repair_pods(&code("y")),
        vec![pod("c")]
    );
}

/// Everything up to a committed Birth whose expert the host has not yet enrolled.
fn surfaced_birth(
    config: impl FnOnce(&mut VcwConfig),
) -> (
    PtrRuntime,
    PodRegistry,
    VcwLoop,
    ptr_runtime::expert::StructureProposal,
) {
    let rt = live_runtime();
    let mut pods = PodRegistry::default();
    pods.register_new(stub_pod("gen")).unwrap();
    pods.register_new(stub_pod("fixer_a")).unwrap();
    let mut cfg = base_config();
    cfg.window_requests = 2;
    config(&mut cfg);
    let mut parts = parts_with(
        cfg,
        roster(&["gen"], &[("fixer_a", &["a"])]),
        RouterConfig::default(),
        UniformScorer,
        7,
        None,
    );
    parts.structure = StructurePolicy::new(just_birth()).unwrap();
    let mut vcw = VcwLoop::new(parts).unwrap();
    for i in 0..2 {
        let mut stub = Stub::new(|_, n| doc(&format!("n{n}"), &["newcode"]));
        run_in_runtime(&mut vcw, &rt, &pods, &format!("sb-{i}"), &mut stub);
    }
    let report = vcw.structure_step(birth_host(&rt)).unwrap();
    (rt, pods, vcw, report.proposals()[0].clone())
}

fn birth_host(rt: &PtrRuntime) -> StructureHost<'_> {
    StructureHost {
        project: project(),
        proposer: proposer(),
        observed_at: CommitIndex(1),
        registry: rt.expert_registry(),
        birth: Some(BirthContext {
            base_model: base(),
            site: BirthSite::Segment,
        }),
        aimer: BTreeMap::new(),
        overlaps: Vec::new(),
    }
}

#[test]
fn a_committed_expert_that_was_fenced_before_enrolment_is_not_enrolled() {
    let (mut rt, _pods, mut vcw, request) = surfaced_birth(|_| {});
    let born = request.request().op().results()[0].pod().clone();
    let birth = request.complete(trained(0x42)).unwrap();
    rt.admit_expert_proposal(&birth, &gate_report(&birth, passing()), &grant())
        .unwrap();
    fence(&mut rt, &born.0);
    assert!(matches!(
        vcw.sync_roster(&project(), rt.expert_registry(), &birth),
        Err(VcwError::NotCommitted { .. })
    ));
    assert!(!vcw.repairers().contains_key(&born));
    // The request stays outstanding; the host can still withdraw it.
    assert_eq!(vcw.outstanding_requests(), 1);
    assert!(vcw.withdraw_request(&birth));
    assert_eq!(vcw.outstanding_requests(), 0);
}

#[test]
fn an_unresolved_request_expires_after_the_configured_number_of_windows() {
    let (rt, pods, mut vcw, _request) = surfaced_birth(|c| c.outstanding_ttl_windows = 2);
    assert_eq!(vcw.outstanding_requests(), 1);
    // The same Birth is held back while it is outstanding.
    assert!(vcw
        .structure_step(birth_host(&rt))
        .unwrap()
        .proposals()
        .is_empty());
    // Two more windows pass without the host resolving the request.
    for i in 0..4 {
        let mut stub = Stub::new(|_, n| doc(&format!("m{n}"), &["newcode"]));
        run_in_runtime(&mut vcw, &rt, &pods, &format!("ttl-{i}"), &mut stub);
    }
    let again = vcw.structure_step(birth_host(&rt)).unwrap();
    assert_eq!(vcw.counters().outstanding_expired, 1);
    // The hold is lifted, so the policy proposes the Birth again, and it is tracked afresh.
    assert_eq!(again.proposals().len(), 1);
    assert_eq!(vcw.outstanding_requests(), 1);
    let refused = VcwLoop::new(parts_with(
        {
            let mut c = base_config();
            c.outstanding_ttl_windows = 0;
            c
        },
        roster(&["gen"], &[]),
        RouterConfig::default(),
        UniformScorer,
        7,
        None,
    ));
    assert!(matches!(refused, Err(VcwError::Config(_))));
}

// ---- the bridge ----

mod bridge {
    use super::*;
    use ptr_pods::{ExecutionManifest, LineageBinding, PodOutput, PodOutputKind};
    use ptr_runtime::{
        ExecutionScope, FencingToken, PlacementEpoch, PodOutputAdmission,
        PodOutputAdmissionRequest, ScopeState,
    };
    use ptr_types::{
        PrincipalId, ProvenanceRef, Revision, ScopeId, ScopeLeaseBinding, SessionId, Timestamp,
    };

    /// Passes the bytes `observation` and nothing else.
    struct Observation;

    impl Verifier<TypedPayload> for Observation {
        fn verify(&self, candidate: &TypedPayload) -> VerificationReport {
            if candidate.bytes == b"observation" {
                report(
                    VerificationStatus::Pass,
                    VerificationLevel::Deterministic,
                    &[],
                )
            } else {
                report(
                    VerificationStatus::Fail,
                    VerificationLevel::Deterministic,
                    &[("bad", true)],
                )
            }
        }
    }

    /// Rejects everything: the runtime's own verifier disagreeing with the loop's.
    struct Reject;

    impl Verifier<TypedPayload> for Reject {
        fn verify(&self, _: &TypedPayload) -> VerificationReport {
            report(
                VerificationStatus::Fail,
                VerificationLevel::Deterministic,
                &[("bad", true)],
            )
        }
    }

    struct Producer;

    impl RolePods<TypedPayload> for Producer {
        fn propose(&mut self, _: ProposeCall<'_>) -> Option<TypedPayload> {
            Some(TypedPayload {
                type_id: TypeId::from("text"),
                bytes: b"observation".to_vec(),
            })
        }

        fn repair(&mut self, _: RepairCall<'_, TypedPayload>) -> Option<TypedPayload> {
            None
        }
    }

    fn manifest() -> ExecutionManifest {
        ExecutionManifest::build(
            Generation(1),
            vec![LineageBinding {
                key: "knowledge:fact".into(),
                generation: Generation(1),
                digest: [3; 32],
            }],
            vec![LineageBinding {
                key: "artifact:pod".into(),
                generation: Generation(1),
                digest: [2; 32],
            }],
            vec!["raw:test".into()],
            None,
            Revision(1),
            [4; 32],
            PrincipalId::from("principal"),
            Revision(1),
        )
        .unwrap()
    }

    /// A runtime with a scope in project `project`, where the base Pod `pod` is live at
    /// generation 1, and a Pod registry holding it.
    fn fixture() -> (PtrRuntime, PodRegistry) {
        let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
        runtime
            .ingest_text(RequestId::from("request"), "ground truth")
            .unwrap();
        runtime
            .create_scope(
                ExecutionScope {
                    id: ScopeId::from("scope"),
                    parent: None,
                    session: SessionId::from("session"),
                    project: ProjectId::from("project"),
                    created_at: Timestamp(1),
                    deadline: None,
                    state: ScopeState::Created,
                    cancellation_requested: false,
                },
                ScopeLeaseBinding::default(),
            )
            .unwrap();
        make_live(&mut runtime, &ProjectId::from("project"), "pod", 1);
        let mut pods = PodRegistry::default();
        pods.register_new(stub_pod_in("project", "pod")).unwrap();
        (runtime, pods)
    }

    fn base_pods() -> Vec<(PodId, Generation)> {
        vec![(pod("pod"), Generation(1))]
    }

    fn admission_request(bytes: &[u8], pod_name: &str, id: &str) -> PodOutputAdmissionRequest {
        let manifest = manifest();
        let manifest_digest = manifest.manifest_digest;
        PodOutputAdmissionRequest {
            request_id: RequestId::from(id),
            manifest_digest: manifest.manifest_digest,
            manifest,
            pod_id: PodId::from(pod_name),
            output: PodOutput {
                kind: PodOutputKind::Observation,
                payload: TypedPayload {
                    type_id: TypeId::from("text"),
                    bytes: bytes.to_vec(),
                },
                generation: Generation(1),
                manifest_digest,
                artifact_digest: [2; 32],
                provenance: vec![ProvenanceRef {
                    source: ptr_types::EvidenceId::from("evidence"),
                    note: None,
                }],
                dependencies: vec![[3; 32]],
                revision: Revision(1),
                verified: false,
            },
            expected_type: TypeId::from("text"),
            session_id: SessionId::from("session"),
            scope_id: ScopeId::from("scope"),
            placement_epoch: PlacementEpoch(1),
            fencing_token: FencingToken(1),
            kv_taint: ptr_model_api::KvReuseTaint::Exact,
        }
    }

    fn answer_in(project_name: &str) -> ptr_runtime::vcw::VerifiedAnswer<TypedPayload> {
        let mut vcw = plain(roster(&["pod"], &[]));
        let req = VcwRequest {
            request_id: RequestId::from("request"),
            project: ProjectId::from(project_name),
            request_digest: digest_of("request"),
            required: VerificationLevel::SampleVerified,
            budget: RequestBudget::new(2, 2).unwrap(),
        };
        let run = vcw
            .run(&req, &eligibility(&["pod"]), &Observation, &mut Producer)
            .unwrap();
        match run.outcome {
            VcwOutcome::Verified(answer) => answer,
            other => panic!("expected Verified, got {other:?}"),
        }
    }

    fn answer() -> ptr_runtime::vcw::VerifiedAnswer<TypedPayload> {
        answer_in("project")
    }

    fn admit(
        rt: &mut PtrRuntime,
        pods: &PodRegistry,
        answer: ptr_runtime::vcw::VerifiedAnswer<TypedPayload>,
        request: PodOutputAdmissionRequest,
        verifier: &dyn Verifier<TypedPayload>,
    ) -> Result<PodOutputAdmission, VcwAdmitError> {
        admit_verified_answer(rt, pods, &base_pods(), answer, request, verifier)
    }

    #[test]
    fn an_accepted_answer_reaches_the_runtime_only_through_output_admission() {
        let (mut rt, pods) = fixture();
        let events = rt.committed_events().len();
        // The loop's acceptance appended nothing: it is provisional.
        let _ = answer();
        assert_eq!(rt.committed_events().len(), events);

        let mut wrong_type = admission_request(b"observation", "pod", "request");
        wrong_type.output.payload.type_id = TypeId::from("other");
        let mut stale = admission_request(b"observation", "pod", "request");
        stale.output.generation = Generation(2);
        let cases: Vec<(PodOutputAdmissionRequest, &str)> = vec![
            (admission_request(b"other", "pod", "request"), "payload"),
            (wrong_type, "payload"),
            (
                admission_request(b"observation", "intruder", "request"),
                "pod",
            ),
            (
                admission_request(b"observation", "pod", "elsewhere"),
                "request",
            ),
            (stale, "generation"),
        ];
        for (request, what) in cases {
            let result = admit(&mut rt, &pods, answer(), request, &Observation);
            let matched = matches!(
                (&result, what),
                (Err(VcwAdmitError::PayloadMismatch), "payload")
                    | (Err(VcwAdmitError::PodMismatch), "pod")
                    | (Err(VcwAdmitError::RequestMismatch), "request")
                    | (Err(VcwAdmitError::GenerationMismatch), "generation")
            );
            assert!(matched, "{what}: {result:?}");
            assert_eq!(rt.committed_events().len(), events, "{what}");
        }
        // An answer accepted for another project is not admitted into this scope.
        let result = admit(
            &mut rt,
            &pods,
            answer_in("elsewhere"),
            admission_request(b"observation", "pod", "request"),
            &Observation,
        );
        assert!(matches!(result, Err(VcwAdmitError::ProjectMismatch)));
        assert_eq!(rt.committed_events().len(), events);

        // The runtime still decides with its own, independent verifier.
        let refused = admit(
            &mut rt,
            &pods,
            answer(),
            admission_request(b"observation", "pod", "request"),
            &Reject,
        );
        assert!(matches!(
            refused,
            Err(VcwAdmitError::Runtime(
                RuntimeError::PodVerificationFailed { .. }
            ))
        ));
        assert_eq!(rt.committed_events().len(), events);

        let admitted = admit(
            &mut rt,
            &pods,
            answer(),
            admission_request(b"observation", "pod", "request"),
            &Observation,
        )
        .unwrap();
        assert!(matches!(
            admitted,
            PodOutputAdmission::ObservationCandidate { .. }
        ));
        assert!(rt.committed_events().len() > events);
    }

    #[test]
    fn a_pod_revoked_or_superseded_after_the_run_is_refused_at_the_bridge() {
        // Revoked.
        let (mut rt, pods) = fixture();
        let held = answer();
        rt.commit(LedgerEvent::Revoked {
            subject: "pod".to_owned(),
            generation: Generation(1),
        })
        .unwrap();
        let events = rt.committed_events().len();
        let result = admit(
            &mut rt,
            &pods,
            held,
            admission_request(b"observation", "pod", "request"),
            &Observation,
        );
        assert!(matches!(
            result,
            Err(VcwAdmitError::AuthorNoLongerEligible { .. })
        ));
        assert_eq!(rt.committed_events().len(), events);

        // Superseded by a newer generation.
        let (mut rt, pods) = fixture();
        let held = answer();
        rt.commit(LedgerEvent::CapsuleSuperseded {
            capsule: ptr_types::CapsuleId::from("pod"),
            old: Generation(1),
            new: Generation(2),
        })
        .unwrap();
        let events = rt.committed_events().len();
        let result = admit(
            &mut rt,
            &pods,
            held,
            admission_request(b"observation", "pod", "request"),
            &Observation,
        );
        assert!(matches!(
            result,
            Err(VcwAdmitError::AuthorNoLongerEligible { .. })
        ));
        assert_eq!(rt.committed_events().len(), events);

        // Withdrawn from the Pod registry.
        let (mut rt, _) = fixture();
        let empty = PodRegistry::default();
        let events = rt.committed_events().len();
        let result = admit(
            &mut rt,
            &empty,
            answer(),
            admission_request(b"observation", "pod", "request"),
            &Observation,
        );
        assert!(matches!(
            result,
            Err(VcwAdmitError::AuthorNoLongerEligible { .. })
        ));
        assert_eq!(rt.committed_events().len(), events);
    }
}

#[test]
fn a_pod_that_moved_to_a_new_generation_mid_request_is_stale_for_the_workspace_and_acceptance() {
    let mut vcw = plain(roster(&["p1", "p2"], &[]));
    let view = LiveView::with(&["p1", "p2"]);
    let verifier = DocVerifier::new();
    let mut stub = Stub::new(|_, n| doc(&format!("g{n}"), &[]));
    let first = Rc::new(RefCell::new(None::<String>));
    {
        let view = view.clone();
        let first = Rc::clone(&first);
        stub.on_propose = Some(Box::new(move |who| {
            // The first Pod asked is replaced by its next generation while it works: generation
            // 1 is stale, the Pod id is still listed.
            if first.borrow().is_none() {
                *first.borrow_mut() = Some(who.to_owned());
                view.0.borrow_mut().insert(pod(who), Generation(2));
            }
        }));
    }
    let run = run_doc(
        &mut vcw,
        &request("stale-gen", 4, 4, VerificationLevel::SampleVerified),
        &view,
        &verifier,
        &mut stub,
    );
    let step0 = &run.trace.steps()[0];
    assert_eq!(step0.decision, StepDecision::AuthorNoLongerEligible);
    assert_eq!(
        step0.candidate.as_ref().unwrap().workspace,
        WorkspaceOutcome::Refused(RefusalKind::AuthorRevoked)
    );
    // Whatever the request ends with, generation 1 of that Pod is not in an answer.
    if let VcwOutcome::Verified(answer) = &run.outcome {
        let revoked = first.borrow().clone().unwrap();
        assert!(answer
            .authors()
            .iter()
            .all(|p| !(p.pod.0 == revoked && p.generation == Generation(1))));
    }
}

#[test]
fn an_expired_request_whose_expert_was_committed_is_enrolled_instead_of_dropped() {
    let (mut rt, pods, mut vcw, request) = surfaced_birth(|c| c.outstanding_ttl_windows = 2);
    let born = request.request().op().results()[0].pod().clone();
    // The host commits the Birth and forgets to sync the roster.
    let birth = request.complete(trained(0x42)).unwrap();
    rt.admit_expert_proposal(&birth, &gate_report(&birth, passing()), &grant())
        .unwrap();
    assert!(!vcw.repairers().contains_key(&born));
    for i in 0..4 {
        let mut stub = Stub::new(|_, n| doc(&format!("m{n}"), &["newcode"]));
        run_in_runtime(&mut vcw, &rt, &pods, &format!("reconcile-{i}"), &mut stub);
    }
    vcw.structure_step(birth_host(&rt)).unwrap();
    // The entry outlived its windows, but the expert is committed: enrolled, not stranded.
    assert_eq!(vcw.counters().outstanding_reconciled, 1);
    assert_eq!(vcw.counters().outstanding_expired, 0);
    assert!(vcw.repairers().contains_key(&born));
    assert_eq!(
        vcw.router().registry().repair_pods(&code("newcode")),
        vec![born]
    );
    // The request is resolved: a late sync is an unknown operation, not a second enrolment.
    assert!(matches!(
        vcw.sync_roster(&project(), rt.expert_registry(), &birth),
        Err(VcwError::UnknownOperation)
    ));
}
