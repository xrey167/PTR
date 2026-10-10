//! The verifier-closed loop over the real deterministic task verifiers of `ptr-verifier`
//! (arithmetic, JSON Schema, code tests, facts), with programmatic stub role Pods.
//!
//! For each verifier: a wrong candidate gets a finding, the finding code is routed to the repair
//! role the verifier's code table names (the roster is built from that table by
//! `RepairRoleAssignments`), the repaired candidate passes and is accepted. An unrepairable task
//! ends `Exhausted`; an `Unknown` fact is never repaired or accepted and ends in a fresh
//! proposal or an escalation. Nothing here scores or hints: acceptance is the verifier's.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, VecDeque};
use std::rc::Rc;

use ptr_router::{
    EligibilitySnapshot, EscalationReason, FallbackReason, NonceSequence, PodEligibility, PodRef,
    RepairRoleAssignments, RouteClass, RouterConfig, SamplingKey, SelfModel, SelfModelLimits,
    UniformScorer,
};
use ptr_runtime::expert::{StructurePolicy, StructurePolicyConfig};
use ptr_runtime::vcw::{
    ClassSummary, EscalateCall, ProposeCall, RepairCall, RequestBudget, RolePods, Roster,
    StepDecision, VcwConfig, VcwLoop, VcwOutcome, VcwParts, VcwPayload, VcwRequest, VcwRun,
};
use ptr_semdb::{SemanticDelta, SemanticHost};
use ptr_types::{Digest, Generation, PodId, ProjectId, RequestId, TypeId, VerificationLevel};
use ptr_verifier::{
    fact_key, ArithmeticVerifier, CodeTask, CodeTest, CodeTestVerifier, FactClaim, FactVerifier,
    JsonSchemaVerifier, RepairRole, VerificationReport, Verifier,
};
use sha2::{Digest as _, Sha256};

// ---------------------------------------------------------------------------------------------
// Payloads and verifier adapters (the orphan rule keeps `VcwPayload` impls in this crate)
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Eq, PartialEq)]
struct Text(String);

impl VcwPayload for Text {
    fn payload_type(&self) -> TypeId {
        TypeId::from("text")
    }
    fn payload_digest(&self) -> Digest {
        Sha256::digest(self.0.as_bytes()).into()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Claims(Vec<FactClaim>);

impl VcwPayload for Claims {
    fn payload_type(&self) -> TypeId {
        TypeId::from("claims")
    }
    fn payload_digest(&self) -> Digest {
        let mut hasher = Sha256::new();
        for claim in &self.0 {
            for field in [&claim.subject, &claim.attribute, &claim.value] {
                hasher.update((field.len() as u64).to_le_bytes());
                hasher.update(field.as_bytes());
            }
        }
        hasher.finalize().into()
    }
}

/// Counts verifier calls, records every finding code (hard and soft) of each report, and
/// delegates to a string verifier.
struct OnText<V> {
    inner: V,
    calls: Cell<u32>,
    seen: RefCell<Vec<Vec<String>>>,
}

impl<V: Verifier<String>> Verifier<Text> for OnText<V> {
    fn verify(&self, candidate: &Text) -> VerificationReport {
        self.calls.set(self.calls.get() + 1);
        let report = self.inner.verify(&candidate.0);
        self.seen.borrow_mut().push(codes_of(&report));
        report
    }
}

fn codes_of(report: &VerificationReport) -> Vec<String> {
    report.findings.iter().map(|f| f.code.clone()).collect()
}

fn on_text<V: Verifier<String>>(inner: V) -> OnText<V> {
    OnText {
        inner,
        calls: Cell::new(0),
        seen: RefCell::new(Vec::new()),
    }
}

struct OnClaims {
    inner: FactVerifier,
    seen: RefCell<Vec<Vec<String>>>,
}

impl Verifier<Claims> for OnClaims {
    fn verify(&self, candidate: &Claims) -> VerificationReport {
        let report = self.inner.verify(&candidate.0);
        self.seen.borrow_mut().push(codes_of(&report));
        report
    }
}

fn on_claims(inner: FactVerifier) -> OnClaims {
    OnClaims {
        inner,
        seen: RefCell::new(Vec::new()),
    }
}

// ---------------------------------------------------------------------------------------------
// Scripted role Pods
// ---------------------------------------------------------------------------------------------

type Repair<P> = Box<dyn Fn(&P) -> Option<P>>;

/// Proposals come from a queue; a repair Pod is a function of the previous candidate.
struct Script<P> {
    proposals: VecDeque<P>,
    repairs: BTreeMap<String, Repair<P>>,
    escalation: Option<(PodRef, P)>,
    on_propose: Option<Box<dyn FnMut()>>,
    log: RefCell<Vec<String>>,
}

impl<P> Script<P> {
    fn new(proposals: Vec<P>) -> Self {
        Self {
            proposals: proposals.into(),
            repairs: BTreeMap::new(),
            escalation: None,
            on_propose: None,
            log: RefCell::new(Vec::new()),
        }
    }

    fn repairing(mut self, role: RepairRole, fix: impl Fn(&P) -> Option<P> + 'static) -> Self {
        self.repairs.insert(role.as_str().to_owned(), Box::new(fix));
        self
    }
}

impl<P: VcwPayload> RolePods<P> for Script<P> {
    fn propose(&mut self, call: ProposeCall<'_>) -> Option<P> {
        self.log
            .borrow_mut()
            .push(format!("propose:{}", call.pod.pod.0));
        if let Some(hook) = self.on_propose.as_mut() {
            hook();
        }
        self.proposals.pop_front()
    }

    fn repair(&mut self, call: RepairCall<'_, P>) -> Option<P> {
        self.log
            .borrow_mut()
            .push(format!("repair:{}:{}", call.pod.pod.0, call.code.as_str()));
        self.repairs.get(&call.pod.pod.0)?(call.previous)
    }

    fn escalate(&mut self, call: EscalateCall<'_, P>) -> Option<(PodRef, P)> {
        self.log
            .borrow_mut()
            .push(format!("escalate:{:?}", call.reason));
        self.escalation.take()
    }
}

// ---------------------------------------------------------------------------------------------
// Loop construction
// ---------------------------------------------------------------------------------------------

fn pod(name: &str) -> PodId {
    PodId::from(name)
}

/// A pod per repair role, named by the role, plus one proposer `gen`; the roster comes from the
/// verifier crate's code table through the router's constructor.
fn role_assignments() -> RepairRoleAssignments {
    RepairRole::ALL
        .iter()
        .fold(RepairRoleAssignments::new(), |assignments, role| {
            assignments.assign(*role, pod(role.as_str()))
        })
}

fn new_loop(assignments: &RepairRoleAssignments) -> VcwLoop {
    VcwLoop::new(VcwParts {
        config: VcwConfig::new("real-verifiers", [9; 32]),
        roster: Roster {
            proposers: vec![pod("gen")],
            repairers: assignments.repairers(),
        },
        router: RouterConfig::default(),
        scorer: UniformScorer,
        key: SamplingKey::from_bytes([5; 32]),
        nonces: NonceSequence::starting_at([1; 8], 0),
        self_model: SelfModel::new(SelfModelLimits::default()),
        schema: None,
        structure: StructurePolicy::new(StructurePolicyConfig::default()).unwrap(),
    })
    .unwrap()
}

fn everyone() -> EligibilitySnapshot {
    let mut view = EligibilitySnapshot::new();
    view.set(pod("gen"), Generation(1));
    view.set(pod("boss"), Generation(1));
    for role in RepairRole::ALL {
        view.set(pod(role.as_str()), Generation(1));
    }
    view
}

fn request(id: &str, calls: u32, rounds: u32) -> VcwRequest {
    VcwRequest {
        request_id: RequestId::from(id),
        project: ProjectId::from("proj"),
        request_digest: Sha256::digest(id.as_bytes()).into(),
        required: VerificationLevel::Deterministic,
        budget: RequestBudget::new(calls, rounds).unwrap(),
    }
}

fn text(value: &str) -> Text {
    Text(value.to_owned())
}

/// Every repair record of the run as `(code, pod)`, in order.
fn repair_routes(run: &VcwRun<impl Clone>) -> Vec<(String, String)> {
    run.trace
        .routes()
        .filter_map(|record| match record.class() {
            RouteClass::Repair(code) => Some((code.as_str().to_owned(), record.pod().0.clone())),
            RouteClass::Regenerate(_) => None,
        })
        .collect()
}

fn answer<P: VcwPayload + std::fmt::Debug>(run: &VcwRun<P>) -> &P {
    match &run.outcome {
        VcwOutcome::Verified(answer) => answer.payload(),
        other => panic!("expected Verified, got {other:?}"),
    }
}

fn route(code: &str, role: RepairRole) -> (String, String) {
    (code.to_owned(), role.as_str().to_owned())
}

/// The classification of every judged candidate, in order.
fn classes<P: Clone>(run: &VcwRun<P>) -> Vec<ClassSummary> {
    run.trace.candidates().map(|c| c.class.clone()).collect()
}

/// A script in which any repair Pod being asked is a test failure.
fn forbid_all_repairs<P: VcwPayload + 'static>(mut pods: Script<P>) -> Script<P> {
    for role in RepairRole::ALL {
        pods = pods.repairing(role, move |_| {
            panic!(
                "{} must not be asked: the finding is not a repair target",
                role.as_str()
            )
        });
    }
    pods
}

/// The run was neither verified nor repaired: every judged candidate was `Unknown` with exactly
/// `code` as its only finding (soft, so absent from the hard codes), classified as the
/// `Unknown` fallback, and the loop regenerated rather than repaired.
fn assert_unknown_regenerates<P: VcwPayload + std::fmt::Debug + Clone>(
    run: &VcwRun<P>,
    seen: &[Vec<String>],
    code: &str,
) {
    assert!(
        !matches!(run.outcome, VcwOutcome::Verified(_)),
        "{:?}",
        run.outcome
    );
    assert!(!seen.is_empty());
    assert!(
        seen.iter().all(|codes| codes.as_slice() == [code]),
        "{seen:?}"
    );
    assert!(repair_routes(run).is_empty());
    let classes = classes(run);
    assert_eq!(classes.len(), seen.len());
    assert!(
        classes
            .iter()
            .all(|c| *c == ClassSummary::Fallback(FallbackReason::Unknown)),
        "{classes:?}"
    );
    assert!(run
        .trace
        .candidates()
        .all(|c| c.report.hard_codes.is_empty() && c.report.soft_findings == 1));
    assert!(run
        .trace
        .steps()
        .iter()
        .any(|step| step.decision == StepDecision::Regenerate));
}

// ---------------------------------------------------------------------------------------------
// Table and roster
// ---------------------------------------------------------------------------------------------

#[test]
fn the_roster_registers_each_hard_code_of_the_table_for_its_role_only() {
    let roster = role_assignments().repairers();
    for spec in ptr_verifier::CODE_TABLE {
        let holders: Vec<&str> = roster
            .iter()
            .filter(|(_, codes)| codes.iter().any(|c| c.as_str() == spec.code))
            .map(|(p, _)| p.0.as_str())
            .collect();
        match spec.role {
            Some(role) => assert_eq!(holders, [role.as_str()], "{}", spec.code),
            None => assert!(holders.is_empty(), "{} is soft", spec.code),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------------------------

#[test]
fn arithmetic_wrong_value_is_routed_to_value_repair_then_accepted() {
    let mut vcw = new_loop(&role_assignments());
    let verifier = on_text(ArithmeticVerifier::new("(3 + 4) * 5 - 6 / 3"));
    let mut pods = Script::new(vec![text("34")]).repairing(RepairRole::Value, |_| Some(text("33")));
    let run = vcw
        .run(
            &request("arith-value", 6, 4),
            &everyone(),
            &verifier,
            &mut pods,
        )
        .unwrap();
    assert_eq!(answer(&run), &text("33"));
    assert_eq!(
        repair_routes(&run),
        [route("arith.value", RepairRole::Value)]
    );
    assert_eq!(verifier.calls.get(), 2);
}

#[test]
fn an_out_of_bounds_arithmetic_candidate_is_routed_to_value_repair() {
    let mut vcw = new_loop(&role_assignments());
    let verifier = on_text(ArithmeticVerifier::new("(3 + 4) * 5 - 6 / 3"));
    let mut pods = Script::new(vec![text(&"9".repeat(40))])
        .repairing(RepairRole::Value, |_| Some(text("33")))
        .repairing(RepairRole::Format, |_| {
            panic!("format repair must not be asked for arith.bounds")
        });
    let run = vcw
        .run(
            &request("arith-bounds", 6, 4),
            &everyone(),
            &verifier,
            &mut pods,
        )
        .unwrap();
    assert_eq!(answer(&run), &text("33"));
    assert_eq!(
        repair_routes(&run),
        [route("arith.bounds", RepairRole::Value)]
    );
    assert_eq!(verifier.seen.borrow()[0], ["arith.bounds"]);
}

#[test]
fn arithmetic_format_error_is_routed_to_format_repair_not_value_repair() {
    let mut vcw = new_loop(&role_assignments());
    let verifier = on_text(ArithmeticVerifier::new("2 + 2"));
    let mut pods = Script::new(vec![text("four")])
        .repairing(RepairRole::Format, |_| Some(text("4")))
        .repairing(RepairRole::Value, |_| {
            panic!("value repair must not be asked")
        });
    let run = vcw
        .run(
            &request("arith-format", 6, 4),
            &everyone(),
            &verifier,
            &mut pods,
        )
        .unwrap();
    assert_eq!(answer(&run), &text("4"));
    assert_eq!(
        repair_routes(&run),
        [route("arith.format", RepairRole::Format)]
    );
}

#[test]
fn an_unrepairable_arithmetic_task_ends_exhausted_never_with_a_wrong_answer() {
    let mut vcw = new_loop(&role_assignments());
    let verifier = on_text(ArithmeticVerifier::new("6 * 7"));
    // The repair Pod keeps producing another wrong value.
    let wrong = Cell::new(100);
    let mut pods = Script::new(vec![text("41")]).repairing(RepairRole::Value, move |_| {
        wrong.set(wrong.get() + 1);
        Some(Text(wrong.get().to_string()))
    });
    let run = vcw
        .run(
            &request("arith-exhaust", 5, 3),
            &everyone(),
            &verifier,
            &mut pods,
        )
        .unwrap();
    assert!(
        matches!(run.outcome, VcwOutcome::Exhausted { .. }),
        "{:?}",
        run.outcome
    );
    assert!(verifier.calls.get() >= 2);
    // No candidate in the trace was accepted.
    assert!(run
        .trace
        .steps()
        .iter()
        .all(|step| step.decision != StepDecision::Accepted));
}

#[test]
fn an_undecidable_arithmetic_task_is_unknown_and_never_accepted() {
    let mut vcw = new_loop(&role_assignments());
    let verifier = on_text(ArithmeticVerifier::new("1 / 0"));
    let mut pods = forbid_all_repairs(Script::new(vec![text("1"), text("2"), text("3")]));
    let run = vcw
        .run(
            &request("arith-unknown", 3, 3),
            &everyone(),
            &verifier,
            &mut pods,
        )
        .unwrap();
    // Unknown regenerates a fresh proposal (the policy), never a repair.
    assert_unknown_regenerates(&run, &verifier.seen.borrow(), "arith.task");
}

// ---------------------------------------------------------------------------------------------
// JSON Schema
// ---------------------------------------------------------------------------------------------

const SCHEMA: &str = r#"{
    "type": "object",
    "required": ["name", "age"],
    "additionalProperties": false,
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "age": {"type": "integer", "minimum": 0}
    }
}"#;

#[test]
fn json_missing_property_goes_to_schema_repair_and_a_parse_error_to_format_repair() {
    // A document missing a property: json.required -> schema-repair.
    let mut vcw = new_loop(&role_assignments());
    let verifier = on_text(JsonSchemaVerifier::new(SCHEMA));
    let mut pods = Script::new(vec![text(r#"{"name":"Ada"}"#)])
        .repairing(RepairRole::Schema, |_| {
            Some(text(r#"{"name":"Ada","age":36}"#))
        });
    let run = vcw
        .run(
            &request("json-required", 6, 4),
            &everyone(),
            &verifier,
            &mut pods,
        )
        .unwrap();
    assert_eq!(answer(&run), &text(r#"{"name":"Ada","age":36}"#));
    assert_eq!(
        repair_routes(&run),
        [route("json.required", RepairRole::Schema)]
    );

    // Not JSON at all: json.parse -> format-repair first, then the schema violation it leaves
    // goes to schema-repair, one code per round.
    let mut vcw = new_loop(&role_assignments());
    let verifier = on_text(JsonSchemaVerifier::new(SCHEMA));
    let mut pods = Script::new(vec![text(r#"{"name": "Ada", "age": "#)])
        .repairing(RepairRole::Format, |_| {
            Some(text(r#"{"name":"Ada","age":-1}"#))
        })
        .repairing(RepairRole::Schema, |_| {
            Some(text(r#"{"name":"Ada","age":1}"#))
        });
    let run = vcw
        .run(
            &request("json-parse", 8, 5),
            &everyone(),
            &verifier,
            &mut pods,
        )
        .unwrap();
    assert_eq!(answer(&run), &text(r#"{"name":"Ada","age":1}"#));
    assert_eq!(
        repair_routes(&run),
        [
            route("json.parse", RepairRole::Format),
            route("json.range", RepairRole::Schema)
        ]
    );
}

#[test]
fn json_with_two_violations_routes_each_code_to_the_schema_role_once() {
    let mut vcw = new_loop(&role_assignments());
    let verifier = on_text(JsonSchemaVerifier::new(SCHEMA));
    let mut pods = Script::new(vec![text(r#"{"name":"","age":1,"x":0}"#)]).repairing(
        RepairRole::Schema,
        |previous| {
            // The loop asks once per distinct code in one round, each call seeing the previous edit.
            let fixed = if previous.0.contains(r#""x""#) {
                r#"{"name":"","age":1}"#
            } else {
                r#"{"name":"Ada","age":1}"#
            };
            Some(text(fixed))
        },
    );
    let run = vcw
        .run(
            &request("json-two", 8, 5),
            &everyone(),
            &verifier,
            &mut pods,
        )
        .unwrap();
    assert_eq!(answer(&run), &text(r#"{"name":"Ada","age":1}"#));
    assert_eq!(
        repair_routes(&run),
        [
            route("json.length", RepairRole::Schema),
            route("json.extra", RepairRole::Schema),
        ]
    );
}

#[test]
fn an_unrepairable_document_ends_exhausted() {
    let mut vcw = new_loop(&role_assignments());
    let verifier = on_text(JsonSchemaVerifier::new(SCHEMA));
    let mut pods = Script::new(vec![text("[]")]).repairing(RepairRole::Schema, |_| None);
    let run = vcw
        .run(
            &request("json-exhaust", 4, 3),
            &everyone(),
            &verifier,
            &mut pods,
        )
        .unwrap();
    assert!(!matches!(run.outcome, VcwOutcome::Verified(_)));
    // The finding was a hard json.type, classified as a repair that nobody could perform.
    assert_eq!(verifier.seen.borrow()[0], ["json.type"]);
    let first = run.trace.candidates().next().unwrap();
    assert_eq!(first.report.hard_codes, ["json.type"]);
    assert!(matches!(&first.class, ClassSummary::Repair(codes) if codes.len() == 1));
    assert!(run
        .trace
        .steps()
        .iter()
        .all(|step| step.decision != StepDecision::Accepted));
}

#[test]
fn a_schema_outside_the_subset_is_unknown_and_regenerates_never_repairs() {
    let mut vcw = new_loop(&role_assignments());
    let verifier = on_text(JsonSchemaVerifier::new(r#"{"pattern":"a+"}"#));
    let mut pods = forbid_all_repairs(Script::new(vec![text("\"a\""), text("\"aa\""), text("1")]));
    let run = vcw
        .run(
            &request("json-schema-unknown", 3, 3),
            &everyone(),
            &verifier,
            &mut pods,
        )
        .unwrap();
    assert_unknown_regenerates(&run, &verifier.seen.borrow(), "json.schema");
}

// ---------------------------------------------------------------------------------------------
// Code tests
// ---------------------------------------------------------------------------------------------

fn square_task() -> CodeTask {
    CodeTask {
        entry: "f".into(),
        tests: vec![
            CodeTest::new("zero", &[0], 0),
            CodeTest::new("three", &[3], 9),
            CodeTest::new("neg", &[-4], 16),
        ],
    }
}

#[test]
fn code_with_a_failing_test_goes_to_code_repair_and_a_hang_to_resource_repair() {
    // A wrong program: code.test_failed -> code-repair.
    let mut vcw = new_loop(&role_assignments());
    let verifier = on_text(CodeTestVerifier::new(square_task()));
    let mut pods = Script::new(vec![text("fn f(n) { return n + n; }")])
        .repairing(RepairRole::Code, |_| {
            Some(text("fn f(n) { return n * n; }"))
        });
    let run = vcw
        .run(
            &request("code-fail", 6, 4),
            &everyone(),
            &verifier,
            &mut pods,
        )
        .unwrap();
    assert_eq!(answer(&run), &text("fn f(n) { return n * n; }"));
    assert_eq!(
        repair_routes(&run),
        [route("code.test_failed", RepairRole::Code)]
    );

    // An infinite loop: code.timeout -> resource-repair, never code-repair.
    let mut vcw = new_loop(&role_assignments());
    let verifier = on_text(CodeTestVerifier::new(square_task()));
    let mut pods = Script::new(vec![text("fn f(n) { while 1 { } return 0; }")])
        .repairing(RepairRole::Resource, |_| {
            Some(text("fn f(n) { return n * n; }"))
        })
        .repairing(RepairRole::Code, |_| {
            panic!("code repair must not be asked")
        });
    let run = vcw
        .run(
            &request("code-hang", 6, 4),
            &everyone(),
            &verifier,
            &mut pods,
        )
        .unwrap();
    assert_eq!(answer(&run), &text("fn f(n) { return n * n; }"));
    assert_eq!(
        repair_routes(&run),
        [route("code.timeout", RepairRole::Resource)]
    );
}

#[test]
fn code_syntax_error_goes_to_code_repair_and_a_hang_forever_ends_exhausted() {
    let mut vcw = new_loop(&role_assignments());
    let verifier = on_text(CodeTestVerifier::new(square_task()));
    let mut pods = Script::new(vec![text("fn f(n) { return n * n }")])
        .repairing(RepairRole::Code, |_| {
            Some(text("fn f(n) { return n * n; }"))
        });
    let run = vcw
        .run(
            &request("code-syntax", 6, 4),
            &everyone(),
            &verifier,
            &mut pods,
        )
        .unwrap();
    assert_eq!(
        repair_routes(&run),
        [route("code.syntax", RepairRole::Code)]
    );
    assert!(matches!(run.outcome, VcwOutcome::Verified(_)));

    // The repair Pod only ever returns programs that spin: bounded, then Exhausted.
    let mut vcw = new_loop(&role_assignments());
    let verifier = on_text(CodeTestVerifier::new(square_task()));
    let mut pods = Script::new(vec![text("fn f(n) { while 1 { } return 0; }")])
        .repairing(RepairRole::Resource, |_| {
            Some(text("fn f(n) { while 1 { } return 1; }"))
        });
    let run = vcw
        .run(
            &request("code-exhaust", 4, 3),
            &everyone(),
            &verifier,
            &mut pods,
        )
        .unwrap();
    assert!(
        matches!(run.outcome, VcwOutcome::Exhausted { .. }),
        "{:?}",
        run.outcome
    );
}

#[test]
fn a_blown_resource_bound_goes_to_resource_repair_not_code_repair() {
    let mut vcw = new_loop(&role_assignments());
    let verifier = on_text(CodeTestVerifier::new(square_task()));
    // Unbounded recursion trips the call-depth bound: code.resource.
    let mut pods = Script::new(vec![text("fn f(n) { return f(n + 1); }")])
        .repairing(RepairRole::Resource, |_| {
            Some(text("fn f(n) { return n * n; }"))
        })
        .repairing(RepairRole::Code, |_| {
            panic!("code repair must not be asked")
        });
    let run = vcw
        .run(
            &request("code-resource", 6, 4),
            &everyone(),
            &verifier,
            &mut pods,
        )
        .unwrap();
    assert_eq!(answer(&run), &text("fn f(n) { return n * n; }"));
    assert_eq!(
        repair_routes(&run),
        [route("code.resource", RepairRole::Resource)]
    );
    assert_eq!(verifier.seen.borrow()[0], ["code.resource"; 3]);
}

#[test]
fn a_code_task_without_tests_is_unknown_and_regenerates_never_repairs() {
    let mut vcw = new_loop(&role_assignments());
    let task = CodeTask {
        entry: "f".into(),
        tests: Vec::new(),
    };
    let verifier = on_text(CodeTestVerifier::new(task));
    let mut pods = forbid_all_repairs(Script::new(vec![
        text("fn f(n) { return n; }"),
        text("fn f(n) { return 1; }"),
        text("fn f(n) { return 2; }"),
    ]));
    let run = vcw
        .run(
            &request("code-task", 3, 3),
            &everyone(),
            &verifier,
            &mut pods,
        )
        .unwrap();
    assert_unknown_regenerates(&run, &verifier.seen.borrow(), "code.task");
}

#[test]
fn code_test_failures_are_hidden_by_default() {
    // The repair Pod sees only the previous candidate and the finding code; the verifier's
    // message for a hidden test names the id, so there is nothing to hard-code from it.
    let verifier = CodeTestVerifier::new(square_task());
    let report = verifier.verify(&"fn f(n) { return n + n; }".to_owned());
    assert!(report
        .findings
        .iter()
        .all(|f| !f.message.contains("expected")));
}

// ---------------------------------------------------------------------------------------------
// Facts
// ---------------------------------------------------------------------------------------------

fn snapshot_host() -> SemanticHost {
    let mut host = SemanticHost::default();
    let mut delta = SemanticDelta::default();
    delta
        .upserts
        .insert(fact_key("paris", "capital-of"), "france".into());
    delta
        .upserts
        .insert(fact_key("rome", "capital-of"), "italy".into());
    host.apply_delta(delta).unwrap();
    host
}

fn claim(subject: &str, value: &str) -> Claims {
    Claims(vec![FactClaim::new(subject, "capital-of", value)])
}

#[test]
fn a_contradicted_fact_goes_to_fact_repair_and_the_repaired_claim_is_accepted() {
    let host = snapshot_host();
    let verifier = on_claims(FactVerifier::new(host.snapshot(), host.revision()));
    let mut vcw = new_loop(&role_assignments());
    let mut pods = Script::new(vec![claim("paris", "spain")])
        .repairing(RepairRole::Fact, |_| Some(claim("paris", "france")));
    let run = vcw
        .run(
            &request("fact-contradicted", 6, 4),
            &everyone(),
            &verifier,
            &mut pods,
        )
        .unwrap();
    assert_eq!(answer(&run), &claim("paris", "france"));
    assert_eq!(
        repair_routes(&run),
        [route("fact.contradicted", RepairRole::Fact)]
    );
}

#[test]
fn malformed_and_oversize_claims_go_to_format_repair() {
    let host = snapshot_host();
    for (id, bad) in [
        (
            "fact-malformed",
            Claims(vec![FactClaim::new("", "capital-of", "x")]),
        ),
        (
            "fact-oversize",
            Claims(vec![FactClaim::new("paris", "capital-of", "france"); 65]),
        ),
    ] {
        let verifier = on_claims(FactVerifier::new(host.snapshot(), host.revision()));
        let mut vcw = new_loop(&role_assignments());
        let mut pods = Script::new(vec![bad])
            .repairing(RepairRole::Format, |_| Some(claim("paris", "france")))
            .repairing(RepairRole::Fact, |_| {
                panic!("fact repair must not be asked for a malformed claim list")
            });
        let run = vcw
            .run(&request(id, 6, 4), &everyone(), &verifier, &mut pods)
            .unwrap();
        assert_eq!(answer(&run), &claim("paris", "france"), "{id}");
        let code = id.replace("fact-", "fact.");
        assert_eq!(
            repair_routes(&run),
            [route(&code, RepairRole::Format)],
            "{id}"
        );
        assert_eq!(verifier.seen.borrow()[0], [code.as_str()], "{id}");
    }
}

#[test]
fn an_unknown_fact_is_never_repaired_and_regenerates_a_fresh_proposal() {
    let host = snapshot_host();
    let verifier = on_claims(FactVerifier::new(host.snapshot(), host.revision()));
    let mut vcw = new_loop(&role_assignments());
    // `mars` is not in the snapshot: Unknown. The second proposal is a known fact.
    let mut pods = Script::new(vec![claim("mars", "olympus"), claim("rome", "italy")])
        .repairing(RepairRole::Fact, |_| {
            panic!("an unknown fact is not a repair target")
        });
    let run = vcw
        .run(
            &request("fact-unknown", 6, 4),
            &everyone(),
            &verifier,
            &mut pods,
        )
        .unwrap();
    assert_eq!(answer(&run), &claim("rome", "italy"));
    assert!(repair_routes(&run).is_empty());
    assert_eq!(run.trace.steps()[0].decision, StepDecision::Regenerate);
}

/// The proposer is revoked once it has proposed.
struct RevokedAfterFirstProposal {
    inner: EligibilitySnapshot,
    proposed: Rc<Cell<bool>>,
}

impl PodEligibility for RevokedAfterFirstProposal {
    fn current_generation(&self, pod: &PodId) -> Option<Generation> {
        if pod.0 == "gen" && self.proposed.get() {
            None
        } else {
            self.inner.current_generation(pod)
        }
    }
}

#[test]
fn an_unknown_fact_with_no_proposer_left_escalates_to_the_host_target() {
    let host = snapshot_host();
    let verifier = on_claims(FactVerifier::new(host.snapshot(), host.revision()));
    let mut vcw = new_loop(&role_assignments());
    let proposed = Rc::new(Cell::new(false));
    let view = RevokedAfterFirstProposal {
        inner: everyone(),
        proposed: Rc::clone(&proposed),
    };
    let mut pods = Script::new(vec![claim("mars", "olympus")]).repairing(RepairRole::Fact, |_| {
        panic!("an unknown fact is not a repair target")
    });
    let flag = Rc::clone(&proposed);
    pods.on_propose = Some(Box::new(move || flag.set(true)));
    pods.escalation = Some((
        PodRef::new(pod("boss"), Generation(1)),
        claim("paris", "france"),
    ));
    let run = vcw
        .run(&request("fact-escalate", 6, 4), &view, &verifier, &mut pods)
        .unwrap();
    // The escalation target's candidate is verified like any other, and passes.
    assert_eq!(answer(&run), &claim("paris", "france"));
    assert!(pods
        .log
        .borrow()
        .iter()
        .any(|line| line.starts_with("escalate:")));
    assert!(repair_routes(&run).is_empty());
    assert!(run
        .trace
        .steps()
        .iter()
        .any(|step| step.decision == StepDecision::Escalate));
}

#[test]
fn an_unknown_fact_nobody_can_answer_ends_escalated_with_no_candidate() {
    let host = snapshot_host();
    let verifier = on_claims(FactVerifier::new(host.snapshot(), host.revision()));
    let mut vcw = new_loop(&role_assignments());
    let proposed = Rc::new(Cell::new(false));
    let view = RevokedAfterFirstProposal {
        inner: everyone(),
        proposed: Rc::clone(&proposed),
    };
    let mut pods = Script::new(vec![claim("mars", "olympus")]);
    let flag = Rc::clone(&proposed);
    pods.on_propose = Some(Box::new(move || flag.set(true)));
    let run = vcw
        .run(
            &request("fact-escalate-none", 6, 4),
            &view,
            &verifier,
            &mut pods,
        )
        .unwrap();
    assert!(
        matches!(
            run.outcome,
            VcwOutcome::Escalated {
                reason: EscalationReason::NoProposer(FallbackReason::Unknown)
                    | EscalationReason::Policy(FallbackReason::Unknown)
            }
        ),
        "{:?}",
        run.outcome
    );
}

#[test]
fn a_stale_snapshot_is_unknown_so_a_contradiction_cannot_be_forged_from_old_state() {
    let mut host = snapshot_host();
    let stale = host.snapshot();
    let mut delta = SemanticDelta::default();
    delta
        .upserts
        .insert(fact_key("paris", "capital-of"), "lyon".into());
    host.apply_delta(delta).unwrap();
    let verifier = on_claims(FactVerifier::new(stale, host.revision()));
    let mut vcw = new_loop(&role_assignments());
    let mut pods = forbid_all_repairs(Script::new(vec![
        claim("paris", "france"),
        claim("paris", "france"),
    ]));
    let run = vcw
        .run(
            &request("fact-stale", 3, 3),
            &everyone(),
            &verifier,
            &mut pods,
        )
        .unwrap();
    assert_unknown_regenerates(&run, &verifier.seen.borrow(), "fact.stale_snapshot");
}

// ---------------------------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------------------------

#[test]
fn the_same_run_twice_gives_the_same_trace_digest() {
    let run_once = || {
        let mut vcw = new_loop(&role_assignments());
        let verifier = on_text(ArithmeticVerifier::new("(3 + 4) * 5"));
        let mut pods =
            Script::new(vec![text("36")]).repairing(RepairRole::Value, |_| Some(text("35")));
        let run = vcw
            .run(&request("replay", 6, 4), &everyone(), &verifier, &mut pods)
            .unwrap();
        (*run.trace.digest(), repair_routes(&run))
    };
    assert_eq!(run_once(), run_once());
}
