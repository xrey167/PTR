//! Expert Pods in the verifier-closed loop: text-generator experts for arithmetic and JSON
//! objects (with scripted generators, so this runs without a model), seeded programmatic
//! experts, manifests registered in the Pod registry, and revocation. The model-backed run is
//! the `vcw_qwen` example and the ignored test in `ptr-pods`.

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use ptr_config::PtrConfig;
use ptr_ledger::LedgerEvent;
use ptr_pods::{
    GenerationError, GenerationOutput, GenerationRequest, PodRegistry, StopReason, TextGenerator,
    TypedPayload,
};
use ptr_router::{
    NonceSequence, PodEligibility, RouterConfig, SamplingKey, SelfModel, SelfModelLimits,
    UniformScorer,
};
use ptr_runtime::expert::{StructurePolicy, StructurePolicyConfig};
use ptr_runtime::vcw::experts::{
    CallOutcome, DefectDoc, DefectDocVerifier, ExpertErrorKind, ExpertRole, ExpertTask,
    LlmExpertPod, LlmExpertPods, RosterError, SimulatedExperts, SimulationConfig, TaskKind,
    TextPayload,
};
use ptr_runtime::vcw::{
    OutcomeTag, ProposeCall, RepairCall, RequestBudget, RolePods, RuntimeStanding, VcwConfig,
    VcwLoop, VcwOutcome, VcwParts, VcwRequest, VcwRun,
};
use ptr_runtime::PtrRuntime;
use ptr_types::{Generation, PodId, ProjectId, RequestId, VerificationLevel};
use sha2::{Digest as _, Sha256};

#[path = "vcw_experts_support/mod.rs"]
mod support;

use support::{
    ArithmeticVerifier, JsonObjectVerifier, Kind, ARITHMETIC_CODES, INVALID_JSON, JSON_CODES,
    MISSING_FIELD, NOT_A_NUMBER, WRONG_ANSWER,
};

fn project() -> ProjectId {
    ProjectId::from("proj")
}

fn pod(name: &str) -> PodId {
    PodId::from(name)
}

fn request(id: &str, calls: u32, rounds: u32) -> VcwRequest {
    VcwRequest {
        request_id: RequestId::from(id),
        project: project(),
        request_digest: Sha256::digest(id.as_bytes()).into(),
        required: VerificationLevel::SampleVerified,
        budget: RequestBudget::new(calls, rounds).unwrap(),
    }
}

fn new_loop(roster: ptr_runtime::vcw::Roster, seed: u8) -> VcwLoop {
    VcwLoop::new(VcwParts {
        config: VcwConfig::new("local-test-verifier", [3; 32]),
        roster,
        router: RouterConfig::default(),
        scorer: UniformScorer,
        key: SamplingKey::from_bytes([seed; 32]),
        nonces: NonceSequence::starting_at([1; 8], 0),
        self_model: SelfModel::new(SelfModelLimits::default()),
        schema: None,
        structure: StructurePolicy::new(StructurePolicyConfig::default()).unwrap(),
    })
    .unwrap()
}

/// A generator that answers from a closure over the request, counting its calls.
struct Scripted<F> {
    reply: F,
    calls: Arc<Mutex<Vec<(String, String)>>>,
    identity: Option<ptr_pods::ModelIdentity>,
}

impl<F: FnMut(&GenerationRequest<'_>) -> String> TextGenerator for Scripted<F> {
    fn generate(
        &mut self,
        request: &GenerationRequest<'_>,
    ) -> Result<GenerationOutput, GenerationError> {
        request.validate()?;
        self.calls
            .lock()
            .unwrap()
            .push((request.system.to_owned(), request.user.to_owned()));
        let text = (self.reply)(request);
        let tokens = u32::try_from(text.split_whitespace().count()).unwrap();
        Ok(GenerationOutput::new(text, tokens, StopReason::EndOfTurn))
    }

    fn identity(&self) -> Option<ptr_pods::ModelIdentity> {
        self.identity.clone()
    }
}

/// A generator that fails every call with a backend error.
struct Broken {
    calls: Arc<Mutex<u32>>,
}

impl TextGenerator for Broken {
    fn generate(
        &mut self,
        _request: &GenerationRequest<'_>,
    ) -> Result<GenerationOutput, GenerationError> {
        *self.calls.lock().unwrap() += 1;
        Err(GenerationError::Backend("logit was not finite".into()))
    }
}

type CallLog = Arc<Mutex<Vec<(String, String)>>>;

fn shared<F>(reply: F) -> (ptr_runtime::vcw::experts::SharedGenerator, CallLog)
where
    F: FnMut(&GenerationRequest<'_>) -> String + Send + 'static,
{
    let calls = CallLog::default();
    let generator: ptr_runtime::vcw::experts::SharedGenerator = Arc::new(Mutex::new(Scripted {
        reply,
        calls: calls.clone(),
        identity: None,
    }));
    (generator, calls)
}

/// How many times the generator ran with `system` as its system prompt.
fn calls_with(calls: &CallLog, system: &str) -> usize {
    calls
        .lock()
        .unwrap()
        .iter()
        .filter(|(s, _)| s == system)
        .count()
}

/// A live eligibility view that tests change while a request runs.
struct LiveView(RefCell<ptr_router::EligibilitySnapshot>);

impl LiveView {
    fn with(pods: &[&str]) -> Self {
        let mut snapshot = ptr_router::EligibilitySnapshot::new();
        for name in pods {
            snapshot.set(pod(name), Generation(1));
        }
        Self(RefCell::new(snapshot))
    }

    fn revoke(&self, name: &str) {
        self.0.borrow_mut().revoke(&pod(name));
    }

    fn restore(&self, name: &str) {
        self.0.borrow_mut().set(pod(name), Generation(1));
    }
}

impl PodEligibility for LiveView {
    fn current_generation(&self, pod: &PodId) -> Option<Generation> {
        self.0.borrow().current_generation(pod)
    }
}

/// What the interposer does to the view, and when.
enum Switch {
    /// Leave the view alone.
    Nothing,
    /// Revoke the Pod the loop is about to call, once `n` calls have been forwarded: the loop
    /// has already routed to it.
    RevokeCalledPodAtCall(usize),
    /// Revoke `pod` right after the `n`-th forwarded call returns.
    RevokeAfterCall(usize, &'static str),
    /// Move the called Pod to generation 2 (superseded) at the `n`-th call.
    SupersedeCalledPodAtCall(usize),
}

/// Sits between the loop and the guarded adapter and changes the live view between the routing
/// decision and the call, as a revocation arriving in between would.
struct Interposer<'a, R> {
    inner: R,
    view: &'a LiveView,
    switch: Switch,
    forwarded: usize,
}

impl<R> Interposer<'_, R> {
    fn before(&mut self, called: &PodId) {
        match self.switch {
            Switch::RevokeCalledPodAtCall(n) if n == self.forwarded => {
                self.view.revoke(&called.0);
            }
            Switch::SupersedeCalledPodAtCall(n) if n == self.forwarded => {
                self.view.0.borrow_mut().set(called.clone(), Generation(2));
            }
            _ => {}
        }
    }

    fn after(&mut self) {
        if let Switch::RevokeAfterCall(n, name) = self.switch {
            if n == self.forwarded {
                self.view.revoke(name);
            }
        }
        self.forwarded += 1;
    }
}

impl<R: RolePods<TextPayload>> RolePods<TextPayload> for Interposer<'_, R> {
    fn propose(&mut self, call: ProposeCall<'_>) -> Option<TextPayload> {
        self.before(&call.pod.pod);
        let out = self.inner.propose(call);
        self.after();
        out
    }

    fn repair(&mut self, call: RepairCall<'_, TextPayload>) -> Option<TextPayload> {
        self.before(&call.pod.pod);
        let out = self.inner.repair(call);
        self.after();
        out
    }
}

const PROPOSER_SYSTEM: &str = "proposer-system";
const REPAIRER_SYSTEM: &str = "repairer-system";

/// A proposer `prop` and the repairers `fix_a` and `fix_b`, all over one generator.
fn arithmetic_experts(generator: &ptr_runtime::vcw::experts::SharedGenerator) -> LlmExpertPods {
    let mut pods = LlmExpertPods::new(ExpertTask::new(TaskKind::Arithmetic, "What is 17 + 25?"));
    let make = |name: &str, role, system: &str| {
        Arc::new(
            LlmExpertPod::new(
                &project(),
                &pod(name),
                role,
                TaskKind::Arithmetic,
                system,
                16,
                generator.clone(),
            )
            .unwrap(),
        )
    };
    pods.add_proposer(make("prop", ExpertRole::Proposer, PROPOSER_SYSTEM))
        .unwrap();
    pods.add_repairer(
        make("fix_a", ExpertRole::Repairer, REPAIRER_SYSTEM),
        &ARITHMETIC_CODES,
    )
    .unwrap();
    pods.add_repairer(
        make("fix_b", ExpertRole::Repairer, REPAIRER_SYSTEM),
        &ARITHMETIC_CODES,
    )
    .unwrap();
    pods
}

/// A runtime in which every named Pod is live at generation 1 in the lifecycle authority, and
/// a Pod registry holding the experts' manifests.
fn runtime_with(pods: &LlmExpertPods, names: &[&str]) -> (PtrRuntime, PodRegistry) {
    let mut rt = PtrRuntime::new(PtrConfig::default()).unwrap();
    for name in names {
        rt.commit(LedgerEvent::CapsuleCommitted {
            project: project(),
            capsule: ptr_types::CapsuleId::from(*name),
            generation: Generation(1),
        })
        .unwrap();
    }
    let mut registry = PodRegistry::default();
    for expert in pods.pods() {
        registry.register_new(expert).unwrap();
    }
    (rt, registry)
}

fn standing<'a>(
    rt: &'a PtrRuntime,
    registry: &'a PodRegistry,
    project: &'a ProjectId,
    names: &[&str],
) -> RuntimeStanding<'a> {
    let mut standing = RuntimeStanding::new(rt, registry, project);
    for name in names {
        standing = standing.with_base_pod(pod(name), Generation(1));
    }
    standing
}

fn verified_text(run: &VcwRun<TextPayload>) -> &str {
    match &run.outcome {
        VcwOutcome::Verified(answer) => answer.payload().text(),
        other => panic!("expected Verified, got {other:?}"),
    }
}

#[test]
fn a_wrong_arithmetic_answer_is_repaired_by_a_repair_pod_on_the_named_code() {
    let (generator, calls) = shared(|request| {
        if request.system == PROPOSER_SYSTEM {
            "The answer is 41.".to_owned()
        } else {
            "42".to_owned()
        }
    });
    let mut experts = arithmetic_experts(&generator);
    let names = ["prop", "fix_a", "fix_b"];
    let (rt, registry) = runtime_with(&experts, &names);
    let mut vcw = new_loop(experts.roster(), 5);
    let project = project();
    let view = standing(&rt, &registry, &project, &names);

    let run = vcw
        .run(
            &request("arith-1", 4, 3),
            &view,
            &ArithmeticVerifier { expected: 42 },
            &mut experts.guarded(&view),
        )
        .unwrap();

    assert_eq!(verified_text(&run), "42");
    assert!(matches!(run.trace.outcome(), OutcomeTag::Verified));
    let log: Vec<_> = experts.calls().iter().collect();
    assert_eq!(log.len(), 2);
    assert_eq!(log[0].role, ExpertRole::Proposer);
    assert_eq!(log[0].pod, pod("prop"));
    assert_eq!(log[1].role, ExpertRole::Repairer);
    assert_eq!(log[1].code.as_deref(), Some(WRONG_ANSWER));
    // The repair prompt names one code and shows the previous answer, never a report or score.
    let sent = calls.lock().unwrap();
    assert_eq!(sent.len(), 2);
    assert!(sent[1].1.contains(WRONG_ANSWER));
    assert!(sent[1].1.contains("A previous answer was: 41"));
    assert!(!sent[1].1.contains("score"));
    // The routing record is in the trace with the repairer's generation.
    let routed: Vec<_> = run.trace.routes().map(|r| r.pod().0.clone()).collect();
    assert!(routed.iter().any(|p| p == "prop"));
    assert!(routed.iter().any(|p| p.starts_with("fix_")));
}

#[test]
fn a_json_object_with_a_missing_field_is_repaired_and_a_second_code_is_not_touched() {
    let (generator, _calls) = shared(|request| {
        if request.system == PROPOSER_SYSTEM {
            "```json\n{\"name\": \"Ada\"}\n```".to_owned()
        } else {
            "{\"name\": \"Ada\", \"age\": 36}".to_owned()
        }
    });
    let mut experts = LlmExpertPods::new(ExpertTask::new(
        TaskKind::JsonObject,
        "Return a JSON object with the string name and the integer age for Ada, aged 36.",
    ));
    for (name, role, system) in [
        ("prop", ExpertRole::Proposer, PROPOSER_SYSTEM),
        ("fix_json", ExpertRole::Repairer, REPAIRER_SYSTEM),
    ] {
        let pod = Arc::new(
            LlmExpertPod::new(
                &project(),
                &PodId::from(name),
                role,
                TaskKind::JsonObject,
                system,
                64,
                generator.clone(),
            )
            .unwrap(),
        );
        match role {
            ExpertRole::Proposer => experts.add_proposer(pod).unwrap(),
            _ => experts.add_repairer(pod, &JSON_CODES).unwrap(),
        }
    }
    let names = ["prop", "fix_json"];
    let (rt, registry) = runtime_with(&experts, &names);
    let mut vcw = new_loop(experts.roster(), 9);
    let project = project();
    let view = standing(&rt, &registry, &project, &names);
    let verifier = JsonObjectVerifier {
        fields: vec![("name", Kind::String), ("age", Kind::Integer)],
    };
    let run = vcw
        .run(
            &request("json-1", 4, 3),
            &view,
            &verifier,
            &mut experts.guarded(&view),
        )
        .unwrap();
    assert_eq!(verified_text(&run), "{\"name\": \"Ada\", \"age\": 36}");
    let repair = experts
        .calls()
        .iter()
        .find(|c| c.role == ExpertRole::Repairer)
        .unwrap();
    assert_eq!(repair.code.as_deref(), Some(MISSING_FIELD));
    assert_ne!(repair.code.as_deref(), Some(INVALID_JSON));
}

#[test]
fn text_that_holds_no_answer_reaches_the_verifier_and_the_request_ends_exhausted() {
    let (generator, _calls) = shared(|_| "I cannot say.".to_owned());
    let mut experts = arithmetic_experts(&generator);
    let names = ["prop", "fix_a", "fix_b"];
    let (rt, registry) = runtime_with(&experts, &names);
    let mut vcw = new_loop(experts.roster(), 5);
    let project = project();
    let view = standing(&rt, &registry, &project, &names);
    let run = vcw
        .run(
            &request("arith-2", 4, 2),
            &view,
            &ArithmeticVerifier { expected: 42 },
            &mut experts.guarded(&view),
        )
        .unwrap();
    // The model never produces a number: every candidate fails `not_a_number`, and the budget
    // ends the request. A rejected candidate is never returned.
    assert!(
        matches!(run.outcome, VcwOutcome::Exhausted { .. }),
        "{:?}",
        run.outcome
    );
    let codes: BTreeSet<_> = experts
        .calls()
        .iter()
        .filter_map(|c| c.code.clone())
        .collect();
    assert_eq!(codes, BTreeSet::from([NOT_A_NUMBER.to_owned()]));
}

#[test]
fn a_revoked_expert_is_never_called_and_its_manifest_is_in_the_registry() {
    let (generator, _calls) = shared(|request| {
        if request.system == PROPOSER_SYSTEM {
            "41".to_owned()
        } else {
            "42".to_owned()
        }
    });
    let mut experts = arithmetic_experts(&generator);
    let names = ["prop", "fix_a", "fix_b"];
    let (mut rt, registry) = runtime_with(&experts, &names);

    // Manifests: registered, Pure, one capability per role, the answer type both ways.
    let manifest = registry
        .get(&project(), &pod("fix_a"))
        .unwrap()
        .manifest()
        .clone();
    assert_eq!(manifest.capabilities[0].0, "repair");
    assert_eq!(manifest.accepts, vec![TaskKind::Arithmetic.payload_type()]);

    rt.commit(LedgerEvent::Revoked {
        subject: "fix_a".to_owned(),
        generation: Generation(1),
    })
    .unwrap();
    let project = project();
    {
        let view = standing(&rt, &registry, &project, &names);
        assert_eq!(view.current_generation(&pod("fix_a")), None);
        assert_eq!(view.current_generation(&pod("fix_b")), Some(Generation(1)));
    }
    let mut vcw = new_loop(experts.roster(), 11);
    for i in 0..25 {
        let view = standing(&rt, &registry, &project, &names);
        let run = vcw
            .run(
                &request(&format!("revoked-{i}"), 4, 3),
                &view,
                &ArithmeticVerifier { expected: 42 },
                &mut experts.guarded(&view),
            )
            .unwrap();
        assert_eq!(verified_text(&run), "42");
    }
    assert!(experts.calls().iter().all(|c| c.pod != pod("fix_a")));
    assert!(experts.calls().iter().any(|c| c.pod == pod("fix_b")));
    assert_eq!(vcw.counters().verified, 25);
}

#[test]
fn a_pod_invoked_through_the_registry_runs_the_generator_with_its_own_system_prompt() {
    let (generator, calls) = shared(|request| format!("echo:{}", request.user));
    let experts = arithmetic_experts(&generator);
    let (_rt, registry) = runtime_with(&experts, &["prop", "fix_a", "fix_b"]);
    let handle = registry.get(&project(), &pod("prop")).unwrap();
    let out = handle
        .invoke(TypedPayload {
            type_id: TaskKind::Arithmetic.payload_type(),
            bytes: b"hello".to_vec(),
        })
        .unwrap();
    assert_eq!(out.bytes, b"echo:hello");
    assert_eq!(calls.lock().unwrap()[0].0, PROPOSER_SYSTEM);
    // A payload of another type is refused before the generator runs.
    let refused = handle.invoke(TypedPayload {
        type_id: ptr_types::TypeId::from("other"),
        bytes: Vec::new(),
    });
    assert!(refused.is_err());
    assert_eq!(calls.lock().unwrap().len(), 1);
}

// ---------------------------------------------------------------------------------------------
// Seeded programmatic experts
// ---------------------------------------------------------------------------------------------

fn simulated(config: SimulationConfig) -> SimulatedExperts<DefectDoc> {
    let mut experts = SimulatedExperts::new(config, "task").unwrap();
    experts.add_proposer(pod("gen")).unwrap();
    experts.add_repairer(pod("fix_a"), &["a", "b"]).unwrap();
    experts.add_repairer(pod("fix_b"), &["a", "b"]).unwrap();
    experts
}

fn simulated_run(
    config: SimulationConfig,
    id: &str,
    rounds: u32,
    loop_seed: u8,
) -> (VcwRun<DefectDoc>, SimulatedExperts<DefectDoc>) {
    let mut experts = simulated(config);
    let mut vcw = new_loop(experts.roster().unwrap(), loop_seed);
    let mut view = ptr_router::EligibilitySnapshot::new();
    for name in ["gen", "fix_a", "fix_b"] {
        view.set(pod(name), Generation(1));
    }
    let run = vcw
        .run(
            &request(id, 8, rounds),
            &view,
            &DefectDocVerifier,
            &mut experts.guarded(&view),
        )
        .unwrap();
    (run, experts)
}

#[test]
fn the_simulation_is_replayable_and_depends_on_the_seed() {
    let config = |seed| {
        SimulationConfig::new(seed)
            .with_error_rate("a", 0.7)
            .with_error_rate("b", 0.4)
            .with_repair_success("a", 0.6)
            .with_repair_success("b", 0.8)
    };
    let digests = |seed: u64| -> Vec<[u8; 32]> {
        (0..12)
            .map(|i| {
                *simulated_run(config(seed), &format!("replay-{i}"), 6, 3)
                    .0
                    .trace
                    .digest()
            })
            .collect()
    };
    assert_eq!(digests(7), digests(7));
    assert_ne!(digests(7), digests(8));
}

#[test]
fn error_rates_and_repair_success_are_the_configured_ones() {
    // Rate 0 never plants a defect, rate 1 always does; a repair that always works clears it.
    let (run, _) = simulated_run(SimulationConfig::new(1), "clean", 3, 3);
    assert!(matches!(run.outcome, VcwOutcome::Verified(_)));
    assert_eq!(run.trace.candidates().count(), 1);

    let config = SimulationConfig::new(1).with_error_rate("a", 1.0);
    let (run, experts) = simulated_run(config, "always-a", 3, 3);
    assert!(matches!(run.outcome, VcwOutcome::Verified(_)));
    let repair = experts
        .calls()
        .iter()
        .find(|c| c.role == ExpertRole::Repairer)
        .unwrap();
    assert_eq!(repair.code.as_deref(), Some("a"));

    // A repair that never works, declining: the request is exhausted, never "best effort".
    let mut config = SimulationConfig::new(1)
        .with_error_rate("a", 1.0)
        .with_repair_success("a", 0.0);
    config.decline_on_failure = true;
    let (run, _) = simulated_run(config, "never-fixed", 3, 3);
    assert!(
        matches!(run.outcome, VcwOutcome::Exhausted { .. }),
        "{:?}",
        run.outcome
    );
}

#[test]
fn the_measured_repair_rate_follows_the_configured_probability_per_code() {
    let config = SimulationConfig::new(99)
        .with_error_rate("a", 1.0)
        .with_error_rate("b", 1.0)
        .with_repair_success("a", 0.25)
        .with_repair_success("b", 0.75);
    let mut tries = BTreeSet::new();
    let (mut ok_a, mut n_a, mut ok_b, mut n_b) = (0_u32, 0_u32, 0_u32, 0_u32);
    for i in 0..300 {
        let (_run, experts) = simulated_run(config.clone(), &format!("rate-{i}"), 8, 21);
        // Per repair call: did the next call on that code still find the defect? Read the
        // outcome from the call log: a repair call is followed by another call on the same
        // code exactly when it failed.
        let calls: Vec<_> = experts
            .calls()
            .iter()
            .filter(|c| c.role == ExpertRole::Repairer)
            .collect();
        for (index, call) in calls.iter().enumerate() {
            let code = call.code.clone().unwrap();
            tries.insert(code.clone());
            let failed = calls[index + 1..]
                .iter()
                .any(|later| later.code == call.code);
            let hit = !failed;
            // Only the first call on a code in a request is an independent trial.
            if calls[..index]
                .iter()
                .any(|earlier| earlier.code == call.code)
            {
                continue;
            }
            match code.as_str() {
                "a" => {
                    n_a += 1;
                    ok_a += u32::from(hit);
                }
                _ => {
                    n_b += 1;
                    ok_b += u32::from(hit);
                }
            }
        }
    }
    assert_eq!(tries, BTreeSet::from(["a".to_owned(), "b".to_owned()]));
    let (rate_a, rate_b) = (
        f64::from(ok_a) / f64::from(n_a),
        f64::from(ok_b) / f64::from(n_b),
    );
    assert!((rate_a - 0.25).abs() < 0.08, "a: {rate_a} over {n_a}");
    assert!((rate_b - 0.75).abs() < 0.08, "b: {rate_b} over {n_b}");
}

#[test]
fn simulated_pods_register_with_manifests_and_a_revoked_one_is_never_called() {
    let experts = simulated(SimulationConfig::new(1).with_error_rate("a", 1.0));
    let mut registry = PodRegistry::default();
    for expert in experts.pods(&project(), &DefectDoc::clean_type()) {
        registry.register_new(expert).unwrap();
    }
    assert!(registry.get(&project(), &pod("gen")).is_some());
    let direct = registry
        .get(&project(), &pod("fix_a"))
        .unwrap()
        .invoke(TypedPayload {
            type_id: DefectDoc::clean_type(),
            bytes: Vec::new(),
        });
    assert!(direct.is_err());

    let mut experts = experts;
    let mut vcw = new_loop(experts.roster().unwrap(), 3);
    let mut view = ptr_router::EligibilitySnapshot::new();
    for name in ["gen", "fix_b"] {
        view.set(pod(name), Generation(1));
    }
    for i in 0..20 {
        let run = vcw
            .run(
                &request(&format!("sim-revoked-{i}"), 6, 3),
                &view,
                &DefectDocVerifier,
                &mut experts.guarded(&view),
            )
            .unwrap();
        assert!(matches!(run.outcome, VcwOutcome::Verified(_)));
    }
    assert!(experts.calls().iter().all(|c| c.pod != pod("fix_a")));
}

#[test]
fn a_bad_probability_is_refused_at_construction() {
    let config = SimulationConfig::new(1).with_error_rate("a", 1.5);
    assert!(SimulatedExperts::<DefectDoc>::new(config, "t").is_err());
}

// ---------------------------------------------------------------------------------------------
// Call-time eligibility (the adapters re-check; the loop's route-time view is not enough)
// ---------------------------------------------------------------------------------------------

fn run_live(
    vcw: &mut VcwLoop,
    id: &str,
    view: &LiveView,
    experts: &mut LlmExpertPods,
    switch: Switch,
    verifier: &dyn ptr_verifier::Verifier<TextPayload>,
) -> VcwRun<TextPayload> {
    let mut interposer = Interposer {
        inner: experts.guarded(view),
        view,
        switch,
        forwarded: 0,
    };
    vcw.run(&request(id, 6, 4), view, verifier, &mut interposer)
        .unwrap()
}

#[test]
fn a_proposer_revoked_between_route_and_call_never_runs_and_the_loop_goes_on() {
    let (generator, calls) = shared(|request| {
        if request.system == PROPOSER_SYSTEM {
            "41".to_owned()
        } else {
            "42".to_owned()
        }
    });
    let mut experts = arithmetic_experts(&generator);
    let view = LiveView::with(&["prop", "fix_a", "fix_b"]);
    let mut vcw = new_loop(experts.roster(), 3);
    let verifier = ArithmeticVerifier { expected: 42 };

    let run = run_live(
        &mut vcw,
        "revoke-prop",
        &view,
        &mut experts,
        Switch::RevokeCalledPodAtCall(0),
        &verifier,
    );
    // The model never ran for the revoked proposer, and nothing was returned as an answer.
    assert_eq!(calls_with(&calls, PROPOSER_SYSTEM), 0);
    assert!(calls.lock().unwrap().is_empty());
    assert!(
        !matches!(run.outcome, VcwOutcome::Verified(_)),
        "{:?}",
        run.outcome
    );
    let first = experts.calls().front().unwrap();
    assert_eq!(
        (first.role, first.pod.clone()),
        (ExpertRole::Proposer, pod("prop"))
    );
    assert_eq!(first.outcome, CallOutcome::IneligibleAtCallTime);
    assert_eq!(first.outcome.to_string(), "ineligible at call time");
    assert!(!first.produced);
    assert_eq!(first.new_tokens, 0);

    // The same experts serve the next request once the Pod is back.
    view.restore("prop");
    let run = run_live(
        &mut vcw,
        "revoke-prop-2",
        &view,
        &mut experts,
        Switch::Nothing,
        &verifier,
    );
    assert_eq!(verified_text(&run), "42");
    assert_eq!(calls_with(&calls, PROPOSER_SYSTEM), 1);
}

#[test]
fn a_repairer_revoked_or_superseded_between_route_and_call_never_runs() {
    for supersede in [false, true] {
        let (generator, calls) = shared(|request| {
            if request.system == PROPOSER_SYSTEM {
                "41".to_owned()
            } else {
                "42".to_owned()
            }
        });
        let mut experts = arithmetic_experts(&generator);
        let view = LiveView::with(&["prop", "fix_a", "fix_b"]);
        let mut vcw = new_loop(experts.roster(), 3);
        let switch = if supersede {
            Switch::SupersedeCalledPodAtCall(1)
        } else {
            Switch::RevokeCalledPodAtCall(1)
        };
        let run = run_live(
            &mut vcw,
            "revoke-fix",
            &view,
            &mut experts,
            switch,
            &ArithmeticVerifier { expected: 42 },
        );
        let log: Vec<_> = experts.calls().iter().cloned().collect();
        assert_eq!(log[0].outcome, CallOutcome::Produced);
        assert_eq!(log[1].role, ExpertRole::Repairer);
        assert_eq!(
            log[1].outcome,
            CallOutcome::IneligibleAtCallTime,
            "{supersede}"
        );
        let withdrawn = log[1].pod.clone();
        // The withdrawn Pod's generator was never invoked: one proposal call, then at most the
        // call of the other repairer that the loop routed to afterwards.
        let sent = calls.lock().unwrap().len();
        let ran: Vec<_> = log.iter().filter(|c| c.produced).collect();
        assert_eq!(
            sent,
            ran.len(),
            "every generator run is a logged produced call"
        );
        // (A superseded Pod is eligible again at its new generation; the old one never ran.)
        assert!(ran
            .iter()
            .all(|c| !(c.pod == withdrawn && c.generation == Generation(1))));
        // The loop went on to the other repairer, which fixed the answer.
        assert_eq!(verified_text(&run), "42", "{supersede}");
        assert!(ran.iter().any(|c| c.role == ExpertRole::Repairer));
    }
}

#[test]
fn a_repairer_revoked_between_two_actions_of_one_plan_is_not_run_for_the_second() {
    // The candidate has two defects (a missing name and a wrongly typed age), so one plan holds
    // two actions on two Pods; the second Pod is revoked after the first action ran.
    let (generator, calls) = shared(|request| match request.system {
        PROPOSER_SYSTEM => "{\"age\": \"old\"}".to_owned(),
        "fix_m" => "{\"name\": \"Ada\", \"age\": \"old\"}".to_owned(),
        _ => "{\"name\": \"Ada\", \"age\": 36}".to_owned(),
    });
    let mut experts = LlmExpertPods::new(ExpertTask::new(TaskKind::JsonObject, "Ada, aged 36"));
    for (name, role, codes) in [
        ("prop", ExpertRole::Proposer, &[][..]),
        ("fix_m", ExpertRole::Repairer, &JSON_CODES[1..2]),
        ("fix_t", ExpertRole::Repairer, &JSON_CODES[2..3]),
    ] {
        let system = if name == "prop" {
            PROPOSER_SYSTEM
        } else {
            name
        };
        let pod = Arc::new(
            LlmExpertPod::new(
                &project(),
                &PodId::from(name),
                role,
                TaskKind::JsonObject,
                system,
                64,
                generator.clone(),
            )
            .unwrap(),
        );
        match role {
            ExpertRole::Proposer => experts.add_proposer(pod).unwrap(),
            _ => experts.add_repairer(pod, codes).unwrap(),
        }
    }
    let view = LiveView::with(&["prop", "fix_m", "fix_t"]);
    let mut vcw = new_loop(experts.roster(), 4);
    let verifier = JsonObjectVerifier {
        fields: vec![("name", Kind::String), ("age", Kind::Integer)],
    };
    // Forwarded call 0 is the proposal, 1 the first action, 2 the second.
    let _ = run_live(
        &mut vcw,
        "plan",
        &view,
        &mut experts,
        Switch::RevokeAfterCall(1, "fix_t"),
        &verifier,
    );
    assert!(calls_with(&calls, "fix_m") >= 1);
    assert_eq!(
        calls_with(&calls, "fix_t"),
        0,
        "the revoked Pod's model ran"
    );
    let second = experts
        .calls()
        .iter()
        .find(|c| c.pod == pod("fix_t"))
        .expect("the second action was attempted and logged");
    assert_eq!(second.outcome, CallOutcome::IneligibleAtCallTime);
    // It is the second action of the plan whose first action (fix_m) ran just before.
    let first = experts
        .calls()
        .iter()
        .find(|c| c.pod == pod("fix_m"))
        .unwrap();
    assert_eq!(
        (first.round, first.outcome),
        (second.round, CallOutcome::Produced)
    );
    assert_eq!(second.code.as_deref(), Some(WRONG_TYPE_CODE));
    assert!(experts
        .calls()
        .iter()
        .all(|c| c.pod != pod("fix_t") || !c.produced));
}

const WRONG_TYPE_CODE: &str = support::WRONG_TYPE;

#[test]
fn a_simulated_expert_revoked_between_route_and_call_is_declined_and_draws_nothing() {
    let mut experts = simulated(SimulationConfig::new(2).with_error_rate("a", 1.0));
    let view = LiveView::with(&["gen", "fix_a", "fix_b"]);
    let mut vcw = new_loop(experts.roster().unwrap(), 3);
    struct RevokeCalled<'a, R> {
        inner: R,
        view: &'a LiveView,
        done: bool,
    }
    impl<R: RolePods<DefectDoc>> RolePods<DefectDoc> for RevokeCalled<'_, R> {
        fn propose(&mut self, call: ProposeCall<'_>) -> Option<DefectDoc> {
            self.inner.propose(call)
        }

        fn repair(&mut self, call: RepairCall<'_, DefectDoc>) -> Option<DefectDoc> {
            if !self.done {
                self.done = true;
                self.view.revoke(&call.pod.pod.0);
            }
            self.inner.repair(call)
        }
    }
    let mut interposer = RevokeCalled {
        inner: experts.guarded(&view),
        view: &view,
        done: false,
    };
    let run = vcw
        .run(
            &request("sim-late", 8, 4),
            &view,
            &DefectDocVerifier,
            &mut interposer,
        )
        .unwrap();
    // The other repairer fixed it after the first was found ineligible.
    assert!(
        matches!(run.outcome, VcwOutcome::Verified(_)),
        "{:?}",
        run.outcome
    );
    let ineligible: Vec<_> = experts
        .calls()
        .iter()
        .filter(|c| c.outcome == CallOutcome::IneligibleAtCallTime)
        .collect();
    assert_eq!(ineligible.len(), 1);
    assert!(!ineligible[0].produced);
}

// ---------------------------------------------------------------------------------------------
// Errors, bounds, identity, roster
// ---------------------------------------------------------------------------------------------

#[test]
fn a_pod_with_a_bad_token_bound_is_refused_at_construction() {
    let (generator, _calls) = shared(|_| String::new());
    for bound in [0_u32, 129, u32::MAX] {
        let built = LlmExpertPod::new(
            &project(),
            &pod("p"),
            ExpertRole::Proposer,
            TaskKind::Arithmetic,
            "s",
            bound,
            generator.clone(),
        );
        assert!(
            matches!(built, Err(GenerationError::InvalidRequest(_))),
            "{bound}"
        );
    }
    for bound in [1_u32, 128] {
        assert!(LlmExpertPod::new(
            &project(),
            &pod("p"),
            ExpertRole::Proposer,
            TaskKind::Arithmetic,
            "s",
            bound,
            generator.clone(),
        )
        .is_ok());
    }
}

#[test]
fn a_generator_error_is_recorded_as_a_failure_not_as_no_answer() {
    let runs = Arc::new(Mutex::new(0_u32));
    let generator: ptr_runtime::vcw::experts::SharedGenerator = Arc::new(Mutex::new(Broken {
        calls: runs.clone(),
    }));
    let mut experts = arithmetic_experts(&generator);
    let view = LiveView::with(&["prop", "fix_a", "fix_b"]);
    let mut vcw = new_loop(experts.roster(), 3);
    let run = run_live(
        &mut vcw,
        "broken",
        &view,
        &mut experts,
        Switch::Nothing,
        &ArithmeticVerifier { expected: 42 },
    );
    assert!(!matches!(run.outcome, VcwOutcome::Verified(_)));
    assert!(*runs.lock().unwrap() >= 1);
    let log = experts.calls();
    assert!(!log.is_empty());
    assert!(log
        .iter()
        .all(|c| { c.outcome == CallOutcome::Failed(ExpertErrorKind::Backend) && !c.produced }));
    assert_ne!(
        CallOutcome::Failed(ExpertErrorKind::Backend),
        CallOutcome::Declined
    );
}

#[test]
fn a_poisoned_generator_lock_is_a_typed_failure() {
    let (generator, _calls) = shared(|_| "42".to_owned());
    let mut experts = arithmetic_experts(&generator);
    let poisoner = generator.clone();
    let _ = std::thread::spawn(move || {
        let _guard = poisoner.lock().unwrap();
        panic!("poison the lock");
    })
    .join();
    let view = LiveView::with(&["prop", "fix_a", "fix_b"]);
    let mut vcw = new_loop(experts.roster(), 3);
    let _ = run_live(
        &mut vcw,
        "poisoned",
        &view,
        &mut experts,
        Switch::Nothing,
        &ArithmeticVerifier { expected: 42 },
    );
    assert!(experts
        .calls()
        .iter()
        .all(|c| c.outcome == CallOutcome::Failed(ExpertErrorKind::LockPoisoned)));
}

#[test]
fn the_model_digests_reach_the_manifest_and_the_call_log() {
    let calls = CallLog::default();
    let identity = ptr_pods::ModelIdentity::new(vec![
        ("weights".into(), "w".repeat(64)),
        ("config".into(), "c".repeat(64)),
        ("tokenizer".into(), "t".repeat(64)),
    ]);
    let generator: ptr_runtime::vcw::experts::SharedGenerator = Arc::new(Mutex::new(Scripted {
        reply: |_: &GenerationRequest<'_>| "42".to_owned(),
        calls: calls.clone(),
        identity: Some(identity.clone()),
    }));
    let mut experts = arithmetic_experts(&generator);
    let (_rt, registry) = runtime_with(&experts, &["prop", "fix_a", "fix_b"]);
    let manifest = registry
        .get(&project(), &pod("prop"))
        .unwrap()
        .manifest()
        .clone();
    let names: Vec<_> = manifest.capabilities.iter().map(|c| c.0.clone()).collect();
    assert!(names.contains(&"propose".to_owned()));
    assert!(names.contains(&format!("model:weights:sha256:{}", "w".repeat(64))));
    assert!(names.contains(&format!("model:config:sha256:{}", "c".repeat(64))));
    assert!(names.contains(&format!("model:tokenizer:sha256:{}", "t".repeat(64))));

    let view = LiveView::with(&["prop", "fix_a", "fix_b"]);
    let mut vcw = new_loop(experts.roster(), 3);
    let run = run_live(
        &mut vcw,
        "identity",
        &view,
        &mut experts,
        Switch::Nothing,
        &ArithmeticVerifier { expected: 42 },
    );
    assert_eq!(verified_text(&run), "42");
    let logged = experts.calls().front().unwrap();
    assert_eq!(logged.model.as_deref(), Some(&identity));
}

#[test]
fn a_pod_id_added_twice_is_refused_in_both_adapters() {
    let (generator, _calls) = shared(|_| String::new());
    let mut experts = arithmetic_experts(&generator);
    let again = Arc::new(
        LlmExpertPod::new(
            &project(),
            &pod("prop"),
            ExpertRole::Proposer,
            TaskKind::Arithmetic,
            "s",
            8,
            generator.clone(),
        )
        .unwrap(),
    );
    assert_eq!(
        experts.add_proposer(again.clone()),
        Err(RosterError::DuplicatePod(pod("prop")))
    );
    assert!(experts.add_repairer(again, &ARITHMETIC_CODES).is_err());
    assert_eq!(experts.pods().len(), 3);
    assert_eq!(experts.roster().proposers.len(), 1);
}

#[test]
fn a_regenerated_proposal_carries_an_attempt_number_but_no_finding() {
    // The proposer is declined first, so the loop asks for a fresh proposal in a later round.
    let (generator, calls) = shared(|_| "41".to_owned());
    let mut experts = arithmetic_experts(&generator);
    let view = LiveView::with(&["prop", "fix_a", "fix_b"]);
    let mut vcw = new_loop(experts.roster(), 8);
    let _ = run_live(
        &mut vcw,
        "attempts",
        &view,
        &mut experts,
        Switch::Nothing,
        &ArithmeticVerifier { expected: 42 },
    );
    let sent = calls.lock().unwrap();
    let proposals: Vec<_> = sent.iter().filter(|(s, _)| s == PROPOSER_SYSTEM).collect();
    assert!(!proposals.is_empty());
    assert!(!proposals[0].1.contains("Attempt"));
    for (_, message) in &proposals {
        assert!(!message.contains(WRONG_ANSWER) && !message.contains("score"));
    }
}
