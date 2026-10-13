//! Approximate-KV taint on the production request paths that cannot report KV reuse
//! (`run_resumable_with_pods`, `run_model_once`, `run_model_with_pods`): input taint (raw text
//! equal to tainted content, or a request id that is already tainted) is recorded taint-first
//! before the run is returned and before any Pod output of the request is promoted; clean input
//! appends nothing.

mod common;

use common::{fixture, session, Probe};
use ptr_config::PtrConfig;
use ptr_core::action_head::ActionIr;
use ptr_ledger::LedgerEvent;
use ptr_model_api::{
    ByteTokenizer, InferenceBackend, KvReuseMode, KvReusePolicy, KvReuseReport, KvReuseTaint,
    ModelError, ModelEvent, ModelRequest, ModelResumeRequest, ModelRun, PromptSegment,
    ResumableInferenceBackend, SegmentedModelRequest, Tokenizer as _,
};
use ptr_pods::{DynPod, PodManifest, PodRegistry};
use ptr_protocol::TypedPayload;
use ptr_runtime::execution::ExecutionError;
use ptr_runtime::kv_taint::KvTaintRequest;
use ptr_runtime::{PtrRuntime, RuntimeError};
use ptr_types::{
    CapabilityId, CapsuleId, Effect, Generation, PodId, Probability, ProjectId, RequestId,
    Revision, SemanticContext, TypeId, VerificationLevel,
};
use ptr_verifier::{VerificationReport, VerificationStatus, Verifier};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// The text the fixture's verifier accepts as an action payload.
const TEXT: &str = "verified payload";
const TTL: Duration = Duration::from_secs(60);

fn project() -> ProjectId {
    ProjectId::from("p")
}

/// Proposes the action `template` with the raw text as payload, at the revision it was given.
struct Proposing {
    template: ActionIr,
}

impl Proposing {
    fn action(&self, revision: Revision, payload: &[u8]) -> ActionIr {
        ActionIr {
            revision,
            payload: payload.to_vec(),
            ..self.template.clone()
        }
    }
}

impl InferenceBackend for Proposing {
    fn name(&self) -> &'static str {
        "proposing-echo"
    }

    fn infer(&self, request: &ModelRequest) -> Result<Vec<ModelEvent>, ModelError> {
        Ok(vec![
            ModelEvent::Token(request.raw_text.clone()),
            ModelEvent::ActionReady(self.action(
                request.semantic_context.revision,
                request.raw_text.as_bytes(),
            )),
        ])
    }
}

impl ResumableInferenceBackend for Proposing {
    fn resume(&self, _: &ModelResumeRequest) -> Result<Vec<ModelEvent>, ModelError> {
        Ok(vec![ModelEvent::Finished])
    }
}

/// Asks the echo Pod for `payload`, then proposes an action whose payload is what the Pod
/// returned.
struct PodAsking {
    template: ActionIr,
    payload: Vec<u8>,
}

impl InferenceBackend for PodAsking {
    fn name(&self) -> &'static str {
        "pod-asking"
    }

    fn infer(&self, _: &ModelRequest) -> Result<Vec<ModelEvent>, ModelError> {
        Ok(vec![ModelEvent::PodRequested {
            capability: CapabilityId::from("Echo<Text>"),
            input_type: TypeId::from("Text"),
            payload: self.payload.clone(),
        }])
    }
}

impl ResumableInferenceBackend for PodAsking {
    fn resume(&self, request: &ModelResumeRequest) -> Result<Vec<ModelEvent>, ModelError> {
        Ok(vec![ModelEvent::ActionReady(ActionIr {
            revision: request.revision(),
            payload: request.observation.payload.clone(),
            ..self.template.clone()
        })])
    }
}

struct Echo {
    manifest: PodManifest,
}

impl DynPod for Echo {
    fn manifest(&self) -> &PodManifest {
        &self.manifest
    }

    fn invoke(&self, input: TypedPayload) -> Result<TypedPayload, String> {
        Ok(input)
    }
}

fn registry() -> PodRegistry {
    let mut registry = PodRegistry::default();
    registry.register(Arc::new(Echo {
        manifest: PodManifest {
            project: project(),
            id: PodId::from("echo"),
            capabilities: vec![CapabilityId::from("Echo<Text>")],
            accepts: vec![TypeId::from("Text")],
            produces: vec![TypeId::from("Text")],
            effects: vec![Effect::Pure],
            protocol_version: 1,
        },
    }));
    registry
}

/// Passing verifier that counts how often it was asked.
#[derive(Clone, Default)]
struct Counting(Arc<AtomicUsize>);

impl Counting {
    fn calls(&self) -> usize {
        self.0.load(Ordering::SeqCst)
    }
}

impl Verifier<TypedPayload> for Counting {
    fn verify(&self, _: &TypedPayload) -> VerificationReport {
        self.0.fetch_add(1, Ordering::SeqCst);
        VerificationReport {
            status: VerificationStatus::Pass,
            level: VerificationLevel::Deterministic,
            score: Probability::new(1.0).unwrap(),
            findings: vec![],
        }
    }
}

fn segmented(id: &str) -> SegmentedModelRequest {
    let tokenizer = ByteTokenizer::new();
    let segment = PromptSegment::raw_from_text(&tokenizer, "seed").unwrap();
    SegmentedModelRequest::new(
        RequestId::from(id),
        SemanticContext {
            revision: Revision(0),
            entries: Vec::new(),
        },
        tokenizer.identity().clone(),
        vec![segment],
    )
    .unwrap()
}

/// Record an Approximate run of request `id` whose generated text is `text` and whose action
/// is `other` (never the action the tests later propose, so only the INPUT taint can refuse
/// that one).
fn taint_run(runtime: &mut PtrRuntime, id: &str, text: &str, other: &ActionIr) {
    let request = segmented(id);
    let run = ModelRun {
        events: vec![
            ModelEvent::Token(text.to_owned()),
            ModelEvent::ActionReady(other.clone()),
        ],
        kv_report: Some(
            KvReuseReport::new(
                4,
                KvReuseMode::Approximate,
                8,
                KvReusePolicy::AllowApproximate,
            )
            .unwrap(),
        ),
    };
    let taint = runtime
        .ingest_model_run_for(
            &KvTaintRequest {
                request: &request,
                producer: "test",
                generation_parameters: b"",
            },
            &run,
        )
        .unwrap();
    assert!(taint.kv_taint.is_approximate());
}

fn other_action(action: &ActionIr) -> ActionIr {
    ActionIr {
        payload: b"some other action".to_vec(),
        ..action.clone()
    }
}

fn taint_records(runtime: &PtrRuntime) -> usize {
    runtime
        .committed_events()
        .iter()
        .filter(|committed| matches!(committed.event, LedgerEvent::KvTaintCommitted { .. }))
        .count()
}

fn effect_attempts(runtime: &PtrRuntime) -> usize {
    runtime
        .committed_events()
        .iter()
        .filter(|committed| matches!(committed.event, LedgerEvent::EffectAttempted { .. }))
        .count()
}

fn is_denied(result: Result<(), RuntimeError>) -> bool {
    matches!(
        result,
        Err(RuntimeError::ApproximateStateAtEffectBoundary { .. })
    )
}

fn is_execution_denial(error: &ExecutionError) -> bool {
    matches!(
        error,
        ExecutionError::Audit(inner)
            if matches!(**inner, RuntimeError::ApproximateStateAtEffectBoundary { .. })
    )
}

fn action_of(events: &[ModelEvent]) -> ActionIr {
    events
        .iter()
        .find_map(|event| match event {
            ModelEvent::ActionReady(action) => Some(action.clone()),
            _ => None,
        })
        .expect("the backend proposed an action")
}

/// The action is refused at `authorize_action` and at `prepare_execution_once`, and nothing is
/// attempted or executed.
fn assert_refused_everywhere(runtime: &mut PtrRuntime, action: &ActionIr) {
    let probe = Probe::default();
    let session = session(runtime, action, &probe, "alice");
    assert!(is_denied(runtime.authorize_action(action)));
    let denied = runtime
        .prepare_execution_once(&session, &project(), action, TTL, "k")
        .unwrap_err();
    assert!(is_execution_denial(&denied), "{denied:?}");
    assert_eq!(effect_attempts(runtime), 0);
    assert_eq!(probe.executions(), 0);
}

#[test]
fn tainted_raw_text_taints_the_resumable_action() {
    let (mut runtime, action) = fixture();
    let backend = Proposing {
        template: action.clone(),
    };
    // Positive control for the fixture: the very action is fine before anything is tainted.
    taint_run(&mut runtime, "source", TEXT, &other_action(&action));
    let before = taint_records(&runtime);

    let run = runtime
        .run_resumable_with_pods(
            "r2".into(),
            &project(),
            TEXT,
            &backend,
            &registry(),
            &Counting::default(),
            2,
        )
        .unwrap();
    let proposed = run.action.clone().expect("action proposed");
    assert!(run.kv_taint.is_approximate());
    // A taint record was appended before the run was returned.
    assert!(taint_records(&runtime) > before);
    assert!(runtime.action_kv_taint(&proposed).is_approximate());
    assert_refused_everywhere(&mut runtime, &proposed);
}

#[test]
fn tainted_raw_text_taints_run_model_once() {
    let (mut runtime, action) = fixture();
    let backend = Proposing {
        template: action.clone(),
    };
    taint_run(&mut runtime, "source", TEXT, &other_action(&action));
    let before = taint_records(&runtime);

    let events = runtime.run_model_once("r2".into(), TEXT, &backend).unwrap();
    assert!(taint_records(&runtime) > before);
    let proposed = action_of(&events);
    assert!(runtime.action_kv_taint(&proposed).is_approximate());
    assert_refused_everywhere(&mut runtime, &proposed);
}

#[test]
fn tainted_raw_text_refuses_run_model_with_pods_outputs() {
    let (mut runtime, action) = fixture();
    let backend = PodAsking {
        template: action.clone(),
        payload: b"clean pod payload".to_vec(),
    };
    taint_run(&mut runtime, "source", TEXT, &other_action(&action));
    let verifier = Counting::default();
    let revision = runtime.revision();

    let refused = runtime
        .run_model_with_pods(
            "r2".into(),
            &project(),
            TEXT,
            &backend,
            &registry(),
            &verifier,
        )
        .unwrap_err();
    assert!(
        matches!(refused, RuntimeError::ApproximateSemanticContent { .. }),
        "{refused:?}"
    );
    assert_eq!(verifier.calls(), 0);
    // Only the request's own text ingress moved the revision; no Pod output was promoted.
    assert_eq!(runtime.revision(), Revision(revision.0 + 1));
    assert_eq!(effect_attempts(&runtime), 0);
}

#[test]
fn a_reused_tainted_request_id_taints_the_resumable_action() {
    let (mut runtime, action) = fixture();
    let backend = Proposing {
        template: action.clone(),
    };
    taint_run(
        &mut runtime,
        "reused",
        "unrelated text",
        &other_action(&action),
    );
    let before = taint_records(&runtime);

    let run = runtime
        .run_resumable_with_pods(
            "reused".into(),
            &project(),
            TEXT,
            &backend,
            &registry(),
            &Counting::default(),
            2,
        )
        .unwrap();
    let proposed = run.action.clone().expect("action proposed");
    assert!(run.kv_taint.is_approximate());
    assert!(taint_records(&runtime) > before);
    assert_refused_everywhere(&mut runtime, &proposed);

    // The same holds for `run_model_once` on that id.
    let events = runtime
        .run_model_once("reused".into(), TEXT, &backend)
        .unwrap();
    assert!(runtime
        .action_kv_taint(&action_of(&events))
        .is_approximate());
}

#[test]
fn clean_input_appends_nothing_and_the_action_is_authorized() {
    let (mut runtime, action) = fixture();
    let backend = Proposing {
        template: action.clone(),
    };
    let before = runtime.committed_events().len();
    let run = runtime
        .run_resumable_with_pods(
            "fresh".into(),
            &project(),
            TEXT,
            &backend,
            &registry(),
            &Counting::default(),
            2,
        )
        .unwrap();
    assert_eq!(run.kv_taint, KvReuseTaint::Exact);
    assert_eq!(taint_records(&runtime), 0);
    // Only the text ingress was appended.
    assert!(runtime.committed_events().len() > before);
    let proposed = run.action.expect("action proposed");
    assert!(runtime.authorize_action(&proposed).is_ok());

    let probe = Probe::default();
    let session = session(&mut runtime, &proposed, &probe, "alice");
    let permit = runtime
        .prepare_execution_once(&session, &project(), &proposed, TTL, "k")
        .unwrap();
    runtime.execute_prepared(&session, permit).unwrap();
    assert_eq!(probe.executions(), 1);

    let events = runtime
        .run_model_once("fresh2".into(), TEXT, &backend)
        .unwrap();
    assert_eq!(taint_records(&runtime), 0);
    assert!(runtime.authorize_action(&action_of(&events)).is_ok());
}

#[test]
fn clean_input_through_a_pod_round_stays_exact() {
    let (mut runtime, action) = fixture();
    let backend = PodAsking {
        template: action.clone(),
        payload: TEXT.as_bytes().to_vec(),
    };
    let run = runtime
        .run_resumable_with_pods(
            "fresh".into(),
            &project(),
            "ask the pod",
            &backend,
            &registry(),
            &Counting::default(),
            2,
        )
        .unwrap();
    assert_eq!(run.kv_taint, KvReuseTaint::Exact);
    assert_eq!(run.observations.len(), 1);
    assert_eq!(taint_records(&runtime), 0);
    assert!(runtime
        .authorize_action(&run.action.expect("action after the pod round"))
        .is_ok());
}

#[test]
fn tainted_input_refuses_the_pod_output_before_it_is_promoted() {
    let (mut runtime, action) = fixture();
    // The Pod payload is clean; only the request's raw text is tainted.
    let backend = PodAsking {
        template: action.clone(),
        payload: b"clean pod payload".to_vec(),
    };
    taint_run(
        &mut runtime,
        "source",
        "tainted request text",
        &other_action(&action),
    );
    let verifier = Counting::default();
    let revision = runtime.revision();

    let refused = runtime
        .run_resumable_with_pods(
            "r2".into(),
            &project(),
            "tainted request text",
            &backend,
            &registry(),
            &verifier,
            2,
        )
        .unwrap_err();
    assert!(
        matches!(refused, RuntimeError::ApproximateSemanticContent { .. }),
        "{refused:?}"
    );
    // Refused before the verifier ran; only the text ingress moved the revision.
    assert_eq!(verifier.calls(), 0);
    assert_eq!(runtime.revision(), Revision(revision.0 + 1));
}

#[test]
fn a_pod_payload_equal_to_tainted_stored_content_is_refused_in_the_loop() {
    let (mut runtime, action) = fixture();
    let backend = PodAsking {
        template: action.clone(),
        payload: b"tainted candidate".to_vec(),
    };
    taint_run(
        &mut runtime,
        "source",
        "tainted candidate",
        &other_action(&action),
    );
    let verifier = Counting::default();

    // Clean raw text and a fresh id: the Pod output itself is the tainted content.
    let refused = runtime
        .run_resumable_with_pods(
            "r2".into(),
            &project(),
            "unrelated clean text",
            &backend,
            &registry(),
            &verifier,
            2,
        )
        .unwrap_err();
    assert!(
        matches!(refused, RuntimeError::ApproximateSemanticContent { .. }),
        "{refused:?}"
    );
    assert_eq!(verifier.calls(), 0);
}

// The Pod bindings (`knowledge`/`artifact`/`reader`/`adapter`) of a Pod in the resume loop are
// not reachable: `DynPod` exposes only a `PodManifest`, which carries none. A Pod bound to
// tainted stored content is therefore only caught here when its output equals that content (the
// test above); the binding join exists on `admit_pod_output` (tests/kv_taint_pods.rs) and, as a
// crate-internal entry, on `promote_verified_pod_output` (unit test in kv_taint.rs).

struct Temp(PathBuf);

impl Temp {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "ptr-kv-taint-prod-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&dir).unwrap();
        Self(dir)
    }

    fn log(&self) -> PathBuf {
        self.0.join("ledger")
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn arm(runtime: &mut PtrRuntime, action: &ActionIr) {
    runtime
        .permissions_mut()
        .capabilities
        .insert(action.capability.clone());
    runtime.permissions_mut().allow_mutation = true;
}

#[test]
fn taint_survives_a_restart_between_the_tainting_run_and_the_resumable_run() {
    let temp = Temp::new();
    let template = ActionIr {
        operation: "write".into(),
        target: "capsule:a".into(),
        capability: CapabilityId::from("file.write"),
        effect: Effect::Mutation,
        input_type: TypeId::from("Bytes"),
        generation: Generation(1),
        revision: Revision(0),
        payload: Vec::new(),
    };
    {
        let mut runtime = PtrRuntime::open_durable(PtrConfig::default(), temp.log()).unwrap();
        runtime
            .commit(LedgerEvent::CapsuleCommitted {
                project: project(),
                capsule: CapsuleId::from("capsule:a"),
                generation: Generation(1),
            })
            .unwrap();
        taint_run(&mut runtime, "source", TEXT, &other_action(&template));
    }
    let mut runtime = PtrRuntime::open_durable(PtrConfig::default(), temp.log()).unwrap();
    arm(&mut runtime, &template);
    let backend = Proposing { template };

    let run = runtime
        .run_resumable_with_pods(
            "after-restart".into(),
            &project(),
            TEXT,
            &backend,
            &registry(),
            &Counting::default(),
            2,
        )
        .unwrap();
    assert!(run.kv_taint.is_approximate());
    let proposed = run.action.expect("action proposed");
    assert!(is_denied(runtime.authorize_action(&proposed)));
    // The taint stays across a second restart as well.
    drop(runtime);
    let mut runtime = PtrRuntime::open_durable(PtrConfig::default(), temp.log()).unwrap();
    arm(&mut runtime, &proposed);
    assert!(runtime.action_kv_taint(&proposed).is_approximate());
    assert!(is_denied(runtime.authorize_action(&proposed)));
}
