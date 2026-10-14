//! The verifier-closed workspace loop as a request path of the server, with simulated experts.
//!
//! Each test builds a server whose `[vcw]` section is enabled, posts tasks to `/v1/vcw/tasks`
//! and checks the typed outcome, the ledger and the runtime's events. The experts are the seeded
//! programmatic ones, so every run is replayable bit for bit.

use std::sync::mpsc::channel;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::{header, Request, StatusCode};
use axum::Router;
use ptr_config::{PtrConfig, VcwLearning, VcwServiceConfig, VcwTaskKind};
use ptr_events::RuntimeEvent;
use ptr_ledger::{LedgerEvent, SemanticOrigin};
use ptr_runtime::execution::RequiredVerification;
use ptr_runtime::{ExecutionScope, ScopeState};
use ptr_runtime::{PtrRuntime, SemanticChange, SemanticGrant};
use ptr_semdb::SemanticDelta;
use ptr_server::{router_from_state, ServerState, VcwBuildError, VcwService, VcwServiceParts};
use ptr_types::{
    CapsuleId, Generation, PodId, PrincipalId, Probability, ProjectId, ScopeId, ScopeLeaseBinding,
    SessionId, Timestamp, VerificationLevel,
};
use ptr_verifier::{fact_key, NamedVerifier, VerificationReport, VerificationStatus, Verifier};
use serde_json::{json, Value};
use tower::ServiceExt;

const SQUARE: &str = "fn f(a0) { return a0 * a0; }";

fn enabled() -> VcwServiceConfig {
    let mut config = VcwServiceConfig {
        enabled: true,
        ..VcwServiceConfig::default()
    };
    config.experts.backend = Some(ptr_config::VcwBackend::Simulated);
    config
}

fn with_errors(mut config: VcwServiceConfig, errors: &[(&str, f64)]) -> VcwServiceConfig {
    for (code, rate) in errors {
        config
            .experts
            .simulated
            .error_rates
            .insert((*code).to_owned(), *rate);
    }
    config
}

fn server_from(runtime: PtrRuntime, config: &VcwServiceConfig) -> (ServerState, Router) {
    let service = VcwService::from_config(config, &VcwServiceParts::new([7; 32], [9; 8])).unwrap();
    let state = ServerState::new(runtime).with_vcw(service).unwrap();
    let app = router_from_state(state.clone());
    (state, app)
}

fn server(config: &VcwServiceConfig) -> (ServerState, Router) {
    server_from(PtrRuntime::new(PtrConfig::default()).unwrap(), config)
}

async fn send(app: &Router, request: Request<Body>) -> (StatusCode, Vec<u8>) {
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .unwrap();
    (status, bytes.to_vec())
}

async fn post(app: &Router, body: &Value) -> (StatusCode, Value) {
    let request = Request::post("/v1/vcw/tasks")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let (status, bytes) = send(app, request).await;
    let value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, value)
}

fn arithmetic(id: &str) -> Value {
    json!({"id": id, "task": {"kind": "arithmetic", "expression": "(3 + 4) * 5", "reference": "35"}})
}

fn admitted_for(runtime: &PtrRuntime, id: &str) -> Vec<String> {
    runtime
        .committed_events()
        .iter()
        .filter_map(|committed| match &committed.event {
            LedgerEvent::PodOutputAdmitted {
                request_id, pod_id, ..
            } if request_id.0 == id => Some(pod_id.0.clone()),
            _ => None,
        })
        .collect()
}

fn finished_events(runtime: &PtrRuntime, id: &str) -> Vec<(String, String)> {
    runtime
        .events()
        .iter()
        .filter_map(|envelope| match &envelope.event {
            RuntimeEvent::VcwFinished {
                request,
                outcome,
                trace_digest,
            } if request.0 == id => Some((
                outcome.clone(),
                trace_digest.iter().map(|b| format!("{b:02x}")).collect(),
            )),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn an_arithmetic_answer_that_is_wrong_is_repaired_admitted_and_committed() {
    let config = with_errors(enabled(), &[("arith.value", 1.0)]);
    let (state, app) = server(&config);
    let (status, body) = post(&app, &arithmetic("arith-1")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["outcome"], "verified");
    assert_eq!(body["answer"], "35");
    assert_eq!(body["answer_type"], "vcw.answer.arithmetic");
    assert_eq!(body["rounds"], 1, "one repair round");
    assert_eq!(
        body["verifier_calls"], 2,
        "the wrong candidate and the repaired one"
    );
    let authors: Vec<&str> = body["authors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a["pod"].as_str().unwrap())
        .collect();
    // The repair is a confined edit of the proposal: the proposer's output is still in the
    // answer, so both are its authors, the repairer last (issue #141).
    assert_eq!(authors, ["vcw-proposer", "vcw-value-repair"]);
    let digest = body["trace_digest"].as_str().unwrap().to_owned();
    assert_eq!(digest.len(), 64);

    // The answer reached the ledger through the runtime's output admission, and the trace digest
    // is both a runtime event and part of the manifest the admission committed.
    let handle = state.runtime_handle();
    let runtime = handle.lock().unwrap();
    assert_eq!(admitted_for(&runtime, "arith-1"), ["vcw-value-repair"]);
    assert_eq!(
        finished_events(&runtime, "arith-1"),
        [("verified".to_owned(), digest)]
    );
    let committed = &body["committed"];
    let first = committed["first_index"].as_u64().unwrap();
    let count = committed["count"].as_u64().unwrap();
    assert!(count >= 3, "raw text, admission and candidate: {committed}");
    let in_range: Vec<_> = runtime
        .committed_events()
        .iter()
        .filter(|e| e.index.0 >= first && e.index.0 < first + count)
        .collect();
    assert_eq!(in_range.len() as u64, count);
    assert!(in_range.iter().any(|e| matches!(
        &e.event,
        LedgerEvent::SemanticDeltaCommitted {
            origin: SemanticOrigin::PodCandidate { request, pod, .. }, ..
        } if request == "arith-1" && pod == "vcw-value-repair"
    )));
    drop(runtime);

    let calls = state.vcw_service().unwrap().expert_calls();
    let roles: Vec<_> = calls
        .iter()
        .map(|c| (c.pod.0.as_str(), c.produced))
        .collect();
    assert_eq!(roles, [("vcw-proposer", true), ("vcw-value-repair", true)]);
}

#[tokio::test]
async fn a_clean_first_proposal_needs_no_repair() {
    let (_state, app) = server(&enabled());
    let (status, body) = post(&app, &arithmetic("clean-1")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["rounds"], 0);
    assert_eq!(body["verifier_calls"], 1);
    assert_eq!(body["authors"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn a_json_schema_violation_is_repaired_by_the_schema_role() {
    let config = with_errors(enabled(), &[("json.required", 1.0)]);
    let (state, app) = server(&config);
    let task = json!({"id": "json-1", "task": {
        "kind": "json_object",
        "instruction": "Describe a person.",
        "schema": {"type": "object", "required": ["age", "name"],
                   "properties": {"age": {"type": "integer"}, "name": {"type": "string"}}},
        "reference": {"name": "Ada", "age": 36}
    }});
    let (status, body) = post(&app, &task).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["outcome"], "verified");
    let answer: Value = serde_json::from_str(body["answer"].as_str().unwrap()).unwrap();
    assert_eq!(answer, json!({"name": "Ada", "age": 36}));
    assert_eq!(body["authors"][0]["pod"], "vcw-proposer");
    assert_eq!(body["authors"][1]["pod"], "vcw-schema-repair");
    let calls = state.vcw_service().unwrap().expert_calls();
    assert_eq!(calls[1].code.as_deref(), Some("json.required"));
}

#[tokio::test]
async fn a_failing_code_test_is_repaired_and_a_hang_goes_to_the_resource_role() {
    for (defect, role) in [
        ("code.test_failed", "vcw-code-repair"),
        ("code.syntax", "vcw-code-repair"),
        ("code.timeout", "vcw-resource-repair"),
        ("code.resource", "vcw-resource-repair"),
    ] {
        let config = with_errors(enabled(), &[(defect, 1.0)]);
        let (_state, app) = server(&config);
        let task = json!({"id": "code-1", "task": {
            "kind": "code", "instruction": "Square the argument.", "entry": "f",
            "tests": [{"id": "zero", "args": [0], "expected": 0},
                      {"id": "three", "args": [3], "expected": 9},
                      {"id": "neg", "args": [-4], "expected": 16}],
            "reference": SQUARE
        }});
        let (status, body) = post(&app, &task).await;
        assert_eq!(status, StatusCode::OK, "{defect}: {body}");
        assert_eq!(body["answer"], SQUARE, "{defect}");
        assert_eq!(body["authors"][0]["pod"], "vcw-proposer", "{defect}");
        assert_eq!(body["authors"][1]["pod"], role, "{defect}");
    }
}

#[tokio::test]
async fn an_unrepairable_task_is_exhausted_and_commits_no_answer() {
    let mut config = with_errors(enabled(), &[("arith.value", 1.0)]);
    config.experts.simulated.default_repair_success = 0.0;
    let (state, app) = server(&config);
    let (status, body) = post(&app, &arithmetic("hopeless")).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(body["outcome"], "exhausted");
    assert!(
        body.get("answer").is_none(),
        "no candidate is returned: {body}"
    );
    assert!(body["verifier_calls"].as_u64().unwrap() >= 1);
    let handle = state.runtime_handle();
    let runtime = handle.lock().unwrap();
    assert!(admitted_for(&runtime, "hopeless").is_empty());
    let finished = finished_events(&runtime, "hopeless");
    assert_eq!(finished.len(), 1);
    assert_eq!(finished[0].0, "exhausted");
    assert_eq!(finished[0].1, body["trace_digest"].as_str().unwrap());
}

#[tokio::test]
async fn a_reference_that_is_itself_wrong_never_passes() {
    let (_state, app) = server(&enabled());
    let task = json!({"id": "bad-ref", "task": {
        "kind": "arithmetic", "expression": "(3 + 4) * 5", "reference": "36"}});
    let (status, body) = post(&app, &task).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(body["outcome"], "exhausted");
    assert!(body.get("answer").is_none());
}

fn fact_task(id: &str) -> Value {
    json!({"id": id, "task": {
        "kind": "fact", "question": "What is the capital of France?",
        "claims": [{"subject": "paris", "attribute": "capital-of", "value": "france"}]}})
}

#[tokio::test]
async fn an_unknown_fact_is_regenerated_or_escalated_and_never_accepted() {
    let (state, app) = server(&enabled());
    let (status, body) = post(&app, &fact_task("fact-unknown")).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert!(
        matches!(body["outcome"].as_str(), Some("exhausted" | "escalated")),
        "{body}"
    );
    assert!(body.get("answer").is_none());
    let handle = state.runtime_handle();
    assert!(admitted_for(&handle.lock().unwrap(), "fact-unknown").is_empty());
    // Unknown is not a repair target: only the proposer was ever asked, and more than once.
    let calls = state.vcw_service().unwrap().expert_calls();
    assert!(calls.len() >= 2, "a fresh proposal was drawn: {calls:?}");
    assert!(calls.iter().all(|c| c.pod.0 == "vcw-proposer"), "{calls:?}");
}

#[derive(Clone)]
struct AllowHostWrites;

impl<'a> Verifier<SemanticChange<'a>> for AllowHostWrites {
    fn verify(&self, _: &SemanticChange<'a>) -> VerificationReport {
        VerificationReport {
            status: VerificationStatus::Pass,
            level: VerificationLevel::Deterministic,
            score: Probability::new(1.0).unwrap(),
            findings: vec![],
        }
    }
}

impl<'a> NamedVerifier<SemanticChange<'a>> for AllowHostWrites {
    fn name(&self) -> &'static str {
        "host-fixture"
    }
}

#[tokio::test]
async fn a_contradicted_fact_is_repaired_against_the_committed_state() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime
        .install_semantic_grant(
            SemanticGrant::new(RequiredVerification::Deterministic)
                .with_verifier(AllowHostWrites)
                .allow_host_writes(),
        )
        .unwrap();
    let mut delta = SemanticDelta::default();
    delta
        .upserts
        .insert(fact_key("paris", "capital-of"), "france".into());
    let revision = runtime.revision();
    runtime
        .apply_verified_semantic_delta(revision, delta, &PrincipalId::from("ops"))
        .unwrap();

    let config = with_errors(enabled(), &[("fact.contradicted", 1.0)]);
    let (state, app) = server_from(runtime, &config);
    let (status, body) = post(&app, &fact_task("fact-1")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["answer_type"], "vcw.answer.fact_claims");
    assert_eq!(body["authors"][0]["pod"], "vcw-proposer");
    assert_eq!(body["authors"][1]["pod"], "vcw-fact-repair");
    let claims: Value = serde_json::from_str(body["answer"].as_str().unwrap()).unwrap();
    assert_eq!(claims[0]["value"], "france");
    let handle = state.runtime_handle();
    assert_eq!(
        admitted_for(&handle.lock().unwrap(), "fact-1"),
        ["vcw-fact-repair"]
    );
}

fn revoke(state: &ServerState, pod: &str) {
    let handle = state.runtime_handle();
    handle
        .lock()
        .unwrap()
        .commit(LedgerEvent::Revoked {
            subject: pod.to_owned(),
            generation: Generation(1),
        })
        .unwrap();
}

#[tokio::test]
async fn a_pod_revoked_before_the_request_is_never_called() {
    // The proposer: nobody can propose, so the request is escalated and no Pod runs.
    let (state, app) = server(&enabled());
    revoke(&state, "vcw-proposer");
    let (status, body) = post(&app, &arithmetic("revoked-proposer")).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(body["outcome"], "escalated");
    assert!(
        body["reason"].as_str().unwrap().contains("NoProposer"),
        "{body}"
    );
    assert!(state.vcw_service().unwrap().expert_calls().is_empty());
    let handle = state.runtime_handle();
    assert!(admitted_for(&handle.lock().unwrap(), "revoked-proposer").is_empty());

    // A repairer: the wrong candidate cannot be repaired by it, so it is not called and the
    // request is not verified.
    let config = with_errors(enabled(), &[("arith.value", 1.0)]);
    let (state, app) = server(&config);
    revoke(&state, "vcw-value-repair");
    let (status, body) = post(&app, &arithmetic("revoked-repairer")).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_ne!(body["outcome"], "verified");
    let calls = state.vcw_service().unwrap().expert_calls();
    assert!(!calls.is_empty());
    assert!(
        calls.iter().all(|c| c.pod.0 != "vcw-value-repair"),
        "{calls:?}"
    );
}

#[tokio::test]
async fn a_disabled_section_has_no_endpoint_and_the_old_paths_are_unchanged() {
    // Off: the plain router has no such route.
    let plain = ptr_server::router(PtrRuntime::new(PtrConfig::default()).unwrap());
    let (status, _) = post(&plain, &arithmetic("x")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(matches!(
        VcwService::from_config(
            &VcwServiceConfig::default(),
            &VcwServiceParts::new([1; 32], [2; 8])
        ),
        Err(VcwBuildError::Config(_))
    ));

    // On: the existing routes answer byte for byte as they do without it.
    let (_state, with_vcw) = server(&enabled());
    let legacy = |app: &Router, method: &str, path: &str, body: &'static str| {
        let request = Request::builder()
            .method(method)
            .uri(path)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .unwrap();
        let app = app.clone();
        async move { send(&app, request).await }
    };
    for (method, path, body) in [
        ("GET", "/health", ""),
        ("POST", "/v1/requests", r#"{"id":"r1","text":"hello"}"#),
        ("POST", "/v1/requests", r#"{"id":"","text":""}"#),
    ] {
        let a = legacy(&plain, method, path, body).await;
        let b = legacy(&with_vcw, method, path, body).await;
        assert_eq!(a, b, "{method} {path}");
    }
}

#[tokio::test]
async fn invalid_input_is_a_typed_4xx() {
    let config = VcwServiceConfig {
        kinds: vec![VcwTaskKind::Arithmetic],
        ..enabled()
    };
    let (_state, app) = server(&config);
    let cases: Vec<(Value, StatusCode)> = vec![
        // Blank id, blank expression.
        (
            json!({"id": " ", "task": {"kind": "arithmetic", "expression": "1+1", "reference": "2"}}),
            StatusCode::BAD_REQUEST,
        ),
        (
            json!({"id": "a", "task": {"kind": "arithmetic", "expression": "  ", "reference": "2"}}),
            StatusCode::BAD_REQUEST,
        ),
        // The simulated backend needs its reference.
        (
            json!({"id": "a", "task": {"kind": "arithmetic", "expression": "1+1"}}),
            StatusCode::BAD_REQUEST,
        ),
        // A kind that is not served.
        (fact_task("a"), StatusCode::BAD_REQUEST),
        // Unknown field, unknown kind.
        (
            json!({"id": "a", "task": {"kind": "arithmetic", "expression": "1+1", "reference": "2", "x": 1}}),
            StatusCode::UNPROCESSABLE_ENTITY,
        ),
        (
            json!({"id": "a", "task": {"kind": "poetry"}}),
            StatusCode::UNPROCESSABLE_ENTITY,
        ),
        // A bound exceeded.
        (
            json!({"id": "a", "task": {"kind": "arithmetic", "expression": "1".repeat(513), "reference": "2"}}),
            StatusCode::PAYLOAD_TOO_LARGE,
        ),
        (
            json!({"id": "i".repeat(300), "task": {"kind": "arithmetic", "expression": "1+1", "reference": "2"}}),
            StatusCode::PAYLOAD_TOO_LARGE,
        ),
    ];
    for (body, expected) in cases {
        let (status, _) = post(&app, &body).await;
        assert_eq!(status, expected, "{body}");
    }
    // A body over the endpoint's own limit is refused before it is read.
    let huge = json!({"id": "a", "task": {"kind": "arithmetic",
        "expression": "1".repeat(300 * 1024), "reference": "2"}});
    let (status, _) = post(&app, &huge).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    // Nothing above reached the ledger.
    let (status, body) = post(&app, &arithmetic("after-errors")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

#[tokio::test]
async fn a_request_id_is_used_once() {
    let (_state, app) = server(&enabled());
    assert_eq!(post(&app, &arithmetic("same")).await.0, StatusCode::OK);
    assert_eq!(
        post(&app, &arithmetic("same")).await.0,
        StatusCode::CONFLICT
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_saturated_mailbox_is_refused_with_503_and_recovers() {
    let mut core = PtrConfig::default();
    core.runtime.mailbox_capacity = 1;
    let (state, app) = server_from(PtrRuntime::new(core).unwrap(), &enabled());

    // Hold the runtime so the first request waits inside its admission slot.
    let (locked_tx, locked_rx) = channel();
    let (release_tx, release_rx) = channel::<()>();
    let handle = state.runtime_handle();
    let holder = std::thread::spawn(move || {
        let _guard = handle.lock().unwrap();
        locked_tx.send(()).unwrap();
        let _ = release_rx.recv_timeout(Duration::from_secs(30));
    });
    locked_rx.recv_timeout(Duration::from_secs(30)).unwrap();

    let first = {
        let app = app.clone();
        tokio::spawn(async move { post(&app, &arithmetic("slot-1")).await })
    };
    for _ in 0..300 {
        if state.requests_in_flight() == 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(state.requests_in_flight(), 1);

    let request = Request::post("/v1/vcw/tasks")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(arithmetic("slot-2").to_string()))
        .unwrap();
    let refused = app.clone().oneshot(request).await.unwrap();
    assert_eq!(refused.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(refused.headers()[header::RETRY_AFTER], "1");

    release_tx.send(()).unwrap();
    holder.join().unwrap();
    let (status, body) = first.await.unwrap();
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(post(&app, &arithmetic("slot-3")).await.0, StatusCode::OK);
}

#[tokio::test]
async fn the_response_is_determined_by_the_seed() {
    async fn run(seed: u32) -> Vec<(StatusCode, Value)> {
        let mut config = with_errors(enabled(), &[("arith.value", 0.6), ("arith.format", 0.4)]);
        config.experts.simulated.seed = seed;
        config.experts.simulated.default_repair_success = 0.5;
        config.max_verifier_calls = 8;
        config.max_rounds = 6;
        let (_state, app) = server(&config);
        let mut out = Vec::new();
        for i in 0..12 {
            out.push(post(&app, &arithmetic(&format!("det-{i}"))).await);
        }
        out
    }
    let first = run(5).await;
    assert_eq!(first, run(5).await, "same seed, same responses");
    assert_ne!(first, run(6).await, "another seed draws differently");
    // The draws matter: not every request is a first-try success, and not every one fails.
    let verified = first.iter().filter(|(s, _)| *s == StatusCode::OK).count();
    assert!(first
        .iter()
        .any(|(_, b)| b["rounds"].as_u64().unwrap_or(0) > 0));
    assert!(verified > 0);
}

#[test]
fn a_misconfigured_simulation_is_refused_at_build_time() {
    let parts = VcwServiceParts::new([1; 32], [2; 8]);
    // A code no kind can plant.
    let config = with_errors(enabled(), &[("arith.nonsense", 0.5)]);
    assert!(matches!(
        VcwService::from_config(&config, &parts),
        Err(VcwBuildError::UnknownCode { .. })
    ));
    // A per-Pod override for a Pod that does not exist.
    let mut config = enabled();
    config
        .experts
        .simulated
        .pod_repair_success
        .entry("no-such-pod".into())
        .or_default()
        .insert("arith.value".into(), 0.5);
    assert!(matches!(
        VcwService::from_config(&config, &parts),
        Err(VcwBuildError::Simulation(_))
    ));
    // Another section problem.
    let mut config = enabled();
    config.max_rounds = 0;
    assert!(matches!(
        VcwService::from_config(&config, &parts),
        Err(VcwBuildError::Config(_))
    ));
}

#[cfg(not(feature = "qwen-cpu"))]
#[test]
fn the_model_backend_is_refused_in_a_build_without_it() {
    let mut config = enabled();
    config.kinds = vec![VcwTaskKind::Arithmetic];
    config.experts.backend = Some(ptr_config::VcwBackend::QwenCpu);
    config.experts.qwen_cpu.model_dir = "/nowhere".into();
    let digest = "ab".repeat(32);
    config.experts.qwen_cpu.weights_sha256 = digest.clone();
    config.experts.qwen_cpu.config_sha256 = digest.clone();
    config.experts.qwen_cpu.tokenizer_sha256 = digest.clone();
    config.experts.qwen_cpu.tokenizer_config_sha256 = digest;
    assert!(matches!(
        VcwService::from_config(&config, &VcwServiceParts::new([1; 32], [2; 8])),
        Err(VcwBuildError::QwenNotCompiled)
    ));
}

// ---------------------------------------------------------------------------------------------
// Round 2: backend marker, deadline, id consumption, install guards, legacy identity, hostile
// references, restarts, concurrency.
// ---------------------------------------------------------------------------------------------

fn parts() -> VcwServiceParts {
    VcwServiceParts::new([7; 32], [9; 8])
}

fn temp_ledger(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("ptr-server-vcw-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir.join("ledger.log")
}

fn cleanup(path: &std::path::Path) {
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
}

fn install_error(runtime: PtrRuntime, config: &VcwServiceConfig) -> VcwBuildError {
    let service = VcwService::from_config(config, &parts()).unwrap();
    match ServerState::new(runtime).with_vcw(service) {
        Ok(_) => panic!("the install was expected to be refused"),
        Err(error) => error,
    }
}

#[tokio::test]
async fn every_outcome_names_the_backend_and_the_pods_carry_it_as_a_capability() {
    let config = with_errors(enabled(), &[("arith.value", 1.0)]);
    let (state, app) = server(&config);
    let (status, body) = post(&app, &arithmetic("mark-ok")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["backend"], "simulated");

    let wrong = json!({"id": "mark-ex", "task": {
        "kind": "arithmetic", "expression": "(3 + 4) * 5", "reference": "36"}});
    let (status, body) = post(&app, &wrong).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body["outcome"], "exhausted");
    assert_eq!(body["backend"], "simulated");

    let (_, body) = post(&app, &fact_task("mark-esc")).await;
    assert!(body["backend"] == "simulated", "{body}");

    let service = state.vcw_service().unwrap();
    for (pod, _) in service.pods() {
        let capabilities = service.pod_capabilities(pod).unwrap();
        assert!(
            capabilities.contains(&"backend:simulated".to_owned()),
            "{pod:?}: {capabilities:?}"
        );
    }
}

#[test]
fn an_enabled_service_without_a_backend_is_refused() {
    let mut config = enabled();
    config.experts.backend = None;
    assert!(matches!(
        VcwService::from_config(&config, &parts()),
        Err(VcwBuildError::Config(_))
    ));
}

#[tokio::test]
async fn a_request_past_its_deadline_is_exhausted_without_a_candidate() {
    let service = VcwService::from_config(&enabled(), &parts())
        .unwrap()
        .with_request_deadline(Duration::ZERO);
    let state = ServerState::new(PtrRuntime::new(PtrConfig::default()).unwrap())
        .with_vcw(service)
        .unwrap();
    let app = router_from_state(state.clone());
    let (status, body) = post(&app, &arithmetic("late")).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(body["outcome"], "exhausted");
    assert_eq!(body["reason"], "Deadline");
    assert!(body.get("answer").is_none());
    // No Pod was asked, nothing was admitted, one VcwFinished says so, and the id is used.
    assert!(state.vcw_service().unwrap().expert_calls().is_empty());
    {
        let handle = state.runtime_handle();
        let runtime = handle.lock().unwrap();
        assert!(admitted_for(&runtime, "late").is_empty());
        let finished = finished_events(&runtime, "late");
        assert_eq!(finished.len(), 1);
        assert_eq!(finished[0].0, "exhausted");
    }
    assert_eq!(
        post(&app, &arithmetic("late")).await.0,
        StatusCode::CONFLICT
    );
}

#[tokio::test]
async fn every_request_that_committed_its_text_records_one_finish_and_consumes_its_id() {
    let config = with_errors(enabled(), &[("arith.value", 0.5)]);
    let (state, app) = server(&config);
    let mut ids = Vec::new();
    for i in 0..6 {
        let id = format!("fin-ok-{i}");
        post(&app, &arithmetic(&id)).await;
        ids.push(id);
    }
    // Exhausted, escalated, and a verified fact request (unknown: never accepted).
    let wrong = json!({"id": "fin-ex", "task": {
        "kind": "arithmetic", "expression": "(3 + 4) * 5", "reference": "36"}});
    post(&app, &wrong).await;
    ids.push("fin-ex".into());
    post(&app, &fact_task("fin-fact")).await;
    ids.push("fin-fact".into());

    let handle = state.runtime_handle();
    {
        let runtime = handle.lock().unwrap();
        for id in &ids {
            assert_eq!(finished_events(&runtime, id).len(), 1, "{id}");
        }
    }
    for id in &ids {
        let body = if id == "fin-ex" {
            wrong.clone()
        } else if id == "fin-fact" {
            fact_task(id)
        } else {
            arithmetic(id)
        };
        assert_eq!(post(&app, &body).await.0, StatusCode::CONFLICT, "{id}");
    }
    // The refusals above recorded nothing more.
    let runtime = handle.lock().unwrap();
    for id in &ids {
        assert_eq!(finished_events(&runtime, id).len(), 1, "{id}");
    }
}

#[test]
fn a_scope_of_an_earlier_incarnation_is_recovered_only_when_it_holds_nothing() {
    let scope = |lease: ScopeLeaseBinding| {
        move |runtime: &mut PtrRuntime| {
            runtime
                .create_scope(
                    ExecutionScope {
                        id: ScopeId::from("vcw-scope-vcw"),
                        parent: None,
                        session: SessionId::from("vcw"),
                        project: ProjectId::from("vcw"),
                        created_at: Timestamp(1),
                        deadline: None,
                        state: ScopeState::Created,
                        cancellation_requested: false,
                    },
                    lease,
                )
                .unwrap();
        }
    };
    for (tag, lease, refused) in [
        ("empty", ScopeLeaseBinding::default(), false),
        (
            "leased",
            ScopeLeaseBinding {
                pod_id: Some(PodId::from("some-pod")),
                ..ScopeLeaseBinding::default()
            },
            true,
        ),
    ] {
        let path = temp_ledger(&format!("scope-{tag}"));
        {
            let mut runtime = PtrRuntime::open_durable(PtrConfig::default(), &path).unwrap();
            scope(lease)(&mut runtime);
        }
        let runtime = PtrRuntime::open_durable(PtrConfig::default(), &path).unwrap();
        assert!(runtime
            .scopes()
            .recovery_required(&ScopeId::from("vcw-scope-vcw")));
        let service = VcwService::from_config(&enabled(), &parts()).unwrap();
        let result = ServerState::new(runtime).with_vcw(service);
        if refused {
            assert!(
                matches!(result, Err(VcwBuildError::ScopeNotEmpty { .. })),
                "{tag}"
            );
        } else {
            assert!(result.is_ok(), "{tag}");
        }
        cleanup(&path);
    }
}

#[test]
fn a_pod_id_that_is_another_projects_capsule_is_refused() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("notes"),
            capsule: CapsuleId::from("demo-note"),
            generation: Generation(1),
        })
        .unwrap();
    let mut config = enabled();
    config.pods.proposer = "demo-note".into();
    match install_error(runtime, &config) {
        VcwBuildError::PodIdCollision {
            pod,
            existing_project,
            ..
        } => {
            assert_eq!(pod, "demo-note");
            assert_eq!(existing_project, "notes");
        }
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn an_active_scope_of_another_project_under_the_service_name_is_not_adopted() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime
        .create_scope(
            ExecutionScope {
                id: ScopeId::from("vcw-scope-vcw"),
                parent: None,
                session: SessionId::from("other"),
                project: ProjectId::from("notes"),
                created_at: Timestamp(1),
                deadline: None,
                state: ScopeState::Created,
                cancellation_requested: false,
            },
            ScopeLeaseBinding::default(),
        )
        .unwrap();
    let (_state, app) = server_from(runtime, &enabled());
    // Adopting the foreign scope would make every admission fail with a project mismatch.
    let (status, body) = post(&app, &arithmetic("own-scope")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["outcome"], "verified");
}

#[tokio::test]
async fn an_active_scope_of_another_session_under_the_service_name_is_not_adopted() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime
        .create_scope(
            ExecutionScope {
                id: ScopeId::from("vcw-scope-vcw"),
                parent: None,
                session: SessionId::from("other"),
                project: ProjectId::from("vcw"),
                created_at: Timestamp(1),
                deadline: None,
                state: ScopeState::Created,
                cancellation_requested: false,
            },
            ScopeLeaseBinding::default(),
        )
        .unwrap();
    let (_state, app) = server_from(runtime, &enabled());
    // Admission requires the request's session; adopting this scope would refuse every answer.
    let (status, body) = post(&app, &arithmetic("own-session")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["outcome"], "verified");
}

#[test]
fn a_refusal_late_in_the_roster_commits_none_of_the_earlier_pods() {
    let service = VcwService::from_config(&enabled(), &parts()).unwrap();
    let last = service.pods().last().unwrap().0.clone();
    assert!(service.pods().len() > 1);
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("notes"),
            capsule: CapsuleId::from(last.0.as_str()),
            generation: Generation(1),
        })
        .unwrap();
    assert!(matches!(
        service.install(&mut runtime),
        Err(VcwBuildError::PodIdCollision { .. })
    ));
    let committed_for_vcw = runtime
        .committed_events()
        .iter()
        .filter(|e| {
            matches!(&e.event, LedgerEvent::CapsuleCommitted { project, .. }
                if project == service.project())
        })
        .count();
    assert_eq!(committed_for_vcw, 0);
}

#[test]
fn a_configured_generation_that_differs_from_the_live_one_is_refused() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("vcw"),
            capsule: CapsuleId::from("vcw-proposer"),
            generation: Generation(1),
        })
        .unwrap();
    let mut config = enabled();
    config.pods.generation = 2;
    let error = install_error(runtime, &config);
    match &error {
        VcwBuildError::GenerationMismatch {
            pod,
            configured,
            live,
        } => {
            assert_eq!((pod.as_str(), *configured, *live), ("vcw-proposer", 2, 1));
        }
        other => panic!("{other:?}"),
    }
    assert!(error.to_string().contains("generation"), "{error}");
}

#[tokio::test]
async fn a_revoked_pod_stays_revoked_after_a_restart_on_the_same_ledger() {
    let path = temp_ledger("revoked");
    {
        let runtime = PtrRuntime::open_durable(PtrConfig::default(), &path).unwrap();
        let (state, app) = server_from(runtime, &enabled());
        assert_eq!(post(&app, &arithmetic("before")).await.0, StatusCode::OK);
        revoke(&state, "vcw-proposer");
    }
    let runtime = PtrRuntime::open_durable(PtrConfig::default(), &path).unwrap();
    let (state, app) = server_from(runtime, &enabled());
    let (status, body) = post(&app, &arithmetic("after")).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(body["outcome"], "escalated");
    assert!(state.vcw_service().unwrap().expert_calls().is_empty());
    // The restart did not commit the Pod a second time (which would have revived it).
    let handle = state.runtime_handle();
    let runtime = handle.lock().unwrap();
    let commits = runtime
        .committed_events()
        .iter()
        .filter(|e| {
            matches!(&e.event, LedgerEvent::CapsuleCommitted { capsule, .. }
                if capsule.to_string() == "vcw-proposer")
        })
        .count();
    assert_eq!(commits, 1);
    assert!(admitted_for(&runtime, "after").is_empty());
    drop(runtime);
    cleanup(&path);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_concurrent_requests_with_one_id_yield_one_answer_and_one_conflict() {
    for round in 0..5 {
        let (state, app) = server(&enabled());
        let id = format!("race-{round}");
        let body = arithmetic(&id);
        let (a, b) = tokio::join!(post(&app, &body), post(&app, &body));
        let mut statuses = [a.0.as_u16(), b.0.as_u16()];
        statuses.sort_unstable();
        assert_eq!(statuses, [200, 409], "round {round}: {a:?} {b:?}");
        let handle = state.runtime_handle();
        let runtime = handle.lock().unwrap();
        assert_eq!(admitted_for(&runtime, &id).len(), 1);
        assert_eq!(finished_events(&runtime, &id).len(), 1);
    }
}

async fn full(
    app: &Router,
    method: &str,
    path: &str,
    body: &str,
) -> (u16, Vec<(String, Vec<u8>)>, Vec<u8>) {
    let request = Request::builder()
        .method(method)
        .uri(path)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_owned()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let mut headers: Vec<(String, Vec<u8>)> = response
        .headers()
        .iter()
        .map(|(name, value)| (name.to_string(), value.as_bytes().to_vec()))
        .collect();
    headers.sort();
    let bytes = to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .unwrap();
    (status, headers, bytes.to_vec())
}

#[tokio::test]
async fn a_sequence_of_legacy_requests_is_identical_with_and_without_the_vcw_section() {
    let plain_state = ServerState::new(PtrRuntime::new(PtrConfig::default()).unwrap());
    let plain = router_from_state(plain_state.clone());
    let (vcw_state, with_vcw) = server(&enabled());
    let installed = vcw_state
        .runtime_handle()
        .lock()
        .unwrap()
        .committed_events()
        .len();

    let sequence: [(&str, &str, &str); 9] = [
        ("GET", "/health", ""),
        (
            "POST",
            "/v1/requests",
            r#"{"id":"a1","text":"hello world"}"#,
        ),
        (
            "POST",
            "/v1/requests",
            r#"{"id":"a2","text":"remember the milk"}"#,
        ),
        ("POST", "/v1/requests", r#"{"id":"a1","text":"again"}"#),
        ("POST", "/v1/requests", r#"{"id":"","text":""}"#),
        ("POST", "/v1/requests", r#"{"id":"a3","text":"third"}"#),
        ("POST", "/v1/requests", "not json"),
        ("GET", "/nope", ""),
        ("GET", "/health", ""),
    ];
    for (method, path, body) in sequence {
        let a = full(&plain, method, path, body).await;
        let b = full(&with_vcw, method, path, body).await;
        assert_eq!(a, b, "{method} {path} {body}");
    }
    // The ledgers agree on everything the legacy requests wrote, event for event, once the
    // vcw install's own events are set aside (the indexes shift by exactly their number).
    let plain_events = plain_state
        .runtime_handle()
        .lock()
        .unwrap()
        .committed_events()
        .to_vec();
    let vcw_events = vcw_state
        .runtime_handle()
        .lock()
        .unwrap()
        .committed_events()
        .to_vec();
    assert!(!plain_events.is_empty());
    assert_eq!(vcw_events.len() - installed, plain_events.len());
    for (plain, vcw) in plain_events.iter().zip(&vcw_events[installed..]) {
        assert_eq!(plain.index.0 + installed as u64, vcw.index.0);
        assert_eq!(plain.event, vcw.event);
    }
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

#[tokio::test]
async fn a_hostile_reference_writes_no_fact_and_leaves_no_bytes_in_the_ledger() {
    let marker = "ZZ-HOSTILE-MARKER-ZZ";
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime
        .install_semantic_grant(
            SemanticGrant::new(RequiredVerification::Deterministic)
                .with_verifier(AllowHostWrites)
                .allow_host_writes(),
        )
        .unwrap();
    let mut delta = SemanticDelta::default();
    delta
        .upserts
        .insert(fact_key("paris", "capital-of"), "france".into());
    let revision = runtime.revision();
    runtime
        .apply_verified_semantic_delta(revision, delta, &PrincipalId::from("ops"))
        .unwrap();
    let fact_keys = |runtime: &PtrRuntime| -> Vec<(String, String)> {
        let snapshot = runtime.snapshot();
        let mut out: Vec<(String, String)> = snapshot
            .keys()
            .filter(|key| key.starts_with("fact/"))
            .map(|key| (key.to_owned(), format!("{:?}", snapshot.value(key))))
            .collect();
        out.sort();
        out
    };
    let before = fact_keys(&runtime);
    assert_eq!(before.len(), 1);

    let (state, app) = server_from(runtime, &enabled());
    let hostile = [
        // Each reference is wrong for its task (so it can never be an answer) and dresses up as a
        // fact or carries the marker.
        json!({"id": "h-arith", "task": {"kind": "arithmetic", "expression": "(3 + 4) * 5",
            "reference": format!("{marker} fact/paris/capital-of=berlin")}}),
        json!({"id": "h-json", "task": {
            "kind": "json_object", "instruction": "Describe a place.",
            "schema": {"type": "object", "required": ["name"],
                       "properties": {"name": {"type": "string"}},
                       "additionalProperties": false},
            "reference": {"name": "x", "fact/paris/capital-of": marker}}}),
        json!({"id": "h-code", "task": {
            "kind": "code", "instruction": "Square the argument.", "entry": "f",
            "tests": [{"id": "three", "args": [3], "expected": 9}],
            "reference": format!("fn f(a0) {{ return 1; }} // {marker} fact/paris/capital-of")}}),
        // A fact request whose claims contradict the committed state: it may be repaired to what
        // is committed, never accepted as claimed, and never written.
        json!({"id": "h-fact", "task": {"kind": "fact", "question": "Capital?",
            "claims": [{"subject": "paris", "attribute": "capital-of", "value": marker}]}}),
    ];
    for body in &hostile {
        let (status, response) = post(&app, body).await;
        let text = response.to_string();
        assert!(!text.contains(marker) || status != StatusCode::OK, "{text}");
        if body["id"] != "h-fact" {
            assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{response}");
        }
    }
    let handle = state.runtime_handle();
    let runtime = handle.lock().unwrap();
    assert_eq!(fact_keys(&runtime), before, "no fact/ key was written");
    let needle = marker.as_bytes();
    for committed in runtime.committed_events() {
        let text = format!("{:?}", committed.event);
        assert!(!text.contains(marker), "{text}");
        if let LedgerEvent::SemanticDeltaCommitted { encoded_delta, .. } = &committed.event {
            assert!(!contains(encoded_delta, needle));
        }
    }
    let snapshot = runtime.snapshot();
    for key in snapshot.keys() {
        assert!(!key.contains(marker));
        assert!(!format!("{:?}", snapshot.value(key)).contains(marker));
    }
}

#[tokio::test]
async fn shadow_learning_learns_and_logs_estimates_but_never_changes_what_is_served() {
    fn config(learning: VcwLearning) -> VcwServiceConfig {
        let mut config = with_errors(enabled(), &[("arith.value", 0.6), ("arith.format", 0.4)]);
        config.experts.simulated.seed = 5;
        config.experts.simulated.default_repair_success = 0.5;
        config.max_verifier_calls = 8;
        config.max_rounds = 6;
        config.kinds = vec![VcwTaskKind::Arithmetic];
        config.learning = learning;
        config
    }
    async fn run(config: &VcwServiceConfig) -> (ServerState, Vec<(StatusCode, Value)>) {
        let (state, app) = server(config);
        let mut out = Vec::new();
        for i in 0..160 {
            out.push(post(&app, &arithmetic(&format!("shadow-{i}"))).await);
        }
        (state, out)
    }
    let (off_state, off) = run(&config(VcwLearning::Off)).await;
    let (shadow_state, shadow) = run(&config(VcwLearning::Shadow)).await;

    // Off (the default) learns nothing and reports nothing.
    assert!(off_state
        .vcw_service()
        .unwrap()
        .learning_report()
        .is_empty());
    // Shadow serves byte for byte what off serves: the learned policy never routes.
    assert_eq!(off, shadow);
    assert!(shadow
        .iter()
        .any(|(_, b)| b["rounds"].as_u64().unwrap_or(0) > 0));

    // It did learn: verdicts were settled, a candidate was frozen, estimates were logged.
    let reports = shadow_state.vcw_service().unwrap().learning_report();
    assert_eq!(reports.len(), 1);
    let report = reports[0];
    assert_eq!(report.kind, VcwTaskKind::Arithmetic);
    assert!(report.settled > 256, "{report:?}");
    assert!(report.candidate.is_some(), "{report:?}");
    let latest = report.latest.expect("an estimate was logged");
    assert!(latest.records >= 128, "{latest:?}");
    // One Pod per role, so a learned policy cannot differ from uniform: the estimated gain is
    // zero, which is also what the doubly robust estimator must give when nothing can change.
    assert!(latest.difference.abs() < 1e-9, "{latest:?}");
    let handle = shadow_state.runtime_handle();
    let runtime = handle.lock().unwrap();
    let logged: Vec<_> = runtime
        .events()
        .iter()
        .filter(|e| matches!(e.event, RuntimeEvent::VcwShadowEstimate { .. }))
        .collect();
    assert!(!logged.is_empty());
    // No policy was promoted: no policy event exists.
    assert!(!runtime
        .events()
        .iter()
        .any(|e| matches!(e.event, RuntimeEvent::VcwPolicyChanged { .. })));
}

#[test]
fn shadow_learning_without_an_exploration_floor_is_refused() {
    let mut config = enabled();
    config.learning = VcwLearning::Shadow;
    config.exploration_floor = 0.0;
    assert!(matches!(
        VcwService::from_config(&config, &VcwServiceParts::new([1; 32], [2; 8])),
        Err(VcwBuildError::Config(_))
    ));
    assert!(
        PtrConfig::from_toml_str("[vcw]\nlearning = \"shadow\"\nexploration_floor = 0.0\n")
            .unwrap()
            .validate()
            .is_err()
    );
}

async fn get(app: &Router, path: &str) -> (StatusCode, Value) {
    let request = Request::get(path).body(Body::empty()).unwrap();
    let (status, bytes) = send(app, request).await;
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

#[tokio::test]
async fn the_trace_and_ledger_events_of_a_request_can_be_read_back() {
    let config = with_errors(enabled(), &[("arith.value", 1.0)]);
    let (state, app) = server(&config);
    let (status, body) = post(&app, &arithmetic("trace-1")).await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let (status, report) = get(&app, "/v1/vcw/traces/trace-1").await;
    assert_eq!(status, StatusCode::OK, "{report}");
    assert_eq!(report["outcome"], "verified");
    assert_eq!(report["backend"], "simulated");
    assert_eq!(report["trace"]["digest"], body["trace_digest"]);
    let steps = report["trace"]["steps"].as_array().unwrap();
    assert_eq!(steps[0]["candidate"]["status"], "Fail");
    assert_eq!(steps[0]["candidate"]["hard_codes"], json!(["arith.value"]));
    let routed = &steps[1]["routes"][0];
    assert_eq!(routed["pod"], "vcw-value-repair");
    assert!(routed["propensity"].as_f64().unwrap() > 0.0);
    assert_eq!(report["trace"]["spent"]["rounds"], body["rounds"]);

    // The events listed are exactly the ledger range the response reported, read from the ledger.
    let ledger = &report["ledger"];
    assert_eq!(ledger["first_index"], body["committed"]["first_index"]);
    assert_eq!(ledger["count"], body["committed"]["count"]);
    let kinds: Vec<&str> = ledger["events"]
        .as_array()
        .unwrap()
        .iter()
        .map(|event| event["kind"].as_str().unwrap())
        .collect();
    assert!(kinds.contains(&"PodOutputAdmitted"), "{kinds:?}");
    let handle = state.runtime_handle();
    let runtime = handle.lock().unwrap();
    assert_eq!(admitted_for(&runtime, "trace-1").len(), 1);
}

#[tokio::test]
async fn an_unanswered_request_reads_back_without_an_admission_and_an_unknown_id_is_404() {
    // The reference is wrong, so no candidate ever verifies.
    let config = with_errors(enabled(), &[("arith.value", 1.0)]);
    let (_state, app) = server(&config);
    let request = json!({"id": "trace-2", "task": {
        "kind": "arithmetic", "expression": "(3 + 4) * 5", "reference": "30"}});
    let (status, body) = post(&app, &request).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(body["outcome"], "exhausted");
    assert!(body.get("answer").is_none());

    let (status, report) = get(&app, "/v1/vcw/traces/trace-2").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(report["outcome"], "exhausted");
    let kinds: Vec<&str> = report["ledger"]["events"]
        .as_array()
        .unwrap()
        .iter()
        .map(|event| event["kind"].as_str().unwrap())
        .collect();
    assert!(
        !kinds.iter().any(|kind| kind.starts_with("PodOutput")),
        "{kinds:?}"
    );
    assert!(report["trace"]["steps"]
        .as_array()
        .unwrap()
        .iter()
        .all(|step| step["candidate"].is_null() || step["candidate"]["status"] != "Pass"));

    let (status, _) = get(&app, "/v1/vcw/traces/never-sent").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn the_trace_route_does_not_exist_without_the_vcw_path() {
    let app = router_from_state(ServerState::new(
        PtrRuntime::new(PtrConfig::default()).unwrap(),
    ));
    let (status, _) = get(&app, "/v1/vcw/traces/anything").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn a_second_eligible_repairer_makes_the_router_sample_with_a_propensity_below_one() {
    let mut config = with_errors(enabled(), &[("arith.value", 1.0)]);
    config
        .pods
        .extra_repairers
        .insert("value".into(), vec!["vcw-generalist".into()]);
    let (_state, app) = server(&config);
    let (status, body) = post(&app, &arithmetic("two-repairers")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (_, report) = get(&app, "/v1/vcw/traces/two-repairers").await;
    let repair = report["trace"]["steps"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|step| step["routes"].as_array().unwrap().iter())
        .find(|route| route["class"].as_str().unwrap().starts_with("Repair"))
        .expect("a repair was routed");
    let support = repair["support"].as_array().unwrap();
    let pods: Vec<&str> = support
        .iter()
        .map(|entry| entry["pod"].as_str().unwrap())
        .collect();
    assert_eq!(pods, ["vcw-generalist", "vcw-value-repair"]);
    let chosen = repair["pod"].as_str().unwrap();
    let width = support.iter().find(|entry| entry["pod"] == chosen).unwrap()["width"]
        .as_f64()
        .unwrap();
    let propensity = repair["propensity"].as_f64().unwrap();
    assert!(propensity > 0.0 && propensity < 1.0);
    assert!((propensity - width / 9_007_199_254_740_992.0).abs() < 1e-9);
}

// The guard is held across awaits on purpose: it stands in for a long request.
#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn trace_reads_and_health_answer_while_a_long_request_holds_the_runtime() {
    let config = with_errors(enabled(), &[("arith.value", 1.0)]);
    let (state, app) = server(&config);
    let (status, _) = post(&app, &arithmetic("held-1")).await;
    assert_eq!(status, StatusCode::OK);

    // Hold the runtime lock the way a long request does, and read from more concurrent clients
    // than there are async workers: none of them may wait for the lock.
    let handle = state.runtime_handle();
    let guard = handle.lock().unwrap();
    let mut reads = Vec::new();
    for _ in 0..8 {
        let app = app.clone();
        reads.push(tokio::spawn(async move {
            get(&app, "/v1/vcw/traces/held-1").await
        }));
    }
    let answered = tokio::time::timeout(Duration::from_secs(10), async {
        let health = get(&app, "/health").await;
        let mut statuses = Vec::new();
        for read in reads {
            statuses.push(read.await.unwrap().0);
        }
        (health.0, statuses)
    })
    .await
    .expect("a trace read or /health waited for the runtime lock");
    drop(guard);
    assert_eq!(answered.0, StatusCode::OK);
    assert!(answered.1.iter().all(|status| *status == StatusCode::OK));
}

#[tokio::test]
async fn the_ledger_summary_has_no_event_content() {
    let config = with_errors(enabled(), &[("arith.value", 1.0)]);
    let (_state, app) = server(&config);
    let (status, _) = post(&app, &arithmetic("summary-1")).await;
    assert_eq!(status, StatusCode::OK);
    let (_, report) = get(&app, "/v1/vcw/traces/summary-1").await;
    let events = report["ledger"]["events"].as_array().unwrap();
    assert!(!events.is_empty());
    for event in events {
        let mut keys: Vec<&str> = event
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        assert_eq!(keys, ["bytes", "digest", "index", "kind"]);
    }
}
