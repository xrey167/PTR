//! Verifier and backpressure behaviour of the HTTP boundary.

use axum::body::{to_bytes, Body};
use axum::http::{header, Request, StatusCode};
use ptr_config::PtrConfig;
use ptr_ledger::LedgerEvent;
use ptr_model_api::{
    InferenceBackend, ModelError, ModelEvent, ModelRequest, ModelResumeRequest,
    ReferenceEchoBackend, ResumableInferenceBackend, ScenarioBackend,
};
use ptr_runtime::execution::{ActionScope, ExecutionGrant, RequiredVerification};
use ptr_runtime::PtrRuntime;
use ptr_server::DemoNoteExecutor;
use ptr_types::{CapabilityId, CapsuleId, Effect, Generation, ProjectId, TypeId};
use ptr_verifier::{ActionContract, ActionContractVerifier, PayloadEncoding};
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tower::ServiceExt;

fn runtime(config: PtrConfig) -> PtrRuntime {
    let mut runtime = PtrRuntime::new(config).unwrap();
    runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: "demo".into(),
            capsule: CapsuleId::from("demo-note"),
            generation: Generation(1),
        })
        .unwrap();
    runtime.permissions_mut().allow_mutation = true;
    runtime
        .permissions_mut()
        .capabilities
        .insert(CapabilityId::from("demo.local-note.create"));
    runtime
}

fn request(id: &str, text: &str, key: &str) -> Request<Body> {
    Request::post("/v1/requests")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"id": id, "text": text, "idempotency_key": key}).to_string(),
        ))
        .unwrap()
}

fn data_dir(name: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("ptr-hardening-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).unwrap();
    path
}

/// The demo grant, but with a verifier that allows only `max_payload_bytes`.
/// The handler's own 4 KiB precheck still passes for the payloads used here, so
/// only the grant's verifier can refuse them.
fn strict_grants(
    dir: std::path::PathBuf,
    max_payload_bytes: usize,
) -> Arc<dyn Fn() -> Vec<ExecutionGrant> + Send + Sync> {
    Arc::new(move || {
        vec![ExecutionGrant::new(
            ActionScope {
                project: ProjectId::from("demo"),
                target: ptr_server::DEMO_NOTE_TARGET.into(),
                operation: ptr_server::DEMO_NOTE_OPERATION.into(),
                capability: CapabilityId::from(ptr_server::DEMO_NOTE_CAPABILITY),
                input_type: TypeId::from(ptr_server::DEMO_NOTE_INPUT_TYPE),
                effect: Effect::Mutation,
            },
            RequiredVerification::Deterministic,
            ActionContractVerifier::new(ActionContract {
                target: ptr_server::DEMO_NOTE_TARGET.into(),
                operation: ptr_server::DEMO_NOTE_OPERATION.into(),
                capability: CapabilityId::from(ptr_server::DEMO_NOTE_CAPABILITY),
                input_type: TypeId::from(ptr_server::DEMO_NOTE_INPUT_TYPE),
                effect: Effect::Mutation,
                max_payload_bytes,
                payload_encoding: PayloadEncoding::Utf8,
            }),
            DemoNoteExecutor::new(dir.clone()),
        )]
    })
}

#[tokio::test]
async fn grant_verifier_refuses_an_action_the_handler_precheck_admits() {
    let dir = data_dir("verifier");
    let app = ptr_server::router_with_dependencies(
        runtime(PtrConfig::default()),
        Arc::new(ScenarioBackend),
        ptr_server::checked_pod_router(ptr_router::SamplingKey::from_bytes([7; 32]), [9; 8]),
        Arc::new(Default::default()),
        ptr_server::default_verifier(),
        strict_grants(dir.clone(), 4),
    );
    let response = app
        .clone()
        .oneshot(request("too-long", "hello world", "key-a"))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert!(
        !dir.join("effects").exists(),
        "a refused action must not execute"
    );
    // The same grant still admits a payload within its contract.
    let ok = app.oneshot(request("fits", "hi", "key-b")).await.unwrap();
    assert_eq!(ok.status(), StatusCode::OK);
    assert_eq!(
        std::fs::read(dir.join("effects/demo-note.txt")).unwrap(),
        b"hi"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// Echo backend whose `infer` reports it started and then waits to be released.
struct GatedBackend {
    calls: AtomicUsize,
    entered: Mutex<Sender<()>>,
    release: Mutex<Receiver<()>>,
}

impl InferenceBackend for GatedBackend {
    fn name(&self) -> &'static str {
        "gated"
    }

    fn infer(&self, request: &ModelRequest) -> Result<Vec<ModelEvent>, ModelError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.entered.lock().unwrap().send(()).unwrap();
        self.release
            .lock()
            .unwrap()
            .recv_timeout(Duration::from_secs(30))
            .map_err(|_| ModelError("gate never released".into()))?;
        ReferenceEchoBackend.infer(request)
    }
}

impl ResumableInferenceBackend for GatedBackend {
    fn resume(&self, request: &ModelResumeRequest) -> Result<Vec<ModelEvent>, ModelError> {
        ReferenceEchoBackend.resume(request)
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_full_mailbox_is_refused_with_503_and_recovers_when_released() {
    let mut config = PtrConfig::default();
    config.runtime.mailbox_capacity = 1;
    let (entered_tx, entered_rx) = channel();
    let (release_tx, release_rx) = channel();
    let backend = Arc::new(GatedBackend {
        calls: AtomicUsize::new(0),
        entered: Mutex::new(entered_tx),
        release: Mutex::new(release_rx),
    });
    let app = ptr_server::router_with_dependencies(
        runtime(config),
        backend.clone(),
        ptr_server::checked_pod_router(ptr_router::SamplingKey::from_bytes([7; 32]), [9; 8]),
        Arc::new(Default::default()),
        ptr_server::default_verifier(),
        Arc::new(Vec::new),
    );
    let plain = |id: &str| {
        Request::post("/v1/requests")
            .header("content-type", "application/json")
            .body(Body::from(json!({"id": id, "text": "hello"}).to_string()))
            .unwrap()
    };

    let first = tokio::spawn(app.clone().oneshot(plain("a")));
    entered_rx
        .recv_timeout(Duration::from_secs(30))
        .expect("request A reaches the backend");
    assert_eq!(backend.calls.load(Ordering::SeqCst), 1);

    let refused = app.clone().oneshot(plain("b")).await.unwrap();
    assert_eq!(refused.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(refused.headers()[header::RETRY_AFTER], "1");
    let body = to_bytes(refused.into_body(), 64 * 1024).await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(body["error"], "request mailbox saturated");
    assert_eq!(
        backend.calls.load(Ordering::SeqCst),
        1,
        "a refused request must not reach the backend"
    );

    release_tx.send(()).unwrap();
    assert_eq!(first.await.unwrap().unwrap().status(), StatusCode::OK);

    release_tx.send(()).unwrap();
    let third = app.oneshot(plain("c")).await.unwrap();
    assert_eq!(third.status(), StatusCode::OK);
    assert_eq!(backend.calls.load(Ordering::SeqCst), 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn saturation_is_explicit_with_more_requests_than_workers() {
    let mut config = PtrConfig::default();
    config.runtime.mailbox_capacity = 4;
    let (entered_tx, entered_rx) = channel();
    let (release_tx, release_rx) = channel();
    let backend = Arc::new(GatedBackend {
        calls: AtomicUsize::new(0),
        entered: Mutex::new(entered_tx),
        release: Mutex::new(release_rx),
    });
    let app = ptr_server::router_with_dependencies(
        runtime(config),
        backend.clone(),
        ptr_server::checked_pod_router(ptr_router::SamplingKey::from_bytes([7; 32]), [9; 8]),
        Arc::new(Default::default()),
        ptr_server::default_verifier(),
        Arc::new(Vec::new),
    );
    let plain = |id: String| {
        Request::post("/v1/requests")
            .header("content-type", "application/json")
            .body(Body::from(json!({"id": id, "text": "hello"}).to_string()))
            .unwrap()
    };

    // Six concurrent requests on two async workers and capacity four.
    let tasks: Vec<_> = (0..6)
        .map(|n| tokio::spawn(app.clone().oneshot(plain(format!("r{n}")))))
        .collect();
    entered_rx
        .recv_timeout(Duration::from_secs(30))
        .expect("one request reaches the backend");

    // The two excess requests are refused while the other four stay in flight.
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    while tasks.iter().filter(|t| t.is_finished()).count() < 2 {
        assert!(
            std::time::Instant::now() < deadline,
            "refusals never arrived"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    // Saturated, yet the server still answers health checks.
    let health = app
        .clone()
        .oneshot(Request::get("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(health.status(), StatusCode::OK);

    for _ in 0..4 {
        release_tx.send(()).unwrap();
    }
    let mut ok = 0;
    let mut refused = 0;
    for task in tasks {
        let response = task.await.unwrap().unwrap();
        match response.status() {
            StatusCode::OK => ok += 1,
            StatusCode::SERVICE_UNAVAILABLE => {
                assert_eq!(response.headers()[header::RETRY_AFTER], "1");
                refused += 1;
            }
            other => panic!("unexpected status {other}"),
        }
    }
    assert_eq!((ok, refused), (4, 2));
    assert_eq!(backend.calls.load(Ordering::SeqCst), 4);
}
