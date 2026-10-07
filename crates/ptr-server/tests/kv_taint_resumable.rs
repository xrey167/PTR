//! KV taint on the HTTP `Resumable` backend path (the one `ptrd reference-echo` and `scenario`
//! use), end to end: a runtime that already holds a tainted Content subject for text T refuses
//! an echo-style action over T with 409 and attempts no effect, while other text is served.

use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use ptr_config::PtrConfig;
use ptr_ledger::LedgerEvent;
use ptr_model_api::{
    ByteTokenizer, KvReuseMode, KvReusePolicy, KvReuseReport, ModelEvent, ModelRun, PromptSegment,
    ScenarioBackend, SegmentedModelRequest, Tokenizer,
};
use ptr_runtime::kv_taint::KvTaintRequest;
use ptr_runtime::PtrRuntime;
use ptr_server::ServerBackend;
use ptr_types::{
    ActionIr, CapabilityId, CapsuleId, Effect, Generation, RequestId, Revision, SemanticContext,
    TypeId,
};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tower::ServiceExt;

const TAINTED_TEXT: &str = "tainted note";

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "ptr-kv-taint-resumable-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn ledger(&self) -> PathBuf {
        self.0.join("ledger.log")
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn demo_action(payload: &[u8]) -> ActionIr {
    ActionIr {
        operation: "create".into(),
        target: "demo-note".into(),
        capability: CapabilityId::from("demo.local-note.create"),
        effect: Effect::Mutation,
        input_type: TypeId::from("ptr.demo-note.v1"),
        generation: Generation(1),
        revision: Revision(0),
        payload: payload.to_vec(),
    }
}

/// A runtime armed for the demo note, holding an Approximate run whose generated text is
/// `TAINTED_TEXT` (its own action is a different one, so only the input taint can refuse the
/// scenario backend's action over that text).
fn seeded_runtime(dir: &TempDir) -> PtrRuntime {
    let mut runtime = PtrRuntime::open_durable(PtrConfig::default(), dir.ledger()).unwrap();
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

    let tokenizer = ByteTokenizer::new();
    let request = SegmentedModelRequest::new(
        RequestId::from("seed"),
        SemanticContext {
            revision: Revision(0),
            entries: Vec::new(),
        },
        tokenizer.identity().clone(),
        vec![PromptSegment::raw_from_text(&tokenizer, "seed").unwrap()],
    )
    .unwrap();
    let run = ModelRun {
        events: vec![
            ModelEvent::Token(TAINTED_TEXT.into()),
            ModelEvent::ActionReady(demo_action(b"an unrelated action")),
        ],
        kv_report: Some(
            KvReuseReport::new(
                1,
                KvReuseMode::Approximate,
                4,
                KvReusePolicy::AllowApproximate,
            )
            .unwrap(),
        ),
    };
    let ingested = runtime
        .ingest_model_run_for(
            &KvTaintRequest {
                request: &request,
                producer: "test",
                generation_parameters: b"",
            },
            &run,
        )
        .unwrap();
    assert!(ingested.kv_taint.is_approximate());
    runtime
}

fn app(dir: &TempDir) -> axum::Router {
    ptr_server::router_with_backend(
        seeded_runtime(dir),
        ServerBackend::Resumable(Arc::new(ScenarioBackend)),
        ptr_server::checked_pod_router(ptr_router::SamplingKey::from_bytes([7; 32]), [9; 8]),
        Arc::new(Default::default()),
        ptr_server::default_verifier(),
        ptr_server::demo_effect_grants(&dir.0),
    )
}

fn request(id: &str, text: &str, key: &str) -> Request<Body> {
    Request::post("/v1/requests")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"id": id, "text": text, "idempotency_key": key}).to_string(),
        ))
        .unwrap()
}

async fn send(app: axum::Router, request: Request<Body>) -> (StatusCode, Value) {
    let response = app.oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

fn effect_attempts(dir: &TempDir) -> usize {
    PtrRuntime::open_durable(PtrConfig::default(), dir.ledger())
        .unwrap()
        .committed_events()
        .iter()
        .filter(|committed| matches!(committed.event, LedgerEvent::EffectAttempted { .. }))
        .count()
}

#[tokio::test]
async fn tainted_text_is_refused_with_409_and_no_effect_other_text_is_served() {
    let dir = TempDir::new();
    {
        // Dropped before the file ledger is reopened below.
        let router = app(&dir);
        let (status, body) = send(router.clone(), request("r1", TAINTED_TEXT, "key-1")).await;
        assert_eq!(status, StatusCode::CONFLICT, "{body}");
        assert!(
            body["error"]
                .as_str()
                .unwrap()
                .contains("ApproximateStateAtEffectBoundary"),
            "{body}"
        );
        assert!(!dir.0.join("effects").exists());

        // The request id of the refused request is itself tainted now: even other text under
        // the same id is refused (and that text is tainted from then on, fail closed).
        let (status, body) = send(router.clone(), request("r1", "reused id note", "key-2")).await;
        assert_eq!(status, StatusCode::CONFLICT, "{body}");

        // Other text under a fresh id is served and its effect is applied.
        let (status, body) = send(router, request("r2", "clean note", "key-3")).await;
        assert_eq!(status, StatusCode::OK, "{body}");
    }
    assert_eq!(effect_attempts(&dir), 1);
    assert_eq!(
        std::fs::read(dir.0.join("effects/demo-note.txt")).unwrap(),
        b"clean note"
    );
}

#[tokio::test]
async fn the_refusal_survives_a_restart_of_the_server() {
    let dir = TempDir::new();
    drop(app(&dir));
    // A second server over the same durable ledger: the seed run is not repeated, the taint is
    // replayed from the ledger (the seeding helper is only used for the first build).
    let mut runtime = PtrRuntime::open_durable(PtrConfig::default(), dir.ledger()).unwrap();
    runtime.permissions_mut().allow_mutation = true;
    runtime
        .permissions_mut()
        .capabilities
        .insert(CapabilityId::from("demo.local-note.create"));
    let router = ptr_server::router_with_backend(
        runtime,
        ServerBackend::Resumable(Arc::new(ScenarioBackend)),
        ptr_server::checked_pod_router(ptr_router::SamplingKey::from_bytes([7; 32]), [9; 8]),
        Arc::new(Default::default()),
        ptr_server::default_verifier(),
        ptr_server::demo_effect_grants(&dir.0),
    );
    let (status, body) = send(router.clone(), request("r1", TAINTED_TEXT, "key-1")).await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    let (status, body) = send(router, request("r2", "clean note", "key-2")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
}
