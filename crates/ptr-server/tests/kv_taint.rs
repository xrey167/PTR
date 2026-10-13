//! KV-reuse taint on the HTTP path: a `KvReusingBackend` behind the server, with every
//! `ModelRun` routed through `PtrRuntime::ingest_model_run`.

use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use ptr_config::PtrConfig;
use ptr_ledger::LedgerEvent;
use ptr_model_api::{
    ByteTokenizer, KvReuseHint, KvReuseMode, KvReusePolicy, KvReuseReport, KvReusingBackend,
    ModelError, ModelEvent, ModelRun, ScenarioBackend, SegmentSource, SegmentedModelRequest,
    Tokenizer,
};
use ptr_runtime::PtrRuntime;
use ptr_server::{KvBackend, ServerBackend};
use ptr_types::{ActionIr, CapabilityId, CapsuleId, Effect, Generation, TypeId};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tower::ServiceExt;

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "ptr-kv-taint-{}-{}",
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

/// What the test double saw, so tests can assert the request shape.
#[derive(Default)]
struct Seen {
    sources: Vec<SegmentSource>,
    texts: Vec<String>,
}

/// Echoes the request as the scenario backend does (token plus a demo-note action), with a
/// reuse report of the given mode, or none.
struct DoubleKv {
    mode: Option<KvReuseMode>,
    seen: Arc<Mutex<Seen>>,
}

impl KvReusingBackend for DoubleKv {
    fn name(&self) -> &str {
        "double-kv"
    }

    fn infer_segmented(
        &self,
        request: &SegmentedModelRequest,
        _hint: Option<&KvReuseHint>,
    ) -> Result<ModelRun, ModelError> {
        let tokenizer = ByteTokenizer::new();
        let mut text = String::new();
        let mut seen = self.seen.lock().unwrap();
        for segment in request.segments() {
            seen.sources.push(segment.source().clone());
            text.push_str(&tokenizer.decode(segment.tokens()).unwrap());
        }
        seen.texts.push(text.clone());
        let kv_report = self.mode.map(|mode| {
            KvReuseReport::new(
                1,
                mode,
                request.total_tokens(),
                KvReusePolicy::AllowApproximate,
            )
            .unwrap()
        });
        Ok(ModelRun {
            events: vec![
                ModelEvent::Token(text.clone()),
                ModelEvent::ActionReady(ActionIr {
                    operation: "create".into(),
                    target: "demo-note".into(),
                    capability: CapabilityId::from("demo.local-note.create"),
                    effect: Effect::Mutation,
                    input_type: TypeId::from("ptr.demo-note.v1"),
                    generation: Generation(1),
                    revision: request.revision(),
                    payload: text.into_bytes(),
                }),
            ],
            kv_report,
        })
    }
}

fn runtime(dir: &TempDir) -> PtrRuntime {
    let mut runtime = PtrRuntime::open_durable(PtrConfig::default(), dir.ledger()).unwrap();
    if runtime.live_generation("demo-note").is_none() {
        runtime
            .commit(LedgerEvent::CapsuleCommitted {
                project: "demo".into(),
                capsule: CapsuleId::from("demo-note"),
                generation: Generation(1),
            })
            .unwrap();
    }
    runtime.permissions_mut().allow_mutation = true;
    runtime
        .permissions_mut()
        .capabilities
        .insert(CapabilityId::from("demo.local-note.create"));
    runtime
}

fn app(dir: &TempDir, backend: ServerBackend) -> axum::Router {
    ptr_server::router_with_backend(
        runtime(dir),
        backend,
        ptr_server::checked_pod_router(ptr_router::SamplingKey::from_bytes([7; 32]), [9; 8]),
        Arc::new(Default::default()),
        ptr_server::default_verifier(),
        ptr_server::demo_effect_grants(&dir.0),
    )
}

fn kv_backend(mode: Option<KvReuseMode>) -> (ServerBackend, Arc<Mutex<Seen>>) {
    let seen = Arc::new(Mutex::new(Seen::default()));
    let double = DoubleKv {
        mode,
        seen: Arc::clone(&seen),
    };
    let hint = KvReuseHint {
        requested_prefix_tokens: 1,
        policy: KvReusePolicy::AllowApproximate,
    };
    (
        ServerBackend::KvReusing(KvBackend::new(
            Arc::new(double),
            Arc::new(ByteTokenizer::new()),
            Some(hint),
        )),
        seen,
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

fn ledger_events(dir: &TempDir) -> usize {
    PtrRuntime::open_durable(PtrConfig::default(), dir.ledger())
        .unwrap()
        .committed_events()
        .len()
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
async fn approximate_reuse_is_refused_with_409_and_no_effect_is_attempted() {
    let dir = TempDir::new();
    let (backend, seen) = kv_backend(Some(KvReuseMode::Approximate));
    let (status, body) = send(app(&dir, backend), request("r1", "note", "key-1")).await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert!(
        body["error"]
            .as_str()
            .unwrap()
            .contains("ApproximateStateAtEffectBoundary"),
        "{body}"
    );
    assert_eq!(effect_attempts(&dir), 0);
    assert!(!dir.0.join("effects").exists());
    // The server handed the backend one Raw segment holding the request text.
    let seen = seen.lock().unwrap();
    assert_eq!(seen.sources, [SegmentSource::Raw]);
    assert_eq!(seen.texts, ["note"]);
}

#[tokio::test]
async fn approximate_taint_survives_a_retry_of_the_same_action() {
    let dir = TempDir::new();
    let (backend, _) = kv_backend(Some(KvReuseMode::Approximate));
    {
        // Dropped before the ledger is reopened below (the file ledger is locked).
        let router = app(&dir, backend);
        for id in ["r1", "r2"] {
            let (status, body) = send(router.clone(), request(id, "note", "key-1")).await;
            assert_eq!(status, StatusCode::CONFLICT, "{body}");
        }
    }
    assert_eq!(effect_attempts(&dir), 0);
}

#[tokio::test]
async fn exact_and_no_reuse_match_the_plain_backend_path() {
    let plain_dir = TempDir::new();
    let plain = ServerBackend::Resumable(Arc::new(ScenarioBackend));
    let (plain_status, plain_body) =
        send(app(&plain_dir, plain), request("r1", "note", "key-1")).await;
    assert_eq!(plain_status, StatusCode::OK, "{plain_body}");
    for mode in [Some(KvReuseMode::Exact), None] {
        let dir = TempDir::new();
        let (backend, _) = kv_backend(mode);
        let (status, body) = send(app(&dir, backend), request("r1", "note", "key-1")).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body, plain_body, "{mode:?}");
        assert_eq!(effect_attempts(&dir), 1);
        assert_eq!(
            std::fs::read(dir.0.join("effects/demo-note.txt")).unwrap(),
            b"note"
        );
    }
}

#[tokio::test]
async fn a_run_without_finish_or_action_is_refused() {
    struct Silent;
    impl KvReusingBackend for Silent {
        fn name(&self) -> &str {
            "silent"
        }
        fn infer_segmented(
            &self,
            _: &SegmentedModelRequest,
            _: Option<&KvReuseHint>,
        ) -> Result<ModelRun, ModelError> {
            Ok(ModelRun {
                events: vec![ModelEvent::Token("x".into())],
                kv_report: None,
            })
        }
    }
    let dir = TempDir::new();
    let backend = ServerBackend::KvReusing(KvBackend::new(
        Arc::new(Silent),
        Arc::new(ByteTokenizer::new()),
        None,
    ));
    let (status, body) = send(app(&dir, backend), request("r1", "note", "key-1")).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(effect_attempts(&dir), 0);
}

#[tokio::test]
async fn a_segmented_run_that_asks_for_a_pod_is_refused() {
    struct PodAsker;
    impl KvReusingBackend for PodAsker {
        fn name(&self) -> &str {
            "pod-asker"
        }
        fn infer_segmented(
            &self,
            _: &SegmentedModelRequest,
            _: Option<&KvReuseHint>,
        ) -> Result<ModelRun, ModelError> {
            Ok(ModelRun {
                events: vec![ModelEvent::PodRequested {
                    capability: CapabilityId::from("x"),
                    input_type: TypeId::from("t"),
                    payload: vec![1],
                }],
                kv_report: None,
            })
        }
    }
    let dir = TempDir::new();
    let backend = ServerBackend::KvReusing(KvBackend::new(
        Arc::new(PodAsker),
        Arc::new(ByteTokenizer::new()),
        None,
    ));
    let (status, body) = send(app(&dir, backend), request("r1", "note", "key-1")).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert!(body["error"]
        .as_str()
        .unwrap()
        .contains("cannot resume a Pod"));
}

#[tokio::test]
async fn an_over_limit_prompt_is_refused_with_413_before_the_ledger_is_touched() {
    let dir = TempDir::new();
    // Creates the demo-note capsule; nothing else is committed.
    drop(app(&dir, kv_backend(None).0));
    let before = ledger_events(&dir);
    let seen = Arc::new(Mutex::new(Seen::default()));
    let backend = ServerBackend::KvReusing(
        KvBackend::new(
            Arc::new(DoubleKv {
                mode: None,
                seen: Arc::clone(&seen),
            }),
            Arc::new(ByteTokenizer::new()),
            None,
        )
        .with_max_prompt_tokens(4),
    );
    let router = app(&dir, backend);
    let (status, body) = send(router.clone(), request("r1", "too long", "key-1")).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{body}");
    // At the limit is accepted.
    let (status, body) = send(router.clone(), request("r2", "four", "key-2")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    drop(router);
    assert_eq!(seen.lock().unwrap().texts, ["four"]);
    // Only the accepted request wrote anything (request text, effect records).
    let after = ledger_events(&dir);
    assert!(after > before);
    let refused_only = TempDir::new();
    drop(app(&refused_only, kv_backend(None).0));
    let base = ledger_events(&refused_only);
    let backend = ServerBackend::KvReusing(
        KvBackend::new(
            Arc::new(DoubleKv {
                mode: None,
                seen: Arc::new(Mutex::new(Seen::default())),
            }),
            Arc::new(ByteTokenizer::new()),
            None,
        )
        .with_max_prompt_tokens(4),
    );
    let (status, _) = send(app(&refused_only, backend), request("r1", "too long", "k")).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(ledger_events(&refused_only), base);
}
