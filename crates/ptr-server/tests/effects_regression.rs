use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use ptr_config::PtrConfig;
use ptr_ledger::LedgerEvent;
use ptr_model_api::{
    InferenceBackend, ModelError, ModelEvent, ModelRequest, ModelResumeRequest,
    ReferenceEchoBackend, ResumableInferenceBackend, ScenarioBackend,
};
use ptr_runtime::PtrRuntime;
use ptr_server::ApiResponse;
use ptr_types::{CapabilityId, CapsuleId, Generation};
use serde_json::json;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tower::ServiceExt;

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "ptr-effects-regression-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn entries(&self) -> Vec<String> {
        let mut entries: Vec<_> = std::fs::read_dir(self.0.join("effects"))
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        entries.sort();
        entries
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn note_app(dir: &TempDir) -> axum::Router {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
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
    ptr_server::router_with_dependencies(
        runtime,
        Arc::new(ScenarioBackend),
        Default::default(),
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

async fn success(response: axum::response::Response) -> ApiResponse {
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    serde_json::from_slice(&bytes).unwrap()
}

#[tokio::test]
async fn shorter_note_replaces_all_bytes_and_replaying_an_older_key_preserves_the_latest_note() {
    let dir = TempDir::new();
    std::fs::create_dir(dir.0.join("effects")).unwrap();
    let abandoned = dir.0.join("effects/demo-note.txt.tmp-abandoned");
    std::fs::write(&abandoned, b"unrelated recovery evidence").unwrap();
    let app = note_app(&dir);
    let first = success(
        app.clone()
            .oneshot(request("first", "a much longer first note", "key-a"))
            .await
            .unwrap(),
    )
    .await;
    let second = success(
        app.clone()
            .oneshot(request("second", "x", "key-b"))
            .await
            .unwrap(),
    )
    .await;
    let note = dir.0.join("effects/demo-note.txt");
    assert_eq!(std::fs::read(&note).unwrap(), b"x");
    let first_receipt = first.effect.unwrap();
    let second_receipt = second.effect.unwrap();
    assert!(second_receipt.attempt > first_receipt.settlement);
    assert!(second_receipt.settlement > second_receipt.attempt);
    assert_ne!(first_receipt.result_digest, second_receipt.result_digest);
    let replay = success(
        app.oneshot(request("first", "a much longer first note", "key-a"))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(replay.effect.unwrap(), first_receipt);
    assert_eq!(std::fs::read(note).unwrap(), b"x");
    assert_eq!(
        std::fs::read(abandoned).unwrap(),
        b"unrelated recovery evidence"
    );
    assert_eq!(
        dir.entries(),
        ["demo-note.txt", "demo-note.txt.tmp-abandoned"]
    );
}

#[tokio::test]
async fn failed_note_rename_cleans_its_temporary_file_and_preserves_the_destination() {
    let dir = TempDir::new();
    let destination = dir.0.join("effects/demo-note.txt");
    std::fs::create_dir_all(&destination).unwrap();
    std::fs::write(destination.join("keep"), b"existing data").unwrap();
    let app = note_app(&dir);
    let response = app
        .oneshot(request("rename-failure", "new note", "key-a"))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let body = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
    let error: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(error["error"].as_str().unwrap().starts_with("Executor("));
    assert_eq!(
        std::fs::read(destination.join("keep")).unwrap(),
        b"existing data"
    );
    assert_eq!(dir.entries(), ["demo-note.txt"]);
}

#[derive(Default)]
struct CountingBackend(AtomicUsize);

impl InferenceBackend for CountingBackend {
    fn name(&self) -> &'static str {
        "counting-echo"
    }

    fn infer(&self, request: &ModelRequest) -> Result<Vec<ModelEvent>, ModelError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        ReferenceEchoBackend.infer(request)
    }
}

impl ResumableInferenceBackend for CountingBackend {
    fn resume(&self, request: &ModelResumeRequest) -> Result<Vec<ModelEvent>, ModelError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        ReferenceEchoBackend.resume(request)
    }
}

#[tokio::test]
async fn malformed_keys_are_rejected_before_inference_or_revision_changes() {
    let backend = Arc::new(CountingBackend::default());
    let app = ptr_server::router_with_dependencies(
        PtrRuntime::new(PtrConfig::default()).unwrap(),
        backend.clone(),
        Default::default(),
        Arc::new(Default::default()),
        ptr_server::default_verifier(),
        Arc::new(Vec::new),
    );
    let too_long = "k".repeat(ptr_runtime::execution::MAX_KEY_BYTES + 1);
    for (i, key) in ["", " key", "key ", "key\ninside", "key\0inside", &too_long]
        .into_iter()
        .enumerate()
    {
        let response = app
            .clone()
            .oneshot(request(&format!("invalid-{i}"), "rejected", key))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{key:?}");
        assert_eq!(backend.0.load(Ordering::SeqCst), 0);
    }
    let valid = "k".repeat(ptr_runtime::execution::MAX_KEY_BYTES);
    let response = success(
        app.oneshot(request("accepted", "hello", &valid))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(
        response.revision, 1,
        "rejected requests must not advance the journal"
    );
    assert_eq!(backend.0.load(Ordering::SeqCst), 1);
}
