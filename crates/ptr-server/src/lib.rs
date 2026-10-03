use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use ptr_ledger::{integrity, LedgerEvent};
use ptr_model_api::{ModelEvent, ReferenceEchoBackend, ResumableInferenceBackend};
use ptr_pods::PodRegistry;
use ptr_protocol::TypedPayload;
use ptr_router::PodRouter;
use ptr_runtime::{
    execution::{ActionExecutor, ActionScope, ExecutionGrant, RequiredVerification},
    PtrRuntime, RuntimeError,
};
use ptr_types::{CapabilityId, Effect, ProjectId, RequestId, TypeId, VerificationLevel};
use ptr_verifier::{VerificationReport, VerificationStatus, Verifier};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ApiRequest {
    pub id: String,
    pub text: String,
    #[serde(default)]
    pub idempotency_key: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ApiResponse {
    pub id: String,
    pub revision: u64,
    pub text: String,
    pub verified_evidence: Vec<String>,
    pub effect: Option<EffectReceipt>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct EffectReceipt {
    pub attempt: u64,
    pub settlement: u64,
    pub result_digest: String,
}

pub const DEMO_NOTE_TARGET: &str = "demo-note";
pub const DEMO_NOTE_OPERATION: &str = "create";
pub const DEMO_NOTE_CAPABILITY: &str = "demo.local-note.create";
pub const DEMO_NOTE_INPUT_TYPE: &str = "ptr.demo-note.v1";
pub const MAX_DEMO_NOTE_BYTES: usize = 4 * 1024;
pub const MAX_IDEMPOTENCY_KEY_BYTES: usize = 256;

pub struct DemoNoteExecutor {
    data_dir: std::path::PathBuf,
}

impl DemoNoteExecutor {
    pub fn new(data_dir: impl Into<std::path::PathBuf>) -> Self {
        Self {
            data_dir: data_dir.into(),
        }
    }
}

impl ActionExecutor for DemoNoteExecutor {
    fn execute(
        &self,
        dispatch: ptr_runtime::execution::VerifiedDispatch<'_>,
    ) -> Result<Vec<u8>, String> {
        let action = dispatch.action();
        if action.effect != Effect::Mutation
            || action.target != DEMO_NOTE_TARGET
            || action.operation != DEMO_NOTE_OPERATION
            || action.capability != CapabilityId::from(DEMO_NOTE_CAPABILITY)
            || action.input_type != TypeId::from(DEMO_NOTE_INPUT_TYPE)
            || action.payload.len() > MAX_DEMO_NOTE_BYTES
            || std::str::from_utf8(&action.payload).is_err()
        {
            return Err("demo-note action contract rejected".into());
        }
        // The runtime executes each idempotency key at most once, so every
        // distinct mutation must be able to write its note. The note is replaced
        // atomically (temporary file, fsync, rename), which also makes the
        // re-execution after a reconciled crash window safe.
        let directory = self.data_dir.join("effects");
        std::fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
        let path = directory.join("demo-note.txt");
        let temporary = directory.join(format!(
            "demo-note.txt.tmp-{}-{}",
            std::process::id(),
            NOTE_WRITE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let write = || -> std::io::Result<()> {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)?;
            file.write_all(&action.payload)?;
            file.sync_all()?;
            std::fs::rename(&temporary, &path)?;
            std::fs::File::open(&directory)?.sync_all()
        };
        if let Err(error) = write() {
            let _ = std::fs::remove_file(&temporary);
            return Err(error.to_string());
        }
        Ok(action.payload.clone())
    }
}

static NOTE_WRITE_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

struct AllowActionVerifier;

impl Verifier<ptr_types::ActionIr> for AllowActionVerifier {
    fn verify(&self, _: &ptr_types::ActionIr) -> VerificationReport {
        VerificationReport {
            status: VerificationStatus::Pass,
            level: VerificationLevel::Deterministic,
            score: ptr_types::Probability::new(1.0).expect("one is valid"),
            findings: vec![],
        }
    }
}

pub fn demo_effect_grants(
    data_dir: impl Into<std::path::PathBuf>,
) -> Arc<dyn Fn() -> Vec<ExecutionGrant> + Send + Sync> {
    let data_dir = data_dir.into();
    Arc::new(move || {
        vec![ExecutionGrant::new(
            ActionScope {
                project: ProjectId::from("demo"),
                target: DEMO_NOTE_TARGET.into(),
                operation: DEMO_NOTE_OPERATION.into(),
                capability: CapabilityId::from(DEMO_NOTE_CAPABILITY),
                input_type: TypeId::from(DEMO_NOTE_INPUT_TYPE),
                effect: Effect::Mutation,
            },
            RequiredVerification::Deterministic,
            AllowActionVerifier,
            DemoNoteExecutor::new(data_dir.clone()),
        )]
    })
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

#[derive(Clone)]
pub struct ServerState {
    runtime: Arc<Mutex<PtrRuntime>>,
    backend: Arc<dyn ResumableInferenceBackend>,
    router: PodRouter,
    pods: Arc<PodRegistry>,
    verifier: Arc<dyn Verifier<TypedPayload> + Send + Sync>,
    effect_grants: Arc<dyn Fn() -> Vec<ExecutionGrant> + Send + Sync>,
}

impl ServerState {
    pub fn new(runtime: PtrRuntime) -> Self {
        Self::with_dependencies(
            runtime,
            Arc::new(ReferenceEchoBackend),
            PodRouter,
            Arc::new(PodRegistry::default()),
            Arc::new(AllowAllVerifier),
            Arc::new(Vec::new),
        )
    }

    pub fn with_dependencies(
        runtime: PtrRuntime,
        backend: Arc<dyn ResumableInferenceBackend>,
        router: PodRouter,
        pods: Arc<PodRegistry>,
        verifier: Arc<dyn Verifier<TypedPayload> + Send + Sync>,
        effect_grants: Arc<dyn Fn() -> Vec<ExecutionGrant> + Send + Sync>,
    ) -> Self {
        Self {
            runtime: Arc::new(Mutex::new(runtime)),
            backend,
            router,
            pods,
            verifier,
            effect_grants,
        }
    }
}

pub fn router(runtime: PtrRuntime) -> Router {
    router_with_state(ServerState::new(runtime))
}

pub fn router_with_dependencies(
    runtime: PtrRuntime,
    backend: Arc<dyn ResumableInferenceBackend>,
    router: PodRouter,
    pods: Arc<PodRegistry>,
    verifier: Arc<dyn Verifier<TypedPayload> + Send + Sync>,
    effect_grants: Arc<dyn Fn() -> Vec<ExecutionGrant> + Send + Sync>,
) -> Router {
    router_with_state(ServerState::with_dependencies(
        runtime,
        backend,
        router,
        pods,
        verifier,
        effect_grants,
    ))
}

fn router_with_state(state: ServerState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/requests", post(request))
        .with_state(state)
}

pub async fn serve(listener: tokio::net::TcpListener, runtime: PtrRuntime) -> std::io::Result<()> {
    axum::serve(listener, router(runtime)).await
}

pub async fn serve_with_dependencies(
    listener: tokio::net::TcpListener,
    runtime: PtrRuntime,
    backend: Arc<dyn ResumableInferenceBackend>,
    router: PodRouter,
    pods: Arc<PodRegistry>,
    verifier: Arc<dyn Verifier<TypedPayload> + Send + Sync>,
    effect_grants: Arc<dyn Fn() -> Vec<ExecutionGrant> + Send + Sync>,
) -> std::io::Result<()> {
    axum::serve(
        listener,
        router_with_dependencies(runtime, backend, router, pods, verifier, effect_grants),
    )
    .await
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".into(),
        version: env!("CARGO_PKG_VERSION").into(),
    })
}

async fn request(
    State(state): State<ServerState>,
    Json(input): Json<ApiRequest>,
) -> Result<Json<ApiResponse>, ApiError> {
    if input.id.trim().is_empty() || input.text.trim().is_empty() {
        return Err(ApiError::BadRequest("id and text must be non-empty".into()));
    }

    // Everything that can be judged from the request alone is judged before the
    // model run commits anything. Whether the run ends in a mutation (and so
    // requires a key) is only known afterwards.
    if let Some(key) = input.idempotency_key.as_deref() {
        if key.trim().is_empty()
            || key.len() > MAX_IDEMPOTENCY_KEY_BYTES
            || key.chars().any(char::is_control)
        {
            return Err(ApiError::BadRequest(
                "idempotency_key must be 1..=256 bytes without control characters".into(),
            ));
        }
    }

    let mut runtime = state
        .runtime
        .lock()
        .map_err(|_| ApiError::Internal("runtime mutex poisoned".into()))?;
    let run = runtime
        .run_resumable_with_pods_using_router(
            RequestId::from(input.id.as_str()),
            &ProjectId::from("demo"),
            input.text,
            state.backend.as_ref(),
            state.pods.as_ref(),
            &state.router,
            state.verifier.as_ref(),
            4,
        )
        .map_err(ApiError::Runtime)?;

    let effect = if let Some(action) = run.action.as_ref() {
        if action.effect == Effect::Mutation
            && (action.payload.len() > MAX_DEMO_NOTE_BYTES
                || std::str::from_utf8(&action.payload).is_err())
        {
            return Err(ApiError::BadRequest(
                "demo-note payload must be UTF-8 and at most 4 KiB".into(),
            ));
        }
        // Resolved before any session is registered, so a missing key leaves no
        // execution session behind.
        let key = input.idempotency_key.as_deref().ok_or_else(|| {
            ApiError::BadRequest("idempotency_key is required for mutations".into())
        })?;
        runtime
            .authorize_action(action)
            .map_err(ApiError::Runtime)?;
        let grants = (state.effect_grants)();
        if grants.is_empty() {
            return Err(ApiError::Effect("no effect grant installed".into()));
        }
        let session = runtime
            .register_execution_session("http", grants, std::time::Duration::from_secs(30))
            .map_err(|error| ApiError::Effect(format!("session: {error:?}")))?;
        let permit = runtime
            .prepare_execution_once(
                &session,
                &ProjectId::from("demo"),
                action,
                std::time::Duration::from_secs(30),
                key,
            )
            .map_err(ApiError::execution)?;
        let output = runtime
            .execute_prepared(&session, permit)
            .map_err(ApiError::execution)?;
        Some(effect_receipt(runtime.committed_events(), key, &output)?)
    } else {
        None
    };

    let text = run
        .model_events
        .into_iter()
        .filter_map(|event| match event {
            ModelEvent::Token(token) => Some(token),
            _ => None,
        })
        .collect::<String>();

    Ok(Json(ApiResponse {
        id: input.id,
        revision: runtime.revision().0,
        text,
        verified_evidence: run
            .observations
            .iter()
            .map(|output| format!("{}:{}", output.type_id, output.bytes.len()))
            .collect(),
        effect,
    }))
}

struct AllowAllVerifier;
pub fn default_verifier() -> Arc<dyn Verifier<TypedPayload> + Send + Sync> {
    Arc::new(AllowAllVerifier)
}
impl Verifier<TypedPayload> for AllowAllVerifier {
    fn verify(&self, _: &TypedPayload) -> VerificationReport {
        VerificationReport {
            status: VerificationStatus::Pass,
            level: VerificationLevel::Deterministic,
            score: ptr_types::Probability::new(1.0).expect("one is valid"),
            findings: vec![],
        }
    }
}

#[derive(Debug)]
pub enum ApiError {
    BadRequest(String),
    Runtime(RuntimeError),
    Internal(String),
    Effect(String),
    EffectConflict(String),
}

impl ApiError {
    fn execution(error: ptr_runtime::execution::ExecutionError) -> Self {
        match error {
            ptr_runtime::execution::ExecutionError::KeyBoundToAnotherAction { .. }
            | ptr_runtime::execution::ExecutionError::RuntimeFenced
            | ptr_runtime::execution::ExecutionError::AmbiguousOutcome { .. } => {
                Self::EffectConflict(format!("{error:?}"))
            }
            other => Self::Effect(format!("{other:?}")),
        }
    }
}

fn effect_receipt(
    events: &[ptr_ledger::CommittedEvent],
    key: &str,
    output: &[u8],
) -> Result<EffectReceipt, ApiError> {
    // One forward pass over the whole log. A key can own several attempts (a
    // crash window that was reconciled, then a fresh attempt); the receipt is
    // the most recent attempt that actually settled, paired with its own
    // settlement. A pure replay appends nothing, so the same rule yields the
    // original receipt.
    let mut attempts = std::collections::BTreeSet::new();
    let mut latest: Option<(u64, u64)> = None;
    for committed in events {
        match &committed.event {
            LedgerEvent::EffectAttempted {
                key: Some(recorded),
                ..
            } if recorded == key => {
                attempts.insert(committed.index.0);
            }
            LedgerEvent::EffectSettled { attempt: index, .. } if attempts.contains(&index.0) => {
                latest = Some((index.0, committed.index.0));
            }
            _ => {}
        }
    }
    let (attempt, settlement) = latest
        .ok_or_else(|| ApiError::Internal("missing effect attempt or settlement for key".into()))?;
    Ok(EffectReceipt {
        attempt,
        settlement,
        result_digest: hex_digest(&integrity::sha256(output)),
    })
}

fn hex_digest(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            Self::BadRequest(message) => (StatusCode::BAD_REQUEST, message),
            Self::Runtime(RuntimeError::StaleRevision { .. })
            | Self::Runtime(RuntimeError::StaleGeneration { .. })
            | Self::Runtime(RuntimeError::ExecutionFenced) => {
                (StatusCode::CONFLICT, format!("{self:?}"))
            }
            Self::Runtime(error) => (StatusCode::UNPROCESSABLE_ENTITY, format!("{error:?}")),
            Self::Effect(message) => (StatusCode::UNPROCESSABLE_ENTITY, message),
            Self::EffectConflict(message) => (StatusCode::CONFLICT, message),
            Self::Internal(message) => (StatusCode::INTERNAL_SERVER_ERROR, message),
        };
        (status, Json(serde_json::json!({ "error": message }))).into_response()
    }
}
