use axum::{
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use ptr_ledger::{integrity, LedgerEvent};
use ptr_model_api::{
    KvReuseHint, KvReusingBackend, ModelError, ModelEvent, ModelRun, PromptSegment,
    ReferenceEchoBackend, ResumableInferenceBackend, SegmentedModelRequest, Tokenizer,
};
use ptr_pods::PodRegistry;
use ptr_protocol::TypedPayload;
use ptr_router::{
    CheckedRouter, NonceSequence, PodRouter, RouteRequest, RoutedChoice, RoutedError, SamplingKey,
};
use ptr_runtime::{
    execution::{ActionExecutor, ActionScope, ExecutionGrant, RequiredVerification},
    PodSampler, PtrRuntime, RuntimeError,
};
use ptr_types::{
    CapabilityId, Effect, ProjectId, RequestId, SemanticContext, TypeId, VerificationLevel,
};
use ptr_verifier::{
    ActionContract, ActionContractVerifier, Finding, PayloadEncoding, VerificationReport,
    VerificationStatus, Verifier,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

mod ingress_diagnostics;
mod options;
mod request_tracing;
pub mod vcw;

pub use options::{
    IngressDiagnosticsLimits, IngressDiagnosticsMode, RequestTracingOptions, ServerOptions,
    ServerOptionsError, DEFAULT_MAX_CONCURRENT_INGRESS_DIAGNOSTICS,
    DEFAULT_MAX_INGRESS_ANALYZED_BYTES, DEFAULT_MAX_TRACE_EXPORT_BYTES,
    MAX_INGRESS_DIAGNOSTICS_BUDGET_BYTES,
};

pub use vcw::{
    VcwApiError, VcwBuildError, VcwResponse, VcwService, VcwServiceParts, VcwTaskRequest,
};

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
/// Longest accepted request id, in bytes.
pub const MAX_REQUEST_ID_BYTES: usize = 256;
/// Longest accepted request text, in bytes.
pub const MAX_REQUEST_TEXT_BYTES: usize = 1024 * 1024;
/// Largest accepted request body. Named here instead of inherited from the
/// framework's default, so a dependency upgrade cannot change it silently.
pub const MAX_REQUEST_BODY_BYTES: usize = 2 * 1024 * 1024;
/// Largest typed Pod output the default verifier accepts, in bytes.
pub const MAX_POD_OUTPUT_BYTES: usize = 64 * 1024;
/// Response header of the KV path: prompt tokens the backend served from its KV cache.
pub const KV_HIT_TOKENS_HEADER: &str = "x-ptr-kv-hit-tokens";

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
        // A temporary name that a restarted process cannot reproduce (process id,
        // clock and a counter), and a retry with a fresh name when one is taken
        // anyway. A temporary file is only ever removed when this call created it.
        let mut attempts = 0;
        let (mut file, temporary) = loop {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|elapsed| elapsed.as_nanos())
                .unwrap_or(0);
            let candidate = directory.join(format!(
                "demo-note.txt.tmp-{}-{nanos}-{}",
                std::process::id(),
                NOTE_WRITE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&candidate)
            {
                Ok(file) => break (file, candidate),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists && attempts < 8 => {
                    attempts += 1;
                }
                Err(error) => return Err(error.to_string()),
            }
        };
        let mut write = || -> std::io::Result<()> {
            use std::io::Write;
            file.write_all(&action.payload)?;
            file.sync_all()?;
            std::fs::rename(&temporary, &path)?;
            // Persist the rename itself. On Unix that is an fsync of the directory.
            // Windows cannot open a directory as a file, so the replaced note is
            // opened for writing and flushed instead, which asks the file system to
            // write out the file's metadata along with its data.
            #[cfg(unix)]
            std::fs::File::open(&directory)?.sync_all()?;
            #[cfg(not(unix))]
            std::fs::OpenOptions::new()
                .write(true)
                .open(&path)?
                .sync_all()?;
            Ok(())
        };
        if let Err(error) = write() {
            let _ = std::fs::remove_file(&temporary);
            return Err(error.to_string());
        }
        Ok(action.payload.clone())
    }
}

static NOTE_WRITE_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The one action the demo grant admits, built from the `DEMO_*` constants.
fn demo_note_contract() -> ActionContract {
    ActionContract {
        target: DEMO_NOTE_TARGET.into(),
        operation: DEMO_NOTE_OPERATION.into(),
        capability: CapabilityId::from(DEMO_NOTE_CAPABILITY),
        input_type: TypeId::from(DEMO_NOTE_INPUT_TYPE),
        effect: Effect::Mutation,
        max_payload_bytes: MAX_DEMO_NOTE_BYTES,
        payload_encoding: PayloadEncoding::Utf8,
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
            ActionContractVerifier::new(demo_note_contract()),
            DemoNoteExecutor::new(data_dir.clone()),
        )]
    })
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

/// A [`KvReusingBackend`] with the tokenizer and reuse hint the server calls it with.
///
/// The server encodes the request text as one `Raw` segment under `tokenizer`, so the
/// backend sees caller-supplied text only; the runtime-side taint check is what keeps an
/// Approximate run away from effects (see [`ServerBackend::KvReusing`]).
#[derive(Clone)]
pub struct KvBackend {
    backend: Arc<dyn KvReusingBackend>,
    tokenizer: Arc<dyn Tokenizer>,
    hint: Option<KvReuseHint>,
    max_prompt_tokens: Option<usize>,
}

impl KvBackend {
    pub fn new(
        backend: Arc<dyn KvReusingBackend>,
        tokenizer: Arc<dyn Tokenizer>,
        hint: Option<KvReuseHint>,
    ) -> Self {
        Self {
            backend,
            tokenizer,
            hint,
            max_prompt_tokens: None,
        }
    }

    /// Refuse (413, before any runtime call) a request text that tokenizes to more than
    /// `max` tokens; pass the backend's own prompt limit.
    pub fn with_max_prompt_tokens(mut self, max: usize) -> Self {
        self.max_prompt_tokens = Some(max);
        self
    }

    fn check_prompt_length(&self, text: &str) -> Result<(), ApiError> {
        let Some(maximum) = self.max_prompt_tokens else {
            return Ok(());
        };
        let tokens = self.tokenizer.encode(text).len();
        if tokens > maximum {
            return Err(ApiError::PromptTooLong { tokens, maximum });
        }
        Ok(())
    }

    fn infer(
        &self,
        request_id: &RequestId,
        context: SemanticContext,
        text: &str,
    ) -> Result<ModelRun, ModelError> {
        let contract = |error: ptr_model_api::TokenContractError| {
            ModelError(format!("segmented request refused: {error}"))
        };
        let segment =
            PromptSegment::raw_from_text(self.tokenizer.as_ref(), text).map_err(contract)?;
        let request = SegmentedModelRequest::new(
            request_id.clone(),
            context,
            self.tokenizer.identity().clone(),
            vec![segment],
        )
        .map_err(contract)?;
        self.backend.infer_segmented(&request, self.hint.as_ref())
    }
}

/// The model backend of a [`ServerState`].
#[derive(Clone)]
pub enum ServerBackend {
    /// A plain backend: events go through the resumable flow (Pods, verifier, rounds).
    Resumable(Arc<dyn ResumableInferenceBackend>),
    /// A KV-reusing backend. KV-INGEST-INVARIANT: every `ModelRun` it returns reaches the
    /// request flow only through `PtrRuntime::ingest_model_run` (via
    /// `run_model_run_resumable`), which records Approximate taint so that
    /// `authorize_action` refuses the action. The server never builds a `ResumableRun`
    /// from a `ModelRun` itself. This path does not resume Pods.
    KvReusing(KvBackend),
}

#[derive(Clone)]
pub struct ServerState {
    runtime: Arc<Mutex<PtrRuntime>>,
    backend: ServerBackend,
    router: Arc<dyn PodSampler>,
    pods: Arc<PodRegistry>,
    verifier: Arc<dyn Verifier<TypedPayload> + Send + Sync>,
    effect_grants: Arc<dyn Fn() -> Vec<ExecutionGrant> + Send + Sync>,
    admission: Arc<RequestAdmission>,
    /// The verifier-closed path; `None` unless installed with [`ServerState::with_vcw`].
    vcw: Option<Arc<VcwService>>,
    /// Opt-in capabilities; all off unless installed with [`ServerState::with_options`].
    options: options::ActiveOptions,
}

impl ServerState {
    pub fn new(runtime: PtrRuntime) -> Self {
        Self::with_dependencies(
            runtime,
            Arc::new(ReferenceEchoBackend),
            Arc::new(NoPods),
            Arc::new(PodRegistry::default()),
            default_verifier(),
            Arc::new(Vec::new),
        )
    }

    pub fn with_dependencies(
        runtime: PtrRuntime,
        backend: Arc<dyn ResumableInferenceBackend>,
        router: Arc<dyn PodSampler>,
        pods: Arc<PodRegistry>,
        verifier: Arc<dyn Verifier<TypedPayload> + Send + Sync>,
        effect_grants: Arc<dyn Fn() -> Vec<ExecutionGrant> + Send + Sync>,
    ) -> Self {
        Self::with_backend(
            runtime,
            ServerBackend::Resumable(backend),
            router,
            pods,
            verifier,
            effect_grants,
        )
    }

    /// Like [`ServerState::with_dependencies`] for either kind of [`ServerBackend`].
    pub fn with_backend(
        runtime: PtrRuntime,
        backend: ServerBackend,
        router: Arc<dyn PodSampler>,
        pods: Arc<PodRegistry>,
        verifier: Arc<dyn Verifier<TypedPayload> + Send + Sync>,
        effect_grants: Arc<dyn Fn() -> Vec<ExecutionGrant> + Send + Sync>,
    ) -> Self {
        // Read before the runtime moves into its mutex.
        let admission = Arc::new(
            RequestAdmission::new(runtime.config.runtime.mailbox_capacity)
                .expect("runtime.mailbox_capacity is validated non-zero by ptr-config"),
        );
        Self {
            runtime: Arc::new(Mutex::new(runtime)),
            backend,
            router,
            pods,
            verifier,
            effect_grants,
            admission,
            vcw: None,
            options: options::ActiveOptions::default(),
        }
    }

    /// Enable the opt-in capabilities of `options` (redacted request tracing, advisory ingress
    /// diagnostics). `ServerOptions::default()` changes nothing.
    ///
    /// # Errors
    /// [`ServerOptionsError`] when the trace export cap is zero or its file is unusable.
    pub fn with_options(mut self, options: ServerOptions) -> Result<Self, ServerOptionsError> {
        self.options = options::ActiveOptions::build(options, self.options.ingress_limits)?;
        Ok(self)
    }

    /// Bound the advisory ingress diagnostics: how many analyses run at once and how many
    /// leading bytes of a request text are analysed (defaults: 2 and 64 KiB). Independent of
    /// the order relative to [`ServerState::with_options`]; without the `Advisory` mode it only
    /// records the limits.
    ///
    /// # Errors
    /// [`ServerOptionsError::ZeroIngressConcurrency`] or
    /// [`ServerOptionsError::ZeroIngressAnalysisCap`] for a zero limit, and
    /// [`ServerOptionsError::IngressDiagnosticsBudgetExceeded`] when `max_concurrent *
    /// max_analyzed_bytes` exceeds [`MAX_INGRESS_DIAGNOSTICS_BUDGET_BYTES`].
    pub fn with_ingress_diagnostics_limits(
        mut self,
        limits: IngressDiagnosticsLimits,
    ) -> Result<Self, ServerOptionsError> {
        self.options = self.options.with_ingress_limits(limits)?;
        Ok(self)
    }

    /// Span batches dropped because the trace export file reached its byte cap
    /// (0 when tracing is off).
    pub fn dropped_trace_exports(&self) -> u64 {
        self.options
            .tracing
            .as_ref()
            .map_or(0, |tracing| tracing.dropped_exports())
    }

    /// Span batches lost to a write error on the trace export file (0 when tracing is off).
    pub fn trace_export_errors(&self) -> u64 {
        self.options
            .tracing
            .as_ref()
            .map_or(0, |tracing| tracing.export_errors())
    }

    /// Serve the verifier-closed path (`POST /v1/vcw/tasks`) from this state. The service's
    /// Pods are made live in the runtime and its scope is created, under the runtime's lock.
    /// Without this call the route does not exist.
    ///
    /// # Errors
    /// [`VcwBuildError::Install`] when the runtime refuses the Pods or the scope.
    pub fn with_vcw(mut self, service: VcwService) -> Result<Self, VcwBuildError> {
        {
            let mut runtime = self
                .runtime
                .lock()
                .map_err(|_| VcwBuildError::Install("runtime mutex poisoned".into()))?;
            service.install(&mut runtime)?;
        }
        self.vcw = Some(Arc::new(service));
        Ok(self)
    }

    /// The runtime behind the server, for hosts and tests that act on it directly (a
    /// revocation, a read of the ledger). Holding the lock stops the server's requests.
    pub fn runtime_handle(&self) -> Arc<Mutex<PtrRuntime>> {
        Arc::clone(&self.runtime)
    }

    /// How many requests hold an admission slot now (working or waiting for the runtime).
    pub fn requests_in_flight(&self) -> usize {
        self.admission.in_flight()
    }

    /// The verifier-closed service, when installed.
    pub fn vcw_service(&self) -> Option<&Arc<VcwService>> {
        self.vcw.as_ref()
    }
}

/// The production Pod router: `PodRouter` as the policy, sampled through `route_checked` with
/// `key` and a nonce sequence of `instance` starting at counter 0. `instance` must be fresh for
/// every process incarnation sharing `key` (nonce persistence across restarts is not done;
/// a fresh instance id per process keeps nonces unique).
pub fn checked_pod_router(key: SamplingKey, instance: [u8; 8]) -> Arc<dyn PodSampler> {
    Arc::new(CheckedRouter::new(
        PodRouter,
        key,
        NonceSequence::starting_at(instance, 0),
    ))
}

/// The sampler of `ServerState::new`, whose registry is empty: it routes nothing, so there is
/// no key and no draw.
struct NoPods;

impl PodSampler for NoPods {
    fn route(&self, _: &RouteRequest<'_>) -> Result<Option<RoutedChoice>, RoutedError> {
        Ok(None)
    }

    fn key_id(&self) -> [u8; 8] {
        [0; 8]
    }
}

pub fn router(runtime: PtrRuntime) -> Router {
    router_with_state(ServerState::new(runtime))
}

pub fn router_with_dependencies(
    runtime: PtrRuntime,
    backend: Arc<dyn ResumableInferenceBackend>,
    router: Arc<dyn PodSampler>,
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

/// Like [`router_with_dependencies`] for either kind of [`ServerBackend`].
pub fn router_with_backend(
    runtime: PtrRuntime,
    backend: ServerBackend,
    router: Arc<dyn PodSampler>,
    pods: Arc<PodRegistry>,
    verifier: Arc<dyn Verifier<TypedPayload> + Send + Sync>,
    effect_grants: Arc<dyn Fn() -> Vec<ExecutionGrant> + Send + Sync>,
) -> Router {
    router_with_state(ServerState::with_backend(
        runtime,
        backend,
        router,
        pods,
        verifier,
        effect_grants,
    ))
}

fn router_with_state(state: ServerState) -> Router {
    let routes = Router::new()
        .route("/health", get(health))
        .route("/v1/requests", post(request))
        .layer(axum::extract::DefaultBodyLimit::max(MAX_REQUEST_BODY_BYTES));
    // The verifier-closed path has its own, smaller body limit and exists only when installed.
    let routes = if state.vcw.is_some() {
        routes.merge(vcw::routes())
    } else {
        routes
    };
    // Only when tracing is enabled; route_layer so the matched route pattern is known.
    let routes = match state.options.tracing.clone() {
        Some(tracing) => routes.route_layer(axum::middleware::from_fn_with_state(
            tracing,
            request_tracing::trace_request,
        )),
        None => routes,
    };
    routes.with_state(state)
}

/// The router of a [`ServerState`], for a state built with [`ServerState::with_vcw`].
pub fn router_from_state(state: ServerState) -> Router {
    router_with_state(state)
}

/// Serve a [`ServerState`].
pub async fn serve_state(
    listener: tokio::net::TcpListener,
    state: ServerState,
) -> std::io::Result<()> {
    axum::serve(listener, router_with_state(state)).await
}

/// Like [`serve_state`], but stops accepting and returns once `shutdown` completes and the
/// requests in flight have finished.
pub async fn serve_state_until(
    listener: tokio::net::TcpListener,
    state: ServerState,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    axum::serve(listener, router_with_state(state))
        .with_graceful_shutdown(shutdown)
        .await
}

pub async fn serve(listener: tokio::net::TcpListener, runtime: PtrRuntime) -> std::io::Result<()> {
    axum::serve(listener, router(runtime)).await
}

pub async fn serve_with_dependencies(
    listener: tokio::net::TcpListener,
    runtime: PtrRuntime,
    backend: Arc<dyn ResumableInferenceBackend>,
    router: Arc<dyn PodSampler>,
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

/// Like [`serve_with_dependencies`] for either kind of [`ServerBackend`].
pub async fn serve_with_backend(
    listener: tokio::net::TcpListener,
    runtime: PtrRuntime,
    backend: ServerBackend,
    router: Arc<dyn PodSampler>,
    pods: Arc<PodRegistry>,
    verifier: Arc<dyn Verifier<TypedPayload> + Send + Sync>,
    effect_grants: Arc<dyn Fn() -> Vec<ExecutionGrant> + Send + Sync>,
) -> std::io::Result<()> {
    axum::serve(
        listener,
        router_with_backend(runtime, backend, router, pods, verifier, effect_grants),
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
) -> Result<Response, ApiError> {
    if input.id.trim().is_empty() || input.text.trim().is_empty() {
        return Err(ApiError::BadRequest("id and text must be non-empty".into()));
    }
    if input.id.len() > MAX_REQUEST_ID_BYTES || input.text.len() > MAX_REQUEST_TEXT_BYTES {
        return Err(ApiError::BadRequest(format!(
            "id must be at most {MAX_REQUEST_ID_BYTES} bytes and text at most {MAX_REQUEST_TEXT_BYTES}"
        )));
    }

    // Bounded admission (Invariant #8): the runtime is serialised behind one
    // mutex, so without a bound every waiting request would queue unboundedly.
    // A request that finds the mailbox full is refused instead of waiting. The
    // slot is taken before any blocking work and moves into the blocking task,
    // so it stays held until that work ends even if this future is cancelled,
    // and async workers stay free to refuse excess requests and serve /health.
    let slot = state.admission.try_enter().ok_or(ApiError::Saturated)?;
    // Only the id is kept for the tracing layer; never the text.
    let traced_id = state
        .options
        .tracing
        .is_some()
        .then(|| request_tracing::TracedRequestId(input.id.clone()));
    let result = tokio::task::spawn_blocking(move || {
        let _slot = slot;
        // Advisory only, computed before and apart from the runtime call, which receives
        // `input` unchanged.
        let diagnostics = state
            .options
            .ingress
            .as_ref()
            .map(|runner| runner.diagnose(&input.text));
        process_request_with(&state, input, diagnostics)
    })
    .await
    .map_err(|error| ApiError::Internal(format!("request task failed: {error}")))?;
    match traced_id {
        Some(id) => {
            let mut response = result.into_response();
            response.extensions_mut().insert(id);
            Ok(response)
        }
        None => result,
    }
}

#[cfg(test)]
fn process_request(state: &ServerState, input: ApiRequest) -> Result<Response, ApiError> {
    process_request_with(state, input, None)
}

fn process_request_with(
    state: &ServerState,
    input: ApiRequest,
    diagnostics: Option<ingress_diagnostics::IngressDiagnostics>,
) -> Result<Response, ApiError> {
    let mut runtime = state
        .runtime
        .lock()
        .map_err(|_| ApiError::Internal("runtime mutex poisoned".into()))?;
    // The key is judged by the runtime's own rule, before the model run commits
    // anything: a malformed key and an over-long key that never settled are
    // refused here (400), while an over-long key that an earlier build settled
    // still passes, so a retry under it is answered from its recorded outcome.
    // Whether the run ends in a mutation, and so needs a key at all, is only
    // known afterwards.
    if let Some(key) = input.idempotency_key.as_deref() {
        if !runtime.is_usable_execution_key(key) {
            return Err(ApiError::BadRequest(
                "idempotency_key must be a non-empty identifier without edge whitespace or control characters, and not longer than the runtime's key limit".into(),
            ));
        }
    }
    let request_id = RequestId::from(input.id.as_str());
    let kv_hit_tokens = std::cell::Cell::new(None);
    let run = match &state.backend {
        ServerBackend::Resumable(backend) => runtime.run_resumable_with_pods_using_router(
            request_id,
            &ProjectId::from("demo"),
            input.text.as_str(),
            backend.as_ref(),
            state.pods.as_ref(),
            state.router.as_ref(),
            state.verifier.as_ref(),
            4,
        ),
        ServerBackend::KvReusing(kv) => {
            let text = input.text.as_str();
            let id = request_id.clone();
            // Before any runtime call: an over-long prompt must not reach the ledger.
            kv.check_prompt_length(text)?;
            runtime.run_model_run_resumable(request_id, text, |context| {
                let run = kv.infer(&id, context, text)?;
                kv_hit_tokens.set(Some(run.kv_report.map_or(0, |report| report.hit_tokens())));
                Ok(run)
            })
        }
    }
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
        let executed = runtime
            .prepare_execution_once(
                &session,
                &ProjectId::from("demo"),
                action,
                std::time::Duration::from_secs(30),
                key,
            )
            .and_then(|permit| runtime.execute_prepared(&session, permit));
        // The session belongs to this request alone. It is ended whatever the outcome, so the
        // runtime's bounded session table never fills with the sessions of finished requests.
        // A session already gone (expired) is no error of this request.
        let _ = runtime.revoke_execution_session(&session);
        let output = executed.map_err(ApiError::execution)?;
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

    let response = ApiResponse {
        id: input.id,
        revision: runtime.revision().0,
        text,
        verified_evidence: run
            .observations
            .iter()
            .map(|output| format!("{}:{}", output.type_id, output.bytes.len()))
            .collect(),
        effect,
    };
    let body = match diagnostics {
        None => Json(response).into_response(),
        Some(ingress_diagnostics) => Json(ingress_diagnostics::WithDiagnostics {
            response: &response,
            ingress_diagnostics,
        })
        .into_response(),
    };
    // Read-only report of the KV path; absent for a plain backend.
    Ok(match kv_hit_tokens.get() {
        Some(hits) => ([(KV_HIT_TOKENS_HEADER, hits.to_string())], body).into_response(),
        None => body.into_response(),
    })
}

/// Sizing and encoding rules for typed Pod output.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PayloadBounds {
    pub max_bytes: usize,
    /// Type ids whose bytes must be UTF-8 text.
    pub textual_types: BTreeSet<TypeId>,
}

impl Default for PayloadBounds {
    fn default() -> Self {
        Self {
            max_bytes: MAX_POD_OUTPUT_BYTES,
            textual_types: BTreeSet::from([TypeId::from(DEMO_NOTE_INPUT_TYPE)]),
        }
    }
}

/// Deterministic verifier for typed Pod output: a non-empty type id, at most
/// `max_bytes` of payload, and UTF-8 for the type ids listed as textual. It
/// replaces an allow-all verifier: it cannot judge meaning, but it refuses
/// output that is malformed or oversized, with hard findings
/// (`type-id`, `payload-size`, `payload-utf8`).
#[derive(Clone, Debug, Default)]
pub struct PayloadBoundsVerifier {
    bounds: PayloadBounds,
}

impl PayloadBoundsVerifier {
    pub fn new(bounds: PayloadBounds) -> Self {
        Self { bounds }
    }
}

impl Verifier<TypedPayload> for PayloadBoundsVerifier {
    fn verify(&self, payload: &TypedPayload) -> VerificationReport {
        let mut findings = Vec::new();
        let mut violation = |code: &str, message: String| {
            findings.push(Finding {
                code: code.into(),
                message,
                hard: true,
            });
        };
        let type_id = payload.type_id.0.as_str();
        if type_id.is_empty() || type_id.trim() != type_id {
            violation(
                "type-id",
                "type id must be non-empty without edge whitespace".into(),
            );
        }
        if payload.bytes.len() > self.bounds.max_bytes {
            violation(
                "payload-size",
                format!(
                    "payload is {} bytes, limit is {}",
                    payload.bytes.len(),
                    self.bounds.max_bytes
                ),
            );
        }
        if self.bounds.textual_types.contains(&payload.type_id)
            && std::str::from_utf8(&payload.bytes).is_err()
        {
            violation("payload-utf8", "textual payload is not UTF-8".into());
        }
        let passed = findings.is_empty();
        VerificationReport {
            status: if passed {
                VerificationStatus::Pass
            } else {
                VerificationStatus::Fail
            },
            level: VerificationLevel::Deterministic,
            score: ptr_types::Probability::new(if passed { 1.0 } else { 0.0 })
                .expect("bounded score"),
            findings,
        }
    }
}

/// The default verifier: [`PayloadBounds::default`] (64 KiB, demo-note UTF-8).
pub fn default_verifier() -> Arc<dyn Verifier<TypedPayload> + Send + Sync> {
    default_verifier_with(PayloadBounds::default())
}

/// [`PayloadBoundsVerifier`] with caller-chosen bounds.
pub fn default_verifier_with(
    bounds: PayloadBounds,
) -> Arc<dyn Verifier<TypedPayload> + Send + Sync> {
    Arc::new(PayloadBoundsVerifier::new(bounds))
}

/// Bounds the requests the server works on or waits to work on at once.
/// Capacity comes from `runtime.mailbox_capacity`; it is validated non-zero
/// by `ptr-config`.
#[derive(Debug)]
pub struct RequestAdmission {
    in_flight: AtomicUsize,
    capacity: usize,
}

impl RequestAdmission {
    /// Fails with [`AdmissionError::ZeroCapacity`] for a capacity of zero,
    /// which would refuse every request.
    pub fn new(capacity: usize) -> Result<Self, AdmissionError> {
        if capacity == 0 {
            return Err(AdmissionError::ZeroCapacity);
        }
        Ok(Self {
            in_flight: AtomicUsize::new(0),
            capacity,
        })
    }

    /// Takes a slot, or `None` when all `capacity` slots are taken. Never waits.
    pub fn try_enter(self: &Arc<Self>) -> Option<AdmissionGuard> {
        let mut current = self.in_flight.load(Ordering::Acquire);
        loop {
            if current >= self.capacity {
                return None;
            }
            match self.in_flight.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    return Some(AdmissionGuard {
                        admission: Arc::clone(self),
                    })
                }
                Err(observed) => current = observed,
            }
        }
    }

    pub fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::Acquire)
    }
}

/// Why a [`RequestAdmission`] could not be built.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdmissionError {
    /// Capacity must be at least one.
    ZeroCapacity,
}

/// A taken admission slot, released on drop.
#[derive(Debug)]
pub struct AdmissionGuard {
    admission: Arc<RequestAdmission>,
}

impl Drop for AdmissionGuard {
    fn drop(&mut self) {
        self.admission.in_flight.fetch_sub(1, Ordering::AcqRel);
    }
}

#[derive(Debug)]
pub enum ApiError {
    BadRequest(String),
    Runtime(RuntimeError),
    Internal(String),
    Effect(String),
    EffectConflict(String),
    /// The request mailbox is full.
    Saturated,
    /// The text tokenizes beyond the KV backend's prompt limit.
    PromptTooLong {
        tokens: usize,
        maximum: usize,
    },
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
            Self::Saturated => {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    [(header::RETRY_AFTER, "1")],
                    Json(serde_json::json!({ "error": "request mailbox saturated" })),
                )
                    .into_response();
            }
            Self::BadRequest(message) => (StatusCode::BAD_REQUEST, message),
            // 409, not 422: the refusal depends on the server's KV-reuse state, not on the
            // request's content, so no edit of the request fixes it and an identical
            // request may be answered exactly once the cache state differs.
            Self::Runtime(RuntimeError::StaleRevision { .. })
            | Self::Runtime(RuntimeError::StaleGeneration { .. })
            | Self::Runtime(RuntimeError::ExecutionFenced)
            | Self::Runtime(RuntimeError::ApproximateStateAtEffectBoundary { .. }) => {
                (StatusCode::CONFLICT, format!("{self:?}"))
            }
            // 507, not 503: the registry never drains, so the condition is not transient
            // and a Retry-After would invite retries that cannot succeed.
            Self::Runtime(RuntimeError::KvTaintRegistryFull { .. }) => {
                (StatusCode::INSUFFICIENT_STORAGE, format!("{self:?}"))
            }
            Self::PromptTooLong { .. } => (StatusCode::PAYLOAD_TOO_LARGE, format!("{self:?}")),
            Self::Runtime(error) => (StatusCode::UNPROCESSABLE_ENTITY, format!("{error:?}")),
            Self::Effect(message) => (StatusCode::UNPROCESSABLE_ENTITY, message),
            Self::EffectConflict(message) => (StatusCode::CONFLICT, message),
            Self::Internal(message) => (StatusCode::INTERNAL_SERVER_ERROR, message),
        };
        (status, Json(serde_json::json!({ "error": message }))).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload(type_id: &str, bytes: &[u8]) -> TypedPayload {
        TypedPayload {
            type_id: TypeId::from(type_id),
            bytes: bytes.to_vec(),
        }
    }

    #[test]
    fn mutating_requests_do_not_accumulate_execution_sessions() {
        use ptr_ledger::LedgerEvent;
        use ptr_types::{CapsuleId, Generation};
        let data_dir = std::env::temp_dir().join(format!(
            "ptr-server-sessions-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&data_dir).unwrap();
        let mut runtime = PtrRuntime::new(ptr_config::PtrConfig::default()).unwrap();
        runtime
            .commit(LedgerEvent::CapsuleCommitted {
                project: "demo".into(),
                capsule: CapsuleId::from(DEMO_NOTE_TARGET),
                generation: Generation(1),
            })
            .unwrap();
        runtime.permissions_mut().allow_mutation = true;
        runtime
            .permissions_mut()
            .capabilities
            .insert(CapabilityId::from(DEMO_NOTE_CAPABILITY));
        let state = ServerState::with_dependencies(
            runtime,
            Arc::new(ptr_model_api::ScenarioBackend),
            Arc::new(NoPods),
            Arc::new(PodRegistry::default()),
            default_verifier(),
            demo_effect_grants(&data_dir),
        );
        // More mutating requests than the runtime holds sessions at once, well inside one
        // session lifetime: each request's session must end with the request.
        for attempt in 0..1100 {
            let response = process_request(
                &state,
                ApiRequest {
                    id: "effect".into(),
                    text: "hello".into(),
                    idempotency_key: Some("key".into()),
                },
            );
            assert!(response.is_ok(), "request {attempt}: {:?}", response.err());
        }
        let _ = std::fs::remove_dir_all(&data_dir);
    }

    #[test]
    fn checked_pod_router_reports_the_key_identity_and_routes_nothing_without_pods() {
        let router = checked_pod_router(SamplingKey::from_bytes([7; 32]), [9; 8]);
        assert_eq!(router.key_id(), SamplingKey::from_bytes([7; 32]).key_id());
        assert_ne!(router.key_id(), NoPods.key_id());
    }

    #[test]
    fn payload_bounds_accept_and_refuse_with_stable_codes() {
        let verifier = PayloadBoundsVerifier::default();
        assert_eq!(
            verifier.verify(&payload("any", b"ok")).status,
            VerificationStatus::Pass
        );
        let codes = |p: &TypedPayload| {
            verifier
                .verify(p)
                .findings
                .into_iter()
                .map(|f| f.code)
                .collect::<Vec<_>>()
        };
        assert_eq!(codes(&payload(" ", b"x")), ["type-id"]);
        assert_eq!(
            codes(&payload("any", &vec![0; MAX_POD_OUTPUT_BYTES + 1])),
            ["payload-size"]
        );
        assert_eq!(
            codes(&payload(DEMO_NOTE_INPUT_TYPE, &[0xff])),
            ["payload-utf8"]
        );
        assert!(codes(&payload("binary", &[0xff])).is_empty());
    }

    #[test]
    fn a_full_taint_registry_is_507_without_retry_after_and_a_long_prompt_is_413() {
        let response =
            ApiError::Runtime(RuntimeError::KvTaintRegistryFull { limit: 1 }).into_response();
        assert_eq!(response.status(), StatusCode::INSUFFICIENT_STORAGE);
        assert!(response.headers().get(header::RETRY_AFTER).is_none());
        let response = ApiError::PromptTooLong {
            tokens: 9,
            maximum: 4,
        }
        .into_response();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[test]
    fn admission_is_bounded_and_released_on_drop() {
        let admission = Arc::new(RequestAdmission::new(2).unwrap());
        let a = admission.try_enter().unwrap();
        let _b = admission.try_enter().unwrap();
        assert!(admission.try_enter().is_none());
        drop(a);
        assert_eq!(admission.in_flight(), 1);
        assert!(admission.try_enter().is_some());
    }

    #[test]
    fn zero_capacity_admission_is_rejected() {
        assert_eq!(
            RequestAdmission::new(0).unwrap_err(),
            AdmissionError::ZeroCapacity
        );
    }

    #[test]
    fn padded_type_ids_are_refused() {
        let verifier = PayloadBoundsVerifier::default();
        for id in [" x", "x ", "\tx"] {
            let report = verifier.verify(&payload(id, b"ok"));
            assert_eq!(report.status, VerificationStatus::Fail, "{id:?}");
            assert_eq!(report.findings[0].code, "type-id");
        }
    }

    #[test]
    fn default_verifier_with_applies_custom_bounds() {
        let verifier = default_verifier_with(PayloadBounds {
            max_bytes: 2,
            textual_types: BTreeSet::new(),
        });
        assert_eq!(
            verifier.verify(&payload("t", b"ab")).status,
            VerificationStatus::Pass
        );
        assert_eq!(
            verifier.verify(&payload("t", b"abc")).status,
            VerificationStatus::Fail
        );
    }

    #[tokio::test]
    async fn with_every_diagnostics_permit_held_the_request_is_served_with_skipped_diagnostics() {
        use axum::body::{to_bytes, Body};
        use axum::http::Request;
        use tower::ServiceExt;

        let state = ServerState::new(PtrRuntime::new(ptr_config::PtrConfig::default()).unwrap())
            .with_options(ServerOptions {
                ingress_diagnostics: IngressDiagnosticsMode::Advisory,
                ..ServerOptions::default()
            })
            .unwrap()
            .with_ingress_diagnostics_limits(IngressDiagnosticsLimits {
                max_concurrent: 2,
                max_analyzed_bytes: 1024,
            })
            .unwrap();
        let runner = state.options.ingress.clone().expect("advisory runner");
        let held = (runner.try_acquire().unwrap(), runner.try_acquire().unwrap());
        let app = router_from_state(state);
        let send = |app: Router| async move {
            let response = app
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/v1/requests")
                        .header("content-type", "application/json")
                        .body(Body::from(r#"{"id":"r1","text":"please delete the file"}"#))
                        .unwrap(),
                )
                .await
                .unwrap();
            let status = response.status();
            let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
            (
                status,
                serde_json::from_slice::<serde_json::Value>(&bytes).unwrap(),
            )
        };
        let (status, json) = send(app.clone()).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["text"], "please delete the file");
        assert_eq!(
            json["ingress_diagnostics"],
            serde_json::json!({"advisory": true, "skipped": "busy"})
        );
        drop(held);
        let (status, json) = send(app).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["ingress_diagnostics"]["kind"], "text");
        assert_eq!(json["ingress_diagnostics"]["truncated"], false);
    }
}
